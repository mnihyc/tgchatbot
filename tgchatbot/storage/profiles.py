"""Durable profile batching and atomic publication over original source spans."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
import uuid

from psycopg.types.json import Jsonb

from tgchatbot.domain.models import MessageRole


async def queue_input(conn, *, session_id, generation, message_id, revision, message, body, parts, canonical, force=False):
    if (message.role != MessageRole.USER or message.metadata.get('synthetic_role')
            or canonical['actor_kind'] == 'bot' or canonical['actor_id'] in {None, 'unknown'}):
        return
    spans = [{'start': part['text_span'][0], 'end': part['text_span'][1]} for part in parts
             if part['kind'] == 'text' and part.get('origin') not in
                {'auto_note', 'provenance', 'attachment_excerpt', 'attachment_reference', 'service_event'}
             and part.get('text_span') and body[part['text_span'][0]:part['text_span'][1]].strip()]
    size = sum(len(body[span['start']:span['end']].encode('utf-8')) for span in spans)
    await conn.execute('''INSERT INTO profile_inputs
        (message_id,session_id,generation,source_revision,actor_id,spans,pending_bytes)
        VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT(message_id) DO UPDATE
        SET source_revision=excluded.source_revision,actor_id=excluded.actor_id,
            spans=excluded.spans,pending_bytes=excluded.pending_bytes
        WHERE profile_inputs.source_revision<>excluded.source_revision OR %s''',
        (message_id, session_id, generation, revision, canonical['actor_id'], Jsonb(spans), size, force))


async def reconcile_sources(conn, session_id, generation, source_ids):
    """Surviving originals re-enter ordinary bounded learning, never current membership."""
    if not source_ids:
        return
    rows = await (await conn.execute('''SELECT m.id,m.source_revision,m.actor_id,m.actor_kind,m.role,
        r.body,r.parts,r.metadata FROM messages m JOIN message_revisions r
        ON (r.message_id,r.revision)=(m.id,m.source_revision)
        WHERE m.session_id=%s AND m.generation=%s AND m.id=ANY(%s) AND NOT m.hidden AND NOT m.deleted''',
        (session_id, generation, list(source_ids)))).fetchall()
    for row in rows:
        await queue_input(conn, session_id=session_id, generation=generation, message_id=row['id'],
            revision=row['source_revision'], message=SimpleNamespace(role=MessageRole(row['role']), metadata=row['metadata']),
            body=row['body'], parts=row['parts'], canonical=row, force=True)


async def publish_current(store, conn, scope, session_id, actor_id, *, add_ids=(), remove_ids=(), max_bytes):
    """Publish selected current facts; immutable revisions remain the evidence ledger."""
    current = await (await conn.execute('''SELECT fact_ids FROM profile_current
        WHERE session_id=%s AND generation=%s AND actor_id=%s FOR UPDATE''',
        (session_id, scope['generation'], actor_id))).fetchone()
    selected = set(current['fact_ids'] if current else []) - set(remove_ids)
    selected.update(add_ids)
    rows = await (await conn.execute('''SELECT * FROM profile_facts WHERE id=ANY(%s)
        AND session_id=%s AND generation=%s AND subject_actor_id=%s AND valid
        AND status IN ('active','superseded') AND (valid_to IS NULL OR valid_to>now()) ORDER BY id DESC''',
        (list(selected), session_id, scope['generation'], actor_id))).fetchall()
    await conn.execute('''INSERT INTO profile_current(session_id,generation,actor_id,fact_ids)
        VALUES (%s,%s,%s,%s) ON CONFLICT(session_id,generation,actor_id)
        DO UPDATE SET fact_ids=excluded.fact_ids,updated_at=now()''',
        (session_id, scope['generation'], actor_id, [row['id'] for row in rows]))
    await conn.execute('''UPDATE profile_facts SET retired_at=NULL,retirement_sources=NULL
        WHERE id=ANY(%s) AND session_id=%s AND generation=%s''',
        ([row['id'] for row in rows if row['id'] in add_ids], session_id, scope['generation']))


async def request_catchup(conn, *, session_id, scope):
    """Coalesce reset requests durably, including while a profile batch is busy."""
    await conn.execute('''INSERT INTO jobs
        (session_id,generation,context_id,scope_revision,kind,policy,dedupe_key)
        SELECT %s,%s,%s,%s,'memory_profile_request','memory','reset'
        WHERE EXISTS (SELECT 1 FROM profile_inputs WHERE session_id=%s
            AND generation=%s AND pending_bytes>0)
        ON CONFLICT(session_id,generation,kind,dedupe_key) DO NOTHING''',
        (session_id, scope['generation'], scope['context_id'], scope['revision'],
         session_id, scope['generation']))


async def claim_batch(store, *, max_bytes, lease_seconds, session_id=None, actor_ids=None, lazy=False,
                      profile_actor_ids=(), profile_bytes=None):
    """One chat owns one in-flight patch; no database lock spans a model call."""
    async with store.pool.connection() as conn:
        if session_id is None:
            # Pending spans are at least one byte. Reading at most max_bytes
            # rows is therefore enough to decide whether a full batch exists.
            candidate = await (await conn.execute('''SELECT s.session_id,
                (SELECT p.created_at FROM profile_patches p WHERE p.session_id=s.session_id
                    AND p.generation=s.generation ORDER BY p.id DESC LIMIT 1) AS last_served
                FROM sessions s
                WHERE EXISTS (SELECT 1 FROM jobs j WHERE j.session_id=s.session_id
                    AND j.generation=s.generation AND j.kind='memory_profile'
                    AND (j.status='pending' OR (j.status='running' AND j.lease_until<now()))
                    AND j.available_at<=now()) OR (
                  NOT EXISTS (SELECT 1 FROM jobs j WHERE j.session_id=s.session_id
                    AND j.generation=s.generation AND j.kind='memory_profile'
                    AND j.status IN ('pending','running','failed')) AND
                  (EXISTS (SELECT 1 FROM jobs j WHERE j.session_id=s.session_id
                    AND j.generation=s.generation AND j.kind='memory_profile_request') OR
                  (SELECT COALESCE(sum(p.pending_bytes),0) FROM
                    (SELECT pending_bytes FROM profile_inputs p WHERE p.session_id=s.session_id
                     AND p.generation=s.generation AND p.pending_bytes>0 ORDER BY p.message_id LIMIT %s) p)>=%s))
                ORDER BY last_served NULLS FIRST,s.session_id FOR UPDATE OF s SKIP LOCKED LIMIT 1''',
                (max_bytes, max_bytes))).fetchone()
            if not candidate:
                return None
            session_id = candidate['session_id']
        scope = await (await conn.execute('SELECT generation,context_id,revision FROM sessions '
            'WHERE session_id=%s FOR UPDATE', (session_id,))).fetchone()
        if scope is None:
            return None
        existing = await (await conn.execute('''SELECT * FROM jobs WHERE session_id=%s AND generation=%s
            AND kind='memory_profile' AND status IN ('pending','running','failed') ORDER BY id LIMIT 1 FOR UPDATE''',
            (session_id, scope['generation']))).fetchone()
        if existing:
            if lazy and actor_ids and not set(actor_ids).intersection(existing['payload'].get('reconcile_actors', [])):
                source_ids = [span['message_id'] for span in existing['payload'].get('spans', [])]
                relevant = await (await conn.execute('''SELECT 1 FROM messages WHERE session_id=%s
                    AND generation=%s AND id=ANY(%s) AND actor_id=ANY(%s) LIMIT 1''',
                    (session_id, scope['generation'], source_ids, actor_ids))).fetchone()
                if not relevant:
                    return None
            if (existing['status'] == 'failed' or existing['available_at'] > datetime.now(timezone.utc)
                    or (existing['status'] == 'running' and existing['lease_until'] >= datetime.now(timezone.utc))):
                return None
            return await (await conn.execute('''UPDATE jobs SET status='running',attempts=attempts+1,
                lease_token=%s,lease_until=now()+(%s * interval '1 second') WHERE id=%s RETURNING *''',
                (str(uuid.uuid4()), lease_seconds, existing['id']))).fetchone()
        # Background consumption waits for any current batch and leaves its
        # lease/retry policy untouched. Explicit subject refreshes do not consume
        # a chat-wide reset request on behalf of unrelated pending participants.
        request = (await (await conn.execute('''SELECT id FROM jobs WHERE session_id=%s
            AND generation=%s AND kind='memory_profile_request' FOR UPDATE''',
            (session_id, scope['generation']))).fetchone()) if actor_ids is None else None
        rows = await (await conn.execute('''WITH candidates AS (
            SELECT p.* FROM profile_inputs p
            JOIN messages m ON m.id=p.message_id AND m.source_revision=p.source_revision
            WHERE p.session_id=%s AND p.generation=%s AND p.pending_bytes>0 AND NOT m.hidden AND NOT m.deleted
            AND (%s::text[] IS NULL OR p.actor_id=ANY(%s::text[])) ORDER BY p.message_id LIMIT %s
          ), prefix AS (
            SELECT *,sum(pending_bytes) OVER (ORDER BY message_id)-pending_bytes AS previous_bytes FROM candidates
          ) SELECT p.*,r.body FROM prefix p
            JOIN message_revisions r ON (r.message_id,r.revision)=(p.message_id,p.source_revision)
            WHERE previous_bytes<%s ORDER BY p.message_id''',
            (session_id, scope['generation'], actor_ids, actor_ids, max_bytes, max_bytes))).fetchall()
        if not lazy and request is None and (not rows or sum(row['pending_bytes'] for row in rows) < max_bytes):
            return None
        spans, remaining, cursors, revisions = [], max_bytes, {}, {}
        for row in rows:
            # Keep the next utterance intact when this batch already has work.
            # An oversized first source still advances in bounded UTF-8 spans.
            if spans and row['pending_bytes'] > remaining:
                break
            pending = []
            for span in row['spans']:
                if pending:
                    pending.append(span)
                    continue
                start, end = span['start'], span['end']
                text = row['body'][start:end].encode('utf-8')[:remaining].decode('utf-8', errors='ignore')
                if text:
                    spans.append({'message_id': row['message_id'], 'start': start, 'end': start + len(text)})
                    revisions[str(row['message_id'])] = row['source_revision']
                    start += len(text)
                    remaining -= len(text.encode('utf-8'))
                if start < end:
                    pending.append({'start': start, 'end': end})
            if str(row['message_id']) in revisions:
                cursors[str(row['message_id'])] = {'spans': pending,
                    'pending_bytes': sum(len(row['body'][p['start']:p['end']].encode('utf-8')) for p in pending)}
            if remaining < 4 and pending:
                break
            if remaining == 0:
                break
        if rows and not spans:
            raise ValueError('MEMORY_WORKER_PROFILE_REQUEST_BYTES must fit one source character')
        if request is not None:
            # The request and its materialized batch commit together. A model
            # failure retries that same batch; a process restart loses neither.
            await conn.execute('DELETE FROM jobs WHERE id=%s', (request['id'],))
        if not spans:
            return None
        return await (await conn.execute('''INSERT INTO jobs
            (session_id,generation,context_id,scope_revision,kind,policy,source_ids,source_revisions,payload,
             status,attempts,lease_token,lease_until)
            VALUES (%s,%s,%s,%s,'memory_profile','memory',%s,%s,%s,'running',1,%s,
             now()+(%s * interval '1 second')) RETURNING *''',
            (session_id, scope['generation'], scope['context_id'], scope['revision'], list(map(int, revisions)),
             Jsonb(revisions), Jsonb({'spans': spans, 'cursors': cursors}),
             str(uuid.uuid4()), lease_seconds))).fetchone()


async def apply_patch(store, job, additions, removals, *, max_bytes, expected_source_revisions):
    from tgchatbot.storage.postgres_store import StaleScopeError
    async with store.pool.connection() as conn:
        locked = await store._locked_job(conn, job)
        if locked is None:
            raise StaleScopeError('Profile evidence or job lease changed before publication')
        scope = await store._session(conn, job['session_id'], lock=True)
        affected = set(job['payload'].get('reconcile_actors', []))
        sources = sorted(set(job['source_ids']) | {mid for item in additions for mid in item['source_ids']})
        retired_ids = [item['fact_id'] for item in removals]
        retired = await (await conn.execute('''SELECT f.id,f.source_ids FROM profile_facts f
            JOIN profile_current c ON c.session_id=f.session_id AND c.generation=f.generation
                AND c.actor_id=f.subject_actor_id AND f.id=ANY(c.fact_ids)
            WHERE f.id=ANY(%s) AND f.session_id=%s AND f.generation=%s AND f.valid''',
            (retired_ids, job['session_id'], scope['generation']))).fetchall() if retired_ids else []
        if set(retired_ids) != {row['id'] for row in retired}:
            raise StaleScopeError('A profile fact selected for retirement has changed')
        sources = sorted(set(sources) | {mid for fact in retired for mid in fact['source_ids']})
        await store._sources(conn, job['session_id'], scope, sources, expected_revisions=expected_source_revisions)
        for removal in removals:
            fact = await (await conn.execute('''UPDATE profile_facts SET retired_at=now(),retirement_sources=%s
                WHERE id=%s AND session_id=%s AND generation=%s AND valid
                RETURNING subject_actor_id''', (sources, removal['fact_id'], job['session_id'], scope['generation']))).fetchone()
            if fact is None:
                raise StaleScopeError('A profile fact selected for retirement has changed')
            affected.add(fact['subject_actor_id'])
        published, add_by_actor = [], {}
        for addition in additions:
            values = {key: value for key, value in addition.items() if key != 'reason'}
            fact = await store._save_profile_fact(conn, scope, job['session_id'], **values,
                expected_source_revisions=expected_source_revisions)
            published.append(fact['id'])
            add_by_actor.setdefault(fact['subject_actor_id'], []).append(fact['id'])
            affected.add(fact['subject_actor_id'])
        for actor in affected:
            await publish_current(store, conn, scope, job['session_id'], actor,
                add_ids=add_by_actor.get(actor, []), remove_ids=retired_ids, max_bytes=max_bytes)
        revisions = {str(mid): expected_source_revisions[str(mid)] for mid in sources}
        await conn.execute('''INSERT INTO profile_patches(session_id,generation,source_ids,source_revisions,patch)
            VALUES (%s,%s,%s,%s,%s)''', (job['session_id'], scope['generation'], sources,
                Jsonb(revisions), Jsonb({'additions': additions, 'removals': removals, 'published_fact_ids': published})))
        for message_id, cursor in job['payload']['cursors'].items():
            await conn.execute('''UPDATE profile_inputs SET spans=%s,pending_bytes=%s
                WHERE message_id=%s AND source_revision=%s''',
                (Jsonb(cursor['spans']), cursor['pending_bytes'], int(message_id), job['source_revisions'][message_id]))
        await conn.execute('DELETE FROM jobs WHERE id=%s', (job['id'],))


async def _identity(conn, session_id, generation, actor_id):
    return await (await conn.execute('''SELECT id AS message_id,source_revision,sent_at,actor_kind,actor_name
        FROM messages WHERE session_id=%s AND generation=%s AND actor_id=%s AND NOT hidden AND NOT deleted
        ORDER BY sent_at DESC,id DESC LIMIT 1''', (session_id, generation, actor_id))).fetchone()


async def add_source_dates(conn, documents):
    """Enrich only the selected learning facts inside their committed snapshot."""
    facts = [fact for document in documents for fact in document['facts']]
    if not facts:
        return
    rows = await (await conn.execute('''SELECT f.id,min(m.sent_at) AS first,max(m.sent_at) AS last
        FROM profile_facts f JOIN messages m ON m.id=ANY(f.source_ids)
        WHERE f.id=ANY(%s) GROUP BY f.id''', ([fact['id'] for fact in facts],))).fetchall()
    dates = {row['id']: {'first': row['first'], 'last': row['last']} for row in rows}
    for fact in facts:
        fact['source_dates'] = dates[fact['id']]
