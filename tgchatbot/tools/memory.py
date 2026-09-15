"""Operator-only PostgreSQL memory inspection, rebuild, and background processing.

No Telegram or SSH client is constructed. Audit reads are intentionally unavailable
as model tools. All commands reuse the deployment's .env and database configuration.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, fields
from datetime import datetime
import json
from pathlib import Path
from typing import Any, AsyncIterator
import uuid

from psycopg import Error as DatabaseError

from tgchatbot.domain.models import MessageRole
from tgchatbot.domain.identities import canonical_actor_id
from tgchatbot.domain.timestamps import format_timestamp
from tgchatbot.embeddings import EmbeddingClient, EmbeddingConfig
from tgchatbot.operational import from_env
from tgchatbot.storage.postgres_store import EMBEDDING_DIMENSIONS, PostgresStore, StaleScopeError
from tgchatbot.storage.retired_batches import UNRESOLVED_PAID

@dataclass(frozen=True)
class OperationsConfig:
    # These select traversal pages and status previews, never total archive size.
    page_size: int = 100
    status_job_limit: int = 100

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'MEMORY_OPERATIONS_{field.name.upper()} must be a positive integer')


def emit(record: dict[str, Any]) -> None:
    # SQL datetime objects are presentation fields. Original message prose and
    # JSON metadata/provider evidence remain their exact stored strings.
    def present(value):
        return format_timestamp(value) if isinstance(value, datetime) else str(value)
    print(json.dumps(record, ensure_ascii=False, default=present), flush=True)


async def existing_scope(store: PostgresStore, session_id: str) -> dict[str, int]:
    async with store.pool.connection() as conn:
        row = await (await conn.execute('SELECT generation,context_id,revision FROM sessions WHERE session_id=%s',
                                       (session_id,))).fetchone()
    if row is None:
        raise ValueError(f'Chat {session_id} does not exist; import or receive messages first')
    return dict(row)


async def _sessions(store: PostgresStore, session_id: str | None, options: OperationsConfig) -> AsyncIterator[str]:
    if session_id is not None:
        await existing_scope(store, session_id)
        yield session_id
        return
    after = ''
    while True:
        async with store.pool.connection() as conn:
            rows = await (await conn.execute('SELECT session_id FROM sessions WHERE session_id>%s ORDER BY session_id LIMIT %s',
                                           (after, options.page_size))).fetchall()
        if not rows:
            return
        for row in rows:
            yield row['session_id']
        after = rows[-1]['session_id']


def _covered_length(ranges: list[tuple[int, int]], length: int) -> int:
    total = end = 0
    for start, stop in sorted(ranges):
        start, stop = max(0, start), min(length, stop)
        if stop > max(end, start):
            total += stop - max(end, start)
        end = max(end, stop)
    return total


async def coverage(store: PostgresStore, session_id: str, scope: dict[str, int], space_id: str, *,
                   options: OperationsConfig | None = None) -> dict[str, Any]:
    """Count exact covered character spans, without treating one fragment as a full message."""
    options = options if options is not None else from_env(OperationsConfig, 'MEMORY_OPERATIONS')
    result = dict(active_originals=0, lexical_indexed_originals=0, semantic_eligible_originals=0,
                  semantic_full_originals=0, semantic_partial_originals=0, semantic_uncovered_originals=0,
                  semantic_eligible_characters=0, semantic_covered_characters=0)
    after = 0
    while True:
        async with store.pool.connection() as conn:
            rows = await (await conn.execute('''SELECT m.id,m.source_revision,m.role,
                r.body,r.parts,char_length(r.body) AS length,
                COALESCE(r.metadata->>'synthetic_role','') AS synthetic_role,
                x.message_id IS NOT NULL AS lexical FROM messages m JOIN message_revisions r
                ON (r.message_id,r.revision)=(m.id,m.source_revision)
                LEFT JOIN message_search x ON x.message_id=m.id
                WHERE m.session_id=%s AND m.generation=%s AND NOT m.hidden AND NOT m.deleted
                AND m.id>%s ORDER BY m.id LIMIT %s''',
                (session_id, scope['generation'], after, options.page_size))).fetchall()
            if not rows:
                break
            eligible = {}
            for row in rows:
                if row['role'] not in {'user', 'assistant'} or row['synthetic_role']:
                    continue
                # Match ExcerptBuilder: retain each nonblank part's canonical
                # offsets, including its whitespace, but not application notes
                # or the separator newlines between parts.
                ranges = [tuple(part['text_span']) for part in row['parts']
                          if part.get('text_span') is not None
                          and part.get('origin') not in {'auto_note', 'provenance'}
                          and row['body'][slice(*part['text_span'])].strip()]
                if ranges:
                    row['eligible_ranges'] = ranges
                    row['eligible_length'] = _covered_length(ranges, row['length'])
                    eligible[row['id']] = row
            intervals: dict[int, list[tuple[int, int]]] = {mid: [] for mid in eligible}
            if eligible:
                # Stream derivative rows too: a long original can have many excerpts.
                # This audit consumes the entire cursor. Plan for that work,
                # rather than optimizing a small initial fetch; the setting
                # ends with this page's transaction.
                await conn.execute('SET LOCAL cursor_tuple_fraction=1')
                async with conn.cursor(name=f'coverage_{uuid.uuid4().hex}') as cursor:
                    cursor.itersize = options.page_size
                    # One containment probe per original keeps the existing
                    # source index selective. An excerpt matching several
                    # originals still contributes its exact spans only once.
                    await cursor.execute('''SELECT DISTINCT e.id,e.source_ids,e.source_revisions,e.spans
                        FROM unnest(%s::bigint[]) wanted(message_id)
                        JOIN excerpts e ON e.source_ids @> ARRAY[wanted.message_id]
                        WHERE e.session_id=%s AND e.generation=%s AND e.valid AND e.embedding IS NOT NULL
                        AND e.model=%s''',
                        (list(eligible), session_id, scope['generation'], space_id))
                    async for excerpt in cursor:
                        if excerpt['spans']:
                            spans = excerpt['spans']
                        else:
                            spans = [{'message_id': mid, 'start': 0, 'end': eligible[mid]['length']}
                                     for mid in excerpt['source_ids'] if mid in eligible]
                        for span in spans:
                            mid = span['message_id']
                            if mid in eligible and excerpt['source_revisions'].get(str(mid)) == eligible[mid]['source_revision']:
                                # An excerpt can cross note/separator ranges.
                                # Count only its intersection with indexable text.
                                intervals[mid].extend((max(span['start'], start), min(span['end'], end))
                                    for start, end in eligible[mid]['eligible_ranges']
                                    if span['start'] < end and span['end'] > start)
            result['active_originals'] += len(rows)
            result['lexical_indexed_originals'] += sum(row['lexical'] for row in rows)
            for mid, row in eligible.items():
                covered = _covered_length(intervals[mid], row['length'])
                result['semantic_eligible_originals'] += 1
                result['semantic_eligible_characters'] += row['eligible_length']
                result['semantic_covered_characters'] += covered
                bucket = 'full' if covered == row['eligible_length'] else 'partial' if covered else 'uncovered'
                result[f'semantic_{bucket}_originals'] += 1
        after = rows[-1]['id']
    # A reset or edit during the scan must not produce a misleading coverage report.
    if await existing_scope(store, session_id) != scope:
        raise StaleScopeError('Chat changed during coverage scan; retry status')
    result['measured'] = True
    result['semantic_complete'] = not (result['semantic_partial_originals'] or result['semantic_uncovered_originals'])
    result['scope'] = ('Eligible nonblank part text in the selected space, using canonical character offsets; '
                       'excludes application notes, provenance, inter-part separators, profile extraction, and sticker coverage.')
    return result


async def status_records(store: PostgresStore, space_id: str, session_id: str | None = None, *,
                         include_coverage: bool = False,
                         options: OperationsConfig | None = None) -> AsyncIterator[dict[str, Any]]:
    options = options if options is not None else from_env(OperationsConfig, 'MEMORY_OPERATIONS')
    async for chat in _sessions(store, session_id, options):
        scope = await existing_scope(store, chat)
        covered = (await coverage(store, chat, scope, space_id, options=options) if include_coverage else
                   {'measured': False, 'reason': 'Use status --coverage for an explicit scan of all active source spans.'})
        async with store.pool.connection() as conn:
            spaces = await (await conn.execute('''SELECT model AS space_id,count(*) AS excerpts,
                count(*) FILTER (WHERE embedding IS NOT NULL) AS embedded_excerpts FROM excerpts
                WHERE session_id=%s AND generation=%s AND valid GROUP BY model ORDER BY model''',
                (chat, scope['generation']))).fetchall()
            pending_spaces = await (await conn.execute('''SELECT payload->>'space_id' AS space_id,kind,status,
                count(*) AS count FROM jobs WHERE session_id=%s AND generation=%s
                AND status IN ('pending','running','failed') AND payload ? 'space_id'
                GROUP BY payload->>'space_id',kind,status ORDER BY kind,status''',
                (chat, scope['generation']))).fetchall()
            batches = await (await conn.execute('''SELECT id,status,payload->>'space_id' AS space_id,
                payload->>'phase' AS phase,payload->>'name' AS name,payload->>'display_name' AS display_name
                FROM jobs WHERE session_id=%s AND generation=%s AND kind='embedding_batch'
                AND status IN ('pending','running','failed') ORDER BY id LIMIT %s''',
                (chat, scope['generation'], options.status_job_limit + 1))).fetchall()
            retired_filter = f'''j.session_id=%s AND j.kind='embedding_batch'
                AND (j.generation<%s OR j.status='stale' OR j.payload->>'retired_reconciliation'='true')
                AND ({UNRESOLVED_PAID})'''
            retired = await (await conn.execute(f'''SELECT count(*) AS count,
                count(*) FILTER (WHERE (j.payload->>'space_id') IS DISTINCT FROM %s) AS blocked_space_count
                FROM jobs j WHERE {retired_filter}''', (space_id, chat, scope['generation']))).fetchone()
            retired['jobs'] = await (await conn.execute(f'''SELECT j.id,j.generation,j.status,j.error,
                j.payload->>'space_id' AS space_id,j.payload->>'phase' AS phase,
                j.payload->>'name' AS name,j.payload->>'display_name' AS display_name
                FROM jobs j WHERE {retired_filter} ORDER BY j.id LIMIT %s''',
                (chat, scope['generation'], options.status_job_limit))).fetchall()
            retired['jobs_limit'] = options.status_job_limit
            retired['jobs_truncated'] = retired['count'] > len(retired['jobs'])
            tails = (await (await conn.execute('SELECT count(*) AS count FROM excerpt_tails WHERE session_id=%s AND generation=%s',
                                              (chat, scope['generation']))).fetchone())['count']
            profile_material = await (await conn.execute('''SELECT count(*) AS sources,
                COALESCE(sum(pending_bytes),0) AS bytes FROM profile_inputs
                WHERE session_id=%s AND generation=%s AND pending_bytes>0''',
                (chat, scope['generation']))).fetchone()
            profile_material['bytes'] = int(profile_material['bytes'])
        yield {'type': 'status', 'session_id': chat, **scope, 'selected_space_id': space_id,
               'coverage': covered, 'jobs': await store.job_status(chat), 'excerpt_spaces': spaces,
               'job_spaces': pending_spaces, 'batch_jobs': batches[:options.status_job_limit],
               'batch_jobs_limit': options.status_job_limit,
               'batch_jobs_truncated': len(batches) > options.status_job_limit,
               'retired_batch_slots': retired, 'pending_excerpt_tails': tails,
               'pending_profile_material': profile_material}


async def _check_unfinished_batches(store: PostgresStore, session_id: str, scope: dict[str, int], space_id: str) -> None:
    async with store.pool.connection() as conn:
        count = (await (await conn.execute('''SELECT count(*) AS count FROM jobs
            WHERE session_id=%s AND generation=%s AND kind='embedding_batch'
            AND status IN ('pending','running','failed') AND (payload->>'space_id') IS DISTINCT FROM %s''',
            (session_id, scope['generation'], space_id))).fetchone())['count']
    if count:
        raise ValueError('Unfinished Batch jobs use a different or unknown embedding space; reconcile them under their original configuration first')


async def retry_failed(store: PostgresStore, session_id: str, space_id: str, *, kinds: tuple[str, ...] | None = None) -> int:
    """Resume matching failed jobs without changing remote batch identity or source revisions."""
    await existing_scope(store, session_id)
    async with store.pool.connection() as conn:
        # Match storage lock ordering: session before jobs. Full resets cannot race this update.
        await conn.execute('SELECT session_id FROM sessions WHERE session_id=%s FOR UPDATE', (session_id,))
        result = await conn.execute('''UPDATE jobs j SET status='pending',attempts=0,error=NULL,
            available_at=now(),finished_at=NULL,lease_token=NULL,lease_until=NULL FROM sessions s
            WHERE j.session_id=s.session_id AND j.session_id=%s AND j.generation=s.generation
            AND j.status='failed' AND (j.policy='memory' OR (j.context_id=s.context_id AND j.scope_revision=s.revision))
            AND (NOT (j.payload ? 'space_id') OR j.payload->>'space_id'=%s)
            AND (j.kind NOT IN ('memory_embed','embedding_batch') OR j.payload->>'space_id'=%s)
            AND (%s::text[] IS NULL OR j.kind=ANY(%s::text[]))''',
            (session_id, space_id, space_id, list(kinds) if kinds else None, list(kinds) if kinds else None))
        return result.rowcount


async def rebuild(store: PostgresStore, session_id: str, space_id: str, *, progress=emit,
                  options: OperationsConfig | None = None) -> dict[str, Any]:
    options = options if options is not None else from_env(OperationsConfig, 'MEMORY_OPERATIONS')
    scope = await existing_scope(store, session_id)
    await _check_unfinished_batches(store, session_id, scope, space_id)
    revived = await retry_failed(store, session_id, space_id, kinds=('memory_embed', 'embedding_batch'))
    run_id, after, messages, jobs = uuid.uuid4().hex, 0, 0, 0
    while True:
        rows = await store.list_canonical_messages(session_id, after_message_id=after, limit=options.page_size, expected_scope=scope)
        if not rows:
            break
        ids = [row.db_id for row in rows if row.message.role in {MessageRole.USER, MessageRole.ASSISTANT}
               and not row.message.metadata.get('synthetic_role')]
        if ids:
            await store.enqueue_job(session_id, 'memory_ingest', source_ids=ids,
                payload={'space_id': space_id, 'rebuild_id': run_id},
                dedupe_key=f'rebuild:{space_id}:{run_id}:{rows[-1].db_id}', expected_scope=scope)
            jobs += 1
            messages += len(ids)
        after = rows[-1].db_id
        if progress:
            progress({'type': 'rebuild_progress', 'session_id': session_id, 'generation': scope['generation'],
                      'rebuild_id': run_id, 'queued_originals': messages, 'queued_jobs': jobs, 'last_message_id': after})
        await asyncio.sleep(0)
    await store.assert_scope(session_id, scope, generation_only=True)
    return {'type': 'rebuild_queued', 'session_id': session_id, 'generation': scope['generation'],
            'space_id': space_id, 'rebuild_id': run_id, 'queued_originals': messages, 'queued_jobs': jobs,
            'retried_embedding_jobs': revived, 'originals_deleted': 0}


async def audit_records(store: PostgresStore, session_id: str, *, message_id: int | None = None,
                        telegram_message_id: int | None = None,
                        generation: int | None = None, after_message_id: int = 0,
                        after_revision: int = 0,
                        options: OperationsConfig | None = None) -> AsyncIterator[dict[str, Any]]:
    options = options if options is not None else from_env(OperationsConfig, 'MEMORY_OPERATIONS')
    await existing_scope(store, session_id)
    source_chat_id = session_id.removeprefix('telegram:')
    while True:
        async with store.pool.connection() as conn:
            rows = await (await conn.execute('''SELECT m.id AS message_id,m.session_id,m.generation,m.context_id,
                m.role,m.source,m.source_chat_id,m.source_message_id,m.source_revision AS current_source_revision,
                m.sent_at,m.hidden,m.deleted,r.revision,r.body,r.parts,r.metadata,r.edited_at,r.created_at
                FROM messages m JOIN message_revisions r ON r.message_id=m.id
                WHERE m.session_id=%s AND (%s::bigint IS NULL OR m.id=%s)
                AND (%s::text IS NULL OR (m.source='telegram' AND m.source_chat_id=%s
                    AND m.source_message_id=%s) OR EXISTS (
                    SELECT 1 FROM message_source_aliases a WHERE a.message_id=m.id
                    AND a.session_id=m.session_id AND a.generation=m.generation
                    AND a.source='telegram' AND a.source_chat_id=%s AND a.source_message_id=%s))
                AND (%s::bigint IS NULL OR m.generation=%s) AND (m.id,r.revision)>(%s,%s)
                ORDER BY m.id,r.revision LIMIT %s''',
                (session_id, message_id, message_id,
                 str(telegram_message_id) if telegram_message_id is not None else None,
                 source_chat_id, str(telegram_message_id), source_chat_id, str(telegram_message_id),
                 generation, generation, after_message_id, after_revision, options.page_size))).fetchall()
        if not rows:
            return
        for row in rows:
            yield {'type': 'audit_revision', **row}
        after_message_id, after_revision = rows[-1]['message_id'], rows[-1]['revision']


async def job_records(store: PostgresStore, session_id: str, *, status: str | None = None,
                      kind: str | None = None, job_id: int | None = None,
                      generation: int | None = None,
                      options: OperationsConfig | None = None) -> AsyncIterator[dict[str, Any]]:
    """Stream saved job state, including errors and accepted remote operation identities."""
    options = options if options is not None else from_env(OperationsConfig, 'MEMORY_OPERATIONS')
    scope = await existing_scope(store, session_id)
    generation = scope['generation'] if generation is None else generation
    after = 0
    while True:
        async with store.pool.connection() as conn:
            rows = await (await conn.execute('''SELECT * FROM jobs
                WHERE session_id=%s AND generation=%s AND id>%s
                AND (%s::text IS NULL OR status=%s) AND (%s::text IS NULL OR kind=%s)
                AND (%s::bigint IS NULL OR id=%s) ORDER BY id LIMIT %s''',
                (session_id, generation, after, status, status, kind, kind, job_id, job_id,
                 options.page_size))).fetchall()
        if not rows:
            return
        for row in rows:
            yield {'type': 'memory_job', **row}
        after = rows[-1]['id']


async def profile_records(store: PostgresStore, session_id: str, *, max_bytes: int,
                          actor_id: str | None = None,
                          options: OperationsConfig | None = None) -> AsyncIterator[dict[str, Any]]:
    """Inspect current profiles and pending material without refreshing them."""
    from tgchatbot.domain.profiles import present_profile
    actor_id = canonical_actor_id(actor_id) if actor_id is not None else None
    options = options if options is not None else from_env(OperationsConfig, 'MEMORY_OPERATIONS')
    scope = await existing_scope(store, session_id)
    after = ''
    while True:
        async with store.pool.connection() as conn:
            if actor_id is not None:
                actors = [actor_id]
            else:
                rows = await (await conn.execute('''SELECT actor_id FROM (
                    SELECT actor_id FROM profile_current WHERE session_id=%s AND generation=%s
                    UNION SELECT actor_id FROM profile_inputs
                        WHERE session_id=%s AND generation=%s AND pending_bytes>0
                    ) actors WHERE actor_id>%s ORDER BY actor_id LIMIT %s''',
                    (session_id, scope['generation'], session_id, scope['generation'], after,
                     options.page_size))).fetchall()
                actors = [row['actor_id'] for row in rows]
        if not actors:
            await store.assert_scope(session_id, scope, generation_only=True)
            return
        snapshot = await store.fetch_profile_snapshot(session_id, actors,
            expected_scope=scope, max_bytes=max_bytes, include_pending=True)
        for document in snapshot['profiles']:
            actor = document['actor_id']
            yield {'type': 'current_profile', 'session_id': session_id, **snapshot['scope'],
                   'as_of': snapshot['as_of'], 'profile': present_profile(document),
                   'evidence': [fact for fact in snapshot['facts'] if fact['subject_actor_id'] == actor],
                   'pending_material': snapshot['pending_material'][actor]}
        if actor_id is not None:
            return
        after = actors[-1]


async def audit_state_records(store: PostgresStore, session_id: str, *,
                              generation: int | None = None,
                              options: OperationsConfig | None = None) -> AsyncIterator[dict[str, Any]]:
    """Operator-only personality, profile patch and hosted operation audit."""
    options = options if options is not None else from_env(OperationsConfig, 'MEMORY_OPERATIONS')
    await existing_scope(store, session_id)
    after = 0
    while True:
        async with store.pool.connection() as conn:
            rows = await (await conn.execute('''SELECT * FROM agent_generations WHERE session_id=%s
                AND (%s::bigint IS NULL OR generation=%s) AND generation>%s ORDER BY generation LIMIT %s''',
                (session_id, generation, generation, after, options.page_size))).fetchall()
        if not rows:
            break
        for row in rows:
            yield {'type': 'agent_generation_state', **row}
        after = rows[-1]['generation']
    after = 0
    while True:
        async with store.pool.connection() as conn:
            rows = await (await conn.execute('''SELECT * FROM profile_patches WHERE session_id=%s
                AND (%s::bigint IS NULL OR generation=%s) AND id>%s ORDER BY id LIMIT %s''',
                (session_id, generation, generation, after, options.page_size))).fetchall()
        if not rows:
            break
        for row in rows:
            yield {'type': 'profile_patch', **row}
        after = rows[-1]['id']
    after = 0
    while True:
        async with store.pool.connection() as conn:
            rows = await (await conn.execute('''SELECT id,session_id,generation,status,payload,created_at,error
                FROM jobs WHERE session_id=%s AND kind='embedding_batch'
                AND (payload ? 'name' OR payload->>'phase' IN ('submitting','polling'))
                AND (%s::bigint IS NULL OR generation=%s) AND id>%s ORDER BY id LIMIT %s''',
                (session_id, generation, generation, after, options.page_size))).fetchall()
        if not rows:
            break
        for row in rows:
            yield {'type': 'embedding_batch_state', **row}
        after = rows[-1]['id']


@asynccontextmanager
async def learning_resources(store: PostgresStore, config, embedding_config: EmbeddingConfig, *, batch: bool):
    # Lazy imports keep inspection/requeue commands free of generation clients.
    from tgchatbot.core.memory_worker import MemoryWorker
    from tgchatbot.providers.factory import build_providers
    if batch and (not embedding_config.enabled or embedding_config.provider != 'gemini'):
        raise ValueError('--batch requires the explicitly configured native Gemini embedding provider')
    async with AsyncExitStack() as stack:
        embeddings = EmbeddingClient(embedding_config)
        stack.push_async_callback(embeddings.aclose)
        providers = build_providers(config)
        for provider in providers.values():
            close = getattr(provider, 'aclose', None)
            if callable(close):
                stack.push_async_callback(close)
        worker = MemoryWorker(store=store, embeddings=embeddings, providers=providers, config=config, batch=batch)
        stack.push_async_callback(worker.close)
        yield worker, embeddings, providers


async def work(store: PostgresStore, config, embedding_config: EmbeddingConfig, *, batch: bool, once: bool) -> dict[str, Any]:
    async with learning_resources(store, config, embedding_config, batch=batch) as (worker, _, _):
        if once:
            processed = await worker.run_once()
            return {'type': 'worker_once', 'processed_dispatch': processed, 'last_error': worker.last_error,
                    'complete': False, 'note': 'One dispatch is not queue completion; inspect status, including deferred Batch jobs.'}
        await worker.run()
        return {'type': 'worker_stopped'}


async def prepare_context(store: PostgresStore, config, embedding_config: EmbeddingConfig,
                          session_id: str) -> dict[str, Any]:
    from tgchatbot.core.memory import MemoryService
    from tgchatbot.core.runtime import AgentRuntime
    from tgchatbot.storage.previews import PreviewCache
    await existing_scope(store, session_id)
    async with learning_resources(store, config, embedding_config, batch=False) as (worker, embeddings, providers):
        memory = MemoryService(store, embeddings, config=config.memory)
        memory.worker = worker
        previews = PreviewCache(store, max_bytes=config.memory.preview_cache_bytes)
        try:
            runtime = AgentRuntime(config=config, store=store, providers=providers,
                tool_registry=None, memory=memory, preview_cache=previews)
            async def progress(event):
                emit({'type': 'context_preparation_progress', 'session_id': session_id,
                    'phase': event.title, 'detail': event.detail})
            result = await runtime.prepare_context(session_id=session_id, emit=progress)
            return {'type': 'context_prepared', 'session_id': session_id, **result}
        finally:
            previews.close()


async def _run(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    from tgchatbot.config import load_config
    load_dotenv(Path.cwd() / '.env')
    config = load_config(require_telegram=False)
    options = from_env(OperationsConfig, 'MEMORY_OPERATIONS')
    store = PostgresStore(config.database_url)
    session_id = f'telegram:{args.chat_id}' if getattr(args, 'chat_id', None) is not None else None
    try:
        await store.initialize()
        if args.command == 'jobs':
            async for record in job_records(store, session_id, status=args.status, kind=args.kind,
                    job_id=args.job_id, generation=args.generation, options=options):
                emit(record)
            return
        if args.command == 'profiles':
            async for record in profile_records(store, session_id, actor_id=args.actor_id,
                    max_bytes=config.memory.profile_bytes, options=options):
                emit(record)
            return
        if args.command == 'context':
            from tgchatbot.tools.memory_inspection import context_records
            async for record in context_records(store, session_id, options=options):
                emit(record)
            return
        if args.command == 'read':
            from tgchatbot.tools.memory_inspection import query_memory
            emit(await query_memory(store, config, None, session_id, message_ids=args.message_id,
                offset=args.offset, length=args.length, include_neighbors=args.neighbors))
            return
        if args.command == 'image':
            from tgchatbot.tools.memory_inspection import export_image
            emit(await export_image(store, session_id, message_id=args.message_id,
                image_id=args.image_id, output=args.output))
            return
        if args.command == 'search' and args.lexical_only:
            from tgchatbot.tools.memory_inspection import query_memory
            emit(await query_memory(store, config, None, session_id, query=args.query,
                actor_id=args.actor_id, before=args.before, after=args.after, limit=args.limit,
                lexical_only=True))
            return
        if args.command == 'audit':
            if args.state:
                async for record in audit_state_records(store, session_id, generation=args.generation, options=options):
                    emit(record)
            async for record in audit_records(store, session_id, message_id=args.message_id,
                    telegram_message_id=args.telegram_message_id,
                    generation=args.generation, after_message_id=args.after_message_id, after_revision=args.after_revision, options=options):
                emit(record)
            return
        embedding_config = EmbeddingConfig.from_env()
        if embedding_config.enabled and embedding_config.dimensions != EMBEDDING_DIMENSIONS:
            raise ValueError(f'Conversation memory requires EMBEDDING_DIMENSIONS={EMBEDDING_DIMENSIONS}')
        if args.command == 'search':
            from tgchatbot.tools.memory_inspection import query_memory
            emit(await query_memory(store, config, embedding_config, session_id, query=args.query,
                actor_id=args.actor_id, before=args.before, after=args.after, limit=args.limit,
                lexical_only=args.lexical_only))
        elif args.command == 'status':
            async for record in status_records(store, embedding_config.space_id, session_id, include_coverage=args.coverage, options=options):
                emit(record)
        elif args.command == 'rebuild':
            emit(await rebuild(store, session_id, embedding_config.space_id, options=options))
        elif args.command == 'retry-jobs':
            count = await retry_failed(store, session_id, embedding_config.space_id)
            emit({'type': 'jobs_retried', 'session_id': session_id, 'space_id': embedding_config.space_id, 'count': count})
        elif args.command == 'prepare-context':
            emit(await prepare_context(store, config, embedding_config, session_id))
        else:
            emit(await work(store, config, embedding_config, batch=args.batch, once=args.once))
    finally:
        await store.close()


def _nonzero(value: str) -> int:
    number = int(value)
    if number == 0:
        raise argparse.ArgumentTypeError('chat ID must be nonzero')
    return number


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError('value must be positive')
    return number


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command', required=True)
    status = commands.add_parser('status', help='Per-chat queue counts and exact selected-space source coverage')
    status.add_argument('--chat-id', type=_nonzero)
    status.add_argument('--coverage', action='store_true', help='Explicitly scan all active original spans for full/partial semantic coverage')
    worker = commands.add_parser('work', help='Run the shared memory worker; stop the live bot first')
    worker.add_argument('--batch', action='store_true', help='Use native Gemini Batch for initial import or rebuild')
    worker.add_argument('--once', action='store_true', help='Perform one dispatch, including deferred-job handling; does not drain the queue')
    prepare = commands.add_parser('prepare-context', help='Compact imported working context in batches before resuming the bot; calls the configured model')
    prepare.add_argument('--chat-id', type=_nonzero, required=True)
    rebuild_parser = commands.add_parser('rebuild', help='Queue active canonical originals for the selected embedding space')
    rebuild_parser.add_argument('--chat-id', type=_nonzero, required=True)
    retry = commands.add_parser('retry-jobs', help='Resume failed jobs in this chat and selected space, preserving paid Batch identity')
    retry.add_argument('--chat-id', type=_nonzero, required=True)
    jobs = commands.add_parser('jobs', help='Stream saved job errors, progress and remote identities; current generation by default')
    jobs.add_argument('--chat-id', type=_nonzero, required=True)
    jobs.add_argument('--status', help='Exact saved job status')
    jobs.add_argument('--kind', help='Exact job kind')
    jobs.add_argument('--job-id', type=_positive)
    jobs.add_argument('--generation', type=_positive, help='Inspect this retained generation instead of the current one')
    profiles = commands.add_parser('profiles', help='Inspect current stored profiles and pending material without model calls')
    profiles.add_argument('--chat-id', type=_nonzero, required=True)
    profiles.add_argument('--actor-id', help='Stable actor ID; omit to discover profiles and pending actors')
    context = commands.add_parser('context', help='Inspect current context layers and counts without model calls')
    context.add_argument('--chat-id', type=_nonzero, required=True)
    search = commands.add_parser('search', help='Search currently retrievable memory; may call embeddings unless --lexical-only')
    search.add_argument('--chat-id', type=_nonzero, required=True)
    search.add_argument('--query', required=True)
    search.add_argument('--actor-id')
    search.add_argument('--before', help='Latest source date/time in the configured timezone')
    search.add_argument('--after', help='Earliest source date/time in the configured timezone')
    search.add_argument('--limit', type=_positive)
    search.add_argument('--lexical-only', action='store_true', help='Search stored text without embedding requests')
    read = commands.add_parser('read', help='Read current retrievable originals without model calls; use audit for old full-reset generations')
    read.add_argument('--chat-id', type=_nonzero, required=True)
    read.add_argument('--message-id', type=_positive, action='append', required=True, help='Internal memory message ID; may repeat')
    read.add_argument('--offset', type=int, default=0)
    read.add_argument('--length', type=_positive)
    read.add_argument('--neighbors', action='store_true')
    image = commands.add_parser('image', help='Export one retained compressed image from currently retrievable memory without model calls')
    image.add_argument('--chat-id', type=_nonzero, required=True)
    image.add_argument('--message-id', type=_positive, required=True, help='Internal memory message ID')
    image.add_argument('--image-id', required=True, help='Image ID shown beside a memory read/search result')
    image.add_argument('--output', type=Path, required=True, help='Destination file for the retained compressed image')
    audit = commands.add_parser('audit', help='Stream original revisions, including old generations and hidden/deleted messages')
    audit.add_argument('--chat-id', type=_nonzero, required=True)
    audit_source = audit.add_mutually_exclusive_group()
    audit_source.add_argument('--message-id', type=_positive, help='Internal PostgreSQL message ID, not Telegram message ID')
    audit_source.add_argument('--telegram-message-id', type=int, help='Telegram message ID in this chat, including historical export IDs and split reply aliases')
    audit.add_argument('--generation', type=_positive)
    audit.add_argument('--state', action='store_true', help='Include retired generation settings/persona and retained hosted Batch identities')
    audit.add_argument('--after-message-id', type=int, default=0, help='Resume after the pair (internal message ID, revision)')
    audit.add_argument('--after-revision', type=int, default=0)
    return result


def main() -> None:
    arguments = parser()
    args = arguments.parse_args()
    if args.command == 'audit' and (args.after_message_id < 0 or args.after_revision < 0):
        arguments.error('audit cursor values must be nonnegative')
    try:
        asyncio.run(_run(args))
    except KeyboardInterrupt:
        arguments.exit(130, 'Memory worker stopped; durable jobs remain resumable.\n')
    except (ValueError, RuntimeError, OSError, DatabaseError) as exc:
        arguments.exit(1, f'Memory operation stopped: {exc}\n')


if __name__ == '__main__':
    main()
