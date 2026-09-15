"""Canonical delivered speech whose replay belongs to an existing model batch."""
from __future__ import annotations

from psycopg.types.json import Jsonb


async def link_replay_owner(conn, *, session_id, scope, message_id, owner_message_id):
    # This relation is meaningful only within the same live model exchange.
    owner = await (await conn.execute('''SELECT m.id FROM messages m JOIN message_revisions r
        ON (r.message_id,r.revision)=(m.id,m.source_revision)
        WHERE m.id=%s AND m.session_id=%s AND m.generation=%s AND m.context_id=%s
        AND NOT m.hidden AND NOT m.deleted AND m.role='tool'
        AND r.metadata->>'tool_phase'='call' ''',
        (owner_message_id, session_id, scope['generation'], scope['context_id']))).fetchone()
    if owner is None:
        from tgchatbot.storage.postgres_store import StaleScopeError
        raise StaleScopeError('The delivered speech no longer belongs to an active model batch')
    await conn.execute('''INSERT INTO message_replay_owners(message_id,owner_message_id)
        VALUES (%s,%s) ON CONFLICT(message_id) DO NOTHING''', (message_id, owner_message_id))


async def detach_replay_owners(conn, *, session_id, generation, source_ids):
    """A withdrawn original cannot survive inside its native replay snapshot."""
    links = await (await conn.execute('''SELECT link.owner_message_id
        FROM message_replay_owners link JOIN messages m ON m.id=link.message_id
        WHERE m.session_id=%s AND m.generation=%s AND NOT link.detached
        AND (link.message_id=ANY(%s) OR link.owner_message_id=ANY(%s))
        FOR UPDATE OF link''', (session_id, generation, source_ids, source_ids))).fetchall()
    owners = sorted({row['owner_message_id'] for row in links})
    if not owners:
        return source_ids
    await conn.execute('UPDATE message_replay_owners SET detached=true WHERE owner_message_id=ANY(%s)', (owners,))
    batches = await (await conn.execute('''SELECT r.metadata->>'tool_batch_id' AS batch
        FROM messages m JOIN message_revisions r ON (r.message_id,r.revision)=(m.id,m.source_revision)
        WHERE m.id=ANY(%s)''', (owners,))).fetchall()
    # Explicit withdrawal already invalidates the context cache. Retain the
    # calls/results through existing portable replay, without the withdrawn
    # prose or an opaque signature claiming the old exchange is unchanged.
    await conn.execute('''UPDATE messages m SET presentation=
        (COALESCE(m.presentation,'{}'::jsonb)-'provider_native') || %s
        FROM message_revisions r WHERE (r.message_id,r.revision)=(m.id,m.source_revision)
        AND m.session_id=%s AND m.generation=%s
        AND (m.id=ANY(%s) OR r.metadata->>'tool_batch_id'=ANY(%s))''',
        (Jsonb({'portable_tool_history': True, 'provider_native_skip_same_provider': False}),
         session_id, generation, owners, [row['batch'] for row in batches if row['batch']]))
    # Existing block invalidation follows the shared model owner as well as
    # the delivered original; neither body is rewritten or deleted.
    return sorted(set(source_ids) | set(owners))
