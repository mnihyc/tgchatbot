"""Observed delivery identities for one canonical answer with multiple chunks."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from tgchatbot.storage.postgres_store import PostgresStore, StaleScopeError


async def bind_message_source(store: PostgresStore, session_id: str, message_id: int, *,
                              source: str, source_chat_id: str, source_message_ids: Sequence[str],
                              actor_id: str, actor_kind: str, actor_name: str,
                              expected_scope: Mapping[str, Any] | None = None) -> None:
    if any(value is None for value in source_message_ids):
        raise ValueError('Source binding requires observed message identities')
    ids = list(dict.fromkeys(str(value) for value in source_message_ids))
    if (not ids or any(not value.strip() for value in ids)
            or any(not isinstance(value, str) or not value.strip()
                   for value in (source, source_chat_id, actor_id, actor_kind))):
        raise ValueError('Source binding requires observed identities and explicit source/actor fields')
    if source == 'telegram' and (any(not value.isascii() or not value.isdecimal() or not value.lstrip('0') for value in ids)
                                 or session_id != f'telegram:{source_chat_id}'):
        raise ValueError('Telegram delivery identifiers must be positive and belong to the bound chat')
    async with store.pool.connection() as conn:
        scope = await store._session(conn, session_id, lock=True)
        store._check_scope(scope, expected_scope)
        row = await (await conn.execute('''SELECT * FROM messages WHERE id=%s AND session_id=%s
            AND generation=%s AND context_id=%s AND NOT hidden AND NOT deleted FOR UPDATE''',
            (message_id, session_id, scope['generation'], scope['context_id']))).fetchone()
        if row is None:
            raise StaleScopeError('Delivered source is no longer active in this context')
        if row['source_message_id'] is not None and (
                row['source'], row['source_chat_id'], row['source_message_id']) != (source, source_chat_id, ids[0]):
            raise ValueError('An existing canonical source cannot be rebound to an unrelated delivery')
        conflict = await (await conn.execute('''SELECT id FROM messages WHERE session_id=%s AND generation=%s
            AND source=%s AND source_chat_id=%s AND source_message_id=ANY(%s) AND id<>%s LIMIT 1''',
            (session_id, scope['generation'], source, source_chat_id, ids, message_id))).fetchone()
        if conflict:
            raise ValueError('A delivery chunk already belongs to another canonical original')
        for source_id in ids:
            conflict = await (await conn.execute('''SELECT message_id FROM message_source_aliases
                WHERE session_id=%s AND generation=%s AND source=%s AND source_chat_id=%s AND source_message_id=%s''',
                (session_id, scope['generation'], source, source_chat_id, source_id))).fetchone()
            if conflict and conflict['message_id'] != message_id:
                raise ValueError('A delivery chunk alias already belongs to another original')
        await conn.execute('''UPDATE messages SET source=%s,source_chat_id=%s,source_message_id=%s,
            actor_id=%s,actor_kind=%s,actor_name=%s WHERE id=%s''',
            (source, source_chat_id, ids[0], actor_id, actor_kind, actor_name, message_id))
        for source_id in ids[1:]:
            await conn.execute('''INSERT INTO message_source_aliases
                (session_id,generation,source,source_chat_id,source_message_id,message_id)
                VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING''',
                (session_id, scope['generation'], source, source_chat_id, source_id, message_id))
        await conn.execute('UPDATE sessions SET context_version=context_version+1 WHERE session_id=%s', (session_id,))
