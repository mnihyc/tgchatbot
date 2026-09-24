"""Bounded conversation expansion; original time, chat and epoch govern relations."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


async def expand_message_ids(store, session_id: str, message_ids: Sequence[int], *,
                             expected_scope: Mapping[str, Any] | None = None,
                             neighbors: int | None = None, limit: int | None = None,
                             include_selected: bool = True) -> list[int]:
    """Keep selected sources first, then explicit replies and nearby originals.

    The caller's read window determines the total result size. Neighbor defaults
    come from the existing environment, independently of the archive's size.
    Import order never determines proximity; timestamps do, with IDs breaking ties.
    Search can request only related originals without spending that read window
    on sources it has already returned. The selected seed window remains bounded.
    """
    ids = list(dict.fromkeys(int(value) for value in message_ids))
    neighbors = store.config.relationship_neighbors if neighbors is None else neighbors
    limit = store.config.relationship_read_limit if limit is None else limit
    if not ids or any(value <= 0 for value in ids):
        raise ValueError('Expand positive message IDs')
    if type(neighbors) is not int or neighbors < 0 or type(limit) is not int or limit < 1:
        raise ValueError('Neighbor count must be nonnegative and read length positive')
    async with store.pool.connection() as conn:
        scope = await store.read_session_scope(conn, session_id)
        if scope is None:
            return []
        store._check_scope(scope, expected_scope)
        seeds = await (await conn.execute('''SELECT m.id,m.source,m.source_chat_id,m.reply_to_source_id,
            m.topic_id,m.sent_at,r.metadata FROM messages m JOIN message_revisions r
            ON (r.message_id,r.revision)=(m.id,m.source_revision)
            WHERE m.session_id=%s AND m.generation=%s AND m.id=ANY(%s) AND NOT m.hidden AND NOT m.deleted''',
            (session_id, scope['generation'], ids))).fetchall()
        by_id = {row['id']: row for row in seeds}
        selected = [message_id for message_id in ids if message_id in by_id][:limit]
        result = list(selected) if include_selected else []
        seen = set(ids)
        if len(result) == limit:
            return result
        for message_id in selected:
            seed = by_id[message_id]
            reply_chat = seed['metadata'].get('reply_to_source_chat_id', seed['source_chat_id'])
            if seed['reply_to_source_id'] is not None and reply_chat == seed['source_chat_id']:
                reply = await (await conn.execute('''SELECT id FROM messages WHERE session_id=%s
                    AND generation=%s AND source=%s AND source_chat_id IS NOT DISTINCT FROM %s
                    AND source_message_id=%s AND NOT hidden AND NOT deleted LIMIT 1''',
                    (session_id, scope['generation'], seed['source'], seed['source_chat_id'],
                     seed['reply_to_source_id']))).fetchone()
                if reply is None:
                    # Telegram may split one delivered answer into multiple real
                    # messages. Replies to later chunks point to the same original.
                    reply = await (await conn.execute('''SELECT m.id FROM message_source_aliases a
                        JOIN messages m ON m.id=a.message_id
                        WHERE a.session_id=%s AND a.generation=%s AND a.source=%s
                        AND a.source_chat_id IS NOT DISTINCT FROM %s AND a.source_message_id=%s
                        AND m.session_id=a.session_id AND m.generation=a.generation
                        AND NOT m.hidden AND NOT m.deleted LIMIT 1''',
                        (session_id, scope['generation'], seed['source'], seed['source_chat_id'],
                         seed['reply_to_source_id']))).fetchone()
                if reply and reply['id'] not in seen:
                    result.append(reply['id'])
                    seen.add(reply['id'])
                    if len(result) == limit:
                        return result
        for message_id in selected:
            seed = by_id[message_id]
            if neighbors:
                topic_predicate = 'topic_id IS NULL' if seed['topic_id'] is None else 'topic_id=%s'
                parameters = [session_id, scope['generation']]
                if seed['topic_id'] is not None:
                    parameters.append(seed['topic_id'])
                parameters.extend([seed['sent_at'], seed['id'], neighbors])
                # Each side is an indexed seek with LIMIT, even in a sparse topic.
                for operator, direction in (('<', 'DESC'), ('>', 'ASC')):
                    rows = await (await conn.execute(f'''SELECT id FROM messages WHERE session_id=%s
                        AND generation=%s AND {topic_predicate} AND NOT hidden AND NOT deleted
                        AND (sent_at,id) {operator} (%s,%s) ORDER BY sent_at {direction},id {direction} LIMIT %s''',
                        parameters)).fetchall()
                    for row in rows:
                        if row['id'] not in seen:
                            result.append(row['id'])
                            seen.add(row['id'])
                            if len(result) == limit:
                                return result
        return result
