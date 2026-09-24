"""Read-only inspection selectors; PostgreSQL owns scope and source membership."""
from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict

from tgchatbot.core.context_state import LiveConversationState
from tgchatbot.domain.identities import canonical_actor_id
from tgchatbot.domain.timestamps import parse_time_filter
from tgchatbot.storage.postgres_store import _timestamp


@asynccontextmanager
async def inspection_snapshot(store, session_id, defaults, *, timezone=None):
    async with store.pool.connection() as conn:
        await conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        row = await (await conn.execute('SELECT *, transaction_timestamp() AS as_of FROM sessions WHERE session_id=%s',
                                       (session_id,))).fetchone()
        if row is None:
            raise ValueError('Chat does not exist; import or receive messages first.')
        reader = InspectionReader(store, conn, session_id, row, defaults)
        reader.timezone = timezone
        yield reader
    await store.assert_scope(session_id, reader.scope)


class InspectionReader:
    def __init__(self, store, conn, session_id, row, defaults):
        self.store, self.conn, self.session_id = store, conn, session_id
        self.scope = store._scope(row)
        self.as_of, self.row = row['as_of'], row
        self.settings = store.decode_settings(row['settings'], defaults)
        self.timezone = None
        self._state = None

    async def state(self):
        if self._state is None:
            blocks, messages, version = await self.store.read_live_context(self.conn, self.session_id)
            self._state = LiveConversationState(session_id=self.session_id, blocks=blocks,
                raw_messages=messages, loaded=True, database_version=version)
            self._state.rebuild_estimate()
        return self._state

    async def calibration(self):
        key = (self.settings.provider, self.settings.model, self.settings.tool_history_mode.value)
        row = await (await self.conn.execute('''SELECT multiplier FROM provider_token_calibration
            WHERE (provider,model,history_mode)=(%s,%s,%s)''', key)).fetchone()
        return max(1., float(row['multiplier'])) if row else 1.

    def message_record(self, row):
        return {'message': self.store._message(row),
                'presentation': self.store._message(row, presentation=True),
                'context_id': row['context_id'],
                'current_context': row['context_id'] == self.scope['context_id'],
                'compacted_by': row['compacted_by_block_id'],
                'replay_owner': row.get('context_owner_message_id') if not row.get('context_replay_detached') else None}

    async def message(self, message_id, *, current_context=False):
        query = self.store._select_message() + ' WHERE m.session_id=%s AND m.id=%s AND NOT m.hidden AND NOT m.deleted'
        if current_context:
            query += ' AND m.context_id=s.context_id'
        row = await (await self.conn.execute(query, (self.session_id, message_id))).fetchone()
        if row is None:
            raise ValueError('Message is unavailable in this chat scope. Open a fresh /context view.')
        return self.message_record(row)

    async def tool_result(self, call):
        """Use the recorded call owner; legacy rows also require the same batch."""
        stored = call['message']
        meta = stored.message.metadata
        query = self.store._select_message() + ''' WHERE m.session_id=%s AND m.context_id=%s
            AND m.role='tool' AND m.id>%s AND NOT m.hidden AND NOT m.deleted
            AND r.metadata->>'tool_phase'='result'
            AND (r.metadata->>'tool_call_message_id'=%s
                OR (NOT r.metadata ? 'tool_call_message_id' AND m.actor_name IS NOT DISTINCT FROM %s
                    AND r.metadata#>>'{tool_payload,call_id}'=%s
                    AND r.metadata->>'tool_batch_id' IS NOT DISTINCT FROM %s
                    AND NOT EXISTS (
                        SELECT 1 FROM messages other JOIN message_revisions original
                            ON (original.message_id,original.revision)=(other.id,other.source_revision)
                        WHERE other.session_id=m.session_id AND other.generation=m.generation
                            AND other.context_id=m.context_id AND other.role='tool'
                            AND NOT other.hidden AND NOT other.deleted AND other.id<m.id AND other.id<>%s
                            AND other.actor_name IS NOT DISTINCT FROM m.actor_name
                            AND original.metadata->>'tool_phase'='call'
                            AND original.metadata#>>'{tool_payload,call_id}'=r.metadata#>>'{tool_payload,call_id}'
                            AND original.metadata->>'tool_batch_id' IS NOT DISTINCT FROM r.metadata->>'tool_batch_id')))
            ORDER BY m.id LIMIT 1'''
        row = await (await self.conn.execute(query, (self.session_id, call['context_id'], stored.db_id,
            str(stored.db_id), stored.message.name, (meta.get('tool_payload') or {}).get('call_id'),
            meta.get('tool_batch_id'), stored.db_id))).fetchone()
        return self.message_record(row) if row else None

    async def recent(self, *, limit, before_id=None, source_time=False, actor_id=None, before=None, after=None):
        query = self.store._select_message() + ''' WHERE m.session_id=%s AND NOT m.hidden AND NOT m.deleted
            AND m.role IN ('user','assistant') AND NOT (r.metadata ? 'synthetic_role')'''
        values = [self.session_id]
        if not source_time:
            query += ' AND m.context_id=s.context_id'
        if before_id is not None:
            anchor = await self.message(before_id, current_context=not source_time)
            if source_time:
                query += ' AND (m.sent_at,m.id)<(%s,%s)'
                values.extend((_timestamp(anchor['message'].message.metadata['sent_at']), before_id))
            else:
                query += ' AND m.id<%s'
                values.append(before_id)
        if actor_id is not None:
            query += ' AND m.actor_id=%s'
            values.append(canonical_actor_id(actor_id))
        for at, operator in ((before, '<'), (after, '>=')):
            if at is not None:
                query += f' AND m.sent_at{operator}%s'
                values.append(parse_time_filter(at, self.timezone))
        query += (' ORDER BY m.sent_at DESC,m.id DESC' if source_time else ' ORDER BY m.id DESC') + ' LIMIT %s'
        values.append(limit + 1)
        rows = await (await self.conn.execute(query, values)).fetchall()
        return {'items': [self.message_record(row) for row in rows[:limit]],
                'next': rows[limit - 1]['id'] if len(rows) > limit else None}

    async def block(self, block_id, *, current_context=False):
        query = '''SELECT b.*,cardinality(b.source_ids) AS source_count FROM memory_blocks b
            JOIN sessions s ON s.session_id=b.session_id AND s.generation=b.generation
            WHERE b.session_id=%s AND b.id=%s AND b.valid'''
        if current_context:
            query += ' AND b.context_id=s.context_id'
        row = await (await self.conn.execute(query, (self.session_id, block_id))).fetchone()
        if row is None:
            raise ValueError('Summary is unavailable in this chat scope. Open a fresh /context view.')
        return {**asdict(self.store._block(row)), 'source_ids': row['source_ids'],
                'current_context': row['context_id'] == self.scope['context_id']}

    async def profile_actors(self, *, limit, after=None, included=()):
        actors = await (await self.conn.execute('''SELECT actor_id FROM (
            SELECT actor_id FROM profile_current WHERE session_id=%s AND generation=%s
            UNION SELECT actor_id FROM profile_inputs WHERE session_id=%s AND generation=%s AND pending_bytes>0
            UNION SELECT unnest(%s::text[])
            ) actors WHERE actor_id>%s ORDER BY actor_id LIMIT %s''',
            (self.session_id, self.scope['generation'], self.session_id, self.scope['generation'],
             list(set(included) | {'agent'}), canonical_actor_id(after) if after else '', limit + 1))).fetchall()
        return [row['actor_id'] for row in actors[:limit]], (actors[limit - 1]['actor_id'] if len(actors) > limit else None)

    async def profiles(self, actors):
        snapshot = await self.store.read_profile_snapshot(self.conn, self.session_id, actors,
            expected_scope=self.scope, include_pending=True, as_of=self.as_of)
        dates = await (await self.conn.execute('''SELECT actor_id,updated_at FROM profile_current
            WHERE session_id=%s AND generation=%s AND actor_id=ANY(%s)''',
            (self.session_id, self.scope['generation'], actors))).fetchall()
        snapshot['updated_at'] = {row['actor_id']: row['updated_at'] for row in dates}
        return snapshot

    async def jobs(self):
        return await self.store.read_job_status(self.conn, self.session_id)
