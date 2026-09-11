"""Durable outcome of a selected sticker; Telegram acceptance is not transactional."""
from __future__ import annotations

from pathlib import Path

from psycopg.types.json import Jsonb


class StickerDeliveryStore:
    def __init__(self, store):
        self.store = store

    async def initialize(self, *, recover_interrupted: bool = False) -> None:
        async with self.store.pool.connection() as conn:
            await conn.execute(Path(__file__).with_suffix('.sql').read_text())
            if recover_interrupted:
                # Called once by the application's startup owner, never during
                # ordinary catalog reads or construction of a second handle.
                await conn.execute("""UPDATE sticker_deliveries SET status='unknown',
                    error='interrupted_delivery',finished_at=now() WHERE status='sending'""")

    async def queue(self, session_id, sticker_id, *, operation_id, expected_scope, timing, metadata=None):
        async with self.store.pool.connection() as conn:
            scope = await self.store._session(conn, session_id, lock=True)
            self.store._check_scope(scope, expected_scope)
            await conn.execute('''INSERT INTO sticker_deliveries
                (operation_id,session_id,generation,context_id,revision,sticker_id,timing,status,metadata)
                VALUES (%s,%s,%s,%s,%s,%s,%s,'queued',%s) ON CONFLICT DO NOTHING''',
                (operation_id,session_id,scope['generation'],scope['context_id'],scope['revision'],
                 sticker_id,timing,Jsonb(metadata or {})))
            row = await (await conn.execute('SELECT * FROM sticker_deliveries WHERE operation_id=%s',
                (operation_id,))).fetchone()
            if (row['session_id'],row['generation'],row['context_id'],row['sticker_id']) != (
                    session_id,scope['generation'],scope['context_id'],sticker_id):
                raise ValueError('Delivery operation already belongs to another selection')
            return row

    async def begin(self, operation_id):
        """Only the winner may send. A repeated/unknown operation never retries."""
        async with self.store.pool.connection() as conn:
            row = await (await conn.execute('SELECT * FROM sticker_deliveries WHERE operation_id=%s',
                (operation_id,))).fetchone()
            if row is None:
                raise ValueError('Unknown delivery operation')
            scope = await self.store._session(conn, row['session_id'], lock=True)
            row = await (await conn.execute('SELECT * FROM sticker_deliveries WHERE operation_id=%s FOR UPDATE',
                (operation_id,))).fetchone()
            if row['status'] != 'queued':
                return {**row, 'may_send': False}
            stale = any(row[key] != scope[key] for key in ('generation','context_id','revision'))
            row = await (await conn.execute('''UPDATE sticker_deliveries SET status=%s,
                error=%s,started_at=CASE WHEN %s THEN started_at ELSE now() END,
                finished_at=CASE WHEN %s THEN now() ELSE NULL END
                WHERE operation_id=%s RETURNING *''',
                ('failed' if stale else 'sending','scope_changed_before_delivery' if stale else None,
                 stale,stale,operation_id))).fetchone()
            return {**row, 'may_send': not stale}

    async def finish(self, operation_id, status, *, telegram_message_id=None, error=None):
        if status not in {'sent','failed','unknown'}:
            raise ValueError('Invalid delivery outcome')
        async with self.store.pool.connection() as conn:
            # Deliberately no active-generation predicate: an old operation's
            # late acknowledgment belongs in its original audit scope.
            row = await (await conn.execute('''UPDATE sticker_deliveries SET status=%s,
                telegram_message_id=%s,error=%s,finished_at=now()
                WHERE operation_id=%s AND status IN ('sending','unknown') RETURNING *''',
                (status,telegram_message_id,error,operation_id))).fetchone()
            if row is None:
                row = await (await conn.execute('SELECT * FROM sticker_deliveries WHERE operation_id=%s',
                    (operation_id,))).fetchone()
            return row

    async def get(self, operation_id):
        async with self.store.pool.connection() as conn:
            return await (await conn.execute('SELECT * FROM sticker_deliveries WHERE operation_id=%s',
                (operation_id,))).fetchone()

    async def recent(self, session_id, *, limit=20):
        async with self.store.pool.connection() as conn:
            return await (await conn.execute('''SELECT d.* FROM sticker_deliveries d
                JOIN sessions s USING(session_id) WHERE d.session_id=%s
                AND d.generation=s.generation AND d.status='sent'
                ORDER BY d.finished_at DESC,d.operation_id LIMIT %s''', (session_id,limit))).fetchall()

    async def unresolved(self, session_id, *, limit=20):
        async with self.store.pool.connection() as conn:
            return await (await conn.execute('''SELECT d.* FROM sticker_deliveries d
                JOIN sessions s USING(session_id) WHERE d.session_id=%s
                AND d.generation=s.generation AND d.status IN ('sending','unknown')
                ORDER BY d.created_at DESC,d.operation_id LIMIT %s''', (session_id,limit))).fetchall()
