"""Audit-only leases for hosted operations whose source scope was invalidated."""
from __future__ import annotations

import uuid

from psycopg.types.json import Jsonb
from tgchatbot.storage.postgres_store import _duration


# A timed-out submission may already be paid even without an operation name.
# Terminal retired metadata remains auditable but no longer reserves capacity.
UNRESOLVED_PAID = """(COALESCE(j.payload->>'name','')<>''
    OR j.payload->>'phase' IN ('submitting','polling'))
    AND COALESCE(j.payload->>'phase','')<>'retired_terminal'
    AND j.status IN ('pending','running','failed','stale')"""


async def pending_batches(store, *, include_prepared=False, exclude_id=None) -> int:
    prepared = """ OR (j.generation=s.generation AND j.status IN ('pending','running')
        AND j.payload->>'phase'='prepared')""" if include_prepared else ''
    async with store.pool.connection() as conn:
        row = await (await conn.execute(f'''SELECT count(*) AS n FROM jobs j
            JOIN sessions s ON s.session_id=j.session_id
            WHERE j.kind='embedding_batch' AND (%s::bigint IS NULL OR j.id<>%s)
            AND (({UNRESOLVED_PAID}){prepared})''', (exclude_id, exclude_id))).fetchone()
        return row['n']


async def claim_retired_batch(store):
    # Ordinary workers intentionally cannot claim retired source generations.
    # This lease authorizes only operation-status observation, never source reads.
    async with store.pool.connection() as conn:
        return await (await conn.execute(f'''WITH candidate AS (
            SELECT j.id FROM jobs j JOIN sessions s ON s.session_id=j.session_id
            WHERE (j.generation<s.generation OR j.status='stale'
                OR j.payload->>'retired_reconciliation'='true') AND j.kind='embedding_batch'
            AND ({UNRESOLVED_PAID}) AND j.available_at<=now()
            AND (j.status IN ('pending','failed','stale')
                OR (j.status='running' AND j.lease_until<now()))
            ORDER BY j.available_at,j.id FOR UPDATE OF j SKIP LOCKED LIMIT 1)
            UPDATE jobs j SET status='running',lease_token=%s,
                payload=j.payload || '{{"retired_reconciliation":true}}'::jsonb,
                lease_until=now()+(%s * interval '1 second')
            FROM candidate c WHERE j.id=c.id RETURNING j.*''',
            (str(uuid.uuid4()), store.config.retired_batch_lease_seconds))).fetchone()


async def save_retired_batch(store, job, payload, *, terminal=False, error=None, delay_seconds=None):
    delay_seconds = _duration(store.config.retired_batch_poll_seconds if delay_seconds is None else delay_seconds)
    async with store.pool.connection() as conn:
        cursor = await conn.execute('''UPDATE jobs j SET payload=%s,status=%s,error=%s,
            lease_token=NULL,lease_until=NULL,available_at=now()+(%s * interval '1 second'),
            finished_at=CASE WHEN %s THEN now() ELSE NULL END
            FROM sessions s WHERE j.session_id=s.session_id
            AND (j.generation<s.generation OR j.payload->>'retired_reconciliation'='true')
            AND j.id=%s AND j.status='running' AND j.lease_token=%s AND j.lease_until>now()''',
            (Jsonb(payload), 'stale', error, delay_seconds,
             terminal, job['id'], job['lease_token']))
        return cursor.rowcount == 1
