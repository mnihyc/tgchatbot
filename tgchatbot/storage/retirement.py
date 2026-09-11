"""Bounded retirement of disposable semantic vectors after a full reset.

Source messages/revisions, profiles, compacted audit evidence, and hosted Batch
identities are retained. PostgreSQL vacuum can reuse retired pages; this does not
promise immediate filesystem shrinkage or run disruptive VACUUM FULL.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from tgchatbot.storage.postgres_store import PostgresStore, StaleScopeError

async def retire_excerpt_chunk(store: PostgresStore, job: Mapping[str, Any], *, limit: int | None = None) -> int:
    limit = store.config.retirement_page_size if limit is None else limit
    if type(limit) is not int or limit < 1:
        raise ValueError('Retirement page must contain a positive number of rows')
    before = job['payload'].get('before_generation')
    if isinstance(before, bool) or not isinstance(before, int) or before != job['generation']:
        raise ValueError('Retirement boundary must equal the job generation; active generation is never retired')
    async with store.pool.connection() as conn:
        # Use the same session-before-job lock ordering and lease/source guard as
        # every durable worker commit. A second reset cannot race this transaction.
        if await store._locked_job(conn, job) is None:
            raise StaleScopeError('Retirement lease or generation changed')
        result = await conn.execute('''WITH candidates AS (
            SELECT id FROM excerpts WHERE session_id=%s AND generation<%s
            AND (valid OR embedding IS NOT NULL) ORDER BY generation,id LIMIT %s
            FOR UPDATE)
            UPDATE excerpts e SET valid=false,embedding=NULL FROM candidates c WHERE e.id=c.id''',
            (job['session_id'], before, limit))
        return result.rowcount
