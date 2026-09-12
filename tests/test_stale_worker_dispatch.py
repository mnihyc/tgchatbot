"""An edited batch member must not abandon other still-current originals."""
from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.memory_worker import MemoryWorker
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind
from tgchatbot.storage.postgres_store import StaleScopeError


class StaleWorkerDispatchTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.settings()
        self.store.config = replace(self.store.config, job_retry_delay_seconds=0)
        self.embeddings = SimpleNamespace(enabled=True, space_id='stale-dispatch-fixture',
            count_tokens=AsyncMock(side_effect=lambda text: len(text.encode())),
            embed_documents=AsyncMock(side_effect=lambda items: [[1.0] + [0.0] * 1535 for _ in items]))
        self.worker = MemoryWorker(store=self.store, embeddings=self.embeddings,
            providers={'openai': self.provider}, config=self.config)

    async def source(self, number, text, *, enriched=None):
        parts = [MessagePart(PartKind.TEXT, text=text)]
        if enriched is not None:
            parts.append(MessagePart(PartKind.IMAGE, mime_type='image/png',
                data_b64='YWJj' if enriched else None, origin='attachment_reference'))
        return await self.store.append_message(self.session, ConversationMessage(MessageRole.USER, parts,
            metadata={'source': 'telegram', 'source_chat_id': '100', 'source_message_id': str(number),
                'actor_id': f'telegram:user:{number}', 'actor_kind': 'user', 'actor_name': 'Participant',
                'topic_id': 'same-topic', 'sent_at': '2026-01-01T00:00:00+00:00'}))

    async def job(self, job_id):
        async with self.store.pool.connection() as conn:
            return await (await conn.execute('SELECT * FROM jobs WHERE id=%s', (job_id,))).fetchone()

    async def test_edited_media_during_claimed_group_defers_current_text_and_recovers_exact_coverage(self):
        text = await self.source(1, 'The unchanged neighboring message must remain searchable.')
        media = await self.source(2, 'A photo awaiting enrichment.', enriched=False)
        jobs = await self.store.claim_jobs(kind='memory_ingest', limit=2)
        entered, release = asyncio.Event(), asyncio.Event()
        build = self.worker.builder.build
        async def gated(*args, **kwargs):
            entered.set()
            await release.wait()
            return await build(*args, **kwargs)
        with patch.object(self.worker.builder, 'build', side_effect=gated):
            task = asyncio.create_task(self.worker._guarded(jobs, self.worker._ingest))
            try:
                await asyncio.wait_for(entered.wait(), timeout=3)
                current_media = await self.source(2, 'A photo awaiting enrichment.', enriched=True)
                release.set()
                self.assertFalse(await asyncio.wait_for(task, timeout=3))
            finally:
                release.set()
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        by_source = {job['source_ids'][0]: await self.job(job['id']) for job in jobs}
        self.assertEqual(by_source[text.db_id]['status'], 'pending')
        self.assertEqual(by_source[text.db_id]['attempts'], 0, 'Another source edit does not spend this job retry budget')
        self.assertEqual(by_source[media.db_id]['status'], 'stale')
        retry = await self.store.claim_jobs(kind='memory_ingest', limit=10)
        self.assertEqual({mid for job in retry for mid in job['source_ids']}, {text.db_id, media.db_id})
        self.assertTrue(await self.worker._guarded(retry, self.worker._ingest))
        async with self.store.pool.connection() as conn:
            await conn.execute("UPDATE excerpt_tails SET updated_at=now()-interval '1 day'")
        tails = await self.store.claim_jobs(kind='memory_tail', limit=1)
        self.assertTrue(await self.worker._guarded(tails, self.worker._tail))
        embeds = await self.store.claim_jobs(kind='memory_embed', limit=10)
        await self.worker._guarded(embeds, self.worker._embed)
        async with self.store.pool.connection() as conn:
            excerpts = await (await conn.execute('SELECT source_ids,source_revisions,spans,embedding IS NOT NULL AS embedded '
                'FROM excerpts WHERE valid ORDER BY id')).fetchall()
        expected = {text.db_id: text, media.db_id: current_media}
        self.assertEqual({mid for row in excerpts for mid in row['source_ids']}, set(expected))
        self.assertTrue(all(row['embedded'] for row in excerpts))
        for message_id, original in expected.items():
            body = '\n'.join(part.text for part in original.message.parts if part.text is not None)
            spans = sorted((span for row in excerpts for span in row['spans'] if span['message_id'] == message_id),
                key=lambda span: span['start'])
            self.assertEqual(''.join(body[span['start']:span['end']] for span in spans),
                ''.join(part.text for part in original.message.parts if part.text is not None),
                'Cover every original part character; storage-added separators are not source spans')
            self.assertTrue(all(row['source_revisions'][str(message_id)] == original.message.metadata['source_revision']
                for row in excerpts if message_id in row['source_ids']))
        self.assertEqual(self.provider.requests, [])

    async def test_stale_dependency_deferral_keeps_durable_operation_payload_and_refunds_attempt(self):
        queued = await self.store.enqueue_job(self.session, 'operation-fixture', payload={'phase': 'prepared'})
        job = (await self.store.claim_jobs(kind='operation-fixture'))[0]
        durable = {'phase': 'polling', 'name': 'operations/already-submitted'}
        async def operation(jobs):
            self.assertTrue(await self.store.update_job_payload(jobs[0], durable))
            raise StaleScopeError('A borrowed dependency changed after intent was saved')
        self.assertFalse(await self.worker._guarded([job], operation))
        stored = await self.job(queued['id'])
        self.assertEqual(stored['status'], 'pending')
        self.assertEqual(stored['payload'], durable, 'Original claim payload must not rewind persisted external intent')
        self.assertEqual(stored['attempts'], 0)
        next_job = (await self.store.claim_jobs(kind='operation-fixture'))[0]
        self.assertTrue(await self.store.defer_job(next_job, {}, 0))
        self.assertEqual((await self.job(queued['id']))['payload'], {}, 'Explicit empty payload still replaces data')

    async def test_old_lease_cannot_defer_or_rewrite_a_new_owner(self):
        queued = await self.store.enqueue_job(self.session, 'lease-fixture', payload={'phase': 'prepared'})
        old = (await self.store.claim_jobs(kind='lease-fixture'))[0]
        async with self.store.pool.connection() as conn:
            await conn.execute("UPDATE jobs SET lease_until=now()-interval '1 second' WHERE id=%s", (queued['id'],))
        current = (await self.store.claim_jobs(kind='lease-fixture'))[0]
        await self.store.update_job_payload(current, {'phase': 'owned-by-new-lease'})
        before = await self.job(queued['id'])
        async def stale(_jobs):
            raise StaleScopeError('Old operation finished after another worker reclaimed it')
        await self.worker._guarded([old], stale)
        self.assertEqual(await self.job(queued['id']), before)

    async def test_heartbeat_ownership_loss_keeps_sibling_recoverable_by_lease_expiry(self):
        text = await self.source(1, 'This sibling still needs its projection.')
        media = await self.source(2, 'Pending media.', enriched=False)
        jobs = await self.store.claim_jobs(kind='memory_ingest', limit=2, lease_seconds=1)
        text_job = next(job for job in jobs if job['source_ids'] == [text.db_id])
        ordinary_limits = self.worker.limits
        self.worker.limits = replace(ordinary_limits, heartbeat_seconds=0.01, lease_seconds=1)
        entered = asyncio.Event()
        async def blocked(_jobs):
            entered.set()
            await asyncio.Event().wait()
        task = asyncio.create_task(self.worker._guarded(jobs, blocked))
        try:
            await asyncio.wait_for(entered.wait(), timeout=3)
            await self.source(2, 'Pending media.', enriched=True)
            self.assertFalse(await asyncio.wait_for(task, timeout=3))
        finally:
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        sibling = await self.job(text_job['id'])
        self.assertEqual(sibling['status'], 'running', 'Heartbeat cancellation keeps the existing lease recovery policy')
        self.assertEqual(sibling['payload'], text_job['payload'])
        self.worker.limits = ordinary_limits
        async with self.store.pool.connection() as conn:
            await conn.execute("UPDATE jobs SET lease_until=now()-interval '1 second' WHERE id=%s", (text_job['id'],))
        reclaimed = await self.store.claim_jobs(kind='memory_ingest', limit=10)
        self.assertEqual({mid for job in reclaimed for mid in job['source_ids']}, {text.db_id, media.db_id})
        self.assertTrue(await self.worker._guarded(reclaimed, self.worker._ingest))
        self.assertEqual(set((await self.store.get_excerpt_tail(self.session, 'same-topic'))['source_ids']),
            {text.db_id, media.db_id})

    async def test_full_reset_retires_old_job_without_requeueing_audit_sources(self):
        source = await self.source(1, 'Old-generation original retained for audit.')
        jobs = await self.store.claim_jobs(kind='memory_ingest')
        async def reset(_jobs):
            await self.store.reset_full(self.session, self.config.default_session_settings())
            raise StaleScopeError('Generation reset before projection commit')
        await self.worker._guarded(jobs, reset)
        self.assertEqual((await self.job(jobs[0]['id']))['status'], 'stale')
        self.assertEqual(await self.store.claim_jobs(kind='memory_ingest'), [])
        self.assertEqual(await self.store.read_messages(self.session, [source.db_id]), [])
        self.assertEqual(len(await self.store.list_message_revisions(self.session, source.db_id)), 1)

    async def test_validation_and_service_failures_keep_their_existing_retry_policy(self):
        for error, expected_status in ((ValueError('Invalid model output'), 'failed'),
                                       (TimeoutError('Endpoint temporarily unavailable'), 'pending')):
            with self.subTest(error=type(error).__name__):
                queued = await self.store.enqueue_job(self.session, type(error).__name__)
                job = (await self.store.claim_jobs(kind=type(error).__name__))[0]
                async def fail(_jobs):
                    raise error
                await self.worker._guarded([job], fail)
                stored = await self.job(queued['id'])
                self.assertEqual(stored['status'], expected_status)
                self.assertEqual(stored['attempts'], 1)
