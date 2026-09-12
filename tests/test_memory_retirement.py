from __future__ import annotations

import os
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.memory_worker import MemoryWorker
from tgchatbot.domain.models import ConversationMessage
from tgchatbot.domain.provenance import original_text
from tgchatbot.embeddings import BatchJob
from tgchatbot.storage.postgres_store import StaleScopeError
from tgchatbot.storage.retirement import retire_excerpt_chunk
from tgchatbot.tools.memory import audit_records, audit_state_records


class GenerationRetirementTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.settings()
        self.space = 'synthetic-space'
        self.vector = np.array([1.] + [0.] * 1535)
        self.embeddings = SimpleNamespace(enabled=True, space_id=self.space,
            count_tokens=AsyncMock(side_effect=NotImplementedError),
            submit_batch=AsyncMock(), find_batch=AsyncMock(), poll_batch=AsyncMock(), read_batch_results=AsyncMock())
        self.worker = MemoryWorker(store=self.store, embeddings=self.embeddings, providers={}, config=self.config)

    async def source(self, text='An immutable original.', source_id=1, *, session=None):
        session = session or self.session
        await self.store.get_or_create_session(session, self.config.default_session_settings())
        return await self.store.append_message(session, ConversationMessage.user_text(text, metadata={
            'source': 'telegram', 'source_chat_id': session.split(':')[-1], 'source_message_id': str(source_id),
            'actor_id': 'telegram:user:7', 'actor_name': 'Alex'}))

    async def test_full_reset_snapshots_settings_and_persona_before_defaults_and_audit_remains_separate(self):
        await self.settings(system_prompt='Speak like the old librarian.', model='old-model')
        persona = {'name': 'Sleepy fox', 'summary': 'A gentle chat-specific persona'}
        await self.store.save_sticker_persona(self.session, persona)
        source = await self.source()
        await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7', asserted_by='telegram:user:7',
                                           claim='An original claim', source_ids=[source.db_id])
        original_scope = await self.store.get_scope(self.session)
        defaults = self.config.default_session_settings()
        fresh_scope = await self.store.reset_full(self.session, defaults)
        settings = await self.store.get_or_create_session(self.session, defaults)
        self.assertEqual(settings.system_prompt, defaults.system_prompt)
        self.assertIsNone(await self.store.get_sticker_persona(self.session))
        self.assertEqual(await self.store.get_profile(self.session, 'telegram:user:7'), [])
        self.assertEqual(await self.store.read_messages(self.session, [source.db_id]), [])
        self.assertEqual(await self.store.search_messages(self.session, 'immutable'), [])
        self.assertEqual((await self.store.job_status(self.session))[0]['kind'], 'memory_retire')
        snapshots = [row async for row in audit_state_records(self.store, self.session)]
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(snapshots[0]['settings']['system_prompt'], 'Speak like the old librarian.')
        self.assertEqual(snapshots[0]['settings']['model'], 'old-model')
        self.assertEqual(snapshots[0]['sticker_persona'], persona)
        self.assertEqual(snapshots[0]['generation'], original_scope['generation'])
        self.assertNotEqual(fresh_scope['generation'], original_scope['generation'])
        self.assertEqual([row['body'] async for row in audit_records(self.store, self.session)], ['An immutable original.'])
        await self.store.reset_full(self.session, defaults)
        with patch.dict(os.environ, {'MEMORY_OPERATIONS_PAGE_SIZE': '1'}):
            snapshots = [row async for row in audit_state_records(self.store, self.session)]
        self.assertEqual([row['generation'] for row in snapshots], [1, 2])
        self.assertEqual(snapshots[0]['settings']['system_prompt'], 'Speak like the old librarian.')
        self.assertEqual(snapshots[1]['settings']['system_prompt'], defaults.system_prompt)
        filtered = [row async for row in audit_state_records(self.store, self.session, generation=1)]
        self.assertEqual(len(filtered), 1)

    async def test_retirement_is_bounded_resumable_preserves_sources_and_prioritizes_reset_cleanup(self):
        source = await self.source()
        await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7', asserted_by='telegram:user:7',
                                           claim='An original claim', source_ids=[source.db_id])
        await self.store.create_memory_block(self.session, summary_text='An original summary.', estimated_tokens=8,
                                            source_message_ids=[source.db_id])
        template = await self.store.create_excerpt(self.session, [source.db_id], model=self.space, embedding=self.vector)
        # 1001 vectors exercise the default 1000-row transaction group. These
        # are synthetic derivatives of one source; no model endpoint is involved.
        async with self.store.pool.connection() as conn:
            await conn.execute("SET LOCAL statement_timeout='20s'")
            await conn.execute('''INSERT INTO excerpts
                (session_id,generation,source_ids,source_revisions,spans,fingerprint,model,embedding)
                SELECT e.session_id,e.generation,e.source_ids,e.source_revisions,e.spans,
                    'retirement-fixture-' || n,e.model,e.embedding FROM excerpts e,generate_series(1,1000) n WHERE e.id=%s''',
                (template['id'],))
        paid = await self.store.enqueue_job(self.session, 'embedding_batch', source_ids=[source.db_id],
            payload={'space_id': self.space, 'phase': 'polling', 'name': 'batches/retired-paid', 'excerpt_ids': [template['id']]})
        before = [row async for row in audit_records(self.store, self.session)]
        await self.store.reset_full(self.session, self.config.default_session_settings())
        fresh = await self.source('Fresh active source.', 2)
        active_excerpt = await self.store.create_excerpt(self.session, [fresh.db_id], model=self.space, embedding=self.vector)
        other = await self.source('Other chat stays indexed.', session='telegram:200')
        other_excerpt = await self.store.create_excerpt('telegram:200', [other.db_id], model=self.space, embedding=self.vector)
        self.assertTrue(await self.worker.run_once())
        async with self.store.pool.connection() as conn:
            remaining = (await (await conn.execute('''SELECT count(*) AS n FROM excerpts
                WHERE session_id=%s AND generation=1 AND embedding IS NOT NULL''', (self.session,))).fetchone())['n']
        self.assertEqual(remaining, 1, 'One dispatch retires at most 1000 vectors')
        # New worker instance resumes the durable pending retirement job.
        restarted = MemoryWorker(store=self.store, embeddings=self.embeddings, providers={}, config=self.config)
        self.assertTrue(await restarted.run_once())
        async with self.store.pool.connection() as conn:
            old = (await (await conn.execute('''SELECT count(*) AS n,count(*) FILTER(WHERE valid OR embedding IS NOT NULL) AS active
                FROM excerpts WHERE session_id=%s AND generation=1''', (self.session,))).fetchone())
            kept = await (await conn.execute('SELECT id,valid,embedding IS NOT NULL AS vector FROM excerpts WHERE id=ANY(%s)',
                                            ([active_excerpt['id'], other_excerpt['id']],))).fetchall()
            facts = (await (await conn.execute('SELECT count(*) AS n FROM profile_facts')).fetchone())['n']
            blocks = (await (await conn.execute('SELECT count(*) AS n FROM memory_blocks')).fetchone())['n']
        self.assertEqual((old['n'], old['active']), (1001, 0))
        self.assertTrue(all(row['valid'] and row['vector'] for row in kept))
        self.assertEqual((facts, blocks), (1, 1))
        self.assertEqual([row async for row in audit_records(self.store, self.session, generation=1)], before)
        await self.store.cleanup_jobs(older_than_seconds=0)
        states = [row async for row in audit_state_records(self.store, self.session, generation=1)]
        retained = [row for row in states if row['type'] == 'embedding_batch_state']
        self.assertEqual(retained[0]['id'], paid['id'])
        self.assertEqual(retained[0]['payload']['name'], 'batches/retired-paid')
        self.embeddings.submit_batch.assert_not_awaited()

    async def test_second_reset_invalidates_old_retirement_lease_but_new_job_covers_all_prior_generations(self):
        source = await self.source()
        await self.store.create_excerpt(self.session, [source.db_id], model=self.space, embedding=self.vector)
        await self.store.reset_full(self.session, self.config.default_session_settings())
        old_job = (await self.store.claim_jobs(kind='memory_retire'))[0]
        source = await self.source('Second generation.', 2)
        await self.store.create_excerpt(self.session, [source.db_id], model=self.space, embedding=self.vector)
        await self.store.reset_full(self.session, self.config.default_session_settings())
        with self.assertRaises(StaleScopeError):
            await retire_excerpt_chunk(self.store, old_job)
        new_job = (await self.store.claim_jobs(kind='memory_retire'))[0]
        self.assertEqual(await retire_excerpt_chunk(self.store, new_job), 2)
        with self.assertRaises(ValueError):
            await retire_excerpt_chunk(self.store, {**new_job, 'payload': {'before_generation': 4}})

    async def test_submission_and_paid_polling_ignore_obsolete_disk_budget_environment(self):
        source = await self.source()
        excerpt = await self.store.create_excerpt(self.session, [source.db_id], model=self.space)
        prepared = await self.store.enqueue_job(self.session, 'embedding_batch', source_ids=[source.db_id],
            payload={'space_id': self.space, 'phase': 'prepared', 'excerpt_ids': [excerpt['id']]})
        self.worker.batch = True
        self.embeddings.submit_batch.return_value = BatchJob(
            name='batches/paid', state='JOB_STATE_PENDING', done=False, space_id=self.space)
        with patch.dict(os.environ, {'MEMORY_DISK_BUDGET_MIB': '1'}):
            self.assertTrue(await self.worker.run_once())
        self.embeddings.submit_batch.assert_awaited_once()
        async with self.store.pool.connection() as conn:
            await conn.execute('UPDATE jobs SET available_at=now() WHERE id=%s', (prepared['id'],))
        self.embeddings.poll_batch.return_value = BatchJob(name='batches/paid', state='JOB_STATE_PENDING', done=False, space_id=self.space)
        # Intervening intake gets a turn; the five queue owners must still let
        # this already-paid job progress within one scheduler rotation.
        for _ in range(5):
            self.assertTrue(await self.worker.run_once())
            if self.embeddings.poll_batch.await_count:
                break
        self.embeddings.poll_batch.assert_awaited_once_with('batches/paid')
        self.embeddings.submit_batch.assert_awaited_once()

    async def test_configured_hosted_capacity_allows_fifth_operation_and_defers_sixth(self):
        source = await self.source()
        excerpt = await self.store.create_excerpt(self.session, [source.db_id], model=self.space)
        for index in range(4):
            await self.store.enqueue_job(self.session, 'embedding_batch', source_ids=[source.db_id],
                payload={'space_id': self.space, 'phase': 'polling', 'name': f'batches/paid-{index}',
                         'excerpt_ids': [excerpt['id']]})
        async with self.store.pool.connection() as conn:
            await conn.execute("UPDATE jobs SET available_at=now()+interval '1 hour' WHERE kind='embedding_batch'")
        self.worker.limits = replace(self.worker.limits, max_active_batches=5)
        self.embeddings.submit_batch.return_value = BatchJob(
            name='batches/fifth', state='JOB_STATE_PENDING', done=False, space_id=self.space)
        for _ in range(2):
            await self.store.enqueue_job(self.session, 'embedding_batch', source_ids=[source.db_id],
                payload={'space_id': self.space, 'phase': 'prepared', 'excerpt_ids': [excerpt['id']]})
            await self.worker._batch(await self.store.claim_jobs(kind='embedding_batch', limit=1, lease_seconds=900))
        self.embeddings.submit_batch.assert_awaited_once()
        async with self.store.pool.connection() as conn:
            jobs = await (await conn.execute("SELECT payload,status FROM jobs WHERE kind='embedding_batch' ORDER BY id")).fetchall()
        self.assertEqual(jobs[4]['payload']['name'], 'batches/fifth')
        self.assertEqual(jobs[5]['payload']['phase'], 'prepared')
        self.assertEqual(jobs[5]['status'], 'pending')

    async def test_four_retired_paid_batches_reconcile_without_sources_and_release_new_generation_capacity(self):
        old_source = await self.source('Old private preference.')
        old_excerpt = await self.store.create_excerpt(self.session, [old_source.db_id], model=self.space)
        old_jobs = [await self.store.enqueue_job(self.session, 'embedding_batch', source_ids=[old_source.db_id],
            payload={'space_id': self.space, 'phase': 'polling', 'name': f'batches/old-{index}',
                     'excerpt_ids': [old_excerpt['id']]}) for index in range(4)]
        # A prior exhausted retry or reset callback does not prove the hosted
        # operation stopped consuming its slot.
        async with self.store.pool.connection() as conn:
            await conn.execute("UPDATE jobs SET status='failed' WHERE id=%s", (old_jobs[1]['id'],))
            await conn.execute("UPDATE jobs SET status='stale' WHERE id=%s", (old_jobs[2]['id'],))
        await self.store.reset_full(self.session, self.config.default_session_settings())
        fresh = await self.source('Fresh generation preference.', 2)
        excerpt = await self.store.create_excerpt(self.session, [fresh.db_id], model=self.space)
        prepared = await self.store.enqueue_job(self.session, 'embedding_batch', source_ids=[fresh.db_id],
            payload={'space_id': self.space, 'phase': 'prepared', 'excerpt_ids': [excerpt['id']]})
        await self.worker._batch(await self.store.claim_jobs(kind='embedding_batch', lease_seconds=900))
        self.embeddings.submit_batch.assert_not_awaited()
        self.assertEqual(await self.worker._pending_batches(), 5)

        self.assertTrue(await self.worker.run_once())  # Bounded derivative retirement first.
        self.embeddings.poll_batch.side_effect = lambda name: BatchJob(
            name=name, state='JOB_STATE_SUCCEEDED', done=True, space_id=self.space)
        with patch.object(self.store, 'read_messages', AsyncMock(side_effect=AssertionError('retired source read'))), \
                patch.object(self.store, 'get_excerpt', AsyncMock(side_effect=AssertionError('retired excerpt read'))):
            for _ in range(4):
                self.assertTrue(await self.worker.run_once())
        self.assertEqual(self.embeddings.poll_batch.await_count, 4)
        self.embeddings.read_batch_results.assert_not_awaited()
        self.embeddings.submit_batch.assert_not_awaited()
        self.assertEqual(await self.worker._pending_batches(), 1)
        self.assertEqual(await self.store.read_messages(self.session, [old_source.db_id]), [])
        self.assertEqual(await self.store.search_messages(self.session, 'private'), [])
        await self.store.cleanup_jobs(older_than_seconds=0)
        audit = [row async for row in audit_state_records(self.store, self.session, generation=1)
                 if row['type'] == 'embedding_batch_state']
        self.assertEqual(len(audit), 4)
        self.assertTrue(all(row['status'] == 'stale' and row['payload']['phase'] == 'retired_terminal' for row in audit))
        self.assertEqual([row['body'] async for row in audit_records(self.store, self.session, generation=1)],
                         ['Old private preference.'])

        async with self.store.pool.connection() as conn:
            await conn.execute('UPDATE jobs SET available_at=now() WHERE id=%s', (prepared['id'],))
        self.embeddings.submit_batch.return_value = BatchJob(
            name='batches/new-generation', state='JOB_STATE_PENDING', done=False, space_id=self.space)
        self.assertTrue(await self.worker.run_once())
        self.embeddings.submit_batch.assert_awaited_once()
        submitted = self.embeddings.submit_batch.await_args.args[0]
        self.assertEqual([item.item_id for item in submitted], [str(excerpt['id'])])

    async def test_retired_ambiguous_submission_stays_reserved_until_found_and_terminal(self):
        source = await self.source()
        job = await self.store.enqueue_job(self.session, 'embedding_batch', source_ids=[source.db_id],
            payload={'space_id': self.space, 'phase': 'submitting', 'display_name': 'persisted-before-timeout',
                     'excerpt_ids': [123]})
        await self.store.reset_full(self.session, self.config.default_session_settings())
        await self.worker.run_once()
        self.embeddings.find_batch.return_value = None
        self.assertTrue(await self.worker.run_once())
        self.assertEqual(await self.worker._pending_batches(), 1)
        self.embeddings.find_batch.assert_awaited_once_with('persisted-before-timeout')
        async with self.store.pool.connection() as conn:
            missing = await (await conn.execute('SELECT payload,error FROM jobs WHERE id=%s', (job['id'],))).fetchone()
            await conn.execute('UPDATE jobs SET available_at=now() WHERE id=%s', (job['id'],))
        self.assertEqual(missing['payload']['phase'], 'submitting')
        self.assertIn('capacity remains reserved', missing['error'])
        self.embeddings.find_batch.return_value = BatchJob(
            name='batches/recovered', state='JOB_STATE_PENDING', done=False, space_id=self.space)
        restarted = MemoryWorker(store=self.store, embeddings=self.embeddings, providers={}, config=self.config)
        self.assertTrue(await restarted.run_once())
        self.assertEqual(await restarted._pending_batches(), 1)
        async with self.store.pool.connection() as conn:
            await conn.execute('UPDATE jobs SET available_at=now() WHERE id=%s', (job['id'],))
        self.embeddings.poll_batch.return_value = BatchJob(
            name='batches/recovered', state='JOB_STATE_CANCELLED', done=True, space_id=self.space)
        self.assertTrue(await restarted.run_once())
        self.assertEqual(await restarted._pending_batches(), 0)
        self.embeddings.poll_batch.assert_awaited_once_with('batches/recovered')
        self.embeddings.submit_batch.assert_not_awaited()
        self.embeddings.read_batch_results.assert_not_awaited()

    async def test_retired_foreign_space_is_explicitly_blocked_without_contacting_selected_provider(self):
        source = await self.source()
        job = await self.store.enqueue_job(self.session, 'embedding_batch', source_ids=[source.db_id],
            payload={'space_id': 'different-space', 'phase': 'polling', 'name': 'batches/other-model', 'excerpt_ids': []})
        await self.store.reset_full(self.session, self.config.default_session_settings())
        await self.worker.run_once()
        self.assertTrue(await self.worker.run_once())
        self.assertEqual(await self.worker._pending_batches(), 1)
        self.assertEqual(self.worker.last_error, 'RetiredBatchSpaceMismatch')
        async with self.store.pool.connection() as conn:
            row = await (await conn.execute('SELECT payload,error FROM jobs WHERE id=%s', (job['id'],))).fetchone()
        self.assertIn('selected embedding space differs', row['error'])
        self.assertEqual(row['payload']['name'], 'batches/other-model')
        self.embeddings.find_batch.assert_not_awaited()
        self.embeddings.poll_batch.assert_not_awaited()
        self.embeddings.submit_batch.assert_not_awaited()

    async def test_unpaid_prepared_batch_stops_reserving_capacity_immediately_after_reset(self):
        source = await self.source()
        job = await self.store.enqueue_job(self.session, 'embedding_batch', source_ids=[source.db_id],
            payload={'space_id': self.space, 'phase': 'prepared', 'excerpt_ids': []})
        self.assertEqual(await self.worker._pending_batches(), 1)
        await self.store.reset_full(self.session, self.config.default_session_settings())
        self.assertEqual(await self.worker._pending_batches(), 0)
        await self.store.cleanup_jobs(older_than_seconds=0)
        async with self.store.pool.connection() as conn:
            row = await (await conn.execute('SELECT id FROM jobs WHERE id=%s', (job['id'],))).fetchone()
        self.assertIsNone(row)
        self.embeddings.submit_batch.assert_not_awaited()

    async def test_source_edit_retires_paid_batch_without_reset_or_stale_result_application(self):
        source = await self.source('Original preference.')
        excerpt = await self.store.create_excerpt(self.session, [source.db_id], model=self.space)
        queued = await self.store.enqueue_job(self.session, 'embedding_batch', source_ids=[source.db_id],
            payload={'space_id': self.space, 'phase': 'polling', 'name': 'batches/edited-source',
                     'excerpt_ids': [excerpt['id']]})
        claimed = (await self.store.claim_jobs(kind='embedding_batch', lease_seconds=900))[0]
        await self.source('Corrected preference.')
        self.assertFalse(await self.store.renew_job(claimed, lease_seconds=900))
        self.assertEqual(await self.store.claim_jobs(kind='embedding_batch'), [])
        self.embeddings.poll_batch.return_value = BatchJob(
            name='batches/edited-source', state='JOB_STATE_PENDING', done=False, space_id=self.space)
        with patch.object(self.store, 'get_excerpt', AsyncMock(side_effect=AssertionError('stale excerpt read'))):
            self.assertTrue(await self.worker.run_once())
        self.assertEqual(await self.worker._pending_batches(), 1)
        self.assertEqual(await self.store.claim_jobs(kind='embedding_batch'), [])
        async with self.store.pool.connection() as conn:
            row = await (await conn.execute('SELECT status,payload FROM jobs WHERE id=%s', (queued['id'],))).fetchone()
            await conn.execute('UPDATE jobs SET available_at=now() WHERE id=%s', (queued['id'],))
        self.assertEqual(row['status'], 'stale')
        self.assertTrue(row['payload']['retired_reconciliation'])
        self.embeddings.poll_batch.return_value = BatchJob(
            name='batches/edited-source', state='JOB_STATE_SUCCEEDED', done=True, space_id=self.space)
        self.assertTrue(await self.worker.run_once())
        self.assertEqual(await self.worker._pending_batches(), 0)
        self.embeddings.submit_batch.assert_not_awaited()
        self.embeddings.read_batch_results.assert_not_awaited()
        current = await self.store.read_messages(self.session, [source.db_id])
        self.assertEqual(original_text(current[0].message), 'Corrected preference.')
