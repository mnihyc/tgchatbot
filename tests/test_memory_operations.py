from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np
from psycopg.types.json import Jsonb

from tests.business_helpers import BusinessTestCase
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind, ProviderResponse
from tgchatbot.embeddings import EmbeddingConfig
from tgchatbot.tools.memory import audit_records, audit_state_records, parser, rebuild, retry_failed, status_records, work


class MemoryOperationsTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.settings()
        self.space = 'synthetic-space'
        self.vector = np.array([1.] + [0.] * 1535)

    async def original(self, text, source_id, *, session=None, **metadata):
        session = session or self.session
        await self.store.get_or_create_session(session, self.config.default_session_settings())
        return await self.store.append_message(session, ConversationMessage.user_text(text, metadata={
            'source': 'telegram', 'source_chat_id': session.split(':')[-1], 'source_message_id': str(source_id),
            'actor_id': 'telegram:user:7', 'actor_name': 'Alex', **metadata}))

    async def fail_job(self, job, payload=None):
        async with self.store.pool.connection() as conn:
            await conn.execute("UPDATE jobs SET status='failed',attempts=3,error='synthetic failure',finished_at=now() WHERE id=%s", (job['id'],))
            if payload is not None:
                await conn.execute('UPDATE jobs SET payload=%s WHERE id=%s', (Jsonb(payload), job['id']))

    async def test_default_status_does_not_scan_source_text_or_claim_complete_coverage(self):
        await self.original('Unindexed source.', 1)
        with patch('tgchatbot.tools.memory.coverage', new_callable=AsyncMock) as scan:
            result = [row async for row in status_records(self.store, self.space, self.session)]
        scan.assert_not_awaited()
        self.assertFalse(result[0]['coverage']['measured'])
        self.assertNotIn('semantic_complete', result[0]['coverage'])
        self.assertIn('--coverage', result[0]['coverage']['reason'])
        self.assertEqual(result[0]['pending_profile_material'], {'sources': 1, 'bytes': len('Unindexed source.')})

    async def test_profile_work_and_operator_audit_do_not_require_embedding_credentials(self):
        await self.original('I prefer tea.I like birds.', 1)
        self.provider.responses = [ProviderResponse(final_text='{"additions": [], "removals": []}')]
        with patch.dict(os.environ, {'MEMORY_WORKER_PROFILE_REQUEST_BYTES': '26'}, clear=True), \
             patch('tgchatbot.providers.factory.build_providers', return_value={'openai': self.provider}):
            result = await work(self.store, self.config, EmbeddingConfig(api_key=''), batch=False, once=True)
        self.assertTrue(result['processed_dispatch'])
        self.assertEqual(len(self.provider.requests), 1)
        audit = [row async for row in audit_state_records(self.store, self.session)]
        self.assertEqual(len([row for row in audit if row['type'] == 'profile_patch']), 1)

    async def test_status_exposes_retired_hosted_slot_blockers_without_reading_old_sources(self):
        source = await self.original('Private original is not status output.', 1)
        old = await self.store.enqueue_job(self.session, 'embedding_batch', source_ids=[source.db_id],
            payload={'space_id': self.space, 'phase': 'polling', 'name': 'batches/old-paid'})
        foreign_space = await self.store.enqueue_job(self.session, 'embedding_batch',
            payload={'space_id': 'other-space', 'phase': 'submitting', 'display_name': 'ambiguous-old'})
        await self.store.enqueue_job(self.session, 'embedding_batch', payload={'space_id': self.space, 'phase': 'prepared'})
        terminal = await self.store.enqueue_job(self.session, 'embedding_batch',
            payload={'space_id': self.space, 'phase': 'retired_terminal', 'name': 'batches/finished'})
        await self.store.reset_full(self.session, self.config.default_session_settings())
        stale = await self.store.enqueue_job(self.session, 'embedding_batch',
            payload={'space_id': self.space, 'phase': 'polling', 'name': 'batches/edited-source'})
        active = await self.store.enqueue_job(self.session, 'embedding_batch',
            payload={'space_id': self.space, 'phase': 'polling', 'name': 'batches/current-paid'})
        await self.store.get_or_create_session('telegram:200', self.config.default_session_settings())
        await self.store.enqueue_job('telegram:200', 'embedding_batch',
            payload={'space_id': self.space, 'phase': 'polling', 'name': 'batches/other-chat'})
        await self.store.reset_full('telegram:200', self.config.default_session_settings())
        async with self.store.pool.connection() as conn:
            await conn.execute("UPDATE jobs SET status='stale' WHERE id=ANY(%s)", ([stale['id'], terminal['id']],))
            await conn.execute("UPDATE jobs SET error='selected embedding space differs' WHERE id=%s", (foreign_space['id'],))
        with patch('tgchatbot.tools.memory.coverage', new_callable=AsyncMock) as scan, \
                patch.object(self.store, 'read_messages', AsyncMock(side_effect=AssertionError('old source read'))):
            result = [row async for row in status_records(self.store, self.space, self.session)]
        scan.assert_not_awaited()
        slots = result[0]['retired_batch_slots']
        self.assertEqual((slots['count'], slots['blocked_space_count']), (3, 1))
        self.assertEqual({row['id'] for row in slots['jobs']}, {old['id'], foreign_space['id'], stale['id']})
        self.assertIn('selected embedding space differs', [row['error'] for row in slots['jobs']])
        self.assertEqual([row['id'] for row in result[0]['batch_jobs']], [active['id']])
        self.assertNotIn('Private original', str(result))

    async def test_configured_status_windows_report_truncation_and_audit_still_traverses_every_job(self):
        retired_jobs = [await self.store.enqueue_job(self.session, 'embedding_batch',
            payload={'space_id': self.space, 'phase': 'polling', 'name': f'batches/retired-{index}'})
            for index in range(4)]
        await self.store.reset_full(self.session, self.config.default_session_settings())
        current_jobs = [await self.store.enqueue_job(self.session, 'embedding_batch',
            payload={'space_id': self.space, 'phase': 'polling', 'name': f'batches/current-{index}'})
            for index in range(3)]
        with patch.dict(os.environ, {'MEMORY_OPERATIONS_PAGE_SIZE': '1', 'MEMORY_OPERATIONS_STATUS_JOB_LIMIT': '2'}):
            result = [row async for row in status_records(self.store, self.space, self.session)]
            audit = [row async for row in audit_state_records(self.store, self.session)]
        status = result[0]
        self.assertEqual([row['id'] for row in status['batch_jobs']], [job['id'] for job in current_jobs[:2]])
        self.assertEqual(status['batch_jobs_limit'], 2)
        self.assertTrue(status['batch_jobs_truncated'])
        retired = status['retired_batch_slots']
        self.assertEqual(retired['count'], 4)
        self.assertEqual(retired['jobs_limit'], 2)
        self.assertTrue(retired['jobs_truncated'])
        self.assertEqual([row['id'] for row in retired['jobs']], [job['id'] for job in retired_jobs[:2]])
        self.assertEqual({row['id'] for row in audit if row['type'] == 'embedding_batch_state'},
                         {job['id'] for job in retired_jobs + current_jobs})
        with patch.dict(os.environ, {'MEMORY_OPERATIONS_STATUS_JOB_LIMIT': '10'}):
            expanded = [row async for row in status_records(self.store, self.space, self.session)]
        self.assertEqual(len(expanded[0]['batch_jobs']), 3)
        self.assertEqual(expanded[0]['batch_jobs_limit'], 10)
        self.assertFalse(expanded[0]['batch_jobs_truncated'])
        self.assertEqual(len(expanded[0]['retired_batch_slots']['jobs']), 4)
        self.assertFalse(expanded[0]['retired_batch_slots']['jobs_truncated'])

    async def test_status_distinguishes_partial_text_other_spaces_and_retained_soft_context(self):
        full = await self.original('abcdefghij', 1)
        partial = await self.original('ABCDEFGHIJ', 2)
        unindexed = await self.original('untouched', 3)
        await self.store.create_excerpt(self.session, [full.db_id], spans=[{'message_id': full.db_id, 'start': 0, 'end': 6}],
                                        model=self.space, embedding=self.vector)
        await self.store.create_excerpt(self.session, [full.db_id], spans=[{'message_id': full.db_id, 'start': 4, 'end': 10}],
                                        model=self.space, embedding=self.vector)
        await self.store.create_excerpt(self.session, [partial.db_id], spans=[{'message_id': partial.db_id, 'start': 0, 'end': 3}],
                                        model=self.space, embedding=self.vector)
        await self.store.create_excerpt(self.session, [unindexed.db_id], model='another-space', embedding=self.vector)
        await self.store.reset_context(self.session)
        await self.original('foreign chat', 1, session='telegram:200')
        status = [row async for row in status_records(self.store, self.space, self.session, include_coverage=True)]
        self.assertEqual(len(status), 1)
        coverage = status[0]['coverage']
        self.assertEqual((coverage['active_originals'], coverage['lexical_indexed_originals']), (3, 3))
        self.assertEqual((coverage['semantic_full_originals'], coverage['semantic_partial_originals'],
                          coverage['semantic_uncovered_originals']), (1, 1, 1))
        self.assertEqual(coverage['semantic_covered_characters'], 13, 'Overlapping spans must not inflate coverage')
        self.assertFalse(coverage['semantic_complete'])
        self.assertEqual({item['space_id'] for item in status[0]['excerpt_spaces']}, {self.space, 'another-space'})
        self.assertEqual({row['session_id'] async for row in status_records(self.store, self.space)}, {self.session, 'telegram:200'})

    async def test_coverage_counts_eligible_part_ranges_without_notes_or_separator_gaps(self):
        first, second = ' 茶🙂 ', 'document fragment'
        parts = [MessagePart(PartKind.TEXT, text=first),
                 MessagePart(PartKind.TEXT, text='Application-only delivery note', origin='auto_note'),
                 MessagePart(PartKind.TEXT, text=' \t '),
                 MessagePart(PartKind.FILE, text=second, origin='attachment_excerpt'),
                 MessagePart(PartKind.TEXT, text='Application-only attribution', origin='provenance')]
        source = await self.store.append_message(self.session, ConversationMessage(MessageRole.USER, parts, metadata={
            'source': 'telegram', 'source_chat_id': '100', 'source_message_id': '1',
            'actor_id': 'telegram:user:7', 'actor_kind': 'user', 'actor_name': 'Alex'}))
        body = '\n'.join(part.text for part in parts)
        first_span = {'message_id': source.db_id, 'start': 0, 'end': len(first)}
        second_start = body.index(second)
        second_span = {'message_id': source.db_id, 'start': second_start, 'end': second_start + len(second)}
        await self.store.create_excerpt(self.session, [source.db_id], spans=[first_span],
                                        model=self.space, embedding=self.vector)
        status = [row async for row in status_records(self.store, self.space, self.session, include_coverage=True)]
        partial = status[0]['coverage']
        self.assertEqual(partial['semantic_eligible_originals'], 1)
        self.assertEqual(partial['semantic_eligible_characters'], len(first) + len(second))
        self.assertEqual(partial['semantic_covered_characters'], len(first))
        self.assertEqual(partial['semantic_partial_originals'], 1)
        self.assertFalse(partial['semantic_complete'])

        await self.store.create_excerpt(self.session, [source.db_id], spans=[second_span],
                                        model=self.space, embedding=self.vector)
        status = [row async for row in status_records(self.store, self.space, self.session, include_coverage=True)]
        complete = status[0]['coverage']
        self.assertEqual(complete['semantic_full_originals'], 1)
        self.assertEqual(complete['semantic_partial_originals'], 0)
        self.assertEqual(complete['semantic_covered_characters'], len(first) + len(second))
        self.assertTrue(complete['semantic_complete'])
        # A wider overlapping excerpt must not inflate the numerator by counting
        # ignored notes, blank parts, or separator characters.
        await self.store.create_excerpt(self.session, [source.db_id],
            spans=[{'message_id': source.db_id, 'start': 0, 'end': len(body)}],
            model=self.space, embedding=self.vector)
        repeated = [row async for row in status_records(self.store, self.space, self.session, include_coverage=True)]
        self.assertEqual(repeated[0]['coverage']['semantic_covered_characters'], len(first) + len(second))

        notes_chat = 'telegram:200'
        await self.store.get_or_create_session(notes_chat, self.config.default_session_settings())
        await self.store.append_message(notes_chat, ConversationMessage(MessageRole.USER,
            [part for part in parts if part not in (parts[0], parts[3])], metadata={
                'source': 'telegram', 'source_chat_id': '200', 'source_message_id': '1',
                'actor_id': 'telegram:user:7', 'actor_kind': 'user', 'actor_name': 'Alex'}))
        status = [row async for row in status_records(self.store, self.space, notes_chat, include_coverage=True)]
        notes = status[0]['coverage']
        self.assertEqual(notes['active_originals'], 1)
        self.assertEqual(notes['semantic_eligible_originals'], 0)
        self.assertEqual(notes['semantic_eligible_characters'], 0)
        self.assertEqual(notes['semantic_covered_characters'], 0)

    async def test_coverage_preserves_multisource_overlap_across_pages_and_source_corrections(self):
        first = await self.original('abcdefghij', 1)
        second = await self.original('ABCDEFGHIJ', 2, actor_id='telegram:user:8')
        await self.store.create_excerpt(self.session, [first.db_id, second.db_id],
            spans=[{'message_id': source.db_id, 'start': 0, 'end': 6} for source in (first, second)],
            model=self.space, embedding=self.vector)
        await self.store.create_excerpt(self.session, [first.db_id],
            spans=[{'message_id': first.db_id, 'start': 4, 'end': 10}], model=self.space, embedding=self.vector)
        await self.store.create_excerpt(self.session, [second.db_id], model=self.space, embedding=self.vector)

        async def measured(page_size):
            with patch.dict(os.environ, {'MEMORY_OPERATIONS_PAGE_SIZE': str(page_size)}):
                return [row async for row in status_records(self.store, self.space, self.session,
                    include_coverage=True)][0]['coverage']

        together, separate = await measured(2), await measured(1)
        self.assertEqual(together, separate, 'An excerpt crossing traversal pages keeps the same original coverage')
        self.assertEqual(together['semantic_covered_characters'], 20,
            'Multi-source matches and overlapping excerpts cannot count a character twice')
        self.assertEqual(together['semantic_full_originals'], 2)

        revised = await self.original('Corrected source', 1)
        self.assertEqual(revised.db_id, first.db_id)
        after = await measured(2)
        self.assertEqual(after['semantic_full_originals'], 1)
        self.assertEqual(after['semantic_uncovered_originals'], 1)
        self.assertEqual(after['semantic_covered_characters'], 10,
            'Old multi-source evidence cannot cover revised text; the other sender\'s independent excerpt remains')
        self.assertFalse(after['semantic_complete'])

    async def test_status_pages_and_excludes_hidden_deleted_old_generations_and_synthetic(self):
        old = await self.original('old generation', 1)
        await self.store.reset_full(self.session, self.config.default_session_settings())
        current = await self.original('current text', 2)
        hidden = await self.original('hidden', 3)
        deleted = await self.original('deleted', 4)
        await self.store.hide_message_ids(self.session, [hidden.db_id])
        await self.store.delete_message_ids(self.session, [deleted.db_id])
        await self.original('memory control', 5, synthetic_role='memory_context')
        with patch.dict(os.environ, {'MEMORY_OPERATIONS_PAGE_SIZE': '1'}):
            result = [row async for row in status_records(self.store, self.space, self.session, include_coverage=True)]
        self.assertEqual(result[0]['coverage']['active_originals'], 2)
        self.assertEqual(result[0]['coverage']['semantic_eligible_originals'], 1)
        self.assertEqual(result[0]['coverage']['semantic_eligible_characters'], len('current text'))

    async def test_audit_streams_exact_original_revisions_old_generations_and_compound_cursor(self):
        original = await self.original('Original claim.', 1)
        await self.original('Corrected claim.', 1, edited_at='2026-02-01T00:00:00Z')
        await self.store.reset_full(self.session, self.config.default_session_settings())
        current = await self.original('New generation.', 2)
        await self.store.hide_message_ids(self.session, [current.db_id])
        await self.original('Other chat.', 1, session='telegram:200')
        with patch.dict(os.environ, {'MEMORY_OPERATIONS_PAGE_SIZE': '1'}):
            rows = [row async for row in audit_records(self.store, self.session)]
            resumed = [row async for row in audit_records(self.store, self.session,
                after_message_id=original.db_id, after_revision=1)]
            filtered = [row async for row in audit_records(self.store, self.session,
                message_id=original.db_id, generation=1)]
        self.assertEqual([row['body'] for row in rows], ['Original claim.', 'Corrected claim.', 'New generation.'])
        self.assertEqual([(row['message_id'], row['revision']) for row in resumed], [(original.db_id, 2), (current.db_id, 1)])
        self.assertEqual([row['body'] for row in filtered], ['Original claim.', 'Corrected claim.'])
        self.assertTrue(rows[-1]['hidden'])
        self.assertEqual(rows[0]['metadata']['actor_id'], 'telegram:user:7')
        self.assertEqual(await self.store.read_messages(self.session, [original.db_id]), [])

    async def test_rebuild_pages_active_canonical_sources_preserves_originals_and_existing_batch_identity(self):
        await self.original('Old generation.', 1)
        await self.store.reset_full(self.session, self.config.default_session_settings())
        first = await self.original('First original.', 2)
        await self.store.reset_context(self.session)
        second = await self.original('Second original.', 3)
        hidden = await self.original('Hidden.', 4)
        await self.store.hide_message_ids(self.session, [hidden.db_id])
        await self.original('Control note.', 5, synthetic_role='memory_context')
        await self.original('Foreign.', 1, session='telegram:200')
        batch_payload = {'space_id': self.space, 'name': 'batches/accepted-paid-job', 'phase': 'polling', 'excerpt_ids': [1]}
        batch = await self.store.enqueue_job(self.session, 'embedding_batch', source_ids=[first.db_id],
                                             payload=batch_payload, dedupe_key='paid')
        await self.fail_job(batch)
        before = [row async for row in audit_records(self.store, self.session)]
        with patch.dict(os.environ, {'MEMORY_OPERATIONS_PAGE_SIZE': '1'}):
            result = await rebuild(self.store, self.session, self.space, progress=None)
        self.assertEqual((result['queued_originals'], result['queued_jobs'], result['retried_embedding_jobs']), (2, 2, 1))
        self.assertEqual([row async for row in audit_records(self.store, self.session)], before)
        async with self.store.pool.connection() as conn:
            jobs = await (await conn.execute("SELECT * FROM jobs WHERE payload ? 'rebuild_id' ORDER BY id")).fetchall()
            resumed = await (await conn.execute('SELECT * FROM jobs WHERE id=%s', (batch['id'],))).fetchone()
        self.assertEqual([job['source_ids'] for job in jobs], [[first.db_id], [second.db_id]])
        self.assertTrue(all(job['payload']['space_id'] == self.space for job in jobs))
        self.assertEqual(resumed['payload'], batch_payload)
        self.assertEqual((resumed['status'], resumed['attempts']), ('pending', 0))

    async def test_large_configured_pages_preserve_all_rebuild_sources_and_coverage_beyond_old_bounds(self):
        sources = [await self.original(f'Original {number}', number) for number in range(1, 1104)]
        await self.store.create_excerpt(self.session, [sources[-1].db_id], model=self.space, embedding=self.vector)
        with patch.dict(os.environ, {'MEMORY_OPERATIONS_PAGE_SIZE': '1100'}):
            result = await rebuild(self.store, self.session, self.space, progress=None)
            status = [row async for row in status_records(self.store, self.space, self.session, include_coverage=True)]
            audit = [row async for row in audit_records(self.store, self.session)]
        self.assertEqual((result['queued_originals'], result['queued_jobs']), (1103, 2))
        async with self.store.pool.connection() as conn:
            jobs = await (await conn.execute("SELECT source_ids FROM jobs WHERE payload ? 'rebuild_id' ORDER BY id")).fetchall()
        self.assertEqual([len(job['source_ids']) for job in jobs], [1100, 3])
        self.assertEqual([mid for job in jobs for mid in job['source_ids']], [source.db_id for source in sources])
        self.assertEqual(len(audit), 1103)
        self.assertEqual(audit[-1]['body'], 'Original 1103')
        covered = status[0]['coverage']
        self.assertEqual((covered['active_originals'], covered['semantic_eligible_originals']), (1103, 1103))
        self.assertEqual((covered['semantic_full_originals'], covered['semantic_uncovered_originals']), (1, 1102))
        self.assertEqual(covered['semantic_covered_characters'], len('Original 1103'))

    async def test_rebuild_refuses_unfinished_foreign_or_unknown_batch_without_writes(self):
        original = await self.original('Preserve paid job.', 1)
        for payload in ({'phase': 'submitting'}, {'space_id': 'old-space', 'name': 'batches/paid'}):
            job = await self.store.enqueue_job(self.session, 'embedding_batch', source_ids=[original.db_id], payload=payload)
            with self.assertRaisesRegex(ValueError, 'different or unknown embedding space'):
                await rebuild(self.store, self.session, self.space, progress=None)
            async with self.store.pool.connection() as conn:
                await conn.execute("UPDATE jobs SET status='done' WHERE id=%s", (job['id'],))
                count = (await (await conn.execute("SELECT count(*) AS n FROM jobs WHERE payload ? 'rebuild_id'")).fetchone())['n']
            self.assertEqual(count, 0)

    async def test_retry_is_chat_generation_space_scoped_and_keeps_paid_payload(self):
        old = await self.store.enqueue_job(self.session, 'memory_ingest')
        await self.fail_job(old)
        await self.store.reset_full(self.session, self.config.default_session_settings())
        current = await self.store.enqueue_job(self.session, 'memory_ingest')
        await self.fail_job(current)
        paid = await self.store.enqueue_job(self.session, 'embedding_batch',
            payload={'space_id': self.space, 'phase': 'submitting', 'display_name': 'durable-id'})
        await self.fail_job(paid)
        foreign_space = await self.store.enqueue_job(self.session, 'memory_embed', payload={'space_id': 'other'})
        await self.fail_job(foreign_space)
        unknown_space = await self.store.enqueue_job(self.session, 'embedding_batch', payload={'phase': 'submitting'})
        await self.fail_job(unknown_space)
        await self.store.get_or_create_session('telegram:200', self.config.default_session_settings())
        foreign_chat = await self.store.enqueue_job('telegram:200', 'memory_ingest')
        await self.fail_job(foreign_chat)
        self.assertEqual(await retry_failed(self.store, self.session, self.space), 2)
        async with self.store.pool.connection() as conn:
            rows = await (await conn.execute('SELECT * FROM jobs ORDER BY id')).fetchall()
        self.assertEqual({row['id'] for row in rows if row['status'] == 'pending' and row['kind'] != 'memory_retire'}, {current['id'], paid['id']})
        self.assertEqual(next(row['payload'] for row in rows if row['id'] == paid['id']), paid['payload'])

    async def test_native_batch_cap_counts_failed_and_ambiguous_jobs_across_chats_before_new_submit(self):
        from tgchatbot.core.memory_worker import MemoryWorker
        from tgchatbot.embeddings import BatchJob
        source = await self.original('Original to embed.', 1)
        excerpt = await self.store.create_excerpt(self.session, [source.db_id], model=self.space)
        occupied = []
        for index in range(4):
            other_chat = f'telegram:{200 + index}'
            await self.store.get_or_create_session(other_chat, self.config.default_session_settings())
            payload = {'space_id': self.space, 'phase': 'submitting' if index == 0 else 'polling'}
            if index:
                payload['name'] = f'batches/paid-{index}'
            occupied.append(await self.store.enqueue_job(other_chat, 'embedding_batch', payload=payload))
        await self.fail_job(occupied[-1])
        pending = await self.store.enqueue_job(self.session, 'embedding_batch', source_ids=[source.db_id],
            payload={'space_id': self.space, 'phase': 'prepared', 'excerpt_ids': [excerpt['id']]})
        async with self.store.pool.connection() as conn:
            await conn.execute("UPDATE jobs SET available_at=now()+interval '1 day' WHERE id=ANY(%s)",
                               ([job['id'] for job in occupied],))
        embeddings = SimpleNamespace(space_id=self.space, submit_batch=AsyncMock(return_value=BatchJob(
            name='batches/new-paid', state='JOB_STATE_PENDING', done=False, space_id=self.space)),
            find_batch=AsyncMock())
        worker = MemoryWorker(store=self.store, embeddings=embeddings, providers={}, config=self.config, batch=True)
        jobs = await self.store.claim_jobs(kind='embedding_batch', lease_seconds=900)
        self.assertEqual([job['id'] for job in jobs], [pending['id']])
        await worker._batch(jobs)
        embeddings.submit_batch.assert_not_awaited()
        async with self.store.pool.connection() as conn:
            held = await (await conn.execute('SELECT * FROM jobs WHERE id=%s', (pending['id'],))).fetchone()
            self.assertEqual(held['payload']['phase'], 'prepared')
            self.assertEqual(held['status'], 'pending')
            await conn.execute("UPDATE jobs SET status='done' WHERE id=%s", (occupied[1]['id'],))
            await conn.execute('UPDATE jobs SET available_at=now() WHERE id=%s', (pending['id'],))
        await worker._batch(await self.store.claim_jobs(kind='embedding_batch', lease_seconds=900))
        embeddings.submit_batch.assert_awaited_once()
        async with self.store.pool.connection() as conn:
            submitted = await (await conn.execute('SELECT payload FROM jobs WHERE id=%s', (pending['id'],))).fetchone()
        self.assertEqual(submitted['payload']['name'], 'batches/new-paid')

    async def test_unknown_chat_audit_status_rebuild_do_not_create_sessions(self):
        for command in ('audit', 'status', 'rebuild'):
            with self.assertRaisesRegex(ValueError, 'does not exist'):
                if command == 'audit':
                    _ = [row async for row in audit_records(self.store, 'telegram:404')]
                elif command == 'status':
                    _ = [row async for row in status_records(self.store, self.space, 'telegram:404')]
                else:
                    await rebuild(self.store, 'telegram:404', self.space, progress=None)
        self.assertEqual(await self.store.count_sessions(), 1)


class MemoryCommandContractTests(unittest.IsolatedAsyncioTestCase):
    def test_operator_scopes_are_explicit_and_worker_once_is_a_dispatch(self):
        args = parser().parse_args(['work', '--batch', '--once'])
        self.assertTrue(args.batch and args.once)
        args = parser().parse_args(['audit', '--chat-id=-100', '--message-id', '5', '--generation', '2'])
        self.assertEqual((args.chat_id, args.message_id, args.generation), (-100, 5, 2))
        with self.assertRaises(SystemExit), patch('sys.stderr'):
            parser().parse_args(['rebuild'])
        with self.assertRaises(SystemExit), patch('sys.stderr'):
            parser().parse_args(['retry-jobs', '--chat-id', '0'])

    async def test_once_constructs_same_worker_and_always_closes_clients(self):
        embedding = SimpleNamespace(aclose=AsyncMock())
        provider = SimpleNamespace(aclose=AsyncMock())
        worker = SimpleNamespace(run_once=AsyncMock(return_value=True), close=AsyncMock(), last_error=None)
        config = object()
        with patch('tgchatbot.tools.memory.EmbeddingClient', return_value=embedding), \
             patch('tgchatbot.providers.factory.build_providers', return_value={'mock': provider}), \
             patch('tgchatbot.core.memory_worker.MemoryWorker', return_value=worker) as factory:
            result = await work(object(), config, EmbeddingConfig(api_key='synthetic-only'), batch=True, once=True)
        self.assertTrue(factory.call_args.kwargs['batch'])
        self.assertIs(factory.call_args.kwargs['config'], config)
        self.assertTrue(result['processed_dispatch'])
        self.assertFalse(result['complete'])
        worker.run_once.assert_awaited_once()
        worker.close.assert_awaited_once()
        embedding.aclose.assert_awaited_once()
        provider.aclose.assert_awaited_once()

    async def test_cancelled_worker_closes_resources_and_unsupported_batch_never_calls_network(self):
        embedding = SimpleNamespace(aclose=AsyncMock())
        provider = SimpleNamespace(aclose=AsyncMock())
        worker = SimpleNamespace(run=AsyncMock(side_effect=asyncio.CancelledError()), close=AsyncMock())
        with patch('tgchatbot.tools.memory.EmbeddingClient', return_value=embedding) as client, \
             patch('tgchatbot.providers.factory.build_providers', return_value={'mock': provider}), \
             patch('tgchatbot.core.memory_worker.MemoryWorker', return_value=worker):
            with self.assertRaises(ValueError):
                await work(None, None, EmbeddingConfig(provider='openai', model='test', base_url='http://localhost:9999'), batch=True, once=True)
            client.assert_not_called()
            with self.assertRaises(asyncio.CancelledError):
                await work(None, None, EmbeddingConfig(api_key='synthetic-only'), batch=True, once=False)
        embedding.aclose.assert_awaited_once()
        provider.aclose.assert_awaited_once()
        worker.close.assert_awaited_once()
