from __future__ import annotations

import json
from dataclasses import replace
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.memory_worker import MemoryWorker, ExcerptBuilder, WorkerConfig
from tgchatbot.domain.models import ConversationMessage, ProviderResponse
from tgchatbot.embeddings import BatchJob, BatchItemResult
from tgchatbot.operational import from_env


class ExcerptConfigurationTests(unittest.IsolatedAsyncioTestCase):
    async def test_larger_token_setting_can_pack_past_initial_probe_and_old_character_ceiling(self):
        original = SimpleNamespace(db_id=17, message=ConversationMessage.user_text(
            'a' * 5000, metadata={'actor_id': 'telegram:user:7', 'actor_name': 'Alex'}))
        with patch.dict(os.environ, {'MEMORY_WORKER_EXCERPT_TOKENS': '6000',
                                     'MEMORY_WORKER_EXCERPT_CANDIDATE_CHARS': '32'}, clear=True):
            builder = ExcerptBuilder(SimpleNamespace(enabled=False))
        chunks = await builder.build([original])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]['spans'], [{'message_id': 17, 'start': 0, 'end': 5000}])
        self.assertTrue(chunks[0]['text'].endswith('a' * 5000))
        self.assertLessEqual(await builder.count(chunks[0]['text']), builder.token_limit)


class WorkerWorkflowTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        async def vectors(items, **kwargs):
            return [np.array([1.0] + [0.0] * 1535) for _ in items]
        self.embeddings = SimpleNamespace(enabled=True, space_id='synthetic-vector-space',
            count_tokens=AsyncMock(side_effect=lambda text, **kwargs: len(text.encode('utf-8'))),
            embed_documents=AsyncMock(side_effect=vectors), submit_batch=AsyncMock(),
            find_batch=AsyncMock(), poll_batch=AsyncMock(), read_batch_results=AsyncMock())
        self.worker = MemoryWorker(store=self.store, embeddings=self.embeddings,
                                   providers={'openai': self.provider}, config=self.config)

    async def source(self, text, number, actor='telegram:user:7', topic=''):
        return await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text(text, metadata={
                'source': 'telegram', 'source_chat_id': '100', 'source_message_id': str(number),
                'actor_id': actor, 'actor_name': 'Alex', 'actor_kind': 'user',
                'sent_at': '2026-01-01T00:00:00+00:00', 'topic_id': topic}))

    async def ingest_pending(self):
        jobs = await self.store.claim_jobs(kind='memory_ingest', limit=16, lease_seconds=900)
        await self.worker._ingest(jobs)

    async def test_live_tail_accumulates_originals_across_batches_and_seals_after_idle(self):
        first = await self.source('I prefer jasmine tea.', 1)
        await self.ingest_pending()
        self.assertEqual((await self.store.get_excerpt_tail(self.session))['source_ids'], [first.db_id])
        second = await self.source('I prefer coffee.', 2, actor='telegram:user:8')
        await self.ingest_pending()
        tail = await self.store.get_excerpt_tail(self.session)
        self.assertEqual(tail['source_ids'], [first.db_id, second.db_id])
        # These sources are already lexically searchable before the semantic tail seals.
        self.assertEqual((await self.store.search_messages(self.session, 'jasmine'))[0]['id'], first.db_id)
        self.embeddings.embed_documents.assert_not_awaited()
        async with self.store.pool.connection() as conn:
            await conn.execute("UPDATE excerpt_tails SET updated_at=now()-interval '31 minutes'")
        jobs = await self.store.claim_jobs(kind='memory_tail', limit=1, lease_seconds=900)
        await self.worker._tail(jobs)
        self.assertIsNone(await self.store.get_excerpt_tail(self.session))
        jobs = await self.store.claim_jobs(kind='memory_embed', limit=16, lease_seconds=900)
        await self.worker._embed(jobs)
        document = self.embeddings.embed_documents.await_args.args[0][0]
        self.assertIn('telegram:user:7 Alex] I prefer jasmine tea.', document.text)
        self.assertIn('telegram:user:8 Alex] I prefer coffee.', document.text)
        self.assertEqual(document.text.count('I prefer jasmine tea.'), 1)
        self.assertLessEqual(len(document.text.encode()), 512)

    async def test_long_original_splits_into_exact_nonoverlapping_spans_and_remains_readable(self):
        text = '原始消息ABCDEFGHIJKLMNOPQRSTUVWXYZ ' * 100
        source = await self.source(text, 1)
        self.worker.batch = True
        await self.ingest_pending()
        async with self.store.pool.connection() as conn:
            rows = await (await conn.execute('SELECT spans FROM excerpts ORDER BY id')).fetchall()
        spans = [span for row in rows for span in row['spans']]
        self.assertGreater(len(spans), 1)
        self.assertEqual(spans[0]['start'], 0)
        self.assertEqual(spans[-1]['end'], len(text))
        self.assertTrue(all(left['end'] == right['start'] for left, right in zip(spans, spans[1:])))
        self.assertEqual(''.join(text[span['start']:span['end']] for span in spans), text)
        self.assertEqual((await self.store.read_messages(self.session, [source.db_id]))[0].message.parts[0].text, text)

    async def test_profile_generation_uses_original_actor_evidence_and_rejects_fabricated_sources(self):
        source = await self.source('I prefer concise replies.', 1)
        await self.ingest_pending()
        jobs = await self.store.claim_jobs(kind='memory_profile', limit=1, lease_seconds=900)
        fact = {'subject_actor_id': 'telegram:user:7', 'asserted_by': 'telegram:user:7',
                'claim': 'Prefers concise replies', 'kind': 'explicit', 'source_ids': [source.db_id],
                'valid_from': None, 'valid_to': None}
        self.provider.responses = [ProviderResponse(final_text=json.dumps({'facts': [fact]}))]
        await self.worker._profile(jobs)
        saved = await self.store.get_profile(self.session, 'telegram:user:7')
        self.assertEqual(saved[0]['source_ids'], [source.db_id])
        self.assertEqual(saved[0]['claim'], fact['claim'])
        request = self.provider.requests[-1]
        self.assertEqual(request['tools'], [])
        self.assertIn('I prefer concise replies.', request['messages'][0].parts[0].text)
        self.assertIn('telegram:user:7', request['messages'][0].parts[0].text)
        await self.store.hide_message_ids(self.session, [source.db_id])
        self.assertEqual(await self.store.get_profile(self.session, 'telegram:user:7'), [])

    async def test_pending_batch_cannot_publish_after_full_reset(self):
        source = await self.source('Remember the riverside cafe.', 1)
        self.worker.batch = True
        await self.ingest_pending()
        jobs = await self.store.claim_jobs(kind='memory_embed', limit=16, lease_seconds=900)
        await self.worker._embed(jobs)
        jobs = await self.store.claim_jobs(kind='embedding_batch', limit=1, lease_seconds=900)
        await self.store.reset_full(self.session, self.config.default_session_settings())
        await self.worker._guarded(jobs, self.worker._batch)
        self.embeddings.submit_batch.assert_not_awaited()
        self.assertEqual(await self.store.search_messages(self.session, 'riverside'), [])

    async def test_ambiguous_paid_submission_is_reconciled_instead_of_submitted_twice(self):
        await self.source('Remember the riverside cafe.', 1)
        self.worker.batch = True
        await self.ingest_pending()
        await self.worker._embed(await self.store.claim_jobs(kind='memory_embed', limit=16, lease_seconds=900))
        jobs = await self.store.claim_jobs(kind='embedding_batch', limit=1, lease_seconds=900)
        job = jobs[0]
        self.embeddings.submit_batch.side_effect = TimeoutError('accepted but response lost')
        await self.worker._guarded(jobs, self.worker._batch)
        async with self.store.pool.connection() as conn:
            await conn.execute('UPDATE jobs SET available_at=now() WHERE id=%s', (job['id'],))
        self.embeddings.find_batch.return_value = BatchJob(name='batches/recovered', state='RUNNING',
            done=False, space_id=self.embeddings.space_id)
        retry = await self.store.claim_jobs(kind='embedding_batch', limit=1, lease_seconds=900)
        await self.worker._batch(retry)
        self.embeddings.submit_batch.assert_awaited_once()
        self.embeddings.find_batch.assert_awaited_once()
        async with self.store.pool.connection() as conn:
            stored = await (await conn.execute('SELECT payload FROM jobs WHERE id=%s', (job['id'],))).fetchone()
        self.assertEqual(stored['payload']['name'], 'batches/recovered')

    async def test_two_import_pages_process_every_original_without_replaying_intake(self):
        originals = [await self.source(f'Original preference number {index}.', index)
                     for index in range(1, 106)]
        scope = await self.store.get_scope(self.session)
        for start in (0, 100):
            await self.store.coalesce_memory_jobs(self.session,
                source_ids=[row.db_id for row in originals[start:start + 100]], expected_scope=scope)
        self.worker.batch = True
        self.assertTrue(await self.worker.run_once())
        async with self.store.pool.connection() as conn:
            excerpts = await (await conn.execute('SELECT source_ids FROM excerpts')).fetchall()
            pending = await (await conn.execute("SELECT count(*) AS n FROM jobs WHERE kind='memory_ingest'")).fetchone()
        self.assertEqual({mid for row in excerpts for mid in row['source_ids']}, {row.db_id for row in originals})
        self.assertEqual(pending['n'], 0)
        self.embeddings.submit_batch.assert_not_awaited()

    async def test_configured_large_ingest_and_profile_groups_retain_every_original_and_fact(self):
        originals = [await self.source(f'I prefer option {index}.', index)
                     for index in range(1, 206)]
        scope = await self.store.get_scope(self.session)
        # Rebuild/import may queue a page larger than the worker grouping default.
        async with self.store.pool.connection() as conn:
            await conn.execute("DELETE FROM jobs WHERE kind='memory_ingest'")
        await self.store.enqueue_job(self.session, 'memory_ingest',
            source_ids=[row.db_id for row in originals], expected_scope=scope)
        limits = from_env(WorkerConfig, 'MEMORY_WORKER', {
            'MEMORY_WORKER_INGEST_SOURCES': '250', 'MEMORY_WORKER_CLAIM_JOBS': '64',
            'MEMORY_WORKER_EXCERPT_TOKENS': '30000', 'MEMORY_WORKER_EXCERPT_CANDIDATE_CHARS': '32',
            'MEMORY_WORKER_PROFILE_REQUEST_BYTES': '50000', 'MEMORY_WORKER_PROFILE_MAX_FACTS': '23',
            'MEMORY_WORKER_PROFILE_CLAIM_CHARS': '1500', 'MEMORY_WORKER_PROFILE_OUTPUT_TOKENS': '8192',
            'MEMORY_WORKER_PROFILE_EXISTING_PER_ACTOR': '0', 'MEMORY_WORKER_PROFILE_EXISTING_TOTAL': '0'})
        self.worker = MemoryWorker(store=self.store, embeddings=self.embeddings,
            providers={'openai': self.provider}, config=self.config, batch=True, limits=limits)
        self.assertTrue(await self.worker.run_once())
        async with self.store.pool.connection() as conn:
            excerpts = await (await conn.execute('SELECT source_ids FROM excerpts')).fetchall()
        self.assertEqual({mid for row in excerpts for mid in row['source_ids']}, {row.db_id for row in originals})
        profile_jobs = await self.store.claim_jobs(kind='memory_profile', limit=10, lease_seconds=900)
        self.assertEqual(len(profile_jobs), 1)
        self.assertEqual(set(profile_jobs[0]['source_ids']), {row.db_id for row in originals})
        facts = [{'subject_actor_id': 'telegram:user:7', 'asserted_by': 'telegram:user:7',
                  'claim': f'Prefers option {index}', 'kind': 'explicit', 'source_ids': [row.db_id],
                  'valid_from': None, 'valid_to': None, 'supersedes': None}
                 for index, row in enumerate(originals[:23], start=1)]
        self.provider.responses = [ProviderResponse(final_text=json.dumps({'facts': facts}))]
        await self.worker._profile(profile_jobs)
        request = self.provider.requests[-1]
        evidence = json.loads(request['messages'][0].parts[0].text)
        self.assertEqual(len(evidence['original_evidence']), 205)
        self.assertEqual(evidence['existing_facts'], [])
        self.assertIn('at most 23 concise claims, each at most 1500 characters', request['instructions'])
        self.assertEqual(request['settings'].max_output_tokens, 8192)
        self.assertEqual(len(await self.store.get_profile(self.session, 'telegram:user:7', limit=100)), 23)

    async def test_profile_spans_include_long_message_end_and_exclude_other_bots(self):
        text = ('I have a long history. ' * 700) + 'My lasting preference is jasmine tea.'
        human = await self.source(text, 1)
        await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text('I prefer diesel fuel.', metadata={
                'source': 'telegram', 'source_chat_id': '100', 'source_message_id': '2',
                'actor_id': 'telegram:user:99', 'actor_kind': 'bot', 'actor_name': 'Another bot'}))
        await self.ingest_pending()
        jobs = await self.store.claim_jobs(kind='memory_profile', limit=16, lease_seconds=900)
        self.assertGreater(len(jobs), 1)
        self.provider.responses = [ProviderResponse(final_text='{"facts": []}') for _ in jobs]
        for job in jobs:
            await self.worker._profile([job])
        evidence = [piece for request in self.provider.requests
                    for piece in json.loads(request['messages'][0].parts[0].text)['original_evidence']]
        self.assertEqual({row['source_span']['message_id'] for row in evidence}, {human.db_id})
        self.assertEqual(''.join(row['text'] for row in evidence), text)
        self.assertTrue(evidence[-1]['text'].endswith('jasmine tea.'))

    async def test_invalid_later_profile_claim_cannot_partially_publish_valid_earlier_claim(self):
        source = await self.source('I prefer tea and concise replies.', 1)
        await self.ingest_pending()
        jobs = await self.store.claim_jobs(kind='memory_profile', limit=1, lease_seconds=900)
        fact = {'subject_actor_id': 'telegram:user:7', 'asserted_by': 'telegram:user:7',
                'claim': 'Prefers concise replies', 'kind': 'explicit', 'source_ids': [source.db_id],
                'valid_from': None, 'valid_to': None, 'supersedes': None}
        self.provider.responses = [ProviderResponse(final_text=json.dumps({'facts': [fact,
            {**fact, 'claim': 'Invented from absent evidence', 'source_ids': [source.db_id + 999]}]}))]
        with self.assertRaisesRegex(ValueError, 'unavailable source'):
            await self.worker._profile(jobs)
        self.assertEqual(await self.store.get_profile(self.session, 'telegram:user:7'), [])

    async def test_partial_paid_batch_retries_only_failed_original_through_standard_route(self):
        await self.source('First independent topic.', 1, topic='first')
        await self.source('Second independent topic.', 2, topic='second')
        self.worker.batch = True
        await self.ingest_pending()
        await self.worker._embed(await self.store.claim_jobs(kind='memory_embed', limit=16, lease_seconds=900))
        jobs = await self.store.claim_jobs(kind='embedding_batch', limit=1, lease_seconds=900)
        job = jobs[0]
        job['payload'].update(name='batches/accepted', phase='polling')
        await self.store.update_job_payload(job, payload=job['payload'])
        first_id, second_id = job['payload']['excerpt_ids']
        self.embeddings.poll_batch.return_value = BatchJob(name='batches/accepted', state='SUCCEEDED',
            done=True, space_id=self.embeddings.space_id)
        self.embeddings.read_batch_results.return_value = [
            BatchItemResult(str(first_id), vector=np.array([1.] + [0.] * 1535)),
            BatchItemResult(str(second_id), error={'message': 'temporary failure'})]
        await self.worker._batch(jobs)
        retry = await self.store.claim_jobs(kind='memory_embed', limit=16, lease_seconds=900)
        self.assertEqual([row['payload']['excerpt_id'] for row in retry], [second_id])
        await self.worker._embed(retry)
        self.embeddings.submit_batch.assert_not_awaited()
        self.embeddings.embed_documents.assert_awaited_once()
        self.assertEqual(len(self.embeddings.embed_documents.await_args.args[0]), 1)
        for excerpt_id in (first_id, second_id):
            self.assertTrue((await self.store.get_excerpt(self.session, excerpt_id))['has_embedding'])

    async def test_configured_bulk_group_above_old_source_ceiling_keeps_every_source(self):
        # Raising the operational source group retains all 1002 originals
        # in 334 documents without the previous hidden 1000-source ceiling.
        originals = [await self.source(f'Original {index}', index) for index in range(1, 1003)]
        scope = await self.store.get_scope(self.session)
        async with self.store.pool.connection() as conn:
            await conn.execute("DELETE FROM jobs WHERE kind='memory_ingest'")
        for start in range(0, len(originals), 3):
            source_ids = [row.db_id for row in originals[start:start + 3]]
            excerpt = await self.store.create_excerpt(self.session, source_ids, model=self.embeddings.space_id)
            await self.store.enqueue_job(self.session, 'memory_embed', source_ids=source_ids,
                payload={'excerpt_id': excerpt['id'], 'space_id': self.embeddings.space_id, 'batch': True},
                expected_scope=scope)
        self.worker.batch = True
        self.worker.limits = replace(self.worker.limits, embed_sources=2000)
        self.assertTrue(await self.worker.run_once())
        async with self.store.pool.connection() as conn:
            batches = await (await conn.execute("SELECT source_ids,payload FROM jobs WHERE kind='embedding_batch'")).fetchall()
            failures = await (await conn.execute("SELECT error FROM jobs WHERE status='failed'")).fetchall()
        self.assertEqual(failures, [])
        self.assertEqual(len(batches), 1)
        self.assertEqual({mid for row in batches for mid in row['source_ids']}, {row.db_id for row in originals})
        self.assertEqual(sum(len(row['payload']['excerpt_ids']) for row in batches), 334)
        self.embeddings.submit_batch.assert_not_awaited()

    async def test_parsed_attachment_is_recallable_without_becoming_a_user_profile_declaration(self):
        from tgchatbot.domain.models import MessagePart, MessageRole, PartKind
        source = await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage(role=MessageRole.USER, parts=[
                MessagePart(kind=PartKind.TEXT, text='[Attached file excerpt: guide.txt]\nI prefer jasmine tea.',
                            origin='attachment_excerpt')], metadata={
                'source': 'telegram', 'source_chat_id': '100', 'source_message_id': '1',
                'actor_id': 'telegram:user:7', 'actor_kind': 'user'}))
        self.worker.batch = True
        await self.ingest_pending()
        self.assertEqual((await self.store.search_messages(self.session, 'jasmine'))[0]['id'], source.db_id)
        jobs = await self.store.claim_jobs(kind='memory_embed', limit=16, lease_seconds=900)
        self.assertTrue(jobs)
        excerpt = await self.store.get_excerpt(self.session, jobs[0]['payload']['excerpt_id'])
        self.assertIn('I prefer jasmine tea.', excerpt['text'])
        self.assertEqual(await self.store.claim_jobs(kind='memory_profile'), [])

    async def test_edit_of_previous_context_stays_searchable_without_reentering_new_context(self):
        old = await self.source('Old preference for tea.', 1)
        await self.store.reset_context(self.session)
        self.runtime.invalidate_session(self.session)
        current = await self.source('New context asks about coffee.', 2)
        edited = await self.source('Corrected old preference for jasmine.', 1)
        self.assertEqual(edited.db_id, old.db_id)
        self.assertEqual([row.db_id for row in (await self.runtime._get_live_state(self.session)).raw_messages], [current.db_id])
        self.assertEqual((await self.store.search_messages(self.session, 'jasmine'))[0]['id'], old.db_id)
