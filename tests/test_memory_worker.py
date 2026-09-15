from __future__ import annotations

import asyncio
import json
from dataclasses import replace
import os
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import numpy as np

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.memory import MemoryService
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

    async def profile_jobs(self):
        job = await self.store.claim_profile_batch(session_id=self.session, lazy=True,
            max_bytes=self.worker.limits.profile_request_bytes, lease_seconds=self.worker.limits.lease_seconds)
        return [job] if job else []

    def patch_response(self, facts=(), removals=()):
        return ProviderResponse(final_text=json.dumps({'additions': [dict(fact, status=fact.get('status', 'active'),
            reason='Supported by the cited original declaration.') for fact in facts], 'removals': list(removals)}))

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
        jobs = await self.profile_jobs()
        fact = {'subject_actor_id': 'telegram:user:7', 'asserted_by': 'telegram:user:7',
                'claim': 'Prefers concise replies', 'kind': 'explicit', 'source_ids': [source.db_id],
                'valid_from': None, 'valid_to': None}
        self.provider.responses = [self.patch_response([fact])]
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
            'MEMORY_WORKER_PROFILE_REQUEST_BYTES': '50000', 'MEMORY_WORKER_PROFILE_OUTPUT_TOKENS': '8192'})
        self.worker = MemoryWorker(store=self.store, embeddings=self.embeddings,
            providers={'openai': self.provider}, config=replace(self.config, memory=replace(self.config.memory, profile_bytes=8192)), batch=True, limits=limits)
        self.assertTrue(await self.worker.run_once())
        async with self.store.pool.connection() as conn:
            excerpts = await (await conn.execute('SELECT source_ids FROM excerpts')).fetchall()
        self.assertEqual({mid for row in excerpts for mid in row['source_ids']}, {row.db_id for row in originals})
        profile_jobs = await self.profile_jobs()
        self.assertEqual(len(profile_jobs), 1)
        self.assertEqual(set(profile_jobs[0]['source_ids']), {row.db_id for row in originals})
        facts = [{'subject_actor_id': 'telegram:user:7', 'asserted_by': 'telegram:user:7',
                  'claim': f'Prefers option {index}', 'kind': 'explicit', 'source_ids': [row.db_id],
                  'valid_from': None, 'valid_to': None, 'supersedes': None}
                 for index, row in enumerate(originals[:23], start=1)]
        self.provider.responses = [self.patch_response(facts)]
        await self.worker._profile(profile_jobs)
        request = self.provider.requests[-1]
        evidence = json.loads(request['messages'][0].parts[0].text)
        self.assertEqual(len(evidence['original_evidence']), 205)
        self.assertTrue(all(not profile['facts'] for profile in evidence['current_profiles']))
        self.assertIn('8192 UTF-8 bytes', request['instructions'])
        self.assertNotIn('must fit', request['instructions'])
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
        batches = 0
        while jobs := await self.profile_jobs():
            self.provider.responses = [self.patch_response()]
            await self.worker._profile(jobs)
            batches += 1
        self.assertGreater(batches, 1)
        evidence = [piece for request in self.provider.requests
                    for piece in json.loads(request['messages'][0].parts[0].text)['original_evidence']]
        self.assertEqual({row['message_id'] for row in evidence}, {human.db_id})
        self.assertEqual(''.join(fragment['text'] for row in evidence for fragment in row['fragments']), text)
        self.assertTrue(evidence[-1]['fragments'][-1]['text'].endswith('jasmine tea.'))

    async def test_invalid_later_profile_claim_cannot_partially_publish_valid_earlier_claim(self):
        source = await self.source('I prefer tea and concise replies.', 1)
        await self.ingest_pending()
        jobs = await self.profile_jobs()
        fact = {'subject_actor_id': 'telegram:user:7', 'asserted_by': 'telegram:user:7',
                'claim': 'Prefers concise replies', 'kind': 'explicit', 'source_ids': [source.db_id],
                'valid_from': None, 'valid_to': None, 'supersedes': None}
        self.provider.responses = [self.patch_response([fact,
            {**fact, 'claim': 'Invented from absent evidence', 'source_ids': [source.db_id + 999]}])]
        with self.assertRaisesRegex(ValueError, 'unavailable source'):
            await self.worker._profile(jobs)
        self.assertEqual(await self.store.get_profile(self.session, 'telegram:user:7'), [])

    async def test_background_profiles_wait_for_a_batch_and_do_not_depend_on_embeddings(self):
        self.embeddings.enabled = False
        self.worker.limits = replace(self.worker.limits, profile_request_bytes=26)
        first = await self.source('I prefer tea.', 1)
        for _ in range(3):
            await self.worker.run_once()
        self.assertEqual(self.provider.requests, [], 'A short message must not trigger profile generation')
        second = await self.source('I like birds.', 2)
        self.provider.responses = [self.patch_response([{
            'subject_actor_id': 'telegram:user:7', 'asserted_by': 'telegram:user:7',
            'claim': 'Prefers tea and likes birds', 'kind': 'explicit',
            'source_ids': [first.db_id, second.db_id], 'valid_from': None, 'valid_to': None, 'supersedes': None}])]
        for _ in range(5):
            await self.worker.run_once()
            if self.provider.requests:
                break
        self.assertEqual(len(self.provider.requests), 1)
        self.assertEqual((await self.store.get_profile(self.session, 'telegram:user:7'))[0]['source_ids'],
                         [first.db_id, second.db_id])
        self.embeddings.embed_documents.assert_not_awaited()

    async def test_bulk_import_does_not_starve_profiles_while_new_originals_keep_arriving(self):
        self.worker.batch = True
        self.embeddings.enabled = False
        self.worker.limits = replace(self.worker.limits, profile_request_bytes=26)
        self.provider.responses = [self.patch_response() for _ in range(5)]
        for number in range(1, 6):
            await self.source('I prefer jasmine tea. I like quiet places.', number)
            await self.worker.run_once()
        self.assertTrue(self.provider.requests, 'Profile processing must progress before the import queue drains')
        self.assertTrue(await self.store.search_messages(self.session, 'jasmine'))

    async def test_lazy_profile_fetch_processes_only_one_unicode_batch_before_indexing(self):
        self.worker.limits = replace(self.worker.limits, profile_request_bytes=32)
        text = '我喜欢安静的地方和有趣的企鹅。' * 4
        original = await self.source(text, 1)
        memory = MemoryService(self.store, self.embeddings)
        memory.worker = self.worker
        seen = ''
        while len(seen) < len(text):
            self.provider.responses = [self.patch_response()]
            before = len(self.provider.requests)
            result = await memory.fetch_profiles(self.session, ['telegram:user:7'])
            self.assertTrue(result['ok'])
            self.assertNotIn('more_available', result)
            self.assertEqual(len(self.provider.requests), before + 1)
            evidence = json.loads(self.provider.requests[-1]['messages'][0].parts[0].text)['original_evidence']
            chunk = ''.join(fragment['text'] for item in evidence for fragment in item['fragments'])
            self.assertLessEqual(len(chunk.encode('utf-8')), 32)
            seen += chunk
        self.assertEqual(seen, text)
        await memory.fetch_profiles(self.session, ['telegram:user:7'])
        self.assertEqual(len(self.provider.requests), before + 1, 'Consumed sources must not learn again')
        self.assertEqual((await self.store.read_messages(self.session, [original.db_id]))[0].message.parts[0].text, text)
        self.embeddings.embed_documents.assert_not_awaited()

    async def test_database_failure_rolls_back_every_profile_fact_and_source_cursor(self):
        original = await self.source('I prefer tea. I like birds.', 1)
        facts = [{'subject_actor_id': 'telegram:user:7', 'asserted_by': 'telegram:user:7',
            'claim': claim, 'kind': 'explicit', 'source_ids': [original.db_id],
            'valid_from': None, 'valid_to': None, 'supersedes': None} for claim in ('Prefers tea', 'Likes birds')]
        async with self.store.pool.connection() as conn:
            await conn.execute('''CREATE FUNCTION reject_second_fact() RETURNS trigger LANGUAGE plpgsql AS $$
                BEGIN IF NEW.claim='Likes birds' THEN RAISE EXCEPTION 'synthetic publication failure'; END IF;
                RETURN NEW; END $$''')
            await conn.execute('''CREATE TRIGGER reject_second_fact BEFORE INSERT ON profile_facts
                FOR EACH ROW EXECUTE FUNCTION reject_second_fact()''')
        self.provider.responses = [self.patch_response(facts)]
        from psycopg.errors import RaiseException
        with self.assertRaises(RaiseException):
            await self.worker._profile(await self.profile_jobs())
        self.assertEqual(await self.store.get_profile(self.session, 'telegram:user:7'), [])
        async with self.store.pool.connection() as conn:
            pending = (await (await conn.execute('SELECT pending_bytes FROM profile_inputs WHERE message_id=%s', (original.db_id,))).fetchone())['pending_bytes']
            audits = (await (await conn.execute('SELECT count(*) AS n FROM profile_patches')).fetchone())['n']
        self.assertEqual(pending, len('I prefer tea. I like birds.'.encode()))
        self.assertEqual(audits, 0)

    async def test_concurrent_lazy_fetches_do_not_learn_or_publish_the_same_batch_twice(self):
        source = await self.source('I prefer jasmine tea.', 1)
        started, release = asyncio.Event(), asyncio.Event()
        async def generate(**kwargs):
            started.set()
            await release.wait()
            return self.patch_response([{'subject_actor_id': 'telegram:user:7', 'asserted_by': 'telegram:user:7',
                'claim': 'Prefers jasmine tea', 'kind': 'explicit', 'source_ids': [source.db_id],
                'valid_from': None, 'valid_to': None, 'supersedes': None}])
        memory = MemoryService(self.store, self.embeddings)
        memory.worker = self.worker
        with patch.object(self.provider, 'generate', AsyncMock(side_effect=generate)) as request:
            first = asyncio.create_task(memory.fetch_profiles(self.session, ['telegram:user:7']))
            await asyncio.wait_for(started.wait(), timeout=5)
            try:
                second = await memory.fetch_profiles(self.session, ['telegram:user:7'])
                self.assertEqual(second['profiles'][0]['facts'], [], 'Read the committed state while learning is in flight')
                request.assert_awaited_once()
            finally:
                release.set()
            result = await first
        self.assertEqual(result['profiles'][0]['facts'][0]['claim'], 'Prefers jasmine tea')
        async with self.store.pool.connection() as conn:
            audits = (await (await conn.execute('SELECT count(*) AS n FROM profile_patches')).fetchone())['n']
        self.assertEqual(audits, 1)

    async def test_valid_patch_above_soft_target_commits_once_and_preserves_retired_evidence(self):
        first = await self.source('I prefer tea.', 1)
        old = await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7',
            asserted_by='telegram:user:7', claim='Prefers tea', source_ids=[first.db_id])
        second = await self.source('I now prefer a quieter place.', 2)
        self.provider.responses = [self.patch_response([{'subject_actor_id': 'telegram:user:7',
            'asserted_by': 'telegram:user:7', 'claim': '安静' * 1000, 'kind': 'explicit',
            'source_ids': [second.db_id], 'valid_from': None, 'valid_to': None, 'supersedes': None}],
            [{'fact_id': old['id'], 'reason': 'Replace redundant detail.'}])]
        await self.worker._profile(await self.profile_jobs())
        facts = await self.store.get_profile(self.session, 'telegram:user:7')
        self.assertEqual([fact['claim'] for fact in facts], ['安静' * 1000])
        self.assertEqual(facts[0]['source_ids'], [second.db_id])
        memory = MemoryService(self.store, self.embeddings)
        memory.worker = self.worker
        result = await memory.fetch_profiles(self.session, ['telegram:user:7'])
        self.assertEqual(result['profiles'][0]['facts'][0]['claim'], '安静' * 1000)
        self.assertEqual(len(self.provider.requests), 1, 'Size alone must not spend another request')
        async with self.store.pool.connection() as conn:
            pending = (await (await conn.execute('SELECT sum(pending_bytes) AS n FROM profile_inputs')).fetchone())['n']
            audit = await (await conn.execute('SELECT claim,retired_at FROM profile_facts WHERE id=%s', (old['id'],))).fetchone()
            patches = (await (await conn.execute('SELECT count(*) AS n FROM profile_patches')).fetchone())['n']
        self.assertEqual(pending, 0)
        self.assertEqual(patches, 1)
        self.assertEqual(audit['claim'], 'Prefers tea')
        self.assertIsNotNone(audit['retired_at'])

    async def test_structured_retirement_keeps_audit_and_withdrawal_restores_if_evidence_is_hidden(self):
        first = await self.source('I prefer tea.', 1)
        old = await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7',
            asserted_by='telegram:user:7', claim='Prefers tea', source_ids=[first.db_id])
        withdrawal = await self.source('That preference no longer applies.', 2)
        self.provider.responses = [self.patch_response([{'subject_actor_id': 'telegram:user:7',
            'asserted_by': 'telegram:user:7', 'claim': 'Withdraws preference for tea', 'kind': 'explicit',
            'status': 'retracted', 'source_ids': [withdrawal.db_id], 'valid_from': None,
            'valid_to': None, 'supersedes': old['id']}])]
        await self.worker._profile(await self.profile_jobs())
        self.assertEqual(await self.store.get_profile(self.session, 'telegram:user:7'), [])
        await self.store.hide_message_ids(self.session, [withdrawal.db_id])
        self.assertEqual(await self.store.get_profile(self.session, 'telegram:user:7'), [])
        self.provider.responses = [self.patch_response([{'subject_actor_id': 'telegram:user:7',
            'asserted_by': 'telegram:user:7', 'claim': 'Prefers tea', 'kind': 'explicit',
            'source_ids': [first.db_id], 'valid_from': None, 'valid_to': None, 'supersedes': None}])]
        await self.worker._profile(await self.profile_jobs())
        self.assertEqual([fact['id'] for fact in await self.store.get_profile(self.session, 'telegram:user:7')], [old['id']])
        async with self.store.pool.connection() as conn:
            audit = (await (await conn.execute('SELECT patch FROM profile_patches')).fetchone())['patch']
        self.assertEqual(audit['additions'][0]['status'], 'retracted')
        self.assertIn('reason', audit['additions'][0])

    async def test_full_reset_during_profile_generation_cannot_publish_or_process_old_evidence(self):
        await self.source('A previous-generation preference.', 1)
        async def reset_in_flight(**kwargs):
            await self.store.reset_full(self.session, self.config.default_session_settings())
            return self.patch_response()
        with patch.object(self.provider, 'generate', AsyncMock(side_effect=reset_in_flight)):
            from tgchatbot.storage.postgres_store import StaleScopeError
            with self.assertRaises(StaleScopeError):
                await self.worker._profile(await self.profile_jobs())
        self.assertEqual(await self.profile_jobs(), [])
        async with self.store.pool.connection() as conn:
            sources = (await (await conn.execute('SELECT count(*) AS n FROM message_revisions')).fetchone())['n']
            patches = (await (await conn.execute('SELECT count(*) AS n FROM profile_patches')).fetchone())['n']
        self.assertEqual(sources, 1)
        self.assertEqual(patches, 0)

    async def test_profile_consolidation_retains_old_and_new_original_evidence(self):
        first = await self.source('I prefer tea.', 1)
        old = await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7',
            asserted_by='telegram:user:7', claim='Prefers tea', source_ids=[first.db_id])
        self.provider.responses = [self.patch_response()]
        await self.worker._profile(await self.profile_jobs())
        second = await self.source('I prefer it unsweetened.', 2)
        self.provider.responses = [self.patch_response([{'subject_actor_id': 'telegram:user:7',
            'asserted_by': 'telegram:user:7', 'claim': 'Prefers unsweetened tea', 'kind': 'explicit',
            'source_ids': [first.db_id, second.db_id], 'valid_from': None, 'valid_to': None, 'supersedes': None}],
            [{'fact_id': old['id'], 'reason': 'The consolidated statement preserves both supported preferences.'}])]
        await self.worker._profile(await self.profile_jobs())
        facts = await self.store.get_profile(self.session, 'telegram:user:7')
        self.assertEqual([fact['source_ids'] for fact in facts], [[first.db_id, second.db_id]])
        self.assertEqual(facts[0]['claim'], 'Prefers unsweetened tea')
        async with self.store.pool.connection() as conn:
            audit = (await (await conn.execute('SELECT source_revisions FROM profile_patches ORDER BY id DESC LIMIT 1')).fetchone())['source_revisions']
        self.assertEqual(audit, {str(first.db_id): 1, str(second.db_id): 1})
        await self.store.hide_message_ids(self.session, [second.db_id])
        self.assertEqual(await self.store.get_profile(self.session, 'telegram:user:7'), [])
        self.provider.responses = [self.patch_response([{'subject_actor_id': 'telegram:user:7',
            'asserted_by': 'telegram:user:7', 'claim': 'Prefers tea', 'kind': 'explicit',
            'source_ids': [first.db_id], 'valid_from': None, 'valid_to': None, 'supersedes': None}])]
        await self.worker._profile(await self.profile_jobs())
        restored = await self.store.get_profile(self.session, 'telegram:user:7')
        self.assertEqual([fact['id'] for fact in restored], [old['id']])
        self.assertEqual(restored[0]['claim'], 'Prefers tea')

    async def test_soft_reset_during_profile_generation_preserves_learning(self):
        source = await self.source('I prefer calm replies.', 1)
        async def reset_in_flight(**kwargs):
            await self.store.reset_context(self.session)
            return self.patch_response([{'subject_actor_id': 'telegram:user:7', 'asserted_by': 'telegram:user:7',
                'claim': 'Prefers calm replies', 'kind': 'explicit', 'source_ids': [source.db_id],
                'valid_from': None, 'valid_to': None, 'supersedes': None}])
        with patch.object(self.provider, 'generate', AsyncMock(side_effect=reset_in_flight)):
            await self.worker._profile(await self.profile_jobs())
        self.assertEqual((await self.store.get_profile(self.session, 'telegram:user:7'))[0]['claim'], 'Prefers calm replies')
        self.assertEqual(await self.profile_jobs(), [])

    async def test_lazy_batches_preserve_order_across_unicode_parts(self):
        from tgchatbot.domain.models import MessagePart, PartKind
        self.worker.limits = replace(self.worker.limits, profile_request_bytes=4)
        original = ConversationMessage.user_text('喜喜', metadata={'source': 'telegram', 'source_chat_id': '100',
            'source_message_id': '1', 'actor_id': 'telegram:user:7', 'actor_kind': 'user'})
        original.parts.append(MessagePart(kind=PartKind.TEXT, text='AB'))
        await self.runtime.ingest_user_message(session_id=self.session, incoming_message=original)
        seen = []
        while jobs := await self.profile_jobs():
            self.provider.responses = [self.patch_response()]
            await self.worker._profile(jobs)
            evidence = json.loads(self.provider.requests[-1]['messages'][0].parts[0].text)['original_evidence']
            seen.extend(fragment['text'] for item in evidence for fragment in item['fragments'])
        self.assertEqual(''.join(seen), '喜喜AB')

    async def test_profile_input_batch_too_small_for_unicode_keeps_source_pending(self):
        source = await self.source('喜', 1)
        with self.assertRaisesRegex(ValueError, 'PROFILE_REQUEST_BYTES must fit one source character'):
            await self.store.claim_profile_batch(session_id=self.session, lazy=True,
                max_bytes=1, lease_seconds=self.worker.limits.lease_seconds)
        async with self.store.pool.connection() as conn:
            pending = (await (await conn.execute('SELECT pending_bytes FROM profile_inputs WHERE message_id=%s',
                (source.db_id,))).fetchone())['pending_bytes']
            jobs = (await (await conn.execute("SELECT count(*) AS n FROM jobs WHERE kind='memory_profile'")).fetchone())['n']
        self.assertEqual(pending, 3)
        self.assertEqual(jobs, 0)
        self.assertEqual(self.provider.requests, [])

    async def test_generated_service_and_attachment_descriptions_are_not_profile_declarations(self):
        for number, origin in enumerate(('attachment_excerpt', 'attachment_reference', 'service_event'), start=1):
            message = ConversationMessage.user_text('I prefer the service-generated description.', metadata={
                'source': 'telegram', 'source_chat_id': '100', 'source_message_id': str(number),
                'actor_id': 'telegram:user:7', 'actor_kind': 'user'})
            message.parts[0].origin = origin
            source = await self.runtime.ingest_user_message(session_id=self.session, incoming_message=message)
            with self.assertRaisesRegex(ValueError, 'original message'):
                await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7',
                    asserted_by='telegram:user:7', claim='A fabricated personal preference', source_ids=[source.db_id])
        self.assertEqual(await self.profile_jobs(), [])
        self.assertEqual(self.provider.requests, [])

    async def test_removal_only_patch_cannot_commit_after_existing_evidence_changes(self):
        first = await self.source('I prefer tea.', 1)
        old = await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7',
            asserted_by='telegram:user:7', claim='Prefers tea', source_ids=[first.db_id])
        self.provider.responses = [self.patch_response()]
        await self.worker._profile(await self.profile_jobs())
        second = await self.source('That detail is no longer useful.', 2)
        async def change_in_flight(**kwargs):
            await self.store.hide_message_ids(self.session, [first.db_id])
            return self.patch_response(removals=[{'fact_id': old['id'], 'reason': 'Redundant profile detail.'}])
        from tgchatbot.storage.postgres_store import StaleScopeError
        with patch.object(self.provider, 'generate', AsyncMock(side_effect=change_in_flight)):
            with self.assertRaises(StaleScopeError):
                await self.worker._profile(await self.profile_jobs())
        async with self.store.pool.connection() as conn:
            audits = (await (await conn.execute('SELECT count(*) AS n FROM profile_patches')).fetchone())['n']
            pending = (await (await conn.execute('SELECT pending_bytes FROM profile_inputs WHERE message_id=%s', (second.db_id,))).fetchone())['pending_bytes']
        self.assertEqual(audits, 1, 'The stale retirement must not publish another audit patch')
        self.assertGreater(pending, 0)

    async def test_lower_profile_target_preserves_human_membership_without_new_paid_request(self):
        source = await self.source('I prefer quiet places and calm, concise replies.', 1)
        old = [await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7',
            asserted_by='telegram:user:7', claim=f'Preference {number}: ' + '安静' * 80,
            source_ids=[source.db_id]) for number in range(3)]
        self.provider.responses = [self.patch_response()]
        await self.worker._profile(await self.profile_jobs())
        self.worker.config = replace(self.config, memory=replace(self.config.memory, profile_bytes=900))
        memory = MemoryService(self.store, self.embeddings, config=self.worker.config.memory)
        memory.worker = self.worker
        before = len(self.provider.requests)
        for _ in range(2):
            profile = (await memory.fetch_profiles(self.session, ['telegram:user:7']))['profiles'][0]
            self.assertEqual([fact['id'] for fact in profile['facts']], [fact['id'] for fact in reversed(old)])
            self.assertGreater(len(json.dumps(profile, ensure_ascii=False).encode('utf-8')), 900)
        self.assertEqual(len(self.provider.requests), before)
        self.assertEqual(await self.profile_jobs(), [])

    async def test_lower_agent_profile_target_preserves_human_assertor_without_new_paid_request(self):
        source = await self.source('Please keep your replies calm and concise.', 1)
        old = [await self.store.save_profile_fact(self.session, subject_actor_id='agent',
            asserted_by='telegram:user:7', claim=f'Reply preference {number}: ' + '安静' * 80,
            source_ids=[source.db_id]) for number in range(3)]
        self.provider.responses = [self.patch_response()]
        await self.worker._profile(await self.profile_jobs())
        self.worker.config = replace(self.config, memory=replace(self.config.memory, profile_bytes=900))
        memory = MemoryService(self.store, self.embeddings, config=self.worker.config.memory)
        memory.worker = self.worker
        before = len(self.provider.requests)
        result = await memory.fetch_profiles(self.session, ['agent'])
        self.assertNotIn('refresh_error', result)
        profile = result['profiles'][0]
        self.assertEqual(len(self.provider.requests), before)
        self.assertGreater(len(json.dumps(profile, ensure_ascii=False).encode('utf-8')), 900)
        self.assertEqual([fact['id'] for fact in profile['facts']], [fact['id'] for fact in reversed(old)])
        self.assertTrue(all(fact['asserted_by'] == 'telegram:user:7' and fact['source_ids'] == [source.db_id]
                            for fact in profile['facts']))

    async def test_lazy_subject_selection_does_not_spend_its_batch_on_unrelated_old_backlog(self):
        older = await self.source('Unrelated older material. ' * 100, 1, actor='telegram:user:8')
        requested = await self.source('I prefer jasmine tea.', 2)
        memory = MemoryService(self.store, self.embeddings)
        memory.worker = self.worker
        self.provider.responses = [self.patch_response([{'subject_actor_id': 'telegram:user:7',
            'asserted_by': 'telegram:user:7', 'claim': 'Prefers jasmine tea', 'kind': 'explicit',
            'source_ids': [requested.db_id], 'valid_from': None, 'valid_to': None, 'supersedes': None}])]
        result = await memory.fetch_profiles(self.session, ['telegram:user:7'])
        evidence = json.loads(self.provider.requests[-1]['messages'][0].parts[0].text)['original_evidence']
        self.assertEqual({row['message_id'] for row in evidence}, {requested.db_id})
        self.assertEqual(result['profiles'][0]['facts'][0]['claim'], 'Prefers jasmine tea')
        async with self.store.pool.connection() as conn:
            pending = (await (await conn.execute('SELECT pending_bytes FROM profile_inputs WHERE message_id=%s', (older.db_id,))).fetchone())['pending_bytes']
        self.assertGreater(pending, 0)

    async def test_lazy_subject_request_leaves_unrelated_pending_job_to_background(self):
        older = await self.source('I enjoy hiking.', 1, actor='telegram:user:8')
        job = (await self.profile_jobs())[0]
        await self.store.defer_job(job, payload=job['payload'], delay_seconds=0)
        await self.source('I prefer jasmine tea.', 2)
        memory = MemoryService(self.store, self.embeddings)
        memory.worker = self.worker
        before = len(self.provider.requests)
        result = await memory.fetch_profiles(self.session, ['telegram:user:7'])
        self.assertEqual(len(self.provider.requests), before)
        self.assertEqual(result['profiles'][0]['facts'], [])
        self.assertNotIn('refresh_error', result)
        self.provider.responses = [self.patch_response()]
        self.assertTrue(await self.worker.run_once())
        evidence = json.loads(self.provider.requests[-1]['messages'][0].parts[0].text)['original_evidence']
        self.assertEqual({row['message_id'] for row in evidence}, {older.db_id})

    async def test_background_profiles_share_progress_between_chats_with_old_backlogs(self):
        await self.source('Earlier imported preferences. ' * 100, 1)
        other = 'telegram:200'
        await self.store.get_or_create_session(other, self.config.default_session_settings())
        await self.store.append_message(other, ConversationMessage.user_text('Recent preferences. ' * 100,
            metadata={'source': 'telegram', 'source_chat_id': '200', 'source_message_id': '1',
                      'actor_id': 'telegram:user:8', 'actor_kind': 'user'}))
        self.provider.responses = [self.patch_response() for _ in range(4)]
        serviced = []
        for _ in range(4):
            job = await self.store.claim_profile_batch(max_bytes=64, lease_seconds=900)
            serviced.append(job['session_id'])
            await self.worker._profile([job])
        self.assertEqual(serviced, [self.session, other, self.session, other])
        async with self.store.pool.connection() as conn:
            pending = await (await conn.execute('SELECT pending_bytes FROM profile_inputs')).fetchall()
        self.assertTrue(all(row['pending_bytes'] > 0 for row in pending))

    async def test_many_restored_audit_facts_reenter_bounded_batches_without_expanding_current_profile(self):
        common = await self.source('I enjoy learning hobbies.', 1)
        selected = await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7',
            asserted_by='telegram:user:7', claim='Enjoys learning hobbies', source_ids=[common.db_id])
        self.provider.responses = [self.patch_response()]
        await self.worker._profile(await self.profile_jobs())
        originals = []
        for number in range(20):
            source = await self.source(f'I prefer option {number}.', number + 2)
            originals.append(source)
            independent = await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7',
                asserted_by='telegram:user:7', claim=f'Prefers option {number}', source_ids=[source.db_id])
            self.provider.responses = [self.patch_response([{'subject_actor_id': 'telegram:user:7',
                'asserted_by': 'telegram:user:7', 'claim': f'Enjoys learning hobbies and prefers option {number}',
                'kind': 'explicit', 'source_ids': [common.db_id, source.db_id],
                'valid_from': None, 'valid_to': None, 'supersedes': None}],
                [{'fact_id': fact['id'], 'reason': 'Consolidated supported preferences.'} for fact in (selected, independent)])]
            await self.worker._profile(await self.profile_jobs())
            selected = (await self.store.get_profile(self.session, 'telegram:user:7'))[0]
        await self.store.hide_message_ids(self.session, [common.db_id])
        self.assertEqual(await self.store.get_profile(self.session, 'telegram:user:7'), [])
        async with self.store.pool.connection() as conn:
            current = (await (await conn.execute('SELECT fact_ids FROM profile_current WHERE actor_id=%s', ('telegram:user:7',))).fetchone())['fact_ids']
            restored = (await (await conn.execute("SELECT count(*) AS n FROM profile_facts WHERE valid AND claim LIKE 'Prefers option %'")).fetchone())['n']
        self.assertEqual(current, [])
        self.assertEqual(restored, 20)
        self.worker.limits = replace(self.worker.limits, profile_request_bytes=64)
        memory = MemoryService(self.store, self.embeddings)
        memory.worker = self.worker
        self.provider.responses = [self.patch_response([{'subject_actor_id': 'telegram:user:7',
            'asserted_by': 'telegram:user:7', 'claim': 'Prefers option 0', 'kind': 'explicit',
            'source_ids': [originals[0].db_id], 'valid_from': None, 'valid_to': None, 'supersedes': None}])]
        before = len(self.provider.requests)
        result = await memory.fetch_profiles(self.session, ['telegram:user:7'])
        self.assertEqual(len(self.provider.requests), before + 1)
        request = json.loads(self.provider.requests[-1]['messages'][0].parts[0].text)
        self.assertTrue(all(not profile['facts'] for profile in request['current_profiles']))
        self.assertLessEqual(sum(len(fragment['text'].encode('utf-8'))
            for item in request['original_evidence'] for fragment in item['fragments']), 64)
        self.assertEqual([fact['claim'] for fact in result['profiles'][0]['facts']], ['Prefers option 0'])
        self.assertLessEqual(len(json.dumps(result['profiles'][0], ensure_ascii=False).encode('utf-8')), 4096)

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
