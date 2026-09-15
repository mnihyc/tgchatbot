"""Profile reads stay cheap; context transitions hand learning to durable jobs."""
from __future__ import annotations

import json
from types import SimpleNamespace

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.memory import MemoryService
from tgchatbot.core.memory_worker import MemoryWorker
from tgchatbot.domain.models import ConversationMessage, MessageRole, ProviderResponse
from tgchatbot.tools.base import ToolContext


class ProfileRefreshOwnership(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.settings()
        self.embeddings = SimpleNamespace(enabled=False)
        self.memory = MemoryService(self.store, self.embeddings)
        self.worker = MemoryWorker(store=self.store, embeddings=self.embeddings,
            providers={'openai': self.provider}, config=self.config)
        self.memory.worker = self.worker
        self.runtime.memory = self.memory

    async def source(self, number, text='I prefer jasmine tea.'):
        return await self.store.append_message(self.session, ConversationMessage.user_text(text, metadata={
            'source': 'telegram', 'source_chat_id': '100', 'source_message_id': str(number),
            'actor_id': 'telegram:user:7', 'actor_kind': 'user', 'actor_name': 'Participant',
            'sent_at': '2026-01-02T03:04:05+00:00'}))

    def response(self, source=None):
        additions = [] if source is None else [{'subject_actor_id': 'telegram:user:7',
            'asserted_by': 'telegram:user:7', 'claim': 'Prefers jasmine tea', 'kind': 'explicit',
            'source_ids': [source.db_id], 'valid_from': None, 'valid_to': None, 'supersedes': None,
            'status': 'active', 'reason': 'The participant explicitly states this preference.'}]
        return ProviderResponse(final_text=json.dumps({'additions': additions, 'removals': []}))

    async def request_count(self):
        async with self.store.pool.connection() as conn:
            return (await (await conn.execute("SELECT count(*) AS n FROM jobs WHERE kind='memory_profile_request'"))
                .fetchone())['n']

    async def test_repeated_profile_tool_reads_do_not_learn_new_tiny_messages(self):
        first = await self.source(1)
        await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7',
            asserted_by='telegram:user:7', claim='Prefers jasmine tea', source_ids=[first.db_id])
        tool = next(tool for tool in self.memory.tools if tool.name == 'user_profile_fetch')
        for number in (2, 3, 4):
            await self.source(number, 'neko')
            result = await tool.runner.run({'actor_ids': ['person_id:7']}, ToolContext(self.session, 'Participant'))
            self.assertEqual(result.output['profiles'][0]['facts'][0]['claim'], 'Prefers jasmine tea')
        self.assertEqual(self.provider.requests, [])
        async with self.store.pool.connection() as conn:
            pending = (await (await conn.execute('SELECT sum(pending_bytes) AS n FROM profile_inputs')).fetchone())['n']
        self.assertEqual(pending, len('I prefer jasmine tea.') + 3 * len('neko'))

    async def test_compaction_catches_up_once_before_appending_the_profile_tool_pair(self):
        source = await self.source(1)
        await self.store.create_memory_block(self.session, summary_text='Earlier preference.',
            estimated_tokens=10, source_message_ids=[source.db_id])
        self.provider.responses = [self.response(source)]
        await self.runtime.prepare_context(session_id=self.session)
        self.assertEqual(len(self.provider.requests), 1)
        rows = await self.store.list_uncompacted_messages(self.session)
        pair = [row.message for row in rows if row.message.metadata.get('synthetic_role') == 'profile_refresh']
        self.assertEqual([row.metadata['tool_phase'] for row in pair], ['call', 'result'])
        self.assertTrue(all(row.role == MessageRole.TOOL for row in pair))
        self.assertEqual(pair[1].metadata['tool_payload']['output']['profiles'][0]['facts'][0]['claim'],
            'Prefers jasmine tea')
        await self.runtime.prepare_context(session_id=self.session)
        await self.memory.fetch_profiles(self.session, ['person_id:7'])
        self.assertEqual(len(self.provider.requests), 1, 'The same compaction and reads do not buy another patch')

    async def test_reset_queues_small_batch_without_model_work_and_survives_restart(self):
        source = await self.source(1)
        old_scope = await self.store.get_scope(self.session)
        new_scope = await self.store.reset_context(self.session)
        self.assertEqual(self.provider.requests, [])
        self.assertEqual(new_scope['generation'], old_scope['generation'])
        self.assertEqual(await self.store.list_uncompacted_messages(self.session), [])
        self.assertEqual(await self.request_count(), 1)
        self.assertTrue((await self.memory.search(self.session, 'jasmine'))['matches'])
        reopened = await self.new_store()
        worker = MemoryWorker(store=reopened, embeddings=self.embeddings,
            providers={'openai': self.provider}, config=self.config)
        self.provider.responses = [self.response(source)]
        self.assertTrue(await worker.run_once())
        self.assertEqual(len(self.provider.requests), 1)
        self.assertEqual((await reopened.get_profile(self.session, 'telegram:user:7'))[0]['claim'],
            'Prefers jasmine tea')
        self.assertEqual(await self.request_count(), 0)

    async def test_reset_during_inflight_batch_preserves_a_coalesced_request_for_new_tail(self):
        first = await self.source(1)
        busy = await self.store.claim_profile_batch(session_id=self.session, lazy=True,
            max_bytes=self.worker.limits.profile_request_bytes, lease_seconds=self.worker.limits.lease_seconds)
        second = await self.source(2, 'I like calm replies.')
        await self.store.reset_context(self.session)
        await self.store.reset_context(self.session)
        self.assertEqual(await self.request_count(), 1)
        self.assertIsNone(await self.store.claim_profile_batch(
            max_bytes=self.worker.limits.profile_request_bytes, lease_seconds=self.worker.limits.lease_seconds))
        self.provider.responses = [self.response(), self.response()]
        await self.worker._profile([busy])
        followup = await self.store.claim_profile_batch(
            max_bytes=self.worker.limits.profile_request_bytes, lease_seconds=self.worker.limits.lease_seconds)
        self.assertEqual(followup['source_ids'], [second.db_id])
        self.assertNotIn(first.db_id, followup['source_ids'])
        await self.worker._profile([followup])
        self.assertEqual(await self.request_count(), 0)
        self.assertEqual(len(self.provider.requests), 2)

    async def test_profile_failure_after_reset_retries_same_batch_without_losing_original(self):
        source = await self.source(1)
        await self.store.reset_context(self.session)
        self.provider.responses = [RuntimeError('temporary provider error'), self.response(source)]
        with self.assertLogs('tgchatbot.core.memory_worker', level='WARNING'):
            await self.worker.run_once()
        async with self.store.pool.connection() as conn:
            before = await (await conn.execute("SELECT * FROM jobs WHERE kind='memory_profile'")).fetchone()
            await conn.execute("UPDATE jobs SET available_at=now() WHERE id=%s", (before['id'],))
        self.assertEqual(before['status'], 'pending')
        self.assertEqual(before['source_ids'], [source.db_id])
        self.assertEqual(await self.request_count(), 0, 'The materialized retryable batch now owns the request')
        self.assertEqual(await self.store.list_uncompacted_messages(self.session), [])
        retry = await self.store.claim_profile_batch(
            max_bytes=self.worker.limits.profile_request_bytes, lease_seconds=self.worker.limits.lease_seconds)
        self.assertEqual(retry['id'], before['id'])
        await self.worker._profile([retry])
        self.assertEqual((await self.store.get_profile(self.session, 'telegram:user:7'))[0]['claim'],
            'Prefers jasmine tea')
        self.assertTrue(await self.store.read_messages(self.session, [source.db_id]))

    async def test_empty_reset_adds_no_learning_and_full_reset_retires_old_requests(self):
        await self.store.reset_context(self.session)
        self.assertEqual(await self.request_count(), 0)
        await self.source(1)
        await self.store.reset_context(self.session)
        self.assertEqual(await self.request_count(), 1)
        await self.store.reset_full(self.session, self.config.default_session_settings())
        self.assertIsNone(await self.store.claim_profile_batch(
            max_bytes=self.worker.limits.profile_request_bytes, lease_seconds=self.worker.limits.lease_seconds))
        self.assertEqual(self.provider.requests, [])
