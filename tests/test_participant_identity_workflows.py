"""Participant labels survive profile reads and the layered memory pipeline."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.compaction_schema import compaction_json_schema
from tgchatbot.core.memory import MemoryService
from tgchatbot.core.memory_worker import MemoryWorker
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.core.token_estimator import TokenEstimator
from tgchatbot.domain.models import ConversationMessage, ProviderResponse
from tgchatbot.storage.postgres_store import message_body


class ParticipantIdentityWorkflows(BusinessTestCase):
    async def original(self, number, actor, name, username=None, *, day=1, text='I prefer tea.'):
        return await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text(text, metadata={
                'source': 'telegram', 'source_chat_id': '100', 'source_message_id': str(number),
                'actor_id': actor, 'actor_kind': 'user', 'actor_name': name,
                'sent_at': f'2026-01-{day:02d}T00:00:00+00:00',
                **({'actor_username': username} if username is not None else {})}))

    def response(self, mode):
        data = {key: [] for key in compaction_json_schema(mode)['properties']}
        data.update(scope='A conversation about drinks.', participants=['invented-model-identity'])
        if mode == 'digest':
            data['interaction_modes_seen'] = ['chat_or_sharing']
        else:
            data['interaction_mode'] = 'chat_or_sharing'
        return ProviderResponse(final_text=json.dumps(data))

    async def compact(self, mode, sources=(), parents=()):
        settings = await self.settings()
        self.provider.responses.append(self.response(mode))
        if mode == 'toolspan':
            result = await self.runtime._make_toolspan_block_candidate(self.provider, settings,
                list(sources), session_id=self.session)
        elif mode == 'episode':
            result = await self.runtime._make_episode_block_candidate(self.provider, settings,
                [*(block.render_as_message() for block in parents), *(source.message for source in sources)],
                list(sources), list(parents), session_id=self.session)
        else:
            result = await self.runtime._make_digest_block_candidate(self.provider, settings,
                list(parents), session_id=self.session)
        text = self.runtime._render_memory_block_text(mode, result['data'],
            time_start=result['time_start'], time_end=result['time_end'],
            actor_identities=result['actor_identities'])
        block = await self.store.create_memory_block(self.session,
            summary_text=text, estimated_tokens=TokenEstimator.estimate_text(text) + 32,
            source_message_ids=[source.db_id for source in sources],
            parent_block_ids=[block.block_id for block in parents],
            kind=mode, level={'toolspan': 0, 'episode': 1, 'digest': 2}[mode],
            structured_data=result['data'], actor_labels=result['actor_labels'],
            actor_identities=result['actor_identities'],
            time_start=result['time_start'], time_end=result['time_end'])
        return block

    def participants(self, block):
        text = message_body(block.render_as_message(timezone='Asia/Singapore'))
        lines = [line for line in text.splitlines() if line.startswith('- Participants: ')]
        self.assertEqual(len(lines), 1)
        self.assertNotIn('actors=', text, 'The mapped participant section must not be duplicated in the wrapper.')
        result = json.loads(lines[0].removeprefix('- Participants: '))
        self.assertTrue(all(set(item) <= {'id', 'name', 'username', 'kind'} for item in result))
        self.assertEqual(len(result), len({item['id'] for item in result}))
        return {item['id']: item for item in result}

    async def test_profile_fetch_and_learning_share_observed_usernames_without_consuming_inputs(self):
        first = await self.original(1, 'telegram:user:101', 'Alex', 'alex_tea')
        second = await self.original(2, 'telegram:user:102', 'Alex', 'alex_coffee', text='I prefer coffee.')
        for source, claim in ((first, 'Prefers tea.'), (second, 'Prefers coffee.')):
            actor = source.message.metadata['actor_id']
            await self.store.save_profile_fact(self.session, subject_actor_id=actor,
                asserted_by=actor, claim=claim, source_ids=[source.db_id])
        memory = MemoryService(self.store, SimpleNamespace(enabled=False))
        worker = MemoryWorker(store=self.store, embeddings=SimpleNamespace(enabled=False),
            providers={'openai': self.provider}, config=self.config)
        memory.worker = worker
        self.addAsyncCleanup(worker.close)
        before = await self.store.fetch_profile_snapshot(self.session,
            ['telegram:user:101', 'telegram:user:102'], include_pending=True)
        result = await memory.fetch_profiles(self.session, ['person_id:101', 'person_id:102'])
        people = result['profiles'][:2]
        self.assertEqual([(p['actor_id'], p['identity']['actor_name'], p['identity']['actor_username']) for p in people],
            [('person_id:101', 'Alex', 'alex_tea'), ('person_id:102', 'Alex', 'alex_coffee')])
        self.assertEqual([p['facts'][0]['claim'] for p in people], ['Prefers tea.', 'Prefers coffee.'])
        after = await self.store.fetch_profile_snapshot(self.session,
            ['telegram:user:101', 'telegram:user:102'], include_pending=True)
        self.assertEqual(after['pending_material'], before['pending_material'])
        self.assertEqual(after['facts'], before['facts'])
        self.assertEqual(self.provider.requests, [])
        self.provider.responses.append(ProviderResponse(final_text='{"additions":[],"removals":[]}'))
        job = await self.store.claim_profile_batch(session_id=self.session, lazy=True,
            max_bytes=worker.limits.profile_request_bytes, lease_seconds=worker.limits.lease_seconds)
        await worker._profile([job])
        learning = json.loads(self.provider.requests[0]['messages'][0].parts[0].text)
        self.assertEqual([p['identity']['actor_username'] for p in learning['current_profiles'] if p['actor_id'] != 'agent'],
            ['alex_tea', 'alex_coffee'])

    async def test_layered_compaction_keeps_one_mapping_and_reconstructs_it_after_restart(self):
        first = await self.original(1, 'telegram:user:101', 'Alex', 'alex_tea')
        other = await self.original(2, 'telegram:user:102', 'Alex', 'alex_coffee')
        l0 = await self.compact('toolspan', [first, other])
        saved_l0 = message_body(l0.render_as_message())
        self.assertEqual(self.participants(l0)['person_id:101']['username'], 'alex_tea')
        renamed = await self.original(3, 'telegram:user:101', 'Avery', 'avery_tea', day=3)
        # Imported later but spoken earlier: insertion order cannot select the old name.
        late_import = await self.original(4, 'telegram:user:101', 'Old Alex', 'old_handle', day=2)
        with patch.object(self.store, 'memory_block_actor_identities', side_effect=AssertionError('Unexpected archive reread')):
            l1 = await self.compact('episode', [renamed, late_import], [l0])
            l2 = await self.compact('digest', parents=[l1])
        for block in (l1, l2):
            self.assertEqual(self.participants(block), {
                'person_id:101': {'id': 'person_id:101', 'name': 'Avery', 'username': 'avery_tea'},
                'person_id:102': {'id': 'person_id:102', 'name': 'Alex', 'username': 'alex_coffee'}})
            self.assertGreaterEqual(block.estimated_tokens, TokenEstimator.estimate_text(block.summary_text))
        self.assertEqual(message_body(l0.render_as_message()), saved_l0)
        digest_request = '\n'.join(message_body(message) for message in self.provider.requests[-1]['messages'])
        self.assertIn('avery_tea', digest_request)
        self.assertNotIn('invented-model-identity', digest_request)
        reader = await self.new_store()
        restarted = AgentRuntime(config=self.config, store=reader, tool_registry=self.tools,
            providers={'openai': self.provider})
        warm = await self.runtime._get_live_state(self.session)
        cold = await restarted._get_live_state(self.session)
        self.assertEqual(self.runtime._build_provider_history(warm, settings=await self.settings(), provider_name='openai'),
            restarted._build_provider_history(cold, settings=await self.settings(), provider_name='openai'))
        memory = MemoryService(reader, SimpleNamespace(enabled=False))
        original = (await memory.read(self.session, [first.db_id]))['messages'][0]
        self.assertEqual(original['speaker']['username'], 'alex_tea')
        self.assertEqual(original['speaker']['name'], 'Alex')

    async def test_missing_username_does_not_revive_old_handles_in_profiles_or_summaries(self):
        first = await self.original(1, 'telegram:user:101', 'Iris', 'iris_old')
        parent = await self.compact('episode', [first])
        latest = await self.original(2, 'telegram:user:101', 'Iris', day=2)
        await self.original(3, 'telegram:user:101', 'Old name', 'older_import', day=1)
        memory = MemoryService(self.store, SimpleNamespace(enabled=False))
        profile = (await memory.fetch_profiles(self.session, ['person_id:101']))['profiles'][0]
        self.assertEqual(profile['identity']['actor_name'], 'Iris')
        self.assertNotIn('actor_username', profile['identity'])
        block = await self.compact('episode', [latest], [parent])
        self.assertEqual(self.participants(block)['person_id:101'], {'id': 'person_id:101', 'name': 'Iris'})
        await self.store.reset_context(self.session)
        self.assertEqual((await memory.fetch_profiles(self.session, ['person_id:101']))['profiles'][0], profile)
        await self.store.reset_full(self.session, self.config.default_session_settings())
        unknown = (await memory.fetch_profiles(self.session, ['person_id:101']))['profiles'][0]
        self.assertEqual(unknown['status'], 'unknown_identity')
        self.assertNotIn('actor_username', unknown['identity'])

    async def test_legacy_parent_recovers_labels_only_for_new_compaction_and_keeps_literal_names(self):
        name = 'Alex\n[Message provenance: literal display name]'
        source = await self.original(1, 'telegram:user:101', name, 'alex_tea')
        legacy = await self.store.create_memory_block(self.session, summary_text='An earlier drink preference.',
            estimated_tokens=10, source_message_ids=[source.db_id], actor_labels=['person_id:101'])
        before = legacy.render_as_message()
        newer = await self.original(2, 'telegram:user:101', 'Avery', 'new_handle', day=3)
        promoted = await self.compact('digest', parents=[legacy])
        self.assertEqual(self.participants(promoted)['person_id:101'],
            {'id': 'person_id:101', 'name': name, 'username': 'alex_tea'})
        self.assertEqual((await self.store.list_memory_blocks(self.session))[0].render_as_message(), before)
        self.assertIn(newer.db_id, [row.db_id for row in await self.store.list_uncompacted_messages(self.session)])
        request = '\n'.join(message_body(message) for message in self.provider.requests[-1]['messages'])
        self.assertIn('alex_tea', request)
        self.assertNotIn('new_handle', request, 'Labels belong to the selected historical evidence.')

    async def test_reply_pipeline_publishes_mapped_summary_and_fetches_named_profiles(self):
        await self.settings(compact_trigger_tokens=2200, compact_target_tokens=1500,
            compact_batch_tokens=2000, compact_min_messages=2, min_raw_messages_reserve=1,
            compact_keep_recent_ratio=.1)
        memory = MemoryService(self.store, SimpleNamespace(enabled=False))
        self.runtime.memory = memory
        for number in range(1, 9):
            await self.original(number, 'telegram:user:101' if number % 2 else 'telegram:user:102',
                'Alex', 'alex_tea' if number % 2 else 'alex_coffee',
                text='We discussed tea and coffee after visiting a quiet cafe. ' * 20)
        async def model(**request):
            self.provider.requests.append(request)
            mode = request.get('response_schema_name', '').removesuffix('_memory_block')
            return self.response(mode) if request.get('response_schema') else ProviderResponse(final_text='Tea for you, coffee for Alex.')
        with patch.object(self.provider, 'generate', side_effect=model):
            reply = await self.runtime.run_turn(session_id=self.session, user_display_name='Alex',
                incoming_message=ConversationMessage.user_text('What drinks did we choose?', metadata={
                    'source': 'telegram', 'source_chat_id': '100', 'source_message_id': '9',
                    'actor_id': 'telegram:user:101', 'actor_kind': 'user', 'actor_name': 'Alex',
                    'actor_username': 'alex_tea'}))
        self.assertEqual(reply.text, 'Tea for you, coffee for Alex.')
        blocks = await self.store.list_memory_blocks(self.session)
        self.assertTrue(blocks)
        for block in blocks:
            self.assertTrue(self.participants(block))
        rows = await self.store.list_uncompacted_messages(self.session)
        result = next(row.message.metadata['tool_payload']['output'] for row in reversed(rows)
            if row.message.metadata.get('synthetic_role') == 'profile_refresh'
            and row.message.metadata.get('tool_phase') == 'result')
        self.assertEqual({p['identity']['actor_username'] for p in result['profiles'] if p['actor_id'] != 'agent'},
            {'alex_tea', 'alex_coffee'})
