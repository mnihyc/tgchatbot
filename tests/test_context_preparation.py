"""Archive preparation reuses the real compaction/storage flow without paid models."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.context_state import CompactionWorkingSet
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind, ProviderResponse, UsageInfo
from tgchatbot.embeddings.config import EmbeddingConfig


class ContextPreparationTests(BusinessTestCase):
    async def test_refresh_acknowledges_its_own_capacity_pass_but_not_a_concurrent_gap(self):
        settings = await self.settings(max_input_images=2, compact_target_images=1)
        for concurrent in (False, True):
            with self.subTest(concurrent=concurrent):
                await self.store.reset_context(self.session)
                image = ConversationMessage(MessageRole.USER, [MessagePart(PartKind.IMAGE, data_b64='YWJj')])
                await self.store.append_messages(self.session, [image, image, image])
                await self.store.retire_context_images(self.session, target_images=2)
                state = await self.runtime._get_live_state(self.session)
                async def fetch(*args, **kwargs):
                    if concurrent:
                        await self.store.retire_context_images(self.session, target_images=1)
                    for _ in range(2 if concurrent else 1):
                        await self.runtime.ingest_user_message(session_id=self.session, incoming_message=image)
                    return {'ok': True, 'profiles': []}
                self.runtime.memory = SimpleNamespace(fetch_profiles=AsyncMock(side_effect=fetch))
                await self.runtime._refresh_profiles_after_compaction(session_id=self.session, state=state,
                    settings=settings, provider=self.provider, instructions='', tools=[], emit=None, trigger=None)
                self.assertEqual(await self.store.get_compaction_version(self.session), 3 if concurrent else 2)
                self.assertEqual(await self.store.compaction_needs_profile_refresh(self.session), concurrent)
                if not concurrent:
                    memory = SimpleNamespace(tools=[], fetch_profiles=AsyncMock(return_value={'ok': True, 'profiles': []}))
                    restarted = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools,
                        providers={'openai': self.provider}, memory=memory)
                    self.provider.responses = [ProviderResponse(final_text='Continued')]
                    await restarted.run_turn(session_id=self.session, user_display_name='Person',
                        incoming_message=ConversationMessage.user_text('Continue'))
                    memory.fetch_profiles.assert_not_awaited()

    async def test_page_boundaries_preserve_unbatched_profile_tools_and_transport_notes(self):
        await self.settings()
        self.store.config = replace(self.store.config, history_page_size=2)
        for kind in ('profile_refresh', 'standalone_tool', 'auto_note'):
            with self.subTest(kind=kind):
                await self.store.reset_context(self.session)
                first = ConversationMessage.user_text('Earlier request')
                if kind == 'auto_note':
                    pair = [ConversationMessage.user_text('Automatic transport observation', metadata={'synthetic_role': 'auto_user_note'}),
                        ConversationMessage.user_text('The actual person said this')]
                else:
                    pair = [self.runtime._tool_observation_message(name='user_profile_fetch', phase=phase,
                        payload={'call_id': kind, 'arguments': {}} if phase == 'call' else {'call_id': kind, 'output': {}},
                        metadata_update={'synthetic_role': kind} if kind == 'profile_refresh' else {})
                        for phase in ('call', 'result')]
                rows = await self.store.append_messages(self.session, [first, *pair, ConversationMessage.user_text('Later')])
                window = await self.store.load_compaction_window(self.session)
                self.assertEqual([row.db_id for row in window.raw_messages], [row.db_id for row in rows[:3]])
                self.assertEqual(len(self.runtime._group_raw_compaction_units(window.raw_messages)[1]), 2)
                self.assertTrue(window.more_raw)

    async def test_preparation_image_retirement_does_not_touch_later_intake(self):
        await self.settings(max_input_images=2, compact_target_images=1)
        image = lambda name: ConversationMessage(MessageRole.USER, [MessagePart(PartKind.TEXT, text=name),
            MessagePart(PartKind.IMAGE, data_b64='YWJj', mime_type='image/png')])
        originals = await self.store.append_messages(self.session, [image(f'Old picture {i}') for i in range(4)])
        original_loader = self.store.load_compaction_window
        latest = None
        async def with_later_intake(*args, **kwargs):
            nonlocal latest
            state = await original_loader(*args, **kwargs)
            if latest is None:
                latest = await self.store.append_message(self.session, image('Later picture'))
            return state
        with patch.object(self.store, 'load_compaction_window', side_effect=with_later_intake):
            result = await self.runtime.prepare_context(session_id=self.session)
        self.assertEqual(result['through_message_id'], originals[-1].db_id)
        _blocks, raw = await self.store.load_live_context(self.session)
        self.assertEqual(raw[-1].db_id, latest.db_id)
        self.assertEqual(raw[-1].image_count, 1)
        self.assertEqual(sum(row.image_count for row in raw[:-1]), 1)

    async def test_image_only_compaction_refresh_survives_restart_and_acknowledges_observed_epoch(self):
        settings = await self.settings()
        image = ConversationMessage(MessageRole.USER, [MessagePart(PartKind.IMAGE, data_b64='YWJj')])
        await self.store.append_messages(self.session, [image, image, image])
        await self.store.retire_context_images(self.session, target_images=2)
        self.assertTrue(await self.store.compaction_needs_profile_refresh(self.session))
        async def concurrent_compaction(*args, **kwargs):
            await self.store.retire_context_images(self.session, target_images=1)
            return {'ok': True, 'profiles': []}
        restarted = AgentRuntime(config=self.config, store=self.store, tool_registry=None,
            providers={'openai': self.provider}, memory=SimpleNamespace(fetch_profiles=AsyncMock(side_effect=concurrent_compaction)))
        state = await restarted._get_live_state(self.session)
        await restarted._refresh_profiles_after_compaction(session_id=self.session, state=state, settings=settings,
            provider=self.provider, instructions='', tools=[], emit=None, trigger=None)
        self.assertTrue(await self.store.compaction_needs_profile_refresh(self.session))
        restarted.memory.fetch_profiles.side_effect = None
        restarted.memory.fetch_profiles.return_value = {'ok': True, 'profiles': []}
        await restarted._refresh_profiles_after_compaction(session_id=self.session, state=state, settings=settings,
            provider=self.provider, instructions='', tools=[], emit=None, trigger=None)
        self.assertFalse(await self.store.compaction_needs_profile_refresh(self.session))
        await self.store.retire_context_images(self.session, target_images=0)
        self.assertTrue(await self.store.compaction_needs_profile_refresh(self.session))
        await self.store.reset_context(self.session)
        self.assertFalse(await self.store.compaction_needs_profile_refresh(self.session))

    async def test_observed_token_calibration_keeps_compaction_timing_after_restart_and_reset(self):
        self.provider.responses = [ProviderResponse(final_text='Hello', usage=UsageInfo(input_tokens=1000000))]
        await self.runtime.run_turn(session_id=self.session, user_display_name='Person',
            incoming_message=ConversationMessage.user_text('Training request'))
        key = ('openai', self.config.openai.model, self.config.default_session_settings().tool_history_mode.value)
        learned = await self.store.load_token_calibration(key)
        self.assertGreater(learned, 1)
        outcomes = []
        warm = self.runtime
        cold = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools, providers={'openai': self.provider})
        for runtime in (warm, cold):
            await self.store.reset_full(self.session, self.config.default_session_settings())
            settings = await self.settings(compact_min_messages=2, min_raw_messages_reserve=1,
                compact_batch_tokens=2000, compact_keep_recent_ratio=0.1)
            await self.store.append_messages(self.session, [ConversationMessage.user_text(
                f'Earlier discussion {i}. ' + 'A participant explained their situation. ' * 100) for i in range(3)])
            from tgchatbot.core.prompting import build_system_prompt
            state = await runtime._get_live_state(self.session)
            raw = self.provider.estimate_request_tokens(settings=settings,
                messages=runtime._build_provider_history(state, settings=settings, provider_name='openai'),
                instructions=build_system_prompt(settings), tools=[]).total_tokens
            await self.settings(compact_trigger_tokens=int(raw * 1.4), compact_target_tokens=raw)
            self.provider.requests.clear()
            async def generate(**request):
                if request.get('response_schema'):
                    return await self.summary(**request)
                return ProviderResponse(final_text='Answer with retained context')
            with patch.object(self.provider, 'generate', side_effect=generate):
                result = await runtime.run_turn(session_id=self.session, user_display_name='Person',
                    incoming_message=ConversationMessage.user_text('Latest question'))
            self.assertEqual(result.text, 'Answer with retained context')
            outcomes.append([request['response_schema_name'] for request in self.provider.requests])
            self.assertEqual(await self.store.load_token_calibration(key), learned)
        self.assertTrue(outcomes[0])
        self.assertEqual(outcomes[0], outcomes[1])

    async def archive(self, count=45):
        await self.settings(compact_min_messages=2, min_raw_messages_reserve=1,
            compact_keep_recent_ratio=0.1, compact_batch_tokens=800)
        self.store.config = replace(self.store.config, history_page_size=8, memory_block_page_size=4)
        return await self.store.append_messages(self.session, [ConversationMessage.user_text(
            f'Record {number}: participant described their day and requested a thoughtful response.', metadata={
                'source': 'telegram', 'source_chat_id': '100', 'source_message_id': str(number),
                'actor_id': f'telegram:user:{number % 2 + 1}', 'actor_kind': 'user'}) for number in range(count)])

    async def summary(self, **request):
        self.provider.requests.append(request)
        if request.get('response_schema_name') == 'profile_patch':
            return ProviderResponse(final_text=json.dumps({'additions': [], 'removals': []}))
        fields = request['response_schema']['properties']
        result = {key: [] for key in fields}
        result['scope'] = 'Participants discussed their days.'
        if 'interaction_mode' in fields:
            result['interaction_mode'] = 'chat_or_sharing'
        if 'interaction_modes_seen' in fields:
            result['interaction_modes_seen'] = ['chat_or_sharing']
        return ProviderResponse(final_text=json.dumps(result))

    async def test_many_pages_prepare_without_loading_or_installing_full_archive(self):
        originals = await self.archive()
        windows = []
        loader = self.store.load_compaction_window
        async def capture(*args, **kwargs):
            window = await loader(*args, **kwargs)
            windows.append(window)
            return window
        with patch.object(self.provider, 'generate', side_effect=self.summary), patch.object(
            self.store, 'load_live_context_versioned', side_effect=AssertionError('Full archive load')), patch.object(
                self.store, 'load_compaction_window', side_effect=capture):
            result = await self.runtime.prepare_context(session_id=self.session)
        self.assertGreater(result['compactions'], 1)
        self.assertLessEqual(result['remaining_raw_messages'], 8)
        self.assertLessEqual(result['remaining_root_blocks'], 4)
        self.assertTrue(all(isinstance(window, CompactionWorkingSet) for window in windows))
        self.assertLessEqual(max(len(window.raw_messages) for window in windows), 8)
        self.assertLessEqual(max(len(window.blocks) for window in windows), 4)
        self.assertEqual(self.runtime._live_sessions, {})
        stored = await self.store.read_messages(self.session, [row.db_id for row in originals])
        self.assertEqual([row.message for row in stored], [row.message for row in originals])
        kinds = {request['response_schema_name'] for request in self.provider.requests}
        self.assertIn('episode_memory_block', kinds)
        self.assertIn('digest_memory_block', kinds)
        restarted = AgentRuntime(config=self.config, store=self.store, tool_registry=None, providers={'openai': self.provider})
        again = await restarted.prepare_context(session_id=self.session)
        self.assertEqual(again['compactions'], 0)
        live = await restarted._get_live_state(self.session)
        blocks, messages = await self.store.load_live_context(self.session)
        self.assertEqual(live.blocks, blocks)
        self.assertEqual(live.raw_messages, messages)
        self.assertEqual(messages[-1].db_id, originals[-1].db_id)

    async def test_preparation_resumes_committed_work_after_cancellation(self):
        originals = await self.archive(25)
        calls = 0
        async def interrupted(**request):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise asyncio.CancelledError()
            return await self.summary(**request)
        with patch.object(self.provider, 'generate', side_effect=interrupted):
            with self.assertRaises(asyncio.CancelledError):
                await self.runtime.prepare_context(session_id=self.session)
        first = await self.store.list_memory_blocks(self.session)
        self.assertEqual(len(first), 1)
        restarted = AgentRuntime(config=self.config, store=self.store, tool_registry=None, providers={'openai': self.provider})
        with patch.object(self.provider, 'generate', side_effect=self.summary):
            result = await restarted.prepare_context(session_id=self.session)
        self.assertGreater(result['compactions'], 0)
        self.assertEqual(len(await self.store.read_messages(self.session, [row.db_id for row in originals])), 25)
        self.assertIn(first[0].block_id, [block.block_id for block in await self.store.list_memory_blocks(self.session, limit=100)])

    async def test_rejected_summaries_stop_preparation_without_losing_or_hiding_records(self):
        originals = await self.archive(12)
        with patch.object(self.provider, 'generate', return_value=ProviderResponse(final_text='{}')):
            with self.assertRaisesRegex(RuntimeError, 'cannot progress.*Originals'):
                await self.runtime.prepare_context(session_id=self.session)
        blocks, messages = await self.store.load_live_context(self.session)
        self.assertEqual(blocks, [])
        self.assertEqual([row.db_id for row in messages], [row.db_id for row in originals])

    async def test_page_boundary_extends_through_all_calls_results_and_interleaved_intake(self):
        await self.settings()
        self.store.config = replace(self.store.config, history_page_size=2)
        await self.store.append_message(self.session, ConversationMessage.user_text('Run both checks'))
        calls = await self.store.append_messages(self.session, [self.runtime._tool_observation_message(
            name='shell_exec', phase='call', payload={'call_id': str(i), 'arguments': {}},
            metadata_update={'tool_batch_id': 'batch'}) for i in range(2)])
        await self.store.append_message(self.session, ConversationMessage.user_text('Additional detail'))
        for index, call in enumerate(calls):
            await self.store.append_message(self.session, self.runtime._tool_observation_message(
                name='shell_exec', phase='result', payload={'call_id': str(index), 'output': {'ok': True}},
                metadata_update={'tool_batch_id': 'batch', 'tool_call_message_id': call.db_id}))
        latest = await self.store.append_message(self.session, ConversationMessage.user_text('Latest question'))
        window = await self.store.load_compaction_window(self.session)
        self.assertEqual(len(window.raw_messages), 6)
        self.assertTrue(window.more_raw)
        self.assertEqual(window.through_message_id, latest.db_id)
        unit = self.runtime._group_raw_compaction_units(window.raw_messages)[1]
        self.assertEqual(len(unit), 5)

    async def test_later_intake_is_active_but_outside_finite_preparation_watermark(self):
        originals = await self.archive(18)
        newest = None
        async def with_intake(**request):
            nonlocal newest
            if newest is None:
                newest = await self.store.append_message(self.session, ConversationMessage.user_text('New live question'))
            return await self.summary(**request)
        with patch.object(self.provider, 'generate', side_effect=with_intake):
            result = await self.runtime.prepare_context(session_id=self.session)
        self.assertEqual(result['through_message_id'], originals[-1].db_id)
        self.assertGreater(newest.db_id, result['through_message_id'])
        live = await self.runtime._get_live_state(self.session)
        self.assertEqual(live.raw_messages[-1].db_id, newest.db_id)

    async def test_operator_command_constructs_only_learning_resources_and_persists_profile_exchange(self):
        from tgchatbot.tools.memory import prepare_context
        await self.archive(18)
        with patch('tgchatbot.providers.factory.build_providers', return_value={'openai': self.provider}), patch.object(
            self.provider, 'generate', side_effect=self.summary), patch('tgchatbot.tools.memory.emit'):
            result = await prepare_context(self.store, self.config, EmbeddingConfig(), session_id=self.session)
        self.assertEqual(result['type'], 'context_prepared')
        self.assertGreater(result['compactions'], 0)
        _blocks, messages = await self.store.load_live_context(self.session)
        refresh = [row for row in messages if row.message.metadata.get('synthetic_role') == 'profile_refresh']
        self.assertEqual([row.message.role for row in refresh], [MessageRole.TOOL, MessageRole.TOOL])
        self.assertEqual([row.message.metadata['tool_phase'] for row in refresh], ['call', 'result'])
        self.assertEqual(refresh[0].message.name, 'user_profile_fetch')
        self.assertNotIn('refresh_error', refresh[1].message.metadata['tool_payload']['output'])
        self.assertFalse(await self.store.compaction_needs_profile_refresh(self.session))
        self.tools.runner.run.assert_not_awaited()

    async def test_completed_preparation_refresh_is_recovered_after_restart_without_new_compaction(self):
        from tgchatbot.tools.memory import prepare_context
        await self.archive(18)
        with patch.object(self.provider, 'generate', side_effect=self.summary):
            await self.runtime.prepare_context(session_id=self.session)
        self.assertTrue(await self.store.compaction_needs_profile_refresh(self.session))
        with patch('tgchatbot.providers.factory.build_providers', return_value={'openai': self.provider}), patch(
            'tgchatbot.tools.memory.emit'), patch.object(self.provider, 'generate', side_effect=self.summary):
            result = await prepare_context(self.store, self.config, EmbeddingConfig(), session_id=self.session)
        self.assertEqual(result['compactions'], 0)
        self.assertFalse(await self.store.compaction_needs_profile_refresh(self.session))
