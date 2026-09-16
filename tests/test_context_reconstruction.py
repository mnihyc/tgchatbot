"""Business invariants for database-owned context and interrupted execution."""
from __future__ import annotations

import asyncio
import json
import httpx
from dataclasses import replace
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from telegram.error import BadRequest

from tests.business_helpers import BusinessTestCase, ScriptedProvider
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import (ChatMode, ConversationMessage, MessagePart, MessageRole,
    OutboundArtifact, PartKind, ProviderResponse, ToolCall, ToolHistoryMode, ToolResult)
from tgchatbot.storage.previews import PreviewCache
from tgchatbot.tools.file_send import FileSendTool
from tgchatbot.tools.remote_workspace import RemoteFileResult
from tgchatbot.transports.artifact_delivery import deliver_artifact


class ContextReconstructionTests(BusinessTestCase):
    async def test_same_name_file_outcomes_keep_exact_paths_through_compaction_restart_and_provider_change(self):
        await self.file_outcome_history(partial_preparation=False)

    async def test_unavailable_same_name_file_keeps_requested_path_through_compaction_restart_and_provider_change(self):
        await self.file_outcome_history(partial_preparation=True)

    async def file_outcome_history(self, *, partial_preparation):
        settings = await self.settings(mode=ChatMode.ASSIST, compact_trigger_tokens=100000,
            tool_history_mode=ToolHistoryMode.NATIVE_SAME_PROVIDER)
        prefix = 'shared-project/' * 20
        paths = [prefix + day + '/report.txt' for day in ('2026-04-29', '2026-04-30')]
        originals, artifacts = [], []
        for number, workspace_path in enumerate(paths):
            original, transfer = self.path / f'original-{number}.txt', self.path / f'transfer-{number}.txt'
            original.write_text(f'Original report {number}')
            originals.append(original)
            if not partial_preparation or number == 1:
                transfer.write_bytes(original.read_bytes())
                artifacts.append(OutboundArtifact(transfer, 'report.txt', temporary=True,
                    workspace_path=workspace_path))
        by_path = {artifact.workspace_path: artifact for artifact in artifacts}
        remote = SimpleNamespace(fetch_files=AsyncMock(return_value=[
            RemoteFileResult(path, artifact=by_path.get(path),
                error=None if path in by_path else 'Remote file transfer failed.') for path in paths]))
        sender = FileSendTool(self.config, remote)
        self.tools.spec = sender.spec
        self.tools.list_tools.return_value = [sender.spec]
        arguments = {'paths': paths}
        self.provider.responses = [ProviderResponse(tool_calls=[ToolCall('file_send', 'reports', arguments)],
            continuation_items=[{'type': 'function_call', 'call_id': 'reports', 'name': 'file_send',
                'arguments': json.dumps(arguments)}]), ProviderResponse(final_text='The reports are prepared.')]
        result = await self.runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Send both dated reports.'))
        deliveries = [SimpleNamespace(message_id=77)]
        if not partial_preparation:
            deliveries.insert(0, BadRequest('This transfer was rejected'))
        bot = SimpleNamespace(send_document=AsyncMock(side_effect=deliveries))
        receipts = []
        with (nullcontext() if partial_preparation else
                self.assertLogs('tgchatbot.transports.artifact_delivery', level='ERROR')):
            for artifact in result.artifacts:
                receipt = await deliver_artifact(bot, chat_id=100, artifact=artifact)
                receipts.append(receipt)
                await self.runtime.record_tool_observation(session_id=self.session, name='file_send',
                    phase='delivery', payload=receipt, expected_scope=result.scope)
        expected_deliveries = [(paths[1], 'sent')] if partial_preparation else [(paths[0], 'failed'), (paths[1], 'sent')]
        self.assertEqual([(item['workspace_path'], item['delivery_state']) for item in receipts], expected_deliveries)
        self.assertTrue(all(not artifact.path.exists() for artifact in artifacts))
        self.assertEqual([original.read_text() for original in originals], ['Original report 0', 'Original report 1'])
        state = await self.runtime._get_live_state(self.session)
        original_rows = await self.store.read_messages(self.session, [row.db_id for row in state.raw_messages])
        prepared = next(row.message.metadata['tool_payload']['output'] for row in original_rows
            if row.message.metadata.get('tool_phase') == 'result')
        self.assertEqual([item['workspace_path'] for item in prepared['files']], paths)
        self.assertEqual([item['status'] for item in prepared['files']],
            ['failed', 'queued'] if partial_preparation else ['queued', 'queued'])
        self.assertEqual([row.message.metadata['tool_payload'] for row in original_rows
            if row.message.metadata.get('tool_phase') == 'delivery'], receipts,
            'Original delivery receipts retain transport IDs for audit.')
        visible_receipts = [{'filename': 'report.txt', 'workspace_path': path, 'status': status,
            **({'error': 'BadRequest'} if status == 'failed' else {})} for path, status in expected_deliveries]
        expected_records = [arguments, prepared, *visible_receipts]
        requests = []

        def assert_file_records(request):
            records = []
            decoder = json.JSONDecoder()
            for message in request['messages']:
                for part in message.parts:
                    text = part.text or ''
                    for index, character in enumerate(text):
                        if character == '{':
                            try:
                                record, _end = decoder.raw_decode(text[index:])
                                records.append(record)
                            except json.JSONDecodeError:
                                pass
            for record in expected_records:
                self.assertIn(record, records,
                    'The model must receive exact file identities with their separate preparation/delivery outcomes.')

        async def summarize(**request):
            requests.append(request)
            assert_file_records(request)
            data = {key: [] for key in request['response_schema']['properties']}
            data.update(scope='Sending dated reports', interaction_mode='task_execution',
                artifacts=paths, results_or_takeaways=[f'Failed: {paths[0]}', f'Sent: {paths[1]}'])
            return ProviderResponse(final_text=json.dumps(data))

        with patch.object(self.provider, 'generate', side_effect=summarize):
            candidate = await self.runtime._make_toolspan_block_candidate(self.provider, settings,
                state.raw_messages, session_id=self.session)
        self.assertEqual(candidate['data']['artifacts'], paths)
        changed = await self.settings(provider='gemini', model='fixture-next-model')
        await self.store.close()
        restored_store = await self.new_store()
        restored_cache = PreviewCache(restored_store, max_bytes=0)
        self.addCleanup(restored_cache.close)
        next_provider = ScriptedProvider(name='gemini', responses=[ProviderResponse(final_text='Only the April 30 report arrived.')])
        restored = AgentRuntime(config=self.config, store=restored_store, tool_registry=self.tools,
            providers={'gemini': next_provider}, preview_cache=restored_cache)
        rebuilt = await restored._get_live_state(self.session)
        with patch.object(next_provider, 'generate', side_effect=summarize):
            await restored._make_toolspan_block_candidate(next_provider, changed,
                rebuilt.raw_messages, session_id=self.session)
        self.assertEqual(requests[0]['messages'], requests[1]['messages'])
        answer = await restored.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Which report arrived?'))
        self.assertEqual(answer.text, 'Only the April 30 report arrived.')
        assert_file_records(next_provider.requests[0])
        self.assertEqual(await restored_store.read_messages(self.session, [row.db_id for row in original_rows]), original_rows)
        remote.fetch_files.assert_awaited_once()
        self.assertEqual(bot.send_document.await_count, len(expected_deliveries),
            'Reconstruction and compaction never resend files.')

    async def test_malformed_generated_calls_cannot_execute_and_retry_preserves_context(self):
        from tgchatbot.providers.gemini import GeminiProvider
        from tgchatbot.providers.openai_responses import OpenAIResponsesProvider
        for name, factory, invalid_arguments in (
            ('openai', OpenAIResponsesProvider, ['not json', '[]', 'null']),
            ('gemini', GeminiProvider, [[], None, 'not an object']),
        ):
            for arguments in invalid_arguments:
                with self.subTest(provider=name, arguments=arguments):
                    provider = factory(replace(getattr(self.config, name), api_key='mock'))
                    await provider.aclose()
                    requests = []
                    def respond(request):
                        requests.append(json.loads(request.content))
                        if name == 'openai':
                            item = ({'type': 'function_call', 'call_id': 'bad', 'name': 'shell_exec', 'arguments': arguments}
                                if len(requests) == 1 else {'type': 'message', 'role': 'assistant',
                                    'content': [{'type': 'output_text', 'text': 'Please clarify the command.'}]})
                            body = {'status': 'completed', 'output': [item]}
                        else:
                            part = ({'functionCall': {'name': 'shell_exec', 'args': arguments}}
                                if len(requests) == 1 else {'text': 'Please clarify the command.'})
                            body = {'candidates': [{'finishReason': 'STOP', 'content': {'role': 'model', 'parts': [part]}}]}
                        return httpx.Response(200, json=body)
                    provider._client = httpx.AsyncClient(base_url='https://test.invalid/', transport=httpx.MockTransport(respond))
                    try:
                        self.runtime.providers[name] = provider
                        await self.settings(provider=name, model=getattr(self.config, name).model,
                            mode=ChatMode.ASSIST, provider_retry_count=1)
                        result = await self.runtime.run_turn(session_id=self.session, user_display_name='Person',
                            incoming_message=ConversationMessage.user_text('Use the requested command'))
                        self.assertEqual(result.text, 'Please clarify the command.')
                        self.assertEqual(len(requests), 2)
                        self.tools.runner.run.assert_not_awaited()
                        state = await self.assert_parity()
                        self.assertFalse(any(row.message.role == MessageRole.TOOL for row in state.raw_messages))
                    finally:
                        await provider.aclose()

    async def assert_parity(self):
        settings = await self.store.get_or_create_session(self.session, self.config.default_session_settings())
        live = await self.runtime._get_live_state(self.session)
        restored = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools,
            providers=self.runtime.providers, preview_cache=PreviewCache(self.store, max_bytes=0))
        reconstructed = await restored._get_live_state(self.session)
        self.assertEqual(live.raw_messages, reconstructed.raw_messages)
        self.assertEqual(live.blocks, reconstructed.blocks)
        cached_history = self.runtime._build_provider_history(live, settings=settings, provider_name=settings.provider)
        rebuilt_history = restored._build_provider_history(reconstructed, settings=settings, provider_name=settings.provider)
        self.assertEqual(cached_history, rebuilt_history)
        self.assertEqual(await self.preview_cache.materialize_many(self.session, cached_history, vision=True),
            await restored.preview_cache.materialize_many(self.session, rebuilt_history, vision=True))
        return live

    async def test_images_native_history_edit_batch_retirement_and_resets_reconstruct_identically(self):
        await self.settings(tool_history_mode=ToolHistoryMode.NATIVE_SAME_PROVIDER)
        def incoming(text, data='YWJj', revision=None):
            return ConversationMessage(MessageRole.USER, [MessagePart(PartKind.TEXT, text=text),
                MessagePart(PartKind.IMAGE, data_b64=data, mime_type='image/png')], metadata={
                    'source': 'telegram', 'source_chat_id': '100', 'source_message_id': '1',
                    'actor_id': 'telegram:user:1', 'actor_kind': 'user',
                    **({'edited_at': revision} if revision else {})})
        await self.runtime.ingest_user_message(session_id=self.session, incoming_message=incoming('Original'))
        await self.assert_parity()
        native = {'provider': 'openai', 'items': [{'type': 'message', 'role': 'assistant',
            'content': [{'type': 'output_text', 'text': 'A picture.'}]}]}
        await self.runtime.record_assistant_text(session_id=self.session, text='A picture.', metadata={'provider_native': native})
        await self.assert_parity()
        await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=incoming('Corrected picture', 'ZGVm', '2026-09-12T08:00:00Z'))
        state = await self.assert_parity()
        await self.runtime._compact_oldest_images(session_id=self.session, settings=await self.settings(),
            state=state, target_images=0)
        await self.assert_parity()
        self.assertEqual(await self.store.list_preview_refs(self.session), set())
        await self.store.reset_context(self.session)
        self.runtime.invalidate_session(self.session)
        await self.assert_parity()
        await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text('New context'))
        await self.store.reset_full(self.session, self.config.default_session_settings())
        self.runtime.invalidate_session(self.session)
        await self.assert_parity()

    async def test_snapshot_loading_does_not_erase_an_append_committed_during_the_read(self):
        await self.settings()
        await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage.user_text('First'))
        state = await self.runtime._get_live_state(self.session)
        snapshot_taken, release = asyncio.Event(), asyncio.Event()
        original = self.store.load_live_context_versioned
        count = 0
        async def snapshot(session_id):
            nonlocal count
            result = await original(session_id)
            count += 1
            if count == 1:
                snapshot_taken.set()
                await release.wait()
            return result
        with patch.object(self.store, 'load_live_context_versioned', side_effect=snapshot):
            reloading = asyncio.create_task(self.runtime._reload_live_state(state))
            await snapshot_taken.wait()
            await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage.user_text('Committed later'))
            release.set()
            await reloading
        await self.assert_parity()
        self.assertEqual([item.message.parts[0].text for item in state.raw_messages], ['First', 'Committed later'])

    async def test_multi_call_exchange_keeps_interleaved_human_intake_outside_native_pair(self):
        await self.settings(mode=ChatMode.ASSIST, tool_history_mode=ToolHistoryMode.NATIVE_SAME_PROVIDER)
        self.provider.responses = [ProviderResponse(tool_calls=[ToolCall('shell_exec', 'one', {}), ToolCall('shell_exec', 'two', {})],
            continuation_items=[{'type': 'function_call', 'call_id': call, 'name': 'shell_exec', 'arguments': '{}'} for call in ('one', 'two')]),
            ProviderResponse(final_text='Finished')]
        calls = 0
        async def execute(arguments, context):
            nonlocal calls
            calls += 1
            if calls == 1:
                await self.runtime.ingest_user_message(session_id=self.session,
                    incoming_message=ConversationMessage.user_text('Additional human detail'))
            return ToolResult('', 'shell_exec', {'ok': True})
        self.tools.runner.run.side_effect = execute
        await self.runtime.run_turn(session_id=self.session, user_display_name='Person',
            incoming_message=ConversationMessage.user_text('Run both checks'))
        state = await self.assert_parity()
        units = self.runtime._group_raw_compaction_units(state.raw_messages)
        exchange = next(unit for unit in units if any(item.message.metadata.get('tool_batch_id') for item in unit))
        self.assertEqual(sum(item.message.role == MessageRole.TOOL for item in exchange), 4)
        self.assertTrue(any(item.message.parts[0].text == 'Additional human detail' for item in exchange))
        history = self.runtime._build_provider_history(state, settings=await self.settings(), provider_name='openai')
        positions = [index for index, item in enumerate(history) if item.role == MessageRole.TOOL]
        human = next(index for index, item in enumerate(history) if any(part.text == 'Additional human detail' for part in item.parts))
        self.assertLess(max(positions), human)

    async def test_failed_tool_has_durable_unknown_result_and_agent_can_answer(self):
        await self.settings(mode=ChatMode.ASSIST)
        self.provider.responses = [ProviderResponse(tool_calls=[ToolCall('shell_exec', 'call', {})]), ProviderResponse(final_text='The command failed.')]
        self.tools.runner.run.side_effect = OSError('connection lost')
        with self.assertLogs('tgchatbot.core.runtime', level='ERROR'):
            result = await self.runtime.run_turn(session_id=self.session, user_display_name='Person',
                incoming_message=ConversationMessage.user_text('Run the command'))
        self.assertEqual(result.text, 'The command failed.')
        state = await self.assert_parity()
        outcomes = [row.message.metadata['tool_payload']['output'] for row in state.raw_messages
                    if row.message.metadata.get('tool_phase') == 'result']
        self.assertFalse(outcomes[0]['ok'])
        self.assertIn('effects may have occurred', outcomes[0]['outcome'])
        self.tools.runner.run.assert_awaited_once()

    async def test_cancelled_tool_is_recovered_once_without_execution_or_guessing_effects(self):
        await self.settings(mode=ChatMode.ASSIST, tool_history_mode=ToolHistoryMode.NATIVE_SAME_PROVIDER)
        self.provider.responses = [ProviderResponse(tool_calls=[ToolCall('shell_exec', 'call', {})]), ProviderResponse(final_text='I cannot confirm the outcome.')]
        running = asyncio.Event()
        async def execute(*args):
            running.set()
            await asyncio.Event().wait()
        self.tools.runner.run.side_effect = execute
        turn = asyncio.create_task(self.runtime.run_turn(session_id=self.session, user_display_name='Person',
            incoming_message=ConversationMessage.user_text('Run the command')))
        await running.wait()
        turn.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await turn
        self.runtime.invalidate_session(self.session)
        result = await self.runtime.run_turn(session_id=self.session, user_display_name='Person',
            incoming_message=ConversationMessage.user_text('What happened?'))
        self.assertEqual(result.text, 'I cannot confirm the outcome.')
        state = await self.assert_parity()
        recovered = [row for row in state.raw_messages if row.message.metadata.get('recovery') == 'interrupted']
        self.assertEqual(len(recovered), 1)
        self.assertEqual(recovered[0].message.metadata['tool_payload']['output']['outcome'], 'unknown')
        self.tools.runner.run.assert_awaited_once()

    async def test_concurrent_turn_waits_without_recovering_an_active_tool(self):
        await self.settings(mode=ChatMode.ASSIST)
        self.provider.responses = [ProviderResponse(tool_calls=[ToolCall('shell_exec', 'call', {})]),
            ProviderResponse(final_text='First answer'), ProviderResponse(final_text='Second answer')]
        running, release = asyncio.Event(), asyncio.Event()
        async def execute(*args):
            running.set()
            await release.wait()
            return ToolResult('', 'shell_exec', {'ok': True})
        self.tools.runner.run.side_effect = execute
        first = asyncio.create_task(self.runtime.run_turn(session_id=self.session, user_display_name='Person',
            incoming_message=ConversationMessage.user_text('First request')))
        await running.wait()
        second = asyncio.create_task(self.runtime.run_turn(session_id=self.session, user_display_name='Person',
            incoming_message=ConversationMessage.user_text('Second request')))
        await asyncio.sleep(0)
        release.set()
        replies = await asyncio.gather(first, second)
        self.assertEqual([reply.text for reply in replies], ['First answer', 'Second answer'])
        state = await self.assert_parity()
        self.assertFalse(any(row.message.metadata.get('recovery') for row in state.raw_messages))

    async def test_failed_turn_discards_only_temporary_tool_downloads(self):
        await self.settings(mode=ChatMode.ASSIST)
        temporary, original = self.path / 'download.txt', self.path / 'original.txt'
        temporary.write_text('Downloaded bytes')
        original.write_text('Retained input')
        self.tools.runner.run.return_value = ToolResult('', 'shell_exec', {'ok': True}, artifacts=[
            OutboundArtifact(temporary, 'download.txt', temporary=True), OutboundArtifact(original, 'original.txt')])
        self.provider.responses = [ProviderResponse(tool_calls=[ToolCall('shell_exec', 'call', {})]), RuntimeError('model failed')]
        with self.assertRaisesRegex(RuntimeError, 'model failed'):
            await self.runtime.run_turn(session_id=self.session, user_display_name='Person',
                incoming_message=ConversationMessage.user_text('Make a report'))
        self.assertFalse(temporary.exists())
        self.assertEqual(original.read_text(), 'Retained input')

    async def test_same_provider_model_switch_keeps_tools_without_foreign_signatures(self):
        from tgchatbot.providers.gemini import GeminiProvider
        provider = GeminiProvider(self.config.gemini)
        self.addAsyncCleanup(provider.aclose)
        self.runtime.providers['gemini'] = provider
        await self.settings(provider='gemini', model='gemini-3.8-flash',
            tool_history_mode=ToolHistoryMode.NATIVE_SAME_PROVIDER)
        native = {'provider': 'gemini', 'model': 'gemini-3.8-flash', 'items': [{'role': 'model',
            'parts': [{'functionCall': {'name': 'shell_exec', 'id': 'one', 'args': {}},
                       'thoughtSignature': 'model-specific-signature'}]}]}
        await self.runtime.record_tool_observation(session_id=self.session, name='shell_exec', phase='call',
            payload={'call_id': 'one', 'arguments': {}}, provider_name='gemini', metadata_update={
                'provider_native': native, 'tool_model': 'gemini-3.8-flash', 'tool_batch_id': 'fixture'})
        await self.runtime.record_tool_observation(session_id=self.session, name='shell_exec', phase='result',
            payload={'call_id': 'one', 'output': {'ok': True, 'stdout': 'confirmed result'}},
            provider_name='gemini', metadata_update={'tool_model': 'gemini-3.8-flash', 'tool_batch_id': 'fixture'})
        state = await self.assert_parity()
        unchanged = self.runtime._build_provider_history(state, settings=await self.settings(), provider_name='gemini')
        self.assertEqual(unchanged[0].metadata['provider_native'], native)
        changed = await self.settings(model='gemini-future-flash')
        history = self.runtime._build_provider_history(state, settings=changed, provider_name='gemini')
        self.assertTrue(all('provider_native' not in item.metadata for item in history))
        wire = [item for message in history for item in provider._message_to_contents(message)]
        self.assertEqual(wire[0]['parts'][0]['thoughtSignature'], 'skip_thought_signature_validator')
        self.assertEqual(wire[1]['parts'][0]['functionResponse']['response']['result']['stdout'], 'confirmed result')
        await self.assert_parity()

    async def test_source_edit_during_compaction_cannot_commit_an_old_summary(self):
        from tgchatbot.storage.postgres_store import StaleScopeError
        settings = await self.settings(min_raw_messages_reserve=1)
        metadata = {'source': 'telegram', 'source_chat_id': '100', 'source_message_id': '1'}
        original = await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text('The appointment is Monday.', metadata=metadata))
        for text in ('Remember the appointment.', 'What should I prepare?'):
            await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage.user_text(text))
        candidate = {'scope': 'Appointment planning', 'interaction_mode': 'chat_or_sharing',
            'participants': [], 'topics': ['appointment'], 'user_profile': [],
            'user_intent_or_shared_context': ['The appointment is Monday.'], 'why_it_mattered': [],
            'interaction_timeline': ['An appointment was scheduled.'], 'results_or_takeaways': [],
            'decisions': [], 'open_loops': [], 'artifacts': [], 'uncertainties': []}
        started, release = asyncio.Event(), asyncio.Event()
        async def summarize(**kwargs):
            started.set()
            await release.wait()
            return ProviderResponse(final_text=json.dumps(candidate))
        state = await self.runtime._get_live_state(self.session)
        with patch.object(self.provider, 'generate', side_effect=summarize):
            compacting = asyncio.create_task(self.runtime._compact_old_context(session_id=self.session,
                settings=settings, provider=self.provider, state=state, pressure=True))
            await started.wait()
            await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage.user_text(
                'Correction: the appointment is Tuesday.', metadata={**metadata, 'edited_at': '2026-09-12T10:00:00Z'}))
            release.set()
            with self.assertRaises(StaleScopeError):
                await compacting
        self.assertEqual(await self.store.list_memory_blocks(self.session), [])
        restored = await self.assert_parity()
        updated = next(item for item in restored.raw_messages if item.db_id == original.db_id)
        self.assertEqual(updated.message.parts[0].text, 'Correction: the appointment is Tuesday.')
        self.assertEqual(updated.message.metadata['source_revision'], 2)

    async def test_actual_telegram_delivery_provenance_and_native_model_reconstruct_identically(self):
        from datetime import datetime, timezone
        from tgchatbot.domain.models import TurnResult
        from tgchatbot.transports.telegram_adapter import TelegramBotApp
        settings = await self.settings(tool_history_mode=ToolHistoryMode.NATIVE_SAME_PROVIDER)
        app = TelegramBotApp.__new__(TelegramBotApp)
        app.config, app.runtime, app.store = self.config, self.runtime, self.store
        chat = SimpleNamespace(id=100)
        bot = SimpleNamespace(id=999, full_name='Fixture Bot', is_bot=True)
        source = SimpleNamespace(chat=chat, message_id=10, get_bot=lambda: bot)
        delivered = [SimpleNamespace(message_id=message_id, chat=chat, from_user=bot,
            date=datetime(2026, 9, 12, tzinfo=timezone.utc)) for message_id in (1001, 1002)]
        native = [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Delivered answer'}]}]
        result = TurnResult('Delivered answer', scope=await self.store.get_scope(self.session),
            provider_name='openai', provider_model=settings.model, provider_history_items=native)
        await app._record_delivered_assistant_text(session_id=self.session, result=result,
            source_message=source, delivered_messages=delivered)
        state = await self.assert_parity()
        message = state.raw_messages[0].message
        self.assertEqual(message.metadata['actor_id'], 'telegram:user:999')
        self.assertEqual(message.metadata['source_message_id'], '1001')
        self.assertEqual(message.metadata['provider_native']['model'], settings.model)
        self.assertEqual(message.metadata['telegram_message_aliases'], ['1002'])

    async def test_mixed_visual_and_text_tools_replay_as_one_complete_portable_exchange(self):
        from tgchatbot.providers.chat_completions import ChatCompletionsProvider
        from tgchatbot.providers.gemini import GeminiProvider
        settings = await self.settings(tool_history_mode=ToolHistoryMode.TRANSLATED)
        for phase, name, call_id, evidence in [
            ('call', 'sticker_query', 'visual', True), ('call', 'memory_search', 'text', False),
            ('result', 'sticker_query', 'visual', True), ('result', 'memory_search', 'text', False),
        ]:
            payload = {'call_id': call_id, **({'arguments': {}} if phase == 'call' else {'output': {'ok': True}})}
            parts = [MessagePart(PartKind.IMAGE, data_b64='YWJj', mime_type='image/png', origin='sticker_candidate:one')] if evidence and phase == 'result' else []
            await self.runtime.record_tool_observation(session_id=self.session, name=name, phase=phase,
                payload=payload, provider_name='gemini', evidence_parts=parts, metadata_update={
                    'tool_batch_id': 'mixed', **({'tool_evidence': True} if evidence else {})})
        state = await self.assert_parity()
        history = self.runtime._build_provider_history(state, settings=settings, provider_name='openai')
        self.assertTrue(all(item.role == MessageRole.TOOL for item in history))
        # Exercise protocol framing without constructing any network client.
        completions = object.__new__(ChatCompletionsProvider)
        completions.name = 'openai'
        wire = completions._messages_for_request(history)
        self.assertEqual([item['role'] for item in wire], ['assistant', 'tool', 'tool'])
        self.assertEqual([item['id'] for item in wire[0]['tool_calls']], ['visual', 'text'])
        self.assertEqual([item['tool_call_id'] for item in wire[1:]], ['visual', 'text'])
        gemini = GeminiProvider(self.config.gemini)
        self.addAsyncCleanup(gemini.aclose)
        materialized = await self.preview_cache.materialize_many(self.session, history, vision=True)
        contents = gemini._prepare_contents_for_request([item for message in materialized for item in gemini._message_to_contents(message)])
        self.assertEqual([part['functionCall']['id'] for part in contents[0]['parts']], ['visual', 'text'])
        self.assertEqual([part['functionResponse']['id'] for content in contents[1:] for part in content['parts']], ['visual', 'text'])
        self.assertEqual(len(contents[1]['parts'][0]['functionResponse']['parts']), 1)

    async def test_independent_writer_is_seen_without_cancelling_the_turn_scope(self):
        await self.settings()
        await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text('Before external import'))
        await self.runtime._get_live_state(self.session)
        scope = await self.store.get_scope(self.session)
        other = await self.new_store()
        await other.append_message(self.session, ConversationMessage.user_text('Imported by another process'))
        await self.store.assert_scope(self.session, scope)
        state = await self.assert_parity()
        self.assertEqual([row.message.parts[0].text for row in state.raw_messages],
                         ['Before external import', 'Imported by another process'])
        await other.reset_context(self.session)
        # No runtime invalidation callback came from this independent writer.
        state = await self.assert_parity()
        self.assertEqual(state.raw_messages, [])

    async def test_append_gap_reloads_external_rows_instead_of_guessing_missing_context(self):
        await self.settings()
        await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text('First'))
        other = await self.new_store()
        append = self.runtime._append_stored
        async def interleaved(session_id, message, **kwargs):
            await other.append_message(session_id, ConversationMessage.user_text('Concurrent import'))
            return await append(session_id, message, **kwargs)
        with patch.object(self.runtime, '_append_stored', side_effect=interleaved):
            await self.runtime.ingest_user_message(session_id=self.session,
                incoming_message=ConversationMessage.user_text('New Telegram message'))
        state = await self.assert_parity()
        self.assertEqual([row.message.parts[0].text for row in state.raw_messages],
                         ['First', 'Concurrent import', 'New Telegram message'])

    async def test_source_binding_by_storage_invalidates_cached_provenance(self):
        await self.settings()
        saved = await self.runtime.record_assistant_text(session_id=self.session, text='Delivered text')
        await self.runtime._get_live_state(self.session)
        await self.store.bind_message_source(self.session, saved.db_id, source='telegram',
            source_chat_id='100', source_message_ids=['991', '992'], actor_id='telegram:user:999',
            actor_kind='bot', actor_name='Observed Bot')
        state = await self.assert_parity()
        metadata = state.raw_messages[0].message.metadata
        self.assertEqual(metadata['actor_id'], 'telegram:user:999')
        self.assertEqual(metadata['source_message_id'], '991')

    async def test_text_only_provider_does_not_retire_images_before_vision_is_used(self):
        from tgchatbot.providers.base import ProviderCapabilities
        text_provider = ScriptedProvider('gemini', [ProviderResponse(final_text='Text-only response')])
        text_provider.capabilities = ProviderCapabilities(multimodal_input=False)
        self.runtime.providers['gemini'] = text_provider
        await self.settings(provider='gemini', model='text-only-fixture', max_input_images=1,
            compact_target_images=1, compact_trigger_tokens=100000)
        for text, data in [('first', 'YWFh'), ('second', 'YmJi')]:
            await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage(
                MessageRole.USER, [MessagePart(PartKind.TEXT, text=text),
                    MessagePart(PartKind.IMAGE, data_b64=data, mime_type='image/png')]))
        await self.runtime.run_turn(session_id=self.session, user_display_name='Person',
            incoming_message=ConversationMessage.user_text('Describe the context'))
        self.assertFalse(any(part.data_b64 for message in text_provider.requests[0]['messages'] for part in message.parts))
        state = await self.assert_parity()
        self.assertEqual(state.estimated_images, 2)
        await self.settings(provider='openai', model=self.config.openai.model)
        self.provider.responses = [ProviderResponse(final_text='Now I can see it')]
        await self.runtime.run_turn(session_id=self.session, user_display_name='Person',
            incoming_message=ConversationMessage.user_text('Use vision now'))
        pixels = [part.data_b64 for message in self.provider.requests[0]['messages'] for part in message.parts if part.kind == PartKind.IMAGE]
        self.assertEqual(pixels, ['YmJi'])
        await self.assert_parity()
