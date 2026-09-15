"""Business counterexamples found while auditing conversation ownership."""
from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import ChatMode, ConversationMessage, PartKind, ProviderResponse, ToolCall, ToolResult
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.storage.previews import PreviewCache
from tgchatbot.tools.read_doc import ReadDocTool


class RuntimeAuditWorkflows(BusinessTestCase):
    async def test_compaction_sees_the_text_and_source_of_a_read_document(self):
        await self.compact_read_document(parallel=False)

    async def test_parallel_document_reads_keep_each_observation_through_compaction_and_restart(self):
        await self.compact_read_document(parallel=True)

    async def test_document_compaction_keeps_text_without_restoring_retired_page_images(self):
        await self.compact_read_document(parallel=False, retire_images=True)

    async def compact_read_document(self, *, parallel, retire_images=False):
        settings = await self.settings(mode=ChatMode.ASSIST, compact_trigger_tokens=100000)
        contents = 'Door code: 1591.\nThe side entrance opens at noon.'
        parts = [{'kind': 'text', 'text': contents}]
        if retire_images:
            self.provider.capabilities = replace(self.provider.capabilities, multimodal_tool_results=True)
            parts.append({'kind': 'image', 'mime_type': 'image/png', 'data_b64': 'ZmFrZQ==',
                'filename': 'arrival-page-1.png'})
        remote = SimpleNamespace(enabled=True, inspect_file=AsyncMock(return_value={
            'ok': True, 'parts': parts, 'start': 1, 'end': 1,
            'total_pages' if retire_images else 'total_lines': 1}))
        reader = ReadDocTool(self.config, remote)
        self.tools.spec = reader.spec
        self.tools.list_tools.return_value = [reader.spec]
        arguments = {'path': 'arrival.pdf' if retire_images else 'arrival.txt',
            'format': 'pdf' if retire_images else 'text'}
        calls = [ToolCall('read_doc', call_id, arguments)
            for call_id in (('arrival', 'verification') if parallel else ('arrival',))]
        self.provider.responses = [ProviderResponse(tool_calls=calls,
            continuation_items=[{'type': 'function_call', 'call_id': call.call_id, 'name': 'read_doc',
                'arguments': json.dumps(arguments)} for call in calls]), ProviderResponse(final_text='I have read the arrival note.')]
        await self.runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Read the arrival note so we can refer to it later.'))
        state = await self.runtime._get_live_state(self.session)
        originals = await self.store.read_messages(self.session, [row.db_id for row in state.raw_messages])
        if retire_images:
            self.assertEqual(await self.runtime._compact_oldest_images(session_id=self.session,
                settings=settings, state=state, target_images=0), 1)
            self.assertTrue(any(part.kind == PartKind.IMAGE for row in originals for part in row.message.parts))
        requests = []

        async def summarize(**request):
            requests.append(request)
            evidence = '\n'.join(part.text or '' for message in request['messages'] for part in message.parts)
            data = {key: [] for key in request['response_schema']['properties']}
            data.update(scope='Reading the arrival instructions.', interaction_mode='task_execution',
                results_or_takeaways=['Door code 1591' if 'Door code: 1591' in evidence else 'No document contents supplied'])
            return ProviderResponse(final_text=json.dumps(data))

        with patch.object(self.provider, 'generate', side_effect=summarize):
            candidate = await self.runtime._make_toolspan_block_candidate(self.provider, settings,
                state.raw_messages, session_id=self.session)
        self.assertEqual(candidate['data']['results_or_takeaways'], ['Door code 1591'])
        evidence = '\n'.join(part.text or '' for message in requests[0]['messages'] for part in message.parts)
        self.assertIn(arguments['path'], evidence)
        self.assertEqual(evidence.count(contents), len(calls), 'Each parsed observation is supplied exactly once.')
        if retire_images:
            self.assertIn('[Image compacted]', evidence)
            self.assertNotIn('ZmFrZQ==', evidence)
            self.assertFalse(any(part.kind == PartKind.IMAGE for message in requests[0]['messages'] for part in message.parts))
        await self.store.close()
        restored_store = await self.new_store()
        restored_cache = PreviewCache(restored_store, max_bytes=0)
        self.addCleanup(restored_cache.close)
        restored = AgentRuntime(config=self.config, store=restored_store, tool_registry=self.tools,
            providers={'openai': self.provider}, preview_cache=restored_cache)
        rebuilt = await restored._get_live_state(self.session)
        with patch.object(self.provider, 'generate', side_effect=summarize):
            await restored._make_toolspan_block_candidate(self.provider, settings, rebuilt.raw_messages,
                session_id=self.session)
        self.assertEqual(requests[1]['messages'], requests[0]['messages'])
        self.assertEqual(await restored_store.read_messages(self.session, [row.db_id for row in originals]), originals)
        self.assertEqual(remote.inspect_file.await_count, len(calls), 'Compaction never rereads the remote files.')

    async def test_summary_prefix_survives_the_first_raw_message(self):
        await self.summary_prefix(block_count=2, block_tokens=1120)

    async def test_summary_prefix_survives_growing_raw_history_and_database_restart(self):
        await self.summary_prefix(block_count=5, block_tokens=700)

    async def summary_prefix(self, *, block_count, block_tokens):
        # Small fixture budgets exercise both the first raw message and later
        # pressure transitions without a costly corpus or model request.
        settings = await self.settings(provider='gemini', model='fixture-flash',
            compact_target_tokens=10000, compact_trigger_tokens=100000)
        sources = []
        canonical_sources = []
        for number in range(block_count):
            source = await self.store.append_message(self.session,
                ConversationMessage.user_text(f'Original discussion {number}.'))
            sources.append(source)
            canonical_sources.extend(await self.store.read_messages(self.session, [source.db_id]))
            await self.store.create_memory_block(self.session, source_message_ids=[source.db_id],
                summary_text=f'Earlier discussion {number}. ' + 'A confirmed fact and its owner. ' * 85,
                estimated_tokens=block_tokens)
        version = await self.store.get_compaction_version(self.session)
        state = await self.runtime._get_live_state(self.session)
        self.assertEqual(state.raw_messages, [])
        wire = []

        def respond(request):
            wire.append(json.loads(request.content))
            return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP',
                'content': {'role': 'model', 'parts': [{'text': 'Acknowledged.'}]}}]})

        provider = GeminiProvider(replace(self.config.gemini, api_key='synthetic-key'))
        await provider.aclose()
        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(provider.aclose)
        self.runtime.providers = {'gemini': provider}
        # Serialize the prepared, summary-only context through the real adapter.
        # The next requests come from ordinary intake and runtime execution.
        history = self.runtime._build_provider_history(state, settings=settings, provider_name='gemini')
        await provider.generate(settings=settings, messages=history, instructions='', tools=[])
        for index, text in enumerate(('Remember our earlier discussions?', 'Continuing daily conversation. ' * 370)):
            trigger = await self.runtime.ingest_user_message(session_id=self.session,
                incoming_message=ConversationMessage.user_text(text))
            response = await self.runtime.run_turn_from_stored(session_id=self.session,
                user_display_name='Participant', trigger_message_id=trigger.db_id)
            if index == 0:
                await self.runtime.record_assistant_text(session_id=self.session, text=response.text)
        self.assertEqual(await self.store.get_compaction_version(self.session), version)
        for earlier, later in zip(wire, wire[1:]):
            self.assertEqual(later['contents'][:len(earlier['contents'])], earlier['contents'],
                'An ordinary append must preserve the selected sealed summaries and earlier request prefix.')
        original_rows = await self.store.read_messages(self.session, [source.db_id for source in sources])
        self.assertEqual(original_rows, canonical_sources)
        await self.store.close()
        restored_store = await self.new_store()
        restored_cache = PreviewCache(restored_store, max_bytes=0)
        self.addCleanup(restored_cache.close)
        restored = AgentRuntime(config=self.config, store=restored_store, tool_registry=self.tools,
            providers={'gemini': provider}, preview_cache=restored_cache)
        await restored.run_turn_from_stored(session_id=self.session,
            user_display_name='Participant', trigger_message_id=trigger.db_id)
        self.assertEqual(wire[-1]['contents'], wire[-2]['contents'],
            'The database alone must reconstruct the same summary selection and request.')
        self.assertEqual(await restored_store.get_compaction_version(self.session), version)

    async def test_compaction_keeps_confirmed_result_when_assistant_names_the_tool(self):
        await self.compact_named_tool(parallel=False)

    async def test_parallel_compaction_keeps_the_ordinary_tool_name_utterance(self):
        await self.compact_named_tool(parallel=True)

    async def compact_named_tool(self, *, parallel):
        settings = await self.settings(mode=ChatMode.ASSIST, compact_trigger_tokens=100000)
        calls = [ToolCall('shell_exec', call_id, {'command': 'calculate'})
            for call_id in (('calculation', 'verification') if parallel else ('calculation',))]
        self.provider.responses = [ProviderResponse(final_text='shell_exec',
            tool_calls=calls,
            continuation_items=[{'type': 'message', 'role': 'assistant',
                'content': [{'type': 'output_text', 'text': 'shell_exec'}]},
                *[{'type': 'function_call', 'call_id': call.call_id, 'name': 'shell_exec',
                    'arguments': '{"command":"calculate"}'} for call in calls]]), ProviderResponse(final_text='The calculation finished.')]
        self.tools.runner.run.return_value = ToolResult('', 'shell_exec',
            {'ok': True, 'exit_code': 0, 'stdout': 'CONFIRMED_CODE=1591'})
        await self.runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Name the tool, then calculate the confirmation code.'),
            emit=AsyncMock())
        state = await self.runtime._get_live_state(self.session)
        requests = []

        async def summarize(**request):
            requests.append(request)
            evidence = '\n'.join(part.text or '' for message in request['messages'] for part in message.parts)
            fields = request['response_schema']['properties']
            data = {key: [] for key in fields}
            data.update(scope='Confirmation code calculation', interaction_mode='task_execution',
                results_or_takeaways=['Confirmed code 1591' if 'CONFIRMED_CODE=1591' in evidence else 'No result supplied'])
            return ProviderResponse(final_text=json.dumps(data))

        with patch.object(self.provider, 'generate', side_effect=summarize):
            candidate = await self.runtime._make_toolspan_block_candidate(self.provider, settings,
                state.raw_messages, session_id=self.session)
        self.assertEqual(candidate['data']['results_or_takeaways'], ['Confirmed code 1591'])
        evidence = '\n'.join(part.text or '' for message in requests[0]['messages'] for part in message.parts)
        self.assertIn('shell_exec', evidence.splitlines(), 'The ordinary utterance remains distinct from the tool record.')
        self.assertIn('CONFIRMED_CODE=1591', evidence)
        self.assertEqual(self.tools.runner.run.await_count, len(calls))
        originals = await self.store.read_messages(self.session, [row.db_id for row in state.raw_messages])
        self.assertEqual(len(originals), len(state.raw_messages), 'Compaction input preparation cannot consume originals.')
