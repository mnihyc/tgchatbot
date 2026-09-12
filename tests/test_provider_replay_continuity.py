"""Completed model exchanges remain usable after another turn and DB-only restart."""
from __future__ import annotations

import base64
import copy
import io
import json
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from PIL import Image

from tests.business_helpers import BusinessTestCase
from tgchatbot.config import ChatCompletionsConfig
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import (ChatMode, ConversationMessage, MessagePart, PartKind,
    ToolHistoryMode, ToolResult)
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.providers.chat_completions import ChatCompletionsProvider
from tgchatbot.providers.openai_responses import OpenAIResponsesProvider
from tgchatbot.storage.previews import PreviewCache
from tgchatbot.transports.telegram_adapter import TelegramBotApp


class ProviderReplayContinuityTests(BusinessTestCase):
    async def test_openai_and_compatible_plain_tool_batches_preserve_reasoning_and_results(self):
        for name in ('openai', 'compatible'):
            with self.subTest(provider=name):
                await self.store.reset_context(self.session)
                self.runtime.invalidate_session(self.session)
                wire = []
                if name == 'openai':
                    provider = OpenAIResponsesProvider(replace(self.config.openai, api_key='synthetic-key'))
                    batch = [
                        {'type': 'reasoning', 'id': 'reasoning', 'encrypted_content': 'opaque-reasoning', 'summary': []},
                        {'type': 'message', 'id': 'introduction', 'role': 'assistant',
                            'content': [{'type': 'output_text', 'text': 'Checking both notes.'}]},
                        *[{'type': 'function_call', 'id': 'item-' + call_id, 'call_id': call_id,
                           'name': 'shell_exec', 'arguments': '{}'} for call_id in ('one', 'two')],
                    ]
                    final = [{'type': 'message', 'id': 'answer', 'role': 'assistant',
                        'content': [{'type': 'output_text', 'text': 'The notes agree.'}]}]
                    scripts = [{'status': 'completed', 'output': items} for items in (batch, final, final, final)]
                    history_key = 'input'
                else:
                    provider = ChatCompletionsProvider(ChatCompletionsConfig(name=name, api_key='synthetic-key',
                        base_url='https://fixture.invalid/v1', model='fixture'))
                    self.config = replace(self.config, chat_completions=(provider.config,))
                    self.runtime.config = self.config
                    batch = [{'role': 'assistant', 'content': 'Checking both notes.',
                        'reasoning_content': 'Opaque provider continuation.',
                        'reasoning_details': [{'type': 'reasoning.encrypted', 'data': 'opaque-reasoning'}],
                        'tool_calls': [{'id': call_id, 'type': 'function',
                            'function': {'name': 'shell_exec', 'arguments': '{}'}} for call_id in ('one', 'two')]}]
                    final = [{'role': 'assistant', 'content': 'The notes agree.',
                        'reasoning_content': 'Final provider continuation.'}]
                    scripts = [{'choices': [{'finish_reason': 'tool_calls' if items is batch else 'stop',
                        'message': items[0]}]} for items in (batch, final, final, final)]
                    history_key = 'messages'

                await provider.aclose()
                def respond(request):
                    wire.append(json.loads(request.content))
                    return httpx.Response(200, json=copy.deepcopy(scripts.pop(0)))
                provider._client = httpx.AsyncClient(base_url='https://fixture.invalid/v1/',
                    transport=httpx.MockTransport(respond))
                self.addAsyncCleanup(provider.aclose)
                settings = await self.settings(provider=name, model=provider.config.model, mode=ChatMode.ASSIST,
                    tool_history_mode=ToolHistoryMode.TRANSLATED, max_interaction_rounds=3,
                    compact_trigger_tokens=100000)
                self.runtime.providers = {name: provider}
                self.tools.runner.run.reset_mock()
                self.tools.runner.run.return_value = ToolResult('', 'shell_exec',
                    {'checked': 'Both notes were inspected.', 'ok': True})
                result = await self.runtime.run_turn(session_id=self.session, user_display_name='Participant',
                    incoming_message=ConversationMessage.user_text('Check both notes.'), emit=AsyncMock())
                await self.runtime.record_assistant_text(session_id=self.session, text=result.text, metadata={
                    'provider_native': {'provider': name, 'model': settings.model, 'items': result.provider_history_items}})
                followup = await self.runtime.ingest_user_message(session_id=self.session,
                    incoming_message=ConversationMessage.user_text('Do those results still agree?'))
                await self.runtime.run_turn_from_stored(session_id=self.session,
                    user_display_name='Participant', trigger_message_id=followup.db_id)
                self.assertEqual(wire[2][history_key][:len(wire[1][history_key])], wire[1][history_key])
                self.assertEqual(self.tools.runner.run.await_count, 2)
                for item in [*batch, *final]:
                    self.assertEqual(wire[2][history_key].count(item), 1)
                restored = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools,
                    providers={name: provider}, preview_cache=PreviewCache(self.store, max_bytes=0))
                self.addCleanup(restored.preview_cache.close)
                await restored.run_turn_from_stored(session_id=self.session, user_display_name='Participant',
                    trigger_message_id=followup.db_id)
                self.assertEqual(wire[3][history_key], wire[2][history_key])
                self.assertEqual(self.tools.runner.run.await_count, 2)

    async def test_signed_parallel_and_sequential_calls_and_delivered_reply_keep_request_prefix(self):
        settings = await self.settings(provider='gemini', model='gemini-3.8-flash',
            mode=ChatMode.ASSIST, tool_history_mode=ToolHistoryMode.TRANSLATED,
            max_interaction_rounds=4, compact_trigger_tokens=100000, max_input_images=10)
        wire = []
        parallel = {'role': 'model', 'parts': [
            {'text': 'I will inspect the picture and check the note.'},
            {'functionCall': {'name': 'shell_exec', 'id': 'picture', 'args': {'task': 'picture'}},
                'thoughtSignature': 'original-parallel-signature'},
            {'functionCall': {'name': 'shell_exec', 'id': 'note', 'args': {'task': 'note'}}},
        ]}
        sequential = {'role': 'model', 'parts': [
            {'functionCall': {'name': 'shell_exec', 'id': 'confirm', 'args': {'task': 'confirm'}},
                'thoughtSignature': 'original-sequential-signature'},
        ]}
        final = {'role': 'model', 'parts': [
            {'text': 'Both checks agree.', 'thoughtSignature': 'original-final-signature'}]}
        scripts = [parallel, sequential, final,
            {'role': 'model', 'parts': [{'text': 'The earlier checks still agree.'}]},
            {'role': 'model', 'parts': [{'text': 'The earlier checks still agree.'}]}]

        def respond(request):
            wire.append(json.loads(request.content))
            return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP',
                'content': copy.deepcopy(scripts.pop(0))}]})

        provider = GeminiProvider(replace(self.config.gemini, api_key='synthetic-key'))
        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(provider.aclose)
        self.runtime.providers = {'gemini': provider}
        image = io.BytesIO()
        Image.new('RGB', (8, 8), 'red').save(image, format='PNG')
        pixels = base64.b64encode(image.getvalue()).decode()

        async def execute(arguments, context):
            task = arguments['task']
            return ToolResult('', 'shell_exec', {'ok': True, 'checked': task},
                evidence_parts=[MessagePart(PartKind.IMAGE, mime_type='image/png',
                    data_b64=pixels, origin='file_read')] if task == 'picture' else [])

        self.tools.runner.run.side_effect = execute
        result = await self.runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Inspect the picture, note and confirmation.'),
            emit=AsyncMock())
        self.assertEqual(result.text, 'Both checks agree.')
        self.assertEqual(self.tools.runner.run.await_count, 3)
        self.assertEqual(wire[2]['contents'][:len(wire[1]['contents'])], wire[1]['contents'])

        # Use the transport's actual delivery commit, including bot identity and
        # the provider/model snapshot. Generated but undelivered text is absent.
        app = TelegramBotApp.__new__(TelegramBotApp)
        app.config, app.runtime, app.store = self.config, self.runtime, self.store
        chat = SimpleNamespace(id=100)
        bot = SimpleNamespace(id=999, full_name='Fixture Bot', is_bot=True)
        source = SimpleNamespace(chat=chat, message_id=10, get_bot=lambda: bot)
        delivered = SimpleNamespace(message_id=1001, chat=chat, from_user=bot,
            date=datetime(2026, 9, 13, tzinfo=timezone.utc))
        await app._record_delivered_assistant_text(session_id=self.session, result=result,
            source_message=source, delivered_messages=[delivered])
        followup = await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text('Do those earlier checks agree?'))
        await self.runtime.run_turn_from_stored(session_id=self.session,
            user_display_name='Participant', trigger_message_id=followup.db_id)
        self.assertEqual(wire[3]['contents'][:len(wire[2]['contents'])], wire[2]['contents'])
        self.assertEqual([content for content in wire[3]['contents'] if content['role'] == 'model'],
            [parallel, sequential, final])
        calls = [part['functionCall']['id'] for content in wire[3]['contents']
            for part in content['parts'] if 'functionCall' in part]
        outcomes = [part['functionResponse']['id'] for content in wire[3]['contents']
            for part in content['parts'] if 'functionResponse' in part]
        self.assertEqual(calls, ['picture', 'note', 'confirm'])
        self.assertEqual(outcomes, calls)

        await self.store.close()
        restored_store = await self.new_store()
        restored_cache = PreviewCache(restored_store, max_bytes=0)
        self.addCleanup(restored_cache.close)
        restored = AgentRuntime(config=self.config, store=restored_store, tool_registry=self.tools,
            providers={'gemini': provider}, preview_cache=restored_cache)
        await restored.run_turn_from_stored(session_id=self.session, user_display_name='Participant',
            trigger_message_id=followup.db_id)
        self.assertEqual(wire[4]['contents'], wire[3]['contents'])
        self.assertEqual(self.tools.runner.run.await_count, 3)
        self.assertEqual(settings.tool_history_mode, ToolHistoryMode.TRANSLATED)

        # Switching changes compatibility, not the authoritative snapshot. The
        # existing portable visual pair excludes foreign signatures; switching
        # back must recover the original grouped output without running tools.
        state = await restored._get_live_state(self.session)
        for provider_name, model in (('gemini', 'gemini-future-flash'), ('openai', 'other-model')):
            switched = replace(settings, provider=provider_name, model=model)
            history = restored._build_provider_history(state, settings=switched, provider_name=provider_name)
            self.assertFalse(any(message.metadata.get('provider_native') for message in history))
            if provider_name == 'gemini':
                changed = provider._prepare_contents_for_request([
                    item for message in history for item in provider._message_to_contents(message)])
                typed_calls = [part['functionCall']['id'] for content in changed
                    for part in content['parts'] if 'functionCall' in part]
                self.assertEqual(typed_calls, ['picture', 'note'])
                self.assertNotIn('original-parallel-signature', json.dumps(changed))
        history = restored._build_provider_history(state, settings=settings, provider_name='gemini')
        history = await restored_cache.materialize_many(self.session, history, vision=True)
        recovered = provider._prepare_contents_for_request([
            item for message in history for item in provider._message_to_contents(message)])
        self.assertEqual(recovered, wire[4]['contents'])
