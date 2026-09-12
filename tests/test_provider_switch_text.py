"""Changing providers keeps ordinary assistant text around inspected tool results."""
from __future__ import annotations

import base64
import copy
import io
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx
from PIL import Image

from tests.business_helpers import BusinessTestCase
from tgchatbot.config import ChatCompletionsConfig
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import (ChatMode, ConversationMessage, MessagePart, PartKind,
    ToolHistoryMode, ToolResult)
from tgchatbot.providers.chat_completions import ChatCompletionsProvider
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.providers.openai_responses import OpenAIResponsesProvider
from tgchatbot.storage.previews import PreviewCache


class ProviderSwitchTextWorkflows(BusinessTestCase):
    @staticmethod
    def visible_texts(payload, provider_name):
        if provider_name == 'gemini':
            return [part['text'] for item in payload['contents'] if item['role'] == 'model'
                for part in item.get('parts', []) if 'text' in part and not part.get('thought')]
        if provider_name == 'openai':
            return [part['text'] for item in payload['input'] if item.get('role') == 'assistant'
                for part in item.get('content', []) if part.get('type') == 'output_text']
        return [item['content'] for item in payload['messages']
            if item['role'] == 'assistant' and isinstance(item.get('content'), str)]

    @staticmethod
    def call_result_ids(payload, provider_name):
        if provider_name == 'gemini':
            parts = [part for item in payload['contents'] for part in item.get('parts', [])]
            return ([part['functionCall']['id'] for part in parts if 'functionCall' in part],
                    [part['functionResponse']['id'] for part in parts if 'functionResponse' in part])
        if provider_name == 'openai':
            return ([item['call_id'] for item in payload['input'] if item.get('type') == 'function_call'],
                    [item['call_id'] for item in payload['input'] if item.get('type') == 'function_call_output'])
        return ([call['id'] for item in payload['messages'] for call in item.get('tool_calls', [])],
                [item['tool_call_id'] for item in payload['messages'] if item['role'] == 'tool'])

    async def test_plain_translated_tool_summary_does_not_erase_matching_assistant_words(self):
        await self.settings(provider='gemini', model='gemini-3.8-flash', mode=ChatMode.ASSIST,
            tool_history_mode=ToolHistoryMode.TRANSLATED, compact_trigger_tokens=100000)
        scripts = [[{'text': 'shell_exec'},
            {'functionCall': {'name': 'shell_exec', 'id': 'check', 'args': {}},
                'thoughtSignature': 'source-signature'}], [{'text': 'The check completed.'}]]
        source = GeminiProvider(replace(self.config.gemini, api_key='synthetic-key'))
        source._client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request:
            httpx.Response(200, json={'candidates': [{'finishReason': 'STOP',
                'content': {'role': 'model', 'parts': scripts.pop(0)}}]})))
        self.addAsyncCleanup(source.aclose)
        runtime = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools,
            providers={'gemini': source}, preview_cache=self.preview_cache)
        await runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Name the tool and perform the check.'), emit=AsyncMock())
        wire = []
        target = OpenAIResponsesProvider(replace(self.config.openai, api_key='synthetic-key'))
        await target.aclose()
        def respond(request):
            wire.append(json.loads(request.content))
            return httpx.Response(200, json={'status': 'completed', 'output': [{'type': 'message',
                'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'The stated tool was shell_exec.'}]}]})
        target._client = httpx.AsyncClient(base_url='https://fixture.invalid/v1/', transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(target.aclose)
        await self.settings(provider='openai', model=target.config.model)
        switched = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools,
            providers={'openai': target}, preview_cache=self.preview_cache)
        await switched.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('What tool did you name?'))
        texts = self.visible_texts(wire[0], 'openai')
        self.assertEqual(texts.count('shell_exec'), 1, 'An ordinary utterance remains its own assistant text.')
        self.assertTrue(any(text != 'shell_exec' and 'shell_exec' in text for text in texts),
            'Its occurrence inside a translated tool record is not the ordinary utterance.')
        self.assertNotIn('source-signature', json.dumps(wire[0]))
        self.tools.runner.run.assert_awaited_once()

    async def test_openai_and_compatible_source_text_parts_keep_repetition_without_reasoning(self):
        repeated = '  同一句话也可能说两次。\n'
        final_text = '已核对。\n'
        for origin in ('openai', 'compatible'):
            with self.subTest(origin=origin):
                await self.store.reset_context(self.session)
                if origin == 'openai':
                    provider = OpenAIResponsesProvider(replace(self.config.openai, api_key='synthetic-key'))
                    batch = [{'type': 'reasoning', 'id': 'reasoning', 'summary': [],
                        'encrypted_content': 'PRIVATE_ENCRYPTED_REASONING'},
                        {'type': 'message', 'role': 'assistant', 'content': [
                            {'type': 'output_text', 'text': repeated}, {'type': 'output_text', 'text': repeated}]},
                        *[{'type': 'function_call', 'call_id': call_id, 'name': 'shell_exec',
                           'arguments': json.dumps({'task': call_id})} for call_id in ('picture', 'note')]]
                    final = [{'type': 'message', 'role': 'assistant',
                        'content': [{'type': 'output_text', 'text': final_text}]}]
                    scripts = [{'status': 'completed', 'output': output} for output in (batch, final)]
                else:
                    provider = ChatCompletionsProvider(ChatCompletionsConfig(name=origin, api_key='synthetic-key',
                        base_url='https://fixture.invalid/v1', model='fixture'))
                    self.config = replace(self.config, chat_completions=(provider.config,))
                    batch = {'role': 'assistant', 'content': [
                        {'type': 'text', 'text': repeated}, {'type': 'text', 'text': repeated}],
                        'reasoning_content': 'PRIVATE_PROVIDER_REASONING',
                        'reasoning_details': [{'type': 'reasoning.encrypted', 'data': 'PRIVATE_ENCRYPTED_REASONING'}],
                        'tool_calls': [{'type': 'function', 'id': call_id, 'function': {'name': 'shell_exec',
                            'arguments': json.dumps({'task': call_id})}} for call_id in ('picture', 'note')]}
                    final = {'role': 'assistant', 'content': final_text}
                    scripts = [{'choices': [{'finish_reason': reason, 'message': message}]}
                        for message, reason in ((batch, 'tool_calls'), (final, 'stop'))]
                await provider.aclose()
                provider._client = httpx.AsyncClient(base_url='https://fixture.invalid/v1/',
                    transport=httpx.MockTransport(lambda request: httpx.Response(200, json=copy.deepcopy(scripts.pop(0)))))
                self.addAsyncCleanup(provider.aclose)
                settings = await self.settings(provider=origin, model=provider.config.model, mode=ChatMode.ASSIST,
                    tool_history_mode=ToolHistoryMode.TRANSLATED, compact_trigger_tokens=100000,
                    max_input_images=10, max_interaction_rounds=3)
                async def execute(arguments, context):
                    return ToolResult('', 'shell_exec', {'ok': True}, evidence_parts=[
                        MessagePart(PartKind.TEXT, text='The selected picture is a red square.', origin='file_read')
                    ] if arguments['task'] == 'picture' else [])
                self.tools.runner.run.reset_mock()
                self.tools.runner.run.side_effect = execute
                runtime = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools,
                    providers={origin: provider}, preview_cache=self.preview_cache)
                result = await runtime.run_turn(session_id=self.session, user_display_name='Participant',
                    incoming_message=ConversationMessage.user_text('Check both.'), emit=None)
                delivered = await runtime.record_assistant_text(session_id=self.session, text=result.text, metadata={
                    'provider_native': {'provider': origin, 'model': settings.model, 'items': result.provider_history_items}})
                original = (await self.store.read_messages(self.session, [delivered.db_id]))[0].message

                wire = []
                target = GeminiProvider(replace(self.config.gemini, api_key='synthetic-key'))
                def respond(request):
                    wire.append(json.loads(request.content))
                    return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP',
                        'content': {'role': 'model', 'parts': [{'text': 'Both statements remain visible.'}]}}]})
                target._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
                self.addAsyncCleanup(target.aclose)
                await self.settings(provider='gemini', model='gemini-3.8-flash')
                switched = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools,
                    providers={'gemini': target}, preview_cache=self.preview_cache)
                await switched.run_turn(session_id=self.session, user_display_name='Participant',
                    incoming_message=ConversationMessage.user_text('Keep the exact earlier statements.'))
                visible = '\n'.join(self.visible_texts(wire[0], 'gemini'))
                self.assertEqual(visible.count(repeated), 2, 'Identical original text parts are not accidental duplicates.')
                self.assertEqual(visible.count(final_text), 1)
                self.assertEqual(self.call_result_ids(wire[0], 'gemini'), (['picture', 'note'], ['picture', 'note']))
                self.assertNotIn('PRIVATE_ENCRYPTED_REASONING', json.dumps(wire[0]))
                self.assertNotIn('PRIVATE_PROVIDER_REASONING', json.dumps(wire[0]))
                self.assertEqual((await self.store.read_messages(self.session, [delivered.db_id]))[0].message, original)
                self.assertEqual(self.tools.runner.run.await_count, 2)

    async def test_mixed_text_and_parallel_results_survive_switch_append_restart_and_switch_back(self):
        first_text = '  先看看图片。\n这不是结论。 \n'
        second_text = '\n再核对备注；保留这句说明。\t'
        final_text = '图和备注已核对。'
        model = 'gemini-3.8-flash'
        native_batch = {'role': 'model', 'parts': [
            {'text': 'PRIVATE_REASONING_NOT_VISIBLE', 'thought': True},
            {'text': first_text},
            {'functionCall': {'name': 'shell_exec', 'id': 'picture', 'args': {'task': 'picture'}},
                'thoughtSignature': 'original-call-signature'},
            {'text': second_text},
            {'functionCall': {'name': 'shell_exec', 'id': 'note', 'args': {'task': 'note'}}},
        ]}
        native_final = {'role': 'model', 'parts': [
            {'text': final_text, 'thoughtSignature': 'original-final-signature'}]}
        buffer = io.BytesIO()
        with Image.new('RGB', (8, 8), 'red') as picture:
            picture.save(buffer, format='PNG')
        pixels = base64.b64encode(buffer.getvalue()).decode()

        for target, use_emit in ((name, emit) for name in ('gemini', 'openai', 'compatible') for emit in (True, False)):
            with self.subTest(target=target, streamed_text=use_emit):
                await self.store.reset_context(self.session)
                source_wire = []
                source_scripts = [native_batch, native_final]
                def source_response(request):
                    source_wire.append(json.loads(request.content))
                    return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP',
                        'content': copy.deepcopy(source_scripts.pop(0))}]})
                source_provider = GeminiProvider(replace(self.config.gemini, api_key='synthetic-key'))
                source_provider._client = httpx.AsyncClient(transport=httpx.MockTransport(source_response))
                self.addAsyncCleanup(source_provider.aclose)
                settings = await self.settings(provider='gemini', model=model, mode=ChatMode.ASSIST,
                    tool_history_mode=ToolHistoryMode.TRANSLATED, compact_trigger_tokens=100000,
                    max_input_images=10, max_interaction_rounds=3)
                runtime = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools,
                    providers={'gemini': source_provider}, preview_cache=self.preview_cache)
                async def execute(arguments, context):
                    return ToolResult('', 'shell_exec', {'ok': True, 'checked': arguments['task']},
                        evidence_parts=[MessagePart(PartKind.IMAGE, mime_type='image/png',
                            data_b64=pixels, origin='file_read')] if arguments['task'] == 'picture' else [])
                self.tools.runner.run.reset_mock()
                self.tools.runner.run.side_effect = execute
                result = await runtime.run_turn(session_id=self.session, user_display_name='Participant',
                    incoming_message=ConversationMessage.user_text('核对图片和备注。'),
                    emit=AsyncMock() if use_emit else None)
                delivered = await runtime.record_assistant_text(session_id=self.session, text=result.text, metadata={
                    'provider_native': {'provider': 'gemini', 'model': model, 'items': result.provider_history_items}})
                original_reply = (await self.store.read_messages(self.session, [delivered.db_id]))[0].message
                if not use_emit:
                    self.assertIn(first_text.strip(), result.text)
                    self.assertIn(second_text.strip(), result.text)
                    self.assertIn(final_text, result.text)
                self.assertEqual(self.tools.runner.run.await_count, 2)

                target_wire = []
                if target == 'gemini':
                    provider = GeminiProvider(replace(self.config.gemini, api_key='synthetic-key'))
                    target_model = 'gemini-future-flash'
                    answer = {'candidates': [{'finishReason': 'STOP', 'content': {
                        'role': 'model', 'parts': [{'text': '保留先前的说明。'}]}}]}
                    history_key = 'contents'
                elif target == 'openai':
                    provider = OpenAIResponsesProvider(replace(self.config.openai, api_key='synthetic-key'))
                    target_model = provider.config.model
                    answer = {'status': 'completed', 'output': [{'type': 'message', 'role': 'assistant',
                        'content': [{'type': 'output_text', 'text': '保留先前的说明。'}]}]}
                    history_key = 'input'
                else:
                    provider = ChatCompletionsProvider(ChatCompletionsConfig(name=target,
                        api_key='synthetic-key', base_url='https://fixture.invalid/v1', model='fixture'))
                    target_model = provider.config.model
                    self.config = replace(self.config, chat_completions=(provider.config,))
                    answer = {'choices': [{'finish_reason': 'stop', 'message': {
                        'role': 'assistant', 'content': '保留先前的说明。'}}]}
                    history_key = 'messages'
                await provider.aclose()
                def target_response(request):
                    target_wire.append(json.loads(request.content))
                    return httpx.Response(200, json=copy.deepcopy(answer))
                provider._client = httpx.AsyncClient(base_url='https://fixture.invalid/v1/',
                    transport=httpx.MockTransport(target_response))
                self.addAsyncCleanup(provider.aclose)
                await self.settings(provider=target, model=target_model)
                switched = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools,
                    providers={target: provider}, preview_cache=self.preview_cache)
                for text in ('换模型后，你还记得刚才的说明吗？', '请继续核对，但不用重新执行工具。'):
                    latest = await switched.ingest_user_message(session_id=self.session,
                        incoming_message=ConversationMessage.user_text(text))
                    await switched.run_turn_from_stored(session_id=self.session,
                        user_display_name='Participant', trigger_message_id=latest.db_id)
                for payload in target_wire:
                    visible = '\n'.join(self.visible_texts(payload, target))
                    for text in (first_text, second_text, final_text):
                        self.assertEqual(visible.count(text), 1, f'Ordinary assistant text must survive once: {text}')
                    self.assertEqual(self.call_result_ids(payload, target), (['picture', 'note'], ['picture', 'note']))
                    if target == 'gemini':
                        batches = [item for item in payload['contents']
                            if any('functionCall' in part for part in item.get('parts', []))]
                        self.assertEqual(len(batches), 1)
                        self.assertEqual([part['functionCall']['id'] for part in batches[0]['parts']
                            if 'functionCall' in part], ['picture', 'note'])
                    elif target == 'compatible':
                        batches = [item for item in payload['messages'] if item.get('tool_calls')]
                        self.assertEqual(len(batches), 1)
                        self.assertEqual([call['id'] for call in batches[0]['tool_calls']], ['picture', 'note'])
                    serialized = json.dumps(payload)
                    self.assertNotIn('original-call-signature', serialized)
                    self.assertNotIn('original-final-signature', serialized)
                    self.assertNotIn('PRIVATE_REASONING_NOT_VISIBLE', serialized)
                self.assertEqual(target_wire[1][history_key][:len(target_wire[0][history_key])],
                    target_wire[0][history_key])

                # Reopen storage and rebuild with no preview cache; no prior
                # runtime/history instance supplies the preserved visible text.
                await self.store.close()
                self.store = await self.new_store()
                cache = PreviewCache(self.store, max_bytes=0)
                self.addCleanup(cache.close)
                self.preview_cache = cache
                restored = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools,
                    providers={target: provider}, preview_cache=cache)
                await restored.run_turn_from_stored(session_id=self.session,
                    user_display_name='Participant', trigger_message_id=latest.db_id)
                self.assertEqual(target_wire[2][history_key], target_wire[1][history_key])
                self.assertEqual(self.tools.runner.run.await_count, 2)
                self.assertEqual((await self.store.read_messages(self.session, [delivered.db_id]))[0].message,
                    original_reply, 'Portable history must not rewrite the delivered source or aggregate its text again.')

                state = await restored._get_live_state(self.session)
                recovered = restored._build_provider_history(state, settings=settings, provider_name='gemini')
                recovered = await cache.materialize_many(self.session, recovered, vision=True)
                contents = source_provider._prepare_contents_for_request([
                    item for message in recovered for item in source_provider._message_to_contents(message)])
                self.assertEqual(contents[:len(source_wire[1]['contents'])], source_wire[1]['contents'])
                self.assertEqual([content for content in contents if content['role'] == 'model'],
                    [native_batch, native_final])
