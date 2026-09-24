"""DeepSeek wire compatibility without credentials or paid model requests."""
from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace

import httpx

from tgchatbot.config import ChatCompletionsConfig
from tgchatbot.core.token_estimator import TokenEstimator
from tgchatbot.domain.models import ConversationMessage, MessagePart, PartKind, SessionSettings, ToolCall
from tgchatbot.providers.deepseek_responses import DeepSeekResponsesProvider
from tgchatbot.tools.base import ToolSpec


class DeepSeekResponsesContracts(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.config = ChatCompletionsConfig(name='deepseek', api_key='mock',
            base_url='https://fixture.invalid/', model='deepseek-flash', multimodal_input=True)
        self.settings = SessionSettings(provider='deepseek', model=self.config.model)
        self.provider = DeepSeekResponsesProvider(self.config)
        self.addAsyncCleanup(self.provider.aclose)
        self.wire = []
        self.body = {'status': 'completed', 'output': [
            {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'done'}]}]}
        def respond(request):
            self.assertEqual(request.url.path, '/responses')
            self.wire.append(json.loads(request.content))
            return httpx.Response(200, json=copy.deepcopy(self.body))
        self.provider._client = httpx.AsyncClient(base_url=self.config.base_url, transport=httpx.MockTransport(respond))

    async def generate(self, **changes):
        return await self.provider.generate(**dict(settings=self.settings,
            messages=[ConversationMessage.user_text('Hello.')], instructions='Use the provided evidence.',
            tools=[], **changes))

    async def test_existing_thinking_and_sampling_configuration_survives_endpoint_change(self):
        for extra, expected in (({}, None), ({'reasoning_effort': 'max'}, {'effort': 'max'}),
                                ({'thinking': {'type': 'disabled'}, 'reasoning_effort': 'high'}, {'effort': 'none'}),
                                ({'thinking': {'type': 'enabled'}}, None)):
            with self.subTest(extra=extra):
                self.provider.config = replace(self.config, extra_body=extra, temperature=0.3, top_p=0.98,
                    max_output_tokens=16384)
                result = await self.generate()
                payload = self.wire[-1]
                self.assertEqual(result.final_text, 'done')
                self.assertEqual(payload.get('reasoning'), expected)
                self.assertNotIn('reasoning_effort', payload)
                self.assertNotIn('thinking', payload)
                self.assertNotIn('include', payload)
                self.assertNotIn('text', payload)
                self.assertEqual(payload['temperature'], 0.3)
                self.assertEqual(payload['top_p'], 0.98)
                self.assertEqual(payload['max_output_tokens'], 16384)
                self.assertEqual(self.provider.config.extra_body, extra)

    async def test_structured_profile_requests_keep_configured_output_mode_and_no_external_tools(self):
        schema = {'type': 'object', 'properties': {'additions': {'type': 'array', 'items': {'type': 'string'}}}}
        for mode in ('json_object', 'json_schema', 'prompt'):
            with self.subTest(mode=mode):
                self.provider.config = replace(self.config, structured_output=mode)
                await self.generate(response_schema=schema, response_schema_name='profile_patch')
                payload = self.wire[-1]
                self.assertFalse(payload['tools'])
                self.assertEqual(payload['tool_choice'], 'none')
                if mode == 'prompt':
                    self.assertNotIn('text', payload)
                else:
                    self.assertEqual(payload['text']['format']['type'], mode)
                if mode == 'json_schema':
                    self.assertEqual(payload['text']['format']['schema'], schema)
                else:
                    self.assertIn(json.dumps(schema), payload['instructions'])

    async def test_mixed_text_reasoning_and_parallel_calls_replay_verbatim(self):
        self.body['output'] = [
            {'type': 'reasoning', 'id': 'r1', 'summary': [], 'content': [{'type': 'reasoning_text', 'text': 'Check both.'}]},
            {'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': 'Checking.'}]},
            *[{'type': 'function_call', 'call_id': call_id, 'name': 'lookup', 'arguments': '{}'} for call_id in ('one', 'two')],
        ]
        self.body['usage'] = {'input_tokens': 1000, 'input_tokens_details': {'cached_tokens': 800},
            'output_tokens': 20, 'output_tokens_details': {'reasoning_tokens': 10}}
        tools = [ToolSpec('lookup', 'Look up a note.', {'type': 'object', 'properties': {}}, None)]
        result = await self.provider.generate(settings=self.settings, messages=[], instructions='Check notes.', tools=tools)
        self.assertEqual(result.final_text, 'Checking.')
        self.assertEqual([call.call_id for call in result.tool_calls], ['one', 'two'])
        self.assertEqual(result.usage.cached_input_tokens, 800)
        native = self.provider.persistent_history_items(result)
        self.assertEqual(native, self.body['output'])
        message = ConversationMessage.assistant_text(result.final_text, metadata={'provider_native': {
            'provider': 'deepseek', 'model': self.settings.model, 'items': native}})
        extras = [item for call in result.tool_calls for item in self.provider.make_tool_result_items(call, {'found': True})]
        await self.provider.generate(settings=self.settings, messages=[message], instructions='Check notes.',
            tools=tools, extra_input_items=extras)
        self.assertEqual(self.wire[-1]['input'], native + extras)
        self.assertEqual(self.wire[-1]['tools'][0]['name'], 'lookup')
        self.assertFalse(self.wire[-1]['tools'][0]['strict'])
        self.assertEqual(self.wire[-1]['tool_choice'], 'auto')

    async def test_legacy_snapshot_preserves_reasoning_text_and_calls_without_mutating_database_shape(self):
        native = [{'role': 'assistant', 'content': 'The accompanying sentence.', 'reasoning_content': 'Earlier reasoning.',
            'tool_calls': [{'id': 'legacy', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"n": 1}'}}]}]
        message = ConversationMessage.assistant_text('The accompanying sentence.', metadata={'provider_native': {
            'provider': 'deepseek', 'model': self.settings.model, 'items': native}})
        before = copy.deepcopy(message)
        await self.provider.generate(settings=self.settings, messages=[message], instructions='Continue.', tools=[],
            extra_input_items=self.provider.make_tool_result_items(ToolCall('lookup', 'legacy', {}), {'found': True}))
        items = self.wire[-1]['input']
        self.assertEqual([item.get('type', 'message') for item in items],
            ['reasoning', 'message', 'function_call', 'function_call_output'])
        self.assertEqual(items[0]['content'][0]['text'], 'Earlier reasoning.')
        self.assertEqual(items[1]['content'], 'The accompanying sentence.')
        self.assertEqual(items[2]['arguments'], '{"n": 1}')
        self.assertEqual(items[2]['call_id'], items[3]['call_id'])
        self.assertEqual(message, before)

    async def test_image_restrictions_and_evidence_capability_remain_unchanged(self):
        pixels = MessagePart(PartKind.IMAGE, data_b64='AAAA', mime_type='image/png')
        user = replace(ConversationMessage.user_text('Picture.'), parts=[pixels])
        assistant = replace(ConversationMessage.assistant_text('Picture.'), parts=[pixels])
        for enabled in (True, False):
            self.provider.config = replace(self.config, multimodal_input=enabled)
            self.provider.capabilities = replace(self.provider.capabilities, multimodal_input=enabled)
            await self.provider.generate(settings=self.settings, messages=[user, assistant], instructions='', tools=[])
            self.assertEqual(json.dumps(self.wire[-1]['input']).count('input_image'), int(enabled))
            self.assertFalse(self.provider.supports_tool_evidence(self.settings))
            result = self.provider.make_tool_result_items(ToolCall('read_doc', 'image', {}), {}, [pixels])
            self.assertNotIn('AAAA', json.dumps(result))
            self.assertIn('visual evidence unavailable', json.dumps(result))

    async def test_reasoning_and_images_are_counted_without_counting_base64_as_text(self):
        reasoning = {'type': 'reasoning', 'content': [{'type': 'reasoning_text', 'text': 'prior thought ' * 1000}]}
        self.assertGreater(self.provider._estimate_input_item_tokens(reasoning), 1000)
        image = replace(ConversationMessage.user_text(''), parts=[MessagePart(PartKind.IMAGE,
            data_b64='AAAA', mime_type='image/png')])
        def estimate(message):
            return self.provider.estimate_request_tokens(settings=self.settings, messages=[message], instructions='', tools=[])
        self.assertGreaterEqual(estimate(image).history_tokens, TokenEstimator.IMAGE_TOKENS)
        larger = replace(image, parts=[replace(image.parts[0], data_b64='AAAA' * 10000)])
        self.assertEqual(estimate(image), estimate(larger))
