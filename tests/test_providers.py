"""Provider contract tests use in-memory HTTP responses; no credentials or API charges."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zoneinfo import ZoneInfo

import httpx

from tgchatbot.config import ChatCompletionsConfig, load_config
from tgchatbot.core.compaction_schema import compaction_json_schema
from tgchatbot.domain.models import ChatMode, ConversationMessage, MessagePart, MessageRole, PartKind, SessionSettings, ToolCall, ToolHistoryMode, ToolResult
from tgchatbot.providers.chat_completions import ChatCompletionsProvider
from tgchatbot.providers.factory import build_provider, build_providers
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.providers.openai_responses import OpenAIResponsesProvider
from tgchatbot.tools.base import ToolSpec


class ProviderConfigTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.directory.cleanup)
        self.env = {'APP_DATA_DIR': self.directory.name, 'TGBOT_TOKEN': 'test-token'}

    def load(self, **env):
        with patch.dict(os.environ, self.env | env, clear=True):
            return load_config()

    def test_any_provider_can_be_the_only_provider(self):
        for name in ('openai', 'gemini', 'deepseek', 'openrouter'):
            with self.subTest(name=name):
                config = self.load(**{f'{name.upper()}_API_KEY': 'mock-key', f'{name.upper()}_MODEL': 'chosen-model'})
                self.assertEqual(config.configured_provider_names(), (name,))
                self.assertEqual(config.default_provider, name)
                self.assertEqual(config.default_session_settings().model, 'chosen-model')

    def test_new_session_metadata_defaults_to_utc_plus_eight(self):
        for env in ({}, {'DEFAULT_METADATA_TIMEZONE': '  '}):
            with self.subTest(env=env):
                settings = self.load(OPENAI_API_KEY='mock', **env).default_session_settings()
                moment = datetime(2026, 1, 1, tzinfo=timezone.utc).astimezone(ZoneInfo(settings.metadata_timezone))
                self.assertEqual(moment.utcoffset(), timedelta(hours=8))
                self.assertEqual(moment.isoformat(), '2026-01-01T08:00:00+08:00')
                self.assertEqual(SessionSettings().metadata_timezone, settings.metadata_timezone)

    def test_explicit_metadata_timezone_overrides_the_new_default(self):
        for zone in ('UTC', 'Asia/Tokyo', 'Europe/Berlin'):
            with self.subTest(zone=zone):
                settings = self.load(OPENAI_API_KEY='mock', DEFAULT_METADATA_TIMEZONE=f' {zone} ').default_session_settings()
                self.assertEqual(settings.metadata_timezone, zone)

    def test_mixed_profiles_select_explicit_default_without_cross_provider_model_fallback(self):
        config = self.load(OPENAI_API_KEY='mock', GEMINI_API_KEY='mock', DEEPSEEK_API_KEY='mock', DEFAULT_PROVIDER='deepseek')
        self.assertEqual(set(config.configured_provider_names()), {'openai', 'gemini', 'deepseek'})
        self.assertEqual(config.default_session_settings().model, config.provider_config('deepseek').model)
        with self.assertRaisesRegex(ValueError, 'Unknown provider'):
            config.default_model_for_provider('typo')

    def test_explicit_missing_default_is_an_error_not_a_fallback(self):
        config = self.load(OPENAI_API_KEY='mock', DEFAULT_PROVIDER='gemini')
        with self.assertRaisesRegex(RuntimeError, 'DEFAULT_PROVIDER'):
            build_providers(config)

    def test_named_profiles_allow_different_models_on_the_same_endpoint(self):
        profiles = [{'name': name, 'api_key_env': 'SHARED_KEY', 'base_url': 'https://example.invalid/v1', 'model': model} for name, model in [('fast', 'small'), ('careful', 'large')]]
        config = self.load(LLM_PROVIDERS_JSON=json.dumps(profiles), SHARED_KEY='mock', DEFAULT_PROVIDER='careful')
        self.assertEqual(config.default_model_for_provider('fast'), 'small')
        self.assertEqual(config.default_session_settings().model, 'large')

    def test_profile_misconfiguration_is_reported_at_startup(self):
        baseline = {'name': 'custom', 'api_key_env': 'SHARED_KEY', 'base_url': 'https://example.invalid/v1', 'model': 'test'}
        for change in ({'name': 'openai'}, {'structured_output': 'typo'}, {'multimodal_input': 'false'}, {'extra_body': {'tools': []}}, {'unknown_field': 1}, {'top_p': 2}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.load(SHARED_KEY='mock', LLM_PROVIDERS_JSON=json.dumps([baseline | change]))
        with self.assertRaisesRegex(ValueError, 'api_key'):
            self.load(LLM_PROVIDERS_JSON=json.dumps([baseline]))

    def test_openrouter_requires_explicit_model(self):
        with self.assertRaisesRegex(ValueError, 'model is required'):
            self.load(OPENROUTER_API_KEY='mock')

    def test_offline_jobs_do_not_require_a_telegram_token(self):
        with patch.dict(os.environ, {'APP_DATA_DIR': self.directory.name, 'DEEPSEEK_API_KEY': 'mock'}, clear=True):
            self.assertEqual(load_config(require_telegram=False).default_provider, 'deepseek')
            with self.assertRaisesRegex(RuntimeError, 'TGBOT_TOKEN'):
                load_config()


class CachedUsageContractTests(unittest.TestCase):
    def test_reported_cache_reads_are_a_subset_of_input_not_added_to_totals(self):
        # Parse documented wire shapes without constructing an HTTP client.
        fixtures = (
            (GeminiProvider, {'candidates': [{'content': {'parts': [{'text': 'done'}]}}]},
             'usageMetadata', {'promptTokenCount': 10000, 'candidatesTokenCount': 23, 'totalTokenCount': 10023},
             lambda value: {'cachedContentTokenCount': value}),
            (OpenAIResponsesProvider, {'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'done'}]}]},
             'usage', {'input_tokens': 10000, 'output_tokens': 23},
             lambda value: {'input_tokens_details': {'cached_tokens': value, 'cache_write_tokens': 500}}),
            (ChatCompletionsProvider, {'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]},
             'usage', {'prompt_tokens': 10000, 'completion_tokens': 23},
             lambda value: {'prompt_tokens_details': {'cached_tokens': value, 'cache_write_tokens': 500}}),
            (ChatCompletionsProvider, {'choices': [{'message': {'role': 'assistant', 'content': 'done'}}]},
             'usage', {'prompt_tokens': 10000, 'completion_tokens': 23},
             lambda value: {'prompt_cache_hit_tokens': value, 'prompt_cache_miss_tokens': 10000 - value}),
        )
        for provider_class, body, usage_key, counters, cache_fields in fixtures:
            provider = object.__new__(provider_class)
            for cached in (8192, 0, None):
                with self.subTest(provider=provider_class.__name__, fields=cache_fields(0), cached=cached):
                    usage = counters | (cache_fields(cached) if cached is not None else {})
                    response = provider._parse_response(body | {usage_key: usage})
                    self.assertEqual(response.final_text, 'done')
                    self.assertEqual(response.usage.cached_input_tokens, cached)
                    self.assertEqual(response.usage.input_tokens, 10000)
                    self.assertEqual(response.usage.output_tokens, 23)
                    self.assertEqual(response.usage.total_tokens, 10023)
            self.assertIsNone(provider._parse_response(body).usage.cached_input_tokens)

    def test_compatible_standard_zero_is_not_replaced_by_an_alias_counter(self):
        provider = object.__new__(ChatCompletionsProvider)
        body = {'choices': [{'message': {'content': 'done'}}], 'usage': {
            'prompt_tokens': 10000, 'completion_tokens': 23,
            'prompt_tokens_details': {'cached_tokens': 0}, 'prompt_cache_hit_tokens': 8192}}
        self.assertEqual(provider._parse_response(body).usage.cached_input_tokens, 0)

    def test_gemini_usage_survives_a_response_without_candidates(self):
        provider = object.__new__(GeminiProvider)
        response = provider._parse_response({'usageMetadata': {
            'promptTokenCount': 10000, 'cachedContentTokenCount': 8192, 'totalTokenCount': 10000}})
        self.assertEqual(response.final_text, '')
        self.assertEqual(response.usage.cached_input_tokens, 8192)
        self.assertEqual(response.usage.input_tokens, 10000)
        self.assertEqual(response.usage.total_tokens, 10000)


class FrameworkProfileToolHistoryTests(unittest.TestCase):
    def test_framework_pair_uses_native_tool_shapes_across_provider_switches(self):
        arguments = {'actor_ids': ['telegram:user:7'], 'include_agent_preferences': True}
        output = {'ok': True, 'profiles': [{'actor_id': 'telegram:user:7', 'facts': [{'claim': 'Prefers jasmine tea.'}]}]}
        for previous_provider in (None, 'other-provider'):
            call, result = [ConversationMessage(role=MessageRole.TOOL, name='user_profile_fetch', parts=[], metadata={
                'synthetic_role': 'profile_refresh', 'refresh_reason': 'compaction',
                'tool_phase': phase, 'tool_provider': previous_provider,
                'tool_payload': {'call_id': 'profile-refresh-fixture', **payload},
            }) for phase, payload in (('call', {'arguments': arguments}), ('result', {'output': output}))]
            for provider_class in (GeminiProvider, OpenAIResponsesProvider, ChatCompletionsProvider):
                with self.subTest(provider=provider_class.__name__, previous_provider=previous_provider):
                    provider = object.__new__(provider_class)
                    if provider_class is ChatCompletionsProvider:
                        provider.name = 'openrouter'
                    encode = provider._message_to_contents if provider_class is GeminiProvider else provider._message_to_input_items
                    encoded_call, encoded_result = encode(call)[0], encode(result)[0]
                    if provider_class is GeminiProvider:
                        part = encoded_call['parts'][0]
                        self.assertEqual(encoded_call['role'], 'model')
                        self.assertEqual(part['thoughtSignature'], 'skip_thought_signature_validator')
                        self.assertEqual(part['functionCall'], {'name': call.name, 'id': 'profile-refresh-fixture', 'args': arguments})
                        self.assertEqual(encoded_result['parts'][0]['functionResponse'], {
                            'name': call.name, 'id': 'profile-refresh-fixture', 'response': {'result': output}})
                    elif provider_class is OpenAIResponsesProvider:
                        self.assertEqual(encoded_call['type'], 'function_call')
                        self.assertEqual(encoded_result['type'], 'function_call_output')
                        self.assertEqual(encoded_call['call_id'], encoded_result['call_id'])
                        self.assertEqual(json.loads(encoded_call['arguments']), arguments)
                        self.assertEqual(json.loads(encoded_result['output']), output)
                    else:
                        self.assertEqual(encoded_call['role'], 'assistant')
                        self.assertEqual(encoded_result['role'], 'tool')
                        tool_call = encoded_call['tool_calls'][0]
                        self.assertEqual(tool_call['id'], encoded_result['tool_call_id'])
                        self.assertEqual(json.loads(tool_call['function']['arguments']), arguments)
                        self.assertEqual(json.loads(encoded_result['content']), output)

    def test_actual_gemini_thought_signature_is_preserved(self):
        provider = object.__new__(GeminiProvider)
        native = {'role': 'model', 'parts': [{'functionCall': {'name': 'user_profile_fetch',
            'args': {'actor_ids': []}, 'id': 'model-call'}, 'thoughtSignature': 'opaque-real-signature'}]}
        message = ConversationMessage(role=MessageRole.TOOL, name='user_profile_fetch', parts=[],
            metadata={'tool_provider': 'gemini', 'tool_phase': 'call',
                      'provider_native': {'provider': 'gemini', 'items': [native]}})
        self.assertEqual(provider._message_to_contents(message), [native])


class ToolEvidenceContractTests(unittest.TestCase):
    def exchange(self, evidence=None):
        metadata = {'tool_evidence': True, 'tool_provider': 'another-provider'}
        call = ConversationMessage(MessageRole.TOOL, [], name='sticker_query', metadata=metadata | {
            'tool_phase': 'call', 'tool_payload': {'call_id': 'portable-query-1', 'arguments': {'intent_core': 'offer comfort'}}})
        output = {'candidates': [{'sticker_id': 'asset-a'}, {'sticker_id': 'asset-b'}]}
        evidence = evidence if evidence is not None else [
            MessagePart(PartKind.TEXT, text='asset-a, frame 1 at 0ms'),
            MessagePart(PartKind.IMAGE, mime_type='image/png', data_b64='AAAA'),
            MessagePart(PartKind.TEXT, text='asset-b, frame 1 at 0ms'),
            MessagePart(PartKind.IMAGE, mime_type='image/png', data_b64='BBBB'),
        ]
        result = ConversationMessage(MessageRole.TOOL,
            [MessagePart(PartKind.TEXT, text='duplicate JSON summary', origin='tool_output'), *evidence],
            name='sticker_query', metadata=metadata | {'tool_phase': 'result', 'tool_payload': {
                'call_id': 'portable-query-1', 'output': output}})
        return call, result

    def test_native_replay_preserves_candidate_frame_pairing_and_tool_provenance(self):
        call, result = self.exchange()
        gemini = object.__new__(GeminiProvider)
        encoded_call = gemini._message_to_contents(call)[0]['parts'][0]
        response = gemini._message_to_contents(result)[0]['parts'][0]['functionResponse']
        self.assertEqual(encoded_call['functionCall']['id'], response['id'])
        self.assertEqual(encoded_call['thoughtSignature'], 'skip_thought_signature_validator')
        refs = response['response']['evidence']
        self.assertEqual([x['text'] for x in refs if 'text' in x], ['asset-a, frame 1 at 0ms', 'asset-b, frame 1 at 0ms'])
        images = [part['inlineData']['data'] for part in response['parts']]
        self.assertEqual([images[x['image_part']-1] for x in refs if 'image_part' in x], ['AAAA', 'BBBB'])
        self.assertNotIn('$ref', json.dumps(response))
        self.assertTrue(all(set(part['inlineData']) == {'mimeType', 'data'} for part in response['parts']))
        self.assertNotIn('duplicate JSON summary', json.dumps(response))
        openai = object.__new__(OpenAIResponsesProvider)
        item = openai._message_to_input_items(result)[0]
        self.assertEqual(item['type'], 'function_call_output')
        self.assertEqual(item['call_id'], openai._message_to_input_items(call)[0]['call_id'])
        self.assertEqual([p['type'] for p in item['output']], ['input_text', 'input_text', 'input_image', 'input_text', 'input_image'])
        self.assertEqual([p['image_url'] for p in item['output'] if p['type'] == 'input_image'], ['data:image/png;base64,AAAA', 'data:image/png;base64,BBBB'])
        self.assertEqual(result.role, MessageRole.TOOL)
        tool_result = ToolResult('portable-query-1', 'sticker_query', {}, evidence_parts=result.parts[1:])
        self.assertEqual((tool_result.artifacts, tool_result.stickers), ([], []))

    def test_text_only_tool_route_keeps_ids_and_never_makes_a_fake_user_image(self):
        call, result = self.exchange()
        compatible = object.__new__(ChatCompletionsProvider)
        compatible.name = 'custom'
        self.assertFalse(compatible.supports_tool_evidence(SessionSettings()))
        wire = compatible._message_to_input_items(result)
        self.assertEqual(len(wire), 1)
        self.assertEqual(wire[0]['role'], 'tool')
        self.assertIn('asset-a', wire[0]['content'])
        self.assertIn('visual evidence unavailable', wire[0]['content'])
        self.assertNotIn('AAAA', wire[0]['content'])
        self.assertEqual(compatible._message_to_input_items(call)[0]['tool_calls'][0]['id'], wire[0]['tool_call_id'])
        gemini = object.__new__(GeminiProvider)
        self.assertFalse(gemini.supports_tool_evidence(SessionSettings(model='gemini-2.5-flash')))
        older = gemini._message_to_contents(result, tool_images=False)[0]['parts'][0]['functionResponse']
        self.assertNotIn('parts', older)
        self.assertIn('visual evidence unavailable', json.dumps(older))

    def test_retired_evidence_remains_text_on_every_adapter_and_nested_image_cost_is_counted(self):
        call, live = self.exchange()
        _, retired = self.exchange([MessagePart(PartKind.TEXT, text='asset-a [Image compacted]')])
        for cls in (GeminiProvider, OpenAIResponsesProvider):
            provider = object.__new__(cls)
            encode = provider._message_to_contents if cls is GeminiProvider else provider._message_to_input_items
            estimate = provider._estimate_content_tokens if cls is GeminiProvider else provider._estimate_input_item_tokens
            first, second = encode(live)[0], encode(retired)[0]
            self.assertGreaterEqual(estimate(first) - estimate(second), 1800)
            self.assertIn('[Image compacted]', json.dumps(second))
            self.assertNotIn('AAAA', json.dumps(second))
            enlarged = replace(live, parts=[replace(p, data_b64=p.data_b64 * 10000) if p.data_b64 else p for p in live.parts])
            self.assertEqual(estimate(first), estimate(encode(enlarged)[0]))


class ServiceTierContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_tier_reaches_native_request_and_errors_do_not_fall_back(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            for name in ('gemini', 'openai', 'deepseek'):
                with self.subTest(provider=name), patch.dict(os.environ, {
                    'APP_DATA_DIR': directory, 'TGBOT_TOKEN': 'mock', f'{name.upper()}_API_KEY': 'mock',
                    f'{name.upper()}_MODEL': 'gemini-3.8-flash' if name == 'gemini' else 'configured-model',
                }, clear=True):
                    config = load_config()
                    provider = build_provider(config, name)
                    await provider.aclose()
                    requests = []
                    reject = False
                    def handle(request):
                        requests.append(json.loads(request.content))
                        if reject:
                            return httpx.Response(503, json={'error': {'message': 'Flex capacity unavailable'}})
                        if name == 'gemini':
                            body = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'done'}]}}],
                                    'usageMetadata': {'promptTokenCount': 12, 'candidatesTokenCount': 3, 'totalTokenCount': 15, 'serviceTier': 'flex'}}
                        elif name == 'openai':
                            body = {'output': [], 'usage': {'input_tokens': 12, 'output_tokens': 3, 'total_tokens': 15}, 'service_tier': 'flex'}
                        else:
                            body = {'choices': [{'message': {'role': 'assistant', 'content': 'done'}}],
                                    'usage': {'prompt_tokens': 12, 'completion_tokens': 3, 'total_tokens': 15}, 'service_tier': 'flex'}
                        return httpx.Response(200, json=body)
                    provider._client = httpx.AsyncClient(base_url='https://test.invalid/', transport=httpx.MockTransport(handle))
                    try:
                        settings = replace(config.default_session_settings(), service_tier='flex', native_web_search_mode='off')
                        args = {'settings': settings, 'messages': [ConversationMessage.user_text('Synthetic annotation')], 'instructions': 'Describe.', 'tools': []}
                        result = await provider.generate(**args)
                        self.assertEqual(requests[-1]['service_tier'], 'flex')
                        self.assertEqual(result.usage.service_tier, 'flex')
                        self.assertEqual(result.usage.input_tokens, 12)
                        reject = True
                        before = len(requests)
                        with self.assertRaises(httpx.HTTPStatusError):
                            await provider.generate(**args)
                        self.assertEqual(len(requests), before + 1)
                        self.assertEqual(requests[-1]['service_tier'], 'flex')
                        reject = False
                        await provider.generate(**(args | {'settings': replace(settings, service_tier=None)}))
                        self.assertNotIn('service_tier', requests[-1])
                    finally:
                        await provider.aclose()


class ChatCompletionsContractTests(unittest.IsolatedAsyncioTestCase):
    async def make_provider(self, **overrides):
        profile = ChatCompletionsConfig(name='custom', api_key='mock', base_url='https://example.invalid/api/v1', model='test-model')
        provider = ChatCompletionsProvider(replace(profile, **overrides))
        await provider.aclose()
        self.requests = []
        self.responses = []

        def handler(request):
            self.requests.append(request)
            return httpx.Response(200, json=self.responses.pop(0))

        provider._client = httpx.AsyncClient(base_url=profile.base_url + '/', transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(provider.aclose)
        return provider

    @staticmethod
    def answer(text='done', **message):
        return {'choices': [{'message': {'role': 'assistant', 'content': text, **message}}], 'usage': {'prompt_tokens': 11, 'completion_tokens': 7}}

    @staticmethod
    def settings(provider):
        return SessionSettings(provider=provider.name, model=provider.config.model, mode=ChatMode.ASSIST)

    async def test_tool_cycle_preserves_reasoning_and_call_ids(self):
        provider = await self.make_provider(name='deepseek')
        calls = [{'id': 'call-a', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{"query":"weather"}'}}]
        details = [{'type': 'reasoning.encrypted', 'data': 'opaque-signature', 'index': 0}]
        self.responses.extend([self.answer('Checking.', tool_calls=calls, reasoning_content='provider thought', reasoning_details=details), self.answer('The lookup succeeded.')])
        tool = ToolSpec(name='lookup', description='Look up a query', parameters_schema={'type': 'object', 'properties': {'query': {'type': 'string'}}, 'required': ['query']}, runner=None)
        kwargs = {'settings': self.settings(provider), 'messages': [ConversationMessage.user_text('Check weather')], 'instructions': 'Be helpful', 'tools': [tool]}
        first = await provider.generate(**kwargs)
        self.assertEqual(first.tool_calls[0].arguments, {'query': 'weather'})
        self.assertEqual(first.reasoning_summaries, [])
        result_items = provider.make_tool_result_items(first.tool_calls[0], {'result': 'sunny'})
        second = await provider.generate(**(kwargs | {'tools': [], 'extra_input_items': first.continuation_items + result_items}))
        first_payload, second_payload = [json.loads(request.content) for request in self.requests]
        self.assertEqual(self.requests[0].url.path, '/api/v1/chat/completions')
        self.assertEqual(first_payload['tools'][0]['function']['parameters'], tool.parameters_schema)
        self.assertNotIn('tools', second_payload)
        self.assertEqual(second_payload['messages'][-2]['reasoning_details'], details)
        self.assertEqual(second_payload['messages'][-2]['reasoning_content'], 'provider thought')
        self.assertEqual(second_payload['messages'][-1]['tool_call_id'], 'call-a')
        self.assertEqual(second.final_text, 'The lookup succeeded.')
        self.assertEqual(second.usage.total_tokens, 18)

    async def test_schema_modes_support_compaction_without_vendor_specific_fields(self):
        schema = {'type': 'object', 'properties': {'summary': {'type': 'string'}}, 'required': ['summary'], 'additionalProperties': False}
        for mode in ('json_schema', 'json_object', 'prompt'):
            with self.subTest(mode=mode):
                provider = await self.make_provider(structured_output=mode, token_limit_parameter='max_completion_tokens')
                self.responses.append(self.answer('{"summary":"remember this"}'))
                response = await provider.generate(settings=self.settings(provider), messages=[ConversationMessage.user_text('history')], instructions='Summarize', tools=[], response_schema=schema, response_schema_name='memory')
                payload = json.loads(self.requests[-1].content)
                self.assertEqual(json.loads(response.final_text), {'summary': 'remember this'})
                self.assertNotIn('max_tokens', payload)
                self.assertEqual(payload['max_completion_tokens'], 4096)
                self.assertNotIn('reasoning', payload)
                if mode == 'json_schema':
                    self.assertEqual(payload['response_format']['json_schema']['schema'], schema)
                else:
                    self.assertIn(json.dumps(schema), payload['messages'][0]['content'])
                    if mode == 'json_object':
                        self.assertEqual(payload['response_format'], {'type': 'json_object'})
                    else:
                        self.assertNotIn('response_format', payload)

    async def test_vision_capability_and_attachment_path_are_preserved(self):
        provider = await self.make_provider(multimodal_input=True)
        message = ConversationMessage(role=MessageRole.USER, parts=[MessagePart(kind=PartKind.IMAGE, data_b64='AAAA', mime_type='image/png'), MessagePart(kind=PartKind.FILE, filename='notes.txt', artifact_path='/work/notes.txt')])
        items = provider._message_to_input_items(message)
        self.assertEqual(items[0]['content'][0]['image_url']['url'], 'data:image/png;base64,AAAA')
        self.assertIn('/work/notes.txt', items[0]['content'][1]['text'])
        provider = await self.make_provider(multimodal_input=False)
        content = provider._message_to_input_items(message)[0]['content']
        self.assertIn('image input is unavailable', content)
        self.assertNotIn('AAAA', content)

    async def test_malformed_tool_arguments_never_become_an_executable_empty_call(self):
        provider = await self.make_provider()
        for arguments in ('not json', '[]', 'null'):
            with self.subTest(arguments=arguments), self.assertRaises((ValueError, TypeError)):
                provider._parse_response(self.answer(tool_calls=[{'id': 'call-1', 'function': {'name': 'delete', 'arguments': arguments}}]))

    async def test_error_envelope_does_not_become_a_successful_empty_reply(self):
        provider = await self.make_provider()
        with self.assertRaises(RuntimeError):
            provider._parse_response({'error': {'message': 'upstream unavailable'}})
        with self.assertRaises(ValueError):
            provider._parse_response({'choices': []})

    async def test_extra_body_and_sampling_overrides_preserve_request_controls(self):
        provider = await self.make_provider(extra_body={'thinking': {'type': 'disabled'}}, temperature=0.3, top_p=0.9, function_tools=False)
        self.responses.append(self.answer())
        settings = replace(self.settings(provider), temperature=0.0)
        await provider.generate(settings=settings, messages=[ConversationMessage.user_text('Hi')], instructions='Test', tools=[ToolSpec('test', 'test', {}, None)])
        payload = json.loads(self.requests[0].content)
        self.assertEqual(payload['temperature'], 0)
        self.assertEqual(payload['top_p'], 0.9)
        self.assertEqual(payload['thinking'], {'type': 'disabled'})
        self.assertNotIn('tools', payload)

    async def test_same_provider_native_history_and_cross_provider_translation(self):
        from tgchatbot.core.runtime import AgentRuntime
        provider = await self.make_provider(name='deepseek')
        body = self.answer('Visible introduction', reasoning_content='private continuation', tool_calls=[{'id': 'a', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}])
        native = provider.persistent_history_items(provider._parse_response(body))
        message = ConversationMessage(role=MessageRole.TOOL, name='lookup', parts=[MessagePart(kind=PartKind.TEXT, text='lookup query')], metadata={'tool_phase': 'call', 'tool_provider': 'deepseek', 'tool_payload': {'call_id': 'a', 'arguments': {}}, 'provider_native': {'provider': 'deepseek', 'model': self.settings(provider).model, 'items': native}})
        runtime = object.__new__(AgentRuntime)
        settings = replace(self.settings(provider), tool_history_mode=ToolHistoryMode.NATIVE_SAME_PROVIDER)
        same = runtime._history_message_for_provider(settings=settings, provider_name='deepseek', message=message)
        self.assertEqual(provider._message_to_input_items(same), native)
        switched = runtime._history_message_for_provider(settings=settings, provider_name='openrouter', message=message)
        self.assertNotIn('provider_native', switched.metadata)
        self.assertIn('Visible introduction', switched.parts[0].text)
        self.assertNotIn('private continuation', switched.parts[0].text)
        self.assertTrue(runtime._message_has_tool_context(ConversationMessage.assistant_text('', metadata={'provider_native': {'provider': 'deepseek', 'items': native}})))

    async def test_token_estimate_counts_images_semantically_not_base64_size(self):
        provider = await self.make_provider(multimodal_input=True)
        def estimate(data):
            return provider.estimate_request_tokens(settings=self.settings(provider), messages=[ConversationMessage(role=MessageRole.USER, parts=[MessagePart(kind=PartKind.IMAGE, data_b64=data, mime_type='image/png')])], instructions='', tools=[])
        self.assertEqual(estimate('A').history_tokens, estimate('A' * 100000).history_tokens)


class AllAdaptersContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_each_provider_can_generate_validated_compaction(self):
        from tgchatbot.core.runtime import AgentRuntime
        schema = compaction_json_schema('episode')
        candidate = {name: [] for name in schema['properties']}
        candidate.update(scope='Preserve requested language', interaction_mode='chat_or_sharing', user_profile=['Use English'])
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
            for name in ('openai', 'gemini', 'deepseek', 'openrouter'):
                with self.subTest(name=name), patch.dict(os.environ, {'APP_DATA_DIR': directory, 'TGBOT_TOKEN': 'mock', f'{name.upper()}_API_KEY': 'mock', f'{name.upper()}_MODEL': 'test-model'}, clear=True):
                    config = load_config()
                    provider = build_provider(config, name)
                    await provider.aclose()
                    captured = []
                    def handler(request):
                        captured.append(json.loads(request.content))
                        text = json.dumps(candidate)
                        if name == 'openai':
                            body = {'output': [{'type': 'message', 'role': 'assistant', 'content': [{'type': 'output_text', 'text': text}]}]}
                        elif name == 'gemini':
                            body = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': text}]}}]}
                        else:
                            body = ChatCompletionsContractTests.answer(text)
                        return httpx.Response(200, json=body)
                    provider._client = httpx.AsyncClient(base_url='https://example.invalid/', transport=httpx.MockTransport(handler))
                    try:
                        runtime = object.__new__(AgentRuntime)
                        runtime.config = config
                        result = await runtime._generate_structured_candidate(provider, config.default_session_settings(), [ConversationMessage.user_text('Please use English')], mode='episode')
                        self.assertEqual(result['user_profile'], ['Use English'])
                        payload = captured[0]
                        self.assertNotIn('tools', payload) if name in {'deepseek', 'openrouter'} else None
                    finally:
                        await provider.aclose()


class ProviderTelegramControlTests(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_image_limits_return_usage_instead_of_crashing(self):
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from tgchatbot.transports.telegram_adapter import TelegramBotApp
        app = object.__new__(TelegramBotApp)
        app._allowed = lambda chat: True
        app._advanced_allowed = lambda update: True
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory, patch.dict(os.environ, {'APP_DATA_DIR': directory, 'OPENAI_API_KEY': 'mock'}, clear=True):
            app.config = load_config(require_telegram=False)
            settings = app.config.default_session_settings()
            app.store = SimpleNamespace(get_or_create_session=AsyncMock(return_value=settings))
            provider = build_provider(app.config, 'openai')
            self.addAsyncCleanup(provider.aclose)
            app.runtime = SimpleNamespace(providers={'openai': provider})
            reply = AsyncMock()
            update = SimpleNamespace(effective_chat=SimpleNamespace(id=100), effective_message=SimpleNamespace(reply_text=reply))
            for name in ('max_input_images', 'compact_target_images', 'native_web_search_max'):
                with self.subTest(name=name):
                    reply.reset_mock()
                    await app.param_command(update, SimpleNamespace(args=[name, '-1']))
                    self.assertIn(f'Invalid {name}', reply.await_args.args[0])
