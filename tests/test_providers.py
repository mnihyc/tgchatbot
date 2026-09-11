"""Provider contract tests use in-memory HTTP responses; no credentials or API charges."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx

from tgchatbot.config import ChatCompletionsConfig, load_config
from tgchatbot.core.compaction_schema import compaction_json_schema
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import ChatMode, ConversationMessage, MessagePart, MessageRole, PartKind, SessionSettings, ToolHistoryMode
from tgchatbot.providers.chat_completions import ChatCompletionsProvider
from tgchatbot.providers.factory import build_provider, build_providers
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
        provider = await self.make_provider(name='deepseek')
        body = self.answer('Visible introduction', reasoning_content='private continuation', tool_calls=[{'id': 'a', 'type': 'function', 'function': {'name': 'lookup', 'arguments': '{}'}}])
        native = provider.persistent_history_items(provider._parse_response(body))
        message = ConversationMessage(role=MessageRole.TOOL, name='lookup', parts=[MessagePart(kind=PartKind.TEXT, text='lookup query')], metadata={'tool_phase': 'call', 'tool_provider': 'deepseek', 'tool_payload': {'call_id': 'a', 'arguments': {}}, 'provider_native': {'provider': 'deepseek', 'items': native}})
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
                        result = await runtime._generate_structured_candidate(provider, config.default_session_settings(), [ConversationMessage.user_text('Please use English')], mode='episode')
                        self.assertEqual(result['user_profile'], ['Use English'])
                        payload = captured[0]
                        self.assertNotIn('tools', payload) if name in {'deepseek', 'openrouter'} else None
                    finally:
                        await provider.aclose()


class StickerAnalysisContractTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.directory.cleanup)
        self.environment = patch.dict(os.environ, {'APP_DATA_DIR': self.directory.name, 'GEMINI_API_KEY': 'mock'}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_offline_analysis_uses_shared_provider_and_one_event_loop(self):
        import asyncio
        from types import SimpleNamespace
        from unittest.mock import AsyncMock
        from scripts.build_sticker_index import StickerAnalysisClient
        from tgchatbot.domain.models import ProviderResponse
        from tgchatbot.providers.base import ProviderCapabilities
        loops = []
        requests = []
        async def generate(**kwargs):
            loops.append(asyncio.get_running_loop())
            requests.append(kwargs)
            if len(requests) == 1:
                return ProviderResponse(final_text='{"caption":"hello"}')
            return ProviderResponse(final_text='{"result":{"caption":"hello"},"alignment_score":1}')
        provider = SimpleNamespace(capabilities=ProviderCapabilities(), generate=generate, aclose=AsyncMock())
        with patch('scripts.build_sticker_index.build_provider', return_value=provider) as factory:
            client = StickerAnalysisClient(config=load_config(require_telegram=False), provider_name='gemini', model='vision-model')
            try:
                result = client.analyze(relative_path='pack/example.webp', source_format_name='webp', ocr_summary={'joined_text': 'hello'}, frame_payloads=[{'mime': 'image/png', 'data': 'AAAA'}])
            finally:
                client.close()
        self.assertEqual(result['caption'], 'hello')
        self.assertIs(loops[0], loops[1])
        self.assertEqual(factory.call_args.args[1], 'gemini')
        self.assertEqual(requests[0]['settings'].model, 'vision-model')
        self.assertEqual(requests[0]['messages'][0].parts[-1].kind, PartKind.IMAGE)
        self.assertEqual(requests[0]['tools'], [])
        self.assertTrue(requests[1]['response_schema'])
        provider.aclose.assert_awaited_once()

    def test_no_embeddings_build_records_disabled_manifest_and_does_not_call_embedding_api(self):
        from scripts import build_sticker_index
        argv = ['build_sticker_index.py', '--stickers-dir', self.directory.name, '--index-db', str(Path(self.directory.name) / 'index.sqlite3'), '--no-embeddings', '--workers', '1']
        with patch('sys.argv', argv), patch.object(build_sticker_index, 'load_dotenv', return_value=False), patch.object(build_sticker_index, 'PaddleOCR', object()), patch.object(build_sticker_index, 'StickerAnalysisClient'), patch.object(build_sticker_index, '_build_rows_parallel', return_value={'ok': 0, 'skipped': 0, 'failed': 0}), patch.object(build_sticker_index.EmbeddingProvider, 'from_env') as embedding, patch('builtins.print'):
            build_sticker_index.main()
        embedding.assert_not_called()
        manifest = json.loads((Path(self.directory.name) / 'embeddings_manifest.json').read_text())
        self.assertFalse(manifest['enabled'])
        self.assertEqual(manifest['dimensions'], 0)
        self.assertTrue((Path(self.directory.name) / 'tantivy_docs.jsonl').exists())


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
