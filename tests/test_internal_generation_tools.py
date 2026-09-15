"""Profile learning and compaction use supplied evidence without external tools."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import httpx

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.compaction_schema import compaction_json_schema
from tgchatbot.core.context_state import MemoryBlock
from tgchatbot.core.memory_worker import MemoryWorker
from tgchatbot.domain.models import ChatMode, ConversationMessage
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.providers.openai_responses import OpenAIResponsesProvider


class InternalGenerationToolsWorkflows(BusinessTestCase):
    async def provider_fixture(self, name):
        self.wire = []
        self.request_urls = []
        self.response_text = 'Ready.'

        def respond(request):
            self.wire.append(json.loads(request.content))
            self.request_urls.append(str(request.url))
            if name == 'gemini':
                return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP',
                    'content': {'role': 'model', 'parts': [{'text': self.response_text}]}}]})
            return httpx.Response(200, json={'status': 'completed', 'output': [
                {'type': 'message', 'role': 'assistant', 'content': [
                    {'type': 'output_text', 'text': self.response_text}]}]})

        config = replace(self.config.provider_config(name), api_key='synthetic-key',
            enable_native_web_search=True)
        provider = GeminiProvider(config) if name == 'gemini' else OpenAIResponsesProvider(config)
        if name == 'openai':
            await provider.aclose()
        provider._client = httpx.AsyncClient(base_url='https://example.invalid/',
            transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(provider.aclose)
        self.runtime.providers[name] = provider
        return provider

    async def live_request(self, provider, settings):
        self.response_text = 'Ready.'
        await provider.generate(settings=settings,
            messages=[ConversationMessage.user_text('Check the current train schedule.')],
            instructions='Help with the question.', tools=[self.tools.spec])
        tools = self.wire[-1]['tools']
        if provider.name == 'gemini':
            self.assertTrue(any('googleSearch' in tool for tool in tools))
            self.assertTrue(any('functionDeclarations' in tool for tool in tools))
        else:
            self.assertEqual({tool['type'] for tool in tools}, {'web_search', 'function'})

    def assert_internal_request(self, provider_name, *, max_output_tokens):
        request = self.wire[-1]
        self.assertFalse(request.get('tools'), 'Internal learning must have no external tool access.')
        if provider_name == 'gemini':
            self.assertIn('/models/gemini-3.8-flash:generateContent', self.request_urls[-1])
            self.assertNotIn('toolConfig', request)
            generation = request['generationConfig']
            self.assertEqual(generation['thinkingConfig'], {'thinkingLevel': 'high'})
            self.assertEqual(generation['maxOutputTokens'], max_output_tokens)
            self.assertIn('responseJsonSchema', generation)
        else:
            self.assertEqual(request['model'], 'gpt-5')
            self.assertEqual(request['tool_choice'], 'none')
            self.assertNotIn('max_tool_calls', request)
            self.assertNotIn('web_search_call.action.sources', request['include'])
            self.assertEqual(request['reasoning']['effort'], 'high')
            self.assertEqual(request['max_output_tokens'], max_output_tokens)
            self.assertEqual(request['text']['format']['type'], 'json_schema')

    async def profile_workflow(self, name):
        provider = await self.provider_fixture(name)
        # Both an explicit session choice and inherited provider defaults must
        # stay available to live chat without leaking into internal learning.
        for web_mode in ('on', 'default'):
            with self.subTest(native_web_search_mode=web_mode):
                self.session = f'telegram:100:{web_mode}'
                settings = await self.settings(provider=name,
                    model='gemini-3.8-flash' if name == 'gemini' else 'gpt-5',
                    mode=ChatMode.AGENT, native_web_search_mode=web_mode,
                    thinking_level='high', reasoning_effort='high', max_output_tokens=24576)
                original_settings = deepcopy(settings)
                source = await self.runtime.ingest_user_message(session_id=self.session,
                    incoming_message=ConversationMessage.user_text('I prefer jasmine tea.', metadata={
                        'actor_id': 'telegram:user:7', 'actor_kind': 'user', 'actor_name': 'Alex'}))
                original_message = deepcopy((await self.store.read_messages(
                    self.session, [source.db_id]))[0].message)
                worker = MemoryWorker(store=self.store, embeddings=SimpleNamespace(enabled=False),
                    providers={name: provider}, config=self.config)
                # A learning request has its own allowance; live chat keeps its
                # independently configured value when the worker finishes.
                worker.limits = replace(worker.limits, profile_output_tokens=16384)
                job = await self.store.claim_profile_batch(session_id=self.session, lazy=True,
                    max_bytes=worker.limits.profile_request_bytes, lease_seconds=worker.limits.lease_seconds)
                self.assertIsNotNone(job)
                await self.live_request(provider, settings)
                self.response_text = json.dumps({'additions': [{
                    'subject_actor_id': 'telegram:user:7', 'asserted_by': 'telegram:user:7',
                    'claim': 'Prefers jasmine tea.', 'kind': 'explicit', 'status': 'active',
                    'source_ids': [source.db_id], 'valid_from': None, 'valid_to': None,
                    'supersedes': None, 'reason': 'The speaker states this preference directly.'}],
                    'removals': []})
                self.assertTrue(await worker._guarded([job], worker._profile))
                self.assert_internal_request(name, max_output_tokens=16384)
                facts = await self.store.get_profile(self.session, 'telegram:user:7')
                self.assertEqual([(fact['claim'], fact['source_ids']) for fact in facts],
                    [('Prefers jasmine tea.', [source.db_id])])
                self.assertEqual((await self.store.read_messages(self.session, [source.db_id]))[0].message,
                    original_message)
                self.assertEqual(await self.settings(), original_settings)
                self.assertEqual(settings, original_settings)
                await self.live_request(provider, settings)

    async def test_gemini_profile_commits_without_external_tools_and_preserves_live_settings(self):
        await self.profile_workflow('gemini')

    async def test_openai_profile_commits_without_external_tools_and_preserves_live_settings(self):
        await self.profile_workflow('openai')

    async def compaction_workflow(self, name):
        provider = await self.provider_fixture(name)
        settings = await self.settings(provider=name,
            model='gemini-3.8-flash' if name == 'gemini' else 'gpt-5',
            mode=ChatMode.AGENT, native_web_search_mode='on', thinking_level='high',
            reasoning_effort='high', max_output_tokens=24576)
        original_settings = deepcopy(settings)
        source = await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text('The ticket is already booked.',
                metadata={'actor_id': 'telegram:user:7', 'actor_name': 'Alex'}))
        parent = MemoryBlock(block_id=1, sequence_no=1, summary_text='Alex booked the ticket.',
            estimated_tokens=12, source_message_count=1, actor_labels=('telegram:user:7',))
        await self.live_request(provider, settings)
        for mode in ('toolspan', 'episode', 'digest'):
            with self.subTest(mode=mode):
                candidate = {key: [] for key in compaction_json_schema(mode)['properties']}
                candidate.update(scope='The ticket booking is settled.', topics=['ticket booking'])
                if mode != 'digest':
                    candidate['interaction_mode'] = 'task_execution'
                self.response_text = json.dumps(candidate)
                if mode == 'toolspan':
                    result = await self.runtime._make_toolspan_block_candidate(provider, settings,
                        [source], session_id=self.session)
                elif mode == 'episode':
                    result = await self.runtime._make_episode_block_candidate(provider, settings,
                        [source.message], [source], [], session_id=self.session)
                else:
                    result = await self.runtime._make_digest_block_candidate(provider, settings, [parent],
                        session_id=self.session)
                self.assertEqual(result['data']['scope'], candidate['scope'])
                self.assert_internal_request(name, max_output_tokens=24576)
                self.assertEqual(settings, original_settings)
                self.assertEqual(await self.settings(), original_settings)
        await self.live_request(provider, settings)

    async def test_gemini_all_compaction_layers_keep_external_tools_out(self):
        await self.compaction_workflow('gemini')

    async def test_openai_all_compaction_layers_keep_external_tools_out(self):
        await self.compaction_workflow('openai')
