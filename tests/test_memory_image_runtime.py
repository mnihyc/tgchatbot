"""Historical image reads exercise real tools, storage and provider wire formats."""
from __future__ import annotations

import base64
from dataclasses import replace
import io
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from PIL import Image

from tests.business_helpers import BusinessTestCase
from tgchatbot.config import ChatCompletionsConfig
from tgchatbot.core.memory import MemoryService
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import ChatMode, ConversationMessage, MessagePart, MessageRole, PartKind, StickerMode, ToolResult
from tgchatbot.providers.chat_completions import ChatCompletionsProvider
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.storage.previews import PreviewCache
from tgchatbot.tools.base import ToolSpec


class MemoryImageRuntimeTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.settings(mode=ChatMode.ASSIST, max_input_images=1, compact_target_images=1,
            compact_trigger_tokens=100000, max_interaction_rounds=2)
        self.tools.list_tools.return_value = []
        self.memory = MemoryService(self.store, SimpleNamespace(enabled=False))
        self.wire = []

    async def photo(self, color, number):
        image = io.BytesIO()
        Image.new('RGB', (8, 8), color).save(image, format='PNG')
        encoded = base64.b64encode(image.getvalue()).decode()
        row = await self.store.append_message(self.session, ConversationMessage(
            role=MessageRole.USER,
            parts=[MessagePart(PartKind.TEXT, text=f'Cup arrangement photo {number}.'),
                MessagePart(PartKind.IMAGE, mime_type='image/png', data_b64=encoded)],
            metadata={'source': 'telegram', 'source_chat_id': '100', 'source_message_id': str(number),
                'actor_id': f'telegram:user:{number}', 'actor_kind': 'user', 'actor_name': 'Participant'}))
        return row, encoded

    async def image_id(self, source):
        result = await self.memory.read(self.session, [source.db_id])
        return result['messages'][0]['images'][0]['image_id']

    async def gemini(self, scripts):
        provider = GeminiProvider(replace(self.config.gemini, api_key='local-fixture'))
        async def respond(request):
            self.wire.append(json.loads(request.content))
            parts = scripts.pop(0)
            return httpx.Response(200, json={'candidates': [{'content': {'role': 'model', 'parts': parts},
                'finishReason': 'STOP'}]})
        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(provider.aclose)
        await self.settings(provider='gemini', model='gemini-3.8-flash')
        return provider

    @staticmethod
    def function(arguments, call_id):
        return [{'functionCall': {'name': 'memory_read', 'args': arguments, 'id': call_id},
            'thoughtSignature': base64.b64encode(b'fixture signature').decode()}]

    def responses(self, wire):
        return [part['functionResponse'] for content in wire['contents'] for part in content.get('parts', [])
            if 'functionResponse' in part and part['functionResponse']['name'] == 'memory_read']

    async def test_parallel_reads_share_the_next_request_image_allowance(self):
        first, first_pixels = await self.photo('red', 201)
        second, _ = await self.photo('blue', 202)
        first_id, second_id = await self.image_id(first), await self.image_id(second)
        await self.store.reset_context(self.session)
        provider = await self.gemini([
            self.function({'message_ids': [first.db_id], 'image_ids': [first_id]}, 'first')
            + self.function({'message_ids': [second.db_id], 'image_ids': [second_id]}, 'second'),
            [{'text': 'I inspected the first photo; the other image did not fit.'}],
        ])
        runtime = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools,
            providers={'gemini': provider}, memory=self.memory, preview_cache=self.preview_cache)
        await runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Inspect both old photos.'))
        results = self.responses(self.wire[1])
        self.assertEqual({item['image_id']: item['status'] for result in results
            for item in result['response']['result']['image_results']},
            {first_id: 'opened', second_id: 'omitted'})
        self.assertEqual([part['inlineData']['data'] for result in results for part in result.get('parts', [])],
            [first_pixels])

    async def test_memory_and_sticker_tools_share_image_allowance_in_either_order(self):
        source, original_pixels = await self.photo('red', 203)
        image_id = await self.image_id(source)
        candidate = io.BytesIO()
        Image.new('RGB', (8, 8), 'blue').save(candidate, format='PNG')
        candidate_pixels = base64.b64encode(candidate.getvalue()).decode()
        memory_call = self.function({'message_ids': [source.db_id], 'image_ids': [image_id]}, 'read')[0]
        sticker_call = {'functionCall': {'name': 'sticker_query', 'args': {}, 'id': 'candidate'},
            'thoughtSignature': base64.b64encode(b'fixture signature').decode()}
        for first in ('memory', 'sticker'):
            with self.subTest(first=first):
                await self.store.reset_context(self.session)
                self.wire.clear()
                calls = [memory_call, sticker_call] if first == 'memory' else [sticker_call, memory_call]
                provider = await self.gemini([calls, [{'text': 'I inspected the image that fit this request.'}]])
                await self.settings(sticker_mode=StickerMode.AUTO)
                # Candidate retrieval quality is outside this admission test;
                # its valid visual result competes with a real original read.
                query = SimpleNamespace(run=AsyncMock(return_value=ToolResult('', 'sticker_query',
                    {'ok': True, 'candidates': [{'sticker_id': 'fixture'}], 'candidate_count': 1},
                    evidence_parts=[
                        MessagePart(PartKind.TEXT, text='A friendly reaction candidate.', origin='sticker_candidate:fixture'),
                        MessagePart(PartKind.IMAGE, mime_type='image/png', data_b64=candidate_pixels,
                            origin='sticker_candidate:fixture'),
                    ])))
                self.tools.list_tools.return_value = [ToolSpec('sticker_query', 'Find reaction candidates',
                    {'type': 'object', 'properties': {}}, query)]
                runtime = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools,
                    providers={'gemini': provider}, memory=self.memory, preview_cache=self.preview_cache)
                await runtime.run_turn(session_id=self.session, user_display_name='Participant',
                    incoming_message=ConversationMessage.user_text('Recall the old photo and consider a friendly sticker.'))
                responses = [part['functionResponse'] for content in self.wire[1]['contents']
                    for part in content.get('parts', []) if 'functionResponse' in part]
                by_name = {result['name']: result for result in responses}
                self.assertEqual(set(by_name), {'memory_read', 'sticker_query'})
                self.assertEqual([part['inlineData']['data'] for result in responses for part in result.get('parts', [])],
                    [original_pixels if first == 'memory' else candidate_pixels])
                read = by_name['memory_read']['response']['result']['image_results'][0]
                self.assertEqual(read['image_id'], image_id)
                self.assertEqual(read['status'], 'opened' if first == 'memory' else 'omitted')
                if first == 'sticker':
                    self.assertTrue(read.get('reason'), 'Omission must explain why retained bytes were not shown')
                shortlist = by_name['sticker_query']['response']['result']
                self.assertEqual(shortlist['candidates'], [] if first == 'memory' else [{'sticker_id': 'fixture'}])
                self.assertEqual(shortlist['candidate_count'], 0 if first == 'memory' else 1)

    async def test_selected_historical_images_fit_real_gemini_requests_and_reconstruct_after_retirement(self):
        first, first_pixels = await self.photo('red', 101)
        second, second_pixels = await self.photo('blue', 102)
        first_id, second_id = await self.image_id(first), await self.image_id(second)
        await self.store.reset_context(self.session)
        provider = await self.gemini([
            self.function({'message_ids': [first.db_id, second.db_id], 'image_ids': [first_id, second_id]}, 'first'),
            [{'text': 'I inspected the first photo.'}],
            self.function({'message_ids': [second.db_id], 'image_ids': [second_id]}, 'second'),
            [{'text': 'I inspected the second photo.'}],
        ])
        runtime = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools,
            providers={'gemini': provider}, memory=self.memory, preview_cache=self.preview_cache)
        await runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Inspect these two old cup photos.'))
        first_response = self.responses(self.wire[1])[0]
        statuses = {item['image_id']: item['status'] for item in first_response['response']['result']['image_results']}
        self.assertEqual(statuses, {first_id: 'opened', second_id: 'omitted'})
        self.assertEqual([part['inlineData']['data'] for part in first_response['parts']], [first_pixels])
        self.assertIn('telegram:user:101', json.dumps(first_response['response']['evidence']))

        await runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Now inspect the other photo.'))
        final_responses = self.responses(self.wire[-1])
        self.assertEqual([part['inlineData']['data'] for response in final_responses for part in response.get('parts', [])],
            [second_pixels], 'Retired first-tool pixels must not reappear through native replay')
        latest = next(response for response in final_responses if response['id'] == 'second')
        self.assertEqual(latest['response']['result']['image_results'][0]['status'], 'opened')
        self.assertIn('Image compacted', json.dumps(final_responses))
        rows = await self.store.list_messages(self.session)
        reads = [row for row in rows if row.name == 'memory_read']
        self.assertEqual([row.metadata['tool_phase'] for row in reads], ['call', 'result', 'call', 'result'])
        self.assertTrue(all(row.role == MessageRole.TOOL for row in reads))

        settings = await self.store.get_or_create_session(self.session, self.config.default_session_settings())
        cold_cache = PreviewCache(self.store, max_bytes=0)
        self.addCleanup(cold_cache.close)
        cold = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools,
            providers={'gemini': provider}, memory=self.memory, preview_cache=cold_cache)
        async def contents(owner):
            state = await owner._get_live_state(self.session)
            history = owner._build_provider_history(state, settings=settings, provider_name='gemini')
            history = await owner.preview_cache.materialize_many(self.session, history, vision=True)
            return [content for message in history for content in provider._message_to_contents(message)]
        self.assertEqual(await contents(runtime), await contents(cold))
        # Working prompt retirement has not made either original image unavailable.
        for source in (first, second):
            result = await self.memory.read(self.session, [source.db_id])
            self.assertTrue(result['messages'][0]['images'][0]['available'])

    async def test_text_only_tool_protocol_reports_omission_without_claiming_missing_bytes(self):
        source, _ = await self.photo('green', 103)
        image_id = await self.image_id(source)
        await self.store.reset_context(self.session)
        provider = ChatCompletionsProvider(ChatCompletionsConfig(name='compatible', api_key='local-fixture',
            base_url='https://fixture.invalid/v1', model='fixture', multimodal_input=True))
        config = replace(self.config, chat_completions=(provider.config,))
        await provider.aclose()
        scripts = [
            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'read', 'type': 'function', 'function': {
                'name': 'memory_read', 'arguments': json.dumps({'message_ids': [source.db_id], 'image_ids': [image_id]})}}]},
            {'role': 'assistant', 'content': 'This route can return the reference but cannot inspect its pixels.'},
        ]
        async def respond(request):
            self.wire.append(json.loads(request.content))
            return httpx.Response(200, json={'choices': [{'message': scripts.pop(0), 'finish_reason': 'stop'}]})
        provider._client = httpx.AsyncClient(base_url='https://fixture.invalid/v1/', transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(provider.aclose)
        await self.settings(provider='compatible', model='fixture')
        runtime = AgentRuntime(config=config, store=self.store, tool_registry=self.tools,
            providers={'compatible': provider}, memory=self.memory, preview_cache=self.preview_cache)
        await runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Inspect that old photo.'))
        tool = next(row for row in self.wire[1]['messages'] if row['role'] == 'tool')
        output = json.loads(tool['content'])
        self.assertEqual(output['image_results'][0]['status'], 'omitted')
        self.assertIn('text-only', output['image_results'][0]['reason'])
        self.assertNotIn('image_url', json.dumps(self.wire[1]))
        self.assertTrue((await self.memory.read(self.session, [source.db_id]))['messages'][0]['images'][0]['available'])
