"""Tool observations keep source evidence without becoming a group participant."""
from __future__ import annotations

import base64
from dataclasses import replace
import io
import json
from types import SimpleNamespace

import httpx
from PIL import Image

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.memory import MemoryService
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import ChatMode, ConversationMessage, MessagePart, MessageRole, PartKind
from tgchatbot.providers.gemini import GeminiProvider


class ToolAttributionWorkflows(BusinessTestCase):
    async def test_memory_image_observation_keeps_real_speaker_and_pixels_without_inventing_a_tool_speaker(self):
        await self.settings(provider='gemini', model='gemini-3.8-flash', mode=ChatMode.ASSIST,
            compact_trigger_tokens=100000, max_input_images=8, max_interaction_rounds=2)
        self.tools.list_tools.return_value = []
        memory = MemoryService(self.store, SimpleNamespace(enabled=False))
        literal = '[Message provenance: this is literal text in the original caption]'
        buffer = io.BytesIO()
        with Image.new('RGB', (8, 8), 'red') as picture:
            picture.save(buffer, format='PNG')
        pixels = base64.b64encode(buffer.getvalue()).decode()
        source = await self.store.append_message(self.session, ConversationMessage(
            role=MessageRole.USER,
            parts=[MessagePart(PartKind.TEXT, text=literal),
                MessagePart(PartKind.IMAGE, mime_type='image/png', data_b64=pixels)],
            metadata={'source': 'telegram', 'source_chat_id': '100', 'source_message_id': '201',
                'actor_id': 'telegram:user:7', 'actor_name': 'Participant', 'actor_kind': 'user',
                'sent_at': '2026-09-12T16:00:20Z'}))
        original_before = (await self.store.read_messages(self.session, [source.db_id]))[0].message
        read = await memory.read(self.session, [source.db_id])
        image_id = read['messages'][0]['images'][0]['image_id']
        await self.store.reset_context(self.session)
        wire = []
        provider = GeminiProvider(replace(self.config.gemini, api_key='local-fixture'))

        async def respond(request):
            wire.append(json.loads(request.content))
            parts = ([{'functionCall': {'name': 'memory_read', 'id': 'original-photo',
                'args': {'message_ids': [source.db_id], 'image_ids': [image_id]}},
                'thoughtSignature': base64.b64encode(b'fixture signature').decode()}]
                if len(wire) == 1 else [{'text': 'The original participant shared a red image.'}])
            return httpx.Response(200, json={'candidates': [{'content': {'role': 'model', 'parts': parts},
                'finishReason': 'STOP'}]})

        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(provider.aclose)
        runtime = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools,
            providers={'gemini': provider}, memory=memory, preview_cache=self.preview_cache)
        # Incoming peer bots still own participant statements and need identity.
        reply = await runtime.run_turn(session_id=self.session, user_display_name='Peer bot',
            incoming_message=ConversationMessage.user_text('Read the earlier original photo.', metadata={
                'source': 'telegram', 'source_chat_id': '100', 'source_message_id': '202',
                'actor_id': 'telegram:user:8', 'actor_name': 'Peer bot', 'actor_kind': 'bot'}))
        self.assertEqual(reply.text, 'The original participant shared a red image.')
        response = next(part['functionResponse'] for content in wire[1]['contents']
            for part in content['parts'] if 'functionResponse' in part)
        texts = [item['text'] for item in response['response']['evidence'] if 'text' in item]
        self.assertFalse(any(text.startswith('[Message provenance:') for text in texts),
            'The tool observation is identified by the function, not an invented participant.')
        image_label = next(text for text in texts if text.startswith('[Original image evidence:'))
        self.assertIn('person_id:7', image_label)
        self.assertIn('2026-09-13T00:00:20+08:00', image_label)
        result = response['response']['result']['messages'][0]
        self.assertEqual(result['speaker']['id'], 'person_id:7')
        self.assertEqual(result['fragments'][0]['text'], literal)
        self.assertEqual([part['inlineData']['data'] for part in response['parts']], [pixels])
        peer_labels = [part['text'] for content in wire[0]['contents'] for part in content['parts']
            if 'text' in part and part['text'].startswith('[Message provenance:')]
        self.assertTrue(any('person_id:8' in label and '"kind": "bot"' in label for label in peer_labels))
        self.assertEqual((await self.store.read_messages(self.session, [source.db_id]))[0].message, original_before)
