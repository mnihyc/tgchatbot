"""The same attachment remains usable across providers without transport-path noise."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tgchatbot.config import ChatCompletionsConfig, load_config
from tests.business_helpers import BusinessTestCase
from tgchatbot.core.memory import MemoryService
from tgchatbot.domain.attachments import generated_attachment_reference
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind
from tgchatbot.providers.chat_completions import ChatCompletionsProvider
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.providers.openai_responses import OpenAIResponsesProvider


class AttachmentPresentationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='fixture-attachment-view-', dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        with patch.dict(os.environ, {'APP_DATA_DIR': temporary.name,
                'OPENAI_API_KEY': 'mock', 'GEMINI_API_KEY': 'mock', 'DEEPSEEK_API_KEY': 'mock'}, clear=True):
            config = load_config(require_telegram=False)
        chat = ChatCompletionsProvider(config.chat_completions[0])
        self.addAsyncCleanup(chat.aclose)
        self.providers = [GeminiProvider(config.gemini), OpenAIResponsesProvider(config.openai), chat]

    @staticmethod
    def wire(provider, message):
        method = getattr(provider, '_message_to_input_items', None) or provider._message_to_contents
        return method(message)

    async def test_relative_file_reference_keeps_original_words_and_one_usable_location(self):
        relative = '2026-09-15/报告_a1b2c3d4e5f67890.pdf'
        message = ConversationMessage(MessageRole.USER, [
            MessagePart(PartKind.TEXT, text='看这份报告；我写的 /literal/path 不要更改。'),
            MessagePart(PartKind.FILE, filename='报告.pdf', mime_type='application/pdf', size_bytes=1234,
                artifact_path='/srv/private-account/session/' + relative, workspace_path=relative),
        ], metadata={'presentation_version': 2})
        for provider in self.providers:
            with self.subTest(provider=provider.name):
                shown = json.dumps(self.wire(provider, message), ensure_ascii=False)
                self.assertEqual(shown.count(relative), 1)
                self.assertIn('application/pdf', shown)
                self.assertIn('1234 bytes', shown)
                self.assertIn(message.parts[0].text, shown)
                self.assertNotIn('/srv/private-account/', shown)

    async def test_missing_original_does_not_hide_available_retained_preview(self):
        message = ConversationMessage(MessageRole.USER, [
            MessagePart(PartKind.TEXT, text='这张图是哪次的？'),
            MessagePart(PartKind.FILE, filename='original.png', mime_type='image/png',
                detail='original file unavailable: missing export file', remote_sync=False),
            MessagePart(PartKind.IMAGE, mime_type='image/png', data_b64='ZmFrZQ==', remote_sync=False),
        ], metadata={'presentation_version': 2})
        for provider in self.providers:
            with self.subTest(provider=provider.name):
                shown = json.dumps(self.wire(provider, message), ensure_ascii=False)
                self.assertIn('original file unavailable: missing export file', shown)
                if provider.capabilities.multimodal_input:
                    self.assertIn('ZmFrZQ==', shown)
                else:
                    self.assertIn('image input is unavailable', shown)

    async def test_legacy_and_recorded_native_history_remain_exact(self):
        file = MessagePart(PartKind.FILE, filename='report.pdf', mime_type='application/pdf', size_bytes=1234,
            artifact_path='/old/workspace/report.pdf', detail='legacy detail')
        message = ConversationMessage(MessageRole.USER, [file])
        expected = '[Attached file: report.pdf, application/pdf, 1234 bytes, remote_path=/old/workspace/report.pdf]'
        for provider in self.providers:
            with self.subTest(provider=provider.name):
                shown = json.dumps(self.wire(provider, message), ensure_ascii=False)
                self.assertIn(expected, shown)
                self.assertNotIn('legacy detail', shown)
                native = self.wire(provider, message)
                replay = ConversationMessage(MessageRole.USER, [file], metadata={
                    'presentation_version': 2, 'provider_native': {'provider': provider.name, 'items': native}})
                self.assertEqual(self.wire(provider, replay), native)

    async def test_image_evidence_cleanup_preserves_custom_context_and_unavailable_routes(self):
        image_id = 'img:1:1:1'
        origin = 'memory_image:' + image_id
        label = MessagePart(PartKind.TEXT, text='[Original image evidence: '
            + json.dumps({'image_id': image_id, 'actor_id': 'person_id:7'}) + ']', origin=origin)
        pixels = MessagePart(PartKind.IMAGE, data_b64='ZmFrZQ==', mime_type='image/png', origin=origin)
        mechanical = replace(pixels, text=generated_attachment_reference(pixels))
        for case, part, labels, version in (
            ('custom_caption', replace(mechanical, text='The cup belongs to Participant 7.'), [label], 2),
            ('meaningful_detail', replace(pixels, detail='Only a cropped preview is available.'), [label], 2),
            ('missing_label', mechanical, [], 2),
            ('wrong_owner', mechanical, [replace(label, origin='memory_image:img:2:1:1')], 2),
            ('unavailable_pixels', replace(mechanical, data_b64=None), [label], 2),
            ('legacy', mechanical, [label], 1),
        ):
            if case == 'meaningful_detail':
                part = replace(part, text=generated_attachment_reference(part))
            for provider in self.providers:
                with self.subTest(case=case, provider=provider.name):
                    message = ConversationMessage(MessageRole.TOOL, [*labels, part], name='memory_read', metadata={
                        'presentation_version': version, 'tool_evidence': True, 'tool_phase': 'result',
                        'tool_provider': provider.name, 'tool_payload': {'call_id': 'read:7', 'output': {'ok': True}}})
                    shown = json.dumps(self.wire(provider, message), ensure_ascii=False)
                    self.assertIn(part.text, shown)
                    if case == 'unavailable_pixels':
                        self.assertIn('visual evidence unavailable', shown)


class StoredMemoryImagePresentationTests(BusinessTestCase):
    async def test_new_rebuilt_image_results_keep_one_owner_label_and_pixels_without_generated_descriptor(self):
        source = await self.store.append_message(self.session, ConversationMessage(MessageRole.USER,
            [MessagePart(PartKind.TEXT, text='My cup is the blue one.'),
             MessagePart(PartKind.IMAGE, data_b64='ZmFrZQ==', mime_type='image/png')], metadata={
                 'source': 'telegram', 'source_chat_id': '100', 'source_message_id': '1',
                 'actor_id': 'telegram:user:7', 'actor_kind': 'user', 'actor_name': 'Participant'}))
        memory = MemoryService(self.store, SimpleNamespace(enabled=False))
        recalled = await memory.read(self.session, [source.db_id])
        image_id = recalled['messages'][0]['images'][0]['image_id']
        resolved = await self.store.resolve_message_images(self.session, [source.db_id], [image_id])
        saved = await self.runtime.record_tool_observation(session_id=self.session, name='memory_read', phase='result',
            payload={'call_id': 'read:8', 'output': {'ok': True, 'image_results': [{'image_id': image_id, 'status': 'opened'}]}},
            provider_name='gemini', metadata_update={'tool_evidence': True}, evidence_parts=resolved['evidence_parts'])
        # Persistence generates a fallback reference for canonical source offsets.
        image = next(part for part in saved.message.parts if part.kind == PartKind.IMAGE)
        descriptor = generated_attachment_reference(image)
        self.assertEqual(image.text, descriptor)
        reader = await self.new_store()
        restored = next(message for message in await reader.list_messages(self.session) if message.name == 'memory_read')
        materialized = (await self.preview_cache.materialize_many(self.session, [restored], vision=True))[0]
        providers = [GeminiProvider(self.config.gemini), OpenAIResponsesProvider(self.config.openai)]
        chat = ChatCompletionsProvider(ChatCompletionsConfig(name='compatible', api_key='mock',
            base_url='https://example.invalid/v1', model='fixture', multimodal_input=True))
        self.addAsyncCleanup(chat.aclose)
        for provider in [*providers, chat]:
            with self.subTest(provider=provider.name):
                message = replace(materialized, metadata=dict(materialized.metadata, tool_provider=provider.name))
                shown = json.dumps(AttachmentPresentationTests.wire(provider, message), ensure_ascii=False)
                self.assertEqual(shown.count('[Original image evidence: '), 1)
                if provider is chat:
                    self.assertIn('visual evidence unavailable', shown)
                    self.assertNotIn('ZmFrZQ==', shown)
                else:
                    self.assertNotIn(descriptor, shown)
                    self.assertIn('ZmFrZQ==', shown)
                legacy = replace(message, metadata=dict(message.metadata, presentation_version=1))
                self.assertIn(descriptor, json.dumps(AttachmentPresentationTests.wire(provider, legacy)))
        self.assertEqual(next(part for part in materialized.parts if part.kind == PartKind.IMAGE).text, descriptor,
            'Serializer cleanup must not mutate reconstructed evidence')
        again = next(message for message in await reader.list_messages(self.session) if message.name == 'memory_read')
        self.assertEqual(again, restored)
