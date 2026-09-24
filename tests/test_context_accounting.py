"""Inspection must explain the real request without rewriting its cache prefix."""
from __future__ import annotations

import copy
import os
import unittest
from dataclasses import replace
from unittest.mock import patch

from tgchatbot.config import load_config
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind
from tgchatbot.providers.factory import build_provider
from tgchatbot.providers.chat_completions import ChatCompletionsProvider
from tgchatbot.providers.inspection import HistoryEntry, inspect_history
from tests.test_memory_image_tools import PIXEL


class ContextAccountingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        with patch.dict(os.environ, {'DEFAULT_PROVIDER': 'gemini', 'DEEPSEEK_API_KEY': 'fixture-key'}, clear=True):
            self.config = load_config(require_telegram=False)

    def tool(self, name, phase, payload, **metadata):
        return ConversationMessage(MessageRole.TOOL, [], name=name, metadata={
            'tool_phase': phase, 'tool_payload': {'call_id': name, **payload}, 'tool_evidence': True, **metadata})

    def wire(self, provider, settings, messages):
        if provider.inspection_format == 'gemini':
            return [item for message in messages for item in provider._message_to_contents(
                message, tool_images=provider.supports_tool_evidence(settings))]
        if provider.inspection_format == 'responses':
            return [item for message in messages for item in provider._message_to_input_items(message)]
        return provider._messages_for_request(messages)

    async def test_mixed_input_has_exclusive_categories_and_unchanged_wire_payload_on_every_route(self):
        providers = [build_provider(self.config, name, require_credentials=False)
                     for name in ('gemini', 'openai', 'deepseek')]
        providers.append(ChatCompletionsProvider(replace(self.config.provider_config('deepseek'), multimodal_input=False)))
        for provider in providers:
            self.addAsyncCleanup(provider.aclose)
            with self.subTest(provider=provider.name):
                settings = replace(self.config.default_session_settings(), provider=provider.name,
                    model='gemini-3.8-flash' if provider.name == 'gemini' else 'fixture-model')
                user = ConversationMessage.user_text('Literal user context. ' * 40)
                user.parts.append(MessagePart(PartKind.IMAGE, mime_type='image/png', data_b64=PIXEL))
                messages = [user, ConversationMessage.assistant_text('Authored response. ' * 40),
                    self.tool('shell_exec', 'call', {'arguments': {'command': 'pwd'}}),
                    self.tool('shell_exec', 'result', {'output': {'stdout': 'workspace\n' * 40}}),
                    self.tool('memory_read', 'result', {'output': {'messages': [{'text': 'Original recall. ' * 40}]}}),
                    self.tool('user_profile_fetch', 'result', {'output': {'profiles': [{'claim': 'Likes tea. ' * 40}]}}),
                    ConversationMessage.assistant_text('Summary of old messages. ' * 40),
                    ConversationMessage.user_text('Framework reply target.', metadata={'synthetic_role': 'reply_target'})]
                entries = [HistoryEntry(message, message_id=i+1, block_id=23 if i == 6 else None)
                           for i, message in enumerate(messages)]
                originals, wire = copy.deepcopy(messages), copy.deepcopy(self.wire(provider, settings, messages))
                raw = provider.estimate_request_tokens(settings=settings, messages=messages,
                    instructions='A fixture preset.', tools=[])
                result = inspect_history(provider, settings, entries, raw, raw.scaled(1.5))
                self.assertIsNotNone(result)
                self.assertEqual(sum(result['categories'].values()), raw.scaled(1.5).total_tokens)
                for category, indexes in {'user': [0], 'assistant': [1], 'tools': [2, 3],
                                          'memory': [4, 6], 'profiles': [5]}.items():
                    expected = sum(result['entries'][i]['categories'][category] for i in indexes)
                    self.assertGreater(expected, 0)
                    self.assertEqual(result['categories'][category], expected)
                self.assertEqual(result['entries'][7]['tokens'], result['entries'][7]['categories']['system'])
                self.assertEqual(result['images']['projected'], int(provider.capabilities.multimodal_input))
                self.assertEqual(result['images']['unsupported'], int(not provider.capabilities.multimodal_input))
                self.assertEqual(messages, originals)
                self.assertEqual(self.wire(provider, settings, messages), wire)

    async def test_signed_native_batch_keeps_prose_calls_and_thoughts_in_distinct_categories(self):
        provider = build_provider(self.config, 'gemini', require_credentials=False)
        self.addAsyncCleanup(provider.aclose)
        settings = replace(self.config.default_session_settings(), model='gemini-3.8-flash')
        text = {'text': 'I will check the original evidence.'}
        thought = {'thought': True, 'text': 'Private continuation.', 'thoughtSignature': 'opaque'}
        call = {'functionCall': {'name': 'memory_read', 'id': 'c1', 'args': {'message_ids': [42]}},
                'thoughtSignature': 'opaque-signed-call'}
        second = {'functionCall': {'name': 'user_profile_fetch', 'id': 'c2', 'args': {'actor_ids': ['person_id:7']}},
                  'thoughtSignature': 'skip_thought_signature_validator'}
        message = self.tool('memory_read', 'call', {'arguments': {'message_ids': [42]}},
            provider_native={'provider': 'gemini', 'items': [{'role': 'model', 'parts': [thought, text, call, second]}]})
        raw = provider.estimate_request_tokens(settings=settings, messages=[message], instructions='', tools=[])
        result = inspect_history(provider, settings, [HistoryEntry(message, message_id=1)], raw, raw)
        row = result['entries'][0]['categories']
        self.assertEqual(row['assistant'], provider._estimate_part_tokens(text))
        self.assertEqual(row['tools'], provider._estimate_part_tokens(call) + provider._estimate_part_tokens(second))
        self.assertGreaterEqual(row['system'], provider._estimate_part_tokens(thought))

    async def test_pending_and_missing_previews_are_not_reported_as_projected_pixels(self):
        provider = build_provider(self.config, 'gemini', require_credentials=False)
        self.addAsyncCleanup(provider.aclose)
        settings = replace(self.config.default_session_settings(), model='gemini-3.8-flash')
        message = ConversationMessage(MessageRole.USER, [
            MessagePart(PartKind.IMAGE, mime_type='image/png', preview_ref='pending-preview'),
            MessagePart(PartKind.IMAGE, mime_type='image/png', text='Unavailable image')])
        raw = provider.estimate_request_tokens(settings=settings, messages=[message], instructions='', tools=[])
        result = inspect_history(provider, settings, [HistoryEntry(message)], raw, raw)
        self.assertEqual(result['images'], {'projected': 0, 'pending': 1, 'unavailable': 1, 'unsupported': 0})
