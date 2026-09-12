"""Requested reasoning allowances reach the selected API without hidden ceilings."""
from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from tgchatbot.config import load_config
from tgchatbot.domain.models import ConversationMessage, SessionSettings
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.transports.telegram_adapter import TelegramBotApp
from tests.business_helpers import BusinessTestCase


class ThinkingControlWorkflows(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='thinking-fixture-', dir=Path(__file__).parent)
        self.addCleanup(directory.cleanup)
        with patch.dict(os.environ, {'APP_DATA_DIR': directory.name, 'GEMINI_API_KEY': 'synthetic-key',
                'GEMINI_THINKING_BUDGET': '131072'}, clear=True):
            self.config = load_config(require_telegram=False)
        self.requests = []
        self.status = 200

    async def generate(self, settings):
        def response(request):
            self.requests.append(json.loads(request.content))
            if self.status != 200:
                return httpx.Response(self.status, json={'error': {'message': 'Synthetic model capacity rejection'}})
            return httpx.Response(200, json={'candidates': [{'content': {'parts': [{'text': 'Answer'}]}}]})
        provider = GeminiProvider(self.config.gemini)
        async with httpx.AsyncClient(transport=httpx.MockTransport(response)) as client:
            provider._client = client
            return await provider.generate(settings=settings, messages=[ConversationMessage.user_text('Question')],
                instructions='Answer the question.', tools=[])

    async def test_larger_configured_allowance_reaches_each_supported_family_unchanged(self):
        self.assertEqual(self.config.gemini.thinking_budget, 131072)
        for model in ('gemini-2.5-flash', 'gemini-2.5-flash-lite', 'gemini-2.5-pro', 'gemini-3.8-flash'):
            with self.subTest(model=model):
                result = await self.generate(SessionSettings(provider='gemini', model=model))
                self.assertEqual(result.final_text, 'Answer')
                self.assertEqual(self.requests[-1]['generationConfig']['thinkingConfig']['thinkingBudget'], 131072)

    async def test_session_allowance_preserves_auto_disabled_and_positive_values(self):
        for value in (-1, 0, 1, 65536):
            with self.subTest(value=value):
                settings = SessionSettings(provider='gemini', model='gemini-2.5-pro', thinking_budget=value)
                await self.generate(settings)
                self.assertEqual(self.requests[-1]['generationConfig']['thinkingConfig']['thinkingBudget'], value)

    async def test_level_precedence_and_unsupported_family_are_preserved(self):
        settings = SessionSettings(provider='gemini', model='gemini-3.8-flash',
            thinking_budget=65536, thinking_level='high')
        await self.generate(settings)
        self.assertEqual(self.requests[-1]['generationConfig']['thinkingConfig'], {'thinkingLevel': 'high'})
        await self.generate(replace(settings, model='gemini-2.0-flash'))
        self.assertNotIn('thinkingConfig', self.requests[-1]['generationConfig'])

    async def test_flash_model_thinking_levels_preserve_supported_wire_values(self):
        for model, level in (
            ('gemini-3.7-flash', 'low'),
            ('gemini-3.8-flash', 'low'),
            ('gemini-3.6-flash', 'minimal'),
        ):
            with self.subTest(model=model, level=level):
                result = await self.generate(SessionSettings(provider='gemini', model=model,
                    thinking_budget=65536, thinking_level=level))
                self.assertEqual(result.final_text, 'Answer')
                self.assertEqual(self.requests[-1]['generationConfig']['thinkingConfig'],
                    {'thinkingLevel': level})

    async def test_saved_minimal_is_not_sent_to_flash_models_that_reject_it(self):
        # A model switch may leave a level accepted by the previous model in
        # saved settings. Do not forward that unsupported value to the API.
        for model in ('gemini-3.7-flash', 'gemini-3.8-flash'):
            with self.subTest(model=model):
                await self.generate(SessionSettings(provider='gemini', model=model,
                    thinking_level='minimal'))
                thinking = self.requests[-1]['generationConfig'].get('thinkingConfig', {})
                self.assertNotIn('thinkingLevel', thinking)

    async def test_api_capacity_error_is_reported_without_silently_changing_allowance(self):
        self.status = 400
        with self.assertRaises(httpx.HTTPStatusError):
            await self.generate(SessionSettings(provider='gemini', model='gemini-2.5-pro', thinking_budget=65536))
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0]['generationConfig']['thinkingConfig']['thinkingBudget'], 65536)


class ThinkingCommandWorkflows(BusinessTestCase):
    async def test_thinking_level_command_matches_flash_model_choices(self):
        self.runtime.providers['gemini'] = GeminiProvider(self.config.gemini)
        app = TelegramBotApp.__new__(TelegramBotApp)
        app.config, app.runtime, app.store = self.config, self.runtime, self.store
        message = SimpleNamespace(reply_text=AsyncMock())
        update = SimpleNamespace(effective_chat=SimpleNamespace(id=100),
            effective_message=message, effective_user=SimpleNamespace(id=7))
        for model, supports_minimal in (
            ('gemini-3.7-flash', False),
            ('gemini-3.8-flash', False),
            ('gemini-3.6-flash', True),
        ):
            with self.subTest(model=model):
                await self.settings(provider='gemini', model=model, thinking_level='high')
                await app.param_command(update, SimpleNamespace(args=[]))
                choices = next(line for line in message.reply_text.call_args.args[0].splitlines()
                    if line.startswith('thinking_level '))
                self.assertEqual(choices, 'thinking_level <' +
                    ('minimal|' if supports_minimal else '') + 'low|medium|high|default>')

                await app.param_command(update, SimpleNamespace(args=['thinking_level', 'minimal']))
                if not supports_minimal:
                    self.assertIn('Invalid thinking_level', message.reply_text.call_args.args[0])
                self.assertEqual((await self.settings()).thinking_level,
                    'minimal' if supports_minimal else 'high')

                await app.param_command(update, SimpleNamespace(args=['thinking_level', 'low']))
                restored = await (await self.new_store()).get_or_create_session(self.session,
                    self.config.default_session_settings())
                self.assertEqual(restored.thinking_level, 'low')

    async def test_requested_budget_survives_command_save_and_new_database_handle(self):
        await self.settings(provider='gemini', model='gemini-3.8-flash', thinking_level='high')
        self.runtime.providers['gemini'] = GeminiProvider(self.config.gemini)
        app = TelegramBotApp.__new__(TelegramBotApp)
        app.config, app.runtime, app.store = self.config, self.runtime, self.store
        chat = SimpleNamespace(id=100)
        message = SimpleNamespace(reply_text=AsyncMock())
        update = SimpleNamespace(effective_chat=chat, effective_message=message, effective_user=SimpleNamespace(id=7))
        for value in (131072, -1, 0):
            with self.subTest(value=value):
                await app.param_command(update, SimpleNamespace(args=['thinking_budget', str(value)]))
                restored = await (await self.new_store()).get_or_create_session(self.session,
                    self.config.default_session_settings())
                self.assertEqual(restored.thinking_budget, value)
                self.assertIsNone(restored.thinking_level)
        await app.param_command(update, SimpleNamespace(args=['thinking_budget', '-2']))
        self.assertIn('Invalid thinking_budget', message.reply_text.call_args.args[0])
        self.assertEqual((await self.settings()).thinking_budget, 0)
