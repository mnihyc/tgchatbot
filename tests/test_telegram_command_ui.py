"""Phone-sized command navigation over real settings/history and fake Telegram."""
from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.memory import MemoryService
from tgchatbot.domain.models import ConversationMessage
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.providers.openai_responses import OpenAIResponsesProvider
from tgchatbot.stickers.catalog import StickerCatalog
from tgchatbot.storage.presets import PresetStore
from tgchatbot.storage.sticker_catalog import StickerCatalogStore
from tgchatbot.tools.registry import ToolRegistry
from tgchatbot.transports.telegram_adapter import TelegramBotApp
from tgchatbot.transports.telegram_command_views import plain_text, status_view


class TelegramCommandUIWorkflows(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.config = replace(self.config,
            gemini=replace(self.config.gemini, thinking_level='high'))
        self.runtime.config = self.config
        self.provider = GeminiProvider(self.config.gemini)
        self.addAsyncCleanup(self.provider.aclose)
        self.provider.generate = AsyncMock(side_effect=AssertionError('Commands must not generate model requests'))
        self.addCleanup(self.provider.generate.assert_not_awaited)
        self.runtime.providers = {'gemini': self.provider}
        await self.settings(provider='gemini', model='gemini-3.8-flash')

        # Construction is the Telegram boundary; handlers/runtime/store are real.
        self.app = TelegramBotApp.__new__(TelegramBotApp)
        self.app.config, self.app.runtime, self.app.store = self.config, self.runtime, self.store
        self.app._chat_states = {}
        self.app.preset_store = PresetStore(self.path / 'presets')
        self.chat = SimpleNamespace(id=100, type='group')
        self.message = SimpleNamespace(reply_text=AsyncMock(), reply_document=AsyncMock())
        self.update = SimpleNamespace(effective_chat=self.chat, effective_message=self.message,
            effective_user=SimpleNamespace(id=7))

    async def command(self, name, *arguments):
        self.message.reply_text.reset_mock()
        self.message.reply_document.reset_mock()
        await getattr(self.app, name + '_command')(self.update, SimpleNamespace(args=list(arguments)))

    def received_text(self):
        call = self.message.reply_text.await_args
        self.assertIsNotNone(call, 'Expected one visible command response')
        text = call.args[0] if call.args else call.kwargs['text']
        return plain_text(text) if call.kwargs.get('parse_mode') == 'HTML' else text

    def received_document(self):
        call = self.message.reply_document.await_args
        self.assertIsNotNone(call, 'Expected a complete downloadable diagnostic')
        return call.kwargs['document'].decode('utf-8')

    async def status_services(self):
        catalog_store = StickerCatalogStore(self.store)
        await catalog_store.initialize()
        catalog = StickerCatalog(catalog_store, self.path, persona_store=self.store)
        remote = SimpleNamespace(enabled=False, _master_started=False)
        self.runtime.tool_registry = ToolRegistry(self.config, remote, catalog)
        self.runtime.memory = MemoryService(self.store, SimpleNamespace(enabled=False))

    async def test_max_reasoning_command_survives_reload_and_reset_to_default(self):
        provider = OpenAIResponsesProvider(self.config.openai)
        self.addAsyncCleanup(provider.aclose)
        self.runtime.providers['openai'] = provider
        await self.settings(provider='openai', model='gpt-6-luna')
        await self.command('param', 'reasoning_effort')
        self.assertIn('|max|', self.received_text())

        await self.command('param', 'reasoning_effort', 'max')
        restored = await (await self.new_store()).get_or_create_session(
            self.session, self.config.default_session_settings())
        self.assertEqual(restored.reasoning_effort, 'max')
        self.assertIn('max', self.received_text())
        requests = []
        def respond(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={'status': 'completed', 'output': []})
        provider._client = httpx.AsyncClient(base_url='https://fixture.invalid/', transport=httpx.MockTransport(respond))
        await provider.generate(settings=restored, messages=[ConversationMessage.user_text('Hello.')], instructions='', tools=[])
        self.assertEqual(requests[0]['reasoning']['effort'], 'max')

        await self.command('param', 'reasoning_effort', 'default')
        restored = await (await self.new_store()).get_or_create_session(
            self.session, self.config.default_session_settings())
        self.assertIsNone(restored.reasoning_effort)
        self.assertIn('configured default', self.received_text())

    async def test_focused_setting_inspect_change_and_default_survive_reload(self):
        await self.command('params', 'model')
        category = self.received_text()
        self.assertIn('thinking_level', category)
        self.assertNotIn('reasoning_effort', category)
        self.assertNotIn('compact_trigger_tokens', category)

        await self.command('param', 'thinking_level')
        details = self.received_text()
        self.assertIn('high', details)
        self.assertIn('configured default', details)
        self.assertIn('/param thinking_level <low|medium|high|default>', details)

        await self.command('param', 'thinking_level', 'low')
        confirmation = self.received_text()
        self.assertIn('low', confirmation)
        self.assertIn('Source: this chat', confirmation)
        self.assertNotIn('<low|medium|high|default>', confirmation)
        self.assertNotIn('temperature', confirmation)
        restored = await (await self.new_store()).get_or_create_session(
            self.session, self.config.default_session_settings())
        self.assertEqual(restored.thinking_level, 'low')

        await self.command('param', 'thinking_level', 'default')
        self.assertIn('high', self.received_text())
        self.assertIn('configured default', self.received_text())
        restored = await (await self.new_store()).get_or_create_session(
            self.session, self.config.default_session_settings())
        self.assertIsNone(restored.thinking_level)

        await self.command('param', 'reasoning_effort')
        self.assertIn('Unavailable', self.received_text())

    async def test_settings_navigation_does_not_load_history_or_contact_services(self):
        original = await self.store.append_message(self.session,
            ConversationMessage.user_text('An unrelated old message.'))
        # A settings request remains usable without constructing history or a catalog.
        self.runtime.tool_registry = None
        with patch.object(self.runtime, '_get_live_state',
                new=AsyncMock(side_effect=AssertionError('Settings must not load history'))) as history:
            await self.command('params')
            self.assertIn('/params context', self.received_text())
            self.assertNotIn('compact_tool_ratio_threshold', self.received_text())
            await self.command('params', 'context')
            self.assertIn('compact_trigger_tokens', self.received_text())
            await self.command('param', 'compact_keep_recent_ratio')
            self.assertIn('not a share of the target', self.received_text())
            await self.command('param', 'provider_retry_count', '3')
            await self.command('params', 'full')
            full = self.received_text()
            self.assertIn('provider_retry_count=3', full)
            self.assertIn('compact_tool_ratio_threshold=', full)
            self.message.reply_document.assert_not_awaited()
            await self.command('help', 'context')
            self.assertIn('audit-only', self.received_text())
            history.assert_not_awaited()
        remaining = await self.store.list_uncompacted_messages(self.session)
        self.assertEqual([item.db_id for item in remaining], [original.db_id])

    async def test_status_is_compact_with_requested_detail_and_complete_file(self):
        await self.status_services()
        await self.settings(compact_trigger_tokens=12000, compact_target_tokens=6000)
        await self.store.append_message(self.session, ConversationMessage.user_text('Remember the test meeting.'))
        await self.command('status')
        brief = self.received_text()
        self.assertIn('/ 12K tokens', brief)
        self.assertIn('░', brief)
        self.assertNotIn('compact_tool_ratio_threshold', brief)
        self.assertNotIn('provider_history_messages', brief)
        self.assertNotIn('Time zone:', brief)
        self.assertEqual(self.message.reply_text.await_count, 1)
        self.message.reply_document.assert_not_awaited()

        await self.command('status', 'context')
        self.assertIn('/ 12K tokens', self.received_text())
        self.assertIn('Compaction target: 6K tokens', self.received_text())
        await self.command('status', 'tools')
        self.assertIn('Remote workspace: not configured', self.received_text())
        self.assertIn('memory_search', self.received_text())
        self.assertIn('user_profile_fetch', self.received_text())
        self.assertNotIn('shell_exec', self.received_text())

        failure = 'Synthetic worker rejection: <missing & retryable>'
        self.runtime.memory.worker = SimpleNamespace(last_error=failure)
        await self.command('status')
        self.assertIn('Worker error', self.received_text())
        self.assertIn('/status memory', self.received_text())
        self.assertNotIn(failure, self.received_text())
        await self.command('status', 'memory')
        self.assertIn(failure, self.received_text())

        await self.command('status', 'full')
        full = self.received_text()
        self.assertIn('estimated_request_tokens=', full)
        self.assertIn('compact_trigger_tokens=12000', full)
        self.assertIn('compact_tool_ratio_threshold=', full)
        self.assertIn(failure, full)
        self.assertIn(self.config.default_metadata_timezone, full)
        self.message.reply_document.assert_not_awaited()

    async def test_prompt_is_literal_and_shown_only_when_requested(self):
        prompt = '  Keep <b>these literal tags</b> & "quoted" text.\nDo not transform **source**.  '
        await self.settings(system_prompt=prompt)
        await self.command('prompt')
        self.assertIn('/prompt show', self.received_text())
        self.assertNotIn('these literal tags', self.received_text())

        await self.command('prompt', 'show')
        self.assertIn(prompt, self.received_text())
        sent = self.message.reply_text.await_args
        self.assertEqual(sent.kwargs['parse_mode'], 'HTML')
        self.assertNotIn('<b>these literal tags</b>', sent.args[0])
        self.message.reply_document.assert_not_awaited()
        self.assertEqual((await self.settings()).system_prompt, prompt)

    async def test_long_emoji_prompt_attaches_full_source_without_truncation(self):
        # Shorter than 4096 Python characters, but over Telegram's UTF-16 limit.
        prompt = '  Fixture start\n' + '😀' * 2100 + '\n<b>literal ending</b>  \n'
        self.assertLess(len(prompt), 4096)
        self.assertGreater(len(prompt.encode('utf-16-le')) // 2, 4096)
        await self.settings(system_prompt=prompt)
        await self.command('prompt', 'show')
        self.assertIn(prompt, self.received_document())
        self.assertNotIn('[truncated]', self.received_document())
        self.message.reply_text.assert_not_awaited()
        self.assertEqual((await self.settings()).system_prompt, prompt)

    async def test_readable_settings_do_not_bypass_existing_advanced_permission(self):
        self.app.config = replace(self.config,
            telegram=replace(self.config.telegram, control_uids=('8',)))
        await self.command('params', 'model')
        self.assertIn('thinking_level', self.received_text())
        self.assertIn('trusted users', self.received_text())
        await self.command('param', 'thinking_level', 'low')
        self.assertIn('not allowed', self.received_text())
        self.assertIsNone((await self.settings()).thinking_level)
        # Existing permission behavior also covers focused advanced inspection.
        await self.command('param', 'thinking_level')
        self.assertIn('not allowed', self.received_text())

    async def test_invalid_choices_explain_rejection_without_changing_settings(self):
        before = await self.settings()
        for name, value, notice in (('provider', 'missing<&>', 'Unknown provider'),
                                    ('mode', 'missing<&>', 'Invalid option')):
            with self.subTest(command=name):
                await self.command(name, value)
                self.assertIn(notice, self.received_text())
                self.assertIn(value, self.received_text())
                self.assertNotIn(value, self.message.reply_text.await_args.args[0])
                self.assertEqual(await self.settings(), before)


class StatusPresentationTests(unittest.TestCase):
    def test_context_usage_shows_overflow_without_claiming_compaction_progress(self):
        status = {'model':'fixture <model>', 'provider':'configured',
                  'estimated_request_tokens':1200000, 'compact_trigger_tokens':800000}
        rendered = status_view(status, {'compacting':True, 'reply_running':True})
        visible = plain_text(rendered)
        self.assertIn('Compacting context', visible)
        self.assertNotIn('Replying', visible)
        self.assertIn('Context ≈ 1.2M / 800K tokens', visible)
        self.assertIn('150% · above ceiling', visible)
        self.assertNotIn('complete', visible)
        self.assertIn('fixture <model>', visible)
        self.assertNotIn('fixture <model>', rendered)
        for used, ceiling in ((0,800000), (None,800000), (1000,None), (1000,0)):
            with self.subTest(used=used, ceiling=ceiling):
                visible = plain_text(status_view({**status, 'estimated_request_tokens':used,
                    'compact_trigger_tokens':ceiling}, {}))
                self.assertNotIn('above ceiling', visible)
                if used is None or not ceiling:
                    self.assertNotIn('%', visible, 'No invented fraction when the budget is unavailable')

    def test_memory_separates_active_work_from_old_completed_jobs_and_keeps_failures(self):
        status = {'semantic_enabled':True, 'scope':{'generation':'private-generation', 'context_id':'private-context'},
            'memory_jobs':[{'kind':'memory_profile','status':'pending','count':2},
                           {'kind':'memory_profile','status':'failed','count':1},
                           {'kind':'memory_embed','status':'done','count':123456}],
            'memory_last_error':'<temporary & retryable>'}
        visible = plain_text(status_view(status, {}, 'memory'))
        self.assertIn('Profiles: 2 queued · 1 failed', visible)
        self.assertNotIn('123456', visible)
        self.assertNotIn('123,456', visible)
        self.assertNotIn('private-generation', visible)
        self.assertNotIn('private-context', visible)
        self.assertIn('<temporary & retryable>', visible)
        self.assertIn('/status full', visible)
        visible = plain_text(status_view({**status, 'memory_jobs':[]}, {}, 'memory'))
        self.assertIn('Background queue: empty', visible)
