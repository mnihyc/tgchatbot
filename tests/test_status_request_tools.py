"""Status describes the tool contract of the next real runtime request."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

from tests.business_helpers import BusinessTestCase, ScriptedProvider
from tgchatbot.core.memory import MemoryService
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import ChatMode, ConversationMessage, ProviderResponse, StickerMode
from tgchatbot.operational import MemoryConfig
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.stickers.catalog import StickerCatalog
from tgchatbot.storage.sticker_catalog import StickerCatalogStore
from tgchatbot.tools.registry import ToolRegistry


class EstimatedProvider(ScriptedProvider):
    def __init__(self):
        super().__init__(responses=[ProviderResponse(final_text='A short fixture answer.')])
        self.estimates = []

    def estimate_request_tokens(self, **kwargs):
        result = super().estimate_request_tokens(**kwargs)
        self.estimates.append((kwargs['tools'], result))
        return result


class StatusRequestToolsTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.catalog_store = StickerCatalogStore(self.store)
        await self.catalog_store.initialize()
        self.catalog = StickerCatalog(self.catalog_store, self.path, persona_store=self.store)

    async def publish_sticker(self, aliases=None):
        revision = await self.catalog_store.begin_revision(source_root=str(self.path), recipe={})
        await self.catalog_store.stage_asset(revision, asset_id='fixture:hello', content_hash='hello',
            aliases=aliases or [{'path': 'greeting/hello.webp', 'pack': 'greeting'}], media={},
            generated_card={}, corrections={}, card={}, provenance={}, state='ready')
        await self.catalog_store.activate(revision)

    async def compare_request(self, *, mode, with_memory, remote_enabled=True,
                              sticker_mode=StickerMode.AUTO):
        await self.settings(mode=mode, sticker_mode=sticker_mode)
        provider = EstimatedProvider()
        memory = MemoryService(self.store, SimpleNamespace(enabled=False), config=MemoryConfig()) if with_memory else None
        remote = SimpleNamespace(enabled=remote_enabled, _master_started=False)
        registry = ToolRegistry(self.config, remote, self.catalog)
        runtime = AgentRuntime(config=self.config, store=self.store, tool_registry=registry,
            providers={'openai': provider}, memory=memory, preview_cache=self.preview_cache)
        trigger = await runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text('A short fixture question.'))
        status = await runtime.describe_session(self.session)
        estimated_tools, estimate = provider.estimates[-1]
        settings_values = await runtime.describe_settings(self.session)
        self.assertEqual(settings_values, {name: status[name] for name in settings_values})
        await runtime.run_turn_from_stored(session_id=self.session, user_display_name='Participant',
            trigger_message_id=trigger.db_id)
        requested_tools = provider.requests[0]['tools']
        # Compare the complete public tool contract, not runner object identity.
        contract = lambda tools: [(tool.name, tool.description, tool.parameters_schema) for tool in tools]
        self.assertEqual(contract(estimated_tools), contract(requested_tools))
        self.assertEqual(status['estimated_request_tokens'], estimate.total_tokens)
        names = {tool.name for tool in requested_tools}
        self.assertEqual(set(status['available_tools']), names)
        self.assertEqual('memory_search' in names, with_memory)
        self.assertEqual('memory_read' in names, with_memory)
        self.assertEqual('user_profile_fetch' in names, with_memory)
        self.assertEqual('shell_exec' in names, remote_enabled and mode != ChatMode.CHAT)
        self.assertEqual('read_doc' in names, remote_enabled and mode != ChatMode.CHAT)
        self.assertEqual('sticker_query' in names,
            mode != ChatMode.CHAT and sticker_mode == StickerMode.AUTO and self.catalog.stats()['stickers'] > 0)
        return status

    async def test_status_estimate_matches_requests_across_modes_memory_and_catalog_publication(self):
        for available in (False, True):
            if available:
                await self.publish_sticker()
            for mode in ChatMode:
                for with_memory in (False, True):
                    with self.subTest(catalog=available, mode=mode.value, memory=with_memory):
                        self.session = f'telegram:fixture-{available}-{mode.value}-{with_memory}'
                        status = await self.compare_request(mode=mode, with_memory=with_memory)
                        self.assertEqual(status['sticker_index_count'], int(available))
                        self.assertEqual(status['sticker_pack_count'], int(available))
                        self.assertTrue(status['sticker_index_loaded'])

    async def test_disabled_remote_and_stickers_are_absent_from_status_and_reply(self):
        await self.publish_sticker()
        await self.compare_request(mode=ChatMode.ASSIST, with_memory=True,
            remote_enabled=False, sticker_mode=StickerMode.OFF)

    async def test_catalog_status_counts_explicit_pack_identity_after_restart(self):
        await self.publish_sticker(aliases=[
            {'path': 'series/first/hello.webp', 'pack': 'greeting'},
            {'path': 'series/second/hello.webp', 'pack': 'greeting'},
            {'path': 'another/path/hello.webp', 'pack': 'kindness'},
            {'path': 'ungrouped.webp', 'pack': ''},
        ])
        # A fresh runtime reconstructs the public counts from the catalog revision.
        self.catalog = StickerCatalog(self.catalog_store, self.path, persona_store=self.store)
        status = await self.compare_request(mode=ChatMode.ASSIST, with_memory=False)
        self.assertEqual(status['sticker_index_count'], 1)
        self.assertEqual(status['sticker_pack_count'], 2)
        self.assertTrue(status['sticker_index_loaded'])

    async def test_settings_inspection_needs_no_history_catalog_or_request_estimation(self):
        await self.settings(provider='gemini', model='gemini-3.8-flash', thinking_level='high',
            temperature=0.75, compact_trigger_tokens=12000, compact_target_tokens=6000)
        original = await self.store.append_message(self.session,
            ConversationMessage.user_text('A stored message does not need loading to inspect settings.'))
        provider = GeminiProvider(self.config.gemini)
        self.addAsyncCleanup(provider.aclose)
        runtime = AgentRuntime(config=self.config, store=self.store, tool_registry=None,
            providers={'gemini': provider})
        runtime._get_live_state = AsyncMock(side_effect=AssertionError('Settings must not read history'))
        values = await runtime.describe_settings(self.session)
        self.assertEqual(values['thinking_level'], 'high')
        self.assertEqual(values['thinking_level_source'], 'session')
        self.assertEqual(values['temperature'], '0.75')
        self.assertEqual(values['compact_trigger_tokens'], 12000)
        self.assertEqual(values['compact_target_tokens'], 6000)
        self.assertEqual(values['metadata_timezone'], self.config.default_metadata_timezone)
        self.assertEqual(values['metadata_timezone_source'], 'environment')
        self.assertEqual(values['metadata_injection_mode_source'], 'session')
        self.assertEqual(values['tool_history_mode_source'], 'session')
        runtime._get_live_state.assert_not_awaited()
        self.assertEqual([item.db_id for item in await self.store.list_uncompacted_messages(self.session)],
            [original.db_id])
