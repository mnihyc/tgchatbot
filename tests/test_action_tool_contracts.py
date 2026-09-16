"""Model-visible action outcomes survive transport, restart and compaction."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from PIL import Image
from telegram.error import TimedOut

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import ChatMode, ConversationMessage, StickerMode
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.stickers.catalog import StickerCatalog
from tgchatbot.stickers.config import StickerConfig
from tgchatbot.storage.postgres_store import message_body
from tgchatbot.storage.sticker_delivery import StickerDeliveryStore
from tgchatbot.tools.base import ToolContext
from tgchatbot.tools.sticker_send import StickerQueryTool, StickerSendSelectedTool
from tgchatbot.transports.sticker_delivery import send_sticker
from tgchatbot.transports.telegram_adapter import TelegramBotApp


class ActionToolContractTests(BusinessTestCase):
    async def sticker_turn(self, timing, *, count=2, uncertain=False):
        asset = self.path / 'greeting.webp'
        Image.new('RGB', (4, 4), 'blue').save(asset, format='WEBP')
        digest = hashlib.sha256(asset.read_bytes()).hexdigest()
        entry = SimpleNamespace(absolute_path=asset, emoji=None, summary='A friendly wave',
            sticker_id='sha256:' + digest, agent_id='sid:7',
            asset=SimpleNamespace(content_hash=digest, card={'caption': 'Hi', 'action': 'Waves one hand'}))
        catalog = SimpleNamespace(config=StickerConfig(), aensure_loaded=AsyncMock(),
            aget_available=AsyncMock(return_value=entry), aagent_sticker_id=AsyncMock(return_value='sid:7'),
            agent_sticker_id=lambda value: 'sid:7', stats=lambda: {'stickers': 1})
        tool = StickerSendSelectedTool(catalog)
        self.tools.sticker_catalog = catalog
        self.tools.list_tools.return_value = [tool.spec]
        replies = [{'candidates': [{'finishReason': 'STOP', 'content': {'role': 'model', 'parts': [{
            'functionCall': {'name': tool.spec.name, 'id': f'call_{number}',
                'args': {'selected_sticker_id': 'sid:7', 'delivery_timing': timing}},
            'thoughtSignature': 'synthetic-signature'}]}}]} for number in range(count)]
        replies.append({'candidates': [{'finishReason': 'STOP', 'content': {'role': 'model',
            'parts': [{'text': 'Here is my response.'}]}}]})
        wire = []
        def respond(request):
            wire.append(json.loads(request.content))
            return httpx.Response(200, json=replies.pop(0))
        provider = GeminiProvider(replace(self.config.gemini, api_key='synthetic-key'))
        await provider.aclose()
        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(provider.aclose)
        settings = await self.settings(provider='gemini', model='gemini-3.8-flash',
            mode=ChatMode.ASSIST, sticker_mode=StickerMode.AUTO, max_interaction_rounds=3,
            compact_trigger_tokens=100000, native_web_search_mode='off')
        deliveries = StickerDeliveryStore(self.store)
        await deliveries.initialize()
        bot = SimpleNamespace(send_sticker=AsyncMock(side_effect=TimedOut() if uncertain else None,
            return_value=SimpleNamespace(message_id=101)))
        runtime = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools,
            providers={'gemini': provider}, preview_cache=self.preview_cache, sticker_delivery=deliveries)
        async def deliver(sticker):
            receipt = await send_sticker(bot, chat_id=100, sticker=sticker, deliveries=deliveries)
            await runtime.record_tool_observation(session_id=self.session, name='sticker_send',
                phase='delivery', payload=receipt)
        async def emit(event):
            if event.kind == 'sticker':
                await deliver(TelegramBotApp._sticker_from_event(event))
        turn = await runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Send a suitable greeting.'), emit=emit)
        def outputs(contents):
            return [part['functionResponse']['response']['result'] for content in contents
                for part in content.get('parts', []) if 'functionResponse' in part
                and part['functionResponse']['name'] == tool.spec.name]
        visible = outputs(wire[-1]['contents'])
        expected = 'unknown' if uncertain else 'queued' if timing == 'after_final' else 'sent'
        self.assertEqual([result['status'] for result in visible], [expected] * count)
        if uncertain:
            self.assertFalse(visible[0]['ok'])
            self.assertIn('may have been sent', visible[0]['error'])
        else:
            self.assertTrue(all(result['ok'] for result in visible))
        self.assertEqual(bot.send_sticker.await_count, 0 if timing == 'after_final' else count)
        if timing == 'after_final':
            for sticker in turn.stickers:
                await deliver(sticker)
        self.assertEqual(bot.send_sticker.await_count, count)
        self.assertEqual(len({sticker.delivery_operation_id for sticker in turn.stickers}), count)
        for sticker in turn.stickers:
            await send_sticker(bot, chat_id=100, sticker=sticker, deliveries=deliveries)
        self.assertEqual(bot.send_sticker.await_count, count, 'Replaying one operation is not a new action call')
        reopened = await self.new_store()
        cold = AgentRuntime(config=self.config, store=reopened, tool_registry=self.tools,
            providers={'gemini': provider}, sticker_delivery=deliveries)
        state = await cold._get_live_state(self.session)
        history = cold._build_provider_history(state, settings=settings, provider_name='gemini')
        contents = [item for message in history for item in provider._message_to_contents(message)]
        self.assertEqual(outputs(contents), visible, 'Restart must not rewrite earlier queued/sent function results')
        compacted = '\n'.join(message_body(message) for message in cold._normalize_compaction_messages(
            [row.message for row in state.raw_messages]))
        self.assertIn('sid:7', compacted)
        self.assertIn('status=' + expected, compacted)
        self.assertIn('Sticker delivery receipt', compacted)

    async def test_two_immediate_calls_are_two_sends_with_confirmed_model_results(self):
        await self.sticker_turn('before_final')

    async def test_two_deferred_calls_execute_once_each_after_the_turn(self):
        await self.sticker_turn('after_final')

    async def test_uncertain_delivery_keeps_identity_and_outcome_through_restart_and_compaction(self):
        await self.sticker_turn('send_now', count=1, uncertain=True)

    async def test_saved_preference_is_acknowledged_even_when_search_fails_and_survives_restart(self):
        await self.settings()
        catalog = StickerCatalog(None, self.path, persona_store=self.store)
        catalog.achoose = AsyncMock(side_effect=RuntimeError('Synthetic search unavailable'))
        tool = StickerQueryTool(catalog)
        context = ToolContext(self.session, 'Participant', scope=await self.store.get_scope(self.session))
        with self.assertLogs('tgchatbot.tools.sticker_send', level='ERROR'):
            failed = await tool.run({'intent_core': 'Greet warmly', 'persona': {
                'visual_identity': {'preferred_pack': 'Friendly drawings'},
                'affect_profile': {'default_tone': 'warm'}}}, context)
        self.assertFalse(failed.output['ok'])
        self.assertEqual(failed.output['persona_update'], 'saved')
        stored_failure = await self.runtime.record_tool_observation(session_id=self.session,
            name='sticker_query', phase='result', payload={'call_id': 'failed-search', 'output': failed.output})
        self.assertIn('persona_update=saved', message_body(stored_failure.message))
        reopened = await self.new_store()
        restarted = StickerCatalog(None, self.path, persona_store=reopened)
        restarted.achoose = AsyncMock(return_value=[])
        inherited = await StickerQueryTool(restarted).run({'intent_core': 'Greet again'}, context)
        self.assertEqual(inherited.output['status'], 'no_candidates')
        self.assertEqual(inherited.output['persona']['visual_identity']['preferred_pack'], 'Friendly drawings')
        self.assertNotIn('persona_update', inherited.output)
        stored = await self.runtime.record_tool_observation(session_id=self.session,
            name='sticker_query', phase='result', payload={'call_id': 'inherited-search', 'output': inherited.output})
        original = (await reopened.read_messages(self.session, [stored.db_id]))[0]
        compacted = '\n'.join(message_body(message) for message in self.runtime._normalize_compaction_messages(
            [original.message]))
        self.assertIn('"preferred_pack": "Friendly drawings"', compacted)
        self.assertIn('"default_tone": "warm"', compacted)
        self.assertEqual(original.message.metadata['tool_payload']['output'], inherited.output)
        restarted.achoose.side_effect = RuntimeError('Synthetic search unavailable')
        with self.assertLogs('tgchatbot.tools.sticker_send', level='ERROR'):
            cleared = await StickerQueryTool(restarted).run(
                {'intent_core': 'Greet', 'persona_mode': 'clear_session_persona'}, context)
        self.assertEqual(cleared.output['persona_update'], 'cleared')
        self.assertIsNone(await reopened.get_sticker_persona(self.session))
