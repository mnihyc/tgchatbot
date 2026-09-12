"""A failed hosted request must not strand the chat or erase its originals.

Telegram transport and Gemini HTTP are mocked; intake, scheduling, the Gemini
adapter, configured runtime retries, persistence and the next reply are real.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx

from tests.business_helpers import BusinessTestCase
from tgchatbot.domain.models import MessageRole, ProcessVisibility
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.storage.artifacts import ArtifactStore
from tgchatbot.storage.postgres_store import message_body
from tgchatbot.transports.telegram_adapter import TelegramBotApp


class TelegramProviderRecoveryTests(BusinessTestCase):
    async def test_exhausted_malformed_call_preserves_context_and_next_keyword_recovers(self):
        await self.check_failure_recovery({'candidates': [{
            'content': {}, 'finishReason': 'MALFORMED_FUNCTION_CALL',
            'finishMessage': 'Function call is empty - no input to parse.'}]})

    async def test_exhausted_truncated_reply_preserves_context_and_next_keyword_recovers(self):
        await self.check_failure_recovery({'candidates': [{
            'content': {'role': 'model', 'parts': [{'text': 'A plausible unfinished answer'}]},
            'finishReason': 'MAX_TOKENS'}]})

    async def test_exhausted_prompt_block_preserves_context_and_next_keyword_recovers(self):
        await self.check_failure_recovery({'promptFeedback': {'blockReason': 'SAFETY'}})

    async def check_failure_recovery(self, failed_response):
        self.config = replace(self.config,
            telegram=replace(self.config.telegram, keywords=('helper',), ignore_keywords=()),
            gemini=replace(self.config.gemini, api_key='mock-gemini-key', base_url='https://provider.invalid'))
        self.runtime.config = self.config
        await self.settings(provider='gemini', model='gemini-3.8-flash',
            process_visibility=ProcessVisibility.OFF, provider_retry_count=1,
            group_reply_delay_s=0, spontaneous_reply_chance=0,
            native_web_search_mode='off', link_prefetch_mode='off')

        requests = []
        malformed = True

        def respond(request):
            requests.append(json.loads(request.content))
            if malformed:
                return httpx.Response(200, json=failed_response)
            return httpx.Response(200, json={'candidates': [{
                'content': {'role': 'model', 'parts': [{'text': 'The spare key is in the blue bag'}]},
                'finishReason': 'STOP'}]})

        provider = GeminiProvider(self.config.gemini)
        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(provider.aclose)
        self.runtime.providers = {'gemini': provider}

        app = TelegramBotApp.__new__(TelegramBotApp)
        app.config, app.runtime, app.store = self.config, self.runtime, self.store
        app.artifact_store = ArtifactStore(self.config.artifact_dir)
        app.remote_workspace = SimpleNamespace(enabled=False)
        app._chat_states = {}
        chat = SimpleNamespace(id=100, type='group')
        sent = []
        bot = SimpleNamespace(id=999, username='fixture_bot', send_chat_action=AsyncMock())

        def message(number, text='', actor=7):
            sender = SimpleNamespace(id=actor, username=f'participant{actor}',
                full_name='Participant' if actor == 7 else 'Fixture Bot', is_bot=actor == 999)
            result = SimpleNamespace(chat=chat, message_id=number, text=text, caption=None,
                from_user=sender, sender_chat=None, date=datetime(2026, 1, 2, tzinfo=timezone.utc),
                edit_date=None, entities=[], caption_entities=[], reply_to_message=None,
                photo=None, sticker=None, document=None, animation=None, video=None,
                audio=None, voice=None, video_note=None, get_bot=lambda: bot,
                reply_text=AsyncMock(), delete=AsyncMock(), edit_text=AsyncMock())
            result.edit_text.return_value = result
            return result

        async def send(**kwargs):
            sent.append(kwargs)
            return message(1000 + len(sent), kwargs['text'], actor=999)

        bot.send_message = AsyncMock(side_effect=send)
        finished = asyncio.Event()
        reply_to_candidate = app._reply_to_candidate

        async def observe_reply(candidate):
            try:
                await reply_to_candidate(candidate)
            finally:
                finished.set()

        # Observe completion without replacing the real scheduler or reply path.
        app._reply_to_candidate = observe_reply

        async def arrive(number, text, *, triggered):
            incoming = message(number, text)
            update = SimpleNamespace(effective_chat=chat, effective_message=incoming,
                effective_user=incoming.from_user)
            finished.clear()
            await app.group_message(update, SimpleNamespace(bot=bot))
            if triggered:
                await asyncio.wait_for(finished.wait(), timeout=3)
            task = app._flow_state(chat.id).reply_task
            if task is not None:
                await asyncio.wait_for(task, timeout=3)

        seed = 'The spare key is in the blue bag.'
        failed = 'helper, confirm you remember it.'
        await arrive(10, seed, triggered=False)
        self.assertEqual(requests, [])
        scope = await self.store.get_scope(self.session)
        with self.assertLogs('tgchatbot', level='WARNING'):
            await arrive(11, failed, triggered=True)
        self.assertEqual(len(requests), 2, 'The configured single retry must exhaust without an unbounded loop')
        self.assertEqual(len(sent), 1, 'Exhaustion produces the existing error notice, not a pretend answer')
        self.assertIn('Reply generation failed', sent[0]['text'])
        self.assertNotIn('empty response', sent[0]['text'])
        self.assertEqual(await self.store.get_scope(self.session), scope)
        originals = await self.store.list_canonical_messages(self.session)
        users = [row for row in originals if row.message.role == MessageRole.USER
                 and row.message.metadata.get('source') == 'telegram']
        self.assertEqual([row.message.metadata['source_message_id'] for row in users], ['10', '11'])
        self.assertTrue(all(row.message.metadata['actor_id'] == 'telegram:user:7' for row in users))
        self.assertFalse([row for row in originals if row.message.role == MessageRole.ASSISTANT])
        preserved = {row.db_id: message_body(row.message) for row in users}

        # The same warm runtime/scheduler receives the next ordinary keyword.
        malformed = False
        followup = 'helper, where did I put the spare key?'
        await arrive(12, followup, triggered=True)
        self.assertEqual(len(requests), 3)
        last_context = json.dumps(requests[-1]['contents'], ensure_ascii=False)
        for original in (seed, failed, followup):
            self.assertIn(original, last_context)
        self.assertEqual(sent[-1]['text'], 'The spare key is in the blue bag')
        self.assertTrue(all(receipt.get('reply_to_message_id') is None for receipt in sent))
        self.assertEqual(await self.store.get_scope(self.session), scope)

        # A new store sees the same originals and the one delivered answer.
        reopened = await self.new_store()
        current = await reopened.list_canonical_messages(self.session)
        by_id = {row.db_id: row for row in current}
        self.assertEqual({key: message_body(by_id[key].message) for key in preserved}, preserved)
        self.assertTrue(all(by_id[key].message.metadata['source_revision'] == 1 for key in preserved))
        self.assertEqual([row.message.metadata['source_message_id'] for row in current
                          if row.message.role == MessageRole.USER
                          and row.message.metadata.get('source') == 'telegram'], ['10', '11', '12'])
        answers = [row for row in current if row.message.role == MessageRole.ASSISTANT]
        self.assertEqual([message_body(row.message) for row in answers], ['The spare key is in the blue bag'])
        state = app._flow_state(chat.id)
        self.assertIsNone(state.latest_reply_candidate)
        self.assertTrue(state.reply_task is None or state.reply_task.done())
