"""Live intake and reply scheduling share the successful request's boundary."""
from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx

from tests.business_helpers import BusinessTestCase
from tgchatbot.domain.models import ChatMode, MessagePart, PartKind, ProcessVisibility, ToolResult
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.storage.artifacts import ArtifactStore
from tgchatbot.storage.postgres_store import message_body
from tgchatbot.transports.telegram_adapter import TelegramBotApp


class TelegramReplyCoverageTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.config = replace(self.config,
            telegram=replace(self.config.telegram, keywords=('helper',), ignore_keywords=()),
            gemini=replace(self.config.gemini, api_key='synthetic-key', base_url='https://provider.invalid'))
        self.runtime.config = self.config
        await self.settings(provider='gemini', model='gemini-3.8-flash', mode=ChatMode.ASSIST,
            process_visibility=ProcessVisibility.OFF, provider_retry_count=0,
            group_reply_delay_s=0, private_reply_delay_s=0, spontaneous_reply_chance=0,
            native_web_search_mode='off', link_prefetch_mode='off', compact_trigger_tokens=100000)
        self.requests, self.sent = [], []
        self.request_hook = self.tool_hook = None
        self.evidence = True
        self.fail_request = None

        async def respond(request):
            payload = json.loads(request.content)
            self.requests.append(payload)
            step = len(self.requests)
            if self.request_hook:
                await self.request_hook(step)
            if step == self.fail_request:
                return httpx.Response(503, json={'error': {'message': 'Synthetic outage'}})
            if step == 1:
                parts = [{'functionCall': {'name': 'shell_exec', 'id': 'check', 'args': {}},
                    'thoughtSignature': 'synthetic-signature'}]
            else:
                parts = [{'text': f'Answer {step}'}]
            return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP',
                'content': {'role': 'model', 'parts': parts}}]})

        provider = GeminiProvider(self.config.gemini)
        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(provider.aclose)
        self.runtime.providers = {'gemini': provider}

        async def tool(*_args):
            if self.tool_hook:
                await self.tool_hook()
            return ToolResult('', 'shell_exec', {'ok': True, 'stdout': 'Checked.'},
                evidence_parts=[MessagePart(PartKind.TEXT, text='Supporting tool evidence.')] if self.evidence else [])
        self.tools.runner.run.side_effect = tool

        self.app = TelegramBotApp.__new__(TelegramBotApp)
        self.app.config, self.app.runtime, self.app.store = self.config, self.runtime, self.store
        self.app.artifact_store = ArtifactStore(self.config.artifact_dir)
        self.app.remote_workspace = SimpleNamespace(enabled=False)
        self.app._chat_states = {}
        self.chat = SimpleNamespace(id=100, type='group')
        self.bot = SimpleNamespace(id=999, username='fixture_bot', send_chat_action=AsyncMock())

        async def send(**kwargs):
            self.sent.append(kwargs['text'])
            return self.message(1000 + len(self.sent), kwargs['text'], actor=999)
        self.bot.send_message = AsyncMock(side_effect=send)
        self.promoted, self.promotion_gates = {}, {}
        self.promotion_tasks = set()
        promote = self.app._promote_candidate_after_delay

        async def observe_promotion(chat_id, token, candidate, delay_s):
            task = asyncio.current_task()
            self.promotion_tasks.add(task)
            try:
                number = candidate.source_message.message_id
                if number in self.promotion_gates:
                    await self.promotion_gates[number].wait()
                await promote(chat_id, token, candidate, delay_s)
                self.promoted[number].set()
            finally:
                self.promotion_tasks.discard(task)
        self.app._promote_candidate_after_delay = observe_promotion
        self.addAsyncCleanup(self.stop_tasks)

    async def stop_tasks(self):
        await self.app._cancel_pending_reply(self.chat.id)
        tasks = list(self.promotion_tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def message(self, number, text, *, actor=7):
        sender = SimpleNamespace(id=actor, username=f'participant{actor}',
            full_name='Participant', is_bot=actor == 999)
        message = SimpleNamespace(chat=self.chat, message_id=number, text=text, caption=None,
            from_user=sender, sender_chat=None, date=datetime(2026, 1, 2, tzinfo=timezone.utc),
            edit_date=None, entities=[], caption_entities=[], reply_to_message=None,
            photo=None, sticker=None, document=None, animation=None, video=None,
            audio=None, voice=None, video_note=None, get_bot=lambda: self.bot,
            reply_text=AsyncMock(), delete=AsyncMock(), edit_text=AsyncMock())
        message.edit_text.return_value = message
        return message

    async def arrive(self, number, text, *, wait_promotion=True):
        message = self.message(number, text)
        update = SimpleNamespace(effective_chat=self.chat, effective_message=message,
            effective_user=message.from_user)
        self.promoted[number] = asyncio.Event()
        if self.chat.type == 'private':
            await self.app.private_message(update, SimpleNamespace(bot=self.bot))
        else:
            await self.app.group_message(update, SimpleNamespace(bot=self.bot))
        if wait_promotion:
            await asyncio.wait_for(self.promoted[number].wait(), timeout=3)

    async def finish_worker(self):
        task = self.app._flow_state(self.chat.id).reply_task
        if task is not None:
            await asyncio.wait_for(asyncio.shield(task), timeout=5)

    async def assert_coalesced(self, *, delayed=False):
        if delayed:
            self.promotion_gates[11] = asyncio.Event()
        async def incoming():
            await self.arrive(11, 'helper, the additional question is about blue paint.', wait_promotion=not delayed)
        self.tool_hook = incoming
        await self.arrive(10, 'helper, check the first question.')
        await self.finish_worker()
        if delayed:
            self.promotion_gates[11].set()
            await asyncio.wait_for(self.promoted[11].wait(), timeout=3)
            await self.finish_worker()
        self.assertIn('additional question', json.dumps(self.requests[1]['contents']))
        self.assertEqual(len(self.requests), 2, 'One tool request and one final request; the included keyword is already handled')
        self.assertEqual(self.sent, ['Answer 2'])
        self.assertIsNone(self.app._flow_state(self.chat.id).latest_reply_candidate)
        originals = await (await self.new_store()).list_canonical_messages(self.session)
        self.assertEqual([row.message.metadata['source_message_id'] for row in originals
            if row.message.metadata.get('actor_id') == 'telegram:user:7'], ['10', '11'])
        self.assertEqual([message_body(row.message) for row in originals
            if row.message.metadata.get('actor_id') == 'telegram:user:999'], ['Answer 2'])

    async def test_group_keyword_in_continuation_does_not_start_another_turn(self):
        await self.assert_coalesced()

    async def test_private_message_in_continuation_does_not_start_another_turn(self):
        self.chat.type = 'private'
        await self.assert_coalesced()

    async def test_delayed_promotion_after_delivery_does_not_reply_again(self):
        await self.assert_coalesced(delayed=True)

    async def test_redelivery_of_absorbed_message_does_not_start_another_turn(self):
        await self.assert_coalesced()
        await self.arrive(11, 'helper, the additional question is about blue paint.')
        await self.finish_worker()
        self.assertEqual((len(self.requests), self.sent), (2, ['Answer 2']))

    async def test_message_after_final_request_is_not_swallowed(self):
        async def incoming():
            await self.arrive(11, 'helper, the additional question is about blue paint.')
        async def request(step):
            if step == 2:
                await self.arrive(12, 'helper, this later question is about green paint.')
        self.tool_hook, self.request_hook = incoming, request
        await self.arrive(10, 'helper, check the first question.')
        await self.finish_worker()
        self.assertIn('additional question', json.dumps(self.requests[1]['contents']))
        self.assertNotIn('later question', json.dumps(self.requests[1]['contents']))
        self.assertIn('later question', json.dumps(self.requests[2]['contents']))
        self.assertEqual((len(self.requests), self.sent), (3, ['Answer 2', 'Answer 3']))

    async def test_ingested_but_not_admitted_message_keeps_its_own_turn(self):
        self.evidence = False
        async def incoming():
            await self.arrive(11, 'helper, the additional question is about blue paint.')
        self.tool_hook = incoming
        await self.arrive(10, 'helper, check the first question.')
        await self.finish_worker()
        self.assertNotIn('additional question', json.dumps(self.requests[1]['contents']))
        self.assertIn('additional question', json.dumps(self.requests[2]['contents']))
        self.assertEqual((len(self.requests), self.sent), (3, ['Answer 2', 'Answer 3']))

    async def test_message_arriving_during_delivery_keeps_its_own_turn(self):
        async def incoming():
            await self.arrive(11, 'helper, the additional question is about blue paint.')
        self.tool_hook = incoming
        send = self.bot.send_message.side_effect
        async def send_with_intake(**kwargs):
            if kwargs['text'] == 'Answer 2':
                await self.arrive(12, 'helper, this later question is about green paint.')
            return await send(**kwargs)
        self.bot.send_message.side_effect = send_with_intake
        await self.arrive(10, 'helper, check the first question.')
        await self.finish_worker()
        self.assertNotIn('later question', json.dumps(self.requests[1]['contents']))
        self.assertIn('later question', json.dumps(self.requests[2]['contents']))
        self.assertEqual((len(self.requests), self.sent), (3, ['Answer 2', 'Answer 3']))

    async def test_failed_delivery_keeps_the_included_candidate_pending(self):
        async def incoming():
            await self.arrive(11, 'helper, the additional question is about blue paint.')
        self.tool_hook = incoming
        send = self.bot.send_message.side_effect
        async def fail_final(**kwargs):
            if kwargs['text'] == 'Answer 2':
                raise TimeoutError('Synthetic Telegram outage')
            return await send(**kwargs)
        self.bot.send_message.side_effect = fail_final
        with self.assertLogs('tgchatbot', level='WARNING'):
            await self.arrive(10, 'helper, check the first question.')
            await self.finish_worker()
        self.assertEqual(len(self.requests), 3)
        self.assertIn('Reply delivery failed', self.sent[0])
        self.assertEqual(self.sent[-1], 'Answer 3')

    async def test_failed_final_request_keeps_the_included_candidate_pending(self):
        self.fail_request = 2
        async def incoming():
            await self.arrive(11, 'helper, the additional question is about blue paint.')
        self.tool_hook = incoming
        with self.assertLogs('tgchatbot', level='WARNING'):
            await self.arrive(10, 'helper, check the first question.')
            await self.finish_worker()
        self.assertIn('additional question', json.dumps(self.requests[1]['contents']))
        self.assertEqual(len(self.requests), 3)
        self.assertIn('Reply generation failed', self.sent[0])
        self.assertEqual(self.sent[-1], 'Answer 3')
