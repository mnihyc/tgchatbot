"""Idle maintenance, waiting replies and retry recovery with real persistence."""
from __future__ import annotations

import asyncio
import json
import os
import unittest
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from tests.business_helpers import BusinessTestCase
from tgchatbot.config import load_config
from tgchatbot.core.events import RuntimeEvent
from tgchatbot.core.idle_compaction import IdleCompaction
from tgchatbot.core.memory import MemoryService
from tgchatbot.core.runtime import AgentRuntime, CompactionModelRequestFailed, ContextLimitExceeded
from tgchatbot.core.token_estimator import TokenEstimator
from tgchatbot.domain.models import ConversationMessage, MessageRole, ProcessVisibility, ProviderResponse
from tgchatbot.storage.postgres_store import StaleScopeError, message_body
from tgchatbot.transports.telegram_adapter import ReplyCandidate, TelegramBotApp


class IdleTimerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.waits = asyncio.Queue()
        self.yield_loop = asyncio.sleep
        async def sleep(delay):
            future = asyncio.get_running_loop().create_future()
            await self.waits.put((delay, future))
            await future
        patcher = patch('tgchatbot.core.idle_compaction.asyncio.sleep', side_effect=sleep)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.runtime = SimpleNamespace(
            store=SimpleNamespace(get_or_create_session=AsyncMock(return_value=object())),
            config=SimpleNamespace(default_session_settings=lambda: object()),
            _effective_compact_idle_trigger_tokens=lambda settings: 600000,
            _effective_compact_idle_seconds=lambda settings: 3600,
            compact_idle_context=AsyncMock(return_value={'status':'ready'}))
        self.scheduler = IdleCompaction(self.runtime, busy=lambda sid:False)
        self.addAsyncCleanup(self.scheduler.close)

    async def test_activity_rearms_configured_delay_and_only_one_operation_runs(self):
        self.scheduler.touch('chat')
        delay, first = await self.waits.get()
        self.assertEqual(delay, 3600)
        self.scheduler.touch('chat')
        _, second = await self.waits.get()
        self.assertTrue(first.cancelled())
        self.runtime.compact_idle_context.assert_not_awaited()
        second.set_result(None)
        for _ in range(4):
            await self.yield_loop(0)
        self.runtime.compact_idle_context.assert_awaited_once()
        self.assertTrue(self.waits.empty(), 'No automatic repeat loop for the same idle period')

    async def test_busy_and_disabled_chats_do_not_compact(self):
        self.scheduler.busy = lambda sid: True
        self.scheduler.touch('busy')
        _, timer = await self.waits.get()
        timer.set_result(None)
        await self.yield_loop(0)
        self.runtime.compact_idle_context.assert_not_awaited()
        self.runtime._effective_compact_idle_trigger_tokens = lambda settings:0
        self.scheduler.touch('disabled')
        await self.yield_loop(0)
        self.assertTrue(self.waits.empty())

    async def test_silent_until_join_and_waiter_cancellation_preserves_work(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def compact(*, session_id, emit):
            await emit(RuntimeEvent(kind='phase', title='Compacting context'))
            entered.set()
            await release.wait()
            return {'status':'ready'}
        self.runtime.compact_idle_context.side_effect = compact
        self.scheduler.touch('chat')
        _, timer = await self.waits.get()
        timer.set_result(None)
        await entered.wait()
        visible = AsyncMock()
        waiter = asyncio.create_task(self.scheduler.join('chat', visible))
        await self.yield_loop(0)
        visible.assert_awaited_once()
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        operation = self.scheduler.sessions['chat'].operation
        self.assertFalse(operation.done())
        # Activity can arm a future timer without cancelling this request.
        self.scheduler.touch('chat')
        await self.waits.get()
        release.set()
        await operation
        self.runtime.compact_idle_context.assert_awaited_once()

    async def test_independent_chats_and_restart_rearm_without_persisting_timers(self):
        self.scheduler.touch('one')
        self.scheduler.touch('two')
        _, first = await self.waits.get()
        _, second = await self.waits.get()
        await self.scheduler.cancel('one')
        self.assertTrue(first.cancelled())
        self.assertFalse(second.done())
        await self.scheduler.close()
        self.assertTrue(second.cancelled())
        restarted = IdleCompaction(self.runtime, busy=lambda sid:False)
        self.addAsyncCleanup(restarted.close)
        restarted.touch('one')
        delay, _ = await self.waits.get()
        self.assertEqual(delay, 3600)

    async def test_status_delivery_failure_does_not_interrupt_waiting_for_compaction(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def compact(*, session_id, emit):
            await emit(RuntimeEvent(kind='phase', title='Compacting context'))
            entered.set()
            await release.wait()
            return {'status':'ready'}
        self.runtime.compact_idle_context.side_effect = compact
        self.scheduler.touch('chat')
        _, timer = await self.waits.get()
        timer.set_result(None)
        await entered.wait()
        waiter = asyncio.create_task(self.scheduler.join('chat', AsyncMock(side_effect=RuntimeError('Telegram unavailable'))))
        await self.yield_loop(0)
        self.assertFalse(waiter.done())
        release.set()
        self.assertTrue(await waiter)

    async def test_failed_idle_attempt_reports_to_waiter_without_automatic_repeat(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def compact(**kwargs):
            entered.set()
            await release.wait()
            raise RuntimeError('Model unavailable after retries')
        self.runtime.compact_idle_context.side_effect = compact
        self.scheduler.touch('chat')
        _, timer = await self.waits.get()
        timer.set_result(None)
        await entered.wait()
        waiter = asyncio.create_task(self.scheduler.join('chat'))
        await self.yield_loop(0)
        release.set()
        with self.assertRaisesRegex(RuntimeError, 'after retries'):
            await waiter
        await self.yield_loop(0)
        self.assertIsNone(self.scheduler.sessions['chat'].operation)
        self.assertTrue(self.waits.empty())
        self.runtime.compact_idle_context.assert_awaited_once()


class IdleCompactionWorkflows(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.config = replace(self.config, context=replace(self.config.context, compact_retry_delay_s=0))
        self.runtime.config = self.config
        await self.settings(compact_trigger_tokens=50000, compact_idle_trigger_tokens=4000,
            compact_target_tokens=2000, compact_batch_tokens=4000, compact_idle_seconds=0,
            compact_keep_recent_ratio=.2, compact_min_messages=4, min_raw_messages_reserve=1)
        self.requests = []
        self.provider.generate = AsyncMock(side_effect=self.generate)

    async def generate(self, **request):
        self.requests.append(request)
        schema = request.get('response_schema')
        if not schema:
            return ProviderResponse(final_text='Ready after compaction.')
        result = {key:[] for key in schema['properties']}
        result['scope'] = 'Alex keeps the blue ticket for the afternoon train.'
        if 'interaction_mode' in result:
            result['interaction_mode'] = 'chat_or_sharing'
        if 'interaction_modes_seen' in result:
            result['interaction_modes_seen'] = ['chat_or_sharing']
        return ProviderResponse(final_text=json.dumps(result))

    async def seed(self):
        return [await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text(
                f'Conversation {i}. ' + 'Keep the blue ticket for the afternoon train. '*30,
                metadata={'actor_id':'telegram:user:7', 'actor_name':'Alex', 'source_message_id':str(i),
                    'source_chat_id':'100', 'source':'telegram'})) for i in range(24)]

    async def test_environment_owns_defaults_and_explicit_session_overrides_survive(self):
        env = {'APP_DATA_DIR':str(self.path), 'APP_TEMP_DIR':str(self.path),
            'DEFAULT_PROVIDER':'openai', 'OPENAI_API_KEY':'dummy'}
        with patch.dict(os.environ, env, clear=True):
            conf = load_config(require_telegram=False)
        self.assertEqual((conf.context.compact_trigger_tokens, conf.context.compact_idle_trigger_tokens,
            conf.context.compact_idle_seconds, conf.context.compact_target_tokens,
            conf.context.compact_batch_tokens, conf.context.compact_retry_count,
            conf.context.compact_retry_delay_s), (800000,600000,3600,50000,40000,3,3))
        with patch.dict(os.environ, {**env,'CONTEXT_COMPACT_IDLE_TRIGGER_TOKENS':'0',
                'CONTEXT_COMPACT_IDLE_SECONDS':'90', 'CONTEXT_COMPACT_RETRY_COUNT':'0',
                'CONTEXT_COMPACT_RETRY_DELAY_S':'10'}, clear=True):
            overridden = load_config(require_telegram=False)
        self.assertEqual((overridden.context.compact_idle_trigger_tokens,
            overridden.context.compact_idle_seconds, overridden.context.compact_retry_count,
            overridden.context.compact_retry_delay_s), (0,90,0,10))
        expected = await self.settings(compact_trigger_tokens=300000, compact_target_tokens=100000,
            compact_idle_trigger_tokens=250000, compact_idle_seconds=120)
        restored = await (await self.new_store()).get_or_create_session(self.session,conf.default_session_settings())
        self.assertEqual(restored, expected)

    async def test_idle_reaches_target_preserves_originals_profiles_and_reconstruction(self):
        originals = await self.seed()
        memory = MemoryService(self.store, SimpleNamespace(enabled=False))
        memory.worker = SimpleNamespace(refresh_profiles=AsyncMock())
        self.runtime.memory = memory
        await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7',
            asserted_by='telegram:user:7', claim='Keeps the blue ticket.', source_ids=[originals[0].db_id])
        result = await self.runtime.compact_idle_context(session_id=self.session)
        self.assertEqual(result['status'], 'ready')
        self.assertLessEqual(result['estimated_request_tokens'], 2000)
        self.assertTrue(self.requests)
        self.assertTrue(all(request.get('response_schema') for request in self.requests), 'No conversational reply')
        memory.worker.refresh_profiles.assert_awaited_once()
        warm = await self.runtime._get_live_state(self.session)
        refresh = [row for row in warm.raw_messages if row.message.metadata.get('synthetic_role')=='profile_refresh']
        self.assertEqual([row.message.metadata['tool_phase'] for row in refresh], ['call','result'])
        self.assertFalse(await self.store.compaction_needs_profile_refresh(self.session))
        retained = await self.store.read_messages(self.session, [row.db_id for row in originals])
        self.assertEqual([message_body(row.message) for row in retained], [message_body(row.message) for row in originals])
        restarted = AgentRuntime(config=self.config, store=await self.new_store(),
            tool_registry=self.tools, providers={'openai':self.provider})
        cold = await restarted._get_live_state(self.session)
        self.assertEqual((warm.blocks, warm.raw_messages), (cold.blocks, cold.raw_messages))

    async def test_active_reply_waits_coalesces_new_triggers_and_respects_visibility(self):
        await self.seed()
        await self.settings(process_visibility=ProcessVisibility.STATUS)
        entered, release = asyncio.Event(), asyncio.Event()
        async def held(**request):
            if request.get('response_schema') and not entered.is_set():
                entered.set()
                await release.wait()
            return await self.generate(**request)
        self.provider.generate.side_effect = held
        app = TelegramBotApp.__new__(TelegramBotApp)
        app.config, app.store, app.runtime = self.config, self.store, self.runtime
        app._chat_states = {}
        app.idle_compaction = IdleCompaction(self.runtime, busy=app._chat_busy)
        self.addAsyncCleanup(app.idle_compaction.close)
        bot = SimpleNamespace(send_chat_action=AsyncMock())
        chat = SimpleNamespace(id=100, type='group')
        notice = SimpleNamespace(edit_text=AsyncMock(), delete=AsyncMock())
        app._send_text_message = AsyncMock(return_value=notice)
        app._deliver_result = AsyncMock(return_value=[])
        app._record_delivered_assistant_text = AsyncMock()
        app.idle_compaction.touch(self.session)
        await asyncio.wait_for(entered.wait(), 5)
        self.assertTrue((await app._flow_snapshot(100))['compacting'])
        app._send_text_message.assert_not_awaited()
        bot.send_chat_action.assert_not_awaited()
        for number in (100,101):
            incoming = await self.runtime.ingest_user_message(session_id=self.session,
                incoming_message=ConversationMessage.user_text(f'New request {number}.'))
            source = SimpleNamespace(chat=chat, message_id=number, get_bot=lambda:bot)
            await app._set_latest_reply_candidate(100, ReplyCandidate(incoming.db_id, 'Alex', source))
            if number == 100:
                # Wait until the reply has attached its progress listener.
                for _ in range(100):
                    if app.idle_compaction.sessions[self.session].emit is not None:
                        break
                    await asyncio.sleep(.001)
        self.assertFalse(any(not request.get('response_schema') for request in self.requests))
        worker = app._flow_state(100).reply_task
        release.set()
        await asyncio.wait_for(worker, 10)
        self.assertEqual(sum(not request.get('response_schema') for request in self.requests), 1)
        app._deliver_result.assert_awaited_once()
        self.assertEqual(app._deliver_result.call_args.args[0].message_id, 101)
        self.assertEqual(app._flow_state(100).last_replied_message_id, incoming.db_id)
        self.assertTrue(notice.edit_text.await_count)

    async def test_reset_during_summary_prevents_stale_commit(self):
        await self.seed()
        entered, release = asyncio.Event(), asyncio.Event()
        async def held(**request):
            entered.set()
            await release.wait()
            return await self.generate(**request)
        self.provider.generate.side_effect = held
        task = asyncio.create_task(self.runtime.compact_idle_context(session_id=self.session))
        try:
            await asyncio.wait_for(entered.wait(), 5)
            await self.store.reset_context(self.session)
            new = await self.runtime.ingest_user_message(session_id=self.session,
                incoming_message=ConversationMessage.user_text('Fresh context.'))
            release.set()
            with self.assertRaises(StaleScopeError):
                await task
            self.assertEqual(await self.store.list_memory_blocks(self.session), [])
            live = await self.runtime._get_live_state(self.session)
            self.assertEqual([row.db_id for row in live.raw_messages], [new.db_id])
        finally:
            release.set()
            await asyncio.gather(task, return_exceptions=True)

    async def test_protected_oversized_message_blocks_generation_without_erasing_it(self):
        await self.settings(compact_trigger_tokens=2500, compact_target_tokens=1000)
        original = await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text('Latest indivisible content. '*2000))
        with self.assertRaises(ContextLimitExceeded):
            await self.runtime.run_turn_from_stored(session_id=self.session, user_display_name='Alex',
                trigger_message_id=original.db_id)
        self.assertEqual(self.requests, [])
        self.assertEqual(message_body((await self.store.read_messages(self.session,[original.db_id]))[0].message),
            message_body(original.message))

    async def test_retry_count_delay_and_retry_after_have_one_owner(self):
        settings = await self.settings(provider_retry_count=9)
        self.runtime.config = replace(self.config, context=replace(self.config.context, compact_retry_delay_s=3))
        calls = 0
        request = httpx.Request('POST','https://model.invalid')
        async def attempt(**kwargs):
            nonlocal calls
            calls += 1
            response = httpx.Response(429 if calls==1 else 503, request=request,
                headers={'retry-after':'7'} if calls==1 else {})
            response.raise_for_status()
        self.provider.generate.side_effect = attempt
        with patch('tgchatbot.core.runtime.asyncio.sleep', new_callable=AsyncMock) as sleep:
            with self.assertRaises(httpx.HTTPStatusError):
                await self.runtime._generate_with_retries(provider=self.provider, settings=settings,
                    messages=[], instructions='', tools=[], extra_input_items=None, compaction=True)
        self.assertEqual(calls, 4, 'Three additional attempts, not multiplied by provider_retry_count')
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [7,3,3])

    async def test_permanent_failure_is_not_retried_and_next_attempt_resumes_checkpoints(self):
        originals = await self.seed()
        calls = 0
        async def fail_second(**request):
            nonlocal calls
            calls += 1
            if calls == 2:
                httpx.Response(401, request=httpx.Request('POST','https://model.invalid')).raise_for_status()
            return await self.generate(**request)
        self.provider.generate.side_effect = fail_second
        with self.assertRaises(CompactionModelRequestFailed):
            await self.runtime.compact_idle_context(session_id=self.session)
        self.assertEqual(calls, 2)
        committed = await self.store.list_memory_blocks(self.session)
        self.assertTrue(committed)
        self.provider.generate.side_effect = self.generate
        await self.runtime.compact_idle_context(session_id=self.session)
        self.assertEqual(len(await self.store.read_messages(self.session,[row.db_id for row in originals])), len(originals))
        async with self.store.pool.connection() as conn:
            row = await (await conn.execute('SELECT summary_text FROM memory_blocks WHERE id=%s',
                (committed[0].block_id,))).fetchone()
        self.assertEqual(row['summary_text'], committed[0].summary_text)

    async def test_live_compaction_cost_does_not_grow_with_off_prompt_archive(self):
        counts = []
        for count, expanding_digest in ((40, False), (900, False), (40, True)):
            with self.subTest(archived_episodes=count, expanding_digest=expanding_digest):
                await self.store.reset_context(self.session)
                settings = await self.settings(compact_trigger_tokens=300000,
                    compact_idle_trigger_tokens=200000, compact_target_tokens=100000,
                    compact_batch_tokens=40000, compact_keep_recent_ratio=.2,
                    compact_min_messages=24, min_raw_messages_reserve=8)
                originals = await self.store.append_messages(self.session, [
                    ConversationMessage.user_text(f'Archived discussion {i}.') for i in range(count)])
                text = 'Archived discussion. '*150
                for row in originals:
                    await self.store.create_memory_block(self.session, summary_text=text,
                        estimated_tokens=TokenEstimator.estimate_text(text)+32,
                        source_message_ids=[row.db_id])
                for i in range(8):
                    await self.runtime.ingest_user_message(session_id=self.session,
                        incoming_message=ConversationMessage.user_text(
                            f'Recent message {i}. '+'Current conversation. '*7000))
                modes = []
                make_digest = self.runtime._make_digest_block_candidate
                async def digest(provider, settings, parents, **kwargs):
                    state = await self.runtime._get_live_state(self.session)
                    visible = {block.block_id for block in self.runtime._select_blocks_for_prompt(state,settings=settings)}
                    self.assertTrue({block.block_id for block in parents} <= visible,
                        'Live compaction must not spend requests on off-prompt archive')
                    return await make_digest(provider,settings,parents,**kwargs)
                async def model(**request):
                    schema=request.get('response_schema') or {}
                    modes.append('digest' if 'durable_state' in schema.get('properties',{}) else 'episode')
                    result = await self.generate(**request)
                    if expanding_digest and modes[-1] == 'digest':
                        data = json.loads(result.final_text)
                        data['scope'] = 'Long older detail. '*2000
                        return ProviderResponse(final_text=json.dumps(data))
                    return result
                self.provider.generate.side_effect=model
                with patch.object(self.runtime,'_make_digest_block_candidate',side_effect=digest):
                    result=await self.runtime.compact_idle_context(session_id=self.session)
                self.assertEqual(result['status'],'ready')
                self.assertLessEqual(modes.index('episode'),1)
                if expanding_digest:
                    self.assertEqual(modes[:2], ['digest','episode'],
                        'A non-reducing digest must yield to raw compaction')
                self.assertEqual(len(await self.store.read_messages(self.session,[row.db_id for row in originals],limit=count)),count)
                counts.append(modes)
        self.assertEqual(counts[0],counts[1], 'More retained archive must not add live summary calls')
