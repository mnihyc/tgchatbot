"""Manual Telegram compaction uses the real layered pipeline and PostgreSQL."""
from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.memory import MemoryService
from tgchatbot.core.memory_worker import MemoryWorker
from tgchatbot.core.prompting import build_system_prompt
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import ConversationMessage, MessageRole, ProviderResponse
from tgchatbot.transports.telegram_adapter import TelegramBotApp
from tgchatbot.transports.telegram_command_views import plain_text


class ManualCompactionWorkflows(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.app = TelegramBotApp.__new__(TelegramBotApp)
        self.app.config, self.app.store, self.app.runtime = self.config, self.store, self.runtime
        self.app._chat_states = {}
        self.progress = SimpleNamespace(edit_text=AsyncMock())
        self.message = SimpleNamespace(reply_text=AsyncMock(return_value=self.progress))
        self.update = SimpleNamespace(effective_chat=SimpleNamespace(id=100, type='group'),
            effective_message=self.message)
        self.context = SimpleNamespace(args=[])
        self.original_number = 0

    async def archive(self, count=18):
        await self.settings(compact_trigger_tokens=50000, compact_target_tokens=3500,
            compact_batch_tokens=1800, compact_min_messages=2, min_raw_messages_reserve=1,
            compact_keep_recent_ratio=.1)
        originals = []
        for number in range(count):
            self.original_number += 1
            originals.append(await self.runtime.ingest_user_message(session_id=self.session,
                incoming_message=ConversationMessage.user_text(
                    f'Discussion {number}. ' + 'We enjoyed tea and a quiet walk after work. ' * 50,
                    metadata={'source':'telegram','source_chat_id':'100','source_message_id':str(self.original_number),
                        'actor_id':f'telegram:user:{101+number%2}','actor_name':'Iris' if number%2==0 else 'Rowan',
                        'actor_kind':'user','sent_at':f'2026-01-01T00:{number:02d}:00+00:00'})))
        return originals

    async def summarize(self, **request):
        self.provider.requests.append(request)
        self.assertTrue(request.get('response_schema'), 'The command must not generate an ordinary chat reply.')
        fields = request['response_schema']['properties']
        result = {key:[] for key in fields}
        if request.get('response_schema_name') != 'profile_patch':
            result['scope'] = 'Participants discussed tea and walks.'
            if 'interaction_mode' in fields:
                result['interaction_mode'] = 'chat_or_sharing'
            if 'interaction_modes_seen' in fields:
                result['interaction_modes_seen'] = ['chat_or_sharing']
        return ProviderResponse(final_text=json.dumps(result))

    def result_text(self):
        return plain_text(self.progress.edit_text.await_args.args[0])

    async def estimate(self):
        settings = await self.store.get_or_create_session(self.session,self.config.default_session_settings())
        state = await self.runtime._get_live_state(self.session)
        return self.runtime._estimate_request_tokens(state,settings=settings,provider=self.provider,
            instructions=build_system_prompt(settings,timezone=self.config.default_metadata_timezone),
            tools=self.runtime._request_tools(settings))

    async def test_command_bypasses_trigger_reaches_target_and_retains_profiles_and_originals(self):
        originals = await self.archive()
        memory = MemoryService(self.store,SimpleNamespace(enabled=False))
        worker = MemoryWorker(store=self.store,embeddings=SimpleNamespace(enabled=False),
            providers={'openai':self.provider},config=self.config)
        self.addAsyncCleanup(worker.close)
        memory.worker = worker
        self.runtime.memory = memory
        await self.store.save_profile_fact(self.session,subject_actor_id='telegram:user:101',
            asserted_by='telegram:user:101',claim='Enjoys tea.',source_ids=[originals[0].db_id])
        settings_before = await self.store.get_or_create_session(self.session,self.config.default_session_settings())
        before = await self.estimate()
        self.assertGreater(before,settings_before.compact_target_tokens)
        self.assertLess(before,settings_before.compact_trigger_tokens)
        unchanged = await self.runtime.prepare_context(session_id=self.session)
        self.assertEqual(unchanged['compactions'],0)
        canonical = await self.store.read_messages(self.session,[row.db_id for row in originals])
        with patch.object(self.provider,'generate',side_effect=self.summarize):
            await self.app.compact_command(self.update,self.context)
        self.assertIn('Context ready',self.result_text())
        self.assertGreater(len(await self.store.list_memory_blocks(self.session)),1)
        self.assertLessEqual(await self.estimate(),settings_before.compact_target_tokens)
        self.assertEqual(settings_before,await self.store.get_or_create_session(self.session,self.config.default_session_settings()))
        self.assertEqual(canonical,await self.store.read_messages(self.session,[row.db_id for row in originals]))
        profiles = await memory.fetch_profiles(self.session,['person_id:101'])
        self.assertIn('Enjoys tea.',json.dumps(profiles))
        self.assertEqual(sum(r['response_schema_name']=='profile_patch' for r in self.provider.requests),1)
        state = await self.runtime._get_live_state(self.session)
        pairs = [row for row in state.raw_messages if row.message.metadata.get('synthetic_role')=='profile_refresh']
        self.assertEqual([row.message.role for row in pairs],[MessageRole.TOOL,MessageRole.TOOL])
        self.assertEqual([row.message.metadata['tool_phase'] for row in pairs],['call','result'])
        self.assertFalse(await self.store.compaction_needs_profile_refresh(self.session))
        self.assertIn(originals[-1].db_id,[row.db_id for row in state.raw_messages])
        restarted = AgentRuntime(config=self.config,store=self.store,tool_registry=self.tools,
            providers={'openai':self.provider},memory=memory)
        cold = await restarted._get_live_state(self.session)
        self.assertEqual(state.blocks,cold.blocks)
        self.assertEqual(state.raw_messages,cold.raw_messages)
        calls = len(self.provider.requests)
        with patch.object(self.provider,'generate',side_effect=self.summarize):
            await self.app.compact_command(self.update,self.context)
        self.assertEqual(len(self.provider.requests),calls,'Already within target: no repeat model work.')
        self.assertIn('Context ready',self.result_text())

    async def test_recent_reserve_cannot_be_discarded_to_claim_target_reached(self):
        originals = await self.archive(1)
        await self.settings(compact_target_tokens=500)
        with self.assertLogs('tgchatbot.transports.telegram_adapter',level='ERROR'):
            await self.app.compact_command(self.update,self.context)
        self.assertIn('stopped before reaching the target',self.result_text())
        self.assertEqual(self.provider.requests,[])
        self.assertEqual(len(await self.store.read_messages(self.session,[originals[0].db_id])),1)
        self.assertEqual(await self.store.list_memory_blocks(self.session),[])

    async def test_failed_request_retains_completed_batches_and_next_command_resumes(self):
        originals = await self.archive()
        calls = 0
        async def interrupted(**request):
            nonlocal calls
            calls += 1
            if calls==2:
                raise RuntimeError('Temporary fixture outage')
            return await self.summarize(**request)
        with patch.object(self.provider,'generate',side_effect=interrupted), self.assertLogs(level='WARNING'):
            await self.app.compact_command(self.update,self.context)
        self.assertIn('stopped before reaching the target',self.result_text())
        committed = await self.store.list_memory_blocks(self.session)
        self.assertTrue(committed)
        self.assertEqual(len(await self.store.read_messages(self.session,[row.db_id for row in originals])),len(originals))
        with patch.object(self.provider,'generate',side_effect=self.summarize):
            await self.app.compact_command(self.update,self.context)
        self.assertIn('Context ready',self.result_text())
        async with self.store.pool.connection() as conn:
            retained = await (await conn.execute('SELECT summary_text FROM memory_blocks WHERE id=%s',
                (committed[0].block_id,))).fetchone()
        self.assertEqual(retained['summary_text'],committed[0].summary_text)

    async def test_new_intake_survives_inflight_command_and_reset_blocks_stale_commit(self):
        for reset in (False,True):
            with self.subTest(reset=reset):
                await self.store.reset_context(self.session)
                originals = await self.archive()
                entered,release = asyncio.Event(),asyncio.Event()
                async def held(**request):
                    entered.set()
                    await release.wait()
                    return await self.summarize(**request)
                with patch.object(self.provider,'generate',side_effect=held):
                    task = asyncio.create_task(self.app.compact_command(self.update,self.context))
                    try:
                        await asyncio.wait_for(entered.wait(),5)
                        if reset:
                            await self.store.reset_context(self.session)
                        newer = await self.runtime.ingest_user_message(session_id=self.session,
                            incoming_message=ConversationMessage.user_text('New arrival while compaction runs.'))
                        release.set()
                        await asyncio.wait_for(task,10)
                    finally:
                        release.set()
                        if not task.done():
                            task.cancel()
                            await asyncio.gather(task,return_exceptions=True)
                live = await self.runtime._get_live_state(self.session)
                self.assertIn(newer.db_id,[row.db_id for row in live.raw_messages])
                if reset:
                    self.assertIn('reset or changed',self.result_text())
                    self.assertEqual(live.blocks,[])
                else:
                    self.assertIn('Context ready',self.result_text())
                    self.assertTrue(live.blocks)
                self.assertEqual(len(await self.store.read_messages(self.session,[row.db_id for row in originals])),len(originals))
