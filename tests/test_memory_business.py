from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import httpx
from psycopg import AsyncServerCursor, OperationalError
from psycopg.errors import QueryCanceled, RaiseException

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.compaction_schema import compaction_json_schema
from tgchatbot.core.memory import MemoryService
from tgchatbot.core.token_estimator import TokenEstimator
from tgchatbot.domain.models import ChatMode, ConversationMessage, MessagePart, MessageRole, PartKind, ProviderResponse, PromptInjectionMode, ToolCall
from tgchatbot.domain.provenance import original_text
from tgchatbot.storage.postgres_store import StaleScopeError
from tgchatbot.operational import MemoryConfig, from_env
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.tools.base import ToolContext


class MemoryBusinessTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.embeddings = SimpleNamespace(enabled=False, space_id="synthetic-space", embed_query=AsyncMock())
        self.memory = MemoryService(self.store, self.embeddings)
        self.runtime.memory = self.memory

    def message(self, text, actor, source_id, *, actor_name="Alex", chat="100"):
        return ConversationMessage.user_text(text, metadata={
            "source": "telegram", "source_chat_id": chat, "source_message_id": str(source_id),
            "actor_id": actor, "actor_kind": "user", "actor_name": actor_name,
            "sent_at": "2026-01-02T03:04:05+00:00",
        })

    async def ingest(self, text, actor="telegram:user:7", source_id=1, **kwargs):
        return await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=self.message(text, actor, source_id, **kwargs))

    async def test_search_unzoned_filters_use_conversation_timezone_without_reinterpreting_offsets(self):
        source = await self.ingest('I prefer jasmine tea.')
        tool = next(tool for tool in self.memory.tools if tool.name == 'memory_search')
        local = {'query': 'jasmine', 'after': '2026-01-02T11:00:00', 'before': '2026-01-02T12:00:00'}
        aware = {'query': 'jasmine', 'after': '2026-01-02T03:00:00+00:00', 'before': '2026-01-02T04:00:00Z'}
        default_context = ToolContext(self.session, 'Alex')
        utc_context = ToolContext(self.session, 'Alex', timezone='UTC')
        local_result = (await tool.runner.run(local, default_context)).output
        aware_result = (await tool.runner.run(aware, default_context)).output
        aware_in_utc = (await tool.runner.run(aware, utc_context)).output
        self.assertTrue(local_result['matches'])
        self.assertEqual(local_result, aware_result)
        self.assertEqual(local_result, aware_in_utc)
        self.assertEqual((await tool.runner.run(local, utc_context)).output['matches'], [])
        stored = (await self.store.read_messages(self.session, [source.db_id]))[0]
        self.assertEqual(stored.message.metadata['sent_at'], '2026-01-02T03:04:05+00:00')

    async def test_profile_fetch_keeps_same_name_people_and_agent_preferences_separate_with_freshness(self):
        alice = await self.ingest('I prefer tea. Please keep your answers concise.', 'telegram:user:7', 1)
        bob = await self.ingest('I might prefer coffee.', 'telegram:user:8', 2)
        await self.ingest('A message with unresolved identity.', 'unknown', 3)
        for actor, asserter, claim, source, kind in (
            ('telegram:user:7', 'telegram:user:7', 'Prefers tea', alice, 'explicit'),
            ('telegram:user:8', 'telegram:user:8', 'May prefer coffee', bob, 'inferred'),
            ('agent', 'telegram:user:7', 'Keep answers concise', alice, 'explicit'),
        ):
            await self.store.save_profile_fact(self.session, subject_actor_id=actor, asserted_by=asserter,
                claim=claim, source_ids=[source.db_id], kind=kind)
        await self.store.save_sticker_persona(self.session, {'opaque_selection_state': 'not a human preference'})
        tool = next(tool for tool in self.memory.tools if tool.name == 'user_profile_fetch')
        scope = await self.store.get_scope(self.session)
        result = (await tool.runner.run({'actor_ids': ['telegram:user:7', 'telegram:user:8', 'unknown', 'Alex']},
            ToolContext(self.session, 'A command issuer', scope=scope))).output
        self.assertTrue(result['ok'])
        profiles = {profile['actor_id']: profile for profile in result['profiles']}
        self.assertEqual(profiles['telegram:user:7']['identity']['actor_name'], 'Alex')
        self.assertEqual(profiles['telegram:user:8']['identity']['actor_name'], 'Alex')
        self.assertEqual(profiles['telegram:user:7']['facts'][0]['claim'], 'Prefers tea')
        self.assertEqual(profiles['telegram:user:8']['facts'][0]['kind'], 'inferred')
        self.assertEqual(profiles['agent']['subject_kind'], 'agent_preferences')
        self.assertEqual(profiles['agent']['facts'][0]['source_ids'], [alice.db_id])
        self.assertEqual(profiles['agent']['facts'][0]['asserted_by'], 'telegram:user:7')
        self.assertNotIn('opaque_selection_state', json.dumps(result))
        for unresolved in ('unknown', 'Alex'):
            self.assertEqual(profiles[unresolved]['status'], 'unknown_identity')
            self.assertFalse(profiles[unresolved]['identity']['known'])
            self.assertEqual(profiles[unresolved]['facts'], [])
        identity_source = profiles['telegram:user:7']['identity']['last_message']
        self.assertEqual((identity_source['message_id'], identity_source['source_revision']), (alice.db_id, 1))
        self.assertEqual(identity_source['sent_at'], '2026-01-02T11:04:05+08:00')
        for key in ('as_of', 'fetched_at'):
            self.assertEqual(datetime.fromisoformat(result[key]).utcoffset(), timedelta(hours=8))
        self.assertLessEqual(result['as_of'], result['fetched_at'])
        utc = await self.memory.fetch_profiles(self.session, ['telegram:user:7'], timezone='UTC', include_agent_preferences=False)
        self.assertEqual(len(utc['profiles']), 1)
        self.assertEqual(utc['profiles'][0]['identity']['last_message']['sent_at'], '2026-01-02T03:04:05+00:00')
        self.assertEqual(self.provider.requests, [])
        self.embeddings.embed_query.assert_not_awaited()

    async def test_profile_fetch_is_bounded_without_paging_or_modifying_original_evidence(self):
        source = await self.ingest('I have several durable preferences.')
        facts = [await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7',
            asserted_by='telegram:user:7', claim=f'Preference {number}: ' + '喜欢安静的地方。' * 30,
            source_ids=[source.db_id]) for number in range(3)]
        memory = MemoryService(self.store, self.embeddings, config=replace(self.memory.config, profile_bytes=1600))
        profile = (await memory.fetch_profiles(self.session, ['telegram:user:7'],
            include_agent_preferences=False))['profiles'][0]
        self.assertLessEqual(len(json.dumps(profile, ensure_ascii=False).encode('utf-8')), 1600)
        self.assertTrue(profile['facts'])
        self.assertNotIn('next_before_fact_id', profile)
        self.assertNotIn('truncated', profile)
        async with self.store.pool.connection() as conn:
            rows = await (await conn.execute('SELECT * FROM profile_facts ORDER BY id')).fetchall()
            revisions = await (await conn.execute('SELECT body,revision FROM message_revisions WHERE message_id=%s', (source.db_id,))).fetchall()
        self.assertEqual(rows, facts)
        self.assertEqual(revisions, [{'body': 'I have several durable preferences.', 'revision': 1}])

    async def test_profile_fetch_retains_soft_reset_history_but_never_crosses_chat_or_full_reset(self):
        source = await self.ingest('I prefer jasmine tea.')
        await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7', asserted_by='telegram:user:7',
            claim='Prefers jasmine tea', source_ids=[source.db_id])
        previous = await self.store.get_scope(self.session)
        await self.store.reset_context(self.session)
        with self.assertRaises(StaleScopeError):
            await self.memory.fetch_profiles(self.session, ['telegram:user:7'], scope=previous)
        current = await self.store.get_scope(self.session)
        profile = (await self.memory.fetch_profiles(self.session, ['telegram:user:7'], scope=current))['profiles'][0]
        self.assertEqual(profile['facts'][0]['source_ids'], [source.db_id])
        foreign = await self.memory.fetch_profiles('telegram:200', ['telegram:user:7'], include_agent_preferences=False)
        self.assertEqual(foreign['profiles'][0]['status'], 'unknown_identity')
        self.assertEqual(foreign['profiles'][0]['facts'], [])
        self.assertEqual(await self.store.count_sessions(), 1, 'Read-only profile fetch must not create a foreign chat')
        await self.store.reset_full(self.session, self.config.default_session_settings())
        fresh = await self.memory.fetch_profiles(self.session, ['telegram:user:7'])
        self.assertGreater(fresh['generation'], current['generation'])
        self.assertTrue(all(not profile['facts'] for profile in fresh['profiles']))
        self.assertFalse(fresh['profiles'][0]['identity']['known'])
        async with self.store.pool.connection() as conn:
            audit = (await (await conn.execute('SELECT count(*) AS n FROM profile_facts')).fetchone())['n']
        self.assertEqual(audit, 1)

    async def test_profile_fetch_follows_corrections_and_evidence_revisions_without_inventing_freshness(self):
        original = await self.ingest('I prefer tea.', source_id=1)
        old = await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7', asserted_by='telegram:user:7',
            claim='Prefers tea', source_ids=[original.db_id], valid_from='2026-01-01T00:00:00Z')
        correction = await self.ingest('I now prefer coffee.', source_id=2)
        new = await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7', asserted_by='telegram:user:7',
            claim='Prefers coffee', source_ids=[correction.db_id], supersedes=old['id'], valid_from='2026-02-01T00:00:00Z')
        await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7', asserted_by='telegram:user:7',
            claim='A future preference', source_ids=[correction.db_id], valid_from='2099-01-01T00:00:00Z')
        profile = (await self.memory.fetch_profiles(self.session, ['telegram:user:7']))['profiles'][0]
        self.assertEqual([fact['id'] for fact in profile['facts']], [new['id']])
        self.assertEqual(profile['facts'][0]['source_ids'], [correction.db_id])
        self.assertEqual(new['supersedes'], old['id'])
        self.assertEqual(new['source_revisions'], {str(correction.db_id): 1})
        await self.store.hide_message_ids(self.session, [correction.db_id])
        restored = (await self.memory.fetch_profiles(self.session, ['telegram:user:7']))['profiles'][0]
        self.assertEqual(restored['facts'], [], 'Restored evidence must be reconciled in a bounded batch')
        self.assertEqual(restored['identity']['last_message']['message_id'], original.db_id)
        await self.ingest('I have withdrawn that preference.', source_id=1)
        revised = (await self.memory.fetch_profiles(self.session, ['telegram:user:7']))['profiles'][0]
        self.assertEqual(revised['facts'], [])
        self.assertEqual(revised['status'], 'no_current_facts')
        self.assertEqual(revised['identity']['last_message']['source_revision'], 2)

    async def test_repeated_framework_snapshots_keep_original_evidence_searchable_and_context_replay_intact(self):
        await self.settings(mode=ChatMode.CHAT)
        source = await self.ingest('I prefer cardamom tea.', actor_name='Cardamom')
        await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7', asserted_by='telegram:user:7',
            claim='Prefers cardamom tea', source_ids=[source.db_id])
        # A narrow search window exposes duplicated snapshots crowding out the
        # source. Ordinary tool observations still belong to searchable history.
        memory = MemoryService(self.store, self.embeddings, config=replace(self.memory.config, search_results=2))
        for number in range(3):
            snapshot = await memory.fetch_profiles(self.session, ['telegram:user:7'])
            call_id = f'profile-refresh-{number}'
            for phase, payload in (
                ('call', {'call_id': call_id, 'arguments': {'actor_ids': ['telegram:user:7'], 'include_agent_preferences': True}}),
                ('result', {'call_id': call_id, 'output': snapshot}),
            ):
                await self.runtime.record_tool_observation(session_id=self.session, name='user_profile_fetch',
                    phase=phase, payload=payload,
                    metadata_update={'synthetic_role': 'profile_refresh', 'refresh_reason': 'compaction'})
            await self.store.append_message(self.session, ConversationMessage.user_text(
                '[Application reply target: Cardamom] Answer this person.',
                metadata={'synthetic_role': 'reply_target', 'reply_target': {'actor_id': 'telegram:user:7'}}))
        await self.runtime.record_tool_observation(session_id=self.session, name='shell_exec', phase='result',
            payload={'output': {'stdout': 'Cardamom lookup completed.'}})
        stored = await self.store.list_uncompacted_messages(self.session)
        controls = [item for item in stored if item.message.metadata.get('synthetic_role') in {'reply_target', 'profile_refresh'}]
        observation = next(item for item in stored if item.message.name == 'shell_exec')
        self.assertEqual(len(controls), 9)
        found = await memory.search(self.session, 'cardamom')
        self.assertEqual({row['message_ids'][0] for row in found['matches']}, {source.db_id, observation.db_id})
        self.assertEqual(next(row['fragments'] for row in found['messages'] if row['message_id'] == source.db_id),
            [{'offset': 0, 'text': 'I prefer cardamom tea.'}])
        read = await memory.read(self.session, [item.db_id for item in controls])
        self.assertEqual({item['message_id']: item['fragments'] for item in read['messages']},
                         {item.db_id: [{'offset': 0, 'text': original_text(item.message)}] for item in controls})
        profile = (await memory.fetch_profiles(self.session, ['telegram:user:7']))['profiles'][0]
        self.assertEqual(profile['identity']['last_message']['message_id'], source.db_id)
        self.assertEqual(profile['facts'][0]['source_ids'], [source.db_id])
        self.runtime.invalidate_session(self.session)
        current = await self.ingest('Continue from our earlier context.', source_id=2)
        self.provider.responses = [ProviderResponse(final_text='Your earlier preference remains available.'),
                                   ProviderResponse(final_text='The same preference is still available.')]
        await self.runtime.run_turn_from_stored(session_id=self.session, user_display_name='Alex', trigger_message_id=current.db_id)
        stored_pairs = [item.message for item in controls if item.message.metadata.get('synthetic_role') == 'profile_refresh']
        before_restart_pairs = [message for message in self.provider.requests[0]['messages']
                                if message.metadata.get('synthetic_role') == 'profile_refresh']
        self.runtime.invalidate_session(self.session)
        later = await self.ingest('Continue after restart.', source_id=3)
        await self.runtime.run_turn_from_stored(session_id=self.session, user_display_name='Alex', trigger_message_id=later.db_id)
        replayed_pairs = [message for message in self.provider.requests[1]['messages']
                          if message.metadata.get('synthetic_role') == 'profile_refresh']
        self.assertEqual(replayed_pairs, before_restart_pairs)
        self.assertEqual([message.metadata['tool_payload'] for message in replayed_pairs],
                         [message.metadata['tool_payload'] for message in stored_pairs])
        self.assertEqual([original_text(message) for message in replayed_pairs],
                         [original_text(message) for message in stored_pairs])
        self.assertEqual(len(replayed_pairs), 6)
        self.assertTrue(all(message.role == MessageRole.TOOL for message in replayed_pairs))
        for call, result in zip(replayed_pairs[::2], replayed_pairs[1::2], strict=True):
            self.assertEqual((call.metadata['tool_phase'], result.metadata['tool_phase']), ('call', 'result'))
            self.assertEqual(call.metadata['tool_payload']['call_id'], result.metadata['tool_payload']['call_id'])
        replay = '\n'.join(original_text(message) for message in self.provider.requests[1]['messages'])
        self.assertIn('Prefers cardamom tea', replay)
        self.assertIn('[Application reply target: Cardamom]', replay)
        self.assertIn('Cardamom lookup completed.', replay)

    def profile_exchange(self, call_id='profile-refresh-atomic'):
        return [self.runtime._tool_observation_message(name='user_profile_fetch', phase=phase, payload=payload,
            metadata_update={'synthetic_role': 'profile_refresh', 'refresh_reason': 'compaction'})
            for phase, payload in (
                ('call', {'call_id': call_id, 'arguments': {'actor_ids': ['telegram:user:7']}}),
                ('result', {'call_id': call_id, 'output': {'ok': True, 'profiles': []}}),
            )]

    async def test_completed_profile_exchange_result_save_failure_leaves_neither_half(self):
        source = await self.ingest('Keep my original question.')
        scope = await self.store.get_scope(self.session)
        # Inject a real database write failure after the call has been inserted.
        # The trigger exists only in this disposable test schema.
        async with self.store.pool.connection() as conn:
            await conn.execute('''CREATE FUNCTION reject_profile_result() RETURNS trigger LANGUAGE plpgsql AS $$
                BEGIN
                    IF NEW.metadata->>'synthetic_role'='profile_refresh' AND NEW.metadata->>'tool_phase'='result' THEN
                        RAISE EXCEPTION 'synthetic interrupted profile result save';
                    END IF;
                    RETURN NEW;
                END $$''')
            await conn.execute('''CREATE TRIGGER reject_profile_result BEFORE INSERT ON message_revisions
                FOR EACH ROW EXECUTE FUNCTION reject_profile_result()''')
        with self.assertRaisesRegex(RaiseException, 'synthetic interrupted profile result save'):
            await self.store.append_messages(self.session, self.profile_exchange(), expected_scope=scope)
        self.runtime.invalidate_session(self.session)
        state = await self.runtime._get_live_state(self.session)
        self.assertEqual([item.db_id for item in state.raw_messages], [source.db_id])
        async with self.store.pool.connection() as conn:
            rows = await (await conn.execute('SELECT id FROM messages WHERE session_id=%s ORDER BY id', (self.session,))).fetchall()
            revisions = await (await conn.execute('SELECT message_id FROM message_revisions ORDER BY message_id')).fetchall()
        self.assertEqual(rows, [{'id': source.db_id}])
        self.assertEqual(revisions, [{'message_id': source.db_id}])

    async def test_concurrent_same_chat_intake_cannot_split_completed_profile_exchange(self):
        source = await self.ingest('Original question.')
        scope = await self.store.get_scope(self.session)
        call_saved, release_result, intake_entered = asyncio.Event(), asyncio.Event(), asyncio.Event()
        append = self.store._append_encoded_message
        session = self.store._session
        intake_task = None

        async def pause_after_call(conn, session_id, message, *args, **kwargs):
            stored = await append(conn, session_id, message, *args, **kwargs)
            if message.metadata.get('tool_phase') == 'call':
                call_saved.set()
                await release_result.wait()
            return stored

        async def observe_intake(conn, session_id, **kwargs):
            if asyncio.current_task() is intake_task:
                intake_entered.set()
            return await session(conn, session_id, **kwargs)

        with patch.object(self.store, '_append_encoded_message', pause_after_call), \
             patch.object(self.store, '_session', observe_intake):
            pair_task = asyncio.create_task(self.store.append_messages(self.session, self.profile_exchange(), expected_scope=scope))
            try:
                await asyncio.wait_for(call_saved.wait(), timeout=5)
                intake_task = asyncio.create_task(self.store.append_message(self.session,
                    self.message('Concurrent new input.', 'telegram:user:7', 2)))
                await asyncio.wait_for(intake_entered.wait(), timeout=5)
                self.assertFalse(intake_task.done())
                visible = await self.store.list_uncompacted_messages(self.session)
                self.assertEqual([item.db_id for item in visible], [source.db_id], 'An uncommitted half must not enter another reader context')
                release_result.set()
                pair, intake = await asyncio.wait_for(asyncio.gather(pair_task, intake_task), timeout=5)
            finally:
                release_result.set()
                pending = [task for task in (pair_task, intake_task) if task is not None and not task.done()]
                for task in pending:
                    task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)
        visible = await self.store.list_uncompacted_messages(self.session)
        self.assertEqual([item.db_id for item in visible], [source.db_id, pair[0].db_id, pair[1].db_id, intake.db_id])
        self.assertEqual([item.message.metadata['tool_phase'] for item in pair], ['call', 'result'])
        self.assertEqual(pair[0].message.metadata['tool_payload']['call_id'], pair[1].message.metadata['tool_payload']['call_id'])

    async def test_configured_retrieval_reads_more_than_old_bounds_without_losing_originals(self):
        with patch.dict(os.environ, {
            'MEMORY_SEARCH_RESULTS': '25', 'MEMORY_READ_MESSAGES': '25',
            'MEMORY_READ_CHARS': '16000', 'MEMORY_SEARCH_RESULT_CHARS': '16000',
            'MEMORY_RESPONSE_CHARS': '80000',
        }):
            memory = MemoryService(self.store, self.embeddings)
        sources = [await self.ingest('saffron ' + ('x' * 13000 if number == 0 else str(number)),
                                    source_id=number + 1) for number in range(25)]
        result = await memory.search(self.session, 'saffron')
        self.assertEqual(len(result['matches']), 25)
        long_result = next(row for row in result['messages'] if row['message_id'] == sources[0].db_id)
        self.assertEqual(long_result['fragments'], [{'offset': 0, 'text': 'saffron ' + 'x' * 13000}])
        self.assertNotIn('partial', long_result)
        self.assertFalse(any(row.get('truncated') for row in result['matches']))
        read = await memory.read(self.session, [source.db_id for source in sources])
        self.assertEqual(len(read['messages']), 25)
        self.assertEqual(read['omitted_ids'], [])
        self.assertEqual(read['messages'][0]['fragments'], [{'offset': 0, 'text': 'saffron ' + 'x' * 13000}])
        self.assertNotIn('next_offset', read['messages'][0])
        # Lowering presentation limits leaves the durable original pageable.
        small = MemoryService(self.store, self.embeddings,
            config=replace(memory.config, read_chars=30, response_chars=30))
        first = (await small.read(self.session, [sources[0].db_id]))['messages'][0]
        second = (await small.read(self.session, [sources[0].db_id], offset=first['next_offset']))['messages'][0]
        self.assertEqual(first['fragments'], [{'offset': 0, 'text': ('saffron ' + 'x' * 13000)[:30]}])
        self.assertEqual(second['fragments'], [{'offset': 30, 'text': ('saffron ' + 'x' * 13000)[30:60]}])
        self.assertEqual((first['next_offset'], second['next_offset']), (30, 60))
        self.assertTrue(first['partial'])
        self.assertTrue(second['partial'])
        self.assertEqual((first['total_characters'], second['total_characters']), (13008, 13008))

    async def test_every_uncompacted_message_survives_cache_eviction_and_reaches_generation(self):
        with patch.dict(os.environ, {'MEMORY_CACHED_SESSIONS': '1'}):
            limits = from_env(MemoryConfig, 'MEMORY')
        self.runtime.config = replace(self.config, memory=limits)
        self.store.config = replace(self.store.config, history_page_size=37)
        sources = [await self.ingest(f'Original {number}', source_id=number + 1) for number in range(272)]
        hot = await self.runtime._get_live_state(self.session)
        expected = [source.db_id for source in sources]
        self.assertEqual([source.db_id for source in hot.raw_messages], expected)
        self.assertEqual(hot.last_message_id, sources[-1].db_id)
        # Ordinary list callers still request a page, not the whole context.
        self.assertEqual(len(await self.store.list_uncompacted_messages(self.session)), 37)
        # Loading another chat evicts only reconstructible process state.
        await self.runtime._get_live_state('another-chat')
        restored = await self.runtime._get_live_state(self.session)
        self.assertIsNot(restored, hot)
        self.assertEqual([source.db_id for source in restored.raw_messages], expected)
        self.assertEqual(restored.last_message_id, sources[-1].db_id)
        self.runtime.memory = None
        self.provider.responses = [ProviderResponse(final_text='All current context is available.')]
        await self.runtime.run_turn_from_stored(session_id=self.session, user_display_name='Alex',
            trigger_message_id=sources[-1].db_id)
        sent_originals = [original_text(message) for message in self.provider.requests[-1]['messages']
                          if message.metadata.get('source') == 'telegram']
        self.assertEqual(sent_originals, [f'Original {number}' for number in range(272)])

    async def test_block_pages_and_enrichment_reload_preserve_all_active_compaction_input(self):
        self.store.config = replace(self.store.config, history_page_size=3, memory_block_page_size=7)
        sources = [await self.ingest(f'Original {number}', source_id=number + 1) for number in range(140)]
        blocks = [await self.store.create_memory_block(self.session,
            summary_text=f'Summary {number}', estimated_tokens=5, source_message_ids=[source.db_id])
            for number, source in enumerate(sources[:130])]
        self.runtime.invalidate_session(self.session)
        state = await self.runtime._get_live_state(self.session)
        self.assertEqual([block.block_id for block in state.blocks], [block.block_id for block in blocks])
        self.assertEqual([source.db_id for source in state.raw_messages], [source.db_id for source in sources[130:]])
        self.assertEqual(len(await self.store.list_memory_blocks(self.session)), 7)
        # An edited compacted source returns to the raw pipeline, while only its
        # invalidated summary disappears. No block page becomes a history cap.
        revised = await self.ingest('Corrected original zero.', source_id=1)
        self.assertEqual(revised.db_id, sources[0].db_id)
        self.assertEqual([block.block_id for block in state.blocks], [block.block_id for block in blocks[1:]])
        expected_raw = [sources[0].db_id, *[source.db_id for source in sources[130:]]]
        self.assertEqual([source.db_id for source in state.raw_messages], expected_raw)
        # Attachment enrichment of a raw original uses the same complete reload.
        enriched = self.message('Original 139', 'telegram:user:7', 140)
        enriched.parts.append(MessagePart(kind=PartKind.TEXT, origin='attachment_excerpt',
                                          text='Extracted attachment detail.'))
        await self.runtime.ingest_user_message(session_id=self.session, incoming_message=enriched)
        self.assertEqual([source.db_id for source in state.raw_messages], expected_raw)
        self.assertEqual(len(state.blocks), 129)
        self.assertIn('Extracted attachment detail.', original_text(state.raw_messages[-1].message))
        self.runtime.invalidate_session(self.session)
        restored = await self.runtime._get_live_state(self.session)
        self.assertEqual([source.db_id for source in restored.raw_messages], expected_raw)
        self.assertEqual([block.block_id for block in restored.blocks], [block.block_id for block in state.blocks])

    async def test_paged_context_load_has_one_snapshot_across_concurrent_compaction(self):
        self.store.config = replace(self.store.config, history_page_size=2, memory_block_page_size=1)
        sources = [await self.ingest(f'Original {number}', source_id=number + 1) for number in range(5)]
        fetch = AsyncServerCursor.fetchmany
        compacted = None

        async def compact_after_first_page(cursor, size=0):
            nonlocal compacted
            rows = await fetch(cursor, size)
            if cursor.name == 'live_context_messages' and rows and compacted is None:
                compacted = await self.store.create_memory_block(self.session,
                    summary_text='The five originals.', estimated_tokens=8,
                    source_message_ids=[source.db_id for source in sources])
            return rows

        self.runtime.invalidate_session(self.session)
        with patch.object(AsyncServerCursor, 'fetchmany', compact_after_first_page):
            snapshot = await self.runtime._get_live_state(self.session)
        self.assertIsNotNone(compacted)
        self.assertEqual([source.db_id for source in snapshot.raw_messages], [source.db_id for source in sources])
        self.assertEqual(snapshot.blocks, [])
        # The next load sees the committed replacement, never a mixture of both.
        self.runtime.invalidate_session(self.session)
        fresh = await self.runtime._get_live_state(self.session)
        self.assertEqual(fresh.raw_messages, [])
        self.assertEqual([block.block_id for block in fresh.blocks], [compacted.block_id])
        self.assertEqual(fresh.last_message_id, sources[-1].db_id)
        # A compacted source redelivery is not new intake, even with no raw tail.
        await self.ingest('Original 4', source_id=5)
        self.assertEqual(fresh.raw_messages, [])
        self.assertEqual([block.block_id for block in fresh.blocks], [compacted.block_id])

    async def test_reset_during_paged_load_cannot_reinstall_pre_reset_raw_messages_or_blocks(self):
        self.store.config = replace(self.store.config, history_page_size=2, memory_block_page_size=1)
        fetch = AsyncServerCursor.fetchmany
        for reset_kind in ('soft', 'full'):
            with self.subTest(reset_kind=reset_kind):
                await self.store.reset_full(self.session, self.config.default_session_settings())
                self.runtime.invalidate_session(self.session)
                sources = [await self.ingest(f'Pre-reset {number}', source_id=number + 1) for number in range(5)]
                await self.store.create_memory_block(self.session, summary_text='Pre-reset summary.',
                    estimated_tokens=5, source_message_ids=[sources[0].db_id])
                fresh = None

                async def reset_after_first_page(cursor, size=0):
                    nonlocal fresh
                    rows = await fetch(cursor, size)
                    if cursor.name == 'live_context_messages' and rows and fresh is None:
                        if reset_kind == 'soft':
                            await self.store.reset_context(self.session)
                        else:
                            await self.store.reset_full(self.session, self.config.default_session_settings())
                        self.runtime.invalidate_session(self.session)
                        fresh = await self.store.append_message(self.session,
                            self.message('After reset.', 'telegram:user:7', 100))
                    return rows

                self.runtime.invalidate_session(self.session)
                with patch.object(AsyncServerCursor, 'fetchmany', reset_after_first_page):
                    loaded = await self.runtime._get_live_state(self.session)
                self.assertIsNotNone(fresh)
                self.assertEqual([source.db_id for source in loaded.raw_messages], [fresh.db_id])
                self.assertEqual(loaded.blocks, [])
                self.assertEqual(loaded.last_message_id, fresh.db_id)
                self.assertIs(await self.runtime._get_live_state(self.session), loaded)
                later = await self.ingest('Later input.', source_id=101)
                self.assertEqual([source.db_id for source in loaded.raw_messages], [fresh.db_id, later.db_id])
                self.assertEqual(loaded.last_message_id, later.db_id)

    async def test_same_display_name_does_not_merge_people_or_profile_subjects(self):
        alice = await self.ingest("I prefer coffee.", "telegram:user:7", 1)
        bob = await self.ingest("I avoid coffee.", "telegram:user:8", 2)
        for source, subject, claim in [(alice, "telegram:user:7", "Prefers coffee"), (bob, "telegram:user:8", "Avoids coffee")]:
            await self.store.save_profile_fact(self.session, subject_actor_id=subject,
                asserted_by=subject, claim=claim, source_ids=[source.db_id])
        result = await self.memory.search(self.session, "coffee")
        ids = {source for row in result["matches"] for source in row["message_ids"]}
        self.assertEqual(ids, {alice.db_id, bob.db_id})
        evidence = await self.memory.read(self.session, [alice.db_id, bob.db_id])
        self.assertEqual({row["speaker"]["id"] for row in evidence["messages"]}, {"telegram:user:7", "telegram:user:8"})
        self.assertEqual({row["speaker"]["name"] for row in evidence["messages"]}, {"Alex"})
        self.assertEqual(evidence['messages'], result['messages'])
        payload = await self.memory.fetch_profiles(self.session, ['telegram:user:8'], include_agent_preferences=False)
        self.assertEqual([(payload['profiles'][0]['actor_id'], fact['claim']) for fact in payload['profiles'][0]['facts']],
                         [("telegram:user:8", "Avoids coffee")])
        self.embeddings.embed_query.assert_not_awaited()

    async def test_soft_reset_clears_working_context_but_retains_historical_evidence_and_profile(self):
        source = await self.ingest("I prefer jasmine tea.")
        await self.settings(system_prompt="Keep this voice.")
        await self.store.save_profile_fact(self.session, subject_actor_id="telegram:user:7",
            asserted_by="telegram:user:7", claim="Prefers jasmine tea", source_ids=[source.db_id])
        old_scope = await self.store.get_scope(self.session)
        new_scope = await self.store.reset_context(self.session)
        self.runtime.invalidate_session(self.session)
        self.assertEqual(new_scope["generation"], old_scope["generation"])
        self.assertNotEqual(new_scope["context_id"], old_scope["context_id"])
        self.assertEqual(await self.store.list_messages(self.session), [])
        self.assertEqual((await self.settings()).system_prompt, "Keep this voice.")
        self.assertTrue((await self.memory.search(self.session, "jasmine"))["matches"])
        self.assertEqual((await self.memory.read(self.session, [source.db_id]))["messages"][0]["fragments"],
            [{'offset': 0, 'text': 'I prefer jasmine tea.'}])
        self.assertEqual(len(await self.store.get_profile(self.session, "telegram:user:7")), 1)
        with self.assertRaises(StaleScopeError):
            await self.memory.read(self.session, [source.db_id], scope=old_scope)

    async def test_full_reset_hides_old_generation_from_search_profile_and_source_reads(self):
        source = await self.ingest("I prefer jasmine tea.")
        await self.store.save_profile_fact(self.session, subject_actor_id="telegram:user:7",
            asserted_by="telegram:user:7", claim="Prefers jasmine tea", source_ids=[source.db_id])
        await self.store.create_excerpt(self.session, [source.db_id])
        await self.store.reset_full(self.session, self.config.default_session_settings())
        self.runtime.invalidate_session(self.session)
        self.assertEqual((await self.memory.search(self.session, "jasmine"))["matches"], [])
        self.assertEqual(await self.store.get_profile(self.session, "telegram:user:7"), [])
        read = await self.memory.read(self.session, [source.db_id])
        self.assertEqual(read["messages"], [])
        self.assertEqual(read["unavailable_ids"], [source.db_id])
        # Full reset leaves an audit record, which normal retrieval cannot read.
        async with self.store.pool.connection() as conn:
            count = (await (await conn.execute("SELECT count(*) AS n FROM messages WHERE id=%s", (source.db_id,))).fetchone())["n"]
        self.assertEqual(count, 1)
        fresh = await self.ingest("I now prefer coffee.", source_id=2)
        self.assertEqual((await self.memory.read(self.session, [fresh.db_id]))["messages"][0]["fragments"],
            [{'offset': 0, 'text': 'I now prefer coffee.'}])

    async def test_deleted_evidence_invalidates_profile_and_cannot_be_read_in_another_chat(self):
        source = await self.ingest("I prefer jasmine tea.")
        await self.store.save_profile_fact(self.session, subject_actor_id="telegram:user:7",
            asserted_by="telegram:user:7", claim="Prefers jasmine tea", source_ids=[source.db_id])
        foreign = await self.memory.read("telegram:200", [source.db_id])
        self.assertEqual(foreign["messages"], [])
        await self.store.delete_message_ids(self.session, [source.db_id])
        self.assertEqual((await self.memory.search(self.session, "jasmine"))["matches"], [])
        self.assertEqual(await self.store.get_profile(self.session, "telegram:user:7"), [])
        self.assertEqual((await self.memory.read(self.session, [source.db_id]))["unavailable_ids"], [source.db_id])

    async def test_chat_memory_tools_share_existing_round_budget_and_preserve_exact_prompt(self):
        await self.settings(mode=ChatMode.CHAT, max_interaction_rounds=1,
            system_prompt="  Keep my exact voice.  ", prompt_injection_mode=PromptInjectionMode.EXACT)
        source = await self.ingest("I prefer jasmine tea.")
        self.provider.responses = [ProviderResponse(tool_calls=[ToolCall("memory_read", "read-1", {"message_ids": [source.db_id]})]),
                                   ProviderResponse(final_text="You prefer jasmine tea.")]
        result = await self.runtime.run_turn_from_stored(session_id=self.session,
            user_display_name="command issuer", trigger_message_id=source.db_id)
        self.assertEqual(result.text, "You prefer jasmine tea.")
        self.assertEqual({tool.name for tool in self.provider.requests[0]["tools"]}, {"memory_search", "memory_read", "user_profile_fetch"})
        self.assertEqual(self.provider.requests[0]["instructions"], "Keep my exact voice.")
        self.assertTrue(self.provider.requests[1]["instructions"].startswith("Keep my exact voice.\n\n[Internal control note]"))
        self.assertEqual(self.provider.requests[1]["tools"], [])
        self.tools.list_tools.assert_not_called()
        self.tools.runner.run.assert_not_awaited()
        output = self.provider.requests[1]["extra_input_items"][-1]["output"]
        self.assertEqual(output["messages"][0]["speaker"]["id"], "telegram:user:7")

    async def test_final_round_cannot_execute_an_extra_memory_read(self):
        await self.settings(mode=ChatMode.CHAT, max_interaction_rounds=1)
        source = await self.ingest("I prefer jasmine tea.")
        self.memory.read = AsyncMock(wraps=self.memory.read)
        self.provider.responses = [ProviderResponse(tool_calls=[ToolCall("memory_read", f"read-{i}", {"message_ids": [source.db_id]})]) for i in range(2)]
        with self.assertRaisesRegex(RuntimeError, "Interaction-round limit"):
            await self.runtime.run_turn_from_stored(session_id=self.session, user_display_name="Alex", trigger_message_id=source.db_id)
        self.assertEqual(self.memory.read.await_count, 1)
        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual(self.provider.requests[1]["tools"], [])

    async def test_durable_reply_target_counts_against_request_budget(self):
        source = await self.ingest("I prefer jasmine tea.")
        self.provider.responses = [ProviderResponse(final_text="Your preference is jasmine tea.")]
        estimator = self.provider.estimate_request_tokens
        self.provider.estimate_request_tokens = Mock(wraps=estimator)
        await self.runtime.run_turn_from_stored(session_id=self.session,
            user_display_name="Alex", trigger_message_id=source.db_id)
        augmented = [call.kwargs for call in self.provider.estimate_request_tokens.call_args_list
            if call.kwargs.get("instructions") and any(
                message.metadata.get("synthetic_role") == "reply_target" for message in call.kwargs["messages"])]
        self.assertTrue(augmented, "The outgoing request must account for the reply target")
        for request in augmented:
            expected = estimator(**{**request, "history_tokens_override": None})
            actual = estimator(**request)
            self.assertEqual(actual.history_tokens, expected.history_tokens,
                "Cached raw-history estimates must not hide the added reply-target tokens")

    async def test_larger_explicit_search_result_reaches_the_reply_without_a_hidden_token_ceiling(self):
        with patch.dict(os.environ, {'MEMORY_SEARCH_RESULT_CHARS': '30000', 'MEMORY_RESPONSE_CHARS': '40000'}):
            limits = from_env(MemoryConfig, 'MEMORY')
        self.runtime.config = replace(self.config, memory=limits)
        self.runtime.memory = MemoryService(self.store, self.embeddings, config=limits)
        original = 'saffron ' * 3500
        source = await self.ingest(original)
        await self.settings(mode=ChatMode.CHAT, max_interaction_rounds=1)
        self.provider.responses = [ProviderResponse(tool_calls=[ToolCall('memory_search', 'search-1', {'query': 'saffron'})]),
                                   ProviderResponse(final_text='Evidence retained.')]
        await self.runtime.run_turn_from_stored(session_id=self.session,
            user_display_name='Alex', trigger_message_id=source.db_id)
        payload = self.provider.requests[1]['extra_input_items'][-1]['output']
        evidence = next(row for row in payload['messages'] if row['message_id'] == source.db_id)
        self.assertGreater(TokenEstimator.estimate_text(evidence['fragments'][0]['text']), 4096)
        self.assertEqual(evidence['fragments'], [{'offset': 0, 'text': original}])
        self.assertNotIn('partial', evidence)
        self.assertEqual(payload['matches'], [{'message_ids': [source.db_id]}])

    async def test_model_completion_started_before_reset_cannot_become_a_reply(self):
        await self.settings(mode=ChatMode.CHAT)
        entered, release = asyncio.Event(), asyncio.Event()
        async def generate(**kwargs):
            entered.set()
            await release.wait()
            return ProviderResponse(final_text="stale answer")
        self.provider.generate = generate
        task = asyncio.create_task(self.runtime.run_turn(session_id=self.session,
            user_display_name="Alex", incoming_message=self.message("old question", "telegram:user:7", 1)))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            await self.store.reset_full(self.session, self.config.default_session_settings())
            self.runtime.invalidate_session(self.session)
            release.set()
            with self.assertRaises(StaleScopeError):
                await task
        finally:
            release.set()
            if not task.done():
                task.cancel()
        self.assertEqual(await self.store.list_messages(self.session), [])

    async def test_ordinary_reply_does_not_search_history_or_duplicate_input(self):
        await self.settings(mode=ChatMode.CHAT, system_prompt="  Keep this exact voice.  ",
            prompt_injection_mode=PromptInjectionMode.EXACT)
        await self.ingest("The meeting is tomorrow.", source_id=1)
        current = await self.ingest("Please summarize the current plan.", source_id=2)
        self.provider.responses = [ProviderResponse(final_text="The meeting is tomorrow.")]
        with patch.object(self.store, "search_excerpts", AsyncMock(side_effect=QueryCanceled("synthetic deadline"))) as search:
            result = await self.runtime.run_turn_from_stored(session_id=self.session,
                user_display_name="Alex", trigger_message_id=current.db_id)
        search.assert_not_awaited()
        self.assertEqual(result.text, "The meeting is tomorrow.")
        request = self.provider.requests[0]
        self.assertEqual(request["instructions"], "Keep this exact voice.")
        texts = [original_text(message) for message in request["messages"]]
        self.assertIn("The meeting is tomorrow.", texts)
        self.assertIn("Please summarize the current plan.", texts)
        self.assertFalse(any(message.metadata.get('synthetic_role') == 'memory_context' for message in request['messages']))
        async with self.store.pool.connection() as conn:
            originals = await (await conn.execute(
                "SELECT id FROM messages WHERE session_id=%s AND source_message_id='2'",
                (self.session,))).fetchall()
            revisions = await (await conn.execute(
                "SELECT revision FROM message_revisions WHERE message_id=%s", (current.db_id,))).fetchall()
        self.assertEqual([row["id"] for row in originals], [current.db_id])
        self.assertEqual(len(revisions), 1)

    async def test_reset_during_post_compaction_profile_timeout_still_cancels_before_generation(self):
        await self.settings(mode=ChatMode.CHAT)
        for source_id, reset_kind in enumerate(("soft", "full"), 1):
            with self.subTest(reset=reset_kind):
                previous = await self.ingest('Previous context', source_id=source_id * 2)
                source = await self.ingest("Old question", source_id=source_id * 2 + 1)
                entered, release = asyncio.Event(), asyncio.Event()

                async def compact_previous(**kwargs):
                    await self.store.create_memory_block(self.session, summary_text='Previous context summarized.',
                        estimated_tokens=5, source_message_ids=[previous.db_id])
                    await self.runtime._reload_live_state(kwargs['state'])
                    return True

                async def timed_out_profiles(*args, **kwargs):
                    entered.set()
                    await release.wait()
                    raise QueryCanceled("synthetic deadline after reset")

                with patch.object(self.runtime, '_compact_if_needed', compact_previous), \
                     patch.object(self.store, 'fetch_profile_snapshot', timed_out_profiles):
                    task = asyncio.create_task(self.runtime.run_turn_from_stored(session_id=self.session,
                        user_display_name="Alex", trigger_message_id=source.db_id))
                    try:
                        await asyncio.wait_for(entered.wait(), timeout=5)
                        if reset_kind == "soft":
                            await self.store.reset_context(self.session)
                        else:
                            await self.store.reset_full(self.session, self.config.default_session_settings())
                        self.runtime.invalidate_session(self.session)
                        release.set()
                        with self.assertRaises(StaleScopeError):
                            await task
                    finally:
                        release.set()
                        if not task.done():
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                self.assertEqual(self.provider.requests, [])
                self.assertEqual(await self.store.list_messages(self.session), [])

    async def test_post_compaction_profile_timeout_keeps_answering_with_explicit_incomplete_profile_signal(self):
        await self.settings(mode=ChatMode.CHAT, system_prompt='Keep this exact voice.',
            prompt_injection_mode=PromptInjectionMode.EXACT)
        previous = await self.ingest('The meeting is tomorrow.', source_id=1)
        current = await self.ingest('Please summarize the current plan.', source_id=2)
        self.provider.responses = [ProviderResponse(final_text='The meeting is tomorrow.')]
        compacted = False

        async def compact_previous(**kwargs):
            nonlocal compacted
            if compacted:
                return False
            await self.store.create_memory_block(self.session, summary_text='The meeting is tomorrow.',
                estimated_tokens=6, source_message_ids=[previous.db_id])
            await self.runtime._reload_live_state(kwargs['state'])
            compacted = True
            return True

        with patch.object(self.runtime, '_compact_if_needed', compact_previous), \
             patch.object(self.store, 'fetch_profile_snapshot', AsyncMock(side_effect=QueryCanceled('synthetic deadline'))):
            result = await self.runtime.run_turn_from_stored(session_id=self.session,
                user_display_name='Alex', trigger_message_id=current.db_id)
        self.assertEqual(result.text, 'The meeting is tomorrow.')
        request = self.provider.requests[0]
        self.assertEqual(request['instructions'], 'Keep this exact voice.')
        self.assertIn('Please summarize the current plan.', [original_text(message) for message in request['messages']])
        refreshes = [item.message for item in await self.store.list_uncompacted_messages(self.session)
                     if item.message.metadata.get('synthetic_role') == 'profile_refresh']
        self.assertEqual(len(refreshes), 2)
        call, refresh = refreshes
        self.assertEqual((call.metadata['tool_phase'], refresh.metadata['tool_phase']), ('call', 'result'))
        self.assertEqual(call.metadata['tool_payload']['call_id'], refresh.metadata['tool_payload']['call_id'])
        self.assertTrue(all(message.role == MessageRole.TOOL for message in refreshes))
        self.assertIn('unavailable', original_text(refresh))
        self.assertIn('do not infer that profiles are empty', original_text(refresh))
        self.assertIn('do not infer that profiles are empty', '\n'.join(original_text(message) for message in request['messages']))
        async with self.store.pool.connection() as conn:
            revisions = await (await conn.execute('SELECT revision FROM message_revisions WHERE message_id=%s', (current.db_id,))).fetchall()
        self.assertEqual(revisions, [{'revision': 1}])

    async def test_explicit_memory_tools_do_not_hide_query_timeout(self):
        await self.ingest("Find the old plan.")
        scope = await self.store.get_scope(self.session)
        ctx = ToolContext(self.session, 'Alex', scope=scope)
        for name, method, arguments in (
            ('memory_search', 'search_excerpts', {'query': 'old plan'}),
            ('user_profile_fetch', 'fetch_profile_snapshot', {'actor_ids': ['telegram:user:7']}),
        ):
            with self.subTest(tool=name):
                tool = next(tool for tool in self.memory.tools if tool.name == name)
                with patch.object(self.store, method, AsyncMock(side_effect=QueryCanceled('synthetic deadline'))):
                    with self.assertRaises(QueryCanceled):
                        await tool.runner.run(arguments, ctx)

    async def test_explicit_memory_tools_do_not_hide_cancellation_or_database_failures(self):
        await self.ingest("Current question")
        scope = await self.store.get_scope(self.session)
        ctx = ToolContext(self.session, 'Alex', scope=scope)
        for name, method, arguments in (
            ('memory_search', 'search_excerpts', {'query': 'old plan'}),
            ('user_profile_fetch', 'fetch_profile_snapshot', {'actor_ids': ['telegram:user:7']}),
        ):
            for error in (asyncio.CancelledError(), OperationalError('synthetic connection failure')):
                with self.subTest(tool=name, error=type(error).__name__):
                    tool = next(tool for tool in self.memory.tools if tool.name == name)
                    with patch.object(self.store, method, AsyncMock(side_effect=error)):
                        with self.assertRaises(type(error)):
                            await tool.runner.run(arguments, ctx)

    def compaction_candidate(self, *, owned):
        return {"scope": "Different drink preferences", "interaction_mode": "chat_or_sharing",
            "participants": ["Alex"], "topics": ["drinks", "preferences"],
            "user_profile": ["telegram:user:7: Prefers coffee", "telegram:user:8: Avoids coffee"] if owned else ["Prefers coffee", "Avoids coffee"],
            "user_intent_or_shared_context": ["Two people share different preferences"],
            "why_it_mattered": [], "interaction_timeline": ["telegram:user:7 stated a preference; telegram:user:8 disagreed"],
            "results_or_takeaways": [], "decisions": [], "open_loops": [], "artifacts": [], "uncertainties": []}

    async def test_compaction_corrects_ownerless_claims_before_committing_two_same_name_people(self):
        settings = await self.settings(min_raw_messages_reserve=1)
        alice = await self.ingest("I prefer coffee.", "telegram:user:7", 1)
        bob = await self.ingest("I avoid coffee.", "telegram:user:8", 2)
        latest = await self.ingest("Latest question stays raw.", "telegram:user:7", 3)
        self.provider.responses = [ProviderResponse(final_text=json.dumps(self.compaction_candidate(owned=False))),
                                   ProviderResponse(final_text=json.dumps(self.compaction_candidate(owned=True)))]
        state = await self.runtime._get_live_state(self.session)
        changed = await self.runtime._compact_old_context(session_id=self.session, settings=settings,
            provider=self.provider, state=state, pressure=True)
        self.assertTrue(changed)
        self.assertEqual(len(self.provider.requests), 2)
        request_text = "\n".join(part.text or "" for message in self.provider.requests[0]["messages"] for part in message.parts)
        for item, actor in [(alice, "telegram:user:7"), (bob, "telegram:user:8")]:
            self.assertIn(actor, request_text)
            self.assertIn(f'"message_id": {item.db_id}', request_text)
        block = state.blocks[0]
        self.assertEqual(list(block.actor_labels), ["telegram:user:7", "telegram:user:8"])
        self.assertEqual(block.structured_data["user_profile"], self.compaction_candidate(owned=True)["user_profile"])
        self.assertEqual([item.db_id for item in state.raw_messages], [latest.db_id])
        originals = await self.memory.read(self.session, [alice.db_id, bob.db_id])
        self.assertEqual([item['fragments'] for item in originals['messages']],
            [[{'offset': 0, 'text': text}] for text in ('I prefer coffee.', 'I avoid coffee.')])

    async def test_gemini_attribution_retry_commits_summary_without_mutating_original_history(self):
        settings = await self.settings(provider='gemini', model='gemini-3.8-flash',
            min_raw_messages_reserve=1)
        alice = await self.ingest('I prefer coffee.', 'telegram:user:7', 1)
        bob = await self.ingest('I avoid coffee.', 'telegram:user:8', 2)
        latest = await self.ingest('Latest question stays raw.', 'telegram:user:7', 3)
        original_rows = await self.store.read_messages(self.session, [alice.db_id, bob.db_id, latest.db_id])
        captured = []

        def handler(request):
            payload = json.loads(request.content)
            captured.append(payload)
            if payload['contents'][-1]['role'] == 'model':
                return httpx.Response(400, json={'error': {
                    'message': 'Requests ending with a model turn are not supported.'}})
            candidate = self.compaction_candidate(owned=len(captured) > 1)
            return httpx.Response(200, json={'candidates': [{'content': {
                'role': 'model', 'parts': [{'text': json.dumps(candidate)}]}}]})

        provider = GeminiProvider(replace(self.config.gemini, api_key='synthetic-key'))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider._client = client
            state = await self.runtime._get_live_state(self.session)
            changed = await self.runtime._compact_old_context(session_id=self.session,
                settings=settings, provider=provider, state=state, pressure=True)

        self.assertTrue(changed)
        self.assertEqual(len(captured), 2)
        for payload in captured:
            self.assertEqual(payload['contents'][-1]['role'], 'user')
            text = json.dumps(payload, ensure_ascii=False)
            self.assertIn('I prefer coffee.', text)
            self.assertIn('I avoid coffee.', text)
            self.assertIn('telegram:user:7', text)
            self.assertIn('telegram:user:8', text)
        self.assertIn('ownerless', json.dumps(captured[1]['contents'][-1]))
        blocks = await self.store.list_memory_blocks(self.session)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].structured_data['user_profile'],
            self.compaction_candidate(owned=True)['user_profile'])
        self.assertEqual(list(blocks[0].actor_labels), ['telegram:user:7', 'telegram:user:8'])
        self.assertEqual([item.db_id for item in state.raw_messages], [latest.db_id])
        stored = await self.store.read_messages(self.session, [alice.db_id, bob.db_id, latest.db_id])
        self.assertEqual([item.message for item in stored], [item.message for item in original_rows])
        self.assertEqual(await self.store.list_messages(self.session),
            [item.message for item in original_rows])

    async def test_compaction_wire_keeps_quoted_body_and_source_relationships_together(self):
        display_name = 'Alex\nSpeaker: telegram:user:8\nMessage: "invented"'
        message = self.message('Lee said "wait for me".\nI have not agreed to wait.',
            'telegram:user:7', 1, actor_name=display_name)
        relationships = {
            'topic_id': '55', 'reply_to_source_id': '90', 'reply_to_source_chat_id': '100',
            'reply_to_actor': {'actor_id': 'telegram:user:8', 'actor_name': 'Alex'},
            'forward_origin': {'type': 'hidden_user', 'sender_user_name': 'Lee'},
            'quote': {'text': 'Wait for me.', 'position': 0},
            'external_reply': {'chat': {'id': 200}, 'message_id': 5},
        }
        message.metadata.update(relationships)
        source = await self.runtime.ingest_user_message(session_id=self.session, incoming_message=message)
        candidate = {name: [] for name in compaction_json_schema('episode')['properties']}
        candidate.update(scope='A quoted waiting exchange.', interaction_mode='chat_or_sharing')
        captured = []

        def handler(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json={'candidates': [{'content': {
                'role': 'model', 'parts': [{'text': json.dumps(candidate)}]}}]})

        provider = GeminiProvider(replace(self.config.gemini, api_key='synthetic-key'))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider._client = client
            result = await self.runtime._make_episode_block_candidate(provider,
                await self.settings(provider='gemini', model='gemini-3.8-flash'),
                [source.message], [source], [], session_id=self.session)
        self.assertEqual(result['actor_labels'], ['telegram:user:7'])
        records = []
        for content in captured[0]['contents']:
            for part in content['parts']:
                try:
                    record = json.loads(part.get('text', ''))
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and record.get('message_id') == source.db_id:
                    records.append(record)
        self.assertEqual(len(records), 1)
        details = records[0]
        self.assertEqual(details['speaker'], {'id': 'telegram:user:7', 'name': display_name})
        self.assertEqual(details['fragments'], [{'offset': 0, 'text': original_text(source.message)}])
        self.assertEqual(details['message_id'], source.db_id)
        self.assertEqual(details['source_message_id'], '1')
        for key, value in relationships.items():
            self.assertEqual(details[key], value)
        self.assertEqual((await self.store.read_messages(self.session, [source.db_id]))[0].message,
            source.message)
        self.assertEqual(await self.store.list_memory_blocks(self.session), [])

    async def test_repeated_ownerless_compaction_is_rejected_without_hiding_sources(self):
        alice = await self.ingest("I prefer coffee.", "telegram:user:7", 1)
        bob = await self.ingest("I avoid coffee.", "telegram:user:8", 2)
        self.provider.responses = [ProviderResponse(final_text=json.dumps(self.compaction_candidate(owned=False))) for _ in range(2)]
        candidate = await self.runtime._make_episode_block_candidate(self.provider, await self.settings(),
            [alice.message, bob.message], [alice, bob], [], session_id=self.session)
        self.assertIsNone(candidate)
        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual([item.db_id for item in await self.store.list_uncompacted_messages(self.session)], [alice.db_id, bob.db_id])
        self.assertEqual(await self.store.list_memory_blocks(self.session), [])

    async def test_compaction_source_actor_list_does_not_drop_the_ninth_person(self):
        rows = [await self.ingest(f"Participant {i}", f"telegram:user:{i}", i) for i in range(1, 11)]
        metadata = self.runtime._compaction_metadata_message(mode="episode", raw_messages=rows,
            parent_blocks=[], time_start=None, time_end=None)
        expected = [f"telegram:user:{i}" for i in range(1, 11)]
        self.assertEqual(metadata.metadata["compaction_actor_ids"], expected)
        for actor in expected:
            self.assertIn(actor, metadata.parts[0].text)
