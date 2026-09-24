"""Read-only operator journeys over originals, summaries, profiles and tools."""
from __future__ import annotations

import html
import io
import json
import os
import time
from contextlib import redirect_stdout
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from psycopg.errors import ReadOnlySqlTransaction

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.inspection import inspect_context
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.core.memory import MemoryService
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind, PromptInjectionMode
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.storage.inspection import inspection_snapshot
from tgchatbot.storage.postgres_store import PostgresStore, StaleScopeError
from tgchatbot.storage.sticker_catalog import StickerCatalogStore
from tgchatbot.tools.memory_inspection import prepared_inspection
from tgchatbot.tools.memory import _run, parser
from tgchatbot.transports.context_views import render_context
from tgchatbot.transports.telegram_adapter import TelegramBotApp
from tgchatbot.transports.telegram_command_views import plain_text


class ContextInspectionTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.settings_value = await self.settings(provider='gemini', model='gemini-3.8-flash')
        self.provider = GeminiProvider(self.config.gemini)
        self.addAsyncCleanup(self.provider.aclose)
        self.provider.generate = AsyncMock(side_effect=AssertionError('Inspection cannot call a model'))
        self.addCleanup(self.provider.generate.assert_not_awaited)
        self.runtime.providers = {'gemini': self.provider}
        self.runtime.memory = MemoryService(self.store, SimpleNamespace(enabled=False))
        self.app = TelegramBotApp.__new__(TelegramBotApp)
        self.app.config, self.app.runtime, self.app.store = self.config, self.runtime, self.store
        self.app._chat_states = {}
        self.message = SimpleNamespace(reply_text=AsyncMock(), reply_document=AsyncMock())
        self.update = SimpleNamespace(effective_chat=SimpleNamespace(id=100, type='group'),
            effective_user=SimpleNamespace(id=7), effective_message=self.message)

    async def original(self, text, at='2026-09-24T01:00:00Z', actor='telegram:user:7'):
        return await self.store.append_message(self.session, ConversationMessage.user_text(text, metadata={
            'actor_id': actor, 'actor_kind': 'user', 'actor_name': 'Alex', 'actor_username': 'fixture_person',
            'sent_at': at}))

    async def show(self, *args):
        self.message.reply_text.reset_mock()
        self.message.reply_document.reset_mock()
        await self.app.context_command(self.update, SimpleNamespace(args=list(args)))
        if self.message.reply_text.await_args:
            self.assertEqual(self.message.reply_text.await_count, 1)
            self.message.reply_document.assert_not_awaited()
            return plain_text(self.message.reply_text.await_args.args[0])
        self.assertEqual(self.message.reply_document.await_count, 1)
        return self.message.reply_document.await_args.kwargs['document'].decode()

    async def test_phone_journey_reads_original_summary_and_saved_profile_without_learning(self):
        literal = '  Literal **message** <&>\n  Another line.  '
        source = await self.original(literal)
        parent = await self.store.create_memory_block(self.session, summary_text='Original participant context.',
            estimated_tokens=20, source_message_ids=[source.db_id], actor_labels=['telegram:user:7'],
            actor_identities=[{'id': 'telegram:user:7', 'name': 'Alex', 'username': 'fixture_person'}])
        digest = await self.store.create_memory_block(self.session, summary_text='A digest covering the older episode.',
            estimated_tokens=20, source_message_ids=[], parent_block_ids=[parent.block_id], kind='digest', level=2)
        await self.original('Fresh context.')
        fact = await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7',
            asserted_by='telegram:user:7', claim='Prefers concise replies.', source_ids=[source.db_id])
        snapshot = await self.runtime.memory.fetch_profiles(self.session, ['telegram:user:7'])
        captured = await self.runtime.record_tool_observation(session_id=self.session, name='user_profile_fetch',
            phase='result', payload={'call_id': 'profile', 'output': snapshot})
        await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7',
            asserted_by='telegram:user:7', claim='Also likes warm tea.', source_ids=[source.db_id])
        scope = await self.store.get_existing_scope(self.session)
        jobs = await self.store.job_status(self.session)
        overview = (await inspect_context(self.runtime, self.session))['data']
        self.assertEqual(overview['memory_blocks'], 2)
        self.assertEqual(overview['selected_memory_blocks'], 1)
        self.assertEqual(sum(overview['input_composition'].values()), overview['estimated_request_tokens'])
        self.assertGreater(overview['input_composition']['profiles'], 0)
        self.assertIn('summarized by block:', await self.show('recent'))
        self.assertIn(literal, await self.show('message', str(source.db_id)))
        included = await self.show('summaries')
        self.assertIn(f'block:{digest.block_id}', included)
        self.assertNotIn(f'block:{parent.block_id} ·', included)
        self.assertIn(f'block:{parent.block_id}', await self.show('summaries', 'all'))
        detail = await self.show('block', str(parent.block_id))
        self.assertIn('Alex', detail)
        self.assertIn('fixture_person', detail)
        self.assertIn(f'message:{source.db_id}', detail)
        profile = await self.show('profile', 'person_id:7')
        self.assertIn('Also likes warm tea.', profile)
        self.assertIn(f'fact:{fact["id"]}', profile)
        self.assertIn(f'/context message {captured.db_id}', profile)
        fetched = await self.show('message', str(captured.db_id))
        self.assertNotIn('Also likes warm tea.', fetched)
        self.assertIn('+08:00', fetched)
        self.assertEqual(await self.store.get_existing_scope(self.session), scope)
        self.assertEqual(await self.store.job_status(self.session), jobs)

    async def test_tools_keep_reused_call_ids_separate_and_show_native_and_delivery_records(self):
        calls = []
        for batch, stdout in [('a', 'first result'), ('b', 'second result')]:
            call = await self.runtime.record_tool_observation(session_id=self.session, name='shell_exec', phase='call',
                payload={'call_id': 'reused', 'arguments': {'command': batch}}, metadata_update={'tool_batch_id': batch})
            calls.append(call)
            await self.runtime.record_tool_observation(session_id=self.session, name='shell_exec', phase='result',
                payload={'call_id': 'reused', 'output': {'stdout': stdout}},
                metadata_update={'tool_batch_id': batch, 'tool_call_message_id': call.db_id})
        await self.runtime.record_tool_observation(session_id=self.session, name='file_send', phase='delivery',
            payload={'delivery_state': 'failed', 'workspace_path': 'notes.txt', 'error': 'fixture transfer error'})
        await self.store.append_message(self.session, ConversationMessage.assistant_text('An accompanied answer.', metadata={
            'provider_native': {'provider': 'gemini', 'items': [{'role': 'model', 'parts': [
                {'text': 'Native batch contains only part of the authored answer.'},
                {'toolCall': {'id': 'native-1', 'toolType': 'google_search', 'args': {'query': 'fixture'}}},
                {'toolResponse': {'id': 'native-1', 'toolType': 'google_search', 'response': {'text': 'Retained search'}}},
            ]}]}}))
        native = (await self.store.list_uncompacted_messages(self.session))[-1]
        opened_native = await self.show('message', str(native.db_id))
        self.assertIn('An accompanied answer.', opened_native)
        self.assertIn('Retained search', opened_native)
        first = await self.show('message', str(calls[0].db_id))
        self.assertIn('first result', first)
        self.assertNotIn('second result', first)
        self.assertIn('second result', await self.show('message', str(calls[1].db_id)))
        self.assertIn('delivery: failed', await self.show('tools', 'file_send'))
        native_view = await self.show('tools', 'google_search')
        self.assertIn('result recorded', native_view)
        self.assertNotIn('Result not recorded', native_view)
        evidence = '  Exact document text.\nWith **literal** formatting.  '
        doc = await self.runtime.record_tool_observation(session_id=self.session, name='read_doc', phase='result',
            payload={'call_id': 'doc', 'output': {'ok': True, 'path': 'note.txt'}},
            evidence_parts=[MessagePart(PartKind.TEXT, text=evidence, origin='document')])
        self.assertIn(evidence, await self.show('message', str(doc.db_id)))

    async def test_legacy_calls_without_batch_or_owner_do_not_borrow_later_results(self):
        calls = [await self.runtime.record_tool_observation(session_id=self.session, name='shell_exec', phase='call',
            payload={'call_id': 'reused', 'arguments': {'command': command}}) for command in ('first', 'second')]
        result = await self.runtime.record_tool_observation(session_id=self.session, name='shell_exec', phase='result',
            payload={'call_id': 'reused', 'output': {'stdout': 'Unlinked legacy outcome'}})
        for call in calls:
            opened = await self.show('message', str(call.db_id))
            self.assertIn('Result not recorded', opened)
            self.assertNotIn('Unlinked legacy outcome', opened)
        self.assertIn('unlinked call', await self.show('tools', 'shell_exec'))
        self.assertIn('Unlinked legacy outcome', await self.show('message', str(result.db_id)))
        await self.store.create_memory_block(self.session, summary_text='An earlier tool call remained unfinished.',
            estimated_tokens=12, source_message_ids=[calls[0].db_id])
        tools = await self.show('tools', 'shell_exec')
        self.assertIn('unlinked call', tools)
        self.assertIn(f'/context message {result.db_id}', tools)

    async def test_paging_source_time_timezone_and_resets_keep_their_declared_scope(self):
        late = await self.original('Newer source.', '2026-09-24T03:00:00Z')
        early = await self.original('Imported later but older source.', '2026-09-23T15:00:00Z')
        tied = await self.original('Same time, higher stored ID.', '2026-09-24T03:00:00Z')
        async with inspection_snapshot(self.store, self.session, self.settings_value, timezone='Asia/Singapore') as reader:
            page = await reader.recent(limit=1, source_time=True)
            self.assertEqual(page['next'], tied.db_id)
            page2 = await reader.recent(limit=1, before_id=page['next'], source_time=True)
            self.assertEqual(page2['next'], late.db_id)
            page3 = await reader.recent(limit=1, before_id=page2['next'], source_time=True)
            self.assertEqual(page3['items'][0]['message'].db_id, early.db_id)
            boundary = await reader.recent(limit=10, source_time=True, before='2026-09-24')
            self.assertEqual([r['message'].db_id for r in boundary['items']], [early.db_id])
            self.assertEqual((await reader.recent(limit=1))['next'], tied.db_id)
        await self.store.reset_context(self.session)
        fresh = await self.original('After soft reset.')
        self.assertNotIn('Newer source.', await self.show('recent'))
        self.assertIn('Newer source.', await self.show('message', str(late.db_id)))
        self.assertIn('unavailable', await self.show('recent', 'before', str(late.db_id)))
        async with inspection_snapshot(self.store, self.session, self.settings_value) as reader:
            self.assertEqual(len((await reader.recent(limit=10, source_time=True))['items']), 4)
        with self.assertRaises(StaleScopeError):
            async with inspection_snapshot(self.store, self.session, self.settings_value) as reader:
                await reader.state()
                await self.store.reset_full(self.session, self.settings_value)
        self.assertIn('unavailable', await self.show('message', str(fresh.db_id)))

    async def test_full_output_and_unicode_boundary_use_files_only_when_needed(self):
        self.runtime.memory = None
        await self.settings(system_prompt='Be concise.', prompt_injection_mode=PromptInjectionMode.EXACT)
        small = await self.show('full')
        self.message.reply_document.assert_not_awaited()
        self.assertIn('Be concise.', small)
        self.assertIn('gemini-3.8-flash', small)
        literal = 'Unabridged <&> **text**.\n' * 500
        original = await self.original(literal)
        large = await self.show('message', str(original.db_id))
        self.assertIn(literal, large)
        self.message.reply_text.assert_not_awaited()
        await self.settings(system_prompt=literal)
        self.assertIn(literal, await self.show('full'))
        self.message.reply_text.assert_not_awaited()
        # Unicode supplementary characters consume two Telegram UTF-16 units.
        boundary = '\U0001f600' * 2048
        for content, file in ((boundary, False), (boundary + 'x', True)):
            self.message.reply_text.reset_mock()
            self.message.reply_document.reset_mock()
            await self.app._send_command(self.message, html.escape(content))
            self.assertEqual(bool(self.message.reply_document.await_count), file)
            self.assertEqual(bool(self.message.reply_text.await_count), not file)

    async def test_prepared_cli_runs_without_credentials_on_readonly_connections_and_creates_nothing(self):
        await self.original('A report fixture.')
        await StickerCatalogStore(self.store).initialize()
        readonly = PostgresStore(self.test_dsn, schema=self.schema)
        self._stores.append(readonly)
        await readonly.open_readonly()
        async with readonly.pool.connection() as conn:
            self.assertEqual((await (await conn.execute('SHOW transaction_read_only')).fetchone())['transaction_read_only'], 'on')
        with self.assertRaises(ReadOnlySqlTransaction):
            async with readonly.pool.connection() as conn:
                await conn.execute('UPDATE sessions SET revision=revision+1')
        config = replace(self.config, openai=replace(self.config.openai, api_key=''),
            gemini=replace(self.config.gemini, api_key=''))
        with patch('httpx.AsyncClient.send', side_effect=AssertionError('Offline inspection cannot call APIs')):
            report = await prepared_inspection(readonly, config, self.session, topic='report')
            text = render_context(report)
            self.assertIn('A report fixture.', text)
            self.assertIn('gemini-3.8-flash', text)
            self.assertIn('generation 1', text)
        with self.assertRaisesRegex(ValueError, 'does not exist'):
            await prepared_inspection(readonly, config, 'telegram:999', topic='report')
        self.assertIsNone(await self.store.get_existing_scope('telegram:999'))

    async def test_cli_messages_prepared_blocks_and_report_file_follow_one_operator_journey(self):
        await StickerCatalogStore(self.store).initialize()
        old = await self.original('An earlier observation.')
        block = await self.store.create_memory_block(self.session, summary_text='The earlier observation was retained.',
            estimated_tokens=15, source_message_ids=[old.db_id])
        newest = await self.original('The latest original.')
        def connection(_dsn):
            store = PostgresStore(self.test_dsn, schema=self.schema)
            self._stores.append(store)
            return store
        async def run(*args):
            output = io.StringIO()
            with patch.dict(os.environ, {'DATABASE_URL': self.test_dsn}, clear=True), patch('dotenv.load_dotenv'), \
                    patch('tgchatbot.config.load_config', return_value=self.config), \
                    patch('tgchatbot.tools.memory.PostgresStore', side_effect=connection), \
                    patch('httpx.AsyncClient.send', side_effect=AssertionError('No API for inspection')), \
                    redirect_stdout(output):
                await _run(parser().parse_args([*args, '--chat-id', '100']))
            return output.getvalue()
        page = [json.loads(line) for line in (await run('messages', '--limit', '1')).splitlines()]
        self.assertEqual(page[0]['message_id'], newest.db_id)
        self.assertEqual(page[1], {'type': 'page', 'before_id': newest.db_id})
        page2 = [json.loads(line) for line in (await run('messages', '--before-id', str(newest.db_id))).splitlines()]
        self.assertEqual([row['message_id'] for row in page2], [old.db_id])
        self.assertIn(f'message:{old.db_id}', await run('context', '--block-id', str(block.block_id)))
        prepared = await run('context', '--prepared')
        self.assertIn('The earlier observation was retained.', prepared)
        self.assertIn('The latest original.', prepared)
        read = json.loads(await run('read', '--message-id', str(old.db_id)))
        self.assertIn('An earlier observation.', json.dumps(read))
        found = json.loads(await run('search', '--query', 'earlier observation', '--lexical-only'))
        self.assertEqual(found['matches'][0]['message_ids'], [old.db_id])
        output = self.path / 'context-report.txt'
        saved = json.loads(await run('report', '--output', str(output)))
        self.assertEqual(saved['output'], str(output))
        self.assertIn('gemini-3.8-flash', output.read_text())
        with self.assertRaises(FileExistsError):
            await run('report', '--output', str(output))

    async def test_commands_keep_arrival_activity_without_creating_history_and_handle_missing_scope(self):
        self.app.idle_compaction = SimpleNamespace(touch=Mock())
        before = await self.store.get_existing_scope(self.session)
        await self.app._observe_activity(self.update, SimpleNamespace(args=[]))
        await self.show('recent')
        self.app.idle_compaction.touch.assert_called_once_with(self.session)
        self.assertEqual(await self.store.get_existing_scope(self.session), before)
        self.assertEqual(await self.store.list_uncompacted_messages(self.session), [])
        self.update.effective_chat.id = 999
        self.message.reply_text.reset_mock()
        await self.app.status_command(self.update, SimpleNamespace(args=[]))
        self.assertIn('does not exist', plain_text(self.message.reply_text.await_args.args[0]))
        self.assertIsNone(await self.store.get_existing_scope('telegram:999'))

    async def test_profiles_page_pending_only_and_captured_only_actors_and_preserve_agent_identity(self):
        source = await self.original('Current pending evidence.', actor='telegram:chat:-200')
        await self.store.save_profile_fact(self.session, subject_actor_id='agent', asserted_by='telegram:chat:-200',
            claim='Use brief replies.', source_ids=[source.db_id])
        await self.runtime.record_tool_observation(session_id=self.session, name='user_profile_fetch', phase='result',
            payload={'call_id': 'capture', 'output': {'as_of': '2026-09-01T00:00:00Z',
                'profiles': [{'actor_id': 'person_id:99', 'facts': [{'claim': 'Old captured preference.'}]}]}})
        actors, after = [], None
        while True:
            page = (await inspect_context(self.runtime, self.session, 'profiles', limit=1, after=after))['data']
            actors.extend(row['profile']['actor_id'] for row in page['items'])
            after = page['next']
            if not after:
                break
        self.assertEqual(actors, ['agent', 'telegram:chat:-200', 'telegram:user:99'])
        pending = (await inspect_context(self.runtime, self.session, 'profile', actor='chat_id:-200'))['data']['items'][0]
        self.assertEqual(pending['pending']['sources'], 1)
        captured = (await inspect_context(self.runtime, self.session, 'profile', actor='person_id:99'))['data']['items'][0]
        self.assertEqual(captured['profile']['facts'], [])
        self.assertEqual(len(captured['snapshots']), 1)

    async def test_inspection_matches_runtime_cold_and_warm_and_keeps_one_snapshot_during_append(self):
        await self.original('First original.')
        await self.runtime.record_tool_observation(session_id=self.session, name='memory_read', phase='result',
            payload={'call_id': 'read', 'output': {'messages': [{'text': 'Source evidence.', 'sent_at': '2026-01-01T00:00:00Z'}]}})
        state = await self.runtime._get_live_state(self.session)
        before = self.runtime._build_provider_history(state, settings=self.settings_value, provider_name='gemini')
        full = await inspect_context(self.runtime, self.session, 'full')
        self.assertEqual([entry.message for entry in full['data']['timeline']], before)
        self.assertEqual(self.runtime._build_provider_history(state, settings=self.settings_value, provider_name='gemini'), before)
        cold = AgentRuntime(config=self.config, store=await self.new_store(), tool_registry=self.tools,
            providers={'gemini': self.provider}, memory=self.runtime.memory)
        restored = await inspect_context(cold, self.session, 'full')
        self.assertEqual(restored['data']['timeline'], full['data']['timeline'])
        self.assertEqual(restored['data']['overview']['input_composition'], full['data']['overview']['input_composition'])
        self.assertEqual(cold._live_sessions, {})
        async with inspection_snapshot(self.store, self.session, self.settings_value) as reader:
            before_page = await reader.recent(limit=10)
            await self.original('Appended during inspection.')
            self.assertEqual(await reader.recent(limit=10), before_page)
        self.assertIn('Appended during inspection.', await self.show('recent'))

    async def test_large_archive_paging_reads_current_context_without_loading_old_originals(self):
        async def seed(count, context_id):
            async with self.store.pool.connection() as conn:
                await conn.execute('''WITH added AS (
                    INSERT INTO messages(session_id,generation,context_id,role,actor_id,actor_kind,actor_name,sent_at)
                    SELECT %s,1,%s,'user','telegram:user:7','user','Alex',
                        '2026-01-01'::timestamptz + n * interval '1 second'
                    FROM generate_series(1,%s) n RETURNING id
                ) INSERT INTO message_revisions(message_id,revision,body,parts,metadata,estimated_tokens,fingerprint)
                SELECT id,1,'Synthetic archived context.',
                    '[{"kind":"text","text":"Synthetic archived context."}]'::jsonb,
                    '{}'::jsonb,10,'fixture' FROM added''', (self.session, context_id, count))
        await seed(100000, 1)
        await self.store.reset_context(self.session)
        await seed(2000, 2)
        async with self.store.pool.connection() as conn:
            await conn.execute('ANALYZE messages')
            await conn.execute('ANALYZE message_revisions')
        start = time.perf_counter()
        overview = (await inspect_context(self.runtime, self.session))['data']
        cold_seconds = time.perf_counter() - start
        self.assertEqual(overview['raw_messages'], 2000)
        start = time.perf_counter()
        warm = (await inspect_context(self.runtime, self.session))['data']
        warm_seconds = time.perf_counter() - start
        self.assertEqual(overview['input_composition'], warm['input_composition'])
        start = time.perf_counter()
        async with inspection_snapshot(self.store, self.session, self.settings_value) as reader:
            page = await reader.recent(limit=20, source_time=True)
            next_page = await reader.recent(limit=20, source_time=True, before_id=page['next'])
            self.assertFalse({row['message'].db_id for row in page['items']} &
                             {row['message'].db_id for row in next_page['items']})
            self.assertEqual(len(next_page['items']), 20)
        paging_seconds = time.perf_counter() - start
        print(f'Inspection fixture: 100K archived + 2K current; cold={cold_seconds:.3f}s, '
              f'warm={warm_seconds:.3f}s, two source-time pages={paging_seconds:.3f}s')
