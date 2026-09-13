"""Operator discovery uses real persisted state without consuming learning work."""
from __future__ import annotations

from contextlib import redirect_stdout
import base64
import io
import json
import os
from unittest.mock import AsyncMock, patch

from tests.business_helpers import BusinessTestCase
from tests.test_memory_image_tools import PIXEL
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind
from tgchatbot.domain.profiles import profile_size
from tgchatbot.tools.memory import OperationsConfig, _run, audit_records, job_records, parser, profile_records


class MemoryOperatorInspectionTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.settings()
        self.page = OperationsConfig(page_size=1)

    async def original(self, text, source_id, *, actor='telegram:user:7', session=None):
        session = session or self.session
        await self.store.get_or_create_session(session, self.config.default_session_settings())
        return await self.store.append_message(session, ConversationMessage.user_text(text, metadata={
            'source': 'telegram', 'source_chat_id': session.removeprefix('telegram:'),
            'source_message_id': str(source_id), 'actor_id': actor, 'actor_name': 'Fixture person',
            'actor_kind': 'user'}))

    async def rows(self, table):
        async with self.store.pool.connection() as conn:
            return await (await conn.execute(f'SELECT * FROM {table} ORDER BY 1')).fetchall()

    async def command(self, *arguments):
        output = io.StringIO()
        with patch.dict(os.environ, {'DATABASE_URL': self.test_dsn}), \
             patch('dotenv.load_dotenv'), patch('tgchatbot.config.load_config', return_value=self.config), \
             patch('tgchatbot.tools.memory.PostgresStore', return_value=self.store), \
             patch.object(self.store, 'close', new_callable=AsyncMock), \
             patch('tgchatbot.tools.memory.EmbeddingConfig.from_env', side_effect=AssertionError('Unexpected embedding configuration')), \
             patch('tgchatbot.tools.memory.EmbeddingClient', side_effect=AssertionError('Unexpected embedding client')), \
             patch('tgchatbot.tools.memory_inspection.EmbeddingClient', side_effect=AssertionError('Unexpected helper embedding client')), \
             patch('tgchatbot.providers.factory.build_providers', side_effect=AssertionError('Unexpected generation client')), \
             redirect_stdout(output):
            await _run(parser().parse_args(list(arguments)))
        return [json.loads(line) for line in output.getvalue().splitlines()]

    async def test_jobs_stream_all_saved_errors_payloads_and_filter_chat_generation_without_mutation(self):
        old_source = await self.original('Old generation evidence.', 1)
        old = await self.store.enqueue_job(self.session, 'embedding_batch', source_ids=[old_source.db_id],
            payload={'phase': 'polling', 'name': 'batches/old-paid-fixture'})
        await self.store.reset_full(self.session, self.config.default_session_settings())
        source = await self.original('Current evidence.', 2)
        payload = {'phase': 'polling', 'name': 'batches/accepted-fixture',
                   'display_name': 'retained-fixture-identity', 'excerpt_ids': [123], 'space_id': 'fixture-space'}
        failed = await self.store.enqueue_job(self.session, 'embedding_batch', source_ids=[source.db_id], payload=payload)
        async with self.store.pool.connection() as conn:
            await conn.execute("UPDATE jobs SET status='failed',attempts=4,error='Temporary quota error',"
                "lease_token='fixture-lease',lease_until=now(),available_at=now()+interval '1 hour' WHERE id=%s", (failed['id'],))
        pending = await self.store.enqueue_job(self.session, 'embedding_batch', source_ids=[source.db_id],
            payload={'phase': 'prepared', 'display_name': 'not-submitted-fixture'})
        foreign = await self.original('Other chat.', 2, session='telegram:200')
        await self.store.enqueue_job('telegram:200', 'embedding_batch', source_ids=[foreign.db_id], payload=payload)
        before = await self.rows('jobs')

        rows = [row async for row in job_records(self.store, self.session, kind='embedding_batch', options=self.page)]
        self.assertEqual([row['id'] for row in rows], [failed['id'], pending['id']])
        failed_rows = await self.command('jobs', '--chat-id', '100', '--status', 'failed', '--kind', 'embedding_batch')
        self.assertEqual(len(failed_rows), 1)
        shown = failed_rows[0]
        self.assertEqual((shown['id'], shown['attempts'], shown['error']),
                         (failed['id'], 4, 'Temporary quota error'))
        self.assertEqual(shown['payload'], payload)
        self.assertEqual(shown['source_ids'], [source.db_id])
        self.assertEqual(shown['source_revisions'], {str(source.db_id): 1})
        self.assertEqual(shown['lease_token'], 'fixture-lease')
        self.assertTrue(shown['lease_until'] and shown['available_at'])
        self.assertEqual([row async for row in job_records(self.store, self.session, job_id=old['id'])], [])
        archived = [row async for row in job_records(self.store, self.session, job_id=old['id'], generation=1)]
        self.assertEqual([row['id'] for row in archived], [old['id']])
        self.assertEqual(await self.rows('jobs'), before)

    async def test_profiles_discover_pending_actors_reuse_bounded_current_evidence_and_never_refresh(self):
        old = await self.original('Old retained preference.', 1, actor='telegram:user:1')
        await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:1',
            asserted_by='telegram:user:1', claim='Old preference.', source_ids=[old.db_id])
        await self.store.reset_full(self.session, self.config.default_session_settings())
        current = await self.original('I prefer warm tea.', 2)
        fact = await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7',
            asserted_by='telegram:user:7', claim='Prefers warm tea.', source_ids=[current.db_id])
        pending = await self.original('New person awaiting learning.', 3, actor='telegram:user:8')
        await self.original('Different chat.', 4, actor='telegram:user:9', session='telegram:200')
        await self.store.reset_context(self.session)
        tables = ('jobs', 'profile_current', 'profile_facts', 'profile_inputs')
        before = {table: await self.rows(table) for table in tables}
        rows = [row async for row in profile_records(self.store, self.session,
            max_bytes=self.config.memory.profile_bytes, options=self.page)]
        by_actor = {row['profile']['actor_id']: row for row in rows}
        self.assertEqual(set(by_actor), {'telegram:user:7', 'telegram:user:8'})
        learned = by_actor['telegram:user:7']
        self.assertEqual(learned['generation'], 2)
        self.assertEqual(learned['profile']['facts'][0]['claim'], 'Prefers warm tea.')
        self.assertEqual(learned['evidence'][0]['id'], fact['id'])
        self.assertEqual(learned['evidence'][0]['source_ids'], [current.db_id])
        self.assertLessEqual(profile_size(learned['profile']), self.config.memory.profile_bytes)
        waiting = by_actor['telegram:user:8']
        self.assertEqual(waiting['profile']['status'], 'no_current_facts')
        self.assertEqual(waiting['evidence'], [])
        self.assertEqual(waiting['pending_material'], {'sources': 1,
            'bytes': len('New person awaiting learning.'.encode()),
            'first_message_id': pending.db_id, 'last_message_id': pending.db_id})
        selected = await self.command('profiles', '--chat-id', '100', '--actor-id', 'telegram:user:8')
        self.assertEqual([row['profile']['actor_id'] for row in selected], ['telegram:user:8'])
        for table in tables:
            self.assertEqual(await self.rows(table), before[table])

    async def test_profiles_do_not_reintroduce_facts_invalidated_by_source_edit(self):
        source = await self.original('I like tea.', 1)
        fact = await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7',
            asserted_by='telegram:user:7', claim='Likes tea.', source_ids=[source.db_id])
        await self.original('Correction: I like water.', 1)
        rows = [row async for row in profile_records(self.store, self.session,
            max_bytes=self.config.memory.profile_bytes)]
        self.assertEqual(rows[0]['profile']['facts'], [])
        self.assertEqual(rows[0]['evidence'], [])
        self.assertEqual(rows[0]['pending_material']['sources'], 1)
        retained = await self.rows('profile_facts')
        self.assertEqual(retained[0]['id'], fact['id'])
        self.assertFalse(retained[0]['valid'])

    async def test_profile_and_pending_material_share_one_snapshot_when_worker_publishes_during_inspection(self):
        source = await self.original('I prefer green tea.', 1)
        scope = await self.store.get_scope(self.session)
        job = await self.store.claim_profile_batch(session_id=self.session,
            max_bytes=1000, lease_seconds=60, lazy=True)
        original_snapshot = self.store.fetch_profile_snapshot

        async def publish_then_read(*args, **kwargs):
            await self.store.apply_profile_patch(job, [{
                'subject_actor_id': 'telegram:user:7', 'asserted_by': 'telegram:user:7',
                'claim': 'Prefers green tea.', 'source_ids': [source.db_id],
                'reason': 'Explicit preference in the source.',
            }], [], max_bytes=self.config.memory.profile_bytes,
                expected_source_revisions=job['source_revisions'])
            self.assertEqual(await self.store.get_scope(self.session), scope)
            return await original_snapshot(*args, **kwargs)

        with patch.object(self.store, 'fetch_profile_snapshot', side_effect=publish_then_read):
            rows = [row async for row in profile_records(self.store, self.session,
                actor_id='telegram:user:7', max_bytes=self.config.memory.profile_bytes)]
        self.assertEqual(rows[0]['profile']['facts'][0]['claim'], 'Prefers green tea.')
        self.assertEqual(rows[0]['pending_material'], {'sources': 0, 'bytes': 0,
            'first_message_id': None, 'last_message_id': None})
        default_snapshot = await original_snapshot(self.session, ['telegram:user:7'],
            max_bytes=self.config.memory.profile_bytes)
        self.assertNotIn('pending_material', default_snapshot)

    async def test_telegram_audit_resolves_revisions_and_reply_chunk_aliases_across_resets(self):
        source = await self.original('Original.', 42)
        await self.original('Edited.', 42)
        answer = await self.store.append_message(self.session, ConversationMessage(role=MessageRole.ASSISTANT,
            parts=[MessagePart(kind=PartKind.TEXT, text='One complete answer with two delivery chunks.')]))
        await self.store.bind_message_source(self.session, answer.db_id, source='telegram', source_chat_id='100',
            source_message_ids=['70', '71'], actor_id='telegram:bot:1', actor_kind='bot', actor_name='Fixture bot')
        await self.store.reset_full(self.session, self.config.default_session_settings())
        latest = await self.original('Current original reusing source ID.', 42)
        await self.store.hide_message_ids(self.session, [latest.db_id])
        await self.original('Other chat same Telegram ID.', 42, session='telegram:200')
        rows = [row async for row in audit_records(self.store, self.session, telegram_message_id=42, options=self.page)]
        self.assertEqual([row['body'] for row in rows], ['Original.', 'Edited.', 'Current original reusing source ID.'])
        self.assertEqual([row['message_id'] for row in rows], [source.db_id, source.db_id, latest.db_id])
        self.assertTrue(rows[-1]['hidden'])
        archived = await self.command('audit', '--chat-id', '100', '--telegram-message-id', '71', '--generation', '1')
        self.assertEqual([row['message_id'] for row in archived], [answer.db_id])
        self.assertEqual(archived[0]['source_message_id'], '70')
        self.assertEqual([row async for row in audit_records(self.store, self.session,
            telegram_message_id=71, generation=2)], [])
        with self.assertRaises(SystemExit), patch('sys.stderr'):
            parser().parse_args(['audit', '--chat-id', '100', '--message-id', '1', '--telegram-message-id', '42'])

    async def test_telegram_audit_preserves_negative_import_ids_under_destination_chat(self):
        from tgchatbot.tools.import_desktop import import_file
        path = self.path / 'result.json'
        path.write_text(json.dumps({'name': 'Fixture group', 'type': 'private_group', 'id': 999,
            'messages': [{'id': -999999958, 'type': 'message', 'date': '2026-09-01T00:00:00',
                'from': 'Fixture person', 'from_id': 'user7', 'text': 'Before group upgrade.'}]}))
        await import_file(self.store, path, chat_id=100, defaults=self.config.default_session_settings())
        records = await self.command('audit', '--chat-id', '100', '--telegram-message-id=-999999958')
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['source_chat_id'], '100')
        self.assertEqual(records[0]['source_message_id'], '-999999958')
        self.assertEqual(records[0]['body'], 'Before group upgrade.')

    async def test_operator_cli_journey_search_read_image_and_context_without_providers(self):
        message = ConversationMessage(MessageRole.USER, [
            MessagePart(PartKind.TEXT, text='The amber lantern has a square handle.'),
            MessagePart(PartKind.IMAGE, mime_type='image/png', data_b64=PIXEL),
        ], metadata={'source': 'telegram', 'source_chat_id': '100', 'source_message_id': '9',
            'actor_id': 'telegram:user:7', 'actor_kind': 'user', 'actor_name': 'Fixture person'})
        source = await self.store.append_message(self.session, message)
        block = await self.store.create_memory_block(self.session, summary_text='A lantern was shown.',
            estimated_tokens=8, source_message_ids=[source.db_id])
        found = await self.command('search', '--chat-id', '100', '--query', 'amber lantern', '--lexical-only')
        self.assertEqual(found[0]['matches'][0]['message_ids'], [source.db_id])
        selected_id = str(found[0]['matches'][0]['message_ids'][0])
        read = await self.command('read', '--chat-id', '100', '--message-id', selected_id)
        recalled = next(item for item in read[0]['messages'] if item['message_id'] == source.db_id)
        self.assertEqual(recalled['fragments'][0]['text'], 'The amber lantern has a square handle.')
        image_id = recalled['images'][0]['image_id']
        output = self.path / 'recalled.png'
        exported = await self.command('image', '--chat-id', '100', '--message-id', selected_id,
            '--image-id', image_id, '--output', str(output))
        self.assertEqual(exported[0]['image_id'], image_id)
        self.assertEqual(output.read_bytes(), base64.b64decode(PIXEL))
        context = await self.command('context', '--chat-id', '100')
        self.assertEqual(context[0]['root_blocks'], 1)
        self.assertEqual(context[0]['uncompacted_messages'], 0)
        self.assertEqual(context[1]['block_id'], block.block_id)
        self.assertEqual(context[1]['summary_text'], 'A lantern was shown.')

    async def test_inspection_unknown_chat_does_not_create_it(self):
        count = await self.store.count_sessions()
        for function, kwargs in ((job_records, {}), (profile_records, {'max_bytes': self.config.memory.profile_bytes})):
            with self.subTest(command=function.__name__), self.assertRaisesRegex(ValueError, 'does not exist'):
                _ = [row async for row in function(self.store, 'telegram:404', **kwargs)]
        self.assertEqual(await self.store.count_sessions(), count)
