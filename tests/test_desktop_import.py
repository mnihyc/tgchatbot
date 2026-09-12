"""Historical Desktop import stays passive, attributable, restartable and reset-safe."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

from psycopg import sql

from tgchatbot.domain.models import MessageRole, PartKind, SessionSettings
from tgchatbot.storage.postgres_store import PostgresStore, StaleScopeError, message_body
from tgchatbot.tools.import_desktop import ExportChat, desktop_message, import_file, inspect_export, iter_records


def record(number: int, text: object = 'original message', **fields) -> dict:
    return {'id': number, 'type': 'message', 'date': '2025-01-01T00:00:00',
            'date_unixtime': '1735689600', 'from': 'Alex', 'from_id': 'user11', 'text': text, **fields}


def export(messages: list[dict], *, chat_id: int = 42) -> dict:
    return {'name': 'Synthetic chat', 'type': 'private_supergroup', 'id': chat_id, 'messages': messages}


class DesktopExportReading(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='desktop-fixture-', dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'result.json'

    def write(self, content):
        self.path.write_text(json.dumps(content, ensure_ascii=False), encoding='utf-8')

    def test_full_export_requires_selection_and_handles_chat_header_after_messages(self):
        second = {'messages': [record(2, 'selected')], 'id': 200, 'name': 'Second', 'type': 'personal_chat'}
        self.write({'chats': {'list': [export([record(1, 'other')], chat_id=100), second]}})
        with self.assertRaisesRegex(ValueError, '--export-chat-id'):
            inspect_export(self.path)
        chat = inspect_export(self.path, '200')
        self.assertEqual(chat.metadata['id'], 200)
        self.assertEqual([item['text'] for item in iter_records(self.path, chat)], ['selected'])
        with self.assertRaisesRegex(ValueError, 'not found'):
            inspect_export(self.path, '300')

    def test_exporting_account_is_a_user_and_unknown_names_are_not_merged_into_people(self):
        chat = ExportChat({'id': 42}, None)
        exporting_user = desktop_message(record(1, from_id='user11'), chat, chat_id=-10042)
        known_bot = desktop_message(record(2, from_id='user99'), chat, chat_id=-10042, bot_user_id=99)
        unknown = desktop_message(record(3, from_id=None, forwarded_from='Someone'), chat, chat_id=-10042)
        channel = desktop_message(record(4, from_id='channel42'), chat, chat_id=-10042)
        self.assertEqual(exporting_user.role, MessageRole.USER)
        self.assertEqual(known_bot.role, MessageRole.ASSISTANT)
        self.assertEqual(unknown.metadata['actor_kind'], 'unknown')
        self.assertEqual(unknown.metadata['forward_origin']['actor_name'], 'Someone')
        self.assertEqual(channel.metadata['actor_id'], 'telegram:chat:-1000000000042')
        local_reply = desktop_message(record(5, reply_to_message_id=1, reply_to_peer_id='channel42'), chat, chat_id=-10042)
        foreign_reply = desktop_message(record(6, reply_to_message_id=1, reply_to_peer_id='channel43'), chat, chat_id=-10042)
        self.assertEqual(local_reply.metadata['reply_to_source_chat_id'], '-10042')
        self.assertEqual(foreign_reply.metadata['reply_to_source_chat_id'], '-1000000000043')

    def test_invalid_export_is_rejected_instead_of_silently_dropping_text(self):
        self.write({'chats': {'list': [{'id': 42}]}})
        with self.assertRaisesRegex(ValueError, 'messages array'):
            inspect_export(self.path)
        for text in (None, ['text', 17], [{'type': 'bold'}]):
            with self.subTest(text=text), self.assertRaises(ValueError):
                desktop_message(record(1, text), ExportChat({}, None), chat_id=42)

    def test_naive_export_time_uses_configured_zone_and_unix_time_stays_authoritative(self):
        original = record(1, date='2025-01-01T08:00:00')
        expected = desktop_message(original, ExportChat({}, None), chat_id=42).metadata['sent_at']
        del original['date_unixtime']
        rebuilt = desktop_message(original, ExportChat({}, None), chat_id=42)
        self.assertEqual(rebuilt.metadata['sent_at'], expected)
        explicit = desktop_message(original, ExportChat({}, None), chat_id=42, timezone='UTC')
        self.assertEqual(explicit.metadata['sent_at'], '2025-01-01T08:00:00+00:00')
        original['date'] = '2025-01-01T03:00:00+03:00'
        self.assertEqual(desktop_message(original, ExportChat({}, None), chat_id=42).metadata['sent_at'], expected)


@unittest.skipUnless(os.environ.get('TEST_DATABASE_URL'), 'requires disposable PostgreSQL/pgvector')
class DesktopImportWorkflows(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='desktop-fixture-', dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'result.json'
        self.schema = 'test_desktop_' + uuid.uuid4().hex
        self.store = PostgresStore(os.environ['TEST_DATABASE_URL'], schema=self.schema)
        await self.store.initialize()
        self.chat_id = -1000000000042
        self.session = f'telegram:{self.chat_id}'

    async def asyncTearDown(self):
        async with self.store.pool.connection() as conn:
            await conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))
        await self.store.close()

    def write(self, messages):
        self.path.write_text(json.dumps(export(messages), ensure_ascii=False), encoding='utf-8')

    async def import_messages(self, **kwargs):
        return await import_file(self.store, self.path, chat_id=self.chat_id, **kwargs)

    async def test_originals_identity_quotes_entities_and_attachment_references_survive_restart(self):
        self.write([
            record(1, ['😀 ', {'type': 'bold', 'text': 'I like tea'}], from_id='user11', reply_to_message_id=80,
                   forwarded_from='Quoted person', forwarded_from_id='user33', forwarded_message_id=4,
                   custom_export_field={'keep': 'metadata'}),
            record(2, 'I like coffee', from_id='user22'),
            record(3, 'attachment caption', file='(File not included. Change data exporting settings to download.)',
                   media_type='document', file_name='notes.pdf', mime_type='application/pdf'),
        ])
        result = await self.import_messages()
        self.assertEqual(result.messages, 3)
        await self.store.close()
        self.store = PostgresStore(os.environ['TEST_DATABASE_URL'], schema=self.schema)
        await self.store.initialize()
        originals = await self.store.list_canonical_messages(self.session)
        first = originals[0].message
        self.assertEqual(first.parts[0].text, '😀 I like tea')
        self.assertEqual(first.metadata['source_chat_id'], str(self.chat_id))
        self.assertEqual(first.metadata['export_chat']['id'], 42)
        self.assertEqual(first.metadata['actor_id'], 'telegram:user:11')
        self.assertEqual(first.metadata['forward_origin']['actor_id'], 'telegram:user:33')
        self.assertEqual(first.metadata['reply_to_source_id'], '80')
        self.assertEqual(first.metadata['entities'], [{'type': 'bold', 'offset': 3, 'length': 10}])
        self.assertEqual(first.metadata['desktop']['custom_export_field'], {'keep': 'metadata'})
        self.assertTrue(first.metadata['sent_at'].startswith('2025-01-01T00:00:00'))
        self.assertEqual(len(await self.store.search_messages(self.session, 'tea', actor_id='telegram:user:11')), 1)
        self.assertEqual(await self.store.search_messages(self.session, 'coffee', actor_id='telegram:user:11'), [])
        media = originals[2].message
        self.assertIn('notes.pdf', message_body(media))
        self.assertIn('unavailable', message_body(media))
        self.assertTrue(all(not part.remote_sync for part in media.parts))
        self.assertEqual([part.kind for part in media.parts], [PartKind.TEXT, PartKind.FILE])
        self.assertEqual(media.metadata['desktop']['mime_type'], 'application/pdf')
        jobs = await self.store.claim_jobs(10, kind='memory_ingest')
        self.assertEqual(len(jobs), 1)
        self.assertEqual(set(jobs[0]['source_ids']), {row.db_id for row in originals})

    async def test_commands_and_service_events_are_inert_without_telegram_or_remote_work(self):
        self.write([record(1, '/reset_full'), record(2, '/shell rm anything'),
                    record(3, '', type='service', action='invite_members', actor='Alex', actor_id='user11')])
        await self.store.get_or_create_session(self.session, SessionSettings(system_prompt='Keep this preset.'))
        initial = await self.store.get_scope(self.session)
        # DNS and process creation are forbidden during historical intake. The
        # already-open PostgreSQL pool is the only external boundary used here.
        with patch('socket.getaddrinfo', side_effect=AssertionError('unexpected network lookup')), \
             patch('asyncio.create_subprocess_exec', side_effect=AssertionError('unexpected process')):
            await self.import_messages()
        self.assertEqual((await self.store.get_scope(self.session))['generation'], initial['generation'])
        self.assertEqual((await self.store.get_or_create_session(self.session, SessionSettings())).system_prompt, 'Keep this preset.')
        rows = await self.store.list_canonical_messages(self.session)
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0].message.parts[0].text, '/reset_full')
        self.assertIn('invite_members', message_body(rows[2].message))
        self.assertTrue(all(row.message.role == MessageRole.USER for row in rows))

    async def test_reimport_deduplicates_and_shares_the_live_telegram_namespace(self):
        self.write([record(1, 'remember this'), record(2, 'another memory')])
        await self.import_messages()
        await self.import_messages()
        first = (await self.store.list_canonical_messages(self.session))[0]
        live_revision = desktop_message(record(1, 'a live correction', edited_unixtime='1735776000'),
                                        ExportChat({}, None), chat_id=self.chat_id)
        live_revision.metadata.pop('imported')
        live_revision.metadata.pop('desktop')
        live_revision.metadata.pop('export_chat')
        corrected = await self.store.append_message(self.session, live_revision)
        self.assertEqual(first.db_id, corrected.db_id)
        await self.import_messages()
        rows = await self.store.list_canonical_messages(self.session)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].message.parts[0].text, 'a live correction')
        revisions = await self.store.list_message_revisions(self.session, first.db_id)
        self.assertEqual(len(revisions), 2)

    async def test_upgraded_group_history_preserves_negative_source_ids_and_reply_links(self):
        from tgchatbot.storage.relationships import expand_message_ids

        previous_id = 12 - 1_000_000_000
        self.write([record(previous_id, 'Before the group upgrade'),
                    record(12, 'After the group upgrade', reply_to_message_id=previous_id)])
        await self.import_messages()
        await self.import_messages()
        rows = await self.store.list_canonical_messages(self.session)
        self.assertEqual(len(rows), 2)
        self.assertEqual([row.message.metadata['source_message_id'] for row in rows],
                         [str(previous_id), '12'])
        self.assertEqual(rows[1].message.metadata['reply_to_source_id'], str(previous_id))
        expanded = await expand_message_ids(self.store, self.session, [rows[1].db_id])
        self.assertEqual(expanded[:2], [rows[1].db_id, rows[0].db_id])
        self.assertTrue(all(row.db_id > 0 for row in rows))

    async def test_soft_reset_during_batched_import_keeps_all_history_active(self):
        self.write([record(number, f'history message {number}') for number in range(1, 206)])
        original_append = self.store.append_message
        count = 0

        async def append_and_reset(*args, **kwargs):
            nonlocal count
            saved = await original_append(*args, **kwargs)
            count += 1
            if count == 100:
                await self.store.reset_context(self.session)
            return saved

        progress = []
        with patch.object(self.store, 'append_message', side_effect=append_and_reset):
            result = await self.import_messages(progress=progress.append)
        self.assertEqual(result.messages, 205)
        self.assertEqual([item.messages for item in progress], [100, 200, 205])
        rows = await self.store.list_canonical_messages(self.session, limit=500)
        self.assertEqual(len(rows), 205)
        self.assertEqual(len(await self.store.list_messages(self.session, limit=500)), 105)
        self.assertTrue(await self.store.search_messages(self.session, 'history message 1'))
        jobs = await self.store.claim_jobs(10, kind='memory_ingest')
        self.assertEqual(sorted(len(job['source_ids']) for job in jobs), [5, 100, 100])

    async def test_full_reset_during_import_stops_writes_and_leaves_prior_rows_audit_only(self):
        self.write([record(number, f'old history {number}') for number in range(1, 4)])
        original_append = self.store.append_message
        imported_ids = []

        async def append_and_reset(*args, **kwargs):
            saved = await original_append(*args, **kwargs)
            imported_ids.append(saved.db_id)
            await self.store.reset_full(self.session, SessionSettings())
            return saved

        with patch.object(self.store, 'append_message', side_effect=append_and_reset):
            with self.assertRaises(StaleScopeError):
                await self.import_messages()
        self.assertEqual(len(imported_ids), 1)
        self.assertEqual(await self.store.list_canonical_messages(self.session), [])
        self.assertEqual(await self.store.search_messages(self.session, 'old history'), [])
        self.assertEqual(await self.store.claim_jobs(10, kind='memory_ingest'), [])
        self.assertEqual(len(await self.store.list_message_revisions(self.session, imported_ids[0])), 1)

    async def test_larger_configured_import_batches_queue_every_original_once(self):
        self.write([record(number, f'preserved original {number}') for number in range(1, 206)])
        progress = []
        with patch.dict(os.environ, {'IMPORT_BATCH_MESSAGES': '150', 'IMPORT_BATCH_BYTES': '1048576'}):
            result = await self.import_messages(progress=progress.append)
        self.assertEqual((result.messages, result.batches), (205, 2))
        self.assertEqual([item.messages for item in progress], [150, 205])
        rows = await self.store.list_canonical_messages(self.session, limit=205)
        self.assertEqual([row.message.parts[0].text for row in rows],
                         [f'preserved original {number}' for number in range(1, 206)])
        jobs = await self.store.claim_jobs(10, kind='memory_ingest')
        self.assertEqual(sorted(len(job['source_ids']) for job in jobs), [55, 150])
        self.assertEqual(sorted(mid for job in jobs for mid in job['source_ids']), [row.db_id for row in rows])

    async def test_record_limit_can_be_increased_and_import_resumed_without_losing_prior_text(self):
        long_text = 'Long original 茶🙂 ' * 100
        self.write([record(1, 'Committed before oversized record'), record(2, long_text)])
        with patch.dict(os.environ, {'IMPORT_MAX_RECORD_BYTES': '300', 'IMPORT_BATCH_MESSAGES': '10'}):
            with self.assertRaisesRegex(ValueError, 'IMPORT_MAX_RECORD_BYTES=300'):
                await self.import_messages()
        before = await self.store.list_canonical_messages(self.session)
        self.assertEqual([row.message.parts[0].text for row in before], ['Committed before oversized record'])
        # A byte target smaller than this one source never splits or truncates
        # the original. The independently configured record limit authorizes it.
        with patch.dict(os.environ, {'IMPORT_MAX_RECORD_BYTES': '8192', 'IMPORT_BATCH_BYTES': '300'}):
            result = await self.import_messages()
        self.assertEqual((result.messages, result.batches), (2, 2))
        rows = await self.store.list_canonical_messages(self.session)
        self.assertEqual(rows[0].db_id, before[0].db_id)
        self.assertEqual(rows[1].message.parts[0].text, long_text)
        self.assertEqual(len(await self.store.list_message_revisions(self.session, before[0].db_id)), 1)


if __name__ == '__main__':
    unittest.main()
