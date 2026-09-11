from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from telegram.error import BadRequest

from tests.business_helpers import BusinessTestCase
from tgchatbot.domain.models import ConversationMessage, OutboundArtifact, OutboundSticker, ProcessVisibility, ResponseDelivery, StickerTiming, TurnResult
from tgchatbot.storage.postgres_store import StaleScopeError
from tgchatbot.transports.telegram_adapter import ReplyCandidate, TelegramBotApp
from tgchatbot.transports.telegram_render import TelegramMessageRenderer


class TelegramDeliverySourceTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.settings(process_visibility=ProcessVisibility.OFF)
        self.app = TelegramBotApp.__new__(TelegramBotApp)
        self.app.config, self.app.runtime, self.app.store = self.config, self.runtime, self.store
        self.app._chat_states = {}
        self.app._notify_user_error = AsyncMock()
        self.chat = SimpleNamespace(id=100)
        self.bot = SimpleNamespace(id=999, username='fixture_bot', send_chat_action=AsyncMock())
        self.next_id = 1000
        self.sent = []
        async def send(**kwargs):
            self.next_id += 1
            message = self.message(self.next_id)
            self.sent.append((message, kwargs['text']))
            return message
        self.bot.send_message = AsyncMock(side_effect=send)
        self.source = self.message(10, actor=7)

    def message(self, message_id, actor=999):
        sender = SimpleNamespace(id=actor, full_name='Fixture Bot' if actor == 999 else 'Alex', is_bot=actor == 999)
        message = SimpleNamespace(message_id=message_id, chat=self.chat, from_user=sender,
            date=datetime(2026, 1, 2, tzinfo=timezone.utc), get_bot=lambda: self.bot,
            delete=AsyncMock(), edit_text=AsyncMock())
        message.edit_text.return_value = message
        return message

    def renderer(self, placeholder=None, delivery=ResponseDelivery.EDIT):
        return TelegramMessageRenderer(placeholder, source_message=self.source,
            response_delivery=delivery, min_edit_interval_s=0, process_visibility=ProcessVisibility.STATUS)

    async def record(self, text, messages):
        await self.app._record_delivered_assistant_text(session_id=self.session,
            result=TurnResult(text, scope=await self.store.get_scope(self.session)),
            source_message=self.source, delivered_messages=messages)

    async def test_direct_multichunk_answer_records_actual_primary_aliases_and_bot_actor(self):
        text = 'Detailed answer. ' * 700
        result = TurnResult(text, scope=await self.store.get_scope(self.session))
        settings = await self.settings(process_visibility=ProcessVisibility.OFF)
        delivered = await self.app._deliver_result(self.source, self.renderer(), settings, result, sent_before_receipts=[])
        self.assertGreater(len(delivered), 1)
        await self.record(text, delivered)
        originals = await self.store.list_canonical_messages(self.session)
        self.assertEqual(len(originals), 1, 'All physical chunks share one canonical answer body')
        stored = originals[0]
        self.assertEqual(stored.message.metadata['source'], 'telegram')
        self.assertEqual(stored.message.metadata['source_message_id'], str(delivered[0].message_id))
        self.assertEqual(stored.message.metadata['actor_id'], 'telegram:user:999')
        self.assertEqual(stored.message.metadata['actor_kind'], 'bot')
        async with self.store.pool.connection() as conn:
            aliases = await (await conn.execute('SELECT * FROM message_source_aliases ORDER BY source_message_id')).fetchall()
            revisions = (await (await conn.execute('SELECT count(*) AS n FROM message_revisions')).fetchone())['n']
        self.assertEqual({row['source_message_id'] for row in aliases}, {str(message.message_id) for message in delivered[1:]})
        self.assertTrue(all(row['message_id'] == stored.db_id for row in aliases))
        self.assertEqual(revisions, 1)
        await self.store.bind_message_source(self.session, stored.db_id, source='telegram', source_chat_id='100',
            source_message_ids=[str(message.message_id) for message in delivered],
            actor_id='telegram:user:999', actor_kind='bot', actor_name='Fixture Bot')
        self.assertEqual((await self.store.list_canonical_messages(self.session))[0].message.metadata['source_revision'], 1)

    async def test_single_chunk_uses_primary_source_index_without_redundant_alias_row(self):
        delivered = []
        last = await self.app._send_text_message(self.source, 'One answer.', delivered_messages=delivered)
        self.assertIs(last, delivered[0], 'Placeholder/error callers retain the Message return contract')
        await self.record('One answer.', delivered)
        async with self.store.pool.connection() as conn:
            count = (await (await conn.execute('SELECT count(*) AS n FROM message_source_aliases')).fetchone())['n']
        self.assertEqual(count, 0)

    async def test_final_edit_includes_edited_placeholder_and_actual_continuation_ids(self):
        placeholder = self.message(50)
        renderer = self.renderer(placeholder)
        delivered = await renderer.finalize('Chunked answer. ' * 700)
        self.assertGreater(len(delivered), 1)
        self.assertEqual(delivered[0].message_id, 50)
        self.assertEqual([message.message_id for message in delivered[1:]], [message.message_id for message, _ in self.sent])
        placeholder.edit_text.assert_awaited_once()
        await self.record('Chunked answer. ' * 700, delivered)

    async def test_final_new_excludes_status_placeholder_and_failed_edit_fallback_excludes_deleted_message(self):
        placeholder = self.message(50)
        renderer = self.renderer(placeholder, ResponseDelivery.FINAL_NEW)
        renderer.state.lines = ['Status: complete']
        delivered = await renderer.finalize('New answer.')
        self.assertEqual(len(delivered), 1)
        self.assertNotEqual(delivered[0].message_id, 50)
        fallback = self.message(60)
        fallback.edit_text.side_effect = BadRequest('message to edit not found')
        renderer = self.renderer(fallback)
        delivered = await renderer.finalize('Replacement answer.')
        self.assertEqual([message.message_id for message in delivered], [self.sent[-1][0].message_id])
        self.assertNotEqual(delivered[0].message_id, 60)
        fallback.delete.assert_awaited_once()

    async def test_partial_text_delivery_failure_does_not_commit_complete_answer_or_aliases(self):
        self.runtime.run_turn_from_stored = AsyncMock(return_value=TurnResult('Long answer. ' * 700))
        self.bot.send_message.side_effect = [self.message(1001), TimeoutError('synthetic delivery failure')]
        await self.app._reply_to_candidate(ReplyCandidate(1, 'Alex', self.source))
        self.assertEqual(await self.store.list_canonical_messages(self.session), [])
        async with self.store.pool.connection() as conn:
            count = (await (await conn.execute('SELECT count(*) AS n FROM message_source_aliases')).fetchone())['n']
        self.assertEqual(count, 0)
        self.app._notify_user_error.assert_awaited_once()

    async def test_reset_after_final_text_cancels_files_and_stickers_without_error_notice(self):
        scope = await self.store.get_scope(self.session)
        path = self.path / 'report.txt'
        path.write_text('Synthetic report')
        result = TurnResult('Delivered before reset.', scope=scope,
            artifacts=[OutboundArtifact(path, 'report.txt')],
            stickers=[OutboundSticker(path, timing=StickerTiming.AFTER_FINAL)])
        self.runtime.run_turn_from_stored = AsyncMock(return_value=result)
        async def reset_after_send(**kwargs):
            await self.store.reset_full(self.session, self.config.default_session_settings())
            self.runtime.invalidate_session(self.session)
            return self.message(1001)
        self.bot.send_message.side_effect = reset_after_send
        self.bot.send_document = AsyncMock()
        self.app._send_stickers_direct = AsyncMock(return_value=[])
        await self.app._reply_to_candidate(ReplyCandidate(1, 'Alex', self.source))
        self.bot.send_document.assert_not_awaited()
        self.app._send_stickers_direct.assert_not_awaited()
        self.app._notify_user_error.assert_not_awaited()
        self.assertEqual(await self.store.list_canonical_messages(self.session), [])

    async def test_reset_during_sticker_delivery_cannot_insert_receipt_into_new_generation(self):
        scope = await self.store.get_scope(self.session)
        result = TurnResult('Final answer.', scope=scope,
            stickers=[OutboundSticker(self.path / 'sticker.webp', timing=StickerTiming.AFTER_FINAL)])
        self.runtime.run_turn_from_stored = AsyncMock(return_value=result)
        async def reset_after_sticker(*args):
            await self.store.reset_full(self.session, self.config.default_session_settings())
            self.runtime.invalidate_session(self.session)
            return [{'sent': True, 'telegram_message_id': 2001}]
        self.app._send_stickers_direct = AsyncMock(side_effect=reset_after_sticker)
        await self.app._reply_to_candidate(ReplyCandidate(1, 'Alex', self.source))
        self.app._notify_user_error.assert_not_awaited()
        self.assertEqual(await self.store.list_canonical_messages(self.session), [])
        async with self.store.pool.connection() as conn:
            count = (await (await conn.execute("SELECT count(*) AS n FROM messages WHERE role='tool'")).fetchone())['n']
        self.assertEqual(count, 0)

    async def test_binding_cannot_reassign_existing_source_or_cross_reset(self):
        await self.record('First answer.', [self.message(1001), self.message(1002)])
        first = (await self.store.list_canonical_messages(self.session))[0]
        await self.record('Second answer.', [self.message(1003)])
        second = (await self.store.list_canonical_messages(self.session))[-1]
        scope = await self.store.get_scope(self.session)
        kwargs = dict(source='telegram', source_chat_id='100', actor_id='telegram:user:999', actor_kind='bot', actor_name='Fixture Bot')
        with self.assertRaises(ValueError):
            await self.store.bind_message_source(self.session, second.db_id, source_message_ids=['1003', '1002'], **kwargs)
        with self.assertRaises(ValueError):
            await self.store.bind_message_source(self.session, first.db_id, source_message_ids=['2000'], **kwargs)
        await self.store.reset_full(self.session, self.config.default_session_settings())
        with self.assertRaises(StaleScopeError):
            await self.store.bind_message_source(self.session, first.db_id, source_message_ids=['1001', '1004'], expected_scope=scope, **kwargs)
        self.assertEqual(await self.store.list_canonical_messages(self.session), [])
