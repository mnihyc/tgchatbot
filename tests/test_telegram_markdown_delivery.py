from __future__ import annotations

import asyncio
import json
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from telegram import Bot, MessageEntity
from telegram.error import BadRequest, TimedOut
from telegram.request import BaseRequest

from tgchatbot.core.events import RuntimeEvent
from tgchatbot.domain.models import ProcessVisibility, ResponseDelivery
from tgchatbot.transports.telegram_render import MAX_TELEGRAM_TEXT_CHARS, TelegramMessageRenderer, bot_message_safe


def entity_text(payload, entity):
    """Read Telegram's actual UTF-16 span, independent of the Markdown converter."""
    encoded = payload['text'].encode('utf-16-le')
    return encoded[entity.offset * 2:(entity.offset + entity.length) * 2].decode('utf-16-le')


class TelegramMarkdownDeliveryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.bot = SimpleNamespace(send_message=AsyncMock(side_effect=self.send))
        self.payloads = []
        self.next_id = 100
        self.source = self.message(7)

    def message(self, message_id):
        message = SimpleNamespace(message_id=message_id, chat=SimpleNamespace(id=20),
            date=datetime(2026, 9, 13, tzinfo=timezone.utc), get_bot=lambda: self.bot,
            edit_text=AsyncMock(), delete=AsyncMock())
        async def edit(**kwargs):
            self.record(kwargs)
            return message
        message.edit_text.side_effect = edit
        return message

    def record(self, payload):
        self.assertIsNone(payload.get('parse_mode'), 'Native entities require no second Markdown parser')
        self.assertLessEqual(len(payload['text'].encode('utf-16-le')) // 2, MAX_TELEGRAM_TEXT_CHARS)
        self.assertTrue(payload['text'].strip(), 'Telegram rejects whitespace-only messages')
        for entity in payload.get('entities', []):
            self.assertIsInstance(entity, MessageEntity)
            self.assertTrue(entity_text(payload, entity))
        self.payloads.append(payload)

    async def send(self, **kwargs):
        self.record(kwargs)
        self.next_id += 1
        return self.message(self.next_id)

    def renderer(self, *, placeholder=None, visibility=ProcessVisibility.OFF, delivery=ResponseDelivery.EDIT):
        return TelegramMessageRenderer(placeholder, source_message=self.source,
            response_delivery=delivery, min_edit_interval_s=0, process_visibility=visibility)

    def spans(self, kind):
        return [entity_text(payload, entity) for payload in self.payloads
                for entity in payload.get('entities', []) if entity.type == kind]

    async def test_normal_markdown_formats_chinese_emoji_code_and_link_in_direct_answer(self):
        raw = '**粗体🙂** *斜体* `x_y = [1]` [网站](https://example.com/a?q=1&x=2)'
        renderer = self.renderer()
        messages = await renderer.finalize(raw)
        self.assertEqual(len(messages), 1)
        self.assertEqual(renderer.state.answer, raw, 'Transport formatting must not replace original assistant text')
        self.assertEqual(self.payloads[0]['text'], '粗体🙂 斜体 x_y = [1] 网站')
        self.assertEqual(self.spans('bold'), ['粗体🙂'])
        self.assertEqual(self.spans('italic'), ['斜体'])
        self.assertEqual(self.spans('code'), ['x_y = [1]'])
        link = next(entity for entity in self.payloads[0]['entities'] if entity.type == 'text_link')
        self.assertEqual(link.url, 'https://example.com/a?q=1&x=2')
        self.assertEqual(entity_text(self.payloads[0], link), '网站')
        self.assertIsNone(self.payloads[0].get('reply_to_message_id'), 'Group replies remain direct by default')

    async def test_placeholder_edit_has_same_formatting_as_new_message(self):
        renderer = self.renderer(placeholder=self.message(50), visibility=ProcessVisibility.STATUS)
        messages = await renderer.finalize('**Ready**: `a_b` 🙂')
        self.assertEqual([message.message_id for message in messages], [50])
        self.assertEqual(self.spans('bold'), ['Ready'])
        self.assertEqual(self.spans('code'), ['a_b'])
        self.bot.send_message.assert_not_awaited()

    async def test_malformed_markdown_does_not_flatten_valid_neighbouring_styles(self):
        await self.renderer().finalize('**Valid** [bad](broken url) and `code_[]` then *unfinished')
        self.assertEqual(self.payloads[0]['text'], 'Valid [bad](broken url) and code_[] then *unfinished')
        self.assertEqual(self.spans('bold'), ['Valid'])
        self.assertEqual(self.spans('code'), ['code_[]'])
        self.assertEqual(self.bot.send_message.await_count, 1, 'Malformed source syntax needs no failed Telegram request')

    async def test_long_bold_answer_keeps_every_character_and_style_across_chunks(self):
        body = ('strong text 🙂 ' * 500).rstrip()
        await self.renderer().finalize('**' + body + '**')
        self.assertGreater(len(self.payloads), 1)
        self.assertEqual(''.join(payload['text'] for payload in self.payloads), body)
        self.assertEqual(''.join(self.spans('bold')), body)

    async def test_long_code_keeps_language_indentation_backslashes_and_special_characters(self):
        line = '    print("你好🙂 _*[]()~`>#+-=|{}.!\\\\")\n'
        body = line * 230
        await self.renderer().finalize('````python\n' + body + '````')
        self.assertGreater(len(self.payloads), 1)
        # The Markdown parser removes the single structural newline before the fence.
        self.assertEqual(''.join(payload['text'] for payload in self.payloads), body.rstrip('\n'))
        self.assertEqual(''.join(self.spans('pre')), body.rstrip('\n'))
        self.assertTrue(all(entity.language == 'python' for payload in self.payloads
            for entity in payload['entities'] if entity.type == 'pre'))

    async def test_emoji_heavy_reply_is_split_using_telegram_utf16_units(self):
        body = '🙂' * 2500
        await self.renderer().finalize(body)
        self.assertEqual(len(self.payloads), 2)
        self.assertEqual(''.join(payload['text'] for payload in self.payloads), body)

    async def test_long_link_keeps_destination_on_every_chunk(self):
        label = '详细说明🙂' * 950
        await self.renderer().finalize('[' + label + '](https://example.com/help)')
        self.assertGreater(len(self.payloads), 1)
        self.assertEqual(''.join(self.spans('text_link')), label)
        self.assertTrue(all(entity.url == 'https://example.com/help' for payload in self.payloads
            for entity in payload['entities'] if entity.type == 'text_link'))

    async def test_full_progress_keeps_code_style_when_a_later_event_appends(self):
        renderer = self.renderer(placeholder=self.message(50), visibility=ProcessVisibility.FULL)
        await renderer.emit(RuntimeEvent(kind='tool', title='Inspection', detail='```python\n' + 'print(1)\n' * 600 + '```'))
        await renderer.emit(RuntimeEvent(kind='phase', title='Ready', detail='**Checked**'))
        # The last edited message combines an already-rendered continuation with a
        # new event. It must not reinterpret code underscores/stars as Markdown.
        latest = self.payloads[-1]
        self.assertIn('print(1)', latest['text'])
        self.assertTrue(any(entity.type == 'pre' for entity in latest['entities']))
        self.assertIn('Checked', self.spans('bold'))
        await renderer.finalize('**Done**')
        self.assertIn('Done', self.spans('bold'))

    async def test_actual_delivery_failure_and_cancellation_do_not_trigger_plaintext_resends(self):
        for failure in (BadRequest("Can't parse entities: synthetic rejected entity"),
                        BadRequest('Chat not found'), TimedOut(), asyncio.CancelledError()):
            with self.subTest(failure=type(failure).__name__):
                self.bot.send_message.reset_mock()
                self.bot.send_message.side_effect = failure
                with self.assertRaises(type(failure)):
                    await self.renderer().finalize('**Important**')
                self.assertEqual(self.bot.send_message.await_count, 1)

    async def test_failed_placeholder_edit_can_send_formatted_answer_without_flattening(self):
        placeholder = self.message(50)
        placeholder.edit_text.side_effect = BadRequest('Message to edit not found')
        messages = await self.renderer(placeholder=placeholder, visibility=ProcessVisibility.STATUS).finalize('**Ready**')
        self.assertEqual(len(messages), 1)
        self.assertNotEqual(messages[0].message_id, 50)
        self.assertEqual(self.spans('bold'), ['Ready'])
        placeholder.delete.assert_awaited_once()

    async def test_status_only_final_edit_failure_does_not_block_final_new_answer(self):
        placeholder = self.message(50)
        placeholder.edit_text.side_effect = TimedOut()
        renderer = self.renderer(placeholder=placeholder, visibility=ProcessVisibility.STATUS,
            delivery=ResponseDelivery.FINAL_NEW)
        renderer.state.lines = ['Status: finished']
        messages = await renderer.finalize('**Answer**')
        self.assertEqual(len(messages), 1)
        self.assertEqual(self.spans('bold'), ['Answer'])

    async def test_cancellation_during_status_finalization_still_stops_answer_delivery(self):
        placeholder = self.message(50)
        placeholder.edit_text.side_effect = asyncio.CancelledError()
        renderer = self.renderer(placeholder=placeholder, visibility=ProcessVisibility.STATUS,
            delivery=ResponseDelivery.FINAL_NEW)
        with self.assertRaises(asyncio.CancelledError):
            await renderer.finalize('**Answer**')
        self.bot.send_message.assert_not_awaited()

    async def test_chunked_replacement_send_failure_does_not_resend_delivered_chunks(self):
        placeholder = self.message(50)
        renderer = self.renderer(placeholder=placeholder, visibility=ProcessVisibility.STATUS)
        self.bot.send_message.side_effect = BadRequest('Chat not found')
        with self.assertRaises(BadRequest):
            await renderer._replace_message_with_chunked_text('**' + 'detail ' * 1500 + 'end**')
        placeholder.edit_text.assert_awaited_once()
        self.assertEqual(self.bot.send_message.await_count, 1,
            'A failed continuation is delivery failure, not evidence that the preceding edit failed')

    async def test_python_telegram_bot_serializes_native_entities_without_parse_mode(self):
        payloads = []

        class Request(BaseRequest):
            @property
            def read_timeout(self):
                return 1

            async def initialize(self):
                pass

            async def shutdown(self):
                pass

            async def do_request(self, url, method, request_data=None, **kwargs):
                payload = request_data.parameters
                payloads.append(payload)
                result = {'message_id': 1, 'date': 1, 'chat': {'id': 20, 'type': 'private'},
                    'text': payload['text'], 'entities': payload.get('entities', [])}
                return 200, json.dumps({'ok': True, 'result': result}).encode()

        bot = Bot('123:test-only-token', request=Request())
        message = await bot_message_safe(bot, 'send_message', chat_id=20,
            text='**粗体🙂** and `x_y`', parse_mode='MarkdownV2')
        self.assertNotIn('parse_mode', payloads[0])
        self.assertEqual(message.text, '粗体🙂 and x_y')
        self.assertEqual(message.parse_entity(message.entities[0]), '粗体🙂')
        self.assertEqual(message.parse_entity(message.entities[1]), 'x_y')
        # Editing a real SDK Message must use the same serialization boundary.
        await bot_message_safe(message, 'edit_text', text='**Updated** 🙂', parse_mode='MarkdownV2')
        self.assertEqual(payloads[1]['message_id'], 1)
        self.assertEqual(payloads[1]['text'], 'Updated 🙂')
        self.assertEqual(payloads[1]['entities'][0]['type'], 'bold')
        self.assertNotIn('parse_mode', payloads[1])
