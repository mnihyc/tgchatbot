"""Delivered tool-step speech is searchable without duplicating model history."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
from telegram.error import BadRequest

from tests.business_helpers import BusinessTestCase, ScriptedProvider
from tgchatbot.core.memory import MemoryService
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import (ChatMode, ConversationMessage, MessageRole,
    ProcessVisibility, ProviderResponse, ToolCall, ToolHistoryMode)
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.storage.postgres_store import message_body
from tgchatbot.transports.telegram_adapter import ReplyCandidate, TelegramBotApp


def model_step(text, *call_ids):
    return {'role': 'model', 'parts': [
        {'text': text, 'thoughtSignature': 'fixture-text-signature-' + text},
        *({'functionCall': {'name': 'shell_exec', 'args': {}, 'id': call_id},
            'thoughtSignature': 'fixture-call-signature-' + call_id} for call_id in call_ids)]}


class AssistantSpokenMemoryTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.selected = await self.settings(provider='gemini', model='gemini-3.8-flash',
            mode=ChatMode.AGENT, process_visibility=ProcessVisibility.OFF,
            tool_history_mode=ToolHistoryMode.NATIVE_SAME_PROVIDER, provider_retry_count=0)
        self.native = [model_step('Saffron note before checking.', 'check-a', 'check-b'),
            model_step('Cardamom final answer.')]
        self.wire = []

        def respond(request):
            self.wire.append(json.loads(request.content))
            if not self.native:
                raise AssertionError('Unexpected model request')
            item = self.native.pop(0)
            candidate = item if 'finishReason' in item else {'finishReason': 'STOP', 'content': item}
            return httpx.Response(200, json={'candidates': [candidate]})

        self.gemini = GeminiProvider(replace(self.config.gemini, api_key='fixture-key'))
        await self.gemini.aclose()
        self.gemini._client = httpx.AsyncClient(base_url='https://fixture.invalid/',
            transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(self.gemini.aclose)
        self.runtime.providers['gemini'] = self.gemini
        self.memory = MemoryService(self.store, SimpleNamespace(enabled=False))
        self.app = TelegramBotApp.__new__(TelegramBotApp)
        self.app.config, self.app.runtime, self.app.store = self.config, self.runtime, self.store
        self.app._chat_states = {}
        self.app._notify_user_error = AsyncMock()
        self.chat = SimpleNamespace(id=100)
        self.sent = []
        self.bot = SimpleNamespace(id=999, username='fixture_bot', send_chat_action=AsyncMock())

        async def send(**kwargs):
            self.sent.append(kwargs['text'])
            return self.telegram_message(1000 + len(self.sent))

        self.successful_send = send
        self.bot.send_message = AsyncMock(side_effect=send)

    def telegram_message(self, number, actor=999):
        return SimpleNamespace(message_id=number, chat=self.chat,
            from_user=SimpleNamespace(id=actor, full_name='Fixture Bot' if actor == 999 else 'Participant',
                is_bot=actor == 999), date=datetime(2026, 1, 2, tzinfo=timezone.utc),
            get_bot=lambda: self.bot, delete=AsyncMock(), edit_text=AsyncMock())

    async def reply(self, number=10):
        incoming = await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text('Check the operation.', metadata={
                'source': 'telegram', 'source_chat_id': '100', 'source_message_id': str(number),
                'actor_id': 'telegram:user:7', 'actor_kind': 'user', 'actor_name': 'Participant'}))
        await self.app._reply_to_candidate(ReplyCandidate(incoming.db_id, 'Participant',
            self.telegram_message(number, actor=7)))
        return incoming

    async def originals(self):
        return await self.store.list_canonical_messages(self.session)

    async def assistants(self):
        return [row for row in await self.originals() if row.message.role == MessageRole.ASSISTANT]

    async def test_delivery_indexes_bot_speech_once_and_reconstruction_preserves_native_batch(self):
        native = copy.deepcopy(self.native)
        await self.reply()
        self.app._notify_user_error.assert_not_awaited()
        self.assertEqual(self.sent, ['Saffron note before checking.', 'Cardamom final answer.'])
        answers = await self.assistants()
        self.assertEqual([message_body(row.message) for row in answers], self.sent)
        self.assertEqual([row.message.metadata['source_message_id'] for row in answers], ['1001', '1002'])
        self.assertTrue(all(row.message.metadata['actor_id'] == 'telegram:user:999'
            and row.message.metadata['actor_kind'] == 'bot' for row in answers))
        recalled = await self.memory.search(self.session, 'saffron')
        self.assertEqual(recalled['matches'][0]['message_ids'], [answers[0].db_id])
        read = await self.memory.read(self.session, [answers[0].db_id])
        self.assertEqual(read['messages'][0]['fragments'], [{'offset': 0, 'text': self.sent[0]}])
        self.assertEqual(read['messages'][0]['role'], 'assistant')
        self.assertEqual(read['messages'][0]['speaker']['kind'], 'bot')
        async with self.store.pool.connection() as conn:
            jobs = await (await conn.execute("SELECT source_ids FROM jobs WHERE kind='memory_ingest'")).fetchall()
            profiles = await (await conn.execute('SELECT message_id FROM profile_inputs')).fetchall()
        self.assertTrue(all(any(row.db_id in job['source_ids'] for job in jobs) for row in answers))
        self.assertFalse({row.db_id for row in answers} & {row['message_id'] for row in profiles})
        self.assertEqual([item for item in self.wire[1]['contents'] if item == native[0]], [native[0]])

        hot = await self.runtime._get_live_state(self.session)
        hot_rows, tokens = copy.deepcopy(hot.raw_messages), hot.estimated_tokens
        self.assertNotIn(answers[0].db_id, [row.db_id for row in hot_rows])
        self.runtime.invalidate_session(self.session)
        cold = await self.runtime._get_live_state(self.session)
        self.assertEqual(cold.raw_messages, hot_rows)
        self.assertEqual(cold.estimated_tokens, tokens)
        preparation = await self.store.load_compaction_window(self.session)
        self.assertEqual(preparation.raw_messages, hot_rows)
        counts = await self.store.context_preparation_counts(self.session, answers[-1].db_id)
        self.assertEqual(counts['remaining_raw_messages'], len(hot_rows))
        self.assertIn(answers[0].db_id, [row.db_id for row in
            await self.store.list_recent_visible_messages(self.session)])
        summary_input = '\n'.join(message_body(message) for message in
            self.runtime._normalize_compaction_messages([row.message for row in cold.raw_messages]))
        self.assertEqual(summary_input.count('Saffron note before checking.'), 1)

        self.native.append(model_step('Restart continuation.'))
        await self.reply(11)
        for original in native:
            self.assertEqual(sum(item == original for item in self.wire[-1]['contents']), 1)

    async def test_provider_switch_keeps_spoken_text_once(self):
        await self.reply()
        other = ScriptedProvider(name='openai', responses=[ProviderResponse(final_text='Portable continuation.')])
        await self.settings(provider='openai', model='fixture-chat')
        reopened = await self.new_store()
        runtime = AgentRuntime(config=self.config, store=reopened, tool_registry=self.tools,
            providers={'openai': other})
        await runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Continue.'))
        text = '\n'.join(message_body(message) for message in other.requests[0]['messages'])
        self.assertEqual(text.count('Saffron note before checking.'), 1)
        self.assertEqual(text.count('Cardamom final answer.'), 1)

    async def test_identical_spoken_sentences_remain_distinct_occurrences(self):
        self.native = [model_step('Saffron note before checking.', 'check-a'),
            model_step('Saffron note before checking.', 'check-b'), model_step('Done.')]
        await self.reply()
        answers = await self.assistants()
        self.assertEqual([message_body(row.message) for row in answers[:2]], [self.sent[0]] * 2)
        self.assertNotEqual(answers[0].db_id, answers[1].db_id)
        self.assertNotEqual(answers[0].message.metadata['context_owner_message_id'],
            answers[1].message.metadata['context_owner_message_id'])
        recalled = await self.memory.search(self.session, 'saffron')
        self.assertEqual({match['message_ids'][0] for match in recalled['matches']},
            {row.db_id for row in answers[:2]})
        state = await self.runtime._get_live_state(self.session)
        text = '\n'.join(message_body(message) for message in
            self.runtime._normalize_compaction_messages([row.message for row in state.raw_messages]))
        self.assertEqual(text.count('Saffron note before checking.'), 2)

    async def test_failed_later_request_keeps_already_delivered_speech_searchable(self):
        self.native[-1] = {'finishReason': 'MAX_TOKENS', 'content': model_step('Unfinished answer.')}
        with self.assertLogs('tgchatbot.transports.telegram_adapter', level='ERROR'):
            await self.reply()
        answers = await self.assistants()
        self.assertEqual([message_body(row.message) for row in answers], ['Saffron note before checking.'])
        self.assertEqual(len((await self.memory.search(self.session, 'saffron'))['matches']), 1)
        self.assertEqual(len(self.wire), 2)
        self.app._notify_user_error.assert_awaited_once()
        self.native.append(model_step('Recovered on the next message.'))
        await self.reply(11)
        self.assertEqual([message_body(row.message) for row in await self.assistants()],
            ['Saffron note before checking.', 'Recovered on the next message.'])
        self.assertEqual(self.tools.runner.run.await_count, 2, 'Completed tool work must not execute again')
        self.assertEqual(sum(part.get('text') == 'Saffron note before checking.'
            for item in self.wire[-1]['contents'] for part in item['parts']), 1)

    async def test_rejected_speech_delivery_does_not_invent_a_delivered_original(self):
        self.bot.send_message.side_effect = BadRequest('fixture rejection')
        with self.assertLogs('tgchatbot.transports.telegram_adapter', level='ERROR'):
            await self.reply()
        self.assertEqual(await self.assistants(), [])
        self.assertEqual((await self.memory.search(self.session, 'saffron'))['matches'], [])
        self.tools.runner.run.assert_not_awaited()
        self.assertEqual(len(self.wire), 1)
        self.bot.send_message.side_effect = self.successful_send
        self.native = [model_step('Recovered after the rejected delivery.')]
        await self.reply(11)
        self.assertEqual([message_body(row.message) for row in await self.assistants()],
            ['Recovered after the rejected delivery.'])
        self.assertEqual(await self.store.list_unfinished_tool_calls(self.session), [])
        self.tools.runner.run.assert_not_awaited()
        self.assertEqual((await self.memory.search(self.session, 'saffron'))['matches'], [])
        self.assertEqual(len(self.wire), 2)
        self.app._notify_user_error.assert_awaited_once()

    async def test_soft_reset_keeps_recall_and_full_reset_retires_it(self):
        await self.reply()
        answers = await self.assistants()
        await self.store.reset_context(self.session)
        self.runtime.invalidate_session(self.session)
        self.assertEqual((await self.runtime._get_live_state(self.session)).raw_messages, [])
        self.assertEqual(len((await self.memory.search(self.session, 'saffron'))['matches']), 1)
        await self.store.reset_full(self.session, self.config.default_session_settings())
        self.assertEqual((await self.memory.search(self.session, 'saffron'))['matches'], [])
        self.assertEqual(await self.store.read_messages(self.session, [answers[0].db_id]), [])
        async with self.store.pool.connection() as conn:
            count = (await (await conn.execute('SELECT count(*) AS n FROM messages WHERE id=ANY(%s)',
                ([row.db_id for row in answers],))).fetchone())['n']
        self.assertEqual(count, len(answers), 'Reset preserves canonical audit evidence')

    async def test_withdrawing_spoken_source_changes_only_its_shared_native_batch(self):
        first, second = model_step('Saffron first observation.', 'first'), model_step('Cinnamon next observation.', 'next-a', 'next-b')
        final = model_step('Cardamom final answer.')
        self.native = copy.deepcopy([first, second, final])
        await self.reply()
        answers = await self.assistants()
        state = await self.runtime._get_live_state(self.session)
        owner = next(row for row in state.raw_messages
            if row.db_id == answers[0].message.metadata['context_owner_message_id'])
        await self.store.create_memory_block(self.session, summary_text='Saffron was mentioned.',
            estimated_tokens=10, source_message_ids=[owner.db_id])
        await self.store.hide_message_ids(self.session, [answers[0].db_id])
        self.runtime.invalidate_session(self.session)
        self.assertEqual((await self.memory.search(self.session, 'saffron'))['matches'], [])
        cold = await self.runtime._get_live_state(self.session)
        self.assertFalse(cold.blocks, 'The summary of the shared prose must be invalidated too')
        self.native.append(model_step('Continuation after withdrawal.'))
        await self.reply(11)
        contents = self.wire[-1]['contents']
        self.assertNotIn('Saffron first observation.', json.dumps(contents))
        self.assertEqual(sum(item == second for item in contents), 1)
        self.assertEqual(sum(item == final for item in contents), 1)
        self.assertTrue(any(part.get('functionCall', {}).get('id') == 'first'
            for item in contents for part in item['parts']), 'Withdrawing prose keeps the actual tool operation')
        self.app._notify_user_error.assert_not_awaited()

    async def test_rollback_keeps_delivered_speech_in_the_visible_bot_block(self):
        incoming = await self.reply()
        answers = await self.assistants()
        targets = await self.app._collect_rollback_message_ids(self.session, 1)
        self.assertTrue({row.db_id for row in answers}.issubset(targets))
        self.assertNotIn(incoming.db_id, targets)
        await self.store.hide_message_ids(self.session, targets)
        self.runtime.invalidate_session(self.session)
        self.assertEqual((await self.memory.search(self.session, 'saffron'))['matches'], [])
        self.assertEqual([row.db_id for row in (await self.runtime._get_live_state(self.session)).raw_messages],
            [incoming.db_id])

    async def test_existing_schema_initialization_adds_relation_without_rewriting_originals(self):
        source = await self.store.append_message(self.session, ConversationMessage.user_text('Existing archive.'))
        before = await self.store.read_messages(self.session, [source.db_id])
        async with self.store.pool.connection() as conn:
            # This isolated fixture models schema-v3 installations created
            # before the additive replay relation existed.
            await conn.execute('DROP TABLE message_replay_owners')
        reopened = await self.new_store()
        self.assertEqual(await reopened.read_messages(self.session, [source.db_id]), before)
        async with reopened.pool.connection() as conn:
            count = (await (await conn.execute('SELECT count(*) AS n FROM message_replay_owners')).fetchone())['n']
        self.assertEqual(count, 0)
        await self.reply()
        self.assertEqual(len((await self.memory.search(self.session, 'saffron'))['matches']), 1)

    async def test_without_native_text_the_delivered_original_owns_replay_itself(self):
        provider = ScriptedProvider(name='gemini', responses=[
            ProviderResponse(final_text='Saffron ordinary speech.', tool_calls=[ToolCall('shell_exec', 'check', {})]),
            ProviderResponse(final_text='Done.')])
        self.runtime.providers['gemini'] = provider
        await self.reply()
        answers = await self.assistants()
        self.assertNotIn('context_owner_message_id', answers[0].message.metadata)
        self.runtime.invalidate_session(self.session)
        state = await self.runtime._get_live_state(self.session)
        self.assertIn(answers[0].db_id, [row.db_id for row in state.raw_messages])
        history = self.runtime._build_provider_history(state, settings=self.selected, provider_name='gemini')
        self.assertEqual('\n'.join(message_body(message) for message in history).count('Saffron ordinary speech.'), 1)
        self.assertEqual(len((await self.memory.search(self.session, 'saffron'))['matches']), 1)
