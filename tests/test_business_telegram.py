from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from tests.business_helpers import BusinessTestCase, ScriptedProvider
from tgchatbot.domain.models import (
    ChatMode, ConversationMessage, MessageRole, OutboundArtifact, OutboundSticker,
    ProcessVisibility, StickerTiming, TurnResult,
)
from tgchatbot.transports.telegram_adapter import ReplyCandidate, TelegramBotApp


class TelegramWorkflowTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        # Bypass only Application construction, which would initialize a live bot.
        self.app = TelegramBotApp.__new__(TelegramBotApp)
        self.app.config = self.config
        self.app.runtime = self.runtime
        self.app.store = self.store
        self.app._chat_states = {}
        self.app.artifact_store = Mock()
        self.bot = SimpleNamespace(id=999, send_chat_action=AsyncMock(), send_sticker=AsyncMock(return_value=SimpleNamespace(message_id=55)), send_photo=AsyncMock(), send_document=AsyncMock())
        self.chat = SimpleNamespace(id=100, type="group")
        self.message = SimpleNamespace(chat=self.chat, message_id=10, text="hello", caption=None, entities=[], reply_to_message=None, reply_text=AsyncMock(), get_bot=lambda: self.bot)
        self.update = SimpleNamespace(effective_chat=self.chat, effective_message=self.message, effective_user=SimpleNamespace(id=7, username="tester", full_name="Test User"))
        self.context = SimpleNamespace(args=[], bot=self.bot)

    def telegram_config(self, **changes):
        self.app.config = replace(self.config, telegram=replace(self.config.telegram, **changes))

    def candidate(self, number=1, spontaneous=False):
        return ReplyCandidate(number, "tester", self.message, spontaneous=spontaneous)

    async def test_whitelist_and_trusted_controls_are_separate(self):
        self.telegram_config(whitelist=("100",), control_uids=("8",))
        self.assertTrue(self.app._command_allowed(self.update))
        self.assertFalse(self.app._advanced_allowed(self.update))
        self.update.effective_user.id = 8
        self.assertTrue(self.app._advanced_allowed(self.update))
        self.chat.id = 200
        self.assertFalse(self.app._command_allowed(self.update))
        self.assertFalse(self.app._advanced_allowed(self.update))

    async def test_empty_whitelists_preserve_open_legacy_behavior(self):
        self.telegram_config(whitelist=(), control_uids=())
        self.assertTrue(self.app._allowed(self.chat))
        self.assertTrue(self.app._advanced_allowed(self.update))

    async def test_denied_chat_cannot_change_session_mode(self):
        self.telegram_config(whitelist=("200",))
        self.context.args = ["agent"]
        await self.app.mode_command(self.update, self.context)
        self.assertEqual(await self.store.count_sessions(), 0)
        self.message.reply_text.assert_not_awaited()

    async def test_untrusted_user_cannot_change_advanced_parameters(self):
        self.telegram_config(control_uids=("8",))
        self.context.args = ["provider_retry_count", "9"]
        await self.app.param_command(self.update, self.context)
        self.assertIsNone((await self.settings()).provider_retry_count)

    async def test_group_keyword_and_bot_reply_trigger_but_ignore_wins(self):
        self.telegram_config(keywords=("helper",), ignore_keywords=("quiet",))
        for text, reply, expected in [
            ("HELPER please", None, True),
            ("hello", SimpleNamespace(from_user=SimpleNamespace(id=999)), True),
            ("helper quiet", None, False),
            ("quiet", SimpleNamespace(from_user=SimpleNamespace(id=999)), False),
            ("hello", None, False),
        ]:
            with self.subTest(text=text, reply=reply is not None):
                self.message.text, self.message.reply_to_message = text, reply
                plan = await self.app._compute_group_reply_plan(self.update, self.context)
                self.assertEqual(plan.should_reply, expected)
                self.assertEqual(plan.explicit, expected)

    async def test_group_media_and_caption_do_not_trigger_generation(self):
        self.message.text = "bot please reply"
        self.message.document = object()
        self.assertFalse((await self.app._compute_group_reply_plan(self.update, self.context)).should_reply)
        self.message.document = None
        self.message.caption = "caption"
        self.assertFalse((await self.app._compute_group_reply_plan(self.update, self.context)).should_reply)

    async def test_debounce_only_promotes_newest_token(self):
        self.app._set_latest_reply_candidate = AsyncMock()
        state = self.app._flow_state(100)
        old_token = await self.app._next_reply_token(state)
        new_token = await self.app._next_reply_token(state)
        with patch("tgchatbot.transports.telegram_adapter.asyncio.sleep", new=AsyncMock()):
            await self.app._promote_candidate_after_delay(100, old_token, self.candidate(1), 5)
            await self.app._promote_candidate_after_delay(100, new_token, self.candidate(2), 5)
        self.app._set_latest_reply_candidate.assert_awaited_once()
        self.assertEqual(self.app._set_latest_reply_candidate.await_args.args[1].stored_message_id, 2)

    async def test_cancel_pending_reply_invalidates_sleeping_candidate(self):
        self.app._set_latest_reply_candidate = AsyncMock()
        state = self.app._flow_state(100)
        token = await self.app._next_reply_token(state)
        await self.app._cancel_pending_reply(100)
        with patch("tgchatbot.transports.telegram_adapter.asyncio.sleep", new=AsyncMock()):
            await self.app._promote_candidate_after_delay(100, token, self.candidate(), 5)
        self.app._set_latest_reply_candidate.assert_not_awaited()

    async def test_reset_history_invalidates_sleeping_candidate(self):
        self.app._set_latest_reply_candidate = AsyncMock()
        token = await self.app._next_reply_token(self.app._flow_state(100))
        self.context.args = ["history"]
        await self.app.reset_command(self.update, self.context)
        with patch("tgchatbot.transports.telegram_adapter.asyncio.sleep", new=AsyncMock()):
            await self.app._promote_candidate_after_delay(100, token, self.candidate(), 5)
        self.app._set_latest_reply_candidate.assert_not_awaited()

    async def test_spontaneous_probability_is_sampled_after_idle_delay(self):
        await self.settings(spontaneous_reply_chance=25)
        self.app._set_latest_reply_candidate = AsyncMock()
        token = await self.app._next_reply_token(self.app._flow_state(100))
        with patch("tgchatbot.transports.telegram_adapter.asyncio.sleep", new=AsyncMock()) as sleep:
            with patch("tgchatbot.transports.telegram_adapter.random.random", return_value=0.24):
                await self.app._promote_spontaneous_candidate_after_delay(100, self.session, token, self.candidate(spontaneous=True), 1200)
            self.app._set_latest_reply_candidate.assert_awaited_once()
            self.app._set_latest_reply_candidate.reset_mock()
            with patch("tgchatbot.transports.telegram_adapter.random.random", return_value=0.25):
                await self.app._promote_spontaneous_candidate_after_delay(100, self.session, token, self.candidate(spontaneous=True), 1200)
            self.app._set_latest_reply_candidate.assert_not_awaited()
            sleep.assert_awaited_with(1200)

    async def test_new_message_cancels_spontaneous_but_keeps_explicit_candidate(self):
        state = self.app._flow_state(100)
        state.latest_reply_candidate = self.candidate(1, spontaneous=True)
        await self.app._clear_pending_spontaneous_candidate(100, newer_message_id=2)
        self.assertIsNone(state.latest_reply_candidate)
        explicit = self.candidate(2)
        state.latest_reply_candidate = explicit
        await self.app._clear_pending_spontaneous_candidate(100, newer_message_id=3)
        self.assertIs(state.latest_reply_candidate, explicit)

    async def test_worker_waits_for_ingestion_and_does_not_reply_twice(self):
        state = self.app._flow_state(100)
        self.app._reply_to_candidate = AsyncMock()
        await self.app._mark_ingest_started(state)
        state.latest_reply_candidate = self.candidate(1)
        worker = asyncio.create_task(self.app._reply_worker(100))
        await asyncio.sleep(0)
        self.app._reply_to_candidate.assert_not_awaited()
        await self.app._mark_ingest_finished(state)
        await worker
        await self.app._reply_worker(100)
        self.app._reply_to_candidate.assert_awaited_once()
        self.assertEqual(state.last_replied_message_id, 1)

    async def test_retry_hides_newer_outputs_without_duplicating_user(self):
        trigger = await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage.user_text("question"))
        await self.runtime.record_tool_observation(session_id=self.session, name="shell_exec", payload={}, phase="result")
        await self.runtime.record_assistant_text(session_id=self.session, text="old answer")
        self.app._reply_to_candidate = AsyncMock()
        await self.app.retry_command(self.update, self.context)
        visible = await self.store.list_uncompacted_messages(self.session)
        self.assertEqual([item.db_id for item in visible], [trigger.db_id])
        self.assertEqual(self.app._reply_to_candidate.await_args.args[0].stored_message_id, trigger.db_id)
        self.assertNotIn(self.session, self.runtime._live_sessions)

    async def test_rollback_groups_tool_and_assistant_as_one_bot_block(self):
        first = await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage.user_text("question"))
        await self.runtime.record_tool_observation(session_id=self.session, name="shell_exec", payload={}, phase="result")
        await self.runtime.record_assistant_text(session_id=self.session, text="answer")
        await self.app.rollback_command(self.update, self.context)
        visible = await self.store.list_uncompacted_messages(self.session)
        self.assertEqual([item.db_id for item in visible], [first.db_id])

    async def test_retry_finds_user_before_more_than_one_page_of_tool_outputs(self):
        trigger = await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage.user_text("long task"))
        # 41 crosses the existing 40-row page boundary with the smallest fixture.
        for number in range(41):
            await self.runtime.record_tool_observation(session_id=self.session, name="shell_exec", payload={"sequence": number}, phase="result")
        self.app._reply_to_candidate = AsyncMock()
        await self.app.retry_command(self.update, self.context)
        self.assertEqual(self.app._reply_to_candidate.await_args.args[0].stored_message_id, trigger.db_id)
        self.assertEqual(len(await self.store.list_recent_visible_messages(self.session)), 1)

    async def test_rollback_hides_entire_bot_block_across_page_boundary(self):
        trigger = await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage.user_text("long task"))
        for number in range(41):
            await self.runtime.record_assistant_text(session_id=self.session, text=f"output {number}")
        await self.app.rollback_command(self.update, self.context)
        self.assertEqual([item.db_id for item in await self.store.list_recent_visible_messages(self.session)], [trigger.db_id])

    async def test_visible_history_cursor_does_not_cross_sessions_or_repeat_rows(self):
        first = await self.store.append_message(self.session, ConversationMessage.user_text("first"))
        await self.store.append_message("telegram:other", ConversationMessage.user_text("other chat"))
        last = await self.store.append_message(self.session, ConversationMessage.user_text("last"))
        page = await self.store.list_recent_visible_messages(self.session, limit=1)
        self.assertEqual([item.db_id for item in page], [last.db_id])
        page = await self.store.list_recent_visible_messages(self.session, limit=1, before_message_id=last.db_id)
        self.assertEqual([item.db_id for item in page], [first.db_id])

    async def test_reset_settings_preserves_history_and_reset_history_preserves_settings(self):
        await self.settings(mode=ChatMode.AGENT)
        await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage.user_text("history"))
        self.context.args = ["session"]
        await self.app.reset_command(self.update, self.context)
        self.assertEqual((await self.settings()).mode, ChatMode.CHAT)
        self.assertEqual(len(await self.store.list_uncompacted_messages(self.session)), 1)
        await self.settings(mode=ChatMode.ASSIST)
        self.context.args = ["history"]
        await self.app.reset_command(self.update, self.context)
        self.assertEqual((await self.settings()).mode, ChatMode.ASSIST)
        self.assertEqual(await self.store.list_uncompacted_messages(self.session), [])

    async def test_provider_command_switches_model_but_keeps_prompt_and_history(self):
        await self.settings(system_prompt="Preserve my voice", model="custom-old-model")
        await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage.user_text("prior context"))
        self.runtime.providers["gemini"] = ScriptedProvider("gemini")
        self.context.args = ["gemini"]
        await self.app.provider_command(self.update, self.context)
        settings = await self.settings()
        self.assertEqual(settings.provider, "gemini")
        self.assertEqual(settings.model, self.config.default_model_for_provider("gemini"))
        self.assertEqual(settings.system_prompt, "Preserve my voice")
        self.assertEqual(len(await self.store.list_uncompacted_messages(self.session)), 1)

    async def test_failed_delivery_does_not_commit_assistant_history(self):
        await self.settings(process_visibility=ProcessVisibility.OFF)
        self.runtime.run_turn_from_stored = AsyncMock(return_value=TurnResult("not delivered"))
        self.app._deliver_result = AsyncMock(side_effect=RuntimeError("Telegram unavailable"))
        self.app._notify_user_error = AsyncMock()
        await self.app._reply_to_candidate(self.candidate())
        self.assertEqual(await self.store.list_messages(self.session), [])
        self.app._notify_user_error.assert_awaited_once()

    async def test_successful_delivery_commits_assistant_history(self):
        await self.settings(process_visibility=ProcessVisibility.OFF)
        self.runtime.run_turn_from_stored = AsyncMock(return_value=TurnResult("delivered"))
        self.app._deliver_result = AsyncMock()
        await self.app._reply_to_candidate(self.candidate())
        self.assertEqual((await self.store.list_messages(self.session))[0].parts[0].text, "delivered")

    async def test_delivery_orders_answer_then_files_then_after_final_stickers(self):
        file_path = self.path / "report.txt"
        file_path.write_text("report")
        before = OutboundSticker(self.path / "before.webp", timing=StickerTiming.SEND_NOW)
        after = OutboundSticker(self.path / "after.webp", timing=StickerTiming.AFTER_FINAL)
        renderer = SimpleNamespace(finalize=AsyncMock(), send_artifacts=AsyncMock(), send_stickers=AsyncMock(return_value=[]))
        order = Mock()
        for name in ("finalize", "send_artifacts", "send_stickers"):
            order.attach_mock(getattr(renderer, name), name)
        settings = await self.settings(process_visibility=ProcessVisibility.STATUS)
        result = TurnResult("answer", artifacts=[OutboundArtifact(file_path, "report.txt")], stickers=[before, after])
        await self.app._deliver_result(self.message, renderer, settings, result, sent_before_receipts=[])
        self.assertEqual([call[0] for call in order.mock_calls], ["finalize", "send_artifacts", "send_stickers"])
        renderer.send_stickers.assert_awaited_once_with([after])

    async def test_sticker_receipts_distinguish_sent_missing_and_network_error(self):
        path = self.path / "sticker.webp"
        path.write_bytes(b"mock sticker")
        sticker = OutboundSticker(path, source_id="fixture-sticker")
        receipts = await self.app._send_stickers_direct(self.message, [sticker, OutboundSticker(self.path / "missing.webp")])
        self.assertTrue(receipts[0]["sent"])
        self.assertEqual(receipts[0]["telegram_message_id"], 55)
        self.assertFalse(receipts[1]["sent"])
        self.assertEqual(receipts[1]["error"], "missing_file")
        self.bot.send_sticker.side_effect = TimeoutError()
        receipt = (await self.app._send_stickers_direct(self.message, [sticker]))[0]
        self.assertFalse(receipt["sent"])
        self.assertEqual(receipt["error"], "TimeoutError")

    async def test_attachment_failure_keeps_caption_and_records_auto_note(self):
        self.message.text = None
        self.message.caption = "Please inspect this"
        with patch("tgchatbot.transports.telegram_adapter.extract_message_parts", new=AsyncMock(side_effect=TimeoutError())):
            parts = await self.app._safe_extract_parts(self.message, self.session)
        self.assertEqual(parts[0].text, "Please inspect this")
        self.assertIn("Attachment download failed", parts[1].text)
        self.assertEqual(parts[1].origin, "auto_note")
