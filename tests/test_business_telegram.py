from __future__ import annotations

import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from tests.business_helpers import BusinessTestCase, ScriptedProvider
from tgchatbot.domain.models import (
    ChatMode, ConversationMessage, MessageRole, OutboundArtifact, OutboundSticker,
    ProcessVisibility, ProviderResponse, StickerTiming, TurnResult,
)
from tgchatbot.transports.telegram_adapter import ReplyCandidate, TelegramBotApp
from tgchatbot.transports.telegram_command_views import plain_text
from tgchatbot.stickers.catalog import StickerCatalog
from tgchatbot.stickers.plan import StickerRetrievalPlan
from tgchatbot.storage.postgres_store import StaleScopeError


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

    async def test_operational_settings_preserve_requested_limits_without_unrelated_caps(self):
        await self.settings()
        for name, value in {'max_output_tokens': 131072, 'max_input_images': 200000,
                'provider_retry_count': 8, 'max_interaction_rounds': 100,
                'private_reply_delay_s': 901, 'group_spontaneous_reply_delay_s': 100000,
                'compact_trigger_tokens': 20000000, 'compact_min_messages': 2000,
                'compact_idle_trigger_tokens': 15000000, 'compact_idle_seconds': 7200,
                'compact_batch_tokens': 1000000}.items():
            with self.subTest(setting=name):
                self.context.args = [name, str(value)]
                await self.app.param_command(self.update, self.context)
                restored = await (await self.new_store()).get_or_create_session(self.session,
                    self.config.default_session_settings())
                self.assertEqual(getattr(restored, name), value)

    async def test_preset_name_cannot_read_outside_the_preset_directory(self):
        from tgchatbot.storage.presets import PresetStore
        await self.settings()
        self.app.preset_store = PresetStore(self.path / 'presets')
        outside = self.path / 'private.txt'
        outside.write_text('Private local text')
        (self.path / 'presets' / 'linked.txt').symlink_to(outside)
        for name in ('../private', 'linked'):
            self.context.args = [name]
            await self.app.preset_command(self.update, self.context)
            self.assertIn('Preset not found', self.message.reply_text.call_args.args[0])
        with self.assertRaises(ValueError):
            self.app.preset_store.save_text('linked', 'Replacement')
        self.assertEqual(outside.read_text(), 'Private local text')
        self.app.preset_store.save_text('quiet', 'Answer calmly.')
        self.context.args = ['quiet']
        await self.app.preset_command(self.update, self.context)
        self.assertIn('Preset loaded', self.message.reply_text.call_args.args[0])

    async def test_nonfinite_sampling_and_invalid_ratio_leave_settings_intact(self):
        await self.settings(temperature=0.5, top_p=0.8, compact_tool_ratio_threshold=10.0)
        self.provider.describe_controls = lambda settings: {
            name: SimpleNamespace(supported=True) for name in ('temperature', 'top_p')}
        for name, value in (('temperature', 'nan'), ('top_p', 'nan'),
                            ('compact_tool_ratio_threshold', 'invalid')):
            self.context.args = [name, value]
            await self.app.param_command(self.update, self.context)
            self.assertIn('Invalid', self.message.reply_text.call_args.args[0])
        restored = await self.store.get_or_create_session(self.session, self.config.default_session_settings())
        self.assertEqual((restored.temperature, restored.top_p, restored.compact_tool_ratio_threshold),
                         (0.5, 0.8, 10.0))

    async def test_mode_command_obeys_chat_access_independently_of_trusted_users(self):
        self.telegram_config(whitelist=("200",), control_uids=("7",))
        self.context.args = ["agent"]
        await self.app.mode_command(self.update, self.context)
        self.assertEqual(await self.store.count_sessions(), 0)

        for whitelist, controls in [(("100",), ("8",)), ((), ())]:
            with self.subTest(whitelist=whitelist, controls=controls):
                await self.settings(mode=ChatMode.CHAT)
                self.telegram_config(whitelist=whitelist, control_uids=controls)
                await self.app.mode_command(self.update, self.context)
                self.assertEqual((await self.settings()).mode, ChatMode.AGENT)

    async def test_advanced_parameter_changes_require_both_chat_and_user_access(self):
        self.context.args = ["provider_retry_count", "2"]
        cases = [
            (("100",), ("8",), 7, False),
            (("100",), ("8",), 8, True),
            (("200",), ("8",), 8, False),
            ((), (), 7, True),
        ]
        for whitelist, controls, user_id, allowed in cases:
            with self.subTest(whitelist=whitelist, controls=controls, user_id=user_id):
                await self.settings(provider_retry_count=0)
                self.telegram_config(whitelist=whitelist, control_uids=controls)
                self.update.effective_user.id = user_id
                await self.app.param_command(self.update, self.context)
                self.assertEqual((await self.settings()).provider_retry_count, 2 if allowed else 0)

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
        self.app._reply_to_candidate = AsyncMock(return_value=None)
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
        self.app._reply_to_candidate = AsyncMock(return_value=None)
        await self.app.retry_command(self.update, self.context)
        visible = await self.store.list_uncompacted_messages(self.session)
        self.assertEqual([item.db_id for item in visible], [trigger.db_id])
        self.assertEqual(self.app._reply_to_candidate.await_args.args[0].stored_message_id, trigger.db_id)
        self.assertNotIn(self.session, self.runtime._live_sessions)

    async def test_retry_preserves_original_author_when_someone_else_issues_command(self):
        trigger = await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text("Original question", metadata={
                "source": "telegram", "source_chat_id": "100", "source_message_id": "42",
                "actor_id": "telegram:user:8", "actor_kind": "user", "actor_name": "Original author"}))
        await self.runtime.record_assistant_text(session_id=self.session, text="old answer")
        self.app._reply_to_candidate = AsyncMock(return_value=None)
        self.update.effective_user.full_name = "Command issuer"
        await self.app.retry_command(self.update, self.context)
        candidate = self.app._reply_to_candidate.await_args.args[0]
        self.assertEqual(candidate.stored_message_id, trigger.db_id)
        self.assertEqual(candidate.user_display_name, "Original author")
        original = (await self.store.read_messages(self.session, [trigger.db_id]))[0]
        self.assertEqual(original.message.metadata["actor_id"], "telegram:user:8")

    async def test_retry_selects_original_question_before_persisted_target_and_framework_notes(self):
        trigger = await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text("Original question", metadata={
                "source": "telegram", "source_chat_id": "100", "source_message_id": "42",
                "actor_id": "telegram:user:8", "actor_kind": "user", "actor_name": "Original author"}))
        for phase, details in (("call", {"arguments": {"actor_ids": ["telegram:user:8"]}}),
                               ("result", {"output": {"ok": True, "profiles": []}})):
            await self.runtime.record_tool_observation(session_id=self.session, name="user_profile_fetch",
                phase=phase, payload={"call_id": "refresh-retry", **details},
                metadata_update={"synthetic_role": "profile_refresh", "refresh_reason": "compaction"})
        self.provider.responses = [ProviderResponse(final_text="Old answer.")]
        await self.runtime.run_turn_from_stored(session_id=self.session, user_display_name="Original author",
                                              trigger_message_id=trigger.db_id)
        await self.runtime.record_assistant_text(session_id=self.session, text="Old answer.")
        # A late framework note is also USER-shaped; neither it nor the durable
        # reply target is an original eligible for /retry.
        await self.runtime.record_auto_user_note(session_id=self.session,
            parts=ConversationMessage.user_text("Automatic attachment status.").parts)
        self.app._reply_to_candidate = AsyncMock(return_value=None)
        await self.app.retry_command(self.update, self.context)
        candidate = self.app._reply_to_candidate.await_args.args[0]
        self.assertEqual(candidate.stored_message_id, trigger.db_id)
        self.assertEqual(candidate.user_display_name, "Original author")
        self.assertEqual([item.db_id for item in await self.store.list_uncompacted_messages(self.session)],
                         [trigger.db_id])

    async def test_rollback_does_not_count_profile_refresh_or_reply_target_as_extra_blocks(self):
        self.provider.responses = [ProviderResponse(final_text="First answer."), ProviderResponse(final_text="Second answer.")]
        originals = []
        first_turn_ids = []
        for number in (1, 2):
            source = await self.runtime.ingest_user_message(session_id=self.session,
                incoming_message=ConversationMessage.user_text(f"Question {number}"))
            originals.append(source.db_id)
            for phase, details in (("call", {"arguments": {"actor_ids": []}}),
                                   ("result", {"output": {"ok": True, "profiles": []}})):
                await self.runtime.record_tool_observation(session_id=self.session, name="user_profile_fetch",
                    phase=phase, payload={"call_id": f"refresh-{number}", **details},
                    metadata_update={"synthetic_role": "profile_refresh", "refresh_reason": "compaction"})
            result = await self.runtime.run_turn_from_stored(session_id=self.session,
                user_display_name="tester", trigger_message_id=source.db_id)
            await self.runtime.record_assistant_text(session_id=self.session, text=result.text)
            if number == 1:
                first_turn_ids = [item.db_id for item in await self.store.list_uncompacted_messages(self.session)]
        await self.app.rollback_command(self.update, self.context)
        self.assertEqual([item.db_id for item in await self.store.list_uncompacted_messages(self.session)],
                         [*first_turn_ids, originals[1]])
        await self.app.rollback_command(self.update, self.context)
        self.assertEqual([item.db_id for item in await self.store.list_uncompacted_messages(self.session)], first_turn_ids)
        await self.app.rollback_command(self.update, self.context)
        self.assertEqual([item.db_id for item in await self.store.list_uncompacted_messages(self.session)], [originals[0]])

    async def test_rollback_failed_generation_hides_question_and_its_control_record_together(self):
        self.provider.responses = [TimeoutError("Model unavailable")]
        with self.assertRaises(TimeoutError):
            await self.runtime.run_turn(session_id=self.session, user_display_name="tester",
                incoming_message=ConversationMessage.user_text("Unanswered question."))
        before = await self.store.list_uncompacted_messages(self.session)
        self.assertEqual(len(before), 2)
        self.assertEqual(before[-1].message.metadata.get("synthetic_role"), "reply_target")
        await self.app.rollback_command(self.update, self.context)
        self.assertEqual(await self.store.list_uncompacted_messages(self.session), [])

    async def test_full_reset_preserves_other_chat_persona_and_rejects_a_stale_tool_write(self):
        catalog = StickerCatalog(None, self.path / "stickers", persona_store=self.store)
        catalog._loaded = True
        catalog.entries_by_id = {"shared-sticker": object()}
        self.tools.sticker_catalog = catalog
        persona = {"affect_profile": {"default_tone": "warm"}}
        await self.store.save_sticker_persona(self.session, persona)
        await self.store.save_sticker_persona("telegram:200", persona)
        self.assertEqual((await catalog.adescribe_persona_context(self.session))['effective_persona'], persona)
        self.assertEqual((await catalog.adescribe_persona_context("telegram:200"))['effective_persona'], persona)
        catalog.style_memory.preload(self.session, recent_sticker_ids=["old-choice"])
        scope = await self.store.get_scope(self.session)
        self.context.args = ["all"]
        await self.app.reset_command(self.update, self.context)
        restarted = StickerCatalog(None, self.path / "stickers", persona_store=self.store)
        self.assertEqual((await catalog.adescribe_persona_context(self.session))['effective_persona'], {})
        self.assertEqual((await catalog.adescribe_style_context(self.session))['recent_sticker_ids'], [])
        self.assertEqual((await catalog.adescribe_persona_context("telegram:200"))['effective_persona'], persona)
        for session in (self.session, "telegram:200"):
            self.assertEqual(await catalog.adescribe_persona_context(session),
                             await restarted.adescribe_persona_context(session))
        self.assertEqual(list(catalog.entries_by_id), ["shared-sticker"])
        stale_plan = StickerRetrievalPlan.from_payload({"intent_core": "hello",
            "persona_mode": "merge_and_remember", "persona": persona})
        with self.assertRaises(StaleScopeError):
            await catalog.aprepare_query_context(plan=stale_plan, session_id=self.session,
                persist_persona=True, expected_scope=scope)
        self.assertIsNone(await self.store.get_sticker_persona(self.session))
        self.assertEqual((await catalog.adescribe_persona_context(self.session))['effective_persona'], {})

    async def test_rollback_groups_tool_and_assistant_as_one_bot_block(self):
        first = await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage.user_text("question"))
        await self.runtime.record_tool_observation(session_id=self.session, name="shell_exec", payload={}, phase="result")
        answer = '<b>Done & saved</b> 😀\n' + 'Full details remain in the original. ' * 5
        stored = await self.runtime.record_assistant_text(session_id=self.session, text=answer)
        await self.app.rollback_command(self.update, self.context)
        visible = await self.store.list_uncompacted_messages(self.session)
        self.assertEqual([item.db_id for item in visible], [first.db_id])
        response = self.message.reply_text.await_args
        self.assertEqual(response.kwargs['parse_mode'], 'HTML')
        preview = plain_text(response.args[0])
        self.assertIn('2 message(s) hidden', preview)
        self.assertIn('tool:shell_exec:result', preview)
        self.assertIn('assistant: <b>Done & saved</b> 😀', preview)
        self.assertNotIn('<b>Done & saved</b>', response.args[0])
        self.assertIn('…', preview)
        self.assertNotIn(answer, preview)
        # Presentation may shorten an excerpt; rollback keeps the original for audit.
        revisions = await self.store.list_message_revisions(self.session, stored.db_id)
        self.assertEqual(revisions[0]['body'], answer)

    async def test_retry_finds_user_before_more_than_one_page_of_tool_outputs(self):
        trigger = await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage.user_text("long task"))
        # 41 crosses the existing 40-row page boundary with the smallest fixture.
        for number in range(41):
            await self.runtime.record_tool_observation(session_id=self.session, name="shell_exec", payload={"sequence": number}, phase="result")
        self.app._reply_to_candidate = AsyncMock(return_value=None)
        await self.app.retry_command(self.update, self.context)
        self.assertEqual(self.app._reply_to_candidate.await_args.args[0].stored_message_id, trigger.db_id)
        self.assertEqual(len(await self.store.list_recent_visible_messages(self.session)), 1)

    async def test_rollback_hides_entire_bot_block_across_page_boundary(self):
        trigger = await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage.user_text("long task"))
        for number in range(41):
            await self.runtime.record_assistant_text(session_id=self.session, text=f"output {number}")
        await self.app.rollback_command(self.update, self.context)
        self.assertEqual([item.db_id for item in await self.store.list_recent_visible_messages(self.session)], [trigger.db_id])
        preview = plain_text(self.message.reply_text.await_args.args[0])
        self.assertIn('41 message(s) hidden', preview)
        self.assertIn('assistant: output 31', preview)
        self.assertIn('assistant: output 40', preview)
        self.assertNotIn('assistant: output 30', preview)
        self.assertIn('31 earlier message(s)', preview)

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
        self.app._deliver_result = AsyncMock(return_value=[SimpleNamespace(message_id=123, chat=self.chat)])
        await self.app._reply_to_candidate(self.candidate())
        self.assertEqual((await self.store.list_messages(self.session))[0].parts[0].text, "delivered")

    async def test_delivery_orders_answer_then_files_then_after_final_stickers(self):
        file_path = self.path / "report.txt"
        file_path.write_text("report")
        before = OutboundSticker(self.path / "before.webp", timing=StickerTiming.SEND_NOW)
        after = OutboundSticker(self.path / "after.webp", timing=StickerTiming.AFTER_FINAL)
        renderer = SimpleNamespace(finalize=AsyncMock(), send_artifacts=AsyncMock(return_value=[]), send_stickers=AsyncMock(return_value=[]))
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
        self.assertEqual(receipt["delivery_state"], "unknown")

    async def test_attachment_failure_keeps_caption_and_records_auto_note(self):
        self.message.text = None
        self.message.caption = "Please inspect this"
        with patch("tgchatbot.transports.telegram_adapter.extract_message_parts", new=AsyncMock(side_effect=TimeoutError())):
            parts = await self.app._safe_extract_parts(self.message, self.session)
        self.assertEqual(parts[0].text, "Please inspect this")
        self.assertIn("Attachment download failed", parts[1].text)
        self.assertEqual(parts[1].origin, "auto_note")
