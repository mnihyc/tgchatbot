from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tests.business_helpers import BusinessTestCase, ScriptedProvider
from tgchatbot.core.memory import MemoryService
from tgchatbot.core.runtime import AgentRuntime, CompactionModelRequestFailed
from tgchatbot.storage.previews import PreviewCache
from tgchatbot.domain.models import (
    ChatMode, ConversationMessage, MessagePart, MessageRole, PartKind,
    PromptInjectionMode, ProviderResponse, StickerMode, ToolCall, ToolHistoryMode,
)


def text_of(messages):
    return "\n".join(part.text or "" for message in messages
        if message.metadata.get("synthetic_role") not in {"memory_context", "reply_target"}
        for part in message.parts if part.origin != "provenance")


class RuntimeWorkflowTests(BusinessTestCase):
    async def turn(self, text="hello"):
        return await self.runtime.run_turn(session_id=self.session, user_display_name="tester", incoming_message=ConversationMessage.user_text(text))

    async def test_delivered_history_and_settings_survive_restart(self):
        await self.settings(model="chosen-model", system_prompt="Keep this voice.")
        self.provider.responses = [ProviderResponse(final_text="first answer")]
        result = await self.turn()
        # Delivery, rather than generation, commits visible assistant text.
        messages = await self.store.list_messages(self.session)
        self.assertEqual([m.role for m in messages if m.metadata.get("synthetic_role") != "reply_target"],
                         [MessageRole.USER])
        self.assertEqual(sum(m.metadata.get("synthetic_role") == "reply_target" for m in messages), 1)
        await self.runtime.record_assistant_text(session_id=self.session, text=result.text)
        second = ScriptedProvider(responses=[ProviderResponse(final_text="second answer")])
        await self.store.close()
        restarted_store = await self.new_store()
        restarted = AgentRuntime(config=self.config, store=restarted_store, tool_registry=self.tools, providers={"openai": second})
        await restarted.run_turn(session_id=self.session, user_display_name="tester", incoming_message=ConversationMessage.user_text("continue"))
        self.assertEqual(text_of(second.requests[0]["messages"]), "hello\nfirst answer\ncontinue")
        self.assertEqual(second.requests[0]["settings"].model, "chosen-model")
        self.assertIn("Keep this voice.", second.requests[0]["instructions"])

    async def test_retry_recovers_without_duplicate_ingestion(self):
        await self.settings(provider_retry_count=1)
        self.provider.responses = [TimeoutError("temporary"), ProviderResponse(final_text="recovered")]
        self.assertEqual((await self.turn()).text, "recovered")
        self.assertEqual(len(self.provider.requests), 2)
        messages = await self.store.list_messages(self.session)
        self.assertEqual(text_of(messages), "hello")
        self.assertEqual(sum(m.metadata.get("synthetic_role") == "reply_target" for m in messages), 1)

    async def test_exhausted_retries_preserve_user_but_no_invented_answer(self):
        await self.settings(provider_retry_count=1)
        self.provider.responses = [TimeoutError("first"), TimeoutError("last")]
        with self.assertRaisesRegex(TimeoutError, "last"):
            await self.turn()
        self.assertEqual(len(self.provider.requests), 2)
        messages = await self.store.list_messages(self.session)
        self.assertEqual([m.role for m in messages if m.metadata.get("synthetic_role") != "reply_target"],
                         [MessageRole.USER])
        self.assertEqual(text_of(messages), "hello")

    async def test_cancellation_never_retries(self):
        await self.settings(provider_retry_count=3)
        self.provider.responses = [asyncio.CancelledError()]
        with self.assertRaises(asyncio.CancelledError):
            await self.turn()
        self.assertEqual(len(self.provider.requests), 1)

    async def test_unconfigured_provider_keeps_history_and_does_not_fallback(self):
        await self.settings(provider="missing")
        with self.assertRaisesRegex(RuntimeError, "not configured"):
            await self.turn()
        self.assertEqual(self.provider.requests, [])
        self.assertEqual(len(await self.store.list_messages(self.session)), 1)

    async def test_exact_preset_preserves_operator_prompt(self):
        await self.settings(system_prompt="Speak as the fixture character.", prompt_injection_mode=PromptInjectionMode.EXACT)
        self.provider.responses = [ProviderResponse(final_text="hello")]
        await self.turn()
        self.assertEqual(self.provider.requests[0]["instructions"], "Speak as the fixture character.")

    async def test_chat_mode_never_executes_unsolicited_tools(self):
        self.provider.responses = [ProviderResponse(tool_calls=[ToolCall("shell_exec", "c1", {})]), ProviderResponse(final_text="done")]
        await self.turn()
        self.assertEqual(self.provider.requests[0]["tools"], [])
        self.tools.list_tools.assert_not_called()
        self.tools.runner.run.assert_not_awaited()

    async def test_tool_round_records_observations_and_forces_final_answer(self):
        await self.settings(mode=ChatMode.ASSIST, max_interaction_rounds=1, sticker_mode=StickerMode.OFF)
        self.provider.responses = [
            ProviderResponse(final_text="Checking.", tool_calls=[ToolCall("shell_exec", "c1", {"command": "mock"})], continuation_items=[{"type": "function_call", "call_id": "c1"}]),
            ProviderResponse(final_text="Done."),
        ]
        result = await self.turn()
        self.assertEqual(result.text, "Checking.\n\nDone.")
        self.tools.list_tools.assert_called_once_with(allow_python_exec=True, allow_stickers=False)
        self.tools.runner.run.assert_awaited_once()
        self.assertEqual(self.provider.requests[1]["tools"], [])
        self.assertIn("final interaction round", self.provider.requests[1]["instructions"])
        self.assertEqual(self.provider.requests[1]["extra_input_items"][-1]["call_id"], "c1")
        messages = await self.store.list_messages(self.session)
        self.assertEqual([m.role for m in messages], [MessageRole.USER, MessageRole.USER, MessageRole.TOOL, MessageRole.TOOL])
        self.assertEqual(messages[1].metadata["synthetic_role"], "reply_target")
        self.assertEqual([m.metadata["tool_phase"] for m in messages[2:]], ["call", "result"])

    async def test_final_round_cannot_extend_tool_execution(self):
        await self.settings(mode=ChatMode.AGENT, max_interaction_rounds=1)
        self.provider.responses = [ProviderResponse(tool_calls=[ToolCall("shell_exec", f"c{i}", {})]) for i in range(2)]
        with self.assertRaisesRegex(RuntimeError, "Interaction-round limit"):
            await self.turn()
        self.assertEqual(len(self.provider.requests), 2)
        self.tools.runner.run.assert_awaited_once()

    async def test_unadvertised_tool_is_rejected_even_when_registered(self):
        await self.settings(mode=ChatMode.ASSIST)
        self.tools.list_tools.return_value = []
        self.provider.responses = [ProviderResponse(tool_calls=[ToolCall("shell_exec", "c1", {})]), ProviderResponse(final_text="done")]
        await self.turn()
        self.tools.runner.run.assert_not_awaited()
        self.assertFalse(self.provider.requests[1]["extra_input_items"][0]["output"]["ok"])

    async def test_switching_provider_translates_native_tool_history(self):
        await self.settings(mode=ChatMode.ASSIST, tool_history_mode=ToolHistoryMode.NATIVE_SAME_PROVIDER)
        self.provider.responses = [ProviderResponse(tool_calls=[ToolCall("shell_exec", "c1", {})], continuation_items=[{"type": "function_call", "call_id": "c1"}]), ProviderResponse(final_text="old answer")]
        result = await self.turn()
        await self.runtime.record_assistant_text(session_id=self.session, text=result.text)
        other = ScriptedProvider("gemini", [ProviderResponse(final_text="continued")])
        self.runtime.providers["gemini"] = other
        await self.settings(provider="gemini", model="other-model")
        await self.turn("continue on another provider")
        history = other.requests[0]["messages"]
        self.assertIn("old answer", text_of(history))
        self.assertIn("shell_exec", text_of(history))
        self.assertFalse(any(m.role == MessageRole.TOOL for m in history))
        self.assertFalse(any("provider_native" in m.metadata for m in history))

    async def test_same_provider_native_history_and_translated_mode_are_distinct(self):
        settings = await self.settings(tool_history_mode=ToolHistoryMode.NATIVE_SAME_PROVIDER)
        native = {"provider": "openai", "model": settings.model, "items": [{"type": "function_call", "call_id": "c1"}]}
        await self.runtime.record_tool_observation(session_id=self.session, name="shell_exec", phase="call", payload={"call_id": "c1"}, provider_name="openai", metadata_update={"provider_native": native})
        state = await self.runtime._get_live_state(self.session)
        history = self.runtime._build_provider_history(state, settings=settings, provider_name="openai")
        self.assertEqual(history[0].metadata["provider_native"], native)
        settings.tool_history_mode = ToolHistoryMode.TRANSLATED
        translated = self.runtime._build_provider_history(state, settings=settings, provider_name="openai")
        self.assertNotIn("provider_native", translated[0].metadata)
        self.assertEqual(translated[0].role, MessageRole.ASSISTANT)


class MemoryWorkflowTests(BusinessTestCase):
    async def test_sticker_hints_and_images_stay_stable_until_oldest_batch_retirement(self):
        # Small image thresholds exercise the same workflow without thousands
        # of fixtures. Token pressure is absent throughout these short turns.
        await self.settings(max_input_images=3, compact_target_images=1)
        self.provider.responses = [ProviderResponse(final_text="Seen.") for _ in range(5)]
        image_counts = []
        requests = []
        for number in range(5):
            await self.runtime.run_turn(session_id=self.session, user_display_name="tester",
                incoming_message=ConversationMessage(MessageRole.USER, [
                    MessagePart(PartKind.STICKER, text=f"[User sent static sticker 🙂 {number}]", remote_sync=False),
                    MessagePart(PartKind.IMAGE, data_b64="ZmFrZQ==", mime_type="image/png", remote_sync=False),
                ]))
            # Inspect image content independently of retained application
            # control records; append-only targets are covered separately.
            request = [message for message in self.provider.requests[-1]["messages"]
                       if message.metadata.get("synthetic_role") != "reply_target"]
            requests.append(request)
            parts = [part for message in request for part in message.parts]
            hints = [part.text for part in parts if part.kind == PartKind.STICKER]
            self.assertEqual(hints, [f"[User sent static sticker 🙂 {item}]" for item in range(number + 1)])
            self.assertNotIn("expired", text_of(request))
            image_counts.append(sum(part.kind == PartKind.IMAGE and bool(part.data_b64) for part in parts))
        self.assertEqual(image_counts, [1, 2, 3, 1, 2])
        self.assertEqual(requests[2][:len(requests[1])], requests[1])
        self.assertEqual(text_of(requests[3]).count("[Image compacted]"), 3)
        self.assertEqual(requests[4][:len(requests[3])], requests[3])

    async def test_valid_compaction_commits_memory_and_keeps_latest_raw_message(self):
        settings = await self.settings(min_raw_messages_reserve=1)
        for text in ("I prefer concise replies.", "Remember this preference.", "Latest question stays raw."):
            await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage.user_text(text))
        candidate = {
            "scope": "Response style preference", "interaction_mode": "chat_or_sharing",
            "participants": ["tester"], "topics": ["preferences", "response style"],
            "user_profile": ["Prefers concise replies"],
            "user_intent_or_shared_context": ["Remember response preference"],
            "why_it_mattered": [], "interaction_timeline": ["tester stated preference"],
            "results_or_takeaways": ["Use concise replies"], "decisions": [],
            "open_loops": [], "artifacts": [], "uncertainties": [],
        }
        self.provider.responses = [ProviderResponse(final_text=json.dumps(candidate))]
        state = await self.runtime._get_live_state(self.session)
        changed = await self.runtime._compact_old_context(session_id=self.session, settings=settings, provider=self.provider, state=state, pressure=True)
        self.assertTrue(changed)
        self.assertEqual(len(state.blocks), 1)
        self.assertEqual(state.blocks[0].validator_status, "passed")
        self.assertEqual(state.blocks[0].structured_data["user_profile"], ["Prefers concise replies"])
        self.assertEqual(text_of([item.message for item in state.raw_messages]), "Latest question stays raw.")
        self.assertEqual(self.provider.requests[0]["tools"], [])
        self.assertEqual(self.provider.requests[0]["response_schema_name"], "episode_memory_block")
        self.runtime.invalidate_session(self.session)
        self.assertEqual(len((await self.runtime._get_live_state(self.session)).blocks), 1)

    async def test_image_compaction_preserves_newest_image_and_original_artifact(self):
        original = self.path / "original.png"
        original.write_bytes(b"fixture image")
        for number in range(3):
            await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage(MessageRole.USER, [MessagePart(PartKind.TEXT, text=f"image {number}"), MessagePart(PartKind.IMAGE, artifact_path=str(original), data_b64="ZmFrZQ==")]))
        state = await self.runtime._get_live_state(self.session)
        removed = await self.runtime._compact_oldest_images(session_id=self.session, settings=await self.settings(), state=state, target_images=1)
        self.assertEqual(removed, 2)
        self.assertEqual(state.estimated_images, 1)
        self.assertEqual(state.raw_messages[-1].message.parts[-1].kind, PartKind.IMAGE)
        self.assertEqual(state.raw_messages[0].message.parts[-1].text, "[Image compacted]")
        self.assertEqual(original.read_bytes(), b"fixture image")
        self.runtime.invalidate_session(self.session)
        self.assertEqual((await self.runtime._get_live_state(self.session)).estimated_images, 1)

    async def test_retry_compaction_keeps_latest_question_raw_when_a_control_record_is_last(self):
        settings = await self.settings(min_raw_messages_reserve=1)
        await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text("Earlier discussion."))
        latest = await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text("Keep this unanswered question verbatim."))
        self.provider.responses = [ProviderResponse(final_text="An attempted answer.")]
        await self.runtime.run_turn_from_stored(session_id=self.session, user_display_name="tester",
                                              trigger_message_id=latest.db_id)
        # A later attempt starts with the existing target control at the tail.
        # It must not satisfy the reserve intended for meaningful conversation.
        state = await self.runtime._get_live_state(self.session)
        self.assertEqual(state.raw_messages[-1].message.metadata.get("synthetic_role"), "reply_target")
        candidate = {key: [] for key in ("user_profile", "user_intent_or_shared_context", "why_it_mattered",
            "interaction_timeline", "results_or_takeaways", "decisions", "open_loops", "artifacts", "uncertainties")}
        candidate.update(scope="Earlier discussion", interaction_mode="chat_or_sharing",
                         participants=["tester"], topics=["conversation", "history"])
        self.provider.requests.clear()
        self.provider.responses = [ProviderResponse(final_text=json.dumps(candidate))]
        changed = await self.runtime._compact_old_context(session_id=self.session, settings=settings,
            provider=self.provider, state=state, pressure=True)
        self.assertTrue(changed)
        self.assertEqual(text_of([item.message for item in state.raw_messages]),
                         "Keep this unanswered question verbatim.")
        self.assertNotIn("Keep this unanswered question verbatim.", text_of(self.provider.requests[0]["messages"]))

    async def test_compacted_history_restarts_in_order_and_reset_hides_both_layers(self):
        first = await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage.user_text("old user message"))
        answer = await self.runtime.record_assistant_text(session_id=self.session, text="old answer")
        await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage.user_text("recent message"))
        await self.store.create_memory_block(self.session, summary_text="A durable preference.", estimated_tokens=10, source_message_ids=[first.db_id, answer.db_id])
        self.runtime.invalidate_session(self.session)
        state = await self.runtime._get_live_state(self.session)
        history = self.runtime._build_provider_history(state, settings=await self.settings(), provider_name="openai")
        self.assertEqual(len(state.raw_messages), 1)
        self.assertEqual(len(state.blocks), 1)
        self.assertIn("A durable preference.", text_of(history[:1]))
        self.assertEqual(text_of(history[-1:]), "recent message")
        await self.store.clear_messages(self.session)
        self.runtime.invalidate_session(self.session)
        self.assertEqual((await self.runtime._get_live_state(self.session)).raw_messages, [])
        self.assertEqual(await self.store.list_memory_blocks(self.session), [])
        # Reset is reversible archival, not physical deletion.
        async with self.store.pool.connection() as conn:
            row = await (await conn.execute("SELECT COUNT(*) AS count FROM messages WHERE session_id=%s", (self.session,))).fetchone()
            self.assertEqual(row["count"], 3)

    async def test_invalid_compaction_output_preserves_original_messages(self):
        stored = await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage.user_text("must survive"))
        self.provider.responses = [ProviderResponse(final_text='{"scope":"incomplete"}')]
        candidate = await self.runtime._make_episode_block_candidate(self.provider, await self.settings(), [stored.message], [stored], [])
        self.assertIsNone(candidate)
        self.assertEqual(len(await self.store.list_uncompacted_messages(self.session)), 1)
        self.assertEqual(await self.store.list_memory_blocks(self.session), [])

    async def test_compaction_network_failure_is_explicit_and_non_destructive(self):
        stored = await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage.user_text("must survive"))
        self.provider.responses = [TimeoutError("model unavailable")]
        with self.assertRaises(CompactionModelRequestFailed):
            await self.runtime._make_episode_block_candidate(self.provider, await self.settings(), [stored.message], [stored], [])
        self.assertEqual(len(await self.store.list_uncompacted_messages(self.session)), 1)
        self.assertEqual(await self.store.list_memory_blocks(self.session), [])

    async def test_compaction_keeps_tool_call_and_result_together(self):
        await self.runtime.ingest_user_message(session_id=self.session, incoming_message=ConversationMessage.user_text("run task"))
        for phase in ("call", "result"):
            await self.runtime.record_tool_observation(session_id=self.session, name="shell_exec", phase=phase, payload={"call_id": "c1"})
        messages = await self.store.list_uncompacted_messages(self.session)
        units = self.runtime._group_raw_compaction_units(messages)
        self.assertEqual([[item.db_id for item in unit] for unit in units], [[messages[0].db_id], [messages[1].db_id, messages[2].db_id]])


class ProfileContextWorkflowTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.memory = MemoryService(self.store, SimpleNamespace(enabled=False, space_id="fixture"))
        self.runtime.memory = self.memory
        self.actor = "telegram:user:7"
        self.claim = "Prefers jasmine tea, with no sugar."

    def incoming(self, text, source_id, *, image=False):
        message = ConversationMessage.user_text(text, metadata={
            "source": "telegram", "source_chat_id": "100", "source_message_id": str(source_id),
            "actor_id": self.actor, "actor_kind": "user", "actor_name": "Alex",
            "sent_at": "2026-01-02T03:04:05+00:00",
        })
        if image:
            message.parts.append(MessagePart(PartKind.IMAGE, data_b64="ZmFrZQ==",
                                            mime_type="image/png", remote_sync=False))
        return message

    async def seed_profile(self):
        source = await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=self.incoming("I prefer jasmine tea, with no sugar.", 1))
        await self.store.save_profile_fact(self.session, subject_actor_id=self.actor,
            asserted_by=self.actor, claim=self.claim, source_ids=[source.db_id],
            valid_from="2026-01-02T03:04:05+00:00")
        return source

    async def turn(self, text, source_id, *, image=False):
        return await self.runtime.run_turn(session_id=self.session, user_display_name="Alex",
            incoming_message=self.incoming(text, source_id, image=image))

    async def observations(self, synthetic_role):
        return [item for item in await self.store.list_uncompacted_messages(self.session)
                if item.message.metadata.get("synthetic_role") == synthetic_role]

    async def refresh_pair(self):
        pair = await self.observations("profile_refresh")
        self.assertEqual(len(pair), 2)
        call, result = pair
        self.assertEqual([item.message.role for item in pair], [MessageRole.TOOL, MessageRole.TOOL])
        self.assertEqual([item.message.name for item in pair], ["user_profile_fetch", "user_profile_fetch"])
        self.assertEqual([item.message.metadata["tool_phase"] for item in pair], ["call", "result"])
        call_id = call.message.metadata["tool_payload"]["call_id"]
        self.assertTrue(call_id)
        self.assertEqual(result.message.metadata["tool_payload"]["call_id"], call_id)
        self.assertTrue(all(item.message.metadata["refresh_reason"] == "compaction" for item in pair))
        return call, result

    async def test_ordinary_turns_and_restart_keep_reply_targets_in_the_request_prefix(self):
        await self.seed_profile()
        self.provider.responses = [ProviderResponse(final_text="First answer."),
                                   ProviderResponse(final_text="Second answer.")]
        with patch.object(self.memory, "fetch_profiles", wraps=self.memory.fetch_profiles) as fetch:
            first = await self.turn("What should we do next?", 2)
            before = self.provider.requests[-1]["messages"]
            await self.runtime.record_assistant_text(session_id=self.session, text=first.text)
            await self.turn("Continue.", 3)
            after = self.provider.requests[-1]["messages"]
            fetch.assert_not_awaited()
        self.assertEqual(after[:len(before)], before)
        self.assertEqual(len(await self.observations("reply_target")), 2)
        self.assertEqual(await self.observations("profile_refresh"), [])
        self.assertFalse(any(item.metadata.get("synthetic_role") == "memory_context" for item in after))
        # The durable control record survives a cold process, not only the hot
        # request cache. A repeat attempt for the same target must not duplicate it.
        await self.store.close()
        restarted_store = await self.new_store()
        next_provider = ScriptedProvider(responses=[ProviderResponse(final_text="Retry answer.")])
        restarted = AgentRuntime(config=self.config, store=restarted_store, tool_registry=self.tools,
            providers={"openai": next_provider}, memory=MemoryService(restarted_store,
                SimpleNamespace(enabled=False, space_id="fixture")))
        sources = await restarted_store.list_uncompacted_messages(self.session)
        target_id = next(item.db_id for item in sources if item.message.metadata.get("source_message_id") == "3")
        await restarted.run_turn_from_stored(session_id=self.session, user_display_name="Alex",
                                            trigger_message_id=target_id)
        self.assertEqual(next_provider.requests[0]["messages"], after)

    async def test_profile_tool_evidence_survives_later_turns_provider_switch_and_compaction_input(self):
        source = await self.seed_profile()
        await self.settings(tool_history_mode=ToolHistoryMode.NATIVE_SAME_PROVIDER)
        self.provider.responses = [
            ProviderResponse(tool_calls=[ToolCall("user_profile_fetch", "profile-1", {"actor_ids": [self.actor]})],
                continuation_items=[{"type": "function_call", "call_id": "profile-1",
                                     "name": "user_profile_fetch", "arguments": json.dumps({"actor_ids": [self.actor]})}]),
            ProviderResponse(final_text="I will keep the tea unsweetened."),
            ProviderResponse(final_text="Certainly."),
        ]
        with patch.object(self.memory, "fetch_profiles", wraps=self.memory.fetch_profiles) as fetch:
            first = await self.turn("Do you remember my drink preference?", 2)
            response_payload = self.provider.requests[1]["extra_input_items"][-1]["output"]
            rendered = json.dumps(response_payload, ensure_ascii=False, default=str)
            self.assertIn(self.claim, rendered)
            self.assertIn("2026-01-02T11:04:05+08:00", rendered)
            facts = [(profile["actor_id"], fact["source_ids"])
                     for profile in response_payload["profiles"] for fact in profile["facts"]]
            self.assertEqual(facts,
                             [(self.actor, [source.db_id])])
            await self.runtime.record_assistant_text(session_id=self.session, text=first.text)
            await self.turn("And for tomorrow?", 3)
            fetch.assert_awaited_once()
        native_history = self.provider.requests[2]["messages"]
        self.assertEqual(native_history[:len(self.provider.requests[0]["messages"])],
                         self.provider.requests[0]["messages"])
        native_result = next(item for item in native_history if item.role == MessageRole.TOOL
                             and item.name == "user_profile_fetch" and item.metadata.get("tool_phase") == "result")
        self.assertEqual(native_result.metadata["tool_payload"]["output"], response_payload)
        other = ScriptedProvider("gemini", [ProviderResponse(final_text="The same preference applies.")])
        self.runtime.providers["gemini"] = other
        await self.settings(provider="gemini", model="fixture-gemini")
        await self.turn("Continue with this provider.", 4)
        translated = other.requests[0]["messages"]
        self.assertIn(self.claim, text_of(translated))
        self.assertIn('"source_ids": [' + str(source.db_id) + ']', text_of(translated))
        self.assertFalse(any(item.role == MessageRole.TOOL for item in translated))
        raw = await self.store.list_uncompacted_messages(self.session)
        compacted_input = self.runtime._normalize_compaction_messages([item.message for item in raw])
        self.assertIn(self.claim, text_of(compacted_input))
        self.assertIn('"source_ids": [' + str(source.db_id) + ']', text_of(compacted_input))
        controls = [item for item in compacted_input if item.metadata.get("source_role") == "transport"]
        self.assertTrue(any(self.actor in text_of([item]) for item in controls))
        self.assertTrue(all(item.role != MessageRole.USER for item in controls))

    async def test_image_retirement_refreshes_profiles_once_without_creating_new_human_evidence(self):
        source = await self.seed_profile()
        await self.settings(max_input_images=1, compact_target_images=1,
                            tool_history_mode=ToolHistoryMode.TRANSLATED)
        self.provider.responses = [ProviderResponse(final_text="Seen.") for _ in range(3)]
        with patch.object(self.memory, "fetch_profiles", wraps=self.memory.fetch_profiles) as fetch:
            await self.turn("First picture.", 2, image=True)
            fetch.assert_not_awaited()
            await self.turn("Second picture.", 3, image=True)
            fetch.assert_awaited_once()
            changed_history = self.provider.requests[-1]["messages"]
            await self.turn("No new picture.", 4)
            fetch.assert_awaited_once()
        self.assertEqual(self.provider.requests[-1]["messages"][:len(changed_history)], changed_history)
        self.assertIn(self.claim, text_of(changed_history))
        call, result = await self.refresh_pair()
        output = result.message.metadata["tool_payload"]["output"]
        self.assertIn(self.actor, call.message.metadata["tool_payload"]["arguments"]["actor_ids"])
        sent_pair = [item for item in changed_history if item.metadata.get("synthetic_role") == "profile_refresh"]
        self.assertEqual([item.role for item in sent_pair], [MessageRole.TOOL, MessageRole.TOOL])
        self.assertEqual(sent_pair[-1].metadata["tool_payload"]["output"], output)
        self.assertFalse(any(item.role == MessageRole.USER and self.claim in "\n".join(part.text or "" for part in item.parts)
                             for item in changed_history))
        raw = await self.store.list_uncompacted_messages(self.session)
        units = self.runtime._group_raw_compaction_units(raw)
        self.assertIn([call.db_id, result.db_id], [[item.db_id for item in unit] for unit in units])
        normalized = self.runtime._normalize_compaction_messages([call.message, result.message])
        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].metadata["source_role"], "tool")
        self.assertIn(self.claim, text_of(normalized))
        for synthetic in [call, result, *(await self.observations("reply_target"))]:
            with self.assertRaisesRegex(ValueError, "original message"):
                await self.store.save_profile_fact(self.session, subject_actor_id=self.actor,
                    asserted_by=self.actor, claim="A circular inference.", source_ids=[synthetic.db_id])
        self.assertEqual([fact["source_ids"] for fact in await self.store.get_profile(self.session, self.actor)],
                         [[source.db_id]])
        before_restart = self.provider.requests[-1]["messages"]
        await self.store.close()
        store = await self.new_store()
        memory = MemoryService(store, SimpleNamespace(enabled=False, space_id="fixture"))
        provider = ScriptedProvider(responses=[ProviderResponse(final_text="Still remembered.")])
        runtime = AgentRuntime(config=self.config, store=store, tool_registry=self.tools,
            providers={"openai": provider}, memory=memory, preview_cache=PreviewCache(store, max_bytes=self.preview_cache.max_bytes))
        with patch.object(memory, "fetch_profiles", wraps=memory.fetch_profiles) as fetch:
            await runtime.run_turn(session_id=self.session, user_display_name="Alex",
                                   incoming_message=self.incoming("Continue after restart.", 5))
            fetch.assert_not_awaited()
        self.assertEqual(provider.requests[0]["messages"][:len(before_restart)], before_restart)
        restored_pair = [item for item in provider.requests[0]["messages"]
                         if item.metadata.get("synthetic_role") == "profile_refresh"]
        self.assertEqual([item.role for item in restored_pair], [MessageRole.TOOL, MessageRole.TOOL])
        self.assertEqual(restored_pair[-1].metadata["tool_payload"]["output"], output)

    async def test_noop_or_failed_compaction_does_not_refresh_profiles(self):
        await self.seed_profile()
        await self.settings(compact_trigger_tokens=9000, compact_target_tokens=8000)
        self.provider.responses = [ProviderResponse(final_text="History is still available.") for _ in range(2)]
        for number, outcome in enumerate((False, CompactionModelRequestFailed(provider_name="openai", mode="episode")), 2):
            with self.subTest(outcome=type(outcome).__name__):
                stage = AsyncMock(return_value=False) if outcome is False else AsyncMock(side_effect=outcome)
                with patch.object(self.runtime, "_estimate_request_tokens", return_value=10000), \
                     patch.object(self.runtime, "_compact_old_context", stage), \
                     patch.object(self.memory, "fetch_profiles", wraps=self.memory.fetch_profiles) as fetch:
                    await self.turn("Continue without losing the original.", number)
                    fetch.assert_not_awaited()
                self.assertEqual(await self.observations("profile_refresh"), [])
                self.assertEqual(await self.store.list_memory_blocks(self.session), [])

    async def test_compaction_refreshes_active_group_people_and_direct_reply_partner_only(self):
        # All three have retained profiles, but none is currently speaking.
        # Only the direct reply counterpart should rejoin the active audience.
        for number, actor in ((90, "telegram:user:99"), (91, "telegram:user:9"), (92, "telegram:user:10")):
            original = self.incoming("An earlier preference.", number)
            original.metadata["actor_id"] = actor
            source = await self.runtime.ingest_user_message(session_id=self.session, incoming_message=original)
            await self.store.save_profile_fact(self.session, subject_actor_id=actor, asserted_by=actor,
                claim=f"Earlier preference for {actor}.", source_ids=[source.db_id])
        await self.store.reset_context(self.session)
        self.runtime.invalidate_session(self.session)
        await self.seed_profile()
        other = self.incoming("I am also taking part.", 2)
        other.metadata["actor_id"] = "telegram:user:8"
        # Both current speakers are named Alex; identity must stay separate.
        await self.runtime.ingest_user_message(session_id=self.session, incoming_message=other)
        await self.settings(max_input_images=1, compact_target_images=1)
        self.provider.responses = [ProviderResponse(final_text="Seen.") for _ in range(2)]
        await self.turn("First picture.", 3, image=True)
        current = self.incoming("About your earlier message, here is another picture.", 4, image=True)
        current.metadata.update(reply_to_source_id="91", reply_to_source_chat_id="100",
            reply_to_actor={"actor_id": "telegram:user:9", "actor_kind": "user", "actor_name": "Earlier partner"},
            forward_origin={"type": "user", "sender_user": {"id": 10, "first_name": "Forwarded person"}})
        with patch.object(self.memory, "fetch_profiles", wraps=self.memory.fetch_profiles) as fetch:
            await self.runtime.run_turn(session_id=self.session, user_display_name="Alex", incoming_message=current)
            fetch.assert_awaited_once()
        _, result = await self.refresh_pair()
        refresh = result.message
        profiles = refresh.metadata["tool_payload"]["output"]["profiles"]
        by_id = {profile["actor_id"]: profile for profile in profiles}
        self.assertEqual(set(by_id), {self.actor, "telegram:user:8", "telegram:user:9", "agent"})
        self.assertEqual(by_id[self.actor]["identity"]["actor_name"], "Alex")
        self.assertEqual(by_id["telegram:user:8"]["identity"]["actor_name"], "Alex")
        self.assertTrue(by_id["telegram:user:9"]["facts"])
        self.assertEqual(len(await self.store.get_profile(self.session, "telegram:user:99")), 1)
        self.assertEqual(len(await self.store.get_profile(self.session, "telegram:user:10")), 1)

    async def test_successful_promotion_followed_by_model_failure_still_refreshes_once(self):
        source = await self.seed_profile()
        await self.settings(compact_trigger_tokens=9000, compact_target_tokens=8000)
        self.provider.responses = [ProviderResponse(final_text="The saved preference remains available.")]
        promoted = False

        async def promote_then_fail(**kwargs):
            nonlocal promoted
            if promoted:
                raise CompactionModelRequestFailed(provider_name="openai", mode="digest")
            await self.store.create_memory_block(self.session, summary_text="Earlier drink discussion.",
                estimated_tokens=10, source_message_ids=[source.db_id])
            await self.runtime._reload_live_state(kwargs["state"])
            promoted = True
            return True

        with patch.object(self.runtime, "_estimate_request_tokens", return_value=10000), \
             patch.object(self.runtime, "_compact_old_context", side_effect=promote_then_fail), \
             patch.object(self.memory, "fetch_profiles", wraps=self.memory.fetch_profiles) as fetch:
            await self.turn("What do you remember?", 2)
            fetch.assert_awaited_once()
        self.assertEqual(len(await self.store.list_memory_blocks(self.session)), 1)
        await self.refresh_pair()
        self.assertIn("Earlier drink discussion.", text_of(self.provider.requests[-1]["messages"]))
        self.assertIn(self.claim, text_of(self.provider.requests[-1]["messages"]))

    async def test_reset_requires_an_explicit_new_fetch_and_full_reset_excludes_old_profiles(self):
        await self.seed_profile()
        self.provider.responses = [ProviderResponse(tool_calls=[ToolCall("user_profile_fetch", "p1", {"actor_ids": [self.actor]})]),
                                   ProviderResponse(final_text="I remember."),
                                   ProviderResponse(final_text="Fresh context."),
                                   ProviderResponse(tool_calls=[ToolCall("user_profile_fetch", "p2", {"actor_ids": [self.actor]})]),
                                   ProviderResponse(final_text="No retained preference.")]
        await self.turn("Fetch my profile.", 2)
        await self.store.reset_context(self.session)
        self.runtime.invalidate_session(self.session)
        with patch.object(self.memory, "fetch_profiles", wraps=self.memory.fetch_profiles) as fetch:
            await self.turn("A fresh conversation.", 3)
            fetch.assert_not_awaited()
        self.assertNotIn(self.claim, text_of(self.provider.requests[-1]["messages"]))
        self.assertEqual(len(await self.store.get_profile(self.session, self.actor)), 1)
        await self.store.reset_full(self.session, self.config.default_session_settings())
        self.runtime.invalidate_session(self.session)
        await self.turn("Fetch my profile again.", 4)
        payload = self.provider.requests[-1]["extra_input_items"][-1]["output"]
        self.assertEqual([fact for profile in payload["profiles"] for fact in profile["facts"]], [])
        self.assertNotIn(self.claim, text_of(self.provider.requests[-1]["messages"]))
