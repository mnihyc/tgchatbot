from __future__ import annotations

import asyncio
import json

from tests.business_helpers import BusinessTestCase, ScriptedProvider
from tgchatbot.core.runtime import AgentRuntime, CompactionModelRequestFailed
from tgchatbot.domain.models import (
    ChatMode, ConversationMessage, MessagePart, MessageRole, PartKind,
    PromptInjectionMode, ProviderResponse, StickerMode, ToolCall, ToolHistoryMode,
)
from tgchatbot.storage.sqlite_store import SQLiteStore


def text_of(messages):
    return "\n".join(part.text or "" for message in messages for part in message.parts)


class RuntimeWorkflowTests(BusinessTestCase):
    async def turn(self, text="hello"):
        return await self.runtime.run_turn(session_id=self.session, user_display_name="tester", incoming_message=ConversationMessage.user_text(text))

    async def test_delivered_history_and_settings_survive_restart(self):
        await self.settings(model="chosen-model", system_prompt="Keep this voice.")
        self.provider.responses = [ProviderResponse(final_text="first answer")]
        result = await self.turn()
        # Delivery, rather than generation, commits visible assistant text.
        self.assertEqual([m.role for m in await self.store.list_messages(self.session)], [MessageRole.USER])
        await self.runtime.record_assistant_text(session_id=self.session, text=result.text)
        second = ScriptedProvider(responses=[ProviderResponse(final_text="second answer")])
        restarted_store = SQLiteStore(self.config.db_path)
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
        self.assertEqual(len(await self.store.list_messages(self.session)), 1)

    async def test_exhausted_retries_preserve_user_but_no_invented_answer(self):
        await self.settings(provider_retry_count=1)
        self.provider.responses = [TimeoutError("first"), TimeoutError("last")]
        with self.assertRaisesRegex(TimeoutError, "last"):
            await self.turn()
        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual([m.role for m in await self.store.list_messages(self.session)], [MessageRole.USER])

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
        self.assertEqual([m.role for m in messages], [MessageRole.USER, MessageRole.TOOL, MessageRole.TOOL])
        self.assertEqual([m.metadata["tool_phase"] for m in messages[1:]], ["call", "result"])

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
        native = {"provider": "openai", "items": [{"type": "function_call", "call_id": "c1"}]}
        await self.runtime.record_tool_observation(session_id=self.session, name="shell_exec", phase="call", payload={"call_id": "c1"}, provider_name="openai", metadata_update={"provider_native": native})
        state = await self.runtime._get_live_state(self.session)
        history = self.runtime._build_provider_history(state, settings=settings, provider_name="openai")
        self.assertEqual(history[0].metadata["provider_native"], native)
        settings.tool_history_mode = ToolHistoryMode.TRANSLATED
        translated = self.runtime._build_provider_history(state, settings=settings, provider_name="openai")
        self.assertNotIn("provider_native", translated[0].metadata)
        self.assertEqual(translated[0].role, MessageRole.ASSISTANT)


class MemoryWorkflowTests(BusinessTestCase):
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
        with self.store._connect() as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0], 3)

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
