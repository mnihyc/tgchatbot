from __future__ import annotations

import asyncio
import json
import os
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from psycopg import OperationalError
from psycopg.errors import QueryCanceled

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.memory import MemoryService
from tgchatbot.core.token_estimator import TokenEstimator
from tgchatbot.domain.models import ChatMode, ConversationMessage, ProviderResponse, PromptInjectionMode, ToolCall
from tgchatbot.domain.provenance import original_text
from tgchatbot.storage.postgres_store import StaleScopeError
from tgchatbot.operational import MemoryConfig, from_env


class MemoryBusinessTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.embeddings = SimpleNamespace(enabled=False, space_id="synthetic-space", embed_query=AsyncMock())
        self.memory = MemoryService(self.store, self.embeddings)
        self.runtime.memory = self.memory

    def message(self, text, actor, source_id, *, actor_name="Alex", chat="100"):
        return ConversationMessage.user_text(text, metadata={
            "source": "telegram", "source_chat_id": chat, "source_message_id": str(source_id),
            "actor_id": actor, "actor_kind": "user", "actor_name": actor_name,
            "sent_at": "2026-01-02T03:04:05+00:00",
        })

    async def ingest(self, text, actor="telegram:user:7", source_id=1, **kwargs):
        return await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=self.message(text, actor, source_id, **kwargs))

    async def test_configured_retrieval_reads_more_than_old_bounds_without_losing_originals(self):
        with patch.dict(os.environ, {
            'MEMORY_SEARCH_RESULTS': '25', 'MEMORY_READ_MESSAGES': '25',
            'MEMORY_READ_CHARS': '16000', 'MEMORY_SEARCH_RESULT_CHARS': '16000',
            'MEMORY_RESPONSE_CHARS': '80000',
        }):
            memory = MemoryService(self.store, self.embeddings)
        sources = [await self.ingest('saffron ' + ('x' * 13000 if number == 0 else str(number)),
                                    source_id=number + 1) for number in range(25)]
        result = await memory.search(self.session, 'saffron')
        self.assertEqual(len(result['results']), 25)
        long_result = next(row for row in result['results'] if sources[0].db_id in row['source_ids'])
        self.assertEqual(len(long_result['text']), 13008)
        self.assertFalse(long_result['truncated'])
        read = await memory.read(self.session, [source.db_id for source in sources])
        self.assertEqual(len(read['messages']), 25)
        self.assertEqual(read['omitted_ids'], [])
        self.assertEqual(read['messages'][0]['text'], 'saffron ' + 'x' * 13000)
        self.assertIsNone(read['messages'][0]['next_offset'])
        # Lowering presentation limits leaves the durable original pageable.
        small = MemoryService(self.store, self.embeddings,
            config=replace(memory.config, read_chars=30, response_chars=30))
        first = (await small.read(self.session, [sources[0].db_id]))['messages'][0]
        second = (await small.read(self.session, [sources[0].db_id], offset=first['next_offset']))['messages'][0]
        self.assertEqual(first['text'] + second['text'], ('saffron ' + 'x' * 13000)[:60])

    async def test_configured_context_window_survives_restart_and_retains_evicted_sources(self):
        with patch.dict(os.environ, {'MEMORY_CONTEXT_MESSAGES': '270', 'MEMORY_CACHED_SESSIONS': '1'}):
            limits = from_env(MemoryConfig, 'MEMORY')
        self.runtime.config = replace(self.config, memory=limits)
        sources = [await self.ingest(f'Original {number}', source_id=number + 1) for number in range(272)]
        hot = await self.runtime._get_live_state(self.session)
        expected = [source.db_id for source in sources[-270:]]
        self.assertEqual([source.db_id for source in hot.raw_messages], expected)
        # Loading another chat evicts only reconstructible process state.
        await self.runtime._get_live_state('another-chat')
        restored = await self.runtime._get_live_state(self.session)
        self.assertEqual([source.db_id for source in restored.raw_messages], expected)
        evidence = await self.memory.read(self.session, [sources[0].db_id])
        self.assertEqual(evidence['messages'][0]['text'], 'Original 0')

    async def test_same_display_name_does_not_merge_people_or_profile_subjects(self):
        alice = await self.ingest("I prefer coffee.", "telegram:user:7", 1)
        bob = await self.ingest("I avoid coffee.", "telegram:user:8", 2)
        for source, subject, claim in [(alice, "telegram:user:7", "Prefers coffee"), (bob, "telegram:user:8", "Avoids coffee")]:
            await self.store.save_profile_fact(self.session, subject_actor_id=subject,
                asserted_by=subject, claim=claim, source_ids=[source.db_id])
        result = await self.memory.search(self.session, "coffee")
        ids = {source for row in result["results"] for source in row["source_ids"]}
        self.assertEqual(ids, {alice.db_id, bob.db_id})
        evidence = await self.memory.read(self.session, [alice.db_id, bob.db_id])
        self.assertEqual({row["actor_id"] for row in evidence["messages"]}, {"telegram:user:7", "telegram:user:8"})
        self.assertEqual({row["actor_name"] for row in evidence["messages"]}, {"Alex"})
        recall = await self.memory.recall(self.session, bob.message)
        payload = json.loads(recall.split("\n", 1)[1])
        self.assertEqual([(fact["subject_actor_id"], fact["claim"]) for fact in payload["profiles"]],
                         [("telegram:user:8", "Avoids coffee")])
        self.embeddings.embed_query.assert_not_awaited()

    async def test_soft_reset_clears_working_context_but_retains_historical_evidence_and_profile(self):
        source = await self.ingest("I prefer jasmine tea.")
        await self.settings(system_prompt="Keep this voice.")
        await self.store.save_profile_fact(self.session, subject_actor_id="telegram:user:7",
            asserted_by="telegram:user:7", claim="Prefers jasmine tea", source_ids=[source.db_id])
        old_scope = await self.store.get_scope(self.session)
        new_scope = await self.store.reset_context(self.session)
        self.runtime.invalidate_session(self.session)
        self.assertEqual(new_scope["generation"], old_scope["generation"])
        self.assertNotEqual(new_scope["context_id"], old_scope["context_id"])
        self.assertEqual(await self.store.list_messages(self.session), [])
        self.assertEqual((await self.settings()).system_prompt, "Keep this voice.")
        self.assertTrue((await self.memory.search(self.session, "jasmine"))["results"])
        self.assertEqual((await self.memory.read(self.session, [source.db_id]))["messages"][0]["text"], "I prefer jasmine tea.")
        self.assertEqual(len(await self.store.get_profile(self.session, "telegram:user:7")), 1)
        with self.assertRaises(StaleScopeError):
            await self.memory.read(self.session, [source.db_id], scope=old_scope)

    async def test_full_reset_hides_old_generation_from_search_profile_and_source_reads(self):
        source = await self.ingest("I prefer jasmine tea.")
        await self.store.save_profile_fact(self.session, subject_actor_id="telegram:user:7",
            asserted_by="telegram:user:7", claim="Prefers jasmine tea", source_ids=[source.db_id])
        await self.store.create_excerpt(self.session, [source.db_id])
        await self.store.reset_full(self.session, self.config.default_session_settings())
        self.runtime.invalidate_session(self.session)
        self.assertEqual((await self.memory.search(self.session, "jasmine"))["results"], [])
        self.assertEqual(await self.store.get_profile(self.session, "telegram:user:7"), [])
        read = await self.memory.read(self.session, [source.db_id])
        self.assertEqual(read["messages"], [])
        self.assertEqual(read["unavailable_ids"], [source.db_id])
        # Full reset leaves an audit record, which normal retrieval cannot read.
        async with self.store.pool.connection() as conn:
            count = (await (await conn.execute("SELECT count(*) AS n FROM messages WHERE id=%s", (source.db_id,))).fetchone())["n"]
        self.assertEqual(count, 1)
        fresh = await self.ingest("I now prefer coffee.", source_id=2)
        self.assertEqual((await self.memory.read(self.session, [fresh.db_id]))["messages"][0]["text"], "I now prefer coffee.")

    async def test_deleted_evidence_invalidates_profile_and_cannot_be_read_in_another_chat(self):
        source = await self.ingest("I prefer jasmine tea.")
        await self.store.save_profile_fact(self.session, subject_actor_id="telegram:user:7",
            asserted_by="telegram:user:7", claim="Prefers jasmine tea", source_ids=[source.db_id])
        foreign = await self.memory.read("telegram:200", [source.db_id])
        self.assertEqual(foreign["messages"], [])
        await self.store.delete_message_ids(self.session, [source.db_id])
        self.assertEqual((await self.memory.search(self.session, "jasmine"))["results"], [])
        self.assertEqual(await self.store.get_profile(self.session, "telegram:user:7"), [])
        self.assertEqual((await self.memory.read(self.session, [source.db_id]))["unavailable_ids"], [source.db_id])

    async def test_chat_memory_tools_share_existing_round_budget_and_preserve_exact_prompt(self):
        await self.settings(mode=ChatMode.CHAT, max_interaction_rounds=1,
            system_prompt="  Keep my exact voice.  ", prompt_injection_mode=PromptInjectionMode.EXACT)
        source = await self.ingest("I prefer jasmine tea.")
        self.provider.responses = [ProviderResponse(tool_calls=[ToolCall("memory_read", "read-1", {"message_ids": [source.db_id]})]),
                                   ProviderResponse(final_text="You prefer jasmine tea.")]
        result = await self.runtime.run_turn_from_stored(session_id=self.session,
            user_display_name="command issuer", trigger_message_id=source.db_id)
        self.assertEqual(result.text, "You prefer jasmine tea.")
        self.assertEqual({tool.name for tool in self.provider.requests[0]["tools"]}, {"memory_search", "memory_read"})
        self.assertEqual(self.provider.requests[0]["instructions"], "Keep my exact voice.")
        self.assertTrue(self.provider.requests[1]["instructions"].startswith("Keep my exact voice.\n\n[Internal control note]"))
        self.assertEqual(self.provider.requests[1]["tools"], [])
        self.tools.list_tools.assert_not_called()
        self.tools.runner.run.assert_not_awaited()
        output = self.provider.requests[1]["extra_input_items"][-1]["output"]
        self.assertEqual(output["messages"][0]["actor_id"], "telegram:user:7")

    async def test_final_round_cannot_execute_an_extra_memory_read(self):
        await self.settings(mode=ChatMode.CHAT, max_interaction_rounds=1)
        source = await self.ingest("I prefer jasmine tea.")
        self.memory.read = AsyncMock(wraps=self.memory.read)
        self.provider.responses = [ProviderResponse(tool_calls=[ToolCall("memory_read", f"read-{i}", {"message_ids": [source.db_id]})]) for i in range(2)]
        with self.assertRaisesRegex(RuntimeError, "Interaction-round limit"):
            await self.runtime.run_turn_from_stored(session_id=self.session, user_display_name="Alex", trigger_message_id=source.db_id)
        self.assertEqual(self.memory.read.await_count, 1)
        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual(self.provider.requests[1]["tools"], [])

    async def test_recalled_evidence_and_reply_target_count_against_request_budget(self):
        source = await self.ingest("I prefer jasmine tea.")
        self.provider.responses = [ProviderResponse(final_text="Your preference is jasmine tea.")]
        estimator = self.provider.estimate_request_tokens
        self.provider.estimate_request_tokens = Mock(wraps=estimator)
        await self.runtime.run_turn_from_stored(session_id=self.session,
            user_display_name="Alex", trigger_message_id=source.db_id)
        augmented = [call.kwargs for call in self.provider.estimate_request_tokens.call_args_list
            if call.kwargs.get("instructions") and any(
                message.metadata.get("synthetic_role") == "memory_context" for message in call.kwargs["messages"])]
        self.assertTrue(augmented, "The outgoing request must account for retrieved memory")
        for request in augmented:
            expected = estimator(**{**request, "history_tokens_override": None})
            actual = estimator(**request)
            self.assertEqual(actual.history_tokens, expected.history_tokens,
                "Cached raw-history estimates must not hide the added recall and reply-target tokens")

    async def test_larger_configured_recall_reaches_the_reply_without_a_hidden_token_ceiling(self):
        with patch.dict(os.environ, {'MEMORY_RECALL_TOKENS': '12000',
                'MEMORY_SEARCH_RESULT_CHARS': '30000', 'MEMORY_RESPONSE_CHARS': '40000'}):
            limits = from_env(MemoryConfig, 'MEMORY')
        self.runtime.config = replace(self.config, memory=limits)
        self.runtime.memory = MemoryService(self.store, self.embeddings, config=limits)
        original = 'saffron ' * 3500
        source = await self.ingest(original)
        self.provider.responses = [ProviderResponse(final_text='Evidence retained.')]
        await self.runtime.run_turn_from_stored(session_id=self.session,
            user_display_name='Alex', trigger_message_id=source.db_id)
        recall = next(original_text(message) for message in self.provider.requests[0]['messages']
                      if message.metadata.get('synthetic_role') == 'memory_context')
        self.assertGreater(TokenEstimator.estimate_text(recall), 4096)
        self.assertLessEqual(TokenEstimator.estimate_text(recall), limits.recall_tokens)
        payload = json.loads(recall.split('\n', 1)[1])
        self.assertEqual(payload['recall']['results'][0]['text'], original)

    async def test_model_completion_started_before_reset_cannot_become_a_reply(self):
        await self.settings(mode=ChatMode.CHAT)
        entered, release = asyncio.Event(), asyncio.Event()
        async def generate(**kwargs):
            entered.set()
            await release.wait()
            return ProviderResponse(final_text="stale answer")
        self.provider.generate = generate
        task = asyncio.create_task(self.runtime.run_turn(session_id=self.session,
            user_display_name="Alex", incoming_message=self.message("old question", "telegram:user:7", 1)))
        try:
            await asyncio.wait_for(entered.wait(), timeout=5)
            await self.store.reset_full(self.session, self.config.default_session_settings())
            self.runtime.invalidate_session(self.session)
            release.set()
            with self.assertRaises(StaleScopeError):
                await task
        finally:
            release.set()
            if not task.done():
                task.cancel()
        self.assertEqual(await self.store.list_messages(self.session), [])

    async def test_automatic_recall_timeout_answers_from_current_context_without_duplicating_input(self):
        await self.settings(mode=ChatMode.CHAT, system_prompt="  Keep this exact voice.  ",
            prompt_injection_mode=PromptInjectionMode.EXACT)
        await self.ingest("The meeting is tomorrow.", source_id=1)
        current = await self.ingest("Please summarize the current plan.", source_id=2)
        self.provider.responses = [ProviderResponse(final_text="The meeting is tomorrow.")]
        with patch.object(self.store, "search_excerpts", AsyncMock(side_effect=QueryCanceled("synthetic deadline"))):
            result = await self.runtime.run_turn_from_stored(session_id=self.session,
                user_display_name="Alex", trigger_message_id=current.db_id)
        self.assertEqual(result.text, "The meeting is tomorrow.")
        request = self.provider.requests[0]
        self.assertEqual(request["instructions"], "Keep this exact voice.")
        texts = [original_text(message) for message in request["messages"]]
        self.assertIn("The meeting is tomorrow.", texts)
        self.assertIn("Please summarize the current plan.", texts)
        recall = [message for message in request["messages"]
                  if message.metadata.get("synthetic_role") == "memory_context"]
        self.assertEqual(len(recall), 1)
        recall_text = original_text(recall[0])
        self.assertIn("historical coverage is incomplete", recall_text)
        self.assertIn("Do not claim a complete search", recall_text)
        payload = json.loads(recall_text.split("\n", 1)[1])
        self.assertFalse(payload["recall"]["ok"])
        self.assertIn("unavailable", payload["recall"]["coverage"])
        self.assertEqual(payload["recall"]["results"], [])
        async with self.store.pool.connection() as conn:
            originals = await (await conn.execute(
                "SELECT id FROM messages WHERE session_id=%s AND source_message_id='2'",
                (self.session,))).fetchall()
            revisions = await (await conn.execute(
                "SELECT revision FROM message_revisions WHERE message_id=%s", (current.db_id,))).fetchall()
        self.assertEqual([row["id"] for row in originals], [current.db_id])
        self.assertEqual(len(revisions), 1)

    async def test_reset_during_automatic_recall_timeout_still_cancels_before_generation(self):
        await self.settings(mode=ChatMode.CHAT)
        for source_id, reset_kind in enumerate(("soft", "full"), 1):
            with self.subTest(reset=reset_kind):
                source = await self.ingest("Old question", source_id=source_id)
                entered, release = asyncio.Event(), asyncio.Event()

                async def timed_out_search(*args, **kwargs):
                    entered.set()
                    await release.wait()
                    raise QueryCanceled("synthetic deadline after reset")

                with patch.object(self.store, "search_excerpts", timed_out_search):
                    task = asyncio.create_task(self.runtime.run_turn_from_stored(session_id=self.session,
                        user_display_name="Alex", trigger_message_id=source.db_id))
                    try:
                        await asyncio.wait_for(entered.wait(), timeout=5)
                        if reset_kind == "soft":
                            await self.store.reset_context(self.session)
                        else:
                            await self.store.reset_full(self.session, self.config.default_session_settings())
                        self.runtime.invalidate_session(self.session)
                        release.set()
                        with self.assertRaises(StaleScopeError):
                            await task
                    finally:
                        release.set()
                        if not task.done():
                            task.cancel()
                            await asyncio.gather(task, return_exceptions=True)
                self.assertEqual(self.provider.requests, [])
                self.assertEqual(await self.store.list_messages(self.session), [])

    async def test_recall_timeout_signal_survives_small_budget_but_tool_timeout_is_not_hidden(self):
        source = await self.ingest("Find the old plan.")
        scope = await self.store.get_scope(self.session)
        with patch.object(self.store, "search_excerpts", AsyncMock(side_effect=QueryCanceled("synthetic deadline"))):
            recall = await self.memory.recall(self.session, source.message, scope=scope, max_tokens=64)
            self.assertIn("memory unavailable", recall)
            self.assertIn("coverage is incomplete", recall)
            self.assertLessEqual(TokenEstimator.estimate_text(recall), 64)
            with self.assertRaises(QueryCanceled):
                await self.memory.search(self.session, "old plan", scope=scope)

    async def test_automatic_recall_does_not_hide_cancellation_or_other_database_failures(self):
        source = await self.ingest("Current question")
        scope = await self.store.get_scope(self.session)
        for error in (asyncio.CancelledError(), OperationalError("synthetic connection failure")):
            with self.subTest(error=type(error).__name__):
                with patch.object(self.store, "search_excerpts", AsyncMock(side_effect=error)):
                    with self.assertRaises(type(error)):
                        await self.memory.recall(self.session, source.message, scope=scope)

    def compaction_candidate(self, *, owned):
        return {"scope": "Different drink preferences", "interaction_mode": "chat_or_sharing",
            "participants": ["Alex"], "topics": ["drinks", "preferences"],
            "user_profile": ["telegram:user:7: Prefers coffee", "telegram:user:8: Avoids coffee"] if owned else ["Prefers coffee", "Avoids coffee"],
            "user_intent_or_shared_context": ["Two people share different preferences"],
            "why_it_mattered": [], "interaction_timeline": ["telegram:user:7 stated a preference; telegram:user:8 disagreed"],
            "results_or_takeaways": [], "decisions": [], "open_loops": [], "artifacts": [], "uncertainties": []}

    async def test_compaction_corrects_ownerless_claims_before_committing_two_same_name_people(self):
        settings = await self.settings(min_raw_messages_reserve=1)
        alice = await self.ingest("I prefer coffee.", "telegram:user:7", 1)
        bob = await self.ingest("I avoid coffee.", "telegram:user:8", 2)
        latest = await self.ingest("Latest question stays raw.", "telegram:user:7", 3)
        self.provider.responses = [ProviderResponse(final_text=json.dumps(self.compaction_candidate(owned=False))),
                                   ProviderResponse(final_text=json.dumps(self.compaction_candidate(owned=True)))]
        state = await self.runtime._get_live_state(self.session)
        changed = await self.runtime._compact_old_context(session_id=self.session, settings=settings,
            provider=self.provider, state=state, pressure=True)
        self.assertTrue(changed)
        self.assertEqual(len(self.provider.requests), 2)
        request_text = "\n".join(part.text or "" for message in self.provider.requests[0]["messages"] for part in message.parts)
        for item, actor in [(alice, "telegram:user:7"), (bob, "telegram:user:8")]:
            self.assertIn(actor, request_text)
            self.assertIn(f'"message_id": {item.db_id}', request_text)
        block = state.blocks[0]
        self.assertEqual(list(block.actor_labels), ["telegram:user:7", "telegram:user:8"])
        self.assertEqual(block.structured_data["user_profile"], self.compaction_candidate(owned=True)["user_profile"])
        self.assertEqual([item.db_id for item in state.raw_messages], [latest.db_id])
        originals = await self.memory.read(self.session, [alice.db_id, bob.db_id])
        self.assertEqual([item["text"] for item in originals["messages"]], ["I prefer coffee.", "I avoid coffee."])

    async def test_repeated_ownerless_compaction_is_rejected_without_hiding_sources(self):
        alice = await self.ingest("I prefer coffee.", "telegram:user:7", 1)
        bob = await self.ingest("I avoid coffee.", "telegram:user:8", 2)
        self.provider.responses = [ProviderResponse(final_text=json.dumps(self.compaction_candidate(owned=False))) for _ in range(2)]
        candidate = await self.runtime._make_episode_block_candidate(self.provider, await self.settings(),
            [alice.message, bob.message], [alice, bob], [])
        self.assertIsNone(candidate)
        self.assertEqual(len(self.provider.requests), 2)
        self.assertEqual([item.db_id for item in await self.store.list_uncompacted_messages(self.session)], [alice.db_id, bob.db_id])
        self.assertEqual(await self.store.list_memory_blocks(self.session), [])

    async def test_compaction_source_actor_list_does_not_drop_the_ninth_person(self):
        rows = [await self.ingest(f"Participant {i}", f"telegram:user:{i}", i) for i in range(1, 11)]
        metadata = self.runtime._compaction_metadata_message(mode="episode", raw_messages=rows,
            parent_blocks=[], time_start=None, time_end=None)
        expected = [f"telegram:user:{i}" for i in range(1, 11)]
        self.assertEqual(metadata.metadata["compaction_actor_ids"], expected)
        for actor in expected:
            self.assertIn(actor, metadata.parts[0].text)
