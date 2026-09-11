import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
import numpy as np
from tgchatbot.embeddings import EmbeddingConfig
from tgchatbot.stickers.catalog import StickerCatalog
from tgchatbot.stickers.plan import StickerRetrievalPlan
from tgchatbot.stickers.semantic_index import EmbeddingProvider, SemanticIndex


def config(**options):
    return EmbeddingConfig(**{"api_key": "fake", "dimensions": 2,
        "base_url": "https://example.invalid/v1beta", "requests_per_minute": 1e12,
        "max_retries": 0, **options})


class StickerProviderTests(unittest.TestCase):
    def test_fresh_install_needs_neither_embedding_key_nor_sticker_assets(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            catalog = StickerCatalog(Path(tmp) / "index.db", Path(tmp) / "stickers")
            try:
                catalog.load()
                self.assertEqual(catalog.stats()["stickers"], 0)
                self.assertFalse(catalog.semantic_enabled)
            finally:
                catalog.retriever.close()

    def test_offline_facade_uses_shared_gemini_contract_and_query_cache(self):
        seen = []
        def handle(request):
            seen.append(json.loads(request.content))
            return httpx.Response(200, json={"embedding": {"values": [3, 4]}})
        http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        provider = EmbeddingProvider(config=config(), http_client=http)
        try:
            np.testing.assert_allclose(provider.embed("query"), [0.6, 0.8])
            provider.embed("query")
            self.assertEqual(len(seen), 1)
            self.assertEqual(seen[0]["content"]["parts"], [{"text": "task: search result | query: query"}])
            self.assertEqual(seen[0]["embedContentConfig"]["outputDimensionality"], 2)
        finally:
            provider.close()
            asyncio.run(http.aclose())

    def test_sticker_adapter_reuses_global_config_instead_of_a_second_secret(self):
        env = {"GEMINI_API_KEY": "shared", "GEMINI_BASE_URL": "https://shared.invalid/v1beta",
               "OPENAI_API_KEY": "generation-only", "EMBEDDING_DIMENSIONS": "1536"}
        with patch.dict(os.environ, env, clear=True):
            provider = EmbeddingProvider.from_env()
            self.assertEqual(provider.config, EmbeddingConfig.from_env())
            self.assertEqual(provider.api_key, "shared")
            self.assertEqual(provider.base_url, "https://shared.invalid/v1beta")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "unrelated"}, clear=True):
            self.assertEqual(EmbeddingProvider(backend="openai").api_key, "")

    def write_index(self, root, settings):
        (root / "embeddings_manifest.json").write_text(json.dumps({"sticker_ids": ["one"],
            "dimensions": settings.dimensions, "embedding_model": settings.model,
            "embedding_backend": settings.provider, "embedding_space_id": settings.space_id,
            "embedding_base_url": settings.base_url}))
        np.save(root / "caption_embeddings.npy", np.asarray([[1, 0]], dtype=np.float32))
        np.save(root / "sticker_embeddings.npy", np.asarray([[0, 1]], dtype=np.float32))

    def test_space_changes_require_rebuild_but_route_relocation_does_not(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = config()
            self.write_index(root, original)
            moved = EmbeddingProvider(config=replace(original, base_url="https://moved.invalid/v1beta"))
            SemanticIndex(root, embedding_provider=moved).ensure_ready()
            changed = EmbeddingProvider(config=replace(original, model="gemini-embedding-001"))
            with self.assertRaisesRegex(RuntimeError, "rebuild"):
                SemanticIndex(root, embedding_provider=changed).ensure_ready()

    def test_old_index_without_format_fingerprint_requires_explicit_rebuild(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            settings = config()
            self.write_index(root, settings)
            manifest_path = root / "embeddings_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            del manifest["embedding_space_id"]
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "rebuild"):
                SemanticIndex(root, embedding_provider=EmbeddingProvider(config=settings)).ensure_ready()

    def test_lexical_mode_ignores_corrupt_unused_embedding_manifest(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"GEMINI_API_KEY": "fake", "STICKER_SEMANTIC_MODE": "off"}, clear=True):
            for name in ("embeddings_manifest.json", "caption_embeddings.npy", "sticker_embeddings.npy"):
                (Path(tmp) / name).write_text("corrupt")
            catalog = StickerCatalog(Path(tmp) / "index.db", Path(tmp) / "stickers")
            try:
                catalog.load()
                self.assertFalse(catalog.semantic_enabled)
            finally:
                catalog.retriever.close()

    def test_offline_build_uses_document_format_and_preserves_response_index_order(self):
        counts = []
        def handle(request):
            inputs = json.loads(request.content)["input"]
            counts.append(len(inputs))
            return httpx.Response(200, json={"data": [{"index": i, "embedding": [3, 4]} for i in reversed(range(len(inputs)))]})
        http = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        provider = EmbeddingProvider(config=config(provider="openai", model="explicit-model"), http_client=http)
        try:
            result = provider.embed_many(["text"] * 129)
            self.assertEqual(counts, [128, 1])
            self.assertEqual(result.shape, (129, 2))
            np.testing.assert_allclose(result, np.tile([0.6, 0.8], (129, 1)))
        finally:
            provider.close()
            asyncio.run(http.aclose())


class AsyncStickerProviderTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = Mock(get_sticker_persona=AsyncMock(return_value=None),
                          save_sticker_persona=AsyncMock(), clear_sticker_persona=AsyncMock())
        with patch.dict(os.environ, {}, clear=True):
            self.catalog = StickerCatalog(Path(self.directory.name) / "index.db", Path(self.directory.name), persona_store=self.store)
        self.catalog._loaded = True
        self.addCleanup(self.catalog.retriever.close)

    async def test_live_adapter_awaits_shared_service_and_preserves_semantic_ranking(self):
        service = Mock(config=config(), embed_query=AsyncMock(side_effect=[np.array([1, 0]), np.array([0, 1])]))
        adapter = EmbeddingProvider(client=service)
        index = SemanticIndex(Path(self.directory.name), embedding_provider=adapter)
        index.loaded = True
        index.sticker_ids = ["caption-match", "visual-match"]
        index.caption_vectors = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
        index.sticker_vectors = np.asarray([[1, 0], [0, 1]], dtype=np.float32)
        hits = await index.asearch(caption_query_text="caption", sticker_query_text="visual", top_k=1)
        self.assertEqual([(hit.sticker_id, hit.channel) for hit in hits],
                         [("caption-match", "caption_semantic"), ("visual-match", "sticker_semantic")])
        self.assertEqual(service.embed_query.await_args_list[0].kwargs, {"purpose": "sticker"})
        self.assertEqual(service.embed_query.await_count, 2)

    async def test_async_catalog_reuses_existing_scoring_after_awaited_retrieval(self):
        plan = StickerRetrievalPlan.from_payload({"intent_core": "warm support"})
        self.catalog.entries_by_id = {"candidate": object()}
        self.catalog.semantic_enabled = True
        self.catalog.semantic_index = Mock(asearch=AsyncMock(return_value=["semantic-hit"]))
        self.catalog._search_lexical = Mock(return_value=["lexical-hit"])
        self.catalog.choose = Mock(return_value=["ranked-match"])
        result = await self.catalog.achoose(plan=plan, session_id="group-a")
        self.assertEqual(result, ["ranked-match"])
        self.catalog.semantic_index.asearch.assert_awaited_once()
        self.assertEqual(self.catalog.choose.call_args.kwargs["_lexical_hits"], ["lexical-hit"])
        self.assertEqual(self.catalog.choose.call_args.kwargs["_semantic_hits"], ["semantic-hit"])
        self.store.get_sticker_persona.assert_awaited_once_with("group-a")

    async def test_persona_load_and_clear_are_awaited_and_session_scoped(self):
        self.store.get_sticker_persona.return_value = {"affect_profile": {"default_tone": "warm"}}
        await self.catalog.adescribe_style_context("group-a")
        self.store.get_sticker_persona.assert_awaited_once_with("group-a")
        plan = StickerRetrievalPlan.from_payload({"intent_core": "hello", "persona_mode": "clear_session_persona"})
        await self.catalog.aprepare_query_context(plan=plan, session_id="group-a", persist_persona=False)
        self.store.clear_sticker_persona.assert_not_awaited()
        self.assertEqual(self.catalog.style_memory.get("group-a").session_persona["affect_profile"]["default_tone"], "warm")
        await self.catalog.aprepare_query_context(plan=plan, session_id="group-a", persist_persona=True)
        self.store.clear_sticker_persona.assert_awaited_once_with("group-a")
        self.assertIsNone(self.catalog.style_memory.get("group-a").session_persona)
        await self.catalog.adescribe_style_context("group-b")
        self.assertEqual(self.store.get_sticker_persona.await_args.args, ("group-b",))

    async def test_remembered_persona_merges_but_one_off_overlay_does_not_write(self):
        self.store.get_sticker_persona.return_value = {"affect_profile": {"default_tone": "warm"}}
        plan = StickerRetrievalPlan.from_payload({"intent_core": "hello", "persona_mode": "merge_and_remember",
            "persona": {"visual_identity": {"character_archetype": "cat"}}})
        await self.catalog.aprepare_query_context(plan=plan, session_id="group-a", persist_persona=True)
        self.store.save_sticker_persona.assert_awaited_once_with("group-a", {
            "visual_identity": {"character_archetype": "cat"}, "affect_profile": {"default_tone": "warm"}})
        self.store.save_sticker_persona.reset_mock()
        one_off = StickerRetrievalPlan.from_payload({"intent_core": "hello", "persona_mode": "use_once",
            "persona": {"affect_profile": {"default_tone": "amused"}}})
        state, context = await self.catalog.aprepare_query_context(plan=one_off, session_id="group-a", persist_persona=True)
        self.assertEqual(context["effective_persona"]["affect_profile"]["default_tone"], "amused")
        self.assertEqual(state.session_persona["affect_profile"]["default_tone"], "warm")
        self.store.save_sticker_persona.assert_not_awaited()

    async def test_failed_persona_write_restores_in_memory_state(self):
        self.store.get_sticker_persona.return_value = {"affect_profile": {"default_tone": "warm"}}
        self.store.clear_sticker_persona.side_effect = RuntimeError("db failed")
        plan = StickerRetrievalPlan.from_payload({"intent_core": "hello", "persona_mode": "clear_session_persona"})
        with self.assertRaisesRegex(RuntimeError, "db failed"):
            await self.catalog.aprepare_query_context(plan=plan, session_id="group-a", persist_persona=True)
        self.assertEqual(self.catalog.style_memory.get("group-a").session_persona["affect_profile"]["default_tone"], "warm")

    async def test_full_reset_discards_only_target_chat_persona_and_implicit_continuity(self):
        self.store.get_sticker_persona.return_value = {"affect_profile": {"default_tone": "warm"}}
        await self.catalog.adescribe_style_context("group-a")
        await self.catalog.adescribe_style_context("group-b")
        self.catalog.style_memory.preload("group-a", recent_sticker_ids=["old-a"], recent_source_pack_ids=["old-pack"])
        self.catalog.style_memory.preload("group-b", recent_sticker_ids=["old-b"])
        self.catalog.entries_by_id = {"shared": object()}
        self.catalog.reset_session("group-a")
        self.store.get_sticker_persona.return_value = None
        await self.catalog.adescribe_style_context("group-a")
        reset = self.catalog.style_memory.get("group-a")
        self.assertIsNone(reset.session_persona)
        self.assertEqual(list(reset.recent_sticker_ids), [])
        self.assertIsNone(reset.source_pack_id)
        self.assertEqual(list(self.catalog.style_memory.get("group-b").recent_sticker_ids), ["old-b"])
        self.assertEqual(list(self.catalog.entries_by_id), ["shared"])
