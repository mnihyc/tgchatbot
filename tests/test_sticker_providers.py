import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
from tgchatbot.stickers.catalog import StickerCatalog
from tgchatbot.stickers.semantic_index import EmbeddingProvider, SemanticIndex


class StickerProviderTests(unittest.TestCase):
    def test_fresh_install_needs_neither_openai_nor_sticker_assets(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {}, clear=True):
            catalog = StickerCatalog(Path(tmp) / "index.db", Path(tmp) / "stickers")
            try:
                catalog.load()
                self.assertEqual(catalog.stats()["stickers"], 0)
                self.assertFalse(catalog.semantic_enabled)
            finally:
                catalog.retriever.close()

    def test_gemini_embeddings_use_native_protocol_and_normalize(self):
        response = Mock()
        response.json.return_value = {"embedding": {"values": [3, 4]}}
        with patch("tgchatbot.stickers.semantic_index.httpx.post", return_value=response) as post:
            provider = EmbeddingProvider(api_key="fake", backend="gemini", model="gemini-embedding-001", dimensions=2,
                                         base_url="https://example.invalid/v1beta")
            np.testing.assert_allclose(provider.embed("query"), [0.6, 0.8])
            self.assertEqual(post.call_args.args[0], "https://example.invalid/v1beta/models/gemini-embedding-001:embedContent")
            self.assertEqual(post.call_args.kwargs["headers"], {"x-goog-api-key": "fake"})
            self.assertEqual(post.call_args.kwargs["json"]["outputDimensionality"], 2)
            provider.embed("query")
            self.assertEqual(post.call_count, 1)

    def test_separate_embedding_credentials_and_endpoint_changes_reach_the_service(self):
        response = Mock()
        response.json.return_value = {"data": [{"embedding": [1, 0]}]}
        with TemporaryDirectory() as tmp, patch("tgchatbot.stickers.semantic_index.httpx.post", return_value=response) as post:
            for url in ["https://first.invalid/v1", "https://second.invalid/v1"]:
                settings = {"OPENAI_API_KEY": "chat-only", "STICKER_EMBEDDING_API_KEY": "embedding-only",
                            "STICKER_EMBEDDING_BASE_URL": url, "STICKER_EMBEDDING_DIMENSIONS": "2"}
                with patch.dict(os.environ, settings, clear=True):
                    p = EmbeddingProvider.from_env(cache_db_path=Path(tmp) / "cache.db")
                    p.embed("same query")
            self.assertEqual(post.call_count, 2)
            self.assertEqual(post.call_args.args[0], "https://second.invalid/v1/embeddings")
            for request in post.call_args_list:
                self.assertEqual(request.kwargs["headers"]["Authorization"], "Bearer embedding-only")

    def test_invalid_embedding_vectors_fail_before_cache_write(self):
        for vector in [[1], [0, 0], [float("nan"), 1]]:
            response = Mock()
            response.json.return_value = {"data": [{"embedding": vector}]}
            with patch("tgchatbot.stickers.semantic_index.httpx.post", return_value=response):
                with self.assertRaises(ValueError):
                    EmbeddingProvider(api_key="fake", dimensions=2).embed("query")

    def test_switching_embedding_model_requires_rebuild(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "embeddings_manifest.json").write_text(json.dumps({"sticker_ids": ["one"], "dimensions": 2, "embedding_model": "old-model"}))
            np.save(root / "caption_embeddings.npy", np.ones((1, 2)))
            np.save(root / "sticker_embeddings.npy", np.ones((1, 2)))
            index = SemanticIndex(root, embedding_provider=EmbeddingProvider(api_key="fake", model="new-model", dimensions=2))
            with self.assertRaisesRegex(RuntimeError, "rebuild"):
                index.ensure_ready()

    def test_lexical_mode_ignores_corrupt_unused_embedding_manifest(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"OPENAI_API_KEY": "fake", "STICKER_SEMANTIC_MODE": "off"}, clear=True):
            for name in ("embeddings_manifest.json", "caption_embeddings.npy", "sticker_embeddings.npy"):
                (Path(tmp) / name).write_text("corrupt")
            catalog = StickerCatalog(Path(tmp) / "index.db", Path(tmp) / "stickers")
            try:
                catalog.load()
                self.assertFalse(catalog.semantic_enabled)
            finally:
                catalog.retriever.close()

    def test_changing_recorded_embedding_endpoint_requires_rebuild(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "embeddings_manifest.json").write_text(json.dumps({"sticker_ids": ["one"], "dimensions": 2, "embedding_model": "same-model", "embedding_base_url": "https://old.invalid/v1"}))
            np.save(root / "caption_embeddings.npy", np.ones((1, 2)))
            np.save(root / "sticker_embeddings.npy", np.ones((1, 2)))
            index = SemanticIndex(root, embedding_provider=EmbeddingProvider(api_key="fake", model="same-model", dimensions=2, base_url="https://new.invalid/v1"))
            with self.assertRaisesRegex(RuntimeError, "rebuild"):
                index.ensure_ready()

    def test_build_embeddings_preserves_batching_and_response_index_order(self):
        def post(url, **kwargs):
            inputs = kwargs["json"]["input"]
            response = Mock()
            response.json.return_value = {"data": [{"index": i, "embedding": [3, 4]} for i in reversed(range(len(inputs)))]}
            return response
        with patch("tgchatbot.stickers.semantic_index.httpx.post", side_effect=post) as request:
            result = EmbeddingProvider(api_key="fake", dimensions=2).embed_many(["text"] * 129)
            self.assertEqual(request.call_count, 2)
            self.assertEqual(result.shape, (129, 2))
            np.testing.assert_allclose(result, np.tile([0.6, 0.8], (129, 1)))
