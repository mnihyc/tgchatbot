from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import httpx
import numpy as np

from tgchatbot.embeddings import EmbeddingClient, EmbeddingConfig, EmbeddingService, SyncEmbeddingClient


@dataclass(slots=True)
class SemanticHit:
    sticker_id: str
    score: float
    channel: str


class EmbeddingProvider:
    """Sticker adapter for the shared embedding service.

    Runtime callers use ``aembed``. ``embed``/``embed_many`` are an offline-only
    facade for the maintenance scripts. Query caches belong to the shared
    client; no second sticker key or persistent plaintext cache is created.
    """

    def __init__(self, *, config: EmbeddingConfig | None = None,
                 client: EmbeddingService | None = None, api_key: str | None = None,
                 model: str | None = None, dimensions: int | None = None,
                 base_url: str | None = None, backend: str | None = None,
                 cache_db_path: Path | None = None,
                 http_client: httpx.AsyncClient | None = None) -> None:
        if client is not None:
            config = client.config
        if config is None:
            selected = backend or 'gemini'
            config = EmbeddingConfig(
                provider=selected,
                api_key=api_key or '',
                model=model or ('gemini-embedding-2' if selected == 'gemini' else 'text-embedding-3-large'),
                dimensions=dimensions if dimensions is not None else 1536,
                base_url=base_url or ('https://generativelanguage.googleapis.com/v1beta' if selected == 'gemini' else 'https://api.openai.com/v1'),
            )
        self.config = config
        self.api_key = config.api_key
        self.backend = config.provider
        self.model = config.model
        self.dimensions = config.dimensions
        self.base_url = config.base_url
        self.space_id = config.space_id
        self._client = client
        self._http_client = http_client
        self._owns_client = client is None
        self._sync: SyncEmbeddingClient | None = None

    @classmethod
    def from_env(cls, *, cache_db_path: Path | None = None) -> 'EmbeddingProvider':
        return cls(config=EmbeddingConfig.from_env())

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _offline(self) -> SyncEmbeddingClient:
        SyncEmbeddingClient._check_offline()
        if self._client is not None:
            raise RuntimeError('This sticker adapter uses an async client; await aembed/asearch instead')
        if self._sync is None:
            self._sync = SyncEmbeddingClient(self.config, http_client=self._http_client)
        return self._sync

    def embed(self, text: str) -> np.ndarray:
        return self._offline().embed_query(text, purpose='sticker')

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        return np.stack(self._offline().embed_documents(texts, purpose='sticker'))

    async def aembed(self, text: str) -> np.ndarray:
        if self._sync is not None:
            raise RuntimeError('An offline sticker embedding adapter cannot be reused on the live event loop')
        if self._client is None:
            self._client = EmbeddingClient(self.config, http_client=self._http_client)
        return await self._client.embed_query(text, purpose='sticker')

    def close(self) -> None:
        if self._sync is not None:
            self._sync.close()

    async def aclose(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()


class SemanticIndex:
    def __init__(self, index_dir: Path, *, embedding_provider: EmbeddingProvider | None = None, require_ready: bool = True) -> None:
        self.index_dir = Path(index_dir)
        self.embedding_provider = embedding_provider or EmbeddingProvider.from_env()
        self.sticker_ids: list[str] = []
        self.caption_vectors: np.ndarray | None = None
        self.sticker_vectors: np.ndarray | None = None
        self.loaded = False
        self.require_ready = require_ready

    def load(self) -> None:
        manifest_path = self.index_dir / 'embeddings_manifest.json'
        caption_path = self.index_dir / 'caption_embeddings.npy'
        sticker_path = self.index_dir / 'sticker_embeddings.npy'
        missing = [str(path.name) for path in (manifest_path, caption_path, sticker_path) if not path.exists()]
        if missing:
            if self.require_ready:
                raise RuntimeError(f'Missing required embedding artifacts: {", ".join(missing)} in {self.index_dir}')
            self.loaded = True
            self.sticker_ids = []
            self.caption_vectors = None
            self.sticker_vectors = None
            return
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if not manifest.get('enabled', True):
            raise RuntimeError('This sticker index was built without embeddings; rebuild embeddings or set STICKER_SEMANTIC_MODE=off.')
        provider = self.embedding_provider
        if (manifest.get('embedding_space_id') != provider.space_id
                or manifest.get('embedding_model') != provider.model
                or int(manifest.get('dimensions', 0)) != provider.dimensions
                or manifest.get('embedding_backend') != provider.backend):
            raise RuntimeError('Embedding space/model/backend/dimensions changed; rebuild embeddings or set STICKER_SEMANTIC_MODE=off.')
        self.sticker_ids = [str(x) for x in manifest.get('sticker_ids', [])]
        self.caption_vectors = np.load(caption_path, mmap_mode='r')
        self.sticker_vectors = np.load(sticker_path, mmap_mode='r')
        expected = (len(self.sticker_ids), provider.dimensions)
        if self.caption_vectors.shape != expected or self.sticker_vectors.shape != expected:
            raise RuntimeError('Embedding matrix shape does not match its manifest; rebuild embeddings.')
        self.loaded = True

    def ensure_ready(self) -> None:
        if not self.loaded:
            self.load()
        if not self.embedding_provider.enabled:
            raise RuntimeError('Configure an embedding API key or use STICKER_SEMANTIC_MODE=off.')
        if not self.sticker_ids or self.caption_vectors is None or self.sticker_vectors is None:
            raise RuntimeError(f'Semantic index is not ready in {self.index_dir}')

    def search(self, *, caption_query_text: str, sticker_query_text: str, top_k: int = 50) -> list[SemanticHit]:
        self.ensure_ready()
        caption_vec = self.embedding_provider.embed(caption_query_text)
        sticker_vec = self.embedding_provider.embed(sticker_query_text)
        return self._rank(caption_vec, sticker_vec, top_k)

    async def asearch(self, *, caption_query_text: str, sticker_query_text: str, top_k: int = 50) -> list[SemanticHit]:
        self.ensure_ready()
        caption_vec = await self.embedding_provider.aembed(caption_query_text)
        sticker_vec = caption_vec if sticker_query_text == caption_query_text else await self.embedding_provider.aembed(sticker_query_text)
        return self._rank(caption_vec, sticker_vec, top_k)

    def _rank(self, caption_vec: np.ndarray, sticker_vec: np.ndarray, top_k: int) -> list[SemanticHit]:
        k = max(1, min(int(top_k), len(self.sticker_ids)))
        hits: list[SemanticHit] = []

        caption_scores = np.asarray(self.caption_vectors @ caption_vec, dtype=np.float32)
        caption_idx = np.argpartition(-caption_scores, k - 1)[:k]
        for idx in caption_idx:
            hits.append(SemanticHit(sticker_id=self.sticker_ids[int(idx)], score=float(caption_scores[int(idx)]), channel='caption_semantic'))

        sticker_scores = np.asarray(self.sticker_vectors @ sticker_vec, dtype=np.float32)
        sticker_idx = np.argpartition(-sticker_scores, k - 1)[:k]
        for idx in sticker_idx:
            hits.append(SemanticHit(sticker_id=self.sticker_ids[int(idx)], score=float(sticker_scores[int(idx)]), channel='sticker_semantic'))

        hits.sort(key=lambda item: -item.score)
        return hits
