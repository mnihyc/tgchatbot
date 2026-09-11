from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import os
import sqlite3

import httpx
import numpy as np


@dataclass(slots=True)
class SemanticHit:
    sticker_id: str
    score: float
    channel: str


class EmbeddingProvider:
    def __init__(self, *, api_key: str | None = None, model: str = 'text-embedding-3-large', dimensions: int = 1024, base_url: str = 'https://api.openai.com/v1', cache_db_path: Path | None = None, backend: str = 'openai') -> None:
        self.api_key = api_key if api_key is not None else os.getenv('OPENAI_API_KEY')
        if backend not in {'openai', 'gemini'}:
            raise ValueError('Embedding backend must be openai (compatible API) or gemini')
        if dimensions <= 0:
            raise ValueError('Embedding dimensions must be positive')
        self.backend = backend
        self.model = model
        self.dimensions = dimensions
        self.base_url = base_url.rstrip('/')
        self._cache: dict[str, np.ndarray] = {}
        self.cache_db_path = Path(cache_db_path) if cache_db_path else None
        if self.cache_db_path is not None:
            self.cache_db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(self.cache_db_path) as con:
                con.execute('CREATE TABLE IF NOT EXISTS query_embeddings (cache_key TEXT PRIMARY KEY, model TEXT NOT NULL, dimensions INTEGER NOT NULL, vector_json TEXT NOT NULL)')
                con.commit()

    @classmethod
    def from_env(cls, *, cache_db_path: Path | None = None) -> 'EmbeddingProvider':
        backend = os.getenv('STICKER_EMBEDDING_PROVIDER', 'openai').strip().lower()
        prefix = 'GEMINI' if backend == 'gemini' else 'OPENAI'
        default_model = 'gemini-embedding-001' if backend == 'gemini' else 'text-embedding-3-large'
        default_url = 'https://generativelanguage.googleapis.com/v1beta' if backend == 'gemini' else 'https://api.openai.com/v1'
        return cls(
            backend=backend,
            api_key=os.getenv('STICKER_EMBEDDING_API_KEY', os.getenv(f'{prefix}_API_KEY', '')),
            base_url=os.getenv('STICKER_EMBEDDING_BASE_URL', os.getenv(f'{prefix}_BASE_URL', default_url)),
            model=os.getenv('STICKER_EMBEDDING_MODEL', default_model),
            dimensions=int(os.getenv('STICKER_EMBEDDING_DIMENSIONS', '1024')),
            cache_db_path=cache_db_path,
        )

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def _cache_lookup(self, cache_key: str) -> np.ndarray | None:
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        if self.cache_db_path is None:
            return None
        with sqlite3.connect(self.cache_db_path) as con:
            row = con.execute('SELECT vector_json FROM query_embeddings WHERE cache_key=? AND model=? AND dimensions=?', (cache_key, self.model, self.dimensions)).fetchone()
        if not row:
            return None
        vector = np.asarray(json.loads(row[0]), dtype=np.float32)
        self._cache_store(cache_key, vector)
        return vector

    def _cache_store(self, cache_key: str, vector: np.ndarray) -> None:
        self._cache[cache_key] = vector
        if self.cache_db_path is None:
            return
        with sqlite3.connect(self.cache_db_path) as con:
            con.execute('INSERT INTO query_embeddings(cache_key, model, dimensions, vector_json) VALUES(?, ?, ?, ?) ON CONFLICT(cache_key) DO UPDATE SET model=excluded.model, dimensions=excluded.dimensions, vector_json=excluded.vector_json', (cache_key, self.model, self.dimensions, json.dumps(vector.tolist())))
            con.commit()

    def embed(self, text: str) -> np.ndarray:
        text = (text or '').strip()
        if not text:
            raise RuntimeError('Semantic retrieval requires a non-empty query string.')
        if not self.api_key:
            raise RuntimeError('Configure an embedding API key or use STICKER_SEMANTIC_MODE=off.')
        # Endpoint and protocol are part of the vector space identity. Switching
        # a compatible endpoint must not reuse another service's cached vector.
        cache_key = hashlib.sha256(f'{self.backend}:{self.base_url}:{self.model}:{self.dimensions}:{text}'.encode('utf-8')).hexdigest()
        cached = self._cache_lookup(cache_key)
        if cached is not None:
            return cached
        if self.backend == 'gemini':
            model = self.model.removeprefix('models/')
            response = httpx.post(
                f'{self.base_url}/models/{model}:embedContent',
                headers={'x-goog-api-key': self.api_key},
                timeout=httpx.Timeout(15.0, connect=3.0),
                json={'model': f'models/{model}', 'content': {'parts': [{'text': text}]},
                      'outputDimensionality': self.dimensions},
            )
        else:
            response = httpx.post(
                f'{self.base_url}/embeddings',
                headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
                timeout=httpx.Timeout(15.0, connect=3.0),
                json={'model': self.model, 'input': text, 'dimensions': self.dimensions},
            )
        response.raise_for_status()
        body = response.json()
        data = body['embedding']['values'] if self.backend == 'gemini' else body['data'][0]['embedding']
        vector = np.asarray(data, dtype=np.float32)
        if vector.shape != (self.dimensions,) or not np.isfinite(vector).all():
            raise ValueError('Embedding response dimensions or values are invalid')
        norm = np.linalg.norm(vector)
        if norm <= 0:
            raise ValueError('Embedding response is a zero vector')
        vector = vector / norm
        self._cache_store(cache_key, vector)
        return vector

    def embed_many(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimensions), dtype=np.float32)
        if self.backend == 'gemini':
            return np.stack([self.embed(text) for text in texts])
        if not self.enabled:
            raise RuntimeError('Configure an embedding API key before building embeddings.')
        batches: list[np.ndarray] = []
        # Preserve the legacy builder's 128-input batches. Sending every sticker
        # individually would multiply HTTP overhead and request-rate usage.
        for start in range(0, len(texts), 128):
            inputs = texts[start:start + 128]
            response = httpx.post(
                f'{self.base_url}/embeddings',
                headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
                timeout=httpx.Timeout(300.0, connect=30.0),
                json={'model': self.model, 'input': inputs, 'dimensions': self.dimensions},
            )
            response.raise_for_status()
            rows = response.json()['data']
            if len(rows) != len(inputs) or sorted(row['index'] for row in rows) != list(range(len(inputs))):
                raise ValueError('Embedding batch response has missing or duplicate input indices')
            batch = np.asarray([row['embedding'] for row in sorted(rows, key=lambda row: row['index'])], dtype=np.float32)
            if batch.shape != (len(inputs), self.dimensions) or not np.isfinite(batch).all():
                raise ValueError('Embedding batch dimensions or values are invalid')
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            if np.any(norms <= 0):
                raise ValueError('Embedding batch contains a zero vector')
            batches.append(batch / norms)
        return np.vstack(batches)


class SemanticIndex:
    def __init__(self, index_dir: Path, *, embedding_provider: EmbeddingProvider | None = None, require_ready: bool = True) -> None:
        self.index_dir = Path(index_dir)
        self.embedding_provider = embedding_provider or EmbeddingProvider()
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
        if (manifest.get('embedding_model') != provider.model
                or int(manifest.get('dimensions', 0)) != provider.dimensions
                or manifest.get('embedding_backend', provider.backend) != provider.backend
                or manifest.get('embedding_base_url', provider.base_url).rstrip('/') != provider.base_url):
            raise RuntimeError('Embedding endpoint/model/backend/dimensions changed; rebuild embeddings or set STICKER_SEMANTIC_MODE=off.')
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
