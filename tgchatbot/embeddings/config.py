"""Embedding settings are deliberately independent of generation profiles."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from typing import Mapping
from urllib.parse import urlsplit

from tgchatbot.operational import from_env as operational_from_env


@dataclass(frozen=True, slots=True)
class EmbeddingConfig:
    provider: str = "gemini"
    model: str = "gemini-embedding-2"
    dimensions: int = 1536
    api_key: str = field(default="", repr=False)
    base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    model_revision: str = ""
    timeout_s: float = 15.0
    connect_timeout_s: float = 3.0
    max_retries: int = 2
    retry_initial_delay_s: float = 0.5
    retry_max_delay_s: float = 8.0
    retry_after_max_s: float = 30.0
    max_concurrency: int = 4
    requests_per_minute: float = 120.0
    cache_entries: int = 512
    # Inline submissions stay comfortably below the provider's 20 MB limit.
    batch_max_items: int = 1000
    batch_max_bytes: int = 8 * 1024 * 1024
    sync_batch_items: int = 128
    batch_list_page_size: int = 100
    batch_reconcile_max_pages: int = 20

    def __post_init__(self) -> None:
        if self.provider not in {"gemini", "openai"}:
            raise ValueError("EMBEDDING_PROVIDER must be gemini or openai (compatible API)")
        object.__setattr__(self, "model", self.model.removeprefix("models/") if self.provider == "gemini" else self.model)
        object.__setattr__(self, "base_url", self.base_url.rstrip("/"))
        if not self.model or self.dimensions <= 0:
            raise ValueError("Embedding model and positive dimensions are required")
        if self.provider == "gemini" and self.model in {"gemini-embedding-2", "gemini-embedding-001"} and self.dimensions > 3072:
            raise ValueError("Gemini embedding dimensions cannot exceed 3072")
        url = urlsplit(self.base_url)
        if url.scheme not in {"http", "https"} or not url.netloc or url.query or url.fragment or url.username or url.password:
            raise ValueError("Embedding base URL must be an HTTP(S) API root without credentials or query parameters")
        for name in ('timeout_s', 'connect_timeout_s', 'requests_per_minute', 'max_concurrency',
                     'batch_max_items', 'batch_max_bytes', 'sync_batch_items', 'batch_list_page_size',
                     'batch_reconcile_max_pages'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'EMBEDDING_{name.upper()} must be finite and positive')
        for name in ('max_retries', 'cache_entries', 'retry_initial_delay_s', 'retry_max_delay_s', 'retry_after_max_s'):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f'EMBEDDING_{name.upper()} must be finite and nonnegative')

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "EmbeddingConfig":
        env = os.environ if env is None else env
        provider = env.get("EMBEDDING_PROVIDER", "gemini").strip().lower() or 'gemini'
        if provider in {"openai-compatible", "openai_compatible"}:
            provider = "openai"
        prefix = "GEMINI" if provider == "gemini" else "OPENAI"
        default_model = "gemini-embedding-2" if provider == "gemini" else "text-embedding-3-large"
        default_url = "https://generativelanguage.googleapis.com/v1beta" if provider == "gemini" else "https://api.openai.com/v1"
        key_env = env.get("EMBEDDING_API_KEY_ENV") or f"{prefix}_API_KEY"
        selected = {**env,
            'EMBEDDING_PROVIDER': provider,
            'EMBEDDING_MODEL': env.get('EMBEDDING_MODEL', '').strip() or default_model,
            'EMBEDDING_API_KEY': env.get('EMBEDDING_API_KEY', env.get(key_env, '')),
            'EMBEDDING_BASE_URL': env.get('EMBEDDING_BASE_URL', '').strip()
                or env.get(f'{prefix}_BASE_URL', '').strip() or default_url}
        return operational_from_env(cls, 'EMBEDDING', selected)

    @property
    def enabled(self) -> bool:
        if self.provider == "gemini" or urlsplit(self.base_url).hostname == "api.openai.com":
            return bool(self.api_key)
        # Explicit compatible routes can be unauthenticated. Never invent a key.
        return True

    @property
    def input_format(self) -> str:
        if self.provider == "gemini":
            return "gemini001-retrieval-task-v1" if self.model == "gemini-embedding-001" else "gemini2-search-title-v1"
        return "openai-text-v1"

    @property
    def space_spec(self) -> dict[str, str | int]:
        return {"provider": self.provider, "model": self.model, "model_revision": self.model_revision,
                "dimensions": self.dimensions, "normalization": "l2-f32-v1", "input_format": self.input_format}

    @property
    def space_id(self) -> str:
        # Moving the same model to another route does not change its vector space.
        return hashlib.sha256(json.dumps(self.space_spec, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
