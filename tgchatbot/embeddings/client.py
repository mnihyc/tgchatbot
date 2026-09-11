"""Shared text retrieval embeddings; transport never selects a generation model.

API contracts: https://ai.google.dev/api/embeddings and
https://ai.google.dev/gemini-api/docs/batch-api . Gemini 001 deliberately uses
the flat fields: the service ignored nested configuration in our live probe.
"""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
import math
import re
from typing import Any, Literal, Protocol, Sequence
from urllib.parse import urlencode

import httpx
import numpy as np

from tgchatbot.embeddings.config import EmbeddingConfig

Purpose = Literal["memory", "sticker"]
Kind = Literal["document", "query"]


@dataclass(frozen=True, slots=True)
class EmbeddingDocument:
    item_id: str
    text: str
    title: str | None = None


@dataclass(slots=True)
class BatchJob:
    name: str
    state: str
    done: bool
    space_id: str
    item_ids: tuple[str, ...] = ()
    output: dict[str, Any] = field(default_factory=dict, repr=False)
    error: dict[str, Any] | None = None
    display_name: str = ""


@dataclass(slots=True)
class BatchItemResult:
    item_id: str
    vector: np.ndarray | None = None
    error: dict[str, Any] | None = None

    @property
    def ok(self) -> bool:
        return self.vector is not None and self.error is None


class EmbeddingService(Protocol):
    config: EmbeddingConfig

    async def embed_documents(self, items: Sequence[EmbeddingDocument | str], *, purpose: Purpose = "memory") -> list[np.ndarray]: ...
    async def embed_query(self, text: str, *, purpose: Purpose = "memory") -> np.ndarray: ...
    async def count_tokens(self, text: str, *, kind: Kind = "document", title: str | None = None) -> int: ...
    async def submit_batch(self, items: Sequence[EmbeddingDocument], *, purpose: Purpose = "memory", display_name: str = "tgchatbot-embeddings") -> BatchJob: ...
    async def poll_batch(self, name: str) -> BatchJob: ...
    async def find_batch(self, display_name: str, *, max_pages: int | None = None) -> BatchJob | None: ...
    async def read_batch_results(self, job: BatchJob, expected_ids: Sequence[str] | None = None) -> list[BatchItemResult]: ...
    async def aclose(self) -> None: ...


class EmbeddingClient:
    def __init__(self, config: EmbeddingConfig | None = None, *, http_client: httpx.AsyncClient | None = None) -> None:
        self.config = config or EmbeddingConfig.from_env()
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.config.timeout_s, connect=self.config.connect_timeout_s),
            limits=httpx.Limits(max_connections=self.config.max_concurrency, max_keepalive_connections=self.config.max_concurrency),
            follow_redirects=False,
        )
        self._slots = asyncio.Semaphore(self.config.max_concurrency)
        self._rate_lock = asyncio.Lock()
        self._next_request = 0.0
        self._cache: OrderedDict[str, np.ndarray | int] = OrderedDict()

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    @property
    def space_id(self) -> str:
        return self.config.space_id

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> "EmbeddingClient":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    def _key(self, kind: str, purpose: str, payload: Any) -> str:
        return hashlib.sha256(json.dumps([self.space_id, kind, purpose, payload], sort_keys=True, ensure_ascii=False).encode()).hexdigest()

    def _cached(self, key: str) -> np.ndarray | int | None:
        value = self._cache.get(key)
        if value is not None:
            self._cache.move_to_end(key)
            return value.copy() if isinstance(value, np.ndarray) else value
        return None

    def _remember(self, key: str, value: np.ndarray | int) -> None:
        if self.config.cache_entries == 0:
            return
        self._cache[key] = value.copy() if isinstance(value, np.ndarray) else value
        self._cache.move_to_end(key)
        while len(self._cache) > self.config.cache_entries:
            self._cache.popitem(last=False)

    async def _pace(self) -> None:
        # Configurable, evenly spaced requests prevent bulk callers from bursts.
        async with self._rate_lock:
            loop = asyncio.get_running_loop()
            delay = self._next_request - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            self._next_request = loop.time() + 60.0 / self.config.requests_per_minute

    async def _request(self, method: str, path: str, *, payload: Any = None,
                       retry: bool = True, absolute_url: str | None = None) -> httpx.Response:
        if not self.enabled:
            raise RuntimeError("Configure the embedding API key; generation credentials are not substituted")
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["x-goog-api-key" if self.config.provider == "gemini" else "Authorization"] = (
                self.config.api_key if self.config.provider == "gemini" else f"Bearer {self.config.api_key}")
        attempts = self.config.max_retries + 1 if retry else 1
        backoff = min(self.config.retry_initial_delay_s, self.config.retry_max_delay_s)
        for attempt in range(attempts):
            await self._pace()
            try:
                async with self._slots:
                    response = await self._http.request(
                        method, absolute_url or f"{self.config.base_url}/{path}",
                        headers=headers, json=payload,
                        timeout=httpx.Timeout(self.config.timeout_s, connect=self.config.connect_timeout_s),
                    )
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code not in {408, 429, 500, 502, 503, 504} or attempt + 1 == attempts:
                    raise
                retry_after = exc.response.headers.get("Retry-After", "")
                try:
                    delay = float(retry_after)
                    if not math.isfinite(delay):
                        raise ValueError
                    # Do not retry earlier than a long server-directed cooldown.
                    if delay > self.config.retry_after_max_s:
                        raise exc
                    delay = max(0.0, delay)
                except ValueError:
                    delay = backoff
            except httpx.TransportError:
                if attempt + 1 == attempts:
                    raise
                delay = backoff
            backoff = min(backoff * 2, self.config.retry_max_delay_s)
            await asyncio.sleep(delay)
        raise AssertionError("unreachable retry state")

    @staticmethod
    def _text(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Embedding input must be non-empty text")
        return text.strip()

    @staticmethod
    def _purpose(purpose: Purpose) -> None:
        if purpose not in {"memory", "sticker"}:
            raise ValueError("Embedding purpose must be memory or sticker")

    def _format(self, text: str, kind: Kind, title: str | None = None) -> str:
        text = self._text(text)
        if kind not in {"document", "query"}:
            raise ValueError("Embedding kind must be document or query")
        if self.config.provider == "gemini" and self.config.model != "gemini-embedding-001":
            if kind == "query":
                return f"task: search result | query: {text}"
            return f"title: {(title or '').strip() or 'none'} | text: {text}"
        return text

    def _gemini_request(self, text: str, kind: Kind, title: str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": f"models/{self.config.model}",
            "content": {"parts": [{"text": self._format(text, kind, title)}]},
        }
        if self.config.model == "gemini-embedding-001":
            # These flat fields are required by the actual 001 service. Never
            # retry with unspecified task/dimension when they are rejected.
            payload.update(outputDimensionality=self.config.dimensions,
                           taskType="RETRIEVAL_QUERY" if kind == "query" else "RETRIEVAL_DOCUMENT")
            if kind == "document" and title:
                payload["title"] = title
        else:
            payload["embedContentConfig"] = {"outputDimensionality": self.config.dimensions, "autoTruncate": False}
        return payload

    def _vector(self, values: Any) -> np.ndarray:
        if not isinstance(values, list) or any(type(value) not in {int, float} for value in values):
            raise ValueError("Embedding response must contain a list of numeric values")
        try:
            vector = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("Embedding response contains invalid values") from exc
        if vector.shape != (self.config.dimensions,) or not np.isfinite(vector).all():
            raise ValueError("Embedding response dimensions or values are invalid")
        # Scaling before normalization avoids overflow/underflow in hostile or
        # broken responses. Never pad, silently truncate, or accept zero vectors.
        scale = float(np.max(np.abs(vector)))
        if scale == 0:
            raise ValueError("Embedding response is a zero vector")
        vector /= scale
        vector /= np.linalg.norm(vector)
        return vector.astype(np.float32)

    @staticmethod
    def _body(response: httpx.Response) -> dict[str, Any]:
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Embedding API response must be a JSON object")
        return body

    def _gemini_vector(self, body: dict[str, Any]) -> np.ndarray:
        embedding = body.get("embedding")
        if not isinstance(embedding, dict):
            raise ValueError("Embedding API response is missing its embedding object")
        return self._vector(embedding.get("values"))

    async def count_tokens(self, text: str, *, kind: Kind = "document", title: str | None = None) -> int:
        if self.config.provider != "gemini":
            raise NotImplementedError("The OpenAI-compatible embedding protocol has no countTokens endpoint")
        formatted = self._format(text, kind, title)
        if self.config.model == "gemini-embedding-001" and title and kind == "document":
            formatted = f"{title}\n{formatted}"
        key = self._key("tokens", kind, formatted)
        cached = self._cached(key)
        if cached is not None:
            return int(cached)
        response = await self._request("POST", f"models/{self.config.model}:countTokens",
                                       payload={"contents": [{"parts": [{"text": formatted}]}]})
        count = self._body(response).get("totalTokens")
        if type(count) is not int or count < 0:
            raise ValueError("Embedding countTokens response is invalid")
        self._remember(key, count)
        return count

    async def _legacy_length_check(self, text: str, kind: Kind, title: str | None = None) -> None:
        if self.config.provider == "gemini" and self.config.model == "gemini-embedding-001":
            # 001 cannot rely on nested autoTruncate either; reject oversized
            # inputs using its native tokenizer before sending an embedding.
            if await self.count_tokens(text, kind=kind, title=title) > 2048:
                raise ValueError("Gemini embedding 001 input exceeds 2048 tokens; split the document")

    async def embed_query(self, text: str, *, purpose: Purpose = "memory") -> np.ndarray:
        self._purpose(purpose)
        payload = self._gemini_request(text, "query") if self.config.provider == "gemini" else {
            "model": self.config.model, "input": self._format(text, "query"), "dimensions": self.config.dimensions,
        }
        key = self._key("query", purpose, payload)
        cached = self._cached(key)
        if cached is not None:
            return np.asarray(cached)
        await self._legacy_length_check(text, "query")
        path = f"models/{self.config.model}:embedContent" if self.config.provider == "gemini" else "embeddings"
        response = self._body(await self._request("POST", path, payload=payload))
        if self.config.provider == "gemini":
            vector = self._gemini_vector(response)
        else:
            vector = self._openai_vectors(response, 1)[0]
        self._remember(key, vector)
        return vector

    def _openai_vectors(self, body: dict[str, Any], count: int) -> list[np.ndarray]:
        rows = body.get("data", [])
        if (not isinstance(rows, list) or len(rows) != count
                or any(not isinstance(row, dict) or type(row.get("index")) is not int for row in rows)
                or sorted(row["index"] for row in rows) != list(range(count))):
            raise ValueError("Embedding batch response has missing or duplicate input indices")
        return [self._vector(row.get("embedding")) for row in sorted(rows, key=lambda row: row["index"])]

    async def embed_documents(self, items: Sequence[EmbeddingDocument | str], *, purpose: Purpose = "memory") -> list[np.ndarray]:
        self._purpose(purpose)
        documents = [item if isinstance(item, EmbeddingDocument) else EmbeddingDocument(str(i), item)
                     for i, item in enumerate(items)]
        result: list[np.ndarray] = []
        # Gemini's native protocol accepts at most 100 requests. Compatible
        # routes use the operator's configured batch size without another cap.
        batch_size = min(100, self.config.sync_batch_items) if self.config.provider == "gemini" else self.config.sync_batch_items
        for offset in range(0, len(documents), batch_size):
            batch = documents[offset:offset + batch_size]
            for item in batch:
                self._text(item.text)
                await self._legacy_length_check(item.text, "document", item.title)
            if self.config.provider == "gemini":
                payload = {"requests": [self._gemini_request(item.text, "document", item.title) for item in batch]}
                body = self._body(await self._request("POST", f"models/{self.config.model}:batchEmbedContents", payload=payload))
                rows = body.get("embeddings", [])
                if not isinstance(rows, list) or len(rows) != len(batch) or any(not isinstance(row, dict) for row in rows):
                    raise ValueError("Embedding batch response count does not match its inputs")
                vectors = [self._vector(row.get("values")) for row in rows]
            else:
                payload = {"model": self.config.model, "dimensions": self.config.dimensions,
                           "input": [self._format(item.text, "document", item.title) for item in batch]}
                vectors = self._openai_vectors(self._body(await self._request("POST", "embeddings", payload=payload)), len(batch))
            result.extend(vectors)
        return result

    @staticmethod
    def _ids(items: Sequence[EmbeddingDocument]) -> tuple[str, ...]:
        ids = tuple(item.item_id for item in items)
        if any(not isinstance(item_id, str) or not item_id.strip() for item_id in ids) or len(set(ids)) != len(ids):
            raise ValueError("Embedding batch item IDs must be non-empty and unique")
        return ids

    @staticmethod
    def _resource(name: str, prefix: str) -> str:
        if not isinstance(name, str) or not re.fullmatch(rf"{prefix}/[A-Za-z0-9_.-]+", name) or name.split("/")[-1] in {".", ".."}:
            raise ValueError(f"Invalid embedding {prefix} resource name")
        return name

    def _job(self, body: dict[str, Any], *, item_ids: tuple[str, ...] = ()) -> BatchJob:
        name = self._resource(body.get("name", ""), "batches")
        metadata = body.get("metadata") or {}
        state = metadata.get("state", body.get("state", "JOB_STATE_PENDING"))
        terminal = state in {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED", "JOB_STATE_PARTIALLY_SUCCEEDED"}
        output = body.get("response") or body.get("output") or {}
        # Some operation responses wrap EmbedContentBatch rather than its output.
        output = output.get("output", output)
        return BatchJob(name=name, state=state, done=bool(body.get("done", terminal)),
                        space_id=self.space_id, item_ids=item_ids, output=output,
                        error=body.get("error"), display_name=metadata.get("displayName", body.get("displayName", "")))

    async def submit_batch(self, items: Sequence[EmbeddingDocument], *, purpose: Purpose = "memory",
                           display_name: str = "tgchatbot-embeddings") -> BatchJob:
        self._purpose(purpose)
        if self.config.provider != "gemini":
            raise NotImplementedError("Async embedding Batch is implemented for native Gemini only")
        if not items or len(items) > self.config.batch_max_items:
            raise ValueError("Split embedding batches within the configured item limit")
        item_ids = self._ids(items)
        requests = []
        for item in items:
            await self._legacy_length_check(item.text, "document", item.title)
            requests.append({"request": self._gemini_request(item.text, "document", item.title),
                             "metadata": {"key": item.item_id, "space_id": self.space_id, "purpose": purpose}})
        payload = {"batch": {"displayName": display_name, "inputConfig": {"requests": {"requests": requests}}}}
        if len(json.dumps(payload, ensure_ascii=False).encode()) > self.config.batch_max_bytes:
            raise ValueError("Split embedding batches within the configured byte limit")
        # Batch creation is not idempotent: do not automatically resubmit after
        # an ambiguous timeout and create duplicate paid jobs. Worker reconciles.
        response = await self._request("POST", f"models/{self.config.model}:asyncBatchEmbedContent", payload=payload, retry=False)
        job = self._job(self._body(response), item_ids=item_ids)
        job.display_name = job.display_name or display_name
        return job

    async def poll_batch(self, name: str) -> BatchJob:
        if self.config.provider != "gemini":
            raise NotImplementedError("Async embedding Batch is implemented for native Gemini only")
        job = self._job(self._body(await self._request("GET", self._resource(name, "batches"))))
        if job.name != name:
            raise ValueError("Batch polling returned a different operation")
        return job

    async def list_batches(self, *, page_token: str | None = None, page_size: int | None = None) -> tuple[list[BatchJob], str | None]:
        if self.config.provider != "gemini":
            raise NotImplementedError("Async embedding Batch is implemented for native Gemini only")
        page_size = self.config.batch_list_page_size if page_size is None else page_size
        if page_size <= 0:
            raise ValueError("Batch list page size must be positive")
        params = {"pageSize": str(page_size)}
        if page_token:
            params["pageToken"] = page_token
        body = self._body(await self._request("GET", f"batches?{urlencode(params)}"))
        if body.get("unreachable"):
            raise RuntimeError("Batch reconciliation is incomplete; some operations are unreachable")
        rows = body.get("operations", [])
        if not isinstance(rows, list) or (not rows and set(body) - {"operations", "nextPageToken", "unreachable"}):
            raise ValueError("Malformed batch listing response")
        return [self._job(row) for row in rows], body.get("nextPageToken") or None

    async def find_batch(self, display_name: str, *, max_pages: int | None = None) -> BatchJob | None:
        """Reconcile an ambiguous submission using its persisted unique name.

        A missing match is not proof that a timed-out submission was rejected:
        listings can lag. The durable worker decides when to reconcile again.
        """
        max_pages = self.config.batch_reconcile_max_pages if max_pages is None else max_pages
        if not display_name or max_pages <= 0:
            raise ValueError("A stable display name and positive page limit are required")
        token: str | None = None
        matches: dict[str, BatchJob] = {}
        seen_tokens: set[str] = set()
        for _ in range(max_pages):
            jobs, token = await self.list_batches(page_token=token)
            matches.update((job.name, job) for job in jobs if job.display_name == display_name)
            if len(matches) > 1:
                raise RuntimeError("Ambiguous batch submission: multiple operations have the same display name")
            if not token:
                return next(iter(matches.values()), None)
            if token in seen_tokens:
                raise RuntimeError("Batch reconciliation listing repeated a page token")
            seen_tokens.add(token)
        raise RuntimeError("Batch reconciliation page limit reached; no safe absence conclusion")

    async def read_batch_results(self, job: BatchJob, expected_ids: Sequence[str] | None = None) -> list[BatchItemResult]:
        if job.space_id != self.space_id:
            raise ValueError("Embedding batch belongs to another embedding space")
        if not job.done:
            raise ValueError("Embedding batch has not finished")
        ids = tuple(expected_ids) if expected_ids is not None else job.item_ids
        if not ids or len(set(ids)) != len(ids) or any(not isinstance(item_id, str) or not item_id for item_id in ids):
            raise ValueError("Expected stable batch item IDs are required")
        output = job.output
        inline = output.get("inlinedResponses", {})
        rows = inline.get("inlinedResponses", []) if isinstance(inline, dict) else inline
        if output.get("responsesFile"):
            name = self._resource(output["responsesFile"], "files")
            base, slash, version = self.config.base_url.rpartition("/")
            if not slash or version not in {"v1beta", "v1"}:
                raise ValueError("Gemini file download requires a versioned API root")
            url = f"{base}/download/{version}/{name}:download?alt=media"
            response = await self._request("GET", "", absolute_url=url)
            rows = [json.loads(line) for line in response.text.splitlines() if line.strip()]
        if not isinstance(rows, list):
            raise ValueError("Malformed embedding batch output")
        by_id: dict[str, BatchItemResult] = {}
        expected = set(ids)
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("metadata", {}), dict):
                raise ValueError("Malformed embedding batch result metadata")
            metadata = row.get("metadata", {})
            item_id = metadata.get("key", row.get("key"))
            if item_id not in expected or item_id in by_id:
                raise ValueError("Embedding batch returned unknown, missing, or duplicate item IDs")
            if metadata.get("space_id", self.space_id) != self.space_id:
                raise ValueError("Embedding batch result has the wrong embedding space")
            if row.get("error"):
                result = BatchItemResult(item_id=item_id, error=row["error"])
            else:
                try:
                    result = BatchItemResult(item_id=item_id, vector=self._gemini_vector(row.get("response", {})))
                except (AttributeError, TypeError, ValueError) as exc:
                    result = BatchItemResult(item_id=item_id, error={"code": "INVALID_EMBEDDING", "message": str(exc)})
            by_id[item_id] = result
        # Preserve every partial failure for durable retries; never zip truncated
        # output against the input and accidentally attach a vector to a person.
        return [by_id.get(item_id, BatchItemResult(item_id=item_id, error=job.error or {
            "code": "MISSING_RESULT", "message": f"No result in terminal batch ({job.state})"})) for item_id in ids]


class SyncEmbeddingClient:
    """Offline-only facade. The live bot awaits EmbeddingClient directly."""

    def __init__(self, config: EmbeddingConfig, *, http_client: httpx.AsyncClient | None = None) -> None:
        self._runner = asyncio.Runner()
        self.client = EmbeddingClient(config, http_client=http_client)
        self._closed = False

    @staticmethod
    def _check_offline() -> None:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        raise RuntimeError("Synchronous embeddings cannot run on an event loop; await the shared embedding client")

    def embed_query(self, text: str, *, purpose: Purpose = "sticker") -> np.ndarray:
        self._check_offline()
        return self._runner.run(self.client.embed_query(text, purpose=purpose))

    def embed_documents(self, items: Sequence[EmbeddingDocument | str], *, purpose: Purpose = "sticker") -> list[np.ndarray]:
        self._check_offline()
        return self._runner.run(self.client.embed_documents(items, purpose=purpose))

    def close(self) -> None:
        if not self._closed:
            self._check_offline()
            self._runner.run(self.client.aclose())
            self._runner.close()
            self._closed = True

    def __enter__(self) -> "SyncEmbeddingClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
