from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import httpx


@dataclass(slots=True)
class LexicalHit:
    sticker_id: str
    score: float
    fields: list[str]
    debug: dict[str, float]


class TantivyRetrieverClient:
    def __init__(self, base_url: str, *, timeout_s: float = 2.0) -> None:
        self.base_url = base_url.rstrip('/')
        self.timeout = httpx.Timeout(timeout_s, connect=min(timeout_s, 0.5))
        self._client = httpx.Client(timeout=self.timeout)

    def close(self) -> None:
        self._client.close()

    def health(self) -> dict[str, Any] | None:
        try:
            response = self._client.get(f'{self.base_url}/health')
            response.raise_for_status()
            body = response.json()
            return body if isinstance(body, dict) else None
        except Exception:
            return None

    def ensure_healthy(self, *, expected_schema_version: str | None = None, expected_service: str | None = None) -> None:
        body = self.health()
        if body is None:
            raise RuntimeError(f'Required Tantivy retriever is unavailable at {self.base_url}')
        if expected_schema_version and str(body.get('schema_version', '')) != expected_schema_version:
            raise RuntimeError(f"Tantivy retriever schema_version mismatch: expected {expected_schema_version}, got {body.get('schema_version')}")
        if expected_service and str(body.get('service', '')) != expected_service:
            raise RuntimeError(f"Tantivy retriever service mismatch: expected {expected_service}, got {body.get('service')}")

    def search(self, payload: dict[str, Any]) -> list[LexicalHit]:
        self.ensure_healthy()
        response = self._client.post(f'{self.base_url}/search', json=payload)
        response.raise_for_status()
        body = response.json()
        hits: list[LexicalHit] = []
        for item in body.get('hits', []):
            hits.append(
                LexicalHit(
                    sticker_id=str(item['sticker_id']),
                    score=float(item.get('score', 0.0)),
                    fields=[str(x) for x in item.get('fields', [])],
                    debug={str(k): float(v) for k, v in (item.get('debug') or {}).items()},
                )
            )
        return hits
