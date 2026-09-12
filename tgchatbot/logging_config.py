from __future__ import annotations

import logging
import os
from typing import Any

import json
from datetime import datetime, timezone
from pathlib import Path

from tgchatbot.domain.timestamps import format_timestamp, resolve_timezone

_DEFAULT_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
_NOISY_LOGGERS = ("httpx", "httpcore")


def resolve_log_level(level: str | None = None) -> int:
    raw = (level or os.getenv("LOG_LEVEL", "INFO")).strip().upper()
    if raw.isdigit():
        return int(raw)
    return getattr(logging, raw, logging.INFO)


def clip_for_log(value: Any, *, limit: int = 160, rlimit: int = 0) -> str:
    if value is None:
        return ''

    text = ' '.join(str(value).split())
    limit = max(0, limit)
    rlimit = max(0, rlimit)

    # No truncation needed
    if len(text) <= limit + rlimit:
        return text

    # Nothing to keep
    if limit == 0 and rlimit == 0:
        return '…'

    # Left only
    if rlimit == 0:
        return text[:limit] + '…'

    # Right only
    if limit == 0:
        return '…' + text[-rlimit:]

    # Both sides
    return text[:limit] + '…' + text[-rlimit:]


class TimezoneFormatter(logging.Formatter):
    def __init__(self, fmt: str, timezone_name: str | None = None) -> None:
        super().__init__(fmt)
        self.timezone_name = resolve_timezone(timezone_name).key

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        rendered = format_timestamp(record.created, self.timezone_name)
        return datetime.fromisoformat(rendered).strftime(datefmt) if datefmt else rendered


def configure_logging(level: str | None = None, *, timezone_name: str | None = None) -> int:
    resolved_level = resolve_log_level(level)
    handler = logging.StreamHandler()
    handler.setFormatter(TimezoneFormatter(_DEFAULT_LOG_FORMAT, timezone_name))
    logging.basicConfig(level=resolved_level, handlers=[handler], force=True)
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger(__name__).info('logging.ready level=%s httpx=WARNING', logging.getLevelName(resolved_level))
    return resolved_level


def dump_llm_exchange(*, provider: str, model: str, url: str, payload: dict[str, Any], response: Any | None = None, error: str | None = None) -> Path | None:
    if not logging.getLogger().isEnabledFor(logging.DEBUG):
        return None

    logs_dir = Path(os.getenv('APP_DATA_DIR', './data')).resolve() / 'logs'
    logs_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc)
    timestamp = now.strftime('%Y%m%dT%H%M%S_%fZ')
    path = logs_dir / f'provider_{timestamp}.json'
    body: dict[str, Any] = {
        'timestamp': now.isoformat(),
        'provider': provider,
        'model': model,
        'url': url,
        'request': {
            'payload': payload,
        },
    }

    if response is not None:
        try:
            response_body: Any = response.json()
        except Exception:
            response_body = response.text

        body['response'] = {
            'status_code': response.status_code,
            'headers': dict(response.headers),
            'body': response_body,
        }

    if error is not None:
        body['error'] = error

    path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding='utf-8')
    return path
