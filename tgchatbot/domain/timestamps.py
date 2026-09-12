"""Timestamp presentation without rewriting stored evidence or participant text."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
import os
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tgchatbot.settings_schema import DEFAULT_METADATA_TIMEZONE


def resolve_timezone(timezone_name: str | None = None) -> ZoneInfo:
    name = timezone_name or os.getenv('DEFAULT_METADATA_TIMEZONE', '').strip() or DEFAULT_METADATA_TIMEZONE
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError as exc:
        raise ValueError(f'Unknown timestamp timezone: {name}') from exc


def format_timestamp(value: Any, timezone_name: str | None = None) -> str | None:
    """Render a known timestamp; unzoned stored values mean UTC, never local time."""
    if value is None:
        return None
    if isinstance(value, (int, float)) or isinstance(value, str) and value.isdigit():
        value = datetime.fromtimestamp(float(value), timezone.utc)
    elif isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(resolve_timezone(timezone_name)).isoformat()


def format_timestamp_fields(record: Mapping[str, Any], fields: Iterable[str],
                            timezone_name: str | None = None) -> dict[str, Any]:
    """Copy a record and format only timestamp fields explicitly owned by its schema."""
    fields = frozenset(fields)
    return {key: format_timestamp(value, timezone_name) if key in fields else value
            for key, value in record.items()}
