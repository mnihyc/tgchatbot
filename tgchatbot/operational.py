"""Optional operational settings read from the application's existing environment.

Defaults belong to the component's configuration, never to a second env file.
Database capacity measurements are deliberately not configuration or quotas.
"""
from __future__ import annotations

from dataclasses import dataclass, fields
import math
import os
from typing import Mapping, TypeVar, get_args, get_type_hints

T = TypeVar('T')


def from_env(cls: type[T], prefix: str, env: Mapping[str, str] | None = None) -> T:
    """Load typed dataclass overrides; absent or blank values retain defaults."""
    env = os.environ if env is None else env
    hints = get_type_hints(cls)
    values = {}
    for field in fields(cls):
        name = f'{prefix}_{field.name.upper()}'
        raw = env.get(name, '').strip()
        if not raw:
            continue
        kind = hints[field.name]
        if type(None) in get_args(kind):
            kind = next(arg for arg in get_args(kind) if arg is not type(None))
        try:
            if kind is bool:
                if raw.lower() not in {'true', 'false', '1', '0', 'yes', 'no', 'on', 'off'}:
                    raise ValueError('expected a boolean')
                value = raw.lower() in {'true', '1', 'yes', 'on'}
            else:
                value = kind(raw)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f'{name}: {exc}') from None
        values[field.name] = value
    return cls(**values)


@dataclass(frozen=True)
class MemoryConfig:
    # Prompt windows and disposable caches; originals remain in PostgreSQL.
    cached_sessions: int = 32
    preview_cache_bytes: int = 256 * 1024 * 1024
    replay_cache_bytes: int = 32 * 1024 * 1024
    query_chars: int = 2048
    query_timeout_s: float = 5.0
    search_results: int = 20
    search_result_chars: int = 4000
    response_chars: int = 24000
    read_messages: int = 20
    read_chars: int = 12000
    profile_facts: int = 20

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            zero_allowed = field.name in {'preview_cache_bytes', 'replay_cache_bytes'}
            if not math.isfinite(value) or value < 0 or (value == 0 and not zero_allowed):
                requirement = 'nonnegative' if zero_allowed else 'positive'
                raise ValueError(f'MEMORY_{field.name.upper()} must be {requirement}')
