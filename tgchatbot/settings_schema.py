from __future__ import annotations

import math
from typing import Iterable


DEFAULT_METADATA_TIMEZONE = "Asia/Singapore"

REASONING_EFFORT_VALUES = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
REASONING_SUMMARY_VALUES = frozenset({"off", "on", "auto", "detailed", "concise"})
TEXT_VERBOSITY_VALUES = frozenset({"low", "medium", "high"})
GEMINI_THINKING_LEVEL_VALUES = frozenset({"minimal", "low", "medium", "high"})
GEMINI_THINKING_BUDGET_MIN = -1

NATIVE_WEB_SEARCH_MAX_MIN = 0

TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 2.0
TOP_P_MIN = 0.0
TOP_P_MAX = 1.0
TOP_K_MIN = 1

MAX_OUTPUT_TOKENS_MIN = 1

IMAGE_LIMIT_DISABLED = 0

COMPACT_TOKEN_MIN = 1
COMPACT_KEEP_RECENT_RATIO_MIN = 0.0
COMPACT_KEEP_RECENT_RATIO_MAX = 1.0
COMPACT_TOOL_RATIO_THRESHOLD_MIN = 0.0
COMPACT_MIN_MESSAGES_MIN = 2
MIN_RAW_MESSAGES_RESERVE_MIN = 0

MAX_INTERACTION_ROUNDS_MIN = 1
SPONTANEOUS_REPLY_CHANCE_MIN = 0
SPONTANEOUS_REPLY_CHANCE_MAX = 100
PROVIDER_RETRY_COUNT_MIN = 0


def normalize_choice(value: str | None, default: str, allowed: Iterable[str]) -> str:
    allowed_set = {item.strip().lower() for item in allowed}
    selected = (value or "").strip().lower()
    return selected if selected in allowed_set else default


def normalize_optional_choice(value: str | None, allowed: Iterable[str]) -> str | None:
    allowed_set = {item.strip().lower() for item in allowed}
    selected = (value or "").strip().lower()
    return selected if selected in allowed_set else None


def parse_bounded_int_env(value: str | None, *, default: int, minimum: int, maximum: int | None = None) -> int:
    raw = (value or "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    if parsed < minimum or (maximum is not None and parsed > maximum):
        return default
    return parsed


def parse_optional_bounded_int_env(value: str | None, *, minimum: int, maximum: int | None = None) -> int | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        parsed = int(raw)
    except ValueError:
        return None
    if parsed < minimum or (maximum is not None and parsed > maximum):
        return None
    return parsed


def parse_bounded_float_env(value: str | None, *, default: float, minimum: float, maximum: float | None = None) -> float:
    raw = (value or "").strip()
    if not raw:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        return default
    if not math.isfinite(parsed) or parsed < minimum or (maximum is not None and parsed > maximum):
        return default
    return parsed


def parse_optional_disabled_int_env(value: str | None, *, default: int, maximum: int | None = None) -> int:
    raw = (value or "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    if parsed == 0:
        return 0
    if parsed < 0 or (maximum is not None and parsed > maximum):
        return default
    return parsed


def clamp_int(value: int | None, *, minimum: int, maximum: int | None = None, default: int) -> int:
    if value is None:
        return default
    parsed = max(minimum, int(value))
    return min(maximum, parsed) if maximum is not None else parsed


def clamp_float(value: float | None, *, minimum: float, maximum: float | None = None, default: float) -> float:
    if value is None:
        return default
    if not math.isfinite(float(value)):
        return default
    parsed = max(minimum, float(value))
    return min(maximum, parsed) if maximum is not None else parsed


def normalize_optional_disabled_int(value: int | None, *, maximum: int | None = None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed == 0:
        return 0
    if parsed < 0:
        return None
    return min(maximum, parsed) if maximum is not None else parsed


def normalize_optional_bounded_int(value: int | None, *, minimum: int, maximum: int | None = None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < minimum or (maximum is not None and parsed > maximum):
        return None
    return parsed


def effective_optional_disabled_int(override: int | None, default: int, *, maximum: int | None = None) -> int | None:
    configured = normalize_optional_disabled_int(override, maximum=maximum)
    if configured is None:
        configured = normalize_optional_disabled_int(default, maximum=maximum)
    if configured in {None, 0}:
        return None
    return configured


def format_optional_disabled_int(value: int | None, *, disabled_label: str = "disabled") -> str:
    return disabled_label if value is None else str(int(value))


def normalize_reasoning_summary_value(value: str | None) -> str | None:
    return normalize_optional_choice(value, REASONING_SUMMARY_VALUES)


def effective_reasoning_summary(value: str | None, *, provider: str, default: str = "off") -> str:
    raw = normalize_reasoning_summary_value(value)
    if raw is None:
        raw = normalize_reasoning_summary_value(default) or "off"
    provider_name = (provider or "").strip().lower()
    if provider_name == "gemini":
        return "off" if raw == "off" else "on"
    if provider_name == "openai":
        return "auto" if raw == "on" else raw
    return raw


def gemini_supports_thinking(model: str) -> bool:
    normalized = (model or "").strip().lower()
    return normalized.startswith("gemini-2.5") or normalized.startswith("gemini-3")


def gemini_supports_native_web_search(model: str) -> bool:
    normalized = (model or "").strip().lower()
    return normalized.startswith("gemini-2.0") or normalized.startswith("gemini-2.5") or normalized.startswith("gemini-3")


def gemini_supports_tool_combination(model: str) -> bool:
    normalized = (model or "").strip().lower()
    return normalized.startswith("gemini-3")


def gemini_allowed_thinking_levels(model: str) -> tuple[str, ...]:
    normalized = (model or "").strip().lower()
    if normalized.startswith("gemini-3"):
        if "-pro" in normalized or normalized.startswith(("gemini-3.7-flash", "gemini-3.8-flash")):
            return ("low", "medium", "high")
        return ("minimal", "low", "medium", "high")
    return ()


def gemini_thinking_budget_is_valid(model: str, value: int) -> bool:
    return gemini_supports_thinking(model) and value >= GEMINI_THINKING_BUDGET_MIN


def gemini_thinking_budget_usage(model: str) -> str:
    return "-1|0|positive integer" if gemini_supports_thinking(model) else "unsupported"
