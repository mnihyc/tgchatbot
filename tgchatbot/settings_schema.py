from __future__ import annotations

from typing import Iterable


REASONING_EFFORT_VALUES = frozenset({"none", "minimal", "low", "medium", "high", "xhigh"})
REASONING_SUMMARY_VALUES = frozenset({"off", "on", "auto", "detailed", "concise"})
TEXT_VERBOSITY_VALUES = frozenset({"low", "medium", "high"})
GEMINI_THINKING_LEVEL_VALUES = frozenset({"minimal", "low", "medium", "high"})
GEMINI_THINKING_BUDGET_MIN = -1
GEMINI_THINKING_BUDGET_MAX = 32768

NATIVE_WEB_SEARCH_MAX_MIN = 0
NATIVE_WEB_SEARCH_MAX_MAX = 100

TEMPERATURE_MIN = 0.0
TEMPERATURE_MAX = 2.0
TOP_P_MIN = 0.0
TOP_P_MAX = 1.0
TOP_K_MIN = 1
TOP_K_MAX = 1000

MAX_OUTPUT_TOKENS_MIN = 64
MAX_OUTPUT_TOKENS_MAX = 65536

IMAGE_LIMIT_DISABLED = 0
IMAGE_LIMIT_MAX = 100000

COMPACT_TOKEN_MIN = 256
COMPACT_TOKEN_MAX = 10000000
COMPACT_KEEP_RECENT_RATIO_MIN = 0.0
COMPACT_KEEP_RECENT_RATIO_MAX = 0.95
COMPACT_TOOL_RATIO_THRESHOLD_MIN = 1.0
COMPACT_TOOL_RATIO_THRESHOLD_MAX = 100.0
COMPACT_MIN_MESSAGES_MIN = 2
COMPACT_MIN_MESSAGES_MAX = 1000
MIN_RAW_MESSAGES_RESERVE_MIN = 0
MIN_RAW_MESSAGES_RESERVE_MAX = 1000

MAX_INTERACTION_ROUNDS_MIN = 1
MAX_INTERACTION_ROUNDS_MAX = 64
SPONTANEOUS_REPLY_CHANCE_MIN = 0
SPONTANEOUS_REPLY_CHANCE_MAX = 100
GROUP_SPONTANEOUS_REPLY_DELAY_MAX_S = 86400.0
REPLY_DELAY_MAX_S = 600.0
PROVIDER_RETRY_COUNT_MIN = 0
PROVIDER_RETRY_COUNT_MAX = 5


def normalize_choice(value: str | None, default: str, allowed: Iterable[str]) -> str:
    allowed_set = {item.strip().lower() for item in allowed}
    selected = (value or "").strip().lower()
    return selected if selected in allowed_set else default


def normalize_optional_choice(value: str | None, allowed: Iterable[str]) -> str | None:
    allowed_set = {item.strip().lower() for item in allowed}
    selected = (value or "").strip().lower()
    return selected if selected in allowed_set else None


def parse_bounded_int_env(value: str | None, *, default: int, minimum: int, maximum: int) -> int:
    raw = (value or "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    if parsed < minimum or parsed > maximum:
        return default
    return parsed


def parse_optional_bounded_int_env(value: str | None, *, minimum: int, maximum: int) -> int | None:
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        parsed = int(raw)
    except ValueError:
        return None
    if parsed < minimum or parsed > maximum:
        return None
    return parsed


def parse_bounded_float_env(value: str | None, *, default: float, minimum: float, maximum: float) -> float:
    raw = (value or "").strip()
    if not raw:
        return default
    try:
        parsed = float(raw)
    except ValueError:
        return default
    if parsed < minimum or parsed > maximum:
        return default
    return parsed


def parse_optional_disabled_int_env(value: str | None, *, default: int, maximum: int) -> int:
    raw = (value or "").strip()
    if not raw:
        return default
    try:
        parsed = int(raw)
    except ValueError:
        return default
    if parsed == 0:
        return 0
    if parsed < 0 or parsed > maximum:
        return default
    return parsed


def clamp_int(value: int | None, *, minimum: int, maximum: int, default: int) -> int:
    if value is None:
        return default
    return min(maximum, max(minimum, int(value)))


def clamp_float(value: float | None, *, minimum: float, maximum: float, default: float) -> float:
    if value is None:
        return default
    return min(maximum, max(minimum, float(value)))


def normalize_optional_disabled_int(value: int | None, *, maximum: int) -> int | None:
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
    return min(maximum, parsed)


def normalize_optional_bounded_int(value: int | None, *, minimum: int, maximum: int) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    if parsed < minimum or parsed > maximum:
        return None
    return parsed


def effective_optional_disabled_int(override: int | None, default: int, *, maximum: int) -> int | None:
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
        if "-pro" in normalized:
            return ("low", "medium", "high")
        return ("minimal", "low", "medium", "high")
    return ()


def gemini_thinking_budget_is_valid(model: str, value: int) -> bool:
    normalized = (model or "").strip().lower()
    if normalized.startswith("gemini-2.5"):
        if "flash-lite" in normalized:
            return value in {-1, 0} or 512 <= value <= 24576
        if "pro" in normalized:
            return value == -1 or 128 <= value <= 32768
        return value == -1 or 0 <= value <= 24576
    if normalized.startswith("gemini-3"):
        return GEMINI_THINKING_BUDGET_MIN <= value <= GEMINI_THINKING_BUDGET_MAX
    return False


def gemini_thinking_budget_usage(model: str) -> str:
    normalized = (model or "").strip().lower()
    if normalized.startswith("gemini-2.5"):
        if "flash-lite" in normalized:
            return "-1|0|512..24576"
        if "pro" in normalized:
            return "-1|128..32768"
        return "-1|0..24576"
    if normalized.startswith("gemini-3"):
        return f"{GEMINI_THINKING_BUDGET_MIN}..{GEMINI_THINKING_BUDGET_MAX}"
    return "unsupported"
