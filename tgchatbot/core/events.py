from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EventKind = Literal["phase", "thinking", "tool_call", "tool_result", "sticker", "assistant_text", "final"]


@dataclass(slots=True)
class RuntimeEvent:
    kind: EventKind
    title: str
    detail: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
