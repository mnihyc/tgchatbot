from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

EventKind = Literal["phase", "tool_call", "tool_result", "sticker", "final"]


@dataclass(slots=True)
class RuntimeEvent:
    kind: EventKind
    title: str
    detail: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
