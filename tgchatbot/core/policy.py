from __future__ import annotations

from dataclasses import dataclass

from tgchatbot.domain.models import ChatMode


@dataclass(frozen=True)
class RuntimePolicy:
    allow_tools: bool
    allow_python_exec: bool
    show_intermediate_events: bool


def policy_for_mode(mode: ChatMode) -> RuntimePolicy:
    if mode == ChatMode.CHAT:
        return RuntimePolicy(
            allow_tools=False,
            allow_python_exec=False,
            show_intermediate_events=False,
        )
    if mode == ChatMode.ASSIST:
        return RuntimePolicy(
            allow_tools=True,
            allow_python_exec=True,
            show_intermediate_events=True,
        )
    return RuntimePolicy(
        allow_tools=True,
        allow_python_exec=True,
        show_intermediate_events=True,
    )
