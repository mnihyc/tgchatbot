from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Protocol

from tgchatbot.core.token_estimator import TokenEstimator
from tgchatbot.domain.models import ConversationMessage, ProviderResponse, SessionSettings, ToolCall
from tgchatbot.tools.base import ToolSpec


@dataclass(frozen=True)
class ProviderCapabilities:
    multimodal_input: bool = True
    function_tools: bool = True
    native_web_search: bool = False
    server_state: bool = False
    structured_output: bool = True


@dataclass(frozen=True)
class ControlDescriptor:
    supported: bool
    effective_value: str
    source: str
    note: str | None = None


@dataclass(frozen=True)
class RequestTokenEstimate:
    history_tokens: int = 0
    instructions_tokens: int = 0
    tools_tokens: int = 0
    extra_input_tokens: int = 0
    framing_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def compose(
        cls,
        *,
        history_tokens: int = 0,
        instructions_tokens: int = 0,
        tools_tokens: int = 0,
        extra_input_tokens: int = 0,
        framing_tokens: int = 0,
    ) -> "RequestTokenEstimate":
        total_tokens = (
            int(history_tokens)
            + int(instructions_tokens)
            + int(tools_tokens)
            + int(extra_input_tokens)
            + int(framing_tokens)
        )
        return cls(
            history_tokens=int(history_tokens),
            instructions_tokens=int(instructions_tokens),
            tools_tokens=int(tools_tokens),
            extra_input_tokens=int(extra_input_tokens),
            framing_tokens=int(framing_tokens),
            total_tokens=total_tokens,
        )

    def scaled(self, multiplier: float) -> "RequestTokenEstimate":
        factor = max(1.0, float(multiplier))

        def scale(value: int) -> int:
            return max(0, int(round(float(value) * factor)))

        return RequestTokenEstimate.compose(
            history_tokens=scale(self.history_tokens),
            instructions_tokens=scale(self.instructions_tokens),
            tools_tokens=scale(self.tools_tokens),
            extra_input_tokens=scale(self.extra_input_tokens),
            framing_tokens=scale(self.framing_tokens),
        )


def estimate_json_schema_tokens(schema: dict[str, Any] | None, *, name: str | None = None) -> int:
    if not schema:
        return 0
    payload = json.dumps(schema, ensure_ascii=False, sort_keys=True)
    total = TokenEstimator.estimate_text(payload) + 12
    if name:
        total += TokenEstimator.estimate_text(name)
    return total


class ModelProvider(Protocol):
    name: str
    capabilities: ProviderCapabilities

    async def generate(
        self,
        *,
        settings: SessionSettings,
        messages: list[ConversationMessage],
        instructions: str,
        tools: list[ToolSpec],
        extra_input_items: list[dict] | None = None,
        response_schema: dict[str, Any] | None = None,
        response_schema_name: str | None = None,
    ) -> ProviderResponse:
        ...

    def make_tool_result_items(self, tool_call: ToolCall, tool_output: dict) -> list[dict]:
        ...

    def describe_controls(self, settings: SessionSettings) -> dict[str, ControlDescriptor]:
        ...

    def persistent_history_items(self, response: ProviderResponse) -> list[dict]:
        ...

    def estimate_request_tokens(
        self,
        *,
        settings: SessionSettings,
        messages: list[ConversationMessage],
        instructions: str,
        tools: list[ToolSpec],
        extra_input_items: list[dict] | None = None,
        response_schema: dict[str, Any] | None = None,
        response_schema_name: str | None = None,
        history_tokens_override: int | None = None,
    ) -> RequestTokenEstimate:
        ...
