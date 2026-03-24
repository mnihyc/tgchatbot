from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

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
