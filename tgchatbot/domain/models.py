from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class ChatMode(str, Enum):
    CHAT = "chat"
    ASSIST = "assist"
    AGENT = "agent"


class ProcessVisibility(str, Enum):
    OFF = "off"
    MINIMAL = "minimal"
    STATUS = "status"
    VERBOSE = "verbose"
    FULL = "full"


class ResponseDelivery(str, Enum):
    EDIT = "edit"
    FINAL_NEW = "final_new"


class StickerMode(str, Enum):
    OFF = "off"
    AUTO = "auto"


class StickerTiming(str, Enum):
    SEND_NOW = "send_now"
    BEFORE_FINAL = "send_now"
    AFTER_FINAL = "after_final"

    @classmethod
    def parse(cls, value: str | None) -> "StickerTiming":
        normalized = str(value or 'after_final').strip().lower()
        if normalized == 'before_final':
            normalized = 'send_now'
        return cls(normalized)


class PromptInjectionMode(str, Enum):
    AUGMENT = "augment"
    EXACT = "exact"


class ToolHistoryMode(str, Enum):
    TRANSLATED = "translated"
    NATIVE_SAME_PROVIDER = "native_same_provider"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class PartKind(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    STICKER = "sticker"


def default_system_prompt() -> str:
    return (
        "You are a concise, reliable assistant inside a messaging app. Preserve continuity in long-running chats, "
        "use tools only when they materially help, and remain aware of attached images, stickers, and files."
    )


@dataclass(slots=True)
class MessagePart:
    kind: PartKind
    text: str | None = None
    mime_type: str | None = None
    filename: str | None = None
    data_b64: str | None = None
    artifact_path: str | None = None
    size_bytes: int | None = None
    detail: str | None = None
    remote_sync: bool = True
    origin: str | None = None


@dataclass(slots=True)
class ConversationMessage:
    role: MessageRole
    parts: list[MessagePart]
    name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def text(cls, role: MessageRole, text: str, *, name: str | None = None, metadata: dict[str, Any] | None = None) -> "ConversationMessage":
        return cls(role=role, parts=[MessagePart(kind=PartKind.TEXT, text=text)], name=name, metadata=metadata or {})

    @classmethod
    def assistant_text(cls, text: str, *, metadata: dict[str, Any] | None = None) -> "ConversationMessage":
        return cls.text(MessageRole.ASSISTANT, text, metadata=metadata)

    @classmethod
    def user_text(cls, text: str, *, metadata: dict[str, Any] | None = None) -> "ConversationMessage":
        return cls.text(MessageRole.USER, text, metadata=metadata)


@dataclass(slots=True)
class SessionSettings:
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    mode: ChatMode = ChatMode.CHAT
    process_visibility: ProcessVisibility = ProcessVisibility.STATUS
    response_delivery: ResponseDelivery = ResponseDelivery.EDIT
    sticker_mode: StickerMode = StickerMode.OFF
    prompt_injection_mode: PromptInjectionMode = PromptInjectionMode.AUGMENT
    tool_history_mode: ToolHistoryMode = ToolHistoryMode.TRANSLATED
    reasoning_effort: str | None = None
    reasoning_summary: str | None = None
    text_verbosity: str | None = None
    include_thoughts: bool | None = None
    thinking_budget: int | None = None
    thinking_level: str | None = None
    native_web_search_mode: str = "default"
    native_web_search_max: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    link_prefetch_mode: str = "default"
    max_output_tokens: int | None = None
    max_input_images: int | None = None
    compact_target_images: int | None = None
    compact_trigger_tokens: int | None = None
    compact_target_tokens: int | None = None
    compact_batch_tokens: int | None = None
    compact_keep_recent_ratio: float | None = None
    compact_tool_ratio_threshold: float | None = None
    compact_tool_min_tokens: int | None = None
    compact_min_messages: int | None = None
    min_raw_messages_reserve: int | None = None
    max_interaction_rounds: int | None = None
    spontaneous_reply_chance: int | None = None
    spontaneous_reply_idle_s: int | None = None
    provider_retry_count: int | None = None
    private_reply_delay_s: float | None = None
    group_reply_delay_s: float | None = None
    group_spontaneous_reply_delay_s: float | None = None
    reply_delay_s: float | None = None
    metadata_injection_mode: str = "on"
    metadata_timezone: str = "UTC"
    system_prompt: str = field(default_factory=default_system_prompt)


@dataclass(slots=True)
class ToolCall:
    name: str
    call_id: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class OutboundArtifact:
    path: Path
    filename: str
    mime_type: str | None = None
    caption: str | None = None


@dataclass(slots=True)
class OutboundSticker:
    path: Path
    emoji: str | None = None
    timing: StickerTiming = StickerTiming.AFTER_FINAL
    label: str | None = None
    source_id: str | None = None
    delivery_state: str | None = None
    telegram_message_id: int | None = None
    error: str | None = None

    def display_reference(self) -> str:
        return str(self.source_id or self.label or self.path.stem or 'sticker').strip() or 'sticker'

    def delivery_receipt(self) -> dict[str, Any]:
        return {
            'sticker_id': self.source_id or self.path.stem,
            'sticker_label': self.label,
            'delivery_timing': self.timing.value,
            'emoji': self.emoji,
            'delivery_state': 'failed',
            'sent': False,
        }


@dataclass(slots=True)
class ToolResult:
    call_id: str
    name: str
    output: dict[str, Any]
    artifacts: list[OutboundArtifact] = field(default_factory=list)
    stickers: list[OutboundSticker] = field(default_factory=list)


@dataclass(slots=True)
class UsageInfo:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(slots=True)
class ProviderResponse:
    final_text: str = ""
    reasoning_summaries: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    native_tool_calls: list[dict] = field(default_factory=list)
    continuation_items: list[dict[str, Any]] = field(default_factory=list)
    usage: UsageInfo = field(default_factory=UsageInfo)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TurnResult:
    text: str
    artifacts: list[OutboundArtifact] = field(default_factory=list)
    usage: UsageInfo = field(default_factory=UsageInfo)
    stickers: list[OutboundSticker] = field(default_factory=list)
    provider_name: str | None = None
    provider_history_items: list[dict[str, Any]] = field(default_factory=list)
