from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from tgchatbot.domain.models import ConversationMessage, PartKind

@dataclass(slots=True)
class StoredConversationMessage:
    db_id: int
    message: ConversationMessage
    estimated_tokens: int
    created_at: int | None = None
    @property
    def image_count(self) -> int:
        return sum(1 for part in self.message.parts if part.kind == PartKind.IMAGE)

@dataclass(slots=True)
class MemoryBlock:
    block_id: int
    sequence_no: int
    summary_text: str
    estimated_tokens: int
    source_message_count: int
    start_message_id: int | None = None
    end_message_id: int | None = None
    level: int = 1
    kind: str = "episode"
    lifecycle: str = "sealed"
    source_kind: str = "raw"
    parent_block_ids: tuple[int, ...] = ()
    topic_labels: tuple[str, ...] = ()
    actor_labels: tuple[str, ...] = ()
    time_start: str | None = None
    time_end: str | None = None
    retained_raw_excerpt_count: int = 0
    validator_status: str | None = None
    validator_score: float | None = None
    structured_data: dict[str, Any] = field(default_factory=dict)
    def render_as_message(self) -> ConversationMessage:
        scope_bits: list[str] = []
        if self.time_start or self.time_end:
            if self.time_start and self.time_end and self.time_start != self.time_end:
                scope_bits.append(f"time={self.time_start}..{self.time_end}")
            else:
                scope_bits.append(f"time={self.time_start or self.time_end}")
        if self.actor_labels:
            scope_bits.append("actors=" + ", ".join(self.actor_labels[:4]))
        if self.topic_labels:
            scope_bits.append("topics=" + ", ".join(self.topic_labels[:4]))
        header = f"[Memory {self.kind} block L{self.level} #{self.sequence_no}; covers {self.source_message_count} earlier messages"
        if scope_bits:
            header += "; " + "; ".join(scope_bits)
        header += "]"
        return ConversationMessage.assistant_text(header + "\n" + self.summary_text)

@dataclass(slots=True)
class LiveConversationState:
    session_id: str
    blocks: list[MemoryBlock] = field(default_factory=list)
    raw_messages: list[StoredConversationMessage] = field(default_factory=list)
    estimated_tokens: int = 0
    estimated_images: int = 0
    loaded: bool = False
    provider_history_cache: list[ConversationMessage] = field(default_factory=list)
    provider_history_cache_key: tuple[str, str, str, tuple[int, ...], int] | None = None
    provider_history_token_cache: dict[tuple[str, str, str, tuple[int, ...], int], int] = field(default_factory=dict)
    provider_history_dirty: bool = True
    def rebuild_estimate(self) -> int:
        self.estimated_tokens = sum(block.estimated_tokens for block in self.blocks) + sum(item.estimated_tokens for item in self.raw_messages)
        self.estimated_images = sum(item.image_count for item in self.raw_messages)
        self.provider_history_dirty = True
        self.provider_history_cache_key = None
        self.provider_history_token_cache.clear()
        return self.estimated_tokens
