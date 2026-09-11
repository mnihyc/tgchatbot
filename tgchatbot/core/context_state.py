from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
from tgchatbot.domain.models import ConversationMessage, MessageRole, PartKind

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
    last_message_id: int = 0
    loaded: bool = False
    provider_history_cache: list[ConversationMessage] = field(default_factory=list)
    provider_history_cache_key: tuple[str, str, str, tuple[int, ...], int] | None = None
    provider_history_token_cache: dict[tuple[str, str, str, tuple[int, ...], int], int] = field(default_factory=dict)
    provider_history_dirty: bool = True

    def active_participant_ids(self, trigger: ConversationMessage | None = None) -> list[str]:
        """Use retained speakers and direct reply identities, without archive reads."""
        participants: dict[str, None] = {}
        reply_participants: dict[str, None] = {}

        def known_actor(metadata: dict[str, Any]) -> str | None:
            actor = metadata.get('actor_id')
            if metadata.get('actor_kind') in {'bot', 'unknown'} or not isinstance(actor, str):
                return None
            actor = actor.strip()
            return actor if actor and actor not in {'unknown', 'agent'} else None

        def collect(message: ConversationMessage) -> None:
            metadata = message.metadata
            if message.role != MessageRole.USER or metadata.get('synthetic_role'):
                return
            actor = known_actor(metadata)
            if actor is not None:
                participants.setdefault(actor, None)
            reply_actor = metadata.get('reply_to_actor')
            source_chat = metadata.get('source_chat_id')
            if (isinstance(reply_actor, dict) and metadata.get('reply_to_source_id') is not None
                    and source_chat not in (None, '', 'unknown')
                    and str(metadata.get('reply_to_source_chat_id')) == str(source_chat)):
                actor = known_actor(reply_actor)
                if actor is not None:
                    reply_participants.setdefault(actor, None)

        if trigger is not None:
            collect(trigger)
        for item in reversed(self.raw_messages):
            collect(item.message)
        for actor in reply_participants:
            participants.setdefault(actor, None)
        return list(participants)

    def rebuild_estimate(self) -> int:
        self.estimated_tokens = sum(block.estimated_tokens for block in self.blocks) + sum(item.estimated_tokens for item in self.raw_messages)
        self.estimated_images = sum(item.image_count for item in self.raw_messages)
        self.last_message_id = max(max((item.db_id for item in self.raw_messages), default=0),
            max((block.end_message_id or 0 for block in self.blocks), default=0))
        self.provider_history_dirty = True
        self.provider_history_cache_key = None
        self.provider_history_token_cache.clear()
        return self.estimated_tokens
