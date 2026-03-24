from __future__ import annotations

import math

from tgchatbot.domain.models import ConversationMessage, MessagePart, PartKind


class TokenEstimator:
    """Fast approximate token estimator.

    This intentionally avoids model-specific tokenization overhead on every turn.
    The estimate is conservative enough for compaction triggers and cheap enough to
    update incrementally in memory.
    """

    TEXT_CHARS_PER_TOKEN = 4.0
    MESSAGE_OVERHEAD = 12
    PART_OVERHEAD = 6
    IMAGE_TOKENS = 900
    STICKER_TOKENS = 200
    FILE_BASE_TOKENS = 48
    TOOL_METADATA_TOKENS = 24

    @classmethod
    def estimate_text(cls, text: str | None) -> int:
        if not text:
            return 0
        return max(1, math.ceil(len(text) / cls.TEXT_CHARS_PER_TOKEN))

    @classmethod
    def estimate_part(cls, part: MessagePart) -> int:
        total = cls.PART_OVERHEAD
        if part.kind == PartKind.TEXT:
            total += cls.estimate_text(part.text)
        elif part.kind == PartKind.IMAGE:
            total += cls.IMAGE_TOKENS + cls.estimate_text(part.filename)
        elif part.kind == PartKind.STICKER:
            total += cls.STICKER_TOKENS + cls.estimate_text(part.filename)
        elif part.kind == PartKind.FILE:
            descriptor = ' '.join(filter(None, [part.filename, part.mime_type, str(part.size_bytes) if part.size_bytes is not None else None]))
            total += cls.FILE_BASE_TOKENS + cls.estimate_text(descriptor)
        else:
            total += cls.estimate_text(part.text)
        return total

    @classmethod
    def estimate_message(cls, message: ConversationMessage) -> int:
        total = cls.MESSAGE_OVERHEAD + cls.estimate_text(message.name)
        for part in message.parts:
            total += cls.estimate_part(part)
        if message.metadata:
            total += cls.TOOL_METADATA_TOKENS
        return total
