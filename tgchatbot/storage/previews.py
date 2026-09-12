"""Disposable read cache for compressed previews owned by PostgreSQL."""
from __future__ import annotations

import base64
from collections import OrderedDict
from dataclasses import replace

from tgchatbot.domain.models import ConversationMessage, MessagePart, PartKind
from tgchatbot.operational import MemoryConfig, from_env


class PreviewCache:
    def __init__(self, store, *, max_bytes: int | None = None) -> None:
        self.store = store
        self.max_bytes = from_env(MemoryConfig, 'MEMORY').preview_cache_bytes if max_bytes is None else max_bytes
        if self.max_bytes < 0:
            raise ValueError('Preview read cache capacity cannot be negative')
        self._entries: OrderedDict[tuple[str, str], bytes] = OrderedDict()
        self._bytes = 0

    def _remember(self, key: tuple[str, str], data: bytes) -> None:
        if key in self._entries:
            self._entries.move_to_end(key)
            return
        if len(data) > self.max_bytes:
            return
        while self._entries and self._bytes + len(data) > self.max_bytes:
            _, removed = self._entries.popitem(last=False)
            self._bytes -= len(removed)
        self._entries[key] = data
        self._bytes += len(data)

    async def materialize_many(self, session_id: str, messages: list[ConversationMessage], *, vision: bool) -> list[ConversationMessage]:
        references = {part.preview_ref for message in messages for part in message.parts
                      if vision and part.preview_ref and not part.data_b64}
        payloads = {}
        for reference in references:
            key = (session_id, reference)
            if key in self._entries:
                self._entries.move_to_end(key)
                payloads[reference] = self._entries[key]
        missing = references - payloads.keys()
        if missing:
            loaded = await self.store.load_preview_data(session_id, list(missing))
            payloads.update(loaded)
            for reference, data in loaded.items():
                self._remember((session_id, reference), data)
        return [self._materialize(message, payloads, vision=vision) for message in messages]

    @staticmethod
    def _materialize(message: ConversationMessage, payloads: dict[str, bytes], *, vision: bool) -> ConversationMessage:
        parts = []
        for part in message.parts:
            if part.kind == PartKind.STICKER and not part.data_b64 and not part.preview_ref:
                parts.append(part)
                continue
            if part.kind not in {PartKind.IMAGE, PartKind.STICKER}:
                parts.append(part)
                continue
            if vision and part.data_b64:
                parts.append(part)
                continue
            data = payloads.get(part.preview_ref) if vision else None
            if data is not None:
                parts.append(replace(part, data_b64=base64.b64encode(data).decode('ascii')))
            else:
                reason = 'selected provider has no vision input' if not vision else 'preview unavailable'
                description = part.text or part.detail or part.filename or part.kind.value
                parts.append(MessagePart(kind=PartKind.TEXT, text=f'[{description}; {reason}]', remote_sync=False, origin=part.origin))
        return replace(message, parts=parts)

    def close(self) -> None:
        self._entries.clear()
        self._bytes = 0

    def forget_session(self, session_id: str) -> None:
        for key in [key for key in self._entries if key[0] == session_id]:
            self._bytes -= len(self._entries.pop(key))
