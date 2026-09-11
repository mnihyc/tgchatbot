"""Disposable compressed vision inputs. References are never permanent media claims."""
from __future__ import annotations

import base64
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
import hashlib
import fcntl
import shutil
import tempfile

from tgchatbot.domain.models import ConversationMessage, MessagePart, PartKind
from tgchatbot.operational import MemoryConfig, from_env


class PreviewCache:
    def __init__(self, root: Path, *, max_bytes: int | None = None) -> None:
        max_bytes = from_env(MemoryConfig, 'MEMORY').preview_cache_bytes if max_bytes is None else max_bytes
        if max_bytes < 0:
            raise ValueError('Preview cache capacity cannot be negative')
        root.mkdir(parents=True, exist_ok=True)
        self._lock = (root / '.preview-cache.lock').open('a')
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise RuntimeError('Another bot owns the temporary preview cache') from None
        # A hard-killed process cannot run TemporaryDirectory cleanup. The lock
        # gives this bot exclusive ownership of this temporary namespace, so
        # crash leftovers cannot multiply the advertised disk bound on restart.
        for stale in root.glob('previews-*'):
            if stale.is_dir() and not stale.is_symlink():
                shutil.rmtree(stale)
        self._directory = tempfile.TemporaryDirectory(prefix='previews-', dir=root)
        self.root = Path(self._directory.name)
        self.max_bytes = max_bytes
        self._entries: OrderedDict[str, int] = OrderedDict()
        self._bytes = 0

    def externalize(self, message: ConversationMessage) -> ConversationMessage:
        parts = []
        for part in message.parts:
            if not part.data_b64:
                parts.append(part)
                continue
            data = base64.b64decode(part.data_b64, validate=True)
            reference = None
            if len(data) <= self.max_bytes:
                reference = hashlib.sha256(data).hexdigest()
                if reference in self._entries:
                    self._entries.move_to_end(reference)
                    parts.append(replace(part, data_b64=None, preview_ref=reference))
                    continue
                while self._bytes + len(data) > self.max_bytes and self._entries:
                    key, size = self._entries.popitem(last=False)
                    (self.root / key).unlink(missing_ok=True)
                    self._bytes -= size
                (self.root / reference).write_bytes(data)
                self._entries[reference] = len(data)
                self._bytes += len(data)
            parts.append(replace(part, data_b64=None, preview_ref=reference))
        return replace(message, parts=parts)

    def materialize(self, message: ConversationMessage, *, vision: bool) -> ConversationMessage:
        parts = []
        for part in message.parts:
            if part.kind not in {PartKind.IMAGE, PartKind.STICKER}:
                parts.append(part)
                continue
            if vision and part.data_b64:
                parts.append(part)
                continue
            if vision and part.preview_ref in self._entries:
                data = (self.root / part.preview_ref).read_bytes()
                self._entries.move_to_end(part.preview_ref)
                parts.append(replace(part, data_b64=base64.b64encode(data).decode('ascii')))
            else:
                reason = 'selected provider has no vision input' if not vision else 'temporary preview expired or unavailable'
                description = part.text or part.detail or part.filename or part.kind.value
                parts.append(MessagePart(kind=PartKind.TEXT, text=f'[{description}; {reason}]', remote_sync=False, origin=part.origin))
        return replace(message, parts=parts)

    def close(self) -> None:
        self._directory.cleanup()
        self._entries.clear()
        self._bytes = 0
        self._lock.close()
