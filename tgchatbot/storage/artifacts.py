from __future__ import annotations

from pathlib import Path
import base64
import mimetypes
import secrets
import threading


class ArtifactStore:
    def __init__(self, root: Path, *, max_bytes: int | None = None) -> None:
        self.root = root
        self.max_bytes = max_bytes
        self._write_lock = threading.Lock()
        if max_bytes is not None and max_bytes < 0:
            raise ValueError('Artifact cache capacity cannot be negative')
        self.root.mkdir(parents=True, exist_ok=True)
        if max_bytes is not None:
            self._make_room(0)

    def _make_room(self, incoming_bytes: int) -> None:
        files = sorted((path for path in self.root.rglob('*') if path.is_file()), key=lambda path: path.stat().st_mtime)
        total = sum(path.stat().st_size for path in files)
        for path in files:
            if total + incoming_bytes <= self.max_bytes:
                break
            size = path.stat().st_size
            path.unlink(missing_ok=True)
            total -= size

    def save_bytes(self, *, chat_id: str, filename: str, data: bytes) -> Path:
        safe_chat = chat_id.replace("/", "_")
        chat_dir = self.root / safe_chat
        chat_dir.mkdir(parents=True, exist_ok=True)
        name = Path(filename).name or f"blob-{secrets.token_hex(4)}"
        # Canonical intake writes replay files in worker threads. Keep eviction
        # and writing together so concurrent chats share this cache allowance.
        with self._write_lock:
            if self.max_bytes is not None:
                if len(data) > self.max_bytes:
                    raise ValueError('Artifact exceeds disposable cache capacity')
                self._make_room(len(data))
            target = chat_dir / f"{secrets.token_hex(6)}-{name}"
            target.write_bytes(data)
            return target

    def save_base64(self, *, chat_id: str, filename: str, data_b64: str) -> Path:
        return self.save_bytes(chat_id=chat_id, filename=filename, data=base64.b64decode(data_b64))

    @staticmethod
    def guess_mime(path: Path) -> str | None:
        mime, _ = mimetypes.guess_type(path.name)
        return mime
