from __future__ import annotations

from pathlib import Path
import base64
import mimetypes
import secrets


class ArtifactStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_bytes(self, *, chat_id: str, filename: str, data: bytes) -> Path:
        safe_chat = chat_id.replace("/", "_")
        chat_dir = self.root / safe_chat
        chat_dir.mkdir(parents=True, exist_ok=True)
        name = Path(filename).name or f"blob-{secrets.token_hex(4)}"
        target = chat_dir / f"{secrets.token_hex(6)}-{name}"
        target.write_bytes(data)
        return target

    def save_base64(self, *, chat_id: str, filename: str, data_b64: str) -> Path:
        return self.save_bytes(chat_id=chat_id, filename=filename, data=base64.b64decode(data_b64))

    @staticmethod
    def guess_mime(path: Path) -> str | None:
        mime, _ = mimetypes.guess_type(path.name)
        return mime
