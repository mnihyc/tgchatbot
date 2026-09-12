"""Shared image encoding, also executed by the remote file reader."""
from __future__ import annotations

import io


def compress_frame(frame, max_size: int) -> tuple[str, bytes]:
    from PIL import Image

    has_alpha = frame.mode in ("RGBA", "LA") or ("transparency" in frame.info)
    if has_alpha:
        mime = "image/png"
        working = frame.convert("RGBA")
        options = {"format": "PNG", "optimize": True}
    else:
        mime = "image/jpeg"
        working = frame.convert("RGB")
        options = {"format": "JPEG", "quality": 85, "optimize": True}

    try:
        buf = io.BytesIO()
        working.save(buf, **options)
        payload = buf.getvalue()
        if len(payload) <= max_size:
            return mime, payload

        for _ in range(8):
            new_w = max(1, int(working.width * 0.8))
            new_h = max(1, int(working.height * 0.8))
            if new_w == working.width and new_h == working.height:
                break
            resized = working.resize((new_w, new_h), Image.LANCZOS)
            working.close()
            working = resized
            buf = io.BytesIO()
            if mime == "image/jpeg":
                working.save(buf, format="JPEG", quality=75, optimize=True)
            else:
                working.save(buf, format="PNG", optimize=True)
            payload = buf.getvalue()
            if len(payload) <= max_size:
                break
        return mime, payload
    finally:
        working.close()
