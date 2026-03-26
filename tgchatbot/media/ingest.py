from __future__ import annotations

import base64
import io
import mimetypes
from pathlib import Path

try:
    import av  # type: ignore
except Exception as _av_exc:
    av = None
    _AV_IMPORT_ERROR = _av_exc
else:
    _AV_IMPORT_ERROR = None
from PIL import Image
from telegram import Message

from tgchatbot.config import TelegramConfig
from tgchatbot.domain.models import MessagePart, PartKind
from tgchatbot.storage.artifacts import ArtifactStore


def _extract_keyframes(video_bytes: bytes, num_key_frames: int = 6) -> list[Image.Image]:
    if av is None:
        raise RuntimeError(f'PyAV is required for video or animated-sticker processing: {_AV_IMPORT_ERROR}')
    container = av.open(io.BytesIO(video_bytes))
    stream = container.streams.video[0]
    stream.skip_frame = "NONKEY"
    keyframes: list[tuple[int | None, Image.Image]] = []
    for frame in container.decode(stream):
        keyframes.append((frame.pts, frame.to_image()))
        if len(keyframes) >= 300:
            break
    container.close()
    keyframes.sort(key=lambda item: item[0] or 0)
    if len(keyframes) <= num_key_frames:
        return [img for _, img in keyframes]
    step = len(keyframes) / float(num_key_frames)
    return [keyframes[int(i * step)][1] for i in range(num_key_frames)]


def _image_frames_from_bytes(data: bytes, split_frames: int = 6) -> list[Image.Image]:
    try:
        img = Image.open(io.BytesIO(data))
        total_frames = getattr(img, "n_frames", 1)
        if total_frames <= 1:
            return [img.copy()]
        if total_frames <= split_frames:
            wanted = list(range(total_frames))
        else:
            wanted = [int(i * (total_frames - 1) / (split_frames - 1)) for i in range(split_frames)]
            wanted = sorted(set(wanted))
        frames: list[Image.Image] = []
        for idx in wanted:
            img.seek(idx)
            frames.append(img.copy())
        return frames
    except Exception:
        return _extract_keyframes(data, num_key_frames=split_frames)


def _compress_frame(frame: Image.Image, max_size: int) -> tuple[str, bytes]:
    has_alpha = frame.mode in ("RGBA", "LA") or ("transparency" in frame.info)
    if has_alpha:
        mime = "image/png"
        working = frame.convert("RGBA")
        options = {"format": "PNG", "optimize": True}
    else:
        mime = "image/jpeg"
        working = frame.convert("RGB")
        options = {"format": "JPEG", "quality": 85, "optimize": True}

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
        working = working.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        if mime == "image/jpeg":
            working.convert("RGB").save(buf, format="JPEG", quality=75, optimize=True)
        else:
            working.convert("RGBA").save(buf, format="PNG", optimize=True)
        payload = buf.getvalue()
        if len(payload) <= max_size:
            break
    return mime, payload


def _compress_frame_if_fits(frame: Image.Image, max_size: int) -> tuple[str, bytes] | None:
    mime, payload = _compress_frame(frame, max_size)
    if len(payload) > max_size:
        return None
    return mime, payload


def _save_inline_visual_part(
    *,
    artifact_store: ArtifactStore,
    session_id: str,
    filename: str,
    payload: bytes,
    mime: str,
    detail: str,
) -> MessagePart:
    return MessagePart(
        kind=PartKind.IMAGE,
        mime_type=mime,
        filename=Path(filename).name,
        data_b64=base64.b64encode(payload).decode("utf-8"),
        size_bytes=len(payload),
        detail=detail,
        remote_sync=False,
    )


def _build_visual_preview_parts(
    *,
    raw: bytes,
    mime: str,
    filename: str,
    artifact_store: ArtifactStore,
    session_id: str,
    telegram_config: TelegramConfig,
) -> list[MessagePart]:
    if not (mime.startswith("image/") or mime.startswith("video/")):
        return []
    frames_target = telegram_config.max_visual_file_frames if (mime.startswith("video/") or mime in {"image/gif", "image/webp"}) else 1
    try:
        frames = _image_frames_from_bytes(raw, split_frames=max(1, frames_target))
    except Exception:
        return []
    if not frames:
        return []
    out: list[MessagePart] = []
    stem = Path(filename).stem or "preview"
    for idx, frame in enumerate(frames):
        compressed = _compress_frame_if_fits(frame, telegram_config.max_photo_bytes)
        if compressed is None:
            continue
        encoded_mime, payload = compressed
        out.append(
            _save_inline_visual_part(
                artifact_store=artifact_store,
                session_id=session_id,
                filename=f"{stem}-preview-{idx}.{ 'png' if encoded_mime == 'image/png' else 'jpg'}",
                payload=payload,
                mime=encoded_mime,
                detail="low" if len(frames) > 1 else "auto",
            )
        )
    return out


def _file_part(*, filename: str, mime: str, artifact_path: str, size_bytes: int, kind: PartKind = PartKind.FILE) -> MessagePart:
    return MessagePart(
        kind=kind,
        filename=filename,
        mime_type=mime,
        artifact_path=artifact_path,
        size_bytes=size_bytes,
        remote_sync=True,
    )


async def extract_message_parts(
    message: Message,
    artifact_store: ArtifactStore,
    session_id: str,
    telegram_config: TelegramConfig,
) -> list[MessagePart]:
    parts: list[MessagePart] = []
    text = message.text or message.caption
    if text:
        parts.append(MessagePart(kind=PartKind.TEXT, text=text))

    if message.photo:
        photo = message.photo[-1]
        file = await photo.get_file()
        buf = io.BytesIO()
        await file.download_to_memory(buf)
        image_bytes = buf.getvalue()
        try:
            frame = _image_frames_from_bytes(image_bytes, 1)[0]
        except Exception:
            frame = None
        if frame is not None:
            compressed = _compress_frame_if_fits(frame, telegram_config.max_photo_bytes)
            if compressed is not None:
                mime, payload = compressed
                parts.append(
                    _save_inline_visual_part(
                        artifact_store=artifact_store,
                        session_id=session_id,
                        filename='photo-preview.jpg',
                        payload=payload,
                        mime=mime,
                        detail='auto',
                    )
                )
            else:
                parts.append(MessagePart(kind=PartKind.TEXT, text='[Photo preview omitted: compressed preview still exceeds inline model-context size limit]', origin='auto_note'))
        else:
            parts.append(MessagePart(kind=PartKind.TEXT, text='[Photo preview unavailable for this image format]', origin='auto_note'))

    if message.sticker:
        file = await message.sticker.get_file()
        buf = io.BytesIO()
        await file.download_to_memory(buf)
        sticker_bytes = buf.getvalue()
        is_video = bool(getattr(message.sticker, 'is_video', False))
        is_animated = bool(getattr(message.sticker, 'is_animated', False))
        sticker_kind = 'video' if is_video else 'animated' if is_animated else 'static'
        sticker_emoji = getattr(message.sticker, 'emoji', None)
        hint = f'[User sent {sticker_kind} sticker' + (f' {sticker_emoji}' if sticker_emoji else '') + ']'
        parts.append(MessagePart(kind=PartKind.STICKER, text=hint, remote_sync=False))
        if telegram_config.max_sticker_bytes <= 0 or telegram_config.max_sticker_frames <= 0:
            parts.append(MessagePart(kind=PartKind.TEXT, text='[Sticker preview disabled by config]', origin='auto_note'))
        else:
            try:
                frames = _image_frames_from_bytes(sticker_bytes, split_frames=telegram_config.max_sticker_frames)
            except Exception:
                frames = []
            if frames:
                previews_added = 0
                for idx, frame in enumerate(frames):
                    compressed = _compress_frame_if_fits(frame, telegram_config.max_sticker_bytes)
                    if compressed is None:
                        continue
                    mime, payload = compressed
                    parts.append(
                        _save_inline_visual_part(
                            artifact_store=artifact_store,
                            session_id=session_id,
                            filename=f'sticker-preview-{idx}.{"png" if mime == "image/png" else "jpg"}',
                            payload=payload,
                            mime=mime,
                            detail='low',
                        )
                    )
                    previews_added += 1
                if previews_added == 0:
                    parts.append(MessagePart(kind=PartKind.TEXT, text='[Sticker preview omitted: compressed preview still exceeds inline model-context size limit]', origin='auto_note'))
            else:
                parts.append(MessagePart(kind=PartKind.TEXT, text='[Sticker preview unavailable for this sticker format]', origin='auto_note'))

    if message.animation:
        parts.extend(await _extract_video_like_message_part(message.animation, artifact_store, session_id, telegram_config, fallback_name='animation'))

    if message.video:
        parts.extend(await _extract_video_like_message_part(message.video, artifact_store, session_id, telegram_config, fallback_name='video'))

    if message.document:
        doc = message.document
        file = await doc.get_file()
        if (doc.file_size or 0) > telegram_config.max_document_bytes:
            parts.append(MessagePart(kind=PartKind.TEXT, text=f"[Attached file omitted: {doc.file_name or 'document'} exceeds size limit]", origin='auto_note'))
            return parts
        buf = io.BytesIO()
        await file.download_to_memory(buf)
        raw = buf.getvalue()
        filename = doc.file_name or 'document'
        path = artifact_store.save_bytes(chat_id=session_id, filename=filename, data=raw)
        mime = doc.mime_type or mimetypes.guess_type(filename)[0] or 'application/octet-stream'

        if mime.startswith('image/') or mime.startswith('video/'):
            parts.append(_file_part(filename=filename, mime=mime, artifact_path=str(path), size_bytes=len(raw)))
            parts.extend(
                _build_visual_preview_parts(
                    raw=raw,
                    mime=mime,
                    filename=filename,
                    artifact_store=artifact_store,
                    session_id=session_id,
                    telegram_config=telegram_config,
                )
            )
        else:
            parts.append(_file_part(filename=filename, mime=mime, artifact_path=str(path), size_bytes=len(raw)))
            text_excerpt = _try_text_excerpt(raw, mime, telegram_config.max_inline_text_chars)
            if text_excerpt:
                parts.append(MessagePart(kind=PartKind.TEXT, text=f"[Attached file excerpt: {filename} ({mime})]\n{text_excerpt}", remote_sync=False, origin='auto_note'))
    return parts


async def _extract_video_like_message_part(
    media,
    artifact_store: ArtifactStore,
    session_id: str,
    telegram_config: TelegramConfig,
    fallback_name: str,
) -> list[MessagePart]:
    if telegram_config.max_document_bytes <= 0 or telegram_config.max_visual_file_frames <= 0:
        return [MessagePart(kind=PartKind.TEXT, text=f'[{fallback_name.title()} processing disabled by config]', origin='auto_note')]
    if (media.file_size or 0) > telegram_config.max_document_bytes:
        return [MessagePart(kind=PartKind.TEXT, text=f"[{fallback_name.title()} omitted: file too large for inline model context]", origin='auto_note')]
    file = await media.get_file()
    buf = io.BytesIO()
    await file.download_to_memory(buf)
    raw = buf.getvalue()
    filename = getattr(media, 'file_name', None) or f'{fallback_name}.mp4'
    mime = getattr(media, 'mime_type', None) or mimetypes.guess_type(filename)[0] or 'video/mp4'
    path = artifact_store.save_bytes(chat_id=session_id, filename=filename, data=raw)
    parts = [_file_part(filename=filename, mime=mime, artifact_path=str(path), size_bytes=len(raw))]
    parts.extend(
        _build_visual_preview_parts(
            raw=raw,
            mime=mime,
            filename=filename,
            artifact_store=artifact_store,
            session_id=session_id,
            telegram_config=telegram_config,
        )
    )
    return parts


def _try_text_excerpt(raw: bytes, mime: str, limit: int) -> str | None:
    text_like = mime.startswith('text/') or mime in {'application/json', 'application/xml', 'application/javascript'}
    if not text_like:
        return None
    try:
        text = raw.decode('utf-8', errors='replace')
    except Exception:
        return None
    text = text.strip()
    if not text:
        return None
    if len(text) > limit:
        return text[:limit] + '\n[truncated]'
    return text
