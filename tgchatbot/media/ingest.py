from __future__ import annotations

import base64
from contextlib import ExitStack, closing
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
from PIL import Image, UnidentifiedImageError
from telegram import Message

from tgchatbot.config import TelegramConfig
from tgchatbot.domain.models import MessagePart, PartKind
from tgchatbot.storage.artifacts import ArtifactStore


def _extract_keyframes(video_bytes: bytes, num_key_frames: int, *, max_candidates: int) -> list[Image.Image]:
    if num_key_frames <= 0 or max_candidates <= 0:
        return []
    if av is None:
        raise RuntimeError(f'PyAV is required for video or animated-sticker processing: {_AV_IMPORT_ERROR}')
    # Preserve the original chronological sample of the first candidate frames,
    # retaining only metadata instead of every full-resolution decoded image.
    keyframes: list[tuple[int | None, int]] = []
    with av.open(io.BytesIO(video_bytes)) as container:
        if not container.streams.video:
            return []
        stream = container.streams.video[0]
        stream.skip_frame = "NONKEY"
        for index, frame in enumerate(container.decode(stream)):
            keyframes.append((frame.pts, index))
            if len(keyframes) >= max_candidates:
                break
    if not keyframes:
        return []
    keyframes.sort(key=lambda item: item[0] or 0)
    if len(keyframes) > num_key_frames:
        step = len(keyframes) / float(num_key_frames)
        keyframes = [keyframes[int(i * step)] for i in range(num_key_frames)]
    wanted = {index for _, index in keyframes}
    last_index = max(wanted)
    images: dict[int, Image.Image] = {}
    try:
        with av.open(io.BytesIO(video_bytes)) as container:
            stream = container.streams.video[0]
            stream.skip_frame = "NONKEY"
            for index, frame in enumerate(container.decode(stream)):
                if index in wanted:
                    images[index] = frame.to_image()
                if index >= last_index:
                    break
        return [images[index] for _, index in keyframes if index in images]
    except Exception:
        for image in images.values():
            image.close()
        raise


def _image_frames_from_bytes(data: bytes, split_frames: int, *, max_keyframe_candidates: int) -> list[Image.Image]:
    if split_frames <= 0:
        return []
    frames: list[Image.Image] = []
    try:
        with closing(Image.open(io.BytesIO(data))) as img:
            total_frames = getattr(img, "n_frames", 1)
            if total_frames <= 1 or split_frames == 1:
                return [img.copy()]
            if total_frames <= split_frames:
                wanted = list(range(total_frames))
            else:
                wanted = [int(i * (total_frames - 1) / (split_frames - 1)) for i in range(split_frames)]
                wanted = sorted(set(wanted))
            for idx in wanted:
                img.seek(idx)
                frames.append(img.copy())
        return frames
    except (UnidentifiedImageError, OSError, EOFError):
        for frame in frames:
            frame.close()
        return _extract_keyframes(data, num_key_frames=split_frames, max_candidates=max_keyframe_candidates)


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


def _compress_frame_if_fits(frame: Image.Image, max_size: int) -> tuple[str, bytes] | None:
    try:
        mime, payload = _compress_frame(frame, max_size)
    except (OSError, ValueError):
        return None
    if len(payload) > max_size:
        return None
    return mime, payload


def _inline_visual_part(
    *,
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
    telegram_config: TelegramConfig,
) -> list[MessagePart]:
    if telegram_config.max_visual_file_frames <= 0 or not (mime.startswith("image/") or mime.startswith("video/")):
        return []
    frames_target = telegram_config.max_visual_file_frames if (mime.startswith("video/") or mime in {"image/gif", "image/webp"}) else 1
    return visual_parts_from_bytes(raw=raw, filename=filename, max_frames=frames_target,
        max_bytes=telegram_config.max_photo_bytes,
        max_keyframe_candidates=telegram_config.max_video_keyframe_candidates)


def visual_parts_from_bytes(*, raw: bytes, filename: str, max_frames: int, max_bytes: int,
                            max_keyframe_candidates: int, detail: str | None = None) -> list[MessagePart]:
    """Decode/compress local or transported bytes through the same visual parser."""
    if max_frames <= 0 or max_bytes <= 0:
        return []
    try:
        frames = _image_frames_from_bytes(raw, split_frames=max_frames,
            max_keyframe_candidates=max_keyframe_candidates)
    except Exception:
        return []
    if not frames:
        return []
    out: list[MessagePart] = []
    stem = Path(filename).stem or "preview"
    with ExitStack() as decoded:
        for frame in frames:
            decoded.callback(frame.close)
        for idx, frame in enumerate(frames):
            compressed = _compress_frame_if_fits(frame, max_bytes)
            if compressed is None:
                continue
            encoded_mime, payload = compressed
            out.append(
                _inline_visual_part(
                    filename=f"{stem}-preview-{idx}.{ 'png' if encoded_mime == 'image/png' else 'jpg'}",
                    payload=payload,
                    mime=encoded_mime,
                    detail=detail or ("low" if len(frames) > 1 else "auto"),
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
            frame = _image_frames_from_bytes(image_bytes, 1,
                max_keyframe_candidates=telegram_config.max_video_keyframe_candidates)[0]
        except Exception:
            frame = None
        if frame is not None:
            with closing(frame):
                compressed = _compress_frame_if_fits(frame, telegram_config.max_photo_bytes)
            if compressed is not None:
                mime, payload = compressed
                parts.append(
                    _inline_visual_part(
                        filename='photo-preview.jpg',
                        payload=payload,
                        mime=mime,
                        detail='auto',
                    )
                )
            else:
                parts.append(MessagePart(kind=PartKind.TEXT, text='[Photo preview omitted: could not encode within inline model-context size limit]', origin='auto_note'))
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
                frames = _image_frames_from_bytes(sticker_bytes, split_frames=telegram_config.max_sticker_frames,
                    max_keyframe_candidates=telegram_config.max_video_keyframe_candidates)
            except Exception:
                frames = []
            if frames:
                previews_added = 0
                with ExitStack() as decoded:
                    for frame in frames:
                        decoded.callback(frame.close)
                    for idx, frame in enumerate(frames):
                        compressed = _compress_frame_if_fits(frame, telegram_config.max_sticker_bytes)
                        if compressed is None:
                            continue
                        mime, payload = compressed
                        parts.append(
                            _inline_visual_part(
                                filename=f'sticker-preview-{idx}.{"png" if mime == "image/png" else "jpg"}',
                                payload=payload,
                                mime=mime,
                                detail='low',
                            )
                        )
                        previews_added += 1
                if previews_added == 0:
                    parts.append(MessagePart(kind=PartKind.TEXT, text='[Sticker preview omitted: could not encode within inline model-context size limit]', origin='auto_note'))
            else:
                parts.append(MessagePart(kind=PartKind.TEXT, text='[Sticker preview unavailable for this sticker format]', origin='auto_note'))

    for field, fallback_name, fallback_mime, visual in (
        ('animation', 'animation.mp4', 'video/mp4', True),
        ('video', 'video.mp4', 'video/mp4', True),
        ('video_note', 'video-note.mp4', 'video/mp4', False),
        ('audio', 'audio.mp3', 'audio/mpeg', False),
        ('voice', 'voice.ogg', 'audio/ogg', False),
        ('document', 'document', 'application/octet-stream', True),
    ):
        media = getattr(message, field, None)
        if media is not None:
            parts.extend(await _extract_file_message_parts(media, artifact_store, session_id, telegram_config,
                fallback_name=fallback_name, fallback_mime=fallback_mime, visual=visual, excerpt=field == 'document'))
    return parts


async def _extract_file_message_parts(
    media,
    artifact_store: ArtifactStore,
    session_id: str,
    telegram_config: TelegramConfig,
    *,
    fallback_name: str,
    fallback_mime: str,
    visual: bool,
    excerpt: bool,
) -> list[MessagePart]:
    filename = getattr(media, 'file_name', None) or fallback_name
    if telegram_config.max_document_bytes <= 0:
        return [MessagePart(kind=PartKind.TEXT, text='[Attached file processing disabled by config]', origin='auto_note')]
    if (media.file_size or 0) > telegram_config.max_document_bytes:
        return [MessagePart(kind=PartKind.TEXT, text=f'[Attached file omitted: {filename} exceeds size limit]', origin='auto_note')]
    file = await media.get_file()
    buf = io.BytesIO()
    await file.download_to_memory(buf)
    raw = buf.getvalue()
    mime = getattr(media, 'mime_type', None) or mimetypes.guess_type(filename)[0] or fallback_mime
    path = artifact_store.save_bytes(chat_id=session_id, filename=filename, data=raw)
    parts = [_file_part(filename=filename, mime=mime, artifact_path=str(path), size_bytes=len(raw))]
    if visual:
        parts.extend(_build_visual_preview_parts(raw=raw, mime=mime, filename=filename, telegram_config=telegram_config))
    if excerpt:
        text_excerpt = _try_text_excerpt(raw, mime, telegram_config.max_inline_text_chars)
        if text_excerpt:
            parts.append(MessagePart(kind=PartKind.TEXT, text=f"[Attached file excerpt: {filename} ({mime})]\n{text_excerpt}",
                remote_sync=False, origin='attachment_excerpt'))
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
