"""Disposable visual evidence sampled in presentation time from original media."""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
import hashlib
import io
from pathlib import Path
from typing import Iterator

import av
from PIL import Image, ImageColor, UnidentifiedImageError

from tgchatbot.operational import from_env

SUPPORTED_EXTENSIONS = frozenset({'.webp', '.webm', '.png', '.jpg', '.jpeg', '.gif', '.apng', '.mp4', '.mov', '.avif'})


@dataclass(frozen=True)
class MediaConfig:
    max_frames: int = 3
    max_dimension: int = 512
    jpeg_quality: int = 85
    background: str = '#ffffff'

    def __post_init__(self):
        if self.max_frames < 1 or self.max_dimension < 1:
            raise ValueError('Sticker frame count and dimension must be positive')
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError('JPEG quality must be between 1 and 100')
        ImageColor.getrgb(self.background)

    @classmethod
    def from_env(cls):
        return from_env(cls, 'STICKER_MEDIA')


@dataclass(frozen=True)
class PreparedFrame:
    data: bytes
    mime_type: str
    timestamp_s: float
    width: int
    height: int

    @property
    def data_b64(self) -> str:
        return base64.b64encode(self.data).decode('ascii')


@dataclass(frozen=True)
class PreparedMedia:
    content_hash: str
    facts: dict
    frames: tuple[PreparedFrame, ...]


def content_hash(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def _decode(path: Path) -> Iterator[tuple[Image.Image, float, float]]:
    """Yield at most one decoded frame at a time; timestamps are presentation times."""
    try:
        picture = Image.open(path)
    except UnidentifiedImageError:
        picture = None
    if picture is not None:
        with picture:
            at = 0.0
            for index in range(getattr(picture, 'n_frames', 1)):
                picture.seek(index)
                # WebP exposes the current frame's duration only after decoding it.
                frame = picture.convert('RGBA')
                duration = max(0.0, float(picture.info.get('duration', 0))) / 1000
                yield frame, at, duration
                at += duration
        return
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise ValueError('Media contains no video frames')
        stream = container.streams.video[0]
        rate = float(stream.average_rate) if stream.average_rate else None
        origin = None
        for index, frame in enumerate(container.decode(stream)):
            if frame.pts is not None and frame.time_base is not None:
                at = float(frame.pts * frame.time_base)
                if origin is None:
                    origin = at
                at -= origin
            elif rate:
                at = index / rate
            else:
                raise ValueError('Video has no usable presentation timestamps or frame rate')
            duration = float(frame.duration * frame.time_base) if getattr(frame, 'duration', 0) and frame.time_base else (1 / rate if rate else 0.0)
            yield frame.to_image().convert('RGBA'), at, max(duration, 0.0)


def prepare_media(path: Path | str, config: MediaConfig | None = None) -> PreparedMedia:
    path, config = Path(path), config or MediaConfig.from_env()
    before = content_hash(path)
    count, end, last_at, width, height = 0, 0.0, 0.0, 0, 0
    for picture, at, duration in _decode(path):
        count += 1
        width, height = picture.size
        last_at, end = at, max(end, at + duration)
    if not count:
        raise ValueError(f'No decodable image frames: {path.name}')
    # Last presentation start includes the final expression even if its dwell is short.
    timing_known = count == 1 or end > 0
    extent = last_at if timing_known else count - 1
    sample_count = min(config.max_frames, count)
    targets = [extent * index / (sample_count - 1) for index in range(sample_count)] if sample_count > 1 else [0.0]
    selected: list[PreparedFrame] = []
    last_digest: str | None = None
    target_index = 0
    previous: tuple[Image.Image, float] | None = None

    def keep(picture: Image.Image, at: float):
        nonlocal last_digest
        picture.thumbnail((config.max_dimension, config.max_dimension), Image.Resampling.LANCZOS)
        canvas = Image.new('RGB', picture.size, ImageColor.getrgb(config.background))
        canvas.paste(picture, mask=picture.getchannel('A'))
        digest = hashlib.sha256(str(canvas.size).encode() + canvas.tobytes()).hexdigest()
        if digest == last_digest:
            return
        # A later return to an earlier expression is meaningful motion evidence.
        last_digest = digest
        buffer = io.BytesIO()
        canvas.save(buffer, format='JPEG', quality=config.jpeg_quality)
        selected.append(PreparedFrame(buffer.getvalue(), 'image/jpeg', float(at), *canvas.size))

    for index, (picture, at, _) in enumerate(_decode(path)):
        position = at if timing_known else float(index)
        # A presentation interval belongs to the preceding frame, not the next frame.
        while target_index < len(targets) and targets[target_index] < position and previous is not None:
            keep(previous[0].copy(), previous[1])
            target_index += 1
        previous = (picture, at)
    if previous is not None:
        while target_index < len(targets):
            keep(previous[0].copy(), previous[1])
            target_index += 1
    if content_hash(path) != before:
        raise ValueError('Source changed during media preparation')
    facts = {'animated': count > 1, 'frame_count': count, 'width': width, 'height': height,
             'duration_s': end, 'supplied_frames': len(selected), 'frame_times_s': [f.timestamp_s for f in selected],
             'frame_order': 'chronological', 'timing': 'presentation' if timing_known else 'unknown; frame-index fallback',
             'sampling': 'Representative positions spanning the timeline; intermediate events may be omitted; duplicate images omitted',
             'preparation': asdict(config), 'format': path.suffix.lower().lstrip('.')}
    return PreparedMedia(before, facts, tuple(selected))
