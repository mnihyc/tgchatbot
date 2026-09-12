"""One sticker delivery path shared by direct and rendered Telegram replies."""
from __future__ import annotations

import asyncio
import hashlib
import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from telegram.error import BadRequest, Forbidden, RetryAfter


class UnsupportedStickerMedia(ValueError):
    pass


def _upload(data: bytes, filename: str) -> io.BytesIO:
    """Encode static source pixels without changing the catalog's original identity."""
    try:
        with Image.open(io.BytesIO(data)) as picture:
            if getattr(picture, 'n_frames', 1) > 1:
                raise UnsupportedStickerMedia('unsupported_sticker_animation')
            if picture.format == 'WEBP':
                upload = io.BytesIO(data)
            else:
                upload = io.BytesIO()
                picture.convert('RGBA').save(upload, format='WEBP', lossless=True, exact=True)
                upload.seek(0)
            upload.name = 'sticker.webp'
            return upload
    except UnidentifiedImageError:
        # Native animation/video decoding and service constraints belong to
        # Telegram. Do not transcode or clip them as an implicit fallback.
        if Path(filename).suffix.lower() not in {'.webp', '.webm', '.tgs'}:
            raise UnsupportedStickerMedia('unsupported_sticker_format')
        upload = io.BytesIO(data)
        upload.name = filename
        return upload


async def send_sticker(bot, *, chat_id, sticker, reply_to_message_id=None, deliveries=None,
                       message_thread_id=None, direct_messages_topic_id=None):
    receipt = sticker.delivery_receipt()
    operation_id = sticker.delivery_operation_id
    if operation_id is not None:
        if deliveries is None:
            raise RuntimeError('Durable sticker delivery is unavailable')
        operation = await deliveries.begin(operation_id, sticker_id=sticker.source_id or sticker.path.stem,
                                           content_sha256=sticker.content_sha256)
        if not operation['may_send']:
            sticker.delivery_state = operation['status']
            sticker.telegram_message_id = operation['telegram_message_id']
            sticker.error = operation['error']
            return sticker.delivery_receipt()

    async def finish(status, *, message_id=None, error=None):
        receipt.update(delivery_state=status,sent=status=='sent')
        if message_id is not None:
            receipt['telegram_message_id'] = message_id
        if error is not None:
            receipt['error'] = error
        if operation_id is not None:
            await deliveries.finish(operation_id,status,telegram_message_id=message_id,error=error)
        sticker.delivery_state = status
        sticker.telegram_message_id = message_id
        sticker.error = error

    # Freeze one original snapshot before checking its selected identity. An
    # in-place file change during upload cannot replace the verified content.
    try:
        data = sticker.path.read_bytes()
    except FileNotFoundError:
        await finish('failed',error='missing_file')
        return receipt
    except OSError:
        await finish('failed',error='unreadable_file')
        return receipt
    if sticker.content_sha256 and hashlib.sha256(data).hexdigest() != sticker.content_sha256:
        await finish('failed',error='asset_content_changed')
        return receipt
    try:
        source = _upload(data, sticker.path.name)
    except UnsupportedStickerMedia as exc:
        await finish('failed',error=str(exc))
        return receipt
    except (OSError, ValueError, Image.DecompressionBombError):
        await finish('failed',error='unreadable_sticker_media')
        return receipt
    try:
        with source:
            topic = {key: value for key, value in (
                ('message_thread_id', message_thread_id), ('direct_messages_topic_id', direct_messages_topic_id))
                if value is not None}
            sent = await bot.send_sticker(chat_id=chat_id,sticker=source,emoji=sticker.emoji,
                reply_to_message_id=reply_to_message_id, **topic)
    except asyncio.CancelledError:
        # Persist ambiguity before propagating cancellation. Startup recovery
        # handles hard process death or database loss during this best effort.
        await asyncio.shield(finish('unknown',error='CancelledError'))
        raise
    except (BadRequest, Forbidden, RetryAfter) as exc:
        await finish('failed',error=type(exc).__name__)
    except Exception as exc:
        await finish('unknown',error=type(exc).__name__)
    else:
        await finish('sent',message_id=getattr(sent,'message_id',None))
    return receipt
