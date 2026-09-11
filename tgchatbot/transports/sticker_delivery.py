"""One sticker delivery path shared by direct and rendered Telegram replies."""
from __future__ import annotations

import asyncio
import hashlib

from telegram.error import BadRequest, Forbidden, RetryAfter


async def send_sticker(bot, *, chat_id, sticker, reply_to_message_id=None, deliveries=None):
    receipt = sticker.delivery_receipt()
    operation_id = sticker.delivery_operation_id
    if operation_id is not None:
        if deliveries is None:
            raise RuntimeError('Durable sticker delivery is unavailable')
        operation = await deliveries.begin(operation_id)
        if not operation['may_send']:
            receipt.update(delivery_state=operation['status'],sent=operation['status']=='sent',
                telegram_message_id=operation['telegram_message_id'],error=operation['error'])
            return receipt

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

    # Failure opening the input proves no Telegram request was attempted.
    try:
        source = sticker.path.open('rb')
    except OSError:
        await finish('failed',error='missing_file')
        return receipt
    if sticker.content_sha256:
        try:
            digest = hashlib.file_digest(source,'sha256').hexdigest()
            source.seek(0)
        except OSError:
            source.close()
            await finish('failed',error='unreadable_file')
            return receipt
        if digest != sticker.content_sha256:
            source.close()
            await finish('failed',error='asset_content_changed')
            return receipt
    try:
        with source:
            sent = await bot.send_sticker(chat_id=chat_id,sticker=source,emoji=sticker.emoji,
                reply_to_message_id=reply_to_message_id)
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
