"""Telegram owns upload acknowledgments; transfer copies are disposable."""
from __future__ import annotations

import logging

from telegram import InputFile
from telegram.error import BadRequest, Forbidden, RetryAfter

from tgchatbot.domain.models import OutboundArtifact

logger = logging.getLogger(__name__)


async def deliver_artifact(bot, *, chat_id: int, artifact: OutboundArtifact,
                           reply_to_message_id: int | None = None,
                           message_thread_id: int | None = None,
                           direct_messages_topic_id: int | None = None) -> dict:
    receipt = {'filename': artifact.filename, 'sent': False, 'delivery_state': 'failed'}
    if artifact.workspace_path:
        receipt['workspace_path'] = artifact.workspace_path
    submitted = False
    destination = {'chat_id': chat_id, 'reply_to_message_id': reply_to_message_id}
    if message_thread_id is not None:
        destination['message_thread_id'] = message_thread_id
    if direct_messages_topic_id is not None:
        destination['direct_messages_topic_id'] = direct_messages_topic_id
    try:
        with artifact.path.open('rb') as stream:
            submitted = True
            if artifact.path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
                message = await bot.send_photo(**destination, photo=stream,
                    caption=artifact.caption)
            else:
                message = await bot.send_document(**destination,
                    document=InputFile(stream, filename=artifact.filename),
                    caption=artifact.caption)
        receipt.update(sent=True, delivery_state='sent', telegram_message_id=message.message_id)
    except Exception as exc:
        logger.exception('Failed to send artifact %s', artifact.filename)
        receipt['error'] = type(exc).__name__
        if submitted and not isinstance(exc, (BadRequest, Forbidden, RetryAfter)):
            receipt['delivery_state'] = 'unknown'
    finally:
        artifact.discard()
    return receipt
