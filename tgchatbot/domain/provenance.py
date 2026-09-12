"""Stable identity and source attribution shared by live intake and Desktop import."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind


def utc_time(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, (int, float)) or str(value).isdigit():
        value = datetime.fromtimestamp(float(value), timezone.utc)
    elif isinstance(value, str):
        value = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec='seconds')


def telegram_actor(user: Any = None, sender_chat: Any = None) -> dict[str, str]:
    # Anonymous administrators and channel posts belong to sender_chat, even
    # when Telegram supplies a compatibility placeholder in from_user.
    if sender_chat is not None:
        return {'actor_id': f'telegram:chat:{sender_chat.id}', 'actor_kind': 'chat',
                'actor_name': str(getattr(sender_chat, 'title', '') or 'Unknown chat')}
    if user is not None:
        return {'actor_id': f'telegram:user:{user.id}',
                'actor_kind': 'bot' if getattr(user, 'is_bot', False) else 'user',
                'actor_name': str(getattr(user, 'full_name', '') or getattr(user, 'username', '') or 'Unknown user')}
    return {'actor_id': 'unknown', 'actor_kind': 'unknown', 'actor_name': 'Unknown sender'}


def telegram_metadata(message: Any, *, is_edit: bool = False) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        'source': 'telegram', 'source_chat_id': str(message.chat.id),
        'source_message_id': str(message.message_id), 'telegram_message_id': message.message_id,
        **telegram_actor(getattr(message, 'from_user', None), getattr(message, 'sender_chat', None)),
        'sent_at': utc_time(getattr(message, 'date', None)),
        'edited_at': utc_time(getattr(message, 'edit_date', None)), 'is_edit': is_edit,
        'topic_id': str(getattr(message, 'message_thread_id', '') or ''),
        'media_group_id': getattr(message, 'media_group_id', None),
    }
    entities = getattr(message, 'entities', None) or getattr(message, 'caption_entities', None) or []
    direct_topic = getattr(message, 'direct_messages_topic', None)
    if direct_topic is not None:
        metadata['direct_messages_topic_id'] = str(direct_topic.topic_id)
    metadata['entities'] = [item.to_dict() for item in entities]
    reply = getattr(message, 'reply_to_message', None)
    if reply is not None:
        metadata['reply_to_source_id'] = str(reply.message_id)
        metadata['reply_to_source_chat_id'] = str(getattr(getattr(reply, 'chat', None), 'id', message.chat.id))
        metadata['reply_to_actor'] = telegram_actor(getattr(reply, 'from_user', None), getattr(reply, 'sender_chat', None))
    external = getattr(message, 'external_reply', None)
    if external is not None:
        metadata['external_reply'] = external.to_dict()
        if getattr(external, 'message_id', None) is not None:
            metadata['reply_to_source_id'] = str(external.message_id)
        if getattr(external, 'chat', None) is not None:
            metadata['reply_to_source_chat_id'] = str(external.chat.id)
    forward = getattr(message, 'forward_origin', None)
    if forward is not None:
        # Keep forwarded identity separate from the person who sent the message.
        metadata['forward_origin'] = forward.to_dict()
    quote = getattr(message, 'quote', None)
    if quote is not None:
        metadata['quote'] = quote.to_dict()
    return metadata


def attribution(message: ConversationMessage, *, message_id: int | None = None) -> dict[str, Any]:
    source = message.metadata or {}
    result = {key: source[key] for key in (
        'actor_id', 'actor_kind', 'actor_name', 'sent_at', 'source', 'source_chat_id',
        'source_message_id', 'topic_id', 'direct_messages_topic_id', 'reply_to_source_id', 'reply_to_source_chat_id', 'reply_to_actor', 'forward_origin', 'external_reply', 'quote',
    ) if source.get(key) is not None}
    if message_id is not None:
        result['message_id'] = message_id
    return result


def attributed_message(message: ConversationMessage, *, message_id: int | None = None) -> ConversationMessage:
    # The assistant role already identifies our own output. Adding transport
    # labels there teaches an output format the agent should never generate.
    # Incoming peers (including other bots) use USER and retain attribution.
    if message.role == MessageRole.ASSISTANT or not message.metadata.get('source'):
        return message
    label = json.dumps(attribution(message, message_id=message_id), ensure_ascii=False, default=str)
    return replace(message, parts=[MessagePart(kind=PartKind.TEXT,
        text=f'[Message provenance: {label}]', origin='provenance', remote_sync=False), *message.parts])


def original_text(message: ConversationMessage) -> str:
    """User content and attachment descriptions, excluding application notes."""
    return '\n'.join(part.text for part in message.parts if part.text and part.origin not in {'auto_note', 'provenance'})
