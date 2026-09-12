"""Stable identity and source attribution shared by live intake and Desktop import."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any
from collections.abc import Mapping

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
        actor = {'actor_id': f'telegram:user:{user.id}',
                'actor_kind': 'bot' if getattr(user, 'is_bot', False) else 'user',
                'actor_name': str(getattr(user, 'full_name', '') or getattr(user, 'username', '') or 'Unknown user')}
        if getattr(user, 'username', None):
            actor['actor_username'] = str(user.username)
        return actor
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
        'actor_id', 'actor_kind', 'actor_name', 'actor_username', 'sent_at', 'edited_at', 'source', 'source_chat_id',
        'source_message_id', 'topic_id', 'direct_messages_topic_id', 'reply_to_source_id', 'reply_to_source_chat_id', 'reply_to_actor', 'forward_origin', 'external_reply', 'quote',
    ) if source.get(key) is not None}
    if message_id is not None:
        result['message_id'] = message_id
    return result


def message_evidence(source: Mapping[str, Any], *, message_id: int | None,
                     role: str | MessageRole | None, fragments: list[dict],
                     total_characters: int, images: list[dict] | None = None) -> dict[str, Any]:
    """Project an original's supplied slices; selection, bounds and persistence belong to callers."""
    metadata = source.get('metadata') or {}

    def value(key):
        return source.get(key, metadata.get(key))

    speaker = {'id': value('actor_id') or 'unknown'}
    if value('actor_name'):
        speaker['name'] = value('actor_name')
    if value('actor_username'):
        speaker['username'] = value('actor_username')
    if value('actor_kind') and value('actor_kind') != 'user':
        speaker['kind'] = value('actor_kind')
    record: dict[str, Any] = {}
    if message_id is not None:
        record['message_id'] = message_id
    record['speaker'] = speaker
    if value('sent_at') is not None:
        record['sent_at'] = utc_time(value('sent_at'))
    if value('edited_at') is not None:
        record['edited_at'] = utc_time(value('edited_at'))
    role = role.value if isinstance(role, MessageRole) else role
    if role and role != 'user':
        record['role'] = role
    record['fragments'] = []
    for fragment in sorted(fragments, key=lambda item: item['offset']):
        if not fragment['text']:
            continue
        previous = record['fragments'][-1] if record['fragments'] else None
        end = previous['offset'] + len(previous['text']) if previous else -1
        if previous and fragment['offset'] <= end:
            previous['text'] += fragment['text'][end - fragment['offset']:]
        else:
            record['fragments'].append({'offset': fragment['offset'], 'text': fragment['text']})
    # Coverage concerns the original body, not an excerpt's rendered labels.
    # Coalesce already displayed overlap; never join an unseen gap.
    covered = 0
    for fragment in sorted(record['fragments'], key=lambda item: item['offset']):
        if fragment['offset'] > covered:
            break
        covered = max(covered, fragment['offset'] + len(fragment['text']))
    if covered < total_characters:
        record.update(partial=True, total_characters=total_characters)
    if source.get('parts'):
        # Typed source spans distinguish application/attachment context from
        # participant words. Literal lookalike text keeps its original owner.
        groups = []
        for part in source['parts']:
            span = part.get('text_span')
            if span is None:
                continue
            origin = part.get('origin')
            kind = ('provenance' if origin == 'provenance' else
                    'application' if origin == 'auto_note' else
                    'attachment' if origin in {'attachment_reference', 'attachment_excerpt', 'image_compacted'}
                    or part.get('kind') != 'text' else 'original')
            if groups and groups[-1][2] == kind and span[0] <= groups[-1][1] + 1:
                groups[-1][1] = span[1]
            else:
                groups.append([*span, kind])
        originals, annotations = [], []
        for fragment in record['fragments']:
            offset, text = fragment['offset'], fragment['text']
            for start, end, kind in groups:
                start, end = max(start, offset), min(end, offset + len(text))
                if start >= end or kind == 'provenance':
                    continue
                shown = text[start - offset:end - offset]
                if kind == 'original':
                    originals.append({'offset': start, 'text': shown})
                else:
                    annotations.append({'kind': kind, 'text': shown})
        record['fragments'] = originals
        if annotations:
            record['annotations'] = annotations
    for key in ('topic_id', 'direct_messages_topic_id', 'reply_to_source_id',
                'reply_to_source_chat_id', 'reply_to_actor', 'forward_origin',
                'external_reply', 'quote'):
        if value(key) not in (None, '', {}, []):
            record[key] = value(key)
    # Ordinary transport coordinates do not help recall. Relationships can
    # require their source namespace to disambiguate a quoted or replied ID.
    if any(key in record for key in ('reply_to_source_id', 'reply_to_actor',
                                     'forward_origin', 'external_reply', 'quote')):
        for key in ('source', 'source_chat_id', 'source_message_id'):
            if value(key) not in (None, ''):
                record[key] = value(key)
    if images:
        record['images'] = images
    return record


def attributed_message(message: ConversationMessage, *, message_id: int | None = None) -> ConversationMessage:
    # The assistant role already identifies our own output. Adding transport
    # labels there teaches an output format the agent should never generate.
    # Incoming peers (including other bots) use USER and retain attribution.
    if message.role == MessageRole.ASSISTANT or not message.metadata.get('source'):
        return message
    identity = message_evidence(message.metadata, message_id=message_id, role=message.role,
        fragments=[], total_characters=0)
    identity.pop('fragments')
    label = json.dumps(identity, ensure_ascii=False, default=str)
    return replace(message, parts=[MessagePart(kind=PartKind.TEXT,
        text=f'[Message provenance: {label}]', origin='provenance', remote_sync=False),
        *(part for part in message.parts if part.origin != 'provenance')])


def original_text(message: ConversationMessage) -> str:
    """User content and attachment descriptions, excluding application notes."""
    return '\n'.join(part.text for part in message.parts if part.text and part.origin not in {'auto_note', 'provenance'})


def evidence_part_spans(message: ConversationMessage) -> list[dict]:
    """Locate typed parts in the canonical body without interpreting their text."""
    spans: list[dict] = []
    offset = 0
    for part in message.parts:
        if part.text is None:
            continue
        if spans:
            offset += 1
        end = offset + len(part.text)
        spans.append({'text_span': [offset, end], 'kind': part.kind.value, 'origin': part.origin})
        offset = end
    return spans
