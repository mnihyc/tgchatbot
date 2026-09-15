"""Stable identity and source attribution shared by live intake and Desktop import."""
from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any
from collections.abc import Mapping

from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind
from tgchatbot.domain.profiles import present_profile
from tgchatbot.domain.identities import actor_reference
from tgchatbot.domain.attachments import attachment_description, generated_attachment_reference
from tgchatbot.domain.timestamps import format_timestamp_fields, resolve_timezone

AGENT_PRESENTATION_VERSION = 2


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


def attribution(message: ConversationMessage, *, message_id: int | None = None,
                timezone: str | None = 'UTC') -> dict[str, Any]:
    source = message.metadata or {}
    result = {key: source[key] for key in (
        'actor_id', 'actor_kind', 'actor_name', 'actor_username', 'sent_at', 'edited_at', 'source', 'source_chat_id',
        'source_message_id', 'topic_id', 'direct_messages_topic_id', 'reply_to_source_id', 'reply_to_source_chat_id', 'reply_to_actor', 'forward_origin', 'external_reply', 'quote',
    ) if source.get(key) is not None}
    if message_id is not None:
        result['message_id'] = message_id
    return present_attribution(result, timezone)


def present_attribution(record: Mapping[str, Any], timezone: str | None = None) -> dict[str, Any]:
    result = format_timestamp_fields(record, ('sent_at', 'edited_at'), timezone)
    if isinstance(result.get('forward_origin'), Mapping):
        result['forward_origin'] = format_timestamp_fields(result['forward_origin'], ('date', 'forwarded_date'), timezone)
    if isinstance(result.get('external_reply'), Mapping):
        external = dict(result['external_reply'])
        if isinstance(external.get('origin'), Mapping):
            external['origin'] = format_timestamp_fields(external['origin'], ('date',), timezone)
        result['external_reply'] = external
    return result


def _related_actor(value: Mapping[str, Any], *, chat: bool = False) -> dict[str, Any]:
    """Keep source identity, not Telegram account or download capabilities."""
    if any(key in value for key in ('actor_id', 'actor_name', 'actor_username')):
        return {key: actor_reference(item) if key == 'actor_id' else item
                for key, item in value.items() if key in
                {'actor_id', 'actor_kind', 'actor_name', 'actor_username'}
                and item is not None and not (key == 'actor_kind' and item == 'user')}
    result = {}
    if value.get('id') is not None:
        result['actor_id'] = ('chat_id:' if chat else 'person_id:') + str(value['id'])
    name = value.get('title') if chat else ' '.join(str(value[key]) for key in ('first_name', 'last_name') if value.get(key))
    if name:
        result['actor_name'] = name
    if value.get('username'):
        result['actor_username'] = value['username']
    if chat:
        result['actor_kind'] = 'chat'
    elif value.get('is_bot'):
        result['actor_kind'] = 'bot'
    return result


def _forward_origin(origin: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: origin[key] for key in ('type', 'date', 'forwarded_date', 'message_id',
              'forwarded_message_id', 'author_signature', 'saved_from') if origin.get(key) is not None}
    if any(key in origin for key in ('actor_id', 'actor_name', 'actor_username')):
        result['actor'] = _related_actor(origin)
    for key in ('sender_user', 'sender_chat', 'chat'):
        if isinstance(origin.get(key), Mapping):
            result['actor'] = _related_actor(origin[key], chat=key != 'sender_user')
            break
    if origin.get('sender_user_name'):
        # Hidden authors have a display name, not an inferred Telegram identity.
        result['actor'] = {'actor_name': origin['sender_user_name']}
    return result


def _external_reply(external: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: external[key] for key in ('message_id', 'has_media_spoiler')
              if external.get(key) is not None}
    if isinstance(external.get('origin'), Mapping):
        result['origin'] = _forward_origin(external['origin'])
    if isinstance(external.get('chat'), Mapping):
        result['chat'] = _related_actor(external['chat'], chat=True)
    if isinstance(external.get('link_preview_options'), Mapping) and external['link_preview_options'].get('url'):
        result['url'] = external['link_preview_options']['url']
    # External attachments are references, not remotely synced originals. Keep
    # their meaningful descriptions without presenting unusable download IDs.
    attachments = []
    for kind in ('animation', 'audio', 'document', 'photo', 'sticker', 'video', 'video_note', 'voice'):
        value = external.get(kind)
        if not value:
            continue
        item = {'kind': kind}
        if isinstance(value, Mapping):
            item.update({key: value[key] for key in ('file_name', 'mime_type', 'file_size',
                'width', 'height', 'duration', 'title', 'performer', 'emoji', 'set_name',
                'is_animated', 'is_video') if value.get(key) is not None})
        attachments.append(item)
    if attachments:
        result['attachments'] = attachments
    fields = {
        'contact': ('phone_number', 'first_name', 'last_name'),
        'dice': ('emoji', 'value'), 'game': ('title', 'description', 'text'),
        'giveaway': ('winners_selection_date', 'winner_count', 'country_codes', 'prize_description',
                     'premium_subscription_month_count', 'prize_star_count'),
        'giveaway_winners': ('winner_count', 'unclaimed_prize_count', 'prize_description'),
        'invoice': ('title', 'description', 'currency', 'total_amount'),
        'location': ('latitude', 'longitude', 'horizontal_accuracy'),
        'poll': ('question', 'type', 'is_closed', 'allows_multiple_answers', 'explanation'),
        'venue': ('title', 'address'),
    }
    for key, names in fields.items():
        value = external.get(key)
        if isinstance(value, Mapping):
            result[key] = {name: value[name] for name in names if value.get(name) is not None}
            if key == 'poll' and value.get('options'):
                result[key]['options'] = [option['text'] for option in value['options'] if 'text' in option]
            if key == 'venue' and isinstance(value.get('location'), Mapping):
                result[key]['location'] = {name: value['location'][name] for name in fields['location']
                    if value['location'].get(name) is not None}
    return result


def _reply_quote(quote: Mapping[str, Any]) -> dict[str, Any]:
    result = {key: quote[key] for key in ('text', 'position', 'is_manual') if quote.get(key) is not None}
    if quote.get('entities'):
        result['entities'] = []
        for entity in quote['entities']:
            item = {key: entity[key] for key in ('type', 'offset', 'length', 'url', 'language')
                    if entity.get(key) is not None}
            if isinstance(entity.get('user'), Mapping):
                item['actor'] = _related_actor(entity['user'])
            result['entities'].append(item)
    return result


def agent_attribution(record: Mapping[str, Any], timezone: str | None = None) -> dict[str, Any]:
    """Project new application-owned evidence only; recorded exchanges stay exact."""
    result = dict(record)
    if isinstance(result.get('actor_id'), str):
        result['actor_id'] = actor_reference(result['actor_id'])
    if isinstance(result.get('speaker'), Mapping):
        result['speaker'] = {**result['speaker'], 'id': actor_reference(result['speaker']['id'])}
    if isinstance(result.get('reply_to_actor'), Mapping):
        result['reply_to_actor'] = _related_actor(result['reply_to_actor'])
    if isinstance(result.get('forward_origin'), Mapping):
        result['forward_origin'] = _forward_origin(result['forward_origin'])
    if isinstance(result.get('external_reply'), Mapping):
        result['external_reply'] = _external_reply(result['external_reply'])
    if isinstance(result.get('quote'), Mapping):
        result['quote'] = _reply_quote(result['quote'])
    return present_attribution(result, timezone)


def present_tool_output(name: str, output: Mapping[str, Any], timezone: str | None = None) -> dict[str, Any]:
    """Project the defined timestamp fields of app-owned memory tool results."""
    result = dict(output)
    if name in {'memory_search', 'memory_read'} and 'messages' in output:
        result['messages'] = [present_attribution(record, timezone) for record in output['messages']]
    elif name == 'user_profile_fetch':
        result = format_timestamp_fields(output, ('as_of', 'fetched_at'), timezone)
        if 'profiles' in output:
            result['profiles'] = [present_profile(profile, timezone) for profile in output['profiles']]
        if 'timezone' in output:
            result['timezone'] = resolve_timezone(timezone).key
    return result


def present_image_evidence(part: MessagePart, timezone: str | None = None, *,
                           presentation_version: int = 1) -> MessagePart:
    """Project the app-owned image label; literal document/user text stays untouched."""
    prefix = '[Original image evidence: '
    if (part.kind != PartKind.TEXT or not (part.origin or '').startswith('memory_image:')
            or not (part.text or '').startswith(prefix) or not part.text.endswith(']')):
        return part
    try:
        record = json.loads(part.text[len(prefix):-1])
    except json.JSONDecodeError:
        return part
    if not isinstance(record, dict):
        return part
    if presentation_version >= 2:
        identity = message_evidence(record, message_id=record.get('message_id'), role=None,
            fragments=[], total_characters=0, timezone=timezone, presentation_version=presentation_version)
        identity.pop('fragments')
        record = {'image_id': record['image_id'], **identity}
    return replace(part, text=prefix + json.dumps(present_attribution(record, timezone),
        ensure_ascii=False, default=str) + ']')


def _quoted_fragments(source: Mapping[str, Any], fragments: list[dict],
                      original: ConversationMessage | None) -> list[dict]:
    metadata = original.metadata if original is not None else source.get('metadata') or source
    entities = [entity for entity in metadata.get('entities', [])
        if isinstance(entity, dict) and entity.get('type') in {'blockquote', 'expandable_blockquote'}]
    if not entities or not fragments:
        return []
    if original is not None:
        body = '\n'.join(part.text for part in original.parts if part.text is not None)
        parts = evidence_part_spans(original)
    else:
        body = source.get('text')
        parts = source.get('parts') or []
    if not isinstance(body, str):
        return []
    # Telegram entities belong to the incoming text/caption, not generated
    # notes or later independent text parts in the canonical body.
    anchor = next((part['text_span'] for part in parts if part.get('text_span') is not None
        and part.get('kind') == 'text' and part.get('origin') in (None, '')), None)
    if anchor is None:
        return []
    encoded = body[anchor[0]:anchor[1]].encode('utf-16-le')
    ranges = []
    for entity in entities:
        offset, length = entity.get('offset'), entity.get('length')
        if not isinstance(offset, int) or not isinstance(length, int) or offset < 0 or length <= 0:
            continue
        if 2 * (offset + length) > len(encoded):
            continue
        try:
            start = anchor[0] + len(encoded[:2 * offset].decode('utf-16-le'))
            end = anchor[0] + len(encoded[:2 * (offset + length)].decode('utf-16-le'))
        except UnicodeDecodeError:
            continue  # An invalid boundary must not mark a different character.
        ranges.append((start, end))
    merged = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    quoted = []
    for fragment in fragments:
        offset, text = fragment['offset'], fragment['text']
        for start, end in merged:
            start, end = max(start, offset), min(end, offset + len(text))
            if start < end:
                quoted.append({'offset': start, 'text': text[start - offset:end - offset]})
    return quoted


def _present_whole_line_quotes(record: dict, source: Mapping[str, Any],
                               original: ConversationMessage | None) -> None:
    """Reference complete quoted lines without editing or repeating their text.

    Partial slices and multipart originals retain explicit quoted wording: their
    displayed line positions can differ from the canonical body.
    """
    fragments = record['fragments']
    if len(fragments) != 1 or fragments[0]['offset'] != 0 or record.get('partial'):
        return
    parts = original.parts if original is not None else source.get('parts')
    if parts is None:
        return
    if original is not None:
        text_parts = [part for part in parts if part.text is not None]
        if len(text_parts) != 1 or text_parts[0].kind != PartKind.TEXT or text_parts[0].origin:
            return
    else:
        text_parts = [part for part in parts if part.get('text_span') is not None]
        if len(text_parts) != 1 or text_parts[0].get('kind') != 'text' or text_parts[0].get('origin'):
            return
    text = fragments[0]['text']
    lines, remaining = [], []
    for quote in record['quoted_fragments']:
        start = quote['offset']
        end = start + len(quote['text'])
        # Entity ranges can include a final newline, but never trim source text.
        content_end = end - 1 if end > start and text[end - 1:end] == '\n' else end
        if ((start == 0 or text[start - 1:start] == '\n')
                and (content_end == len(text) or text[content_end:content_end + 1] == '\n')):
            lines.append([text.count('\n', 0, start) + 1, text.count('\n', 0, content_end) + 1])
        else:
            remaining.append(quote)
    if lines:
        record['quoted_lines'] = lines
        if remaining:
            record['quoted_fragments'] = remaining
        else:
            del record['quoted_fragments']


def message_evidence(source: Mapping[str, Any], *, message_id: int | None,
                     role: str | MessageRole | None, fragments: list[dict],
                     total_characters: int, images: list[dict] | None = None,
                     timezone: str | None = None,
                     original: ConversationMessage | None = None,
                     presentation_version: int = 1) -> dict[str, Any]:
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
        record['sent_at'] = value('sent_at')
    if value('edited_at') is not None:
        record['edited_at'] = value('edited_at')
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
    groups = []
    if source.get('parts'):
        # Typed source spans distinguish application/attachment context from
        # participant words. Literal lookalike text keeps its original owner.
        for index, part in enumerate(source['parts']):
            span = part.get('text_span')
            if span is None:
                continue
            origin = part.get('origin')
            kind = ('provenance' if origin == 'provenance' else
                    'application' if origin == 'auto_note' else
                    'attachment' if origin in {'attachment_reference', 'attachment_excerpt', 'image_compacted'}
                    or part.get('kind') != 'text' else 'original')
            if (groups and groups[-1][2] == kind and span[0] <= groups[-1][1] + 1
                    and not (presentation_version >= 2 and kind == 'attachment')):
                groups[-1][1] = span[1]
            else:
                groups.append([*span, kind, part, part.get('part_index', index)])
    # Completeness follows the same typed projection as the displayed evidence.
    # Hidden provenance and its joining separators are not missing user words;
    # canonical offsets and caller-selected slices remain unchanged.
    required = ([(start, end) for start, end, kind, *_ in groups if kind != 'provenance']
                if source.get('parts') else [(0, total_characters)])
    for start, end in required:
        covered = start
        for fragment in record['fragments']:
            if fragment['offset'] > covered:
                break
            covered = max(covered, fragment['offset'] + len(fragment['text']))
            if covered >= end:
                break
        if covered < end:
            record.update(partial=True, total_characters=total_characters)
            break
    if source.get('parts'):
        image_ids = {image['image_id'] for image in images or []}
        originals, annotations = [], []
        for fragment in record['fragments']:
            offset, text = fragment['offset'], fragment['text']
            for start, end, kind, part, index in groups:
                start, end = max(start, offset), min(end, offset + len(text))
                if start >= end or kind == 'provenance':
                    continue
                shown = text[start - offset:end - offset]
                if kind == 'original':
                    originals.append({'offset': start, 'text': shown})
                else:
                    if (presentation_version >= 2 and kind == 'attachment'
                            and shown == generated_attachment_reference(part)):
                        if part.get('kind') in {'image', 'sticker'} and any(
                                image_id.startswith(f'img:{message_id}:') and image_id.endswith(f':{index}')
                                for image_id in image_ids):
                            # Image occurrences already report selection IDs and
                            # availability; retain custom captions/descriptions.
                            if part.get('detail') in (None, 'auto', 'low', 'high'):
                                continue
                        if part.get('kind') == 'file':
                            shown = attachment_description(MessagePart(PartKind.FILE, **{
                                key: part[key] for key in ('filename', 'mime_type', 'size_bytes',
                                'artifact_path', 'workspace_path', 'detail') if key in part}))
                    annotations.append({'kind': kind, 'text': shown})
        record['fragments'] = originals
        if annotations:
            record['annotations'] = annotations
    quoted = _quoted_fragments(source, record['fragments'], original)
    if quoted:
        record['quoted_fragments'] = quoted
        if presentation_version >= 2:
            _present_whole_line_quotes(record, source, original)
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
    return (agent_attribution if presentation_version >= 2 else present_attribution)(record, timezone)


def attributed_message(message: ConversationMessage, *, message_id: int | None = None,
                       timezone: str | None = None, presentation_version: int | None = None) -> ConversationMessage:
    # Assistant and tool roles already identify their own output. Participant
    # labels belong to incoming peers (including other bots), not the outer
    # tool observation. Source labels inside tool evidence remain untouched.
    if presentation_version is None:
        presentation_version = message.metadata.get('presentation_version', 1)
    if (message.role in {MessageRole.ASSISTANT, MessageRole.TOOL} or not message.metadata.get('source')
            or presentation_version >= 2 and message.metadata.get('synthetic_role')):
        return message
    body = '\n'.join(part.text for part in message.parts if part.text is not None)
    identity = message_evidence(message.metadata, message_id=message_id, role=message.role,
        fragments=[{'offset': 0, 'text': body}], total_characters=len(body), timezone=timezone, original=message,
        presentation_version=presentation_version)
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
    for index, part in enumerate(message.parts):
        if part.text is None:
            continue
        if spans:
            offset += 1
        end = offset + len(part.text)
        spans.append({'text_span': [offset, end], 'kind': part.kind.value, 'origin': part.origin,
            'part_index': index, **{field: getattr(part, field) for field in
                ('filename', 'mime_type', 'size_bytes', 'artifact_path', 'workspace_path', 'detail')
                if getattr(part, field) is not None}})
        offset = end
    return spans
