"""Stream one Telegram Desktop JSON conversation through ordinary message intake.

Desktop's export uses bare peer IDs and mixed string/entity text. The explicit
destination chat supplies the live Bot API namespace; original export metadata
is retained. Import never invokes Telegram handlers, tools, or media uploads.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, fields
import json
import mimetypes
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any, Callable, Iterator
from datetime import datetime
from zoneinfo import ZoneInfo

import ijson
from ijson.common import ObjectBuilder

from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind, SessionSettings
from tgchatbot.config import TelegramConfig
from tgchatbot.domain.provenance import telegram_actor, telegram_metadata, utc_time
from tgchatbot.operational import from_env
from tgchatbot.settings_schema import DEFAULT_METADATA_TIMEZONE
from tgchatbot.storage.postgres_store import PostgresStore, StaleScopeError


_PEER = re.compile(r'^(user|chat|channel)([0-9]+)$')


@dataclass(frozen=True)
class ImportConfig:
    # Streaming defaults, overridable in the same .env as the bot. Batch bytes
    # are a target: a single intact record may exceed them, up to max_record_bytes.
    max_record_bytes: int = 4 * 1024 * 1024
    batch_bytes: int = 1024 * 1024
    batch_messages: int = 100

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f'IMPORT_{field.name.upper()} must be a positive integer')


@dataclass(frozen=True)
class ExportChat:
    metadata: dict[str, Any]
    index: int | None  # None is a single-chat export; otherwise chats.list index.


@dataclass(frozen=True)
class ImportResult:
    messages: int
    batches: int
    generation: int


def inspect_export(path: Path, export_chat_id: str | None = None) -> ExportChat:
    """Read only chat headers in a streaming pass, never whole chat objects."""
    single: dict[str, Any] = {}
    selected: ExportChat | None = None
    current: dict[str, Any] = {}
    current_messages = False
    count = 0
    full_export = False
    single_messages = False
    with path.open('rb') as stream:
        for prefix, event, value in ijson.parse(stream, use_float=True):
            if prefix == 'chats.list' and event == 'start_array':
                full_export = True
            elif prefix == 'messages' and event == 'start_array':
                single_messages = True
            elif prefix in {'id', 'name', 'type'} and event in {'string', 'number', 'null'}:
                single[prefix] = value
            elif prefix == 'chats.list.item' and event == 'start_map':
                current = {}
                current_messages = False
            elif prefix == 'chats.list.item.messages' and event == 'start_array':
                current_messages = True
            elif prefix in {'chats.list.item.id', 'chats.list.item.name', 'chats.list.item.type'} and event in {'string', 'number', 'null'}:
                current[prefix.rsplit('.', 1)[-1]] = value
            elif prefix == 'chats.list.item' and event == 'end_map':
                if (export_chat_id is None and count == 0) or (export_chat_id is not None and str(current.get('id')) == str(export_chat_id)):
                    if selected is not None:
                        raise ValueError('The selected exported chat ID occurs more than once.')
                    if not current_messages:
                        raise ValueError('The selected exported chat has no messages array.')
                    selected = ExportChat(dict(current), count)
                count += 1
    if full_export:
        if export_chat_id is None and count > 1:
            raise ValueError('This export contains multiple chats; select one with --export-chat-id.')
        if selected is None:
            raise ValueError('The requested chat was not found in chats.list.')
        return selected
    if not single_messages:
        raise ValueError('Expected a Telegram Desktop single-chat export or chats.list export.')
    if export_chat_id is not None and str(single.get('id')) != str(export_chat_id):
        raise ValueError('The requested chat ID does not match this single-chat export.')
    return ExportChat(single, None)


def iter_records(path: Path, chat: ExportChat) -> Iterator[dict[str, Any]]:
    with path.open('rb') as stream:
        if chat.index is None:
            for record in ijson.items(stream, 'messages.item', use_float=True):
                if not isinstance(record, dict):
                    raise ValueError('Every exported message must be an object.')
                yield record
            return
        index = -1
        builder: ObjectBuilder | None = None
        depth = 0
        for prefix, event, value in ijson.parse(stream, use_float=True):
            if prefix == 'chats.list.item' and event == 'start_map':
                index += 1
            if index != chat.index:
                continue
            if prefix == 'chats.list.item.messages.item' and builder is None:
                if event != 'start_map':
                    raise ValueError('Every exported message must be an object.')
                builder = ObjectBuilder()
            if builder is not None:
                builder.event(event, value)
                if event in {'start_map', 'start_array'}:
                    depth += 1
                elif event in {'end_map', 'end_array'}:
                    depth -= 1
                if depth == 0:
                    yield builder.value
                    builder = None


def _actor(peer_id: Any, name: Any, bot_user_id: int | None) -> tuple[Any, Any]:
    match = _PEER.fullmatch(str(peer_id or ''))
    if not match:
        return None, None
    kind, bare = match.group(1), int(match.group(2))
    if kind == 'user':
        return SimpleNamespace(id=bare, full_name=str(name or 'Unknown user'), username=None,
                               is_bot=bare == bot_user_id), None
    chat_id = -bare if kind == 'chat' else -(10**12 + bare)
    return None, SimpleNamespace(id=chat_id, title=str(name or 'Unknown chat'))


def _text(record: dict[str, Any]) -> str:
    value = record.get('text', '')
    if isinstance(value, str):
        return value
    if not isinstance(value, list):
        raise ValueError('Exported message text must be a string or entity list.')
    fragments = []
    for fragment in value:
        text = fragment if isinstance(fragment, str) else fragment.get('text') if isinstance(fragment, dict) else None
        if not isinstance(text, str):
            raise ValueError('An exported text entity has no text.')
        fragments.append(text)
    return ''.join(fragments)


def _entities(record: dict[str, Any]) -> list[Any]:
    # Store offsets into the original body rather than duplicating each entity's
    # text. Telegram offsets count UTF-16 code units, including for emoji.
    fragments = record.get('text_entities', record.get('text', []))
    if not isinstance(fragments, list):
        return []
    entities = []
    offset = 0
    for fragment in fragments:
        text = fragment if isinstance(fragment, str) else fragment.get('text') if isinstance(fragment, dict) else None
        if not isinstance(text, str):
            raise ValueError('An exported text entity has no text.')
        length = len(text.encode('utf-16-le')) // 2
        if isinstance(fragment, dict):
            entity = {key: value for key, value in fragment.items() if key != 'text'}
            entity.update(offset=offset, length=length)
            entities.append(SimpleNamespace(to_dict=lambda value=entity: value))
        offset += length
    return entities


def _attachment(record: dict[str, Any]) -> tuple[str, str, str | None, PartKind] | None:
    kind = record.get('media_type') or ('photo' if record.get('photo') else 'file' if record.get('file') else None)
    if not kind:
        return None
    filename = str(record.get('file_name') or record.get('file') or record.get('photo') or kind)
    mime = record.get('mime_type') or mimetypes.guess_type(filename)[0]
    visual_kind = (PartKind.STICKER if kind == 'sticker' else PartKind.IMAGE
                   if record.get('photo') or str(mime or '').startswith(('image/', 'video/')) else PartKind.TEXT)
    return str(kind), filename, mime, visual_kind


def _attachment_hint(record: dict[str, Any], kind: str, filename: str, *, available: bool = False) -> str:
    emoji = record.get('sticker_emoji') if kind == 'sticker' else None
    emoji_hint = f'; emoji={json.dumps(emoji, ensure_ascii=False)}' if emoji else ''
    state = 'image available' if available else 'media unavailable in bot workspace'
    return f'[Imported attachment: {kind}{emoji_hint}; reference={json.dumps(filename, ensure_ascii=False)}; {state}]'


def _import_visual(message: ConversationMessage, record: dict[str, Any], export_root: Path,
                   config: TelegramConfig) -> None:
    attachment = _attachment(record)
    if attachment is None or attachment[3] == PartKind.TEXT:
        return
    kind, filename, mime, _ = attachment
    reference = record.get('photo') or record.get('file')
    if not isinstance(reference, str):
        return
    try:
        source = (export_root / reference).resolve()
        # An exported relative reference owns only files inside its bundle.
        # Absolute/traversing/symlink references cannot read unrelated local data.
        if not source.is_relative_to(export_root) or not source.is_file():
            return
        photo, sticker = bool(record.get('photo')), kind == 'sticker'
        if not photo and not sticker and (config.max_document_bytes <= 0
                or source.stat().st_size > config.max_document_bytes):
            return
        raw = source.read_bytes()
    except (OSError, ValueError):
        return
    from tgchatbot.media.ingest import _build_visual_preview_parts, visual_parts_from_bytes
    if photo or sticker:
        parts = visual_parts_from_bytes(raw=raw, filename=filename,
            max_frames=1 if photo else config.max_sticker_frames,
            max_bytes=config.max_photo_bytes if photo else config.max_sticker_bytes,
            max_keyframe_candidates=config.max_video_keyframe_candidates,
            detail='auto' if photo else 'low')
    else:
        parts = _build_visual_preview_parts(raw=raw, mime=mime or 'application/octet-stream',
            filename=filename, telegram_config=config)
    if not parts:
        return
    for index, part in enumerate(message.parts):
        if part.origin == 'attachment_reference':
            message.parts[index] = MessagePart(PartKind.TEXT,
                text=_attachment_hint(record, kind, filename, available=True),
                origin='attachment_reference', remote_sync=False)
            break
    message.parts.extend(parts)
    message.metadata['media_availability'] = 'imported'


async def _retained_import(store: PostgresStore, session_id: str, message: ConversationMessage, scope):
    """An unchanged export without its files cannot revoke retained visual evidence."""
    if message.metadata.get('media_availability') != 'not_imported' or not any(
            part.kind in {PartKind.IMAGE, PartKind.STICKER} for part in message.parts):
        return None
    metadata = message.metadata
    previous = await store.read_message_by_source(session_id,
        source=metadata['source'], source_chat_id=metadata['source_chat_id'],
        source_message_id=metadata['source_message_id'], expected_scope=scope, generation_only=True)
    if previous is None:
        return None
    original = previous.message
    if (not original.metadata.get('imported') or original.role != message.role or original.name != message.name
            or original.parts[0].text != message.parts[0].text
            or any(original.metadata.get(key) != value for key, value in metadata.items()
                   if key not in {'media_availability', 'source_revision'})):
        return None
    images = (await store.describe_message_images(session_id, [previous.db_id],
        expected_scope={'generation': scope['generation']})).get(previous.db_id, [])
    if not images or not all(image['available'] for image in images):
        return None
    # The no-op path needs the same full-reset boundary as an ordinary append.
    await store.assert_scope(session_id, scope, generation_only=True)
    return previous


def desktop_message(record: dict[str, Any], chat: ExportChat, *, chat_id: int,
                    bot_user_id: int | None = None,
                    timezone: str = DEFAULT_METADATA_TIMEZONE) -> ConversationMessage:
    try:
        source_id = str(record['id'])
        if not re.fullmatch(r'-?[0-9]+', source_id):
            raise ValueError('Expected an integer source ID')
        message_id = int(source_id)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('An exported message has no numeric message ID.') from exc
    # Desktop shifts pre-upgrade basic-group IDs by -1,000,000,000 when
    # merging history into a supergroup export. Preserve that source namespace;
    # only the database's own message IDs must be positive.
    if message_id == 0:
        raise ValueError('An exported message ID must be nonzero.')
    sender, sender_chat = _actor(record.get('from_id', record.get('actor_id')),
                                 record.get('from', record.get('actor')), bot_user_id)
    timestamp = record.get('date_unixtime', record.get('date'))
    if timestamp is None:
        raise ValueError(f'Exported message {message_id} has no timestamp.')
    edited = record.get('edited_unixtime', record.get('edited'))
    def export_time(value):
        if isinstance(value, str) and not value.isdigit():
            value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            if value.tzinfo is None:
                value = value.replace(tzinfo=ZoneInfo(timezone))
        return utc_time(value)

    message = SimpleNamespace(chat=SimpleNamespace(id=chat_id), message_id=message_id,
        from_user=sender, sender_chat=sender_chat, date=export_time(timestamp), edit_date=export_time(edited),
        message_thread_id=record.get('message_thread_id', record.get('topic_id')),
        media_group_id=record.get('grouped_id', record.get('media_group_id')),
        entities=_entities(record), caption_entities=[])
    metadata = telegram_metadata(message, is_edit=edited is not None)
    metadata['imported'] = True
    metadata['export_chat'] = chat.metadata
    # Keep unknown export fields too; no source message metadata is silently
    # dropped. Text lives in the message body and entities use body offsets.
    metadata['desktop'] = {key: value for key, value in record.items() if key not in {'text', 'text_entities'}}
    if record.get('reply_to_message_id') is not None:
        metadata['reply_to_source_id'] = str(record['reply_to_message_id'])
    if record.get('reply_to_peer_id') is not None:
        metadata['reply_to_peer_id'] = record['reply_to_peer_id']
        peer = _PEER.fullmatch(str(record['reply_to_peer_id']))
        if peer and str(int(peer.group(2))) == str(chat.metadata.get('id')):
            metadata['reply_to_source_chat_id'] = str(chat_id)
        elif peer:
            user, peer_chat = _actor(record['reply_to_peer_id'], None, None)
            metadata['reply_to_source_chat_id'] = str(user.id if user is not None else peer_chat.id)
        else:
            metadata['reply_to_source_chat_id'] = 'unknown'
    if record.get('forwarded_from_id') is not None or record.get('forwarded_from') is not None:
        forwarded_user, forwarded_chat = _actor(record.get('forwarded_from_id'), record.get('forwarded_from'), bot_user_id)
        metadata['forward_origin'] = telegram_actor(forwarded_user, forwarded_chat)
        if metadata['forward_origin']['actor_kind'] == 'unknown':
            metadata['forward_origin']['actor_name'] = str(record.get('forwarded_from') or 'Unknown sender')
        for key in ('forwarded_message_id', 'forwarded_date', 'saved_from'):
            if record.get(key) is not None:
                metadata['forward_origin'][key] = record[key]
    text = _text(record)
    parts = [MessagePart(PartKind.TEXT, text=text, remote_sync=False)]
    attachment = _attachment(record)
    if attachment:
        media_kind, filename, mime, visual_kind = attachment
        parts.append(MessagePart(visual_kind,
            text=_attachment_hint(record, media_kind, filename),
            filename=filename if visual_kind != PartKind.TEXT else None,
            mime_type=mime if visual_kind != PartKind.TEXT else None,
            origin='attachment_reference', remote_sync=False))
        metadata['media_availability'] = 'not_imported'
    if record.get('type') == 'service' and not text:
        parts.append(MessagePart(PartKind.TEXT, text=f'[Telegram service event: {record.get("action", "unknown")}]',
                                 origin='service_event', remote_sync=False))
    role = MessageRole.ASSISTANT if sender is not None and sender.is_bot else MessageRole.USER
    return ConversationMessage(role=role, parts=parts, name=metadata['actor_name'], metadata=metadata)


def _batches(records: Iterator[dict[str, Any]], options: ImportConfig) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    size = 0
    for record in records:
        record_size = len(json.dumps(record, ensure_ascii=False).encode('utf-8'))
        if record_size > options.max_record_bytes:
            # Commit any preceding valid records before reporting the rejected
            # source; increasing the configured limit and rerunning is safe.
            if batch:
                yield batch
            raise ValueError(f'Exported message {record.get("id", "unknown")} is {record_size} bytes; '
                             f'IMPORT_MAX_RECORD_BYTES={options.max_record_bytes}. '
                             'Increase that setting and rerun; no original text was truncated.')
        if batch and size + record_size > options.batch_bytes:
            yield batch
            batch, size = [], 0
        batch.append(record)
        size += record_size
        if len(batch) >= options.batch_messages or size >= options.batch_bytes:
            yield batch
            batch, size = [], 0
    if batch:
        yield batch


async def import_file(store: PostgresStore, path: Path, *, chat_id: int,
                      export_chat_id: str | None = None, bot_user_id: int | None = None,
                      defaults: SessionSettings | None = None,
                      options: ImportConfig | None = None,
                      telegram_config: TelegramConfig | None = None,
                      progress: Callable[[ImportResult], None] | None = None) -> ImportResult:
    options = options if options is not None else from_env(ImportConfig, 'IMPORT')
    session_id = f'telegram:{chat_id}'
    settings = await store.get_or_create_session(session_id, defaults or SessionSettings())
    scope = await store.get_scope(session_id)
    chat = inspect_export(path, export_chat_id)
    export_root = path.parent.resolve()
    processed = batches = 0
    for batch in _batches(iter_records(path, chat), options):
        source_ids = []
        for record in batch:
            message = desktop_message(record, chat, chat_id=chat_id, bot_user_id=bot_user_id,
                timezone=settings.metadata_timezone or DEFAULT_METADATA_TIMEZONE)
            if any(part.kind in {PartKind.IMAGE, PartKind.STICKER} for part in message.parts):
                if telegram_config is None:
                    from tgchatbot.config import load_config
                    telegram_config = load_config(require_telegram=False).telegram
                await asyncio.to_thread(_import_visual, message, record, export_root, telegram_config)
            stored = await _retained_import(store, session_id, message, scope)
            if stored is None:
                stored = await store.append_message(session_id, message, expected_scope=scope, generation_only=True)
            source_ids.append(stored.db_id)
            processed += 1
        # Append has already durably queued every original. Coalescing only
        # pending work reduces backfill overhead without a crash-loss window.
        await store.coalesce_memory_jobs(session_id, source_ids=source_ids, expected_scope=scope)
        batches += 1
        if progress is not None:
            progress(ImportResult(processed, batches, scope['generation']))
        await asyncio.sleep(0)
    return ImportResult(processed, batches, scope['generation'])


async def _run(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    from tgchatbot.config import load_config
    load_dotenv(Path.cwd() / '.env')
    config = load_config(require_telegram=False)
    options = from_env(ImportConfig, 'IMPORT')
    store = PostgresStore(config.database_url)
    token_id = config.telegram.token.split(':', 1)[0]
    try:
        await store.initialize()
        result = await import_file(store, args.file, chat_id=args.chat_id,
            export_chat_id=args.export_chat_id,
            bot_user_id=int(token_id) if token_id.isdigit() else None,
            defaults=config.default_session_settings(),
            telegram_config=config.telegram,
            options=options,
            progress=lambda status: print(f'Processed {status.messages} messages in {status.batches} bounded batches.', flush=True))
    finally:
        await store.close()
    print(f'Import complete: {result.messages} messages processed. New or changed messages are queued for memory processing.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path, required=True, help='Telegram Desktop JSON export')
    parser.add_argument('--chat-id', type=int, required=True, help='Destination Telegram chat ID used by the bot')
    parser.add_argument('--export-chat-id', help='Select one chat ID from a full chats.list export')
    args = parser.parse_args()
    if args.chat_id == 0 or not args.file.is_file():
        parser.error('--chat-id must be nonzero and --file must be an existing JSON file')
    try:
        asyncio.run(_run(args))
    except StaleScopeError:
        parser.exit(1, 'Import stopped because the chat was fully reset. Already imported messages remain in the previous audit epoch.\n')
    except (ValueError, ijson.JSONError) as exc:
        parser.exit(1, f'Import stopped: {exc}\n')


if __name__ == '__main__':
    main()
