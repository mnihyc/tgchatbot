"""Stream one Telegram Desktop JSON conversation through ordinary message intake.

Desktop's export uses bare peer IDs and mixed string/entity text. The explicit
destination chat owns the imported memory; user-only private archives keep their
message IDs separate from live Telegram IDs. Original export metadata is retained.
Import never invokes Telegram handlers, agent tools, or model interpretation.
Available ordinary files use the same remote upload path as live intake.
"""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass, fields, replace
import json
import mimetypes
from pathlib import Path
import re
import shutil
import tempfile
from types import SimpleNamespace
from typing import Any, Callable, Iterator
from datetime import datetime

import ijson
from ijson.common import ObjectBuilder

from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind, SessionSettings
from tgchatbot.config import TelegramConfig
from tgchatbot.domain.provenance import telegram_actor, telegram_metadata, utc_time
from tgchatbot.domain.timestamps import resolve_timezone
from tgchatbot.media.attachments import sync_attachment_parts
from tgchatbot.storage.artifacts import ArtifactStore
from tgchatbot.operational import from_env
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
    skipped: int = 0


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
                   if record.get('photo') or (kind not in {'video_message', 'video_note', 'voice_message', 'audio_file'}
                       and str(mime or '').startswith(('image/', 'video/'))) else PartKind.FILE)
    return str(kind), filename, mime, visual_kind


def _attachment_hint(record: dict[str, Any], kind: str, filename: str, *, available: bool = False,
                     thumbnail: bool = False) -> str:
    emoji = record.get('sticker_emoji') if kind == 'sticker' else None
    emoji_hint = f'; emoji={json.dumps(emoji, ensure_ascii=False)}' if emoji else ''
    state = 'image available' if available else 'media unavailable in bot workspace'
    if thumbnail:
        state += '; export thumbnail only (one still image; animation unavailable)'
    return f'[Imported attachment: {kind}{emoji_hint}; export_reference={json.dumps(filename, ensure_ascii=False)}; {state}]'


def _import_visual(message: ConversationMessage, record: dict[str, Any], export_root: Path,
                   config: TelegramConfig) -> None:
    attachment = _attachment(record)
    if attachment is None or attachment[3] not in {PartKind.IMAGE, PartKind.STICKER}:
        return
    kind, filename, mime, _ = attachment
    source = _export_attachment_path(record, export_root)
    photo, sticker = bool(record.get('photo')), kind == 'sticker'
    from tgchatbot.media.ingest import _build_visual_preview_parts, visual_parts_from_bytes
    parts = []
    if source is not None:
        try:
            if not photo and not sticker and (config.max_document_bytes <= 0
                    or source.stat().st_size > config.max_document_bytes):
                return
            raw = source.read_bytes()
            if photo or sticker:
                parts = visual_parts_from_bytes(raw=raw, filename=filename,
                    max_frames=1 if photo else config.max_sticker_frames,
                    max_bytes=config.max_photo_bytes if photo else config.max_sticker_bytes,
                    max_keyframe_candidates=config.max_video_keyframe_candidates,
                    detail='auto' if photo else 'low')
            else:
                parts = _build_visual_preview_parts(raw=raw, mime=mime or 'application/octet-stream',
                    filename=filename, telegram_config=config)
        except (OSError, ValueError):
            pass
    thumbnail = False
    if not parts and sticker:
        # Desktop supplies a raster thumbnail even for stickers whose original
        # format cannot be decoded here. It is preview evidence, not a replacement
        # original or a claim that the complete animation was observed.
        source = _export_media_path(record.get('thumbnail'), export_root)
        if source is not None:
            try:
                parts = visual_parts_from_bytes(raw=source.read_bytes(), filename=source.name,
                    max_frames=min(1, config.max_sticker_frames), max_bytes=config.max_sticker_bytes,
                    max_keyframe_candidates=config.max_video_keyframe_candidates, detail='low')
            except (OSError, ValueError):
                pass
            thumbnail = bool(parts)
    if not parts:
        return
    for index, part in enumerate(message.parts):
        if part.origin == 'attachment_reference':
            message.parts[index] = MessagePart(PartKind.TEXT,
                text=_attachment_hint(record, kind, filename, available=True, thumbnail=thumbnail),
                origin='attachment_reference', remote_sync=False)
            break
    message.parts.extend(parts)
    message.metadata['media_availability'] = 'imported'


def _export_attachment_path(record: dict[str, Any], export_root: Path) -> Path | None:
    return _export_media_path(record.get('photo') or record.get('file'), export_root)


def _export_media_path(reference: Any, export_root: Path) -> Path | None:
    if not isinstance(reference, str):
        return None
    try:
        source = (export_root / reference).resolve()
        # Export references own only files inside that bundle, including symlinks.
        return source if source.is_relative_to(export_root) and source.is_file() else None
    except (OSError, ValueError):
        return None


def _same_import_source(original: ConversationMessage, incoming: ConversationMessage) -> bool:
    return (original.metadata.get('imported') and original.role == incoming.role
        and original.name == incoming.name and original.parts[0].text == incoming.parts[0].text
        and all(original.metadata.get(key) == value for key, value in incoming.metadata.items()
                if key not in {'media_availability', 'source_revision'}))


async def _sync_import_attachment(message: ConversationMessage, record: dict[str, Any], *,
        session_id: str, export_root: Path, config: TelegramConfig,
        remote_workspace, artifact_store: ArtifactStore | None) -> None:
    attachment = _attachment(record)
    if attachment is None or record.get('photo') or attachment[0] == 'sticker':
        return
    _kind, filename, mime, _visual_kind = attachment
    source = _export_attachment_path(record, export_root)
    try:
        size_bytes = source.stat().st_size if source is not None else None
    except OSError:
        source, size_bytes = None, None
    descriptor = MessagePart(PartKind.FILE, filename=Path(filename).name,
        mime_type=mime or 'application/octet-stream', size_bytes=size_bytes, remote_sync=False,
        origin='attachment_reference')
    if source is None:
        descriptor.detail = 'remote copy unavailable: file was not included in this export'
    elif config.max_document_bytes <= 0:
        descriptor.detail = 'remote copy unavailable: attached file processing disabled by config'
    elif size_bytes > config.max_document_bytes:
        descriptor.detail = 'remote copy unavailable: file exceeds size limit'
    elif not remote_workspace or not remote_workspace.enabled:
        descriptor.detail = 'remote copy unavailable: SSH is disabled'
    else:
        if artifact_store is None:
            raise ValueError('Remote import needs the configured artifact store for temporary copies')
        # Only disposable copies reach the shared sync/cleanup owner. Remote
        # placement uses the source timestamp and original attachment filename.
        try:
            with tempfile.TemporaryDirectory(prefix='import-', dir=artifact_store.root) as temporary:
                staging = ArtifactStore(Path(temporary))
                staged = staging.save_bytes(chat_id=session_id, filename=filename, data=b'')
                await asyncio.to_thread(shutil.copyfile, source, staged)
                pending = replace(descriptor, artifact_path=str(staged), remote_sync=True,
                    size_bytes=staged.stat().st_size)
                synced = await sync_attachment_parts(session_id, [pending], remote_workspace,
                    sent_at=message.metadata.get('sent_at'))
            descriptor = next(part for part in synced if part.kind == PartKind.FILE)
            # Keep the shared live workflow's account of the upload.
            message.parts.extend(part for part in synced if part.kind == PartKind.TEXT)
        except OSError:
            descriptor.detail = 'remote copy unavailable: exported file could not be read'
    message.parts = [part for part in message.parts
        if not (part.origin == 'attachment_reference' and part.kind == PartKind.FILE)]
    message.parts.append(descriptor)
    if descriptor.remote_sync and message.metadata.get('media_availability') != 'imported':
        message.metadata['media_availability'] = 'synced'


async def _retained_import(store: PostgresStore, session_id: str, message: ConversationMessage, scope):
    """An unchanged export without its files cannot revoke retained visual evidence."""
    if not any(part.kind in {PartKind.IMAGE, PartKind.STICKER}
            and not (part.preview_ref or part.data_b64) for part in message.parts):
        return None
    metadata = message.metadata
    previous = await store.read_message_by_source(session_id,
        source=metadata['source'], source_chat_id=metadata['source_chat_id'],
        source_message_id=metadata['source_message_id'], expected_scope=scope, generation_only=True)
    if previous is None:
        return None
    original = previous.message
    if not _same_import_source(original, message):
        return None
    images = (await store.describe_message_images(session_id, [previous.db_id],
        expected_scope={'generation': scope['generation']})).get(previous.db_id, [])
    if not images or not all(image['available'] for image in images):
        return None
    # File synchronization and retained pixels have independent ownership:
    # preserve the original visual evidence while applying this attempt's file
    # availability, instead of returning an older upload descriptor wholesale.
    if any(part.kind == PartKind.FILE for part in message.parts):
        retained = [replace(part) for part in original.parts if part.preview_ref or part.data_b64
            or (part.origin == 'attachment_reference' and part.kind == PartKind.TEXT)]
        restored = []
        for part in message.parts:
            if part.kind in {PartKind.IMAGE, PartKind.STICKER} and not (part.preview_ref or part.data_b64):
                restored.extend(retained)
            else:
                restored.append(part)
        message.parts = restored
        message.metadata['media_availability'] = 'imported'
        return None
    # The no-op path needs the same full-reset boundary as an ordinary append.
    await store.assert_scope(session_id, scope, generation_only=True)
    return previous


def desktop_message(record: dict[str, Any], chat: ExportChat, *, chat_id: int,
                    bot_user_id: int | None = None,
                    timezone: str | None = None) -> ConversationMessage:
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
                value = value.replace(tzinfo=resolve_timezone(timezone))
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
                metadata['forward_origin'][key] = export_time(record[key]) if key == 'forwarded_date' else record[key]
    text = _text(record)
    parts = [MessagePart(PartKind.TEXT, text=text, remote_sync=False)]
    attachment = _attachment(record)
    if attachment:
        media_kind, filename, mime, visual_kind = attachment
        parts.append(MessagePart(visual_kind,
            text=_attachment_hint(record, media_kind, filename),
            filename=filename, mime_type=mime,
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
                      user_only: bool = False,
                      defaults: SessionSettings | None = None,
                      options: ImportConfig | None = None,
                      telegram_config: TelegramConfig | None = None,
                      remote_workspace=None, artifact_store: ArtifactStore | None = None,
                      timezone: str | None = None,
                      progress: Callable[[ImportResult], None] | None = None) -> ImportResult:
    options = options if options is not None else from_env(ImportConfig, 'IMPORT')
    if user_only and chat_id <= 0:
        raise ValueError('--user-only needs the positive Telegram user ID as --chat-id')
    chat = inspect_export(path, export_chat_id)
    if user_only and chat.metadata.get('id') is None:
        raise ValueError('--user-only needs the exported chat ID in the JSON header')
    session_id = f'telegram:{chat_id}'
    await store.get_or_create_session(session_id, defaults or SessionSettings())
    timezone = resolve_timezone(timezone).key
    scope = await store.get_scope(session_id)
    export_root = path.parent.resolve()
    processed = batches = skipped = 0

    def selected_records():
        nonlocal skipped
        for record in iter_records(path, chat):
            # Select the human by stable identity, never by display name or
            # the current token's bot identity. Skipped media does no work.
            if user_only and (record.get('type') != 'message'
                    or record.get('from_id') != f'user{chat_id}'):
                skipped += 1
                continue
            yield record

    for batch in _batches(selected_records(), options):
        source_ids = []
        for record in batch:
            message = desktop_message(record, chat, chat_id=chat_id,
                bot_user_id=None if user_only else bot_user_id,
                timezone=timezone)
            if user_only:
                # Private Desktop IDs belong to the exporting account, not the
                # bot's live message sequence. This stable archive key permits
                # reruns and multiple old bot chats without inventing live IDs.
                message.metadata['source'] = 'telegram_desktop'
                message.metadata['source_chat_id'] = f'user{chat_id}:peer{chat.metadata["id"]}'
                for key in ('telegram_message_id', 'reply_to_source_id',
                            'reply_to_source_chat_id', 'reply_to_peer_id'):
                    message.metadata.pop(key, None)
            previous = None
            attachment = _attachment(record)
            if attachment is not None:
                if telegram_config is None:
                    from tgchatbot.config import load_config
                    telegram_config = load_config(require_telegram=False).telegram
                is_file = not record.get('photo') and attachment[0] != 'sticker'
                if is_file:
                    previous = await store.read_message_by_source(session_id, source=message.metadata['source'],
                        source_chat_id=message.metadata['source_chat_id'], source_message_id=message.metadata['source_message_id'],
                        expected_scope=scope, generation_only=True)
                    if remote_workspace and remote_workspace.enabled and (previous is None
                            or not _same_import_source(previous.message, message)):
                        # Commit source identity/text before any remote transfer.
                        await store.append_message(session_id, message,
                            expected_scope=scope, generation_only=True)
                await asyncio.to_thread(_import_visual, message, record, export_root, telegram_config)
                await _sync_import_attachment(message, record, session_id=session_id, export_root=export_root,
                    config=telegram_config, remote_workspace=remote_workspace,
                    artifact_store=artifact_store)
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
            progress(ImportResult(processed, batches, scope['generation'], skipped))
        await asyncio.sleep(0)
    return ImportResult(processed, batches, scope['generation'], skipped)


async def _run(args: argparse.Namespace) -> None:
    from dotenv import load_dotenv
    from tgchatbot.config import load_config
    load_dotenv(Path.cwd() / '.env')
    config = load_config(require_telegram=False)
    options = from_env(ImportConfig, 'IMPORT')
    store = PostgresStore(config.database_url)
    from tgchatbot.tools.remote_workspace import RemoteWorkspaceClient
    remote_workspace = RemoteWorkspaceClient(config) if config.ssh_exec.enabled and config.ssh_exec.host else None
    artifact_store = ArtifactStore(config.artifact_dir) if remote_workspace else None
    token_id = config.telegram.token.split(':', 1)[0]
    try:
        await store.initialize()
        result = await import_file(store, args.file, chat_id=args.chat_id,
            export_chat_id=args.export_chat_id,
            bot_user_id=int(token_id) if token_id.isdigit() else None,
            user_only=args.user_only,
            defaults=config.default_session_settings(),
            telegram_config=config.telegram, remote_workspace=remote_workspace, artifact_store=artifact_store,
            timezone=config.default_metadata_timezone,
            options=options,
            progress=lambda status: print(f'Processed {status.messages} messages in {status.batches} bounded batches.', flush=True))
    finally:
        if remote_workspace is not None:
            await remote_workspace.aclose()
        await store.close()
    skipped = f' {result.skipped} other messages skipped.' if args.user_only else ''
    print(f'Import complete: {result.messages} messages processed.{skipped} New or changed messages are queued for memory processing.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', type=Path, required=True, help='Telegram Desktop JSON export')
    parser.add_argument('--chat-id', type=int, required=True, help='Destination Telegram chat ID used by the bot')
    parser.add_argument('--export-chat-id', help='Select one chat ID from a full chats.list export')
    parser.add_argument('--user-only', action='store_true',
        help='Keep only messages sent by the user identified by --chat-id; skip bot replies and old Telegram reply links')
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
