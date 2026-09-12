"""Remote upload and disposal of temporary attachment copies, shared by intake paths."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import logging
from pathlib import Path
import posixpath

from tgchatbot.domain.models import MessagePart, PartKind
from tgchatbot.logging_config import clip_for_log

logger = logging.getLogger(__name__)


async def sync_attachment_parts(session_id: str, parts: list[MessagePart], remote_workspace, *,
                                sent_at: datetime | str | None = None) -> list[MessagePart]:
    syncable_parts = [part for part in parts if part.artifact_path and part.remote_sync]
    local_paths = [part.artifact_path for part in syncable_parts if part.artifact_path]
    if not local_paths:
        return parts
    if not remote_workspace or not remote_workspace.enabled:
        local_path_objs = tuple(Path(value) for value in local_paths)
        for path in local_path_objs:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                logger.warning('attachment.sync.cleanup_failed sid=%s file=%s', clip_for_log(session_id, limit=48), path.name)
        return [replace(part, artifact_path=None, remote_sync=False,
                        detail=((part.detail + '; ') if part.detail else '') + 'remote copy unavailable: SSH is disabled')
                if part.artifact_path and part.remote_sync else part for part in parts]
    local_path_objs = tuple(Path(value) for value in local_paths)
    filenames = {str(Path(part.artifact_path).resolve()): part.filename
                 for part in syncable_parts if part.filename}
    paths_by_source: dict[str, str] = {}
    try:
        sync_result = await remote_workspace.sync_inputs(session_id, local_path_objs,
            sent_at=sent_at, filenames=filenames)
        paths_by_source = sync_result.paths_by_source
        logger.info('attachment.sync.ok sid=%s requested=%s synced=%s', clip_for_log(session_id, limit=48), len(local_path_objs), len(paths_by_source))
    except Exception as exc:
        logger.exception('attachment.sync.failed sid=%s files=%s err=%s', clip_for_log(session_id, limit=48), len(local_path_objs), exc.__class__.__name__)
    finally:
        for path in local_path_objs:
            try:
                path.unlink(missing_ok=True)
            except Exception:
                logger.warning('attachment.sync.cleanup_failed sid=%s file=%s', clip_for_log(session_id, limit=48), path.name)
    updated_parts: list[MessagePart] = []
    synced_entries: list[str] = []
    for part in parts:
        if not (part.artifact_path and part.remote_sync):
            updated_parts.append(part)
            continue
        local_key = str(Path(part.artifact_path).resolve())
        remote_path = paths_by_source.get(local_key)
        if remote_path:
            shown_path = posixpath.relpath(remote_path, remote_workspace.session_paths(session_id).root)
            synced_entries.append(f"{part.filename or Path(remote_path).name} -> {shown_path}")
            updated_parts.append(replace(part, artifact_path=remote_path, remote_sync=True))
            continue
        filename = part.filename or 'file'
        updated_parts.append(replace(part, artifact_path=None, remote_sync=False,
            detail=((part.detail + '; ') if part.detail else '') + 'remote copy unavailable: upload failed'))
        updated_parts.append(
            MessagePart(
                kind=PartKind.TEXT,
                text=f'[Attachment sync failed for remote use: {filename}]',
                remote_sync=False,
                origin='auto_note',
            )
        )
    note_parts: list[MessagePart] = []
    if synced_entries:
        note_parts.append(MessagePart(kind=PartKind.TEXT, text='[Attachment synced to remote for tool use (paths relative to workspace): ' + '; '.join(synced_entries) + ']', remote_sync=False, origin='auto_note'))
    return [*note_parts, *updated_parts]
