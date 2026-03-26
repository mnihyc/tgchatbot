#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from tgchatbot.core.token_estimator import TokenEstimator
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind

AUTO_NOTE_PREFIX = '[Auto-generated bot message, do not reply.]'


@dataclass(frozen=True)
class MessageRow:
    db_id: int
    session_id: str
    role: str
    name: str | None
    message: ConversationMessage
    estimated_tokens: int
    compacted: int
    compacted_level: int | None
    compacted_by_block_id: int | None
    hidden: int
    created_at: str | None


@dataclass(frozen=True)
class SessionPlan:
    session_id: str
    merged_rows: list[MessageRow]
    deleted_note_ids: list[int]
    merged_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Merge legacy auto-note rows into real user-message rows.')
    parser.add_argument('--db', required=True, help='Path to tgchatbot.sqlite3')
    parser.add_argument('--session-id', action='append', help='Only migrate the given session id. Repeatable.')
    parser.add_argument('--dry-run', action='store_true', help='Analyze only; do not modify the database.')
    return parser.parse_args()


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    return conn


def _deserialize_part(part: dict[str, Any]) -> MessagePart:
    payload = dict(part)
    kind = payload.pop('kind')
    return MessagePart(kind=PartKind(kind), **payload)


def decode_message_row(row: sqlite3.Row) -> MessageRow:
    payload = json.loads(str(row['payload_json'] or '{}'))
    parts = [_deserialize_part(part) for part in payload.get('parts', [])]
    message = ConversationMessage(
        role=MessageRole(str(row['role'])),
        name=row['name'],
        parts=parts,
        metadata=payload.get('metadata', {}) if isinstance(payload.get('metadata', {}), dict) else {},
    )
    return MessageRow(
        db_id=int(row['id']),
        session_id=str(row['session_id']),
        role=str(row['role']),
        name=row['name'],
        message=message,
        estimated_tokens=int(row['estimated_tokens'] or 0),
        compacted=int(row['compacted'] or 0),
        compacted_level=(int(row['compacted_level']) if row['compacted_level'] is not None else None),
        compacted_by_block_id=(int(row['compacted_by_block_id']) if row['compacted_by_block_id'] is not None else None),
        hidden=int(row['hidden'] or 0),
        created_at=(str(row['created_at']) if row['created_at'] is not None else None),
    )


def encode_message(message: ConversationMessage) -> str:
    payload = {
        'parts': [serialize_part(part) for part in message.parts],
        'metadata': message.metadata,
    }
    return json.dumps(payload, ensure_ascii=False)


def serialize_part(part: MessagePart) -> dict[str, Any]:
    data = {
        'kind': part.kind.value,
        'text': part.text,
        'mime_type': part.mime_type,
        'filename': part.filename,
        'data_b64': part.data_b64,
        'artifact_path': part.artifact_path,
        'size_bytes': part.size_bytes,
        'detail': part.detail,
        'remote_sync': part.remote_sync,
        'origin': part.origin,
    }
    return {key: value for key, value in data.items() if value is not None}


def is_auto_note_row(row: MessageRow) -> bool:
    metadata = row.message.metadata if isinstance(row.message.metadata, dict) else {}
    return row.message.role == MessageRole.USER and str(metadata.get('synthetic_role') or '').strip().lower() == 'auto_user_note'


def has_embedded_auto_notes(message: ConversationMessage) -> bool:
    return any((part.origin or '').strip().lower() == 'auto_note' for part in message.parts)


def _message_id(metadata: dict[str, Any]) -> int | None:
    value = metadata.get('telegram_message_id')
    try:
        return int(value) if value is not None else None
    except Exception:
        return None


def can_merge_auto_notes(note_rows: list[MessageRow], target_row: MessageRow) -> bool:
    if not note_rows:
        return False
    if target_row.message.role != MessageRole.USER or is_auto_note_row(target_row) or has_embedded_auto_notes(target_row.message):
        return False
    target_hidden = int(target_row.hidden or 0)
    target_meta = target_row.message.metadata if isinstance(target_row.message.metadata, dict) else {}
    target_message_id = _message_id(target_meta)
    previous_note_id: int | None = None
    for note_row in note_rows:
        if not is_auto_note_row(note_row):
            return False
        if int(note_row.hidden or 0) != target_hidden:
            return False
        if previous_note_id is not None and note_row.db_id <= previous_note_id:
            return False
        if note_row.db_id >= target_row.db_id:
            return False
        previous_note_id = note_row.db_id
        note_meta = note_row.message.metadata if isinstance(note_row.message.metadata, dict) else {}
        note_message_id = _message_id(note_meta)
        if note_message_id is not None or target_message_id is not None:
            if note_message_id is None or target_message_id is None or note_message_id != target_message_id:
                return False
            continue
        source = str(note_meta.get('source') or '').strip().lower()
        if source not in {'telegram_ingest', 'legacy_uid_json'}:
            return False
    return True


def cleaned_auto_note_parts(message: ConversationMessage) -> list[MessagePart]:
    cleaned: list[MessagePart] = []
    for part in message.parts:
        if (part.origin or '').strip().lower() != 'auto_note':
            continue
        if part.kind == PartKind.TEXT and (part.text or '').strip() == AUTO_NOTE_PREFIX:
            continue
        cleaned.append(
            MessagePart(
                kind=part.kind,
                text=part.text,
                mime_type=part.mime_type,
                filename=part.filename,
                data_b64=part.data_b64,
                artifact_path=part.artifact_path,
                size_bytes=part.size_bytes,
                detail=part.detail,
                remote_sync=part.remote_sync,
                origin='auto_note',
            )
        )
    return cleaned


def merge_auto_note_rows(note_rows: list[MessageRow], target_row: MessageRow) -> MessageRow:
    merged_prefix: list[MessagePart] = []
    for note_row in note_rows:
        merged_prefix.extend(cleaned_auto_note_parts(note_row.message))
    merged_message = ConversationMessage(
        role=target_row.message.role,
        name=target_row.message.name,
        parts=[*merged_prefix, *target_row.message.parts],
        metadata=dict(target_row.message.metadata),
    )
    return replace(
        target_row,
        message=merged_message,
        estimated_tokens=TokenEstimator.estimate_message(merged_message),
    )


def plan_session_rows(rows: list[MessageRow]) -> SessionPlan:
    merged_rows: list[MessageRow] = []
    deleted_note_ids: list[int] = []
    pending_notes: list[MessageRow] = []

    for row in rows:
        if is_auto_note_row(row):
            pending_notes.append(row)
            continue
        if pending_notes and can_merge_auto_notes(pending_notes, row):
            merged_rows.append(merge_auto_note_rows(pending_notes, row))
            deleted_note_ids.extend(note.db_id for note in pending_notes)
            pending_notes = []
            continue
        if pending_notes:
            merged_rows.extend(pending_notes)
            pending_notes = []
        merged_rows.append(row)

    if pending_notes:
        merged_rows.extend(pending_notes)

    return SessionPlan(
        session_id=rows[0].session_id if rows else '',
        merged_rows=merged_rows,
        deleted_note_ids=deleted_note_ids,
        merged_count=len(deleted_note_ids),
    )


def load_session_rows(conn: sqlite3.Connection, session_id: str) -> list[MessageRow]:
    rows = conn.execute(
        """
        SELECT id, session_id, role, name, payload_json, estimated_tokens,
               compacted, compacted_level, compacted_by_block_id, hidden, created_at
        FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
        """,
        (session_id,),
    ).fetchall()
    return [decode_message_row(row) for row in rows]


def apply_session_plan(conn: sqlite3.Connection, plan: SessionPlan) -> None:
    if plan.merged_count <= 0:
        return
    for row in plan.merged_rows:
        conn.execute(
            "UPDATE messages SET role = ?, name = ?, payload_json = ?, estimated_tokens = ?, hidden = ? WHERE id = ? AND session_id = ?",
            (
                row.message.role.value,
                row.message.name,
                encode_message(row.message),
                row.estimated_tokens,
                int(row.hidden or 0),
                row.db_id,
                row.session_id,
            ),
        )
    if plan.deleted_note_ids:
        placeholders = ','.join('?' for _ in plan.deleted_note_ids)
        conn.execute(
            f"DELETE FROM messages WHERE session_id = ? AND id IN ({placeholders})",
            (plan.session_id, *plan.deleted_note_ids),
        )
    conn.execute(
        "UPDATE messages SET compacted = 0, compacted_level = NULL, compacted_by_block_id = NULL WHERE session_id = ?",
        (plan.session_id,),
    )
    conn.execute("DELETE FROM compaction_blocks WHERE session_id = ?", (plan.session_id,))


def migrate(db_path: Path, *, session_ids: list[str] | None = None, dry_run: bool = False) -> dict[str, int]:
    stats = {'sessions_scanned': 0, 'sessions_changed': 0, 'auto_note_rows_deleted': 0}
    with _connect(db_path) as conn:
        if session_ids:
            ordered_session_ids = list(dict.fromkeys(session_ids))
        else:
            ordered_session_ids = [str(row['session_id']) for row in conn.execute("SELECT DISTINCT session_id FROM messages ORDER BY session_id ASC").fetchall()]
        plans: list[SessionPlan] = []
        for session_id in ordered_session_ids:
            rows = load_session_rows(conn, session_id)
            if not rows:
                continue
            stats['sessions_scanned'] += 1
            plan = plan_session_rows(rows)
            if plan.merged_count > 0:
                plans.append(plan)
                stats['sessions_changed'] += 1
                stats['auto_note_rows_deleted'] += plan.merged_count
        if dry_run:
            return stats
        with conn:
            for plan in plans:
                apply_session_plan(conn, plan)
    return stats


def main() -> int:
    args = parse_args()
    stats = migrate(Path(args.db), session_ids=args.session_id, dry_run=args.dry_run)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
