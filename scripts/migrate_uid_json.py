#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

TIMESTAMP_RE = re.compile(r"\((\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\)")
LEGACY_CHAT_METADATA_RE = re.compile(
    r"^\s*\[(?P<label>[^\]]+)\]\s*\((?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\):(?P<remainder>[\s\S]*)$"
)
HANDLE_LIKE_RE = re.compile(r"^[A-Za-z0-9_]{1,64}$")
UTC = ZoneInfo("UTC")


@dataclass(frozen=True)
class LegacyTransportMetadata:
    username: str
    nickname: str
    local_time: str
    created_at_utc: str


@dataclass(frozen=True)
class PreparedMessage:
    role: str
    payload_parts: list[dict[str, Any]]
    metadata: dict[str, Any]
    created_at: str | None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Import legacy UID.json chat history into tgchatbot SQLite schema")
    p.add_argument("--project-root", required=True, help="Path to the new tgchatbot project root")
    p.add_argument("--json", required=True, help="Path to legacy UID.json")
    p.add_argument("--db", required=True, help="Path to target tgchatbot.sqlite3")
    p.add_argument("--session-id", help="Target session id (default: telegram:<UID stem>)")
    p.add_argument("--provider", default="openai")
    p.add_argument("--model", default="gpt-5.4-nano")
    p.add_argument("--mode", default="agent", choices=["chat", "assist", "agent"])
    p.add_argument("--process-visibility", default="full", choices=["off", "minimal", "status", "verbose", "full"])
    p.add_argument("--response-delivery", default="final_new", choices=["edit", "final_new"])
    p.add_argument("--metadata-timezone", default="Asia/Shanghai")
    p.add_argument("--prompt-injection-mode", default="augment", choices=["exact", "augment"])
    p.add_argument("--tool-history-mode", default="native_same_provider", choices=["translated", "native_same_provider"])
    p.add_argument("--replace-session", action="store_true", help="Delete existing messages/blocks for this session before import")
    p.add_argument("--append", action="store_true", help="Append into an existing session instead of replacing/aborting")
    return p.parse_args()


def text_from_legacy_part(part: Any) -> str | None:
    return part if isinstance(part, str) else None


def resolve_metadata_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return UTC


def convert_legacy_timestamp(timestamp_text: str, metadata_timezone: ZoneInfo) -> tuple[str, str]:
    naive_dt = datetime.strptime(timestamp_text, "%Y-%m-%d %H:%M:%S")
    local_dt = naive_dt.replace(tzinfo=metadata_timezone)
    return (
        local_dt.isoformat(timespec="seconds"),
        local_dt.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S"),
    )


def normalize_transport_identity(label: str) -> tuple[str, str]:
    cleaned = re.sub(r"\s+", " ", (label or "")).strip()
    nickname = cleaned.replace('"', "'")
    handle = cleaned[1:] if cleaned.startswith("@") else cleaned
    username = f"@{handle}" if HANDLE_LIKE_RE.fullmatch(handle) else "-"
    return username, nickname


def normalize_legacy_remainder(remainder: str) -> str:
    if not remainder:
        return ""
    if remainder[:1] in {" ", "\t", "\r", "\n"}:
        return remainder.lstrip(" \t\r\n")
    return remainder


def extract_legacy_transport_metadata(parts: list[Any], metadata_timezone: ZoneInfo) -> tuple[list[Any], LegacyTransportMetadata | None]:
    if not parts:
        return list(parts), None

    first_text = text_from_legacy_part(parts[0])
    if not first_text:
        return list(parts), None

    match = LEGACY_CHAT_METADATA_RE.match(first_text)
    if not match:
        return list(parts), None

    username, nickname = normalize_transport_identity(match.group("label"))
    local_time, created_at_utc = convert_legacy_timestamp(match.group("timestamp"), metadata_timezone)
    remainder = normalize_legacy_remainder(match.group("remainder"))

    remaining_parts = list(parts[1:])
    if remainder:
        remaining_parts.insert(0, remainder)

    return remaining_parts, LegacyTransportMetadata(
        username=username,
        nickname=nickname,
        local_time=local_time,
        created_at_utc=created_at_utc,
    )


def build_transport_metadata_dict(metadata: LegacyTransportMetadata) -> dict[str, str]:
    return {
        "username": metadata.username,
        "nickname": metadata.nickname,
        "local_time": metadata.local_time,
        "created_at_utc": metadata.created_at_utc,
    }


def build_auto_note_payload_parts(metadata: LegacyTransportMetadata) -> list[dict[str, Any]]:
    return [
        {
            "kind": "text",
            "text": "[Auto-generated bot message, do not reply.]",
            "remote_sync": False,
            "origin": "auto_note",
        },
        {
            "kind": "text",
            "text": f'[Message metadata: username={metadata.username} nickname="{metadata.nickname}" time={metadata.local_time}]',
            "remote_sync": False,
            "origin": "auto_note",
        },
    ]


def parse_created_at(parts: list[Any], metadata_timezone: ZoneInfo) -> str | None:
    for part in parts:
        text = text_from_legacy_part(part)
        if not text:
            continue
        match = TIMESTAMP_RE.search(text)
        if match:
            return convert_legacy_timestamp(match.group(1), metadata_timezone)[1]
    return None


def normalize_role(old_role: str) -> str:
    old = (old_role or "").strip().lower()
    if old == "model":
        return "assistant"
    if old in {"system", "user", "assistant", "tool"}:
        return old
    return "user"


def infer_session_id(json_path: Path) -> str:
    return f"telegram:{json_path.stem}"


def build_system_prompt(parts: list[Any]) -> str:
    texts = [p for p in parts if isinstance(p, str) and p.strip()]
    prompt = "\n".join(texts).strip()
    if prompt.startswith("[System prompt]:"):
        prompt = prompt[len("[System prompt]:"):].lstrip()
    return prompt


def estimate_text(text: str | None) -> int:
    import math
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4.0))


def estimate_parts(parts: list[dict[str, Any]]) -> int:
    total = 12
    for part in parts:
        total += 6
        kind = part.get("kind")
        if kind == "text":
            total += estimate_text(part.get("text"))
        elif kind == "image":
            total += 900 + estimate_text(part.get("filename"))
        elif kind == "sticker":
            total += 200 + estimate_text(part.get("filename"))
        elif kind == "file":
            desc = " ".join(
                x for x in [
                    part.get("filename"),
                    part.get("mime_type"),
                    str(part.get("size_bytes")) if part.get("size_bytes") is not None else None,
                ] if x
            )
            total += 48 + estimate_text(desc)
        else:
            total += estimate_text(part.get("text"))
    return total


def build_payload_parts(parts: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for part in parts:
        if isinstance(part, str):
            out.append({"kind": "text", "text": part})
            continue
        if isinstance(part, dict):
            mime = str(part.get("mime_type") or "").strip()
            data = part.get("data")
            if mime.startswith("image/") and isinstance(data, str) and data:
                out.append({"kind": "image", "mime_type": mime, "data_b64": data})
                continue
            filename = part.get("filename")
            size_bytes = part.get("size_bytes")
            payload: dict[str, Any] = {"kind": "file"}
            if mime:
                payload["mime_type"] = mime
            if isinstance(filename, str) and filename:
                payload["filename"] = filename
            if isinstance(size_bytes, int):
                payload["size_bytes"] = size_bytes
            if isinstance(data, str) and data:
                payload["text"] = f"[Legacy attachment omitted inline: {filename or mime or 'file'}]"
            out.append(payload)
    return out


def prepare_migrated_messages(*, role: str, parts: list[Any], metadata_timezone: ZoneInfo) -> list[PreparedMessage]:
    normalized_role = normalize_role(role)
    source_parts = list(parts)
    legacy_transport = None

    if normalized_role == "user":
        source_parts, legacy_transport = extract_legacy_transport_metadata(source_parts, metadata_timezone)

    created_at = legacy_transport.created_at_utc if legacy_transport else parse_created_at(parts, metadata_timezone)
    transport_metadata = build_transport_metadata_dict(legacy_transport) if legacy_transport else None

    prepared: list[PreparedMessage] = []
    if legacy_transport:
        prepared.append(
            PreparedMessage(
                role="user",
                payload_parts=build_auto_note_payload_parts(legacy_transport),
                metadata={
                    "synthetic_role": "auto_user_note",
                    "source": "legacy_uid_json",
                    "transport_metadata": transport_metadata,
                },
                created_at=legacy_transport.created_at_utc,
            )
        )

    payload_parts = build_payload_parts(source_parts)
    if not payload_parts and legacy_transport and normalized_role == "user":
        payload_parts = [{"kind": "text", "text": ""}]
    if payload_parts:
        message_metadata: dict[str, Any] = {"source": "legacy_uid_json"}
        if transport_metadata is not None:
            message_metadata["transport_metadata"] = transport_metadata
        prepared.append(
            PreparedMessage(
                role=normalized_role,
                payload_parts=payload_parts,
                metadata=message_metadata,
                created_at=created_at,
            )
        )

    return prepared


def main() -> int:
    args = parse_args()
    project_root = Path(args.project_root).expanduser().resolve()
    json_path = Path(args.json).expanduser().resolve()
    db_path = Path(args.db).expanduser().resolve()
    if not project_root.exists():
        raise SystemExit(f"project root not found: {project_root}")
    if not json_path.exists():
        raise SystemExit(f"json file not found: {json_path}")
    metadata_timezone = resolve_metadata_timezone(args.metadata_timezone)

    sys.path.insert(0, str(project_root))
    from tgchatbot.storage.sqlite_store import SQLiteStore  # type: ignore
    from tgchatbot.domain.models import (  # type: ignore
        ChatMode,
        ProcessVisibility,
        PromptInjectionMode,
        ResponseDelivery,
        SessionSettings,
        StickerMode,
        ToolHistoryMode,
    )

    with json_path.open("r", encoding="utf-8") as f:
        legacy = json.load(f)
    if not isinstance(legacy, list):
        raise SystemExit("legacy json must be a top-level list")

    session_id = args.session_id or infer_session_id(json_path)
    store = SQLiteStore(db_path)

    system_prompt = None
    remaining: list[dict[str, Any]] = []
    for idx, item in enumerate(legacy):
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "")
        parts = item.get("parts") if isinstance(item.get("parts"), list) else []
        if idx == 0 and role == "system":
            candidate = build_system_prompt(parts)
            system_prompt = candidate or None
            continue
        remaining.append({"role": role, "parts": parts})

    settings = SessionSettings(
        provider=args.provider,
        model=args.model,
        mode=ChatMode(args.mode),
        process_visibility=ProcessVisibility(args.process_visibility),
        response_delivery=ResponseDelivery(args.response_delivery),
        sticker_mode=StickerMode.OFF,
        prompt_injection_mode=PromptInjectionMode(args.prompt_injection_mode),
        tool_history_mode=ToolHistoryMode(args.tool_history_mode),
        metadata_timezone=args.metadata_timezone,
        system_prompt=system_prompt or SessionSettings().system_prompt,
    )
    store._save_session_sync(session_id, settings)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute("SELECT COUNT(*) AS c FROM messages WHERE session_id = ? AND COALESCE(hidden, 0) = 0", (session_id,)).fetchone()
        existing = int(row["c"] or 0)
        if existing and not args.replace_session and not args.append:
            raise SystemExit(
                f"session {session_id!r} already has {existing} visible messages; use --replace-session or --append"
            )
        inserted = 0
        metadata_notes_imported = 0
        message_rows_inserted = 0
        image_parts = 0
        with conn:
            if args.replace_session:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                conn.execute("DELETE FROM compaction_blocks WHERE session_id = ?", (session_id,))

            for item in remaining:
                raw_parts = item["parts"]
                prepared_messages = prepare_migrated_messages(
                    role=str(item["role"]),
                    parts=raw_parts,
                    metadata_timezone=metadata_timezone,
                )
                if not prepared_messages:
                    continue

                for prepared in prepared_messages:
                    payload_json = json.dumps({"parts": prepared.payload_parts, "metadata": prepared.metadata}, ensure_ascii=False)
                    estimated_tokens = estimate_parts(prepared.payload_parts)
                    if prepared.created_at:
                        conn.execute(
                            "INSERT INTO messages (session_id, role, name, payload_json, estimated_tokens, compacted, compacted_level, compacted_by_block_id, created_at) VALUES (?, ?, NULL, ?, ?, 0, NULL, NULL, ?)",
                            (session_id, prepared.role, payload_json, estimated_tokens, prepared.created_at),
                        )
                    else:
                        conn.execute(
                            "INSERT INTO messages (session_id, role, name, payload_json, estimated_tokens, compacted, compacted_level, compacted_by_block_id) VALUES (?, ?, NULL, ?, ?, 0, NULL, NULL)",
                            (session_id, prepared.role, payload_json, estimated_tokens),
                        )
                    if prepared.metadata.get("synthetic_role") == "auto_user_note":
                        metadata_notes_imported += 1
                    else:
                        inserted += 1
                        image_parts += sum(1 for part in prepared.payload_parts if part.get("kind") == "image")
                    message_rows_inserted += 1
    finally:
        conn.close()

    print(json.dumps({
        "session_id": session_id,
        "messages_imported": inserted,
        "metadata_notes_imported": metadata_notes_imported,
        "message_rows_inserted": message_rows_inserted,
        "image_parts_imported": image_parts,
        "system_prompt_imported": bool(system_prompt),
        "db": str(db_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

