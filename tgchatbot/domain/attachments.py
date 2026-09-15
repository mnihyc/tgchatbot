"""Attachment presentation; transport paths and retained pixels stay with their owners."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
import json
from typing import Any

from tgchatbot.domain.models import MessagePart, PartKind


def generated_attachment_reference(part: MessagePart | Mapping[str, Any]) -> str:
    """Canonical fallback text; its exact format also identifies synthesized spans."""
    if isinstance(part, MessagePart):
        attributes = {'kind': part.kind.value, 'filename': part.filename, 'mime': part.mime_type,
            'size': part.size_bytes, 'location': part.artifact_path, 'description': part.detail}
    else:
        attributes = {'kind': part.get('kind'), 'filename': part.get('filename'), 'mime': part.get('mime_type'),
            'size': part.get('size_bytes'), 'location': part.get('artifact_path'), 'description': part.get('detail')}
    return '[Attachment reference: ' + ', '.join(f'{key}={value}' for key, value in attributes.items()
        if value is not None) + ']'


def compact_memory_image_evidence(parts: list[MessagePart]) -> list[MessagePart]:
    """Avoid repeating database-generated image bookkeeping beside its owning label."""
    labeled = set()
    prefix = '[Original image evidence: '
    for part in parts:
        if (part.kind == PartKind.TEXT and (part.origin or '').startswith('memory_image:')
                and (part.text or '').startswith(prefix) and part.text.endswith(']')):
            try:
                label = json.loads(part.text[len(prefix):-1])
            except json.JSONDecodeError:
                continue
            if isinstance(label, dict) and part.origin == f"memory_image:{label.get('image_id')}":
                labeled.add(part.origin)
    return [replace(part, text=None) if (
        part.kind == PartKind.IMAGE and part.data_b64 and part.origin in labeled
        and part.text == generated_attachment_reference(part)
        and part.detail in (None, 'auto', 'low', 'high')
        and not (part.filename or part.artifact_path or part.workspace_path)
    ) else part for part in parts]


def attachment_description(part: MessagePart, *, presentation_version: int = 2) -> str:
    """Describe the original file once, independently of any retained image preview."""
    descriptor = f"[Attached file: {part.filename or 'file'}"
    if part.mime_type:
        descriptor += f', {part.mime_type}'
    if part.size_bytes is not None:
        descriptor += f', {part.size_bytes} bytes'
    if presentation_version < 2:
        if part.artifact_path:
            descriptor += f', remote_path={part.artifact_path}'
    else:
        if part.workspace_path:
            descriptor += f', workspace_path={part.workspace_path}'
        elif part.artifact_path:
            # Legacy records may not know their workspace root. Preserve the
            # usable path rather than guessing a relative path from its shape.
            descriptor += f', remote_path={part.artifact_path}'
        if part.detail:
            descriptor += f', {part.detail}'
    return descriptor + ']'
