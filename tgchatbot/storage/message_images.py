"""Source-scoped image occurrences backed by retained compressed DB evidence."""
from __future__ import annotations

import base64
import json

from tgchatbot.domain.models import MessagePart, MessageRole, PartKind
from tgchatbot.domain.provenance import attribution


def _eligible(message):
    if message.metadata.get('synthetic_role'):
        return False
    return not (message.role == MessageRole.TOOL
        and message.name in {'memory_search', 'memory_read', 'user_profile_fetch'}
        and message.metadata.get('tool_phase') in ('call', 'result'))


def _occurrences(row):
    parts = row.message.parts
    has_frames = any(part.kind == PartKind.IMAGE for part in parts)
    for index, part in enumerate(parts):
        if part.kind not in {PartKind.IMAGE, PartKind.STICKER}:
            continue
        if (part.origin or '').startswith('memory_image:'):
            continue
        # Live stickers keep a text/emoji hint beside their decoded frames.
        # Imported unavailable attachment placeholders remain real occurrences.
        if (part.kind == PartKind.STICKER and not part.preview_ref and not part.data_b64
                and part.origin != 'attachment_reference' and has_frames):
            continue
        image_id = f'img:{row.db_id}:{row.message.metadata["source_revision"]}:{index}'
        yield image_id, part


async def _originals(store, conn, session_id, message_ids, expected_scope):
    # Share the source owner's session lock and canonical read path. Permission
    # remains stable until any selected payloads have been read in this transaction.
    # Operator connections cannot lock rows; pin originals and pixels to the same
    # snapshot and recheck reset scope before returning the completed read.
    scope = await store.read_session_scope(conn, session_id)
    if scope is None:
        return []
    store._check_scope(scope, expected_scope)
    rows = await store._read_ids(conn, session_id, message_ids)
    return [row for row in rows if _eligible(row.message)]


async def describe_message_images(store, session_id, message_ids, *, expected_scope=None):
    async with store.pool.connection() as conn:
        rows = await _originals(store, conn, session_id, message_ids, expected_scope)
        occurrences = {row.db_id: list(_occurrences(row)) for row in rows}
        references = {part.preview_ref for items in occurrences.values() for _image_id, part in items if part.preview_ref}
        available = set()
        if references:
            # Descriptor lookup never reads or decodes all matched image bytes.
            found = await (await conn.execute('SELECT reference FROM message_previews '
                'WHERE session_id=%s AND reference=ANY(%s)', (session_id, list(references)))).fetchall()
            available = {item['reference'] for item in found}
        result = {message_id: [{'image_id': image_id, 'available': part.preview_ref in available,
            **({'mime_type': part.mime_type} if part.mime_type else {})} for image_id, part in items]
            for message_id, items in occurrences.items()}
    if store._read_only and expected_scope is not None:
        await store.assert_scope(session_id, expected_scope)
    return result


async def resolve_message_images(store, session_id, message_ids, image_ids, *, expected_scope=None, timezone='UTC'):
    requested = list(dict.fromkeys(image_ids))
    async with store.pool.connection() as conn:
        rows = await _originals(store, conn, session_id, message_ids, expected_scope)
        occurrences = {image_id: (row, part) for row in rows for image_id, part in _occurrences(row)}
        # The source occurrence authorizes this selection; a matching payload
        # hash, another occurrence or a warm cache grants no access.
        references = {occurrences[image_id][1].preview_ref for image_id in requested
            if image_id in occurrences and occurrences[image_id][1].preview_ref}
        payloads = {}
        if references:
            found = await (await conn.execute('SELECT reference,payload FROM message_previews '
                'WHERE session_id=%s AND reference=ANY(%s)', (session_id, list(references)))).fetchall()
            payloads = {item['reference']: bytes(item['payload']) for item in found}
        results, evidence = [], []
        for image_id in requested:
            original = occurrences.get(image_id)
            if original is None:
                results.append({'image_id': image_id, 'status': 'unavailable',
                    'reason': 'Image is unavailable in the requested original messages.'})
                continue
            row, part = original
            data = payloads.get(part.preview_ref)
            if data is None:
                results.append({'image_id': image_id, 'status': 'unavailable',
                    'reason': 'No retained image bytes are available for this occurrence.'})
                continue
            label = {'image_id': image_id, 'source_revision': row.message.metadata['source_revision'],
                **attribution(row.message, message_id=row.db_id, timezone=timezone)}
            origin = f'memory_image:{image_id}'
            evidence.extend([
                MessagePart(PartKind.TEXT, text='[Original image evidence: '
                    + json.dumps(label, ensure_ascii=False, default=str) + ']', origin=origin, remote_sync=False),
                MessagePart(PartKind.IMAGE, mime_type=part.mime_type, data_b64=base64.b64encode(data).decode('ascii'),
                    preview_ref=part.preview_ref, detail=part.detail, origin=origin, remote_sync=False),
            ])
            # Selection is provisional; runtime owns final model admission.
            results.append({'image_id': image_id, 'status': 'selected'})
    if store._read_only and expected_scope is not None:
        await store.assert_scope(session_id, expected_scope)
    return {'image_results': results, 'evidence_parts': evidence}
