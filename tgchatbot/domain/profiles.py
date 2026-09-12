"""The small, source-backed profile document shared by learning and retrieval."""
from __future__ import annotations

from datetime import datetime
import json


def profile_size(value: dict) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=lambda item: item.isoformat()
                          if isinstance(item, datetime) else str(item)).encode('utf-8'))


def profile_document(actor_id: str, identity: dict | None, facts: list[dict], *,
                     max_bytes: int | None, known_agent: bool = False, strict: bool = False) -> dict:
    """Bound the complete per-person tool payload, including its attribution.

    Historical fact revisions and patch reasoning are audit records, not chat
    context. A smaller operational setting takes effect on the next read without
    deleting evidence. Learning uses strict mode so a bad patch cannot silently
    evict a preference chosen by the model.
    """
    agent = actor_id == 'agent'
    document = {'actor_id': actor_id,
        'identity': {'known': identity is not None or (agent and known_agent),
                     'actor_kind': identity['actor_kind'] if identity else 'agent' if agent else 'unknown',
                     'actor_name': identity['actor_name'] if identity else None},
        'subject_kind': 'agent_preferences' if agent else 'actor',
        'facts': [], 'status': 'no_current_facts' if identity or (agent and known_agent) else 'unknown_identity'}
    if identity:
        document['identity']['last_message'] = {key: identity[key] for key in
            ('message_id', 'source_revision', 'sent_at')}
    if max_bytes is not None and profile_size(document) > max_bytes:
        # Display names are presentation; stable actor IDs must never be cut.
        name = document['identity']['actor_name'] or ''
        while name and profile_size(document) > max_bytes:
            excess = profile_size(document) - max_bytes
            name = name.encode('utf-8')[:-max(1, excess)].decode('utf-8', errors='ignore')
            document['identity']['actor_name'] = name
        if max_bytes is not None and profile_size(document) > max_bytes:
            raise ValueError('MEMORY_PROFILE_BYTES cannot fit this actor identity')
    for fact in facts:
        item = {key: fact.get(key) for key in
            ('id', 'asserted_by', 'claim', 'kind', 'source_ids', 'valid_from', 'valid_to')}
        document['facts'].append(item)
        document['status'] = 'available'
        if max_bytes is not None and profile_size(document) > max_bytes:
            document['facts'].pop()
            if strict:
                raise ValueError('Profile patch exceeds MEMORY_PROFILE_BYTES; shorten or retire redundant facts')
    document['status'] = ('available' if document['facts'] else 'no_current_facts'
                          if document['identity']['known'] else 'unknown_identity')
    return document
