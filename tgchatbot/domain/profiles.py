"""The source-backed profile document shared by learning and retrieval."""
from __future__ import annotations

from datetime import datetime
import json

from tgchatbot.domain.timestamps import format_timestamp_fields
from tgchatbot.domain.identities import actor_reference


def present_profile(document: dict, timezone: str | None = None) -> dict:
    """Format profile temporal fields for a reader without changing current membership."""
    result = dict(document)
    if document.get('identity'):
        identity = dict(document['identity'])
        if identity.get('last_message'):
            identity['last_message'] = format_timestamp_fields(identity['last_message'], ('sent_at',), timezone)
        result['identity'] = identity
    if 'facts' in document:
        result['facts'] = []
        for fact in document['facts']:
            rendered = format_timestamp_fields(fact, ('valid_from', 'valid_to'), timezone)
            if fact.get('source_dates'):
                rendered['source_dates'] = format_timestamp_fields(fact['source_dates'], ('first', 'last'), timezone)
            result['facts'].append(rendered)
    return result


def profile_size(value: dict) -> int:
    return len(json.dumps(value, ensure_ascii=False, default=lambda item: item.isoformat()
                          if isinstance(item, datetime) else str(item)).encode('utf-8'))


def chat_profile(document: dict) -> dict:
    """Present all claims; their database fact references open evidence on demand.

    Learning and operator snapshots keep the full document. Apply this only to
    newly fetched tool results, never to previously recorded conversation history.
    """
    result = dict(document)
    result['actor_id'] = actor_reference(document['actor_id'])
    if document.get('identity'):
        identity = dict(document['identity'])
        if identity.get('last_message'):
            identity['last_message'] = {key: value for key, value in identity['last_message'].items()
                                      if key != 'source_revision'}
        result['identity'] = identity
    result['facts'] = []
    for fact in document.get('facts', []):
        item = {'fact_id': fact['id'], 'claim': fact['claim'], 'kind': fact['kind']}
        for field in ('valid_from', 'valid_to'):
            if fact.get(field) is not None:
                item[field] = fact[field]
        if fact.get('asserted_by') != document['actor_id']:
            item['asserted_by'] = actor_reference(fact['asserted_by'])
        result['facts'].append(item)
    return result


def profile_document(actor_id: str, identity: dict | None, facts: list[dict], *,
                     max_bytes: int | None = None, known_agent: bool = False, strict: bool = False) -> dict:
    """Present every selected fact with its identity and attribution intact.

    Size is a learning target, not a publication or read constraint. The legacy
    sizing arguments remain accepted for callers updating independently.
    """
    agent = actor_id == 'agent'
    document = {'actor_id': actor_id,
        'identity': {'known': identity is not None or (agent and known_agent),
                     'actor_kind': identity['actor_kind'] if identity else 'agent' if agent else 'unknown',
                     'actor_name': identity['actor_name'] if identity else None},
        'subject_kind': 'agent_preferences' if agent else 'actor',
        'facts': [], 'status': 'no_current_facts' if identity or (agent and known_agent) else 'unknown_identity'}
    if identity:
        if identity.get('actor_username'):
            document['identity']['actor_username'] = identity['actor_username']
        document['identity']['last_message'] = {key: identity[key] for key in
            ('message_id', 'source_revision', 'sent_at')}
    for fact in facts:
        item = {key: fact.get(key) for key in
            ('id', 'asserted_by', 'claim', 'kind', 'source_ids', 'valid_from', 'valid_to')}
        document['facts'].append(item)
        document['status'] = 'available'
    document['status'] = ('available' if document['facts'] else 'no_current_facts'
                          if document['identity']['known'] else 'unknown_identity')
    return document
