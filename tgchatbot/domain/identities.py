"""Reversible participant references; persistence retains the source identity."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import datetime
import json
from typing import Any


def telegram_user_id(actor_id: str | None) -> int | None:
    """Return the actual user identity; a sender chat is not its posting account."""
    if actor_id and actor_id.startswith('telegram:user:'):
        return int(actor_id.removeprefix('telegram:user:'))
    return None


def actor_reference(actor_id: str) -> str:
    """Use the complete Telegram numeric identity with a concise namespace."""
    for canonical, reference in (('telegram:user:', 'person_id:'), ('telegram:chat:', 'chat_id:')):
        if actor_id.startswith(canonical):
            number = actor_id[len(canonical):]
            if number.lstrip('-').isdigit() and number.count('-') <= 1:
                return reference + number
    return actor_id


def canonical_actor_id(reference: str) -> str:
    """Accept new references and historical canonical IDs without guessing names."""
    for compact, canonical in (('person_id:', 'telegram:user:'), ('chat_id:', 'telegram:chat:')):
        if reference.startswith(compact):
            number = reference[len(compact):]
            if number.lstrip('-').isdigit() and number.count('-') <= 1:
                return canonical + number
    return reference


def actor_observation(metadata: Mapping[str, Any], message_id: int) -> dict | None:
    """One source-owned identity; its ordering fields never enter the agent label."""
    actor = metadata.get('actor_id')
    if not actor or actor == 'unknown':
        return None
    result = {'id': canonical_actor_id(actor), 'message_id': message_id}
    for source, target in (('actor_name', 'name'), ('actor_username', 'username')):
        if metadata.get(source):
            result[target] = metadata[source]
    if metadata.get('actor_kind') not in (None, '', 'user', 'unknown'):
        result['kind'] = metadata['actor_kind']
    at = metadata.get('sent_at')
    result['sent_at'] = at.isoformat() if isinstance(at, datetime) else at or ''
    return result


def latest_actor_observations(observations: Iterable[Mapping[str, Any]]) -> list[dict]:
    """Keep one whole observation per ID; absent fields do not revive old handles."""
    latest: dict[str, dict] = {}
    def order(value):
        at = value.get('sent_at')
        return (datetime.fromisoformat(at).timestamp() if at else float('-inf'), value['message_id'])
    for value in observations:
        actor = canonical_actor_id(value['id'])
        previous = latest.get(actor)
        if previous is None or order(value) > order(previous):
            latest[actor] = {**value, 'id': actor}
    return list(latest.values())


def format_actor_labels(labels: Iterable[str], observations: Iterable[Mapping[str, Any]] | None = None) -> str:
    """Use provenance's vocabulary; legacy recorded summaries retain their text."""
    if observations is None:
        return ', '.join(labels)
    by_id = {canonical_actor_id(item['id']): item for item in observations}
    result = []
    for label in dict.fromkeys(canonical_actor_id(label) for label in labels):
        observed = by_id.get(label, {})
        result.append({'id': actor_reference(label), **{key: observed[key]
            for key in ('name', 'username', 'kind') if observed.get(key)}})
    return json.dumps(result, ensure_ascii=False, separators=(',', ':'))
