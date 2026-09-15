"""Reversible participant references; persistence retains the source identity."""
from __future__ import annotations


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
