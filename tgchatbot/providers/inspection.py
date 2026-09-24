"""Input accounting beside the provider payload, never inside model history.

Adapters own serialization and token estimation. This visitor attributes those
same estimated parts; it does not tokenize independently or modify requests.
"""
from __future__ import annotations

from dataclasses import dataclass

from tgchatbot.core.context_state import is_auto_note_message
from tgchatbot.core.token_estimator import TokenEstimator
from tgchatbot.domain.models import ConversationMessage, MessageRole, PartKind
from tgchatbot.providers.base import pending_image_tokens


CATEGORIES = ('user', 'assistant', 'tools', 'memory', 'profiles', 'system')


@dataclass(frozen=True)
class HistoryEntry:
    message: ConversationMessage
    message_id: int | None = None
    block_id: int | None = None


def result_category(name: str | None) -> str:
    if name == 'user_profile_fetch':
        return 'profiles'
    return 'memory' if name in {'memory_search', 'memory_read'} else 'tools'


def content_category(entry: HistoryEntry) -> str:
    if entry.block_id is not None:
        return 'memory'
    message, meta = entry.message, entry.message.metadata
    if message.role == MessageRole.TOOL or meta.get('source_role') == 'tool':
        return (result_category(message.name or meta.get('tool_name'))
                if meta.get('tool_phase') == 'result' else 'tools')
    if is_auto_note_message(message) or meta.get('synthetic_role'):
        return 'system'
    return message.role.value if message.role in {MessageRole.USER, MessageRole.ASSISTANT} else 'system'


def _native_text_category(entry: HistoryEntry, role: str | None) -> str:
    if entry.block_id is not None or entry.message.metadata.get('synthetic_role'):
        return content_category(entry)
    # One native model batch may own prose as well as several function calls.
    if role in {'assistant', 'model'} and entry.message.metadata.get('provider_native'):
        return 'assistant'
    return content_category(entry)


def _gemini_images(part: dict) -> int:
    if 'inlineData' in part:
        return int(str(part['inlineData'].get('mimeType', '')).startswith('image/'))
    if 'functionResponse' in part:
        return sum(_gemini_images(p) for p in part['functionResponse'].get('parts', []) if isinstance(p, dict))
    return 0


def _responses_images(item: dict) -> int:
    if item.get('type') == 'input_image':
        return 1
    parts = item.get('output') if item.get('type') == 'function_call_output' else item.get('content')
    return sum(_responses_images(p) for p in parts if isinstance(p, dict)) if isinstance(parts, list) else 0


def _allocate(values: list[int], total: int) -> list[int]:
    """Preserve the estimator's rounded component total after calibration."""
    source = sum(values)
    if not source:
        return [0] * len(values)
    floors = [value * total // source for value in values]
    order = sorted(range(len(values)), key=lambda i: (-(values[i] * total % source), i))
    for i in order[:total - sum(floors)]:
        floors[i] += 1
    return floors


def inspect_history(provider, settings, entries: list[HistoryEntry], raw_estimate, estimate) -> dict | None:
    kind = getattr(provider, 'inspection_format', None)
    if kind not in {'gemini', 'responses', 'chat_completions'}:
        return None
    rows = [{key: 0 for key in CATEGORIES} for _ in entries]
    images = {'projected': 0, 'pending': 0, 'unavailable': 0, 'unsupported': 0}

    def add(index, category, tokens):
        rows[index][category] += int(tokens)

    for index, entry in enumerate(entries):
        message = entry.message
        tool_images = provider.supports_tool_evidence(settings)
        supported = (provider.capabilities.multimodal_input and
                     message.role in getattr(provider, 'input_image_roles', tuple(MessageRole)))
        pending_messages = [message] if supported else []
        pending = pending_image_tokens(pending_messages, tool_images=tool_images)
        add(index, content_category(entry), pending)
        images['pending'] += pending // TokenEstimator.IMAGE_TOKENS
        pixels = [p for p in message.parts if p.kind == PartKind.IMAGE]
        supported = supported and (tool_images or message.role != MessageRole.TOOL)
        images['unsupported'] += sum(bool(p.data_b64 or p.preview_ref) for p in pixels) if not supported else 0
        images['unavailable'] += sum(not p.data_b64 and not p.preview_ref for p in pixels) if supported else 0

        if kind == 'gemini':
            for content in provider._message_to_contents(message, tool_images=tool_images):
                parts_total = 0
                for part in content.get('parts', []):
                    if not isinstance(part, dict):
                        continue
                    tokens = provider._estimate_part_tokens(part)
                    parts_total += tokens
                    category = _native_text_category(entry, content.get('role'))
                    if 'functionResponse' in part:
                        category = result_category(part['functionResponse'].get('name'))
                    elif any(key in part for key in ('functionCall', 'toolCall', 'toolResponse')):
                        category = 'tools'
                    elif part.get('thought') or 'text' not in part and 'thoughtSignature' in part:
                        category = 'system'
                    add(index, category, tokens)
                    images['projected'] += _gemini_images(part)
                add(index, 'system', provider._estimate_content_tokens(content) - parts_total)
        elif kind == 'responses':
            def visit(item):
                if not isinstance(item, dict):
                    return
                tokens = provider._estimate_input_item_tokens(item)
                parts = item.get('content')
                if isinstance(parts, list) and ('role' in item or item.get('type') == 'message'):
                    for part in parts:
                        visit(part)
                    add(index, 'system', tokens - sum(provider._estimate_input_item_tokens(p) for p in parts if isinstance(p, dict)))
                    return
                item_type = item.get('type')
                category = _native_text_category(entry, 'assistant' if item_type == 'output_text' else message.role.value)
                if item_type in {'function_call', 'web_search_call'}:
                    category = 'tools'
                elif item_type == 'function_call_output':
                    category = result_category(message.name or message.metadata.get('tool_name'))
                elif item_type == 'reasoning':
                    category = 'system'
                add(index, category, tokens)
                images['projected'] += _responses_images(item)
            for item in provider._message_to_input_items(message):
                visit(item)

    if kind == 'chat_completions':
        for item, index in provider._messages_with_owners([entry.message for entry in entries]):
            entry = entries[index]
            cost = provider._estimate_value_tokens
            calls, body = item.get('tool_calls'), item.get('content')
            category = _native_text_category(entry, item.get('role'))
            if item.get('role') == 'tool':
                category = result_category(entry.message.name or entry.message.metadata.get('tool_name'))
            if item.get('role') == 'tool':
                add(index, category, cost(item))
            else:
                add(index, category, cost(body))
                add(index, 'tools', cost(calls))
                add(index, 'system', cost(item) - cost(body) - cost(calls))
            if isinstance(body, list):
                images['projected'] += sum(isinstance(p, dict) and p.get('type') == 'image_url' for p in body)

    values = [row[category] for row in rows for category in CATEGORIES]
    # An unfamiliar estimator extension must not acquire plausible but invented
    # shares. The total remains available; tests cover every supported adapter.
    if sum(values) != raw_estimate.history_tokens:
        return None
    scaled = iter(_allocate(values, estimate.history_tokens))
    categories = {key: 0 for key in CATEGORIES}
    details = []
    for entry in entries:
        attributed = {key: next(scaled) for key in CATEGORIES}
        for key, value in attributed.items():
            categories[key] += value
        details.append({'message_id': entry.message_id, 'block_id': entry.block_id,
                        'tokens': sum(attributed.values()), 'categories': attributed})
    categories['system'] += estimate.total_tokens - estimate.history_tokens
    return {'categories': categories, 'entries': details, 'images': images}
