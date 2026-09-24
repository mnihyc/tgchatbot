"""Readable inspection output shared by Telegram and the operator CLI."""
from __future__ import annotations

import json

from tgchatbot.domain.identities import actor_reference, format_actor_labels
from tgchatbot.domain.models import PartKind
from tgchatbot.domain.provenance import present_tool_output
from tgchatbot.domain.timestamps import format_timestamp
from tgchatbot.logging_config import clip_for_log
from tgchatbot.providers.base import tool_message_evidence
from tgchatbot.transports.telegram_command_views import plain_text, status_view, _short_number


def parse_context_arguments(arguments):
    args = list(arguments)
    topic = args.pop(0).lower() if args else 'overview'
    if topic not in {'overview', 'recent', 'message', 'summaries', 'block', 'tools', 'profiles', 'profile', 'full'}:
        raise ValueError('Use /context recent, summaries, tools, profiles or full.')
    options = {}

    def number(value, prefix=None):
        if prefix and value.startswith(prefix + ':'):
            value = value[len(prefix) + 1:]
        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError('Use a positive count or message/summary ID. See /help context.') from exc
        if parsed <= 0:
            raise ValueError('Counts and message/summary IDs must be positive.')
        return parsed

    if topic in {'message', 'block', 'profile'}:
        if len(args) != 1:
            raise ValueError(f'Use /context {topic} <' + ('actor>' if topic == 'profile' else 'id>'))
        options['actor' if topic == 'profile' else 'object_id'] = args[0] if topic == 'profile' else number(args[0], topic)
        return topic, options
    while args:
        arg = args.pop(0)
        if arg == 'all' and topic == 'summaries':
            options['all_blocks'] = True
        elif arg == 'before' and topic in {'recent', 'summaries', 'tools'} and args:
            options['before_id'] = number(args.pop(0), 'block' if topic == 'summaries' else 'message')
        elif arg == 'after' and topic == 'profiles' and args:
            options['after'] = args.pop(0)
        elif arg.isdigit() and topic in {'recent', 'summaries', 'tools', 'profiles'}:
            options['limit'] = number(arg)
        elif topic == 'tools' and 'name' not in options and arg not in {'before', 'after'}:
            options['name'] = arg
        else:
            raise ValueError('Unrecognized context argument. Use /help context.')
    return topic, options


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def message_text(message, *, timezone=None, projected=False):
    """Readable payload; binary data and opaque reasoning are not prose."""
    payload = message.metadata.get('tool_payload')
    native = (message.metadata.get('provider_native') or {}) if projected or isinstance(payload, dict) else {}
    lines = []
    for item in native.get('items', []):
        if item.get('type') == 'reasoning':
            continue
        parts = item.get('parts', item.get('content', []))
        if isinstance(parts, str):
            lines.append(parts)
        elif isinstance(parts, list):
            for part in parts:
                if part.get('thought'):
                    continue
                if part.get('text') is not None:
                    lines.append(part['text'])
                if 'inlineData' in part:
                    lines.append('[' + str(part['inlineData'].get('mimeType') or 'media') + ']')
                for key in ('functionCall', 'functionResponse', 'toolCall', 'toolResponse'):
                    if key in part:
                        value = part[key]
                        lines.append(str(value.get('name') or value.get('toolType') or key) + '\n' +
                                     _json({k: v for k, v in value.items() if k not in {'id', 'parts'}}))
        if item.get('type') == 'function_call':
            lines.append(str(item.get('name') or 'tool') + '\n' + str(item.get('arguments') or '{}'))
        for call in item.get('tool_calls', []):
            function = call.get('function') or {}
            lines.append(str(function.get('name') or 'tool') + '\n' + str(function.get('arguments') or '{}'))
        if item.get('type') == 'web_search_call':
            lines.append('web_search\n' + _json(item.get('action')) + '\nResult not recorded')
    if isinstance(payload, dict) and (not native or message.metadata.get('tool_phase') == 'result'):
        lines = [str(message.name or 'tool') + ' · ' + str(message.metadata.get('tool_phase') or 'event')]
        output = payload.get('output', payload.get('arguments', payload))
        if isinstance(output, dict) and message.metadata.get('tool_phase') == 'result':
            output = present_tool_output(message.name, output, timezone)
        lines.append(_json(output))
        evidence = (tool_message_evidence(message, images=False) if projected else
                    [part for part in message.parts if part.origin != 'tool_output'])
        lines.extend(part.text for part in evidence
                     if part.kind == PartKind.TEXT and part.text is not None)
    if not native and not isinstance(payload, dict):
        lines = [part.text for part in message.parts if part.kind == PartKind.TEXT and part.text is not None]
    for part in message.parts:
        if part.kind != PartKind.TEXT:
            lines.append('[' + part.kind.value + (': ' + part.filename if part.filename else '') +
                         (' · ' + part.text if part.text else '') + ']')
    return '\n'.join(lines)


def _date(value, zone):
    return (format_timestamp(value, zone) or 'date unknown').replace('T', ' ')


def _speaker(message):
    meta = message.metadata
    actor = meta.get('actor_id')
    name = meta.get('actor_name') or ('Bot' if message.role.value == 'assistant' else message.role.value)
    return name + (' · ' + actor_reference(actor) if actor else '')


def _recent(data, zone):
    lines = ['Recent messages · excerpts']
    for record in reversed(data['items']):
        stored = record['message']
        message = stored.message
        suffix = f" · summarized by block:{record['compacted_by']}" if record['compacted_by'] else ''
        lines.extend([f"{_date(message.metadata.get('sent_at'), zone)} · {_speaker(message)} · message:{stored.db_id}{suffix}",
                      clip_for_log(message_text(message, timezone=zone), limit=80)])
    if not data['items']:
        lines.append('No recent conversation messages in this context.')
    else:
        lines.append(f"/context message {data['items'][0]['message'].db_id}")
    if data['next']:
        lines.append(f"/context recent before {data['next']}")
    return '\n'.join(lines)


def _summaries(data, zone):
    lines = [f"Summaries · {data['selected']} included · {data['stored']} stored"]
    for block in data['items']:
        tokens = block.get('request_tokens')
        size = f" · ≈{_short_number(tokens)} tokens" if tokens is not None else ''
        lines.append(f"block:{block['block_id']} · {block['kind']} L{block['level']} · " +
                     ('included' if block['included'] else 'stored') + (' · parent' if block['parent'] else '') + size)
        lines.append(f"{_date(block['time_start'], zone)} — {_date(block['time_end'], zone)}")
        lines.append(clip_for_log(block['summary_text'], limit=80))
    if data['items']:
        lines.append(f"/context block {data['items'][0]['block_id']}")
    else:
        lines.append('No summaries in this view.')
    if data['next']:
        lines.append(f"/context summaries {'all ' if data['all'] else ''}before {data['next']}")
    if not data['all'] and data['stored'] > data['selected']:
        lines.append('/context summaries all')
    return '\n'.join(lines)


def _profiles(data, zone):
    lines = ['Profiles · saved facts and included snapshots']
    for row in data['items']:
        profile, snapshots = row['profile'], row['snapshots']
        actor = actor_reference(profile['actor_id'])
        identity = profile.get('identity') or {}
        label = identity.get('actor_name') or actor
        if identity.get('actor_username'):
            label += ' · @' + identity['actor_username'].lstrip('@')
        if label != actor:
            label += ' · ' + actor
        lines.append(f"{label} · {len(profile.get('facts', []))} facts")
        if row['updated_at']:
            lines.append('Saved update: ' + _date(row['updated_at'], zone))
        if row['pending']['sources']:
            lines.append(f"Learning pending: {row['pending']['sources']} source messages")
        if snapshots:
            latest = snapshots[-1]
            lines.append(f"In context: {len(snapshots)} snapshot(s) · latest {_date(latest['captured_at'], zone)}")
        else:
            lines.append('No fetched snapshot in context.')
        if data['detail']:
            for fact in profile.get('facts', []):
                reference = fact.get('id', fact.get('fact_id'))
                lines.append(f"fact:{reference} · {fact['claim']}")
                qualifier = [str(fact[key]) for key in ('kind', 'asserted_by', 'valid_from', 'valid_to') if fact.get(key) is not None]
                if qualifier:
                    lines.append(' · '.join(qualifier))
                sources = fact.get('source_ids', [])
                if sources:
                    lines.append('Sources: ' + ', '.join('message:' + str(value) for value in sources))
            for snapshot in snapshots:
                lines.append(f"Captured {_date(snapshot['captured_at'], zone)} · /context message {snapshot['message_id']}")
        else:
            lines.append('/context profile ' + actor)
    if not data['items']:
        lines.append('No saved profiles or included snapshots.')
    if data['next']:
        lines.append('/context profiles after ' + data['next'])
    return '\n'.join(lines)


def _tools(data, zone):
    lines = ['Tool records · excerpts']
    for owner in data['items']:
        lines.append(f"message:{owner['message_id']} · {_date(owner['at'], zone)}")
        for event in owner['events']:
            lines.append(f"{event['name']} · {event['status']}" + (' · provider-native' if event['native'] else ''))
            if event['arguments'] is not None:
                lines.append(clip_for_log(_json(event['arguments']), limit=80))
            if event['native'] and event['result'] is None and event['status'] not in {'call recorded', 'result recorded'}:
                lines.append('Result not recorded')
        lines.append(f"/context message {owner['message_id']}")
    if not data['items']:
        lines.append('No matching recorded tools in this context.')
    if data['next']:
        lines.append('/context tools ' + (data['name'] + ' ' if data['name'] else '') + 'before ' + str(data['next']))
    return '\n'.join(lines)


def render_context(view):
    topic, data, zone = view['topic'], view['data'], view['timezone']
    if topic == 'overview':
        return plain_text(status_view(data, {}, 'context'))
    if topic == 'recent':
        return _recent(data, zone)
    if topic == 'summaries':
        return _summaries(data, zone)
    if topic in {'profiles', 'profile'}:
        return _profiles(data, zone)
    if topic == 'tools':
        return _tools(data, zone)
    if topic == 'block':
        participants = format_actor_labels(data['actor_labels'], data.get('actor_identities'))
        return '\n'.join([f"Summary · block:{data['block_id']} · {data['kind']} L{data['level']}",
            f"{_date(data['time_start'], zone)} — {_date(data['time_end'], zone)}",
            ('Current context' if data['current_context'] else 'Stored outside current context'),
            'Participants: ' + (participants or 'not recorded'), data['summary_text'],
            'Sources: ' + ', '.join('message:' + str(mid) for mid in data['source_ids']),
            'Parents: ' + (', '.join('block:' + str(bid) for bid in data['parent_block_ids']) or 'none')])
    if topic == 'message':
        stored = data['message']
        message = stored.message
        state = 'Current context' if data['current_context'] else 'Stored outside current context'
        if data['compacted_by']:
            state += f" · summarized by block:{data['compacted_by']}"
        if data['replay_owner']:
            state += f" · presentation owned by message:{data['replay_owner']}"
        lines = [f"message:{stored.db_id} · {_speaker(message)}",
            _date(message.metadata.get('sent_at'), zone), state, message_text(message, timezone=zone)]
        if 'tool_result' in data:
            result = data['tool_result']
            if result:
                stored_result = result['message']
                lines.extend([f'Result · message:{stored_result.db_id}',
                    _date(stored_result.message.metadata.get('sent_at'), zone),
                    message_text(stored_result.message, timezone=zone)])
            else:
                lines.append('Result not recorded')
        for event in data.get('native_tools', []):
            lines.append(f"Provider-native tool · {event['name']} · {event['status']}")
            if event['arguments'] is not None:
                lines.extend(['Arguments', _json(event['arguments'])])
            if event['result'] is not None:
                lines.extend(['Recorded result', _json(event['result'])])
            elif event['status'] not in {'call recorded', 'result recorded'}:
                lines.append('Result not recorded')
        return '\n'.join(lines)
    if topic in {'full', 'report'}:
        lines = [plain_text(status_view(data['overview'], {}, 'context')),
                 f"{data['overview']['provider']} · {data['overview']['model']}",
                 f"History: {data['overview']['tool_history_mode']} · estimator ×{data['overview']['estimator_multiplier']:.3g}\n"
                 f"Scope: generation {data['overview']['scope']['generation']}, context {data['overview']['scope']['context_id']}",
                 'Saved context; media are references and opaque provider continuations are omitted.']
        if data['complete_context']:
            lines.extend(['System instructions', data['instructions'], 'Tool definitions', _json(data['tool_definitions']), 'Context timeline'])
            for entry in data['timeline']:
                label = f'block:{entry.block_id}' if entry.block_id is not None else f'message:{entry.message_id}'
                lines.extend([label + ' · ' + entry.message.role.value, message_text(entry.message, timezone=zone, projected=True)])
        else:
            lines.extend([f"Preset: {len(data['preset'] or '')} characters · excerpt\n" + clip_for_log(data['preset'] or '', limit=80),
                          'Recent preview', _recent(data['recent'], zone),
                          'Summary preview', _summaries(data['summaries'], zone),
                          'Tool preview', _tools(data['tools'], zone),
                          plain_text(status_view(data['overview'], {}, 'memory'))])
        lines.extend(['Current saved profiles (preview; distinct from included snapshots)', _profiles(data['profiles'], zone)])
        return '\n\n'.join(lines)
    raise ValueError('Unknown inspection view')
