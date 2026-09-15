"""Compact command responses; effective settings belong to the runtime.

These app-authored views use Telegram HTML. Dynamic content is escaped here;
the transport owns sending, permissions and oversized diagnostic attachments.
"""
from __future__ import annotations

from dataclasses import dataclass
from html import escape
from html.parser import HTMLParser


@dataclass(frozen=True)
class Parameter:
    group: str
    label: str
    explanation: str
    status_key: str = ''
    unit: str = ''


PARAMETERS = {
    'reasoning_effort': Parameter('model', 'Reasoning effort', 'How much reasoning the provider should use before answering.'),
    'reasoning_summary': Parameter('model', 'Reasoning summary', 'Request a summary of reasoning where the provider supports it.'),
    'text_verbosity': Parameter('model', 'Answer detail', 'Ask the provider for shorter or more detailed answers.'),
    'include_thoughts': Parameter('model', 'Reasoning text', 'Request available reasoning text; this does not set thinking effort.'),
    'thinking_level': Parameter('model', 'Thinking level', 'How much thinking the model should use; a selected level takes precedence over a thinking budget.'),
    'thinking_budget': Parameter('model', 'Thinking budget', 'Token budget for thinking where supported; a selected thinking level takes precedence.'),
    'temperature': Parameter('model', 'Temperature', 'Adjust sampling randomness; lower values tend to give more consistent answers.'),
    'top_p': Parameter('model', 'Sampling probability', 'Restrict sampling to the most probable tokens whose combined probability reaches this value.'),
    'top_k': Parameter('model', 'Sampling candidates', 'Restrict sampling to this many likely next tokens.'),
    'max_output_tokens': Parameter('model', 'Output limit', 'Maximum generated tokens; some providers also count reasoning.', unit='tokens'),
    'provider_retry_count': Parameter('model', 'Request retries', 'Additional attempts after a provider request fails.'),
    'compact_trigger_tokens': Parameter('context', 'Compaction trigger', 'Start compaction when the estimated request reaches this threshold; this is not the model context window.', unit='tokens'),
    'compact_target_tokens': Parameter('context', 'Compaction target', 'Aim to reduce the estimated request to this size; protected recent messages can prevent reaching it.', unit='tokens'),
    'compact_batch_tokens': Parameter('context', 'Compaction batch', 'Estimated input size for an episode compaction batch; pressure can enlarge the batch.', unit='tokens'),
    'compact_keep_recent_ratio': Parameter('context', 'Recent raw preference', 'Prefer keeping this share of the current raw history at each compaction step; pressure can relax it, and it is not a share of the target.'),
    'compact_min_messages': Parameter('context', 'Episode minimum', 'Minimum source messages eligible for ordinary episode compaction; pressure can bypass this minimum.'),
    'min_raw_messages_reserve': Parameter('context', 'Recent message reserve', 'Prefer retaining this many recent meaningful conversation units; pressure can reduce the reserve.'),
    'compact_tool_ratio_threshold': Parameter('context', 'Tool compaction ratio', 'Minimum tool-token to user-token ratio for compacting a tool-heavy span.'),
    'compact_tool_min_tokens': Parameter('context', 'Tool compaction minimum', 'Minimum estimated span size before tool-span compaction.', unit='tokens'),
    'max_input_images': Parameter('context', 'Image limit', 'Above this image count, replace the oldest batch in the prompt with text; 0 removes the count cap.'),
    'compact_target_images': Parameter('context', 'Image compaction target', 'Image count to aim for after retiring an old image batch; 0 uses the image limit as the target.'),
    'metadata': Parameter('context', 'Message metadata', 'Include additional message metadata with incoming messages.', status_key='metadata_injection_mode'),
    'spontaneous_reply_chance': Parameter('replies', 'Spontaneous replies', 'Chance of replying to an eligible group message without a direct trigger.', unit='%'),
    'group_spontaneous_reply_delay_s': Parameter('replies', 'Spontaneous reply delay', 'Wait this long before an eligible spontaneous group reply.', unit='s'),
    'private_reply_delay_s': Parameter('replies', 'Private reply delay', 'Wait for nearby private messages before replying.', unit='s'),
    'group_reply_delay_s': Parameter('replies', 'Group reply delay', 'Wait for nearby triggered group messages before replying.', unit='s'),
    'max_interaction_rounds': Parameter('tools', 'Soft round limit', 'Tool interaction rounds before a reminder to finish. Tools remain available.'),
    'native_web_search': Parameter('tools', 'Provider web search', 'Allow the provider’s built-in web search when supported by the current model.'),
    'native_web_search_max': Parameter('tools', 'Provider web search limit', 'Cap built-in web search calls where supported; 0 removes this explicit cap.'),
    'link_prefetch': Parameter('tools', 'Link previews', 'Fetch a title or short snippet for links in incoming messages.', status_key='link_prefetch_mode'),
    'tool_history_mode': Parameter('tools', 'Tool history', 'Reuse compatible native tool history, or translate it for the selected provider.'),
    # Existing combined setter remains available through explicit lookup.
    'reply_delay_s': Parameter('', 'Private and group reply delay', 'Set both private and triggered group reply delays together.', unit='s'),
}

GROUP_LABELS = {'model': 'Model', 'context': 'Context', 'replies': 'Replies', 'tools': 'Tools'}


def plain_text(rendered_html: str) -> str:
    """Extract visible text from these app-owned inline-HTML responses."""
    class VisibleText(HTMLParser):
        def __init__(self) -> None:
            super().__init__(convert_charrefs=True)
            self.parts: list[str] = []

        def handle_data(self, data: str) -> None:
            self.parts.append(data)

    parser = VisibleText()
    parser.feed(rendered_html)
    parser.close()
    return ''.join(parser.parts)


def _text(value: object) -> str:
    return escape('—' if value is None else str(value))


def _code(value: object) -> str:
    return f'<code>{_text(value)}</code>'


def _number(value: object) -> str:
    return _text(f'{value:,}' if isinstance(value, int) else value)


def _value(status: dict, name: str) -> str:
    spec = PARAMETERS[name]
    if name == 'reply_delay_s':
        return (f"private {_code(status.get('private_reply_delay_s'))} s · "
                f"group {_code(status.get('group_reply_delay_s'))} s")
    value = status.get(spec.status_key or name)
    rendered = _code(value)
    return rendered + (f' {spec.unit}' if spec.unit else '')


def _source(status: dict, name: str) -> str:
    spec = PARAMETERS[name]
    if name == 'reply_delay_s':
        return (f"private: {_source(status, 'private_reply_delay_s')} · "
                f"group: {_source(status, 'group_reply_delay_s')}")
    source = status.get(f'{spec.status_key or name}_source')
    return _text({'session': 'this chat', 'default': 'configured default',
                  'environment': 'deployment configuration'}.get(source, source or 'not reported'))


def _queue_counts(status: dict) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in status.get('memory_jobs', []):
        key = job['status']
        counts[key] = counts.get(key, 0) + job['count']
    return counts


def status_view(status: dict, flow: dict, topic: str = '') -> str:
    if topic == 'context':
        return '\n'.join([
            '🧠 <b>Context</b>',
            f"Estimated request: <b>{_number(status.get('estimated_request_tokens'))}</b> tokens",
            f"Compaction: trigger {_number(status.get('compact_trigger_tokens'))} → target {_number(status.get('compact_target_tokens'))}",
            f"Episode batch: {_number(status.get('compact_batch_tokens'))} tokens",
            f"Recent raw preference: {_text(status.get('compact_keep_recent_ratio'))} per step; softer under pressure",
            f"Recent messages: {_number(status.get('raw_messages'))} · summaries: {_number(status.get('memory_blocks'))}",
            f"Summary layers: L0 {_number(status.get('l0_blocks'))} · L1 {_number(status.get('l1_blocks'))} · L2 {_number(status.get('l2_blocks'))}",
            f"Images: {_number(status.get('estimated_request_images'))} · limit {_text(status.get('max_input_images'))} · target {_text(status.get('compact_target_images'))}",
            'Compaction keeps originals searchable.',
            '',
            '/params context · /status full (file)',
        ])
    if topic == 'memory':
        semantic = 'semantic + text' if status.get('semantic_enabled') else 'text; semantic search not configured'
        scope = status.get('scope') or {}
        lines = [
            '🗂 <b>Memory</b>',
            f'Search: {semantic}',
            f"Agent generation: {_code(scope.get('generation'))} · context: {_code(scope.get('context_id'))}",
        ]
        jobs = status.get('memory_jobs', [])
        if jobs:
            lines.append('<b>Background work</b>')
            lines.extend(f"{_code(job['kind'])}: {_text(job['status'])} {_number(job['count'])}" for job in jobs)
        else:
            lines.append('No recorded background jobs.')
        if status.get('memory_last_error'):
            lines.extend(['', '<b>Last worker error (all chats)</b>', _text(status['memory_last_error'])])
        lines.extend(['', '/reset keeps searchable history and profiles.',
                      '/reset_full makes prior generations audit-only.'])
        return '\n'.join(lines)
    if topic == 'tools':
        remote = 'configured' if status.get('remote_enabled') else 'not configured'
        if status.get('remote_enabled'):
            remote += '; SSH master started' if status.get('remote_master_ready') else '; SSH master not started'
        lines = [
            '🛠 <b>Tools</b>',
            f"Mode: {_code(status.get('mode'))} · soft round limit: {_number(status.get('max_interaction_rounds'))}",
            f"Stickers: {_text(status.get('stickers'))} · {_number(status.get('sticker_index_count'))} in {_number(status.get('sticker_pack_count'))} packs",
            f"Catalog: {'loaded' if status.get('sticker_index_loaded') else 'not loaded'}",
            f'Remote workspace: {remote}',
        ]
        if 'available_tools' in status:
            lines.append('Available: ' + (', '.join(_code(name) for name in status['available_tools']) or 'none'))
        if status.get('native_web_search_supported'):
            lines.append(f"Provider web search: {_text(status.get('native_web_search'))}")
        lines.extend(['', '/params tools · /help tools'])
        return '\n'.join(lines)
    if topic:
        return 'Choose /status context, /status memory, /status tools or /status full (file).'

    activity = 'Replying' if flow.get('reply_running') else 'Receiving messages' if flow.get('ingest_inflight') else 'Idle'
    counts = _queue_counts(status)
    if counts.get('failed'):
        memory = '⚠️ Memory needs attention · /status memory'
    elif status.get('memory_last_error'):
        memory = '⚠️ Worker reported an error · /status memory'
    else:
        queued = counts.get('pending', 0) + counts.get('running', 0)
        memory = f"Memory: {'semantic + text' if status.get('semantic_enabled') else 'text search'}"
        if queued:
            memory += f' · {_number(queued)} jobs pending/running'
    return '\n'.join([
        f'📊 <b>Chat status</b> · {activity}',
        f"{_text(status.get('provider'))} · {_code(status.get('model'))}",
        f"Mode {_code(status.get('mode'))} · progress {_text(status.get('process'))}",
        f"Context ≈ {_number(status.get('estimated_request_tokens'))} tokens · compaction trigger {_number(status.get('compact_trigger_tokens'))}",
        f"Images {_number(status.get('estimated_request_images'))} · stickers {_text(status.get('stickers'))}",
        memory,
        f"Time zone: {_text(status.get('metadata_timezone'))}",
        '',
        '/status context · /status memory · /status tools',
        '/params · /help',
    ])


def settings_view(status: dict, topic: str = '', can_change: bool = True) -> str:
    if topic not in GROUP_LABELS:
        lines = [
            '⚙️ <b>Chat settings</b>',
            '/params model — model and generation',
            '/params context — compaction and images',
            '/params replies — timing and spontaneous replies',
            '/params tools — rounds, search and history',
            '',
        ]
        if can_change:
            lines.extend([f"Inspect: {_code('/param <name>')}",
                          f"Set: {_code('/param <name> <value>')}",
                          f"Use configured default: {_code('/param <name> default')}"])
        else:
            lines.append('Only trusted users can change advanced settings.')
        lines.extend(['/params full — all values as a file', 'Settings apply to this entire chat.'])
        return '\n'.join(lines)
    lines = [f'⚙️ <b>{GROUP_LABELS[topic]} settings</b>']
    if topic == 'model':
        lines.extend([f"/provider — {_text(status.get('provider'))}",
                      f"/model — {_code(status.get('model'))}"])
    elif topic == 'replies':
        lines.append('/mode · /process · /delivery — /help replies')
    elif topic == 'tools':
        lines.append(f"/stickers — {_text(status.get('stickers'))}")
    for name, spec in PARAMETERS.items():
        key = spec.status_key or name
        if spec.group == topic and status.get(f'{key}_supported', True):
            lines.append(f'{_code(name)} · {_value(status, name)}')
    if can_change:
        lines.extend(['', f"Effective values. Details: {_code('/param <name>')}"])
    else:
        lines.extend(['', 'Effective values. /params full — all details as a file'])
        lines.append('Only trusted users can change advanced settings.')
    return '\n'.join(lines)


def parameter_view(status: dict, name: str, usage: str | None, changed: bool = False) -> str:
    if name not in PARAMETERS:
        return f'Unknown setting {_code(name)}. Use /params to find a setting.'
    spec = PARAMETERS[name]
    key = spec.status_key or name
    lines = [f"{'✅' if changed else '⚙️'} <b>{spec.label}</b>",
             f'{_code(name)} · {_value(status, name)}',
             f'Source: {_source(status, name)}']
    if not status.get(f'{key}_supported', True):
        lines.append('Unavailable for the current provider/model.')
        if status.get(f'{key}_note'):
            lines.append(_text(status[f'{key}_note']))
        return '\n'.join(lines)
    if not changed:
        lines.extend(['', _text(spec.explanation)])
        if usage:
            lines.extend(['', _code('/param ' + usage)])
        lines.append(f"Use configured default: {_code('/param ' + name + ' default')}")
    return '\n'.join(lines)


def help_view(topic: str = '') -> str:
    if topic == 'model':
        return '\n'.join([
            '🤖 <b>Model and prompt</b>',
            '/provider — show configured providers; select one by name',
            f"{_code('/model <name>')} — use an exact model name",
            '/model default — use the configured model for this provider',
            '/params model — thinking, sampling and output',
            '/presets — list prompt presets',
            f"{_code('/preset <name> [augment|exact]')} — apply a preset",
            'augment keeps the preset plus framework guidance; exact uses the preset alone.',
            '/preset clear — clear the preset',
            '/prompt — inspect prompt controls',
        ])
    if topic == 'replies':
        return '\n'.join([
            '💬 <b>Replies</b>',
            f"{_code('/mode chat|assist|agent')}",
            'chat keeps memory/profile tools; assist and agent enable other tools and use their configured round defaults.',
            f"{_code('/process off|minimal|status|verbose|full')}",
            'Choose no progress, a brief acknowledgment, current status, more detail, or the full progress log.',
            f"{_code('/delivery edit|final_new')}",
            'Choose replacing the progress message or sending the answer separately; off/minimal progress can bypass this choice.',
            '/params replies — reply delays and spontaneous replies',
        ])
    if topic == 'context':
        return '\n'.join([
            '🧠 <b>History and memory</b>',
            '/status context — context size and compaction',
            '/status memory — searchable memory and background work',
            '/params context — compaction and image settings',
            '/reset — fresh context; keep searchable history, profiles and settings',
            '/reset_full — new agent with defaults; old history and profiles become audit-only',
            '/reset session — restore settings only',
            '/retry — retry the latest visible user message; hide newer assistant/tool output',
            f"{_code('/rollback [count]')} — hide the latest user/bot conversation blocks",
        ])
    if topic == 'tools':
        return '\n'.join([
            '🛠 <b>Tools</b>',
            f"{_code('/stickers off|auto')} — allow automatic sticker selection",
            '/mode — choose chat, assist or agent',
            '/status tools — current catalog and workspace configuration',
            '/params tools — interaction rounds, web search, link previews and history',
            'Remote file and document tools need a configured SSH workspace.',
        ])
    return '\n'.join([
        '💬 <b>Commands</b>',
        '/status — this chat at a glance',
        '/params — inspect or change settings',
        '/help replies — reply modes, progress and delivery',
        '/help model — provider, model and prompt',
        '/help context — compaction, resets, retry and rollback',
        '/help tools — stickers, search and remote files',
    ])
