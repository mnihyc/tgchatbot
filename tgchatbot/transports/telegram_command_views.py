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
    'provider_retry_count': Parameter('model', 'Reply retries', 'Additional attempts after a reply-model request fails.'),
    'compact_trigger_tokens': Parameter('context', 'Context ceiling', 'Compact before replying at this size. Generation stops if the request remains above it; this is not the model context window.', unit='tokens'),
    'compact_idle_trigger_tokens': Parameter('context', 'Idle compaction trigger', 'Compact silently above this size after the idle delay; 0 disables idle compaction.', unit='tokens'),
    'compact_idle_seconds': Parameter('context', 'Idle compaction delay', 'Wait this long without incoming messages or bot replies before idle compaction.', unit='s'),
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
    'max_interaction_rounds': Parameter('tools', 'Tool-round limit', 'Tool rounds that may execute per turn. Further calls return a limit notice.'),
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


def _short_number(value: int | float | None) -> str:
    if value is None:
        return '—'
    for scale, suffix in ((1_000_000, 'M'), (1_000, 'K')):
        if abs(value) >= scale:
            return f'{value / scale:.1f}'.rstrip('0').rstrip('.') + suffix
    return f'{value:g}'


def _duration(seconds: float | None) -> str:
    if seconds is None:
        return '—'
    if seconds >= 3600 and seconds % 3600 == 0:
        return f'{seconds / 3600:g} h'
    if seconds >= 60 and seconds % 60 == 0:
        return f'{seconds / 60:g} min'
    return f'{seconds:g} s'


def _context_meter(status: dict) -> list[str]:
    used, ceiling = status.get('estimated_request_tokens'), status.get('compact_trigger_tokens')
    lines = [f'Context ≈ {_short_number(used)} / {_short_number(ceiling)} tokens']
    if used is not None and ceiling and ceiling > 0:
        # Ten cells fit a phone; the number retains finer detail and overflow.
        filled = min(10, max(0, int(used * 10 / ceiling)))
        percent = used * 100 / ceiling
        label = f'{percent:.0f}%' + (' · above ceiling' if used > ceiling else '')
        lines.append(_code('█' * filled + '░' * (10 - filled)) + ' ' + label)
    return lines


def _value(status: dict, name: str, *, compact: bool = False) -> str:
    spec = PARAMETERS[name]
    if name == 'reply_delay_s':
        return (f"private {_code(status.get('private_reply_delay_s'))} s · "
                f"group {_code(status.get('group_reply_delay_s'))} s")
    value = status.get(spec.status_key or name)
    if compact and spec.unit == 'tokens' and isinstance(value, (int, float)):
        return _short_number(value) + ' tokens'
    if compact and spec.unit == 's' and isinstance(value, (int, float)):
        return _duration(value)
    if compact and name == 'compact_keep_recent_ratio' and value is not None:
        return f'{float(value):.0%}'
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
        idle = status.get('compact_idle_trigger_tokens')
        lines = [f"<b>Saved context</b> · {_text(status.get('as_of') or '')}", *_context_meter(status),
                 *_composition_lines(status, percentages=False)]
        composition = status.get('input_composition')
        if composition and status.get('summary_tokens') is not None:
            summaries = status['summary_tokens']
            lines.append(f"Memory: {_short_number(summaries)} summaries + {_short_number(composition['memory'] - summaries)} recalled")
        lines.append(f"Summaries: {_number(status.get('selected_memory_blocks', 0))} included · {_number(status.get('memory_blocks'))} stored")
        images, limit = status.get('estimated_request_images'), status.get('max_input_images')
        projection = status.get('projected_images')
        if projection is not None:
            label = f"Images: {projection['projected']} projected"
            for key, suffix in (('pending', 'pending previews'), ('unsupported', 'unsupported'), ('unavailable', 'unavailable')):
                if projection[key]:
                    label += f" · {projection[key]} {suffix}"
            lines.append(label)
        elif images or limit:
            image_line = f'Images: {_number(images)}'
            image_line += f' / {_number(limit)}' if limit else ' · no count limit'
            target = status.get('compact_target_images')
            if target and target != limit:
                image_line += f' · target {_number(target)}'
            lines.append(image_line)
        lines.extend([f"Compaction target: {_short_number(status.get('compact_target_tokens'))} tokens",
            (f"Idle compaction: {_short_number(idle)} tokens after {_duration(status.get('compact_idle_seconds'))}" if idle else 'Idle compaction: off'),
            '/context recent · /context summaries', '/context profiles · /context tools',
            '/context full · /params context'])
        return '\n'.join(lines)
    if topic == 'memory':
        semantic = 'semantic + text' if status.get('semantic_enabled') else 'text only'
        lines = ['<b>Memory</b>', f'Search: {semantic}']
        jobs: dict[str, list[str]] = {}
        for job in status.get('memory_jobs', []):
            if job['status'] in {'done', 'stale'}:
                continue
            label = {'pending': 'queued'}.get(job['status'], job['status'])
            jobs.setdefault(job['kind'], []).append(f"{_number(job['count'])} {_text(label)}")
        names = {'memory_ingest': 'Indexing', 'memory_embed': 'Embeddings',
                 'memory_profile': 'Profiles', 'memory_tail': 'Recent history',
                 'embedding_batch': 'Embedding batches'}
        lines.extend(_text(names.get(kind, kind)) + ': ' + ' · '.join(counts)
                     for kind, counts in jobs.items())
        if not jobs:
            lines.append('Background queue: empty')
        if status.get('memory_last_error'):
            lines.append('Last worker error (all chats): ' + _text(status['memory_last_error']))
        lines.extend(['', '/status full · /help context'])
        return '\n'.join(lines)
    if topic == 'tools':
        remote = 'configured' if status.get('remote_enabled') else 'not configured'
        lines = [
            '<b>Tools</b>',
            f"Mode: {_code(status.get('mode'))} · tool-round limit: {_number(status.get('max_interaction_rounds'))}",
            f"Stickers: {_text(status.get('stickers'))} · {_number(status.get('sticker_index_count'))} in {_number(status.get('sticker_pack_count'))} packs",
            f'Remote workspace: {remote}',
        ]
        if not status.get('sticker_index_loaded'):
            lines.append('Sticker catalog not loaded.')
        if 'available_tools' in status:
            lines.append('Available: ' + (', '.join(_code(name) for name in status['available_tools']) or 'none'))
        if status.get('native_web_search_supported'):
            lines.append(f"Provider web search: {_text(status.get('native_web_search'))}")
        lines.extend(['', '/params tools'])
        return '\n'.join(lines)
    if topic:
        return 'Choose /status context, /status memory, /status tools or /status full (file).'

    activity = ('Compacting context' if flow.get('compacting') else 'Replying' if flow.get('reply_running')
                else 'Receiving messages' if flow.get('ingest_inflight') else 'Idle')
    lines = [f'<b>Chat status</b> · {activity}',
        f"{_text(status.get('provider'))} · {_code(status.get('model'))}",
        *_context_meter(status)]
    if status.get('input_composition') is not None:
        lines.extend(['Share of estimated input:', *_composition_lines(status, percentages=True)])
    counts = _queue_counts(status)
    if counts.get('failed'):
        lines.append(f"Memory: {_number(counts['failed'])} failed jobs · /status memory")
    elif status.get('memory_last_error'):
        lines.append('Worker error · /status memory')
    lines.extend(['', '/context · /help'])
    return '\n'.join(lines)


def _composition_lines(status: dict, *, percentages: bool) -> list[str]:
    values = status.get('input_composition')
    if values is None:
        return [] if percentages else ['Input shares unavailable for this provider.']
    def value(key):
        tokens = values[key]
        if not percentages:
            return _short_number(tokens)
        percent = tokens * 100 / max(1, status['estimated_request_tokens'])
        return '&lt;1%' if 0 < percent < 1 else f'{percent:.0f}%'
    return [' · '.join(f'{label} {value(key)}' for key, label in group) for group in (
        (('user', 'User'), ('assistant', 'Assistant'), ('tools', 'Tools')),
        (('memory', 'Memory'), ('profiles', 'Profiles'), ('system', 'System')))]


def settings_view(status: dict, topic: str = '', can_change: bool = True) -> str:
    if topic not in GROUP_LABELS:
        lines = [
            '<b>Chat settings</b>',
            '/params model — model and generation',
            '/params context — compaction and images',
            '/params replies — timing and spontaneous replies',
            '/params tools — rounds, search and history',
            '',
        ]
        if can_change:
            lines.extend([f"Inspect: {_code('/param <name>')}",
                          f"Set/reset: {_code('/param <name> <value|default>')}"])
        else:
            lines.append('Only trusted users can change advanced settings.')
        lines.extend(['/params full — all values as a file', 'Settings apply to this entire chat.'])
        return '\n'.join(lines)
    lines = [f'<b>{GROUP_LABELS[topic]} settings</b>']
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
            lines.append(f'{_code(name)} · {_value(status, name, compact=True)}')
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
    lines = [f"<b>{spec.label}{' updated' if changed else ''}</b>",
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
    return '\n'.join(lines)


def compaction_result(result: dict[str, int]) -> str:
    remaining, target = result['estimated_request_tokens'], result['target_tokens']
    title = 'Context ready' if remaining <= target else 'Target not reached'
    lines = [f'<b>{title}</b>', f'Estimated request: {_number(remaining)} / {_number(target)} tokens']
    if remaining > target:
        lines.append('/params context — inspect compaction limits')
    return '\n'.join(lines)


def rollback_result(hidden: int, previews: list[tuple[str, str]]) -> str:
    lines = [f'Rollback complete. {hidden} message(s) hidden from context.']
    for role, text in previews:
        # Preserve the selected previews; each excerpt is one short line.
        text = ' '.join(text.split())
        if len(text) > 40:
            text = text[:39] + '…'
        lines.append(f'- {_text(role)}: {_text(text)}')
    if hidden > len(previews):
        lines.append(f'… {hidden - len(previews)} earlier message(s)')
    return '\n'.join(lines)


def help_view(topic: str = '') -> str:
    if topic == 'model':
        return '\n'.join([
            '<b>Model and prompt</b>',
            '/provider — show configured providers; select one by name',
            f"{_code('/model <name>')} — use an exact model name",
            '/model default — use the configured model for this provider',
            '/params model — thinking, sampling and output',
            '/presets — list prompt presets',
            f"{_code('/preset <name> [augment|exact]')} — apply a preset",
            'augment keeps the preset plus framework guidance; exact uses the preset alone.',
            '/preset clear — restore the configured prompt',
            '/prompt — inspect prompt controls',
        ])
    if topic == 'replies':
        return '\n'.join([
            '<b>Replies</b>',
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
            '<b>History and memory</b>',
            '/context — saved context size, composition and compaction',
            '/context recent — message previews; add a count or before &lt;message_id&gt;',
            '/context message &lt;id&gt; — read an original or tool record',
            '/context summaries [all] — included summaries or all stored blocks',
            '/context block &lt;id&gt; — summary, participants and sources',
            '/context tools [tool_name] — recorded calls and results',
            '/context profiles — saved facts and included snapshots',
            '/context profile &lt;actor&gt; — use a listed person_id:, chat_id: or agent',
            '/context full — complete readable context; file only when too long',
            'Shares divide estimated input, including media; the bar divides input by the configured ceiling. Memory includes summaries and recalled results; Profiles counts fetched snapshots. Tools excludes those results; System includes instructions, definitions and framing.',
            '/status memory — searchable memory and background work',
            '/params context — compaction and image settings',
            '/compact — compact now to the configured target; keep originals and profiles',
            '/reset — fresh context; keep searchable history, profiles and settings',
            '/reset_full — new agent with defaults; old history and profiles become audit-only',
            '/reset session — restore settings only',
            '/retry — retry the latest visible user message; hide newer assistant/tool output',
            f"{_code('/rollback [count]')} — hide the latest user/bot conversation blocks",
        ])
    if topic == 'tools':
        return '\n'.join([
            '<b>Tools</b>',
            f"{_code('/stickers off|auto')} — allow automatic sticker selection",
            '/mode — choose chat, assist or agent',
            '/status tools — current catalog and workspace configuration',
            '/params tools — interaction rounds, web search, link previews and history',
            'Remote file and document tools need a configured SSH workspace.',
        ])
    return '\n'.join([
        '<b>Commands</b>',
        '/status — this chat at a glance',
        '/context — inspect messages, summaries, profiles and tools',
        '/params — inspect or change settings',
        '/help replies — reply modes, progress and delivery',
        '/help model — provider, model and prompt',
        '/help context — compaction, resets, retry and rollback',
        '/help tools — stickers, search and remote files',
    ])
