"""Shared operator context views. No generation, learning, recovery or mutation."""
from __future__ import annotations

from dataclasses import asdict

from tgchatbot.core.prompting import build_system_prompt
from tgchatbot.domain.identities import actor_reference, canonical_actor_id
from tgchatbot.domain.models import MessageRole
from tgchatbot.domain.profiles import present_profile
from tgchatbot.domain.timestamps import format_timestamp
from tgchatbot.providers.inspection import inspect_history
from tgchatbot.storage.inspection import inspection_snapshot


def native_tools(message):
    """Only durable native items are inspectable; missing results stay missing."""
    native = message.metadata.get('provider_native') or {}
    events = []
    responses = {part['toolResponse']['id']: part['toolResponse']
        for item in native.get('items', []) for part in item.get('parts', [])
        if isinstance(part.get('toolResponse'), dict) and part['toolResponse'].get('id')}
    paired = set()
    for item in native.get('items', []):
        if item.get('type') == 'web_search_call':
            events.append({'name': 'web_search', 'native': True, 'arguments': item.get('action'),
                           'status': item.get('status', 'unknown'), 'result': None})
        for part in item.get('parts', []):
            for key in ('toolCall', 'toolResponse'):
                if key in part:
                    payload = part[key]
                    native_id = payload.get('id')
                    if key == 'toolResponse' and native_id in paired:
                        continue
                    response = responses.get(native_id) if key == 'toolCall' else payload
                    if key == 'toolCall' and response is not None:
                        paired.add(native_id)
                    events.append({'name': str(payload.get('toolType') or 'google_search').lower(),
                        'native': True, 'arguments': payload.get('args') if key == 'toolCall' else None,
                        'result': response.get('response') if response is not None else None,
                        'status': 'result recorded' if response is not None else 'call recorded'})
    return events


def profile_captures(state, entries):
    included = {entry.message_id for entry in entries}
    captures = {}
    for stored in state.raw_messages:
        message, meta = stored.message, stored.message.metadata
        if stored.db_id not in included or message.name != 'user_profile_fetch' or meta.get('tool_phase') != 'result':
            continue
        output = (meta.get('tool_payload') or {}).get('output') or {}
        for profile in output.get('profiles', []):
            actor = canonical_actor_id(str(profile.get('actor_id') or 'unknown'))
            captures.setdefault(actor, []).append({'message_id': stored.db_id,
                'captured_at': output.get('as_of') or meta.get('sent_at'), 'profile': profile})
    return captures


class ContextInspector:
    def __init__(self, runtime, reader):
        self.runtime, self.reader = runtime, reader
        self.settings = reader.settings
        self.timezone = runtime.config.default_metadata_timezone
        self._prepared = False

    async def prepare(self):
        if self._prepared:
            return
        self.state = await self.reader.state()
        self.provider = self.runtime._require_provider(self.settings.provider)
        self.entries = self.runtime.project_context(self.state, settings=self.settings, provider_name=self.provider.name)
        catalog = getattr(self.runtime.tool_registry, 'sticker_catalog', None)
        self.catalog_stats = catalog.stats() if catalog else {}
        self.tools = self.runtime._request_tools(self.settings) if self.provider.capabilities.function_tools else []
        self.instructions = build_system_prompt(self.settings, timezone=self.timezone)
        raw = self.provider.estimate_request_tokens(settings=self.settings,
            messages=[entry.message for entry in self.entries], instructions=self.instructions, tools=self.tools)
        self.calibration = await self.reader.calibration()
        self.estimate = raw.scaled(self.calibration)
        self.attribution = inspect_history(self.provider, self.settings, self.entries, raw, self.estimate)
        self.captures = profile_captures(self.state, self.entries)
        self._prepared = True

    async def overview(self):
        await self.prepare()
        selected = {entry.block_id for entry in self.entries if entry.block_id is not None}
        stats, remote = self.catalog_stats, getattr(self.runtime.tool_registry, 'remote_workspace', None)
        breakdown = self.attribution
        return {**self.runtime._describe_settings_values(self.settings, self.provider),
            'scope': self.reader.scope, 'as_of': format_timestamp(self.reader.as_of, self.timezone),
            'memory_jobs': await self.reader.jobs(),
            'memory_last_error': getattr(getattr(self.runtime.memory, 'worker', None), 'last_error', None),
            'semantic_enabled': bool(self.runtime.memory is not None and self.runtime.memory.embeddings.enabled),
            'raw_messages': len(self.state.raw_messages),
            'tool_history_messages': sum(item.message.role == MessageRole.TOOL for item in self.state.raw_messages),
            'memory_blocks': len(self.state.blocks), 'selected_memory_blocks': len(selected),
            'l0_blocks': sum(block.level == 0 for block in self.state.blocks),
            'l1_blocks': sum(block.level == 1 for block in self.state.blocks),
            'l2_blocks': sum(block.level == 2 for block in self.state.blocks),
            'estimated_history_tokens': self.estimate.history_tokens,
            'estimated_request_tokens': self.estimate.total_tokens,
            'estimator_multiplier': self.calibration,
            'estimated_request_images': self.state.estimated_images,
            'input_composition': breakdown['categories'] if breakdown else None,
            'projected_images': breakdown['images'] if breakdown else None,
            'summary_tokens': sum(row['categories']['memory'] for row in breakdown['entries'] if row['block_id'] is not None) if breakdown else None,
            'provider_history_messages': len(self.entries), 'available_tools': [tool.name for tool in self.tools],
            'loaded_in_memory': self.reader.session_id in self.runtime._live_sessions,
            'sticker_index_loaded': stats.get('loaded', False), 'sticker_index_count': stats.get('stickers', 0),
            'sticker_pack_count': stats.get('packs', 0), 'remote_enabled': bool(remote and remote.enabled),
            'remote_master_ready': bool(remote and remote.enabled and getattr(remote, '_master_started', False))}

    async def summaries(self, *, all_blocks=False, limit=5, before_id=None):
        await self.prepare()
        selected = {entry.block_id for entry in self.entries if entry.block_id is not None}
        parents = {parent for block in self.state.blocks for parent in block.parent_block_ids}
        blocks = [block for block in self.state.blocks if all_blocks or block.block_id in selected]
        if before_id is not None:
            anchor = await self.reader.block(before_id, current_context=True)
            blocks = [block for block in blocks if (block.sequence_no, block.block_id) < (anchor['sequence_no'], before_id)]
        blocks.sort(key=lambda block: (block.sequence_no, block.block_id), reverse=True)
        estimated = {row['block_id']: row['tokens'] for row in self.attribution['entries'] if row['block_id']} if self.attribution else {}
        return {'items': [{**asdict(block), 'included': block.block_id in selected, 'parent': block.block_id in parents,
                            'request_tokens': estimated.get(block.block_id)} for block in blocks[:limit]],
                'next': blocks[limit - 1].block_id if len(blocks) > limit else None,
                'all': all_blocks, 'selected': len(selected), 'stored': len(self.state.blocks)}

    async def tools_view(self, *, limit=5, before_id=None, name=None):
        await self.prepare()
        if before_id is not None:
            await self.reader.message(before_id, current_context=True)
        items = []
        calls = {row.db_id: row for row in self.state.raw_messages if row.message.metadata.get('tool_phase') == 'call'}
        call_keys = {}
        for row in self.state.raw_messages:
            if row.db_id in calls:
                key = (row.message.name, row.message.metadata.get('tool_batch_id'),
                       (row.message.metadata.get('tool_payload') or {}).get('call_id'))
                call_keys.setdefault(key, []).append(row.db_id)
        resolved = {}

        async def call_result(call_id):
            if call_id not in calls:
                return None
            if call_id not in resolved:
                resolved[call_id] = await self.reader.tool_result(
                    {'message': calls[call_id], 'context_id': self.reader.scope['context_id']})
            return resolved[call_id]
        included = {entry.message_id for entry in self.entries}
        for row in reversed(self.state.raw_messages):
            if before_id is not None and row.db_id >= before_id:
                continue
            message, meta = row.message, row.message.metadata
            events = [] if meta.get('provider_native_skip_same_provider') else native_tools(message)
            if message.role == MessageRole.TOOL and meta.get('tool_phase') == 'call' and (not name or message.name == name):
                payload = meta.get('tool_payload') or {}
                result = await call_result(row.db_id)
                output = (result['message'].message.metadata.get('tool_payload') or {}).get('output') if result else None
                events.insert(0, {'name': message.name, 'arguments': payload.get('arguments'),
                    'result': output, 'result_id': result['message'].db_id if result else None, 'native': False,
                    'status': (str(output.get('delivery_state') or output.get('status') or ('failed' if output.get('ok') is False else 'result recorded'))
                               if isinstance(output, dict) else 'result recorded' if result else 'result not recorded')})
            if name:
                events = [event for event in events if event['name'] == name]
            if message.role == MessageRole.TOOL and meta.get('tool_phase') == 'result' and (not name or message.name == name):
                payload = meta.get('tool_payload') or {}
                owner = meta.get('tool_call_message_id')
                legacy_calls = call_keys.get((message.name, meta.get('tool_batch_id'), payload.get('call_id')), [])
                candidate = int(owner) if owner is not None else next(
                    (call_id for call_id in reversed(legacy_calls) if call_id < row.db_id), None)
                linked = await call_result(candidate)
                paired = linked is not None and linked['message'].db_id == row.db_id
                if not paired:
                    output = payload.get('output')
                    events.append({'name': message.name, 'arguments': None, 'result': output, 'native': False,
                        'status': ('failed' if isinstance(output, dict) and output.get('ok') is False else 'result recorded') +
                                  ' · unlinked call'})
            if message.role == MessageRole.TOOL and meta.get('tool_phase') == 'delivery' and (not name or message.name == name):
                payload = meta.get('tool_payload') or {}
                events.append({'name': message.name, 'arguments': None, 'result': payload,
                    'native': False, 'status': 'delivery: ' + str(payload.get('delivery_state') or 'unknown')})
            if events:
                items.append({'message_id': row.db_id, 'at': message.metadata.get('sent_at'),
                              'included': row.db_id in included, 'events': events})
            if len(items) > limit:
                break
        return {'items': items[:limit], 'next': items[limit - 1]['message_id'] if len(items) > limit else None, 'name': name}

    async def profiles(self, *, limit=5, after=None, actor=None):
        await self.prepare()
        if actor is not None:
            actors, next_actor = [canonical_actor_id(actor)], None
        else:
            actors, next_actor = await self.reader.profile_actors(limit=limit, after=after, included=self.captures)
        snapshot = await self.reader.profiles(actors)
        return {'items': [{'profile': present_profile(document, timezone=self.timezone),
                          'updated_at': snapshot['updated_at'].get(document['actor_id']),
                          'pending': snapshot['pending_material'][document['actor_id']],
                          'snapshots': self.captures.get(document['actor_id'], [])}
                         for document in snapshot['profiles']],
                'next': actor_reference(next_actor) if next_actor else None,
                'as_of': snapshot['as_of'], 'detail': actor is not None}


async def inspect_context(runtime, session_id, topic='overview', *, limit=5, before_id=None,
                          all_blocks=False, actor=None, after=None, name=None, object_id=None):
    # Loading immutable published catalog data preserves cold/warm tool parity.
    # It performs no sticker generation, embedding or workspace synchronization.
    catalog = getattr(runtime.tool_registry, 'sticker_catalog', None)
    if catalog is not None and topic not in {'recent', 'message', 'block'}:
        await catalog.aensure_loaded()
    async with inspection_snapshot(runtime.store, session_id, runtime.config.default_session_settings(),
            timezone=runtime.config.default_metadata_timezone) as reader:
        view = ContextInspector(runtime, reader)
        if topic == 'overview':
            result = await view.overview()
        elif topic == 'recent':
            result = await reader.recent(limit=limit, before_id=before_id)
        elif topic == 'message':
            result = await reader.message(object_id)
            if result['message'].message.metadata.get('tool_phase') == 'call':
                result['tool_result'] = await reader.tool_result(result)
            result['native_tools'] = native_tools(result['presentation'].message)
        elif topic == 'block':
            result = await reader.block(object_id)
        elif topic == 'summaries':
            result = await view.summaries(all_blocks=all_blocks, limit=limit, before_id=before_id)
        elif topic == 'tools':
            result = await view.tools_view(limit=limit, before_id=before_id, name=name)
        elif topic in {'profiles', 'profile'}:
            result = await view.profiles(limit=limit, after=after, actor=actor)
        elif topic in {'full', 'report'}:
            result = {'overview': await view.overview(), 'instructions': view.instructions,
                      'preset': reader.settings.system_prompt,
                      'tool_definitions': request_tool_definitions(view.provider, view.settings, view.tools),
                      'timeline': view.entries if topic == 'full' else [],
                      'recent': await reader.recent(limit=limit),
                      'summaries': await view.summaries(limit=limit),
                      'profiles': await view.profiles(limit=limit),
                      'tools': await view.tools_view(limit=limit), 'complete_context': topic == 'full'}
        else:
            raise ValueError(f'Unknown context view: {topic}')
        return {'topic': topic, 'data': result, 'timezone': view.timezone,
                'as_of': format_timestamp(reader.as_of, view.timezone)}


def request_tool_definitions(provider, settings, tools):
    kind = getattr(provider, 'inspection_format', None)
    if kind == 'gemini':
        definitions, config, _native = provider._build_request_tools(settings, tools)
        return {'tools': definitions, 'toolConfig': config}
    if kind == 'responses':
        return {'tools': provider._tool_defs_for_request(settings, tools)}
    # Chat Completions wraps generic function declarations on dispatch.
    return {'tools': [{'type': 'function', 'function': tool.generic_function_declaration()} for tool in tools]}
