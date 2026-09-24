"""Explicit source-backed memory and profiles. PostgreSQL owns read visibility."""
from __future__ import annotations

import asyncio
from datetime import datetime
import json
import logging
from typing import Any

from tgchatbot.domain.models import ToolResult
from tgchatbot.domain.profiles import chat_profile
from tgchatbot.domain.identities import actor_reference, canonical_actor_id
from tgchatbot.domain.provenance import (AGENT_PRESENTATION_VERSION, evidence_part_spans, message_evidence,
                                      present_image_evidence, present_tool_output)
from tgchatbot.domain.timestamps import parse_time_filter, resolve_timezone
from tgchatbot.tools.base import ToolContext, ToolSpec
from tgchatbot.storage.postgres_store import message_body
from tgchatbot.operational import MemoryConfig, from_env

logger = logging.getLogger(__name__)


def _unseen_fragments(fragment: dict, displayed: list[dict]):
    """Only newly displayed original characters consume the shared allowance."""
    start = fragment['offset']
    end = start + len(fragment['text'])
    cursor = start
    for previous in sorted(displayed, key=lambda item: item['offset']):
        if previous['offset'] >= end:
            break
        previous_end = previous['offset'] + len(previous['text'])
        if previous_end <= cursor:
            continue
        if previous['offset'] > cursor:
            yield {'offset': cursor, 'text': fragment['text'][cursor - start:previous['offset'] - start]}
        cursor = max(cursor, previous_end)
    if cursor < end:
        yield {'offset': cursor, 'text': fragment['text'][cursor - start:]}


class MemoryService:
    def __init__(self, store, embeddings, *, config: MemoryConfig | None = None) -> None:
        self.store = store
        self.embeddings = embeddings
        self.config = config or from_env(MemoryConfig, 'MEMORY')
        self.tools = [MemorySearchTool(self).spec, MemoryReadTool(self).spec, UserProfileFetchTool(self).spec]

    async def fetch_profiles(self, session_id: str, actor_ids: list[str], *, scope=None,
                             timezone: str | None = None,
                             include_agent_preferences: bool = True) -> dict[str, Any]:
        if not isinstance(actor_ids, list) or any(not isinstance(actor, str) or not actor.strip() for actor in actor_ids):
            raise ValueError('actor_ids must be an array of explicit stable actor IDs; use [] for agent preferences only')
        zone = resolve_timezone(timezone)
        subjects = list(dict.fromkeys(canonical_actor_id(actor.strip()) for actor in actor_ids))
        if include_agent_preferences and 'agent' not in subjects:
            subjects.append('agent')
        if scope is not None:
            await self.store.assert_scope(session_id, scope)
        snapshot = await self.store.fetch_profile_snapshot(session_id, subjects,
            expected_scope=scope)
        result = {'ok': True, 'as_of': snapshot['as_of'],
            'profiles': [chat_profile(profile) for profile in snapshot['profiles']]}
        return present_tool_output('user_profile_fetch', result, zone.key)

    async def search(self, session_id: str, query: str, *, scope=None, actor_id=None,
                     before=None, after=None, limit=None, timezone=None) -> dict[str, Any]:
        zone = resolve_timezone(timezone)
        before, after = parse_time_filter(before, zone.key), parse_time_filter(after, zone.key)
        if scope is not None:
            await self.store.assert_scope(session_id, scope)
        query = str(query).strip()
        if not query or len(query) > self.config.query_chars:
            raise ValueError(f'Search query must contain 1–{self.config.query_chars} characters')
        vector = None
        status = 'lexical only: embeddings are not configured'
        if self.embeddings.enabled:
            try:
                # A slow hosted query cannot hold a foreground database connection.
                vector = await asyncio.wait_for(self.embeddings.embed_query(query), timeout=self.config.query_timeout_s)
                status = 'lexical and semantic'
            except Exception as exc:
                logger.warning('memory.query_embedding_unavailable error=%s', type(exc).__name__)
                status = 'lexical only: embedding query failed; semantic coverage unavailable'
        rows = await self.store.search_excerpts(session_id, query, embedding=vector,
            model=self.embeddings.space_id if vector is not None else None,
            actor_id=canonical_actor_id(actor_id) if actor_id is not None else None, before=before, after=after,
            limit=self.config.search_results if limit is None else min(self.config.search_results, max(1, int(limit))))
        if scope is not None:
            await self.store.assert_scope(session_id, scope)
        # The store owns ranked sources and exact slices. Bound displayed text
        # here, then project originals once; rank and chronological order have
        # different purposes. Computational fields never enter the tool output.
        matches, selected, remaining = [], {}, self.config.response_chars
        for row in rows:
            allowance = min(self.config.search_result_chars, remaining)
            if allowance <= 0:
                break
            sources = row.get('sources', [row])
            match = {'message_ids': [source['id'] for source in sources]}
            for source in sources:
                entry = selected.setdefault(source['id'], {'source': source, 'fragments': []})
                for fragment in source['fragments']:
                    for unseen in list(_unseen_fragments(fragment, entry['fragments'])):
                        text = unseen['text'][:allowance]
                        allowance -= len(text)
                        remaining -= len(text)
                        if len(text) < len(unseen['text']):
                            match['truncated'] = True
                        if text:
                            entry['fragments'].append({'offset': unseen['offset'], 'text': text})
            matches.append(match)
        source_ids = list(selected)
        images = await self.store.describe_message_images(session_id, source_ids, expected_scope=scope)
        messages = []
        for message_id, entry in selected.items():
            source_images = images.get(message_id, [])
            remaining -= len(json.dumps(source_images, ensure_ascii=False))
            source = entry['source']
            messages.append(message_evidence(source, message_id=message_id, role=source['role'],
                fragments=entry['fragments'], total_characters=source['total_characters'], images=source_images,
                timezone=zone.key, presentation_version=AGENT_PRESENTATION_VERSION))
        # These originals provide conversational context, not additional ranked
        # matches or extensions of an excerpt's exact source spans.
        related = await self._related_context(session_id, [source_id for source_id in source_ids if source_id in images], source_ids,
            remaining=remaining, scope=scope, timezone=zone.key)
        messages.extend(related)
        messages.sort(key=lambda item: (datetime.fromisoformat(item['sent_at']).timestamp()
            if item.get('sent_at') else float('-inf'), item['message_id']))
        if scope is not None:
            await self.store.assert_scope(session_id, scope)
        output = {'ok': True, 'coverage': status, 'matches': matches, 'messages': messages}
        if related:
            output['related_context'] = [item['message_id'] for item in related]
        return output

    async def _related_context(self, session_id, seeds, ranked_ids, *, remaining, scope, timezone=None):
        if not seeds or remaining <= 2:
            return []
        from tgchatbot.storage.relationships import expand_message_ids
        related_ids = await expand_message_ids(self.store, session_id, seeds,
            limit=self.config.read_messages, expected_scope=scope, include_selected=False)
        ranked = set(ranked_ids)
        related_ids = [message_id for message_id in related_ids if message_id not in ranked]
        images = await self.store.describe_message_images(session_id, related_ids, expected_scope=scope)
        rows = await self.store.read_messages(session_id, list(images), limit=self.config.read_messages)
        by_id = {item.db_id: item for item in rows}
        result = []
        remaining -= 2  # The related-context array itself also uses the allowance.
        for message_id in related_ids:
            if message_id not in by_id:
                continue
            stored = by_id[message_id]
            text = message_body(stored.message)
            shown = text[:min(self.config.search_result_chars, remaining)]
            # Charge attribution, descriptors and JSON escaping as well as text.
            # Ranked text keeps its existing allowance; related context uses only
            # what remains, with complete originals available via memory_read.
            while True:
                item = message_evidence({**stored.message.metadata, 'parts': evidence_part_spans(stored.message)}, message_id=message_id,
                    role=stored.message.role, fragments=[{'offset': 0, 'text': shown}],
                    total_characters=len(text), images=images[message_id], timezone=timezone, original=stored.message,
                    presentation_version=AGENT_PRESENTATION_VERSION)
                size = len(json.dumps(item, ensure_ascii=False, default=str)) + (2 if result else 0)
                if size <= remaining or not shown:
                    break
                shown = shown[:max(0, len(shown) - (size - remaining))]
            if size <= remaining:
                result.append(item)
                remaining -= size
        return result

    async def read(self, session_id: str, message_ids: list[int] | None = None, *, scope=None, offset=0, length=None,
                   include_neighbors=False, timezone=None, profile_fact_ids: list[int] | None = None):
        message_ids = message_ids or []
        profile_fact_ids = profile_fact_ids or []
        if len(message_ids) > self.config.read_messages or any(int(value) <= 0 for value in message_ids):
            raise ValueError(f'Read 1–{self.config.read_messages} positive message IDs returned by memory_search')
        if not message_ids and not profile_fact_ids:
            raise ValueError('Provide message_ids or profile_fact_ids to read their original evidence')
        if any(int(value) <= 0 for value in profile_fact_ids):
            raise ValueError('profile_fact_ids must contain positive fact_id values from user_profile_fetch')
        offset, length = int(offset), self.config.read_chars if length is None else int(length)
        if offset < 0 or not 1 <= length <= self.config.read_chars:
            raise ValueError(f'Offset must be nonnegative; length must be 1–{self.config.read_chars} characters')
        if profile_fact_ids and scope is None:
            # Keep resolving a fact and reading its originals under one source
            # revision even for callers outside an active agent turn.
            scope = await self.store.get_existing_scope(session_id)
        if scope is not None:
            await self.store.assert_scope(session_id, scope)
        requested = list(dict.fromkeys(message_ids))
        facts = []
        if profile_fact_ids:
            facts = await self.store.resolve_profile_fact_sources(session_id, profile_fact_ids, expected_scope=scope)
            requested = list(dict.fromkeys([*requested, *(mid for fact in facts for mid in fact['source_ids'])]))
        # Existing original-read limits own text admission. A fact can cite many
        # originals; return their remaining IDs for explicit follow-up reads.
        message_ids = requested[:self.config.read_messages]
        deferred_ids = requested[self.config.read_messages:]
        if include_neighbors and message_ids:
            from tgchatbot.storage.relationships import expand_message_ids
            message_ids = await expand_message_ids(self.store, session_id, message_ids,
                limit=self.config.read_messages, expected_scope=scope)
        rows = await self.store.read_messages(session_id, message_ids, limit=self.config.read_messages)
        by_id = {item.db_id: item for item in rows}
        rows = [by_id[mid] for mid in message_ids if mid in by_id]
        images = await self.store.describe_message_images(session_id,
            [item.db_id for item in rows], expected_scope=scope)
        remaining = self.config.response_chars
        results = []
        for item in rows:
            text = message_body(item.message)
            end = offset + min(length, remaining)
            excerpt = text[offset:end]
            record = message_evidence({**item.message.metadata, 'parts': evidence_part_spans(item.message)}, message_id=item.db_id,
                role=item.message.role, fragments=[{'offset': offset, 'text': excerpt}],
                total_characters=len(text), timezone=timezone, original=item.message,
                images=images.get(item.db_id),
                presentation_version=AGENT_PRESENTATION_VERSION)
            if end < len(text):
                record['next_offset'] = end
            results.append(record)
            remaining -= len(excerpt)
            if remaining <= 0:
                break
        if scope is not None:
            await self.store.assert_scope(session_id, scope)
        result = {'ok': True, 'messages': results,
            'unavailable_ids': sorted(set(requested[:self.config.read_messages]) - set(by_id)),
            'omitted_ids': list(dict.fromkeys([
                *(mid for mid in message_ids if mid in by_id and mid not in {item['message_id'] for item in results}),
                *deferred_ids]))}
        if profile_fact_ids:
            result['profile_facts'] = [{**fact, 'actor_id': actor_reference(fact['actor_id'])} for fact in facts]
            result['unavailable_profile_fact_ids'] = sorted(set(profile_fact_ids) - {fact['fact_id'] for fact in facts})
        return result



class MemorySearchTool:
    def __init__(self, memory: MemoryService) -> None:
        self.memory = memory
        self.spec = ToolSpec('memory_search',
            'Find earlier messages in this chat. Use a natural-language query and optional participant or time filters. '
            'Ordered matches identify contributing message_ids; messages contains their exact fragments in chronological order, each with its speaker and time. '
            'quoted_lines marks 1-based inclusive quoted line ranges, distinct from the sender\'s new assertions. '
            'Use memory_read for omitted text, nearby context or selected images; the returned fragments may already suffice. '
            'Image references belong to their originals; related_context lists nearby originals separately from ranked matches.',
            {'type': 'object', 'properties': {
                'query': {'type': 'string', 'description': 'Describe the event, fact or exchange you need; include distinctive words when known.'},
                'actor_id': {'type': 'string', 'description': 'Optional participant reference from message provenance, e.g. person_id:123. Finds passages involving this participant; surrounding sources retain their own speakers. Omit when unknown.'},
                'after': {'type': 'string', 'description': 'Inclusive ISO timestamp; unzoned values use the conversation timezone'},
                'before': {'type': 'string', 'description': 'Exclusive ISO timestamp; unzoned values use the conversation timezone'}},
             'required': ['query'], 'additionalProperties': False}, self)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            output = await self.memory.search(ctx.session_id, args.get('query', ''), scope=ctx.scope,
                actor_id=args.get('actor_id'), before=args.get('before'), after=args.get('after'),
                timezone=ctx.timezone)
        except (ValueError, TypeError) as exc:
            output = {'ok': False, 'error': str(exc)}
        return ToolResult('', self.spec.name, output)


class MemoryReadTool:
    def __init__(self, memory: MemoryService) -> None:
        self.memory = memory
        self.spec = ToolSpec('memory_read',
            'Read original message_ids or the supporting originals for profile_fact_ids from user_profile_fetch. '
            'Returns the same message records as memory_search. '
            'Paginate long text with offset and length, following next_offset to continue a partial read. '
            'include_neighbors adds reply and nearby context. '
            'Images remain descriptions unless image_ids selects them for visual examination. '
            'Each selected image must belong to an explicitly requested original, not an incidental neighbor. '
            'Results report unavailable or omitted evidence.',
            {'type': 'object', 'properties': {
                'message_ids': {'type': 'array', 'items': {'type': 'integer'},
                    'description': 'Original message_id values from evidence records or provenance.'},
                'profile_fact_ids': {'type': 'array', 'items': {'type': 'integer'},
                    'description': 'Optional fact_id values from user_profile_fetch. Opens their supporting originals; current=false marks historical facts. Use returned omitted_ids for further reads.'},
                'image_ids': {'type': 'array', 'items': {'type': 'string'},
                    'description': 'Optional image references beside those originals; include each owning original in message_ids. Absent or empty keeps the read text-only.'},
                'offset': {'type': 'integer', 'description': 'Zero-based character offset in each requested original; '
                    'use its returned next_offset to continue reading.'},
                'length': {'type': 'integer', 'description': 'Maximum characters to read per original from offset, '
                    'within the configured shared response allowance.'},
                'include_neighbors': {'type': 'boolean', 'description': 'Also read explicit reply target and nearby messages in the same topic'}},
             'additionalProperties': False}, self)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        evidence_parts = []
        try:
            offset = args.get('offset')
            output = await self.memory.read(ctx.session_id, args.get('message_ids', []), scope=ctx.scope,
                offset=0 if offset is None else offset, length=args.get('length'),
                include_neighbors=args.get('include_neighbors') is True, timezone=ctx.timezone,
                profile_fact_ids=args.get('profile_fact_ids'))
            image_ids = args.get('image_ids')
            if image_ids is None:
                image_ids = []  # Strict provider schemas express omitted optional fields as null.
            if not isinstance(image_ids, list) or any(not isinstance(image_id, str) for image_id in image_ids):
                raise ValueError('image_ids must be an array of image references returned by memory_search or memory_read')
            if image_ids:
                selected = await self.memory.store.resolve_message_images(ctx.session_id,
                    args.get('message_ids') or [], image_ids, expected_scope=ctx.scope, timezone=ctx.timezone)
                output['image_results'] = selected['image_results']
                evidence_parts = [present_image_evidence(part, ctx.timezone,
                    presentation_version=AGENT_PRESENTATION_VERSION) for part in selected['evidence_parts']]
        except (ValueError, TypeError) as exc:
            output = {'ok': False, 'error': str(exc)}
        return ToolResult('', self.spec.name, output, evidence_parts=evidence_parts)


class UserProfileFetchTool:
    def __init__(self, memory: MemoryService) -> None:
        self.memory = memory
        self.spec = ToolSpec('user_profile_fetch',
            'Read stored, asynchronously updated profiles for explicit actor IDs and, by default, the agent\'s continuing style preferences. '
            'Fetching does not trigger learning. Claims retain kind, attribution and known validity dates; '
            'asserted_by defaults to the containing actor_id. Use memory_read(profile_fact_ids=[fact_id]) for supporting originals. '
            'Names do not select identities; empty facts do not establish absence of preferences.',
            {'type': 'object', 'properties': {
                'actor_ids': {'type': 'array', 'items': {'type': 'string'},
                    'description': 'Participant references from message provenance, e.g. person_id:123; [] fetches only agent preferences'},
                'include_agent_preferences': {'type': 'boolean', 'description': 'Include source-backed agent style preferences; defaults to true'}},
             'required': ['actor_ids'], 'additionalProperties': False}, self)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            include_agent = args.get('include_agent_preferences')
            output = await self.memory.fetch_profiles(ctx.session_id, args.get('actor_ids'), scope=ctx.scope,
                timezone=getattr(ctx, 'timezone', None),
                include_agent_preferences=True if include_agent is None else include_agent)
        except (ValueError, TypeError) as exc:
            output = {'ok': False, 'error': str(exc)}
        return ToolResult('', self.spec.name, output)
