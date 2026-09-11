"""Explicit source-backed memory and profiles. PostgreSQL owns read visibility."""
from __future__ import annotations

import asyncio
from datetime import datetime, timezone as utc_timezone
import logging
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from tgchatbot.domain.models import ToolResult
from tgchatbot.domain.provenance import attribution
from tgchatbot.tools.base import ToolContext, ToolSpec
from tgchatbot.storage.postgres_store import message_body
from tgchatbot.operational import MemoryConfig, from_env
from tgchatbot.settings_schema import DEFAULT_METADATA_TIMEZONE

logger = logging.getLogger(__name__)


def search_attribution(row: dict) -> dict:
    metadata = row.get('metadata') or {}
    result = {key: row.get(key, metadata.get(key)) for key in (
        'id', 'role', 'actor_id', 'actor_kind', 'actor_name', 'sent_at', 'source',
        'source_chat_id', 'source_message_id', 'topic_id', 'reply_to_source_id', 'reply_to_source_chat_id')}
    for key in ('reply_to_actor', 'forward_origin', 'external_reply', 'quote'):
        if metadata.get(key):
            result[key] = metadata[key]
    if metadata.get('quote'):
        result['contains_quote'] = True
    return result


class MemoryService:
    def __init__(self, store, embeddings, *, config: MemoryConfig | None = None) -> None:
        self.store = store
        self.embeddings = embeddings
        self.config = config or from_env(MemoryConfig, 'MEMORY')
        self.tools = [MemorySearchTool(self).spec, MemoryReadTool(self).spec, UserProfileFetchTool(self).spec]

    async def fetch_profiles(self, session_id: str, actor_ids: list[str], *, scope=None,
                             timezone: str = DEFAULT_METADATA_TIMEZONE,
                             include_agent_preferences: bool = True, before_fact_id: int | None = None) -> dict[str, Any]:
        if not isinstance(actor_ids, list) or any(not isinstance(actor, str) or not actor.strip() for actor in actor_ids):
            raise ValueError('actor_ids must be an array of explicit stable actor IDs; use [] for agent preferences only')
        if before_fact_id is not None and (not isinstance(before_fact_id, int) or isinstance(before_fact_id, bool) or before_fact_id <= 0):
            raise ValueError('before_fact_id must be a positive fact ID returned as next_before_fact_id')
        try:
            zone = ZoneInfo(timezone or DEFAULT_METADATA_TIMEZONE)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f'Unknown profile timestamp timezone: {timezone}') from exc
        subjects = list(dict.fromkeys(actor.strip() for actor in actor_ids))
        if include_agent_preferences and 'agent' not in subjects:
            subjects.append('agent')
        snapshot = await self.store.fetch_profile_snapshot(session_id, subjects, expected_scope=scope,
            before_fact_id=before_fact_id, limit=self.config.profile_facts)
        profiles = []
        for profile in snapshot['profiles']:
            source = profile['identity']
            agent = profile['actor_id'] == 'agent'
            identity = {'known': source is not None or (agent and snapshot['scope'] is not None),
                'actor_kind': source['actor_kind'] if source else 'agent' if agent else 'unknown',
                'actor_name': source['actor_name'] if source else None,
                'last_message': {key: value for key, value in source.items()
                                 if key not in {'actor_kind', 'actor_name'}} if source else None}
            facts = [{key: fact.get(key) for key in (
                'id', 'subject_actor_id', 'asserted_by', 'claim', 'kind', 'status', 'valid_from', 'valid_to',
                'supersedes', 'source_ids', 'source_revisions', 'created_at')} for fact in profile['facts']]
            status = ('available' if facts else 'unknown_identity' if not identity['known']
                      else 'no_more_facts' if before_fact_id else 'no_current_facts')
            profiles.append({**profile, 'identity': identity, 'facts': facts,
                'subject_kind': 'agent_preferences' if agent else 'actor',
                'status': status})
        result = {'ok': True, 'session_id': session_id,
            'generation': snapshot['scope']['generation'] if snapshot['scope'] is not None else None,
            'as_of': snapshot['as_of'], 'fetched_at': datetime.now(utc_timezone.utc), 'timezone': zone.key,
            'profiles': profiles}

        def encode_dates(value):
            if isinstance(value, datetime):
                return value.astimezone(zone).isoformat()
            if isinstance(value, dict):
                return {key: encode_dates(item) for key, item in value.items()}
            if isinstance(value, list):
                return [encode_dates(item) for item in value]
            return value

        return encode_dates(result)

    async def search(self, session_id: str, query: str, *, scope=None, actor_id=None,
                     before=None, after=None, limit=None) -> dict[str, Any]:
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
            actor_id=actor_id, before=before, after=after,
            limit=self.config.search_results if limit is None else min(self.config.search_results, max(1, int(limit))))
        if scope is not None:
            await self.store.assert_scope(session_id, scope)
        # Keep provider requests independent of archive size. Full source text is
        # available through memory_read; truncation is explicit in each result.
        results, remaining = [], self.config.response_chars
        for row in rows:
            text = str(row.get('text', ''))
            allowance = min(self.config.search_result_chars, remaining)
            if allowance <= 0:
                break
            item = {key: value for key, value in row.items()
                    if key not in {'embedding', 'source_revisions', 'metadata', 'sources', 'text'}}
            item.update(text=text[:allowance], truncated=len(text) > allowance)
            item.update(search_attribution(row))
            if 'sources' in row:
                item['sources'] = [search_attribution(source) for source in row['sources']]
            results.append(item)
            remaining -= len(item['text'])
        return {'ok': True, 'coverage': status, 'results': results}

    async def read(self, session_id: str, message_ids: list[int], *, scope=None, offset=0, length=None, include_neighbors=False):
        if len(message_ids) > self.config.read_messages or not message_ids or any(int(value) <= 0 for value in message_ids):
            raise ValueError(f'Read 1–{self.config.read_messages} positive message IDs returned by memory_search')
        offset, length = int(offset), self.config.read_chars if length is None else int(length)
        if offset < 0 or not 1 <= length <= self.config.read_chars:
            raise ValueError(f'Offset must be nonnegative; length must be 1–{self.config.read_chars} characters')
        if scope is not None:
            await self.store.assert_scope(session_id, scope)
        requested = list(message_ids)
        if include_neighbors:
            from tgchatbot.storage.relationships import expand_message_ids
            message_ids = await expand_message_ids(self.store, session_id, message_ids,
                limit=self.config.read_messages, expected_scope=scope)
        rows = await self.store.read_messages(session_id, message_ids, limit=self.config.read_messages)
        by_id = {item.db_id: item for item in rows}
        rows = [by_id[mid] for mid in message_ids if mid in by_id]
        remaining = self.config.response_chars
        results = []
        for item in rows:
            text = message_body(item.message)
            end = offset + min(length, remaining)
            excerpt = text[offset:end]
            results.append({**attribution(item.message, message_id=item.db_id), 'text': excerpt,
                'role': item.message.role.value,
                'offset': offset, 'next_offset': end if end < len(text) else None,
                'total_characters': len(text)})
            remaining -= len(excerpt)
            if remaining <= 0:
                break
        if scope is not None:
            await self.store.assert_scope(session_id, scope)
        return {'ok': True, 'messages': results,
            'unavailable_ids': sorted(set(requested) - set(by_id)),
            'omitted_ids': [mid for mid in message_ids if mid in by_id and mid not in {item['message_id'] for item in results}]}



class MemorySearchTool:
    def __init__(self, memory: MemoryService) -> None:
        self.memory = memory
        self.spec = ToolSpec('memory_search',
            'Search active original messages in this chat, including prior contexts. Returns source IDs and attribution. '
            'Use actor_id only when its stable identity is known. Full-reset audit data is inaccessible.',
            {'type': 'object', 'properties': {
                'query': {'type': 'string'}, 'actor_id': {'type': 'string'},
                'after': {'type': 'string', 'description': 'Inclusive ISO timestamp'},
                'before': {'type': 'string', 'description': 'Exclusive ISO timestamp'}},
             'required': ['query'], 'additionalProperties': False}, self)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            output = await self.memory.search(ctx.session_id, args.get('query', ''), scope=ctx.scope,
                actor_id=args.get('actor_id'), before=args.get('before'), after=args.get('after'))
        except (ValueError, TypeError) as exc:
            output = {'ok': False, 'error': str(exc)}
        return ToolResult('', self.spec.name, output)


class MemoryReadTool:
    def __init__(self, memory: MemoryService) -> None:
        self.memory = memory
        self.spec = ToolSpec('memory_read',
            'Read original active messages by the database IDs from memory_search. Paginate long text with offset. '
            'Unavailable IDs are not authorized or no longer active.',
            {'type': 'object', 'properties': {
                'message_ids': {'type': 'array', 'items': {'type': 'integer'}},
                'offset': {'type': 'integer'}, 'length': {'type': 'integer'},
                'include_neighbors': {'type': 'boolean', 'description': 'Also read explicit reply target and nearby messages in the same topic, within the configured read window'}},
             'required': ['message_ids'], 'additionalProperties': False}, self)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            output = await self.memory.read(ctx.session_id, args.get('message_ids', []), scope=ctx.scope,
                offset=args.get('offset', 0), length=args.get('length'),
                include_neighbors=args.get('include_neighbors') is True)
        except (ValueError, TypeError) as exc:
            output = {'ok': False, 'error': str(exc)}
        return ToolResult('', self.spec.name, output)


class UserProfileFetchTool:
    def __init__(self, memory: MemoryService) -> None:
        self.memory = memory
        self.spec = ToolSpec('user_profile_fetch',
            'Fetch current source-backed profiles for explicit stable actor IDs in this chat, plus agent style preferences. '
            'Call proactively when personal or style context matters and earlier profile evidence is absent or may be old '
            'relative to the conversation. Compare evidence/source dates with fetched_at; fetching does not learn or update facts. '
            'Names never select identities. Empty facts are not proof that a person has no preferences. '
            'Continue a truncated subject with its next_before_fact_id and that actor ID. Full-reset audit data is inaccessible.',
            {'type': 'object', 'properties': {
                'actor_ids': {'type': 'array', 'items': {'type': 'string'},
                    'description': 'Stable IDs from message provenance, such as telegram:user:123; [] fetches only agent preferences'},
                'include_agent_preferences': {'type': 'boolean', 'description': 'Include source-backed agent style preferences; defaults to true'},
                'before_fact_id': {'type': 'integer', 'description': 'Continue before a returned next_before_fact_id; omit for newest facts'}},
             'required': ['actor_ids'], 'additionalProperties': False}, self)

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            include_agent = args.get('include_agent_preferences')
            output = await self.memory.fetch_profiles(ctx.session_id, args.get('actor_ids'), scope=ctx.scope,
                timezone=getattr(ctx, 'timezone', DEFAULT_METADATA_TIMEZONE),
                include_agent_preferences=True if include_agent is None else include_agent,
                before_fact_id=args.get('before_fact_id'))
        except (ValueError, TypeError) as exc:
            output = {'ok': False, 'error': str(exc)}
        return ToolResult('', self.spec.name, output)
