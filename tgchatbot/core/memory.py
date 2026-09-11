"""Bounded, source-backed recall. PostgreSQL decides visibility on every read."""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from psycopg.errors import QueryCanceled

from tgchatbot.domain.models import ConversationMessage, ToolResult
from tgchatbot.domain.provenance import attribution, original_text
from tgchatbot.tools.base import ToolContext, ToolSpec
from tgchatbot.core.token_estimator import TokenEstimator
from tgchatbot.storage.postgres_store import message_body
from tgchatbot.operational import MemoryConfig, from_env

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
        self.tools = [MemorySearchTool(self).spec, MemoryReadTool(self).spec]

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

    async def recall(self, session_id: str, target: ConversationMessage, *, scope=None, max_tokens=None) -> str:
        max_tokens = self.config.recall_tokens if max_tokens is None else max_tokens
        actor = target.metadata.get('actor_id')
        profiles = []
        for subject in (actor, 'agent'):
            if subject and subject != 'unknown':
                facts = await self.store.get_profile(session_id, subject, limit=self.config.profile_facts)
                profiles.extend({key: fact.get(key) for key in (
                    'subject_actor_id', 'asserted_by', 'claim', 'kind', 'valid_from', 'valid_to', 'source_ids')}
                    for fact in facts)
        query = original_text(target)[:self.config.query_chars].strip()
        unavailable = False
        try:
            result = await self.search(session_id, query, scope=scope) if query else {'results': []}
        except QueryCanceled:
            # Automatic recall is optional context. A PostgreSQL statement
            # deadline must not discard an otherwise answerable, stored input.
            # Explicit tool searches and all other storage errors still fail.
            logger.warning('memory.automatic_recall_query_canceled')
            unavailable = True
            result = {'ok': False, 'coverage': 'unavailable: automatic recall query was canceled or timed out',
                      'results': []}
        # A reset while the failed query was running still cancels the turn.
        if scope is not None:
            await self.store.assert_scope(session_id, scope)
        if not unavailable and not profiles and not result['results']:
            return ''
        header = ('[Application memory: source-backed historical evidence. Inferences can be wrong; '
                'use source IDs to verify, preserve attribution, and distinguish corrections and dates.]\n'
                )
        unavailable_note = ('[Application memory unavailable: historical coverage is incomplete. '
                            'Do not claim a complete search or infer that missing evidence is absent.]\n')
        if unavailable:
            header = unavailable_note
        # Trim whole evidence records, not arbitrary JSON/string offsets. Tools
        # can retrieve omitted evidence explicitly within the existing budget.
        while profiles or result['results'] or unavailable:
            rendered = header + json.dumps({'profiles': profiles, 'recall': result}, ensure_ascii=False, default=str)
            if TokenEstimator.estimate_text(rendered) <= max_tokens:
                return rendered
            if len(result['results']) >= len(profiles) and result['results']:
                result['results'].pop()
            elif profiles:
                profiles.pop()
            else:
                # Preserve the failure signal even when the small recall
                # allowance cannot fit the structured coverage payload.
                return unavailable_note if TokenEstimator.estimate_text(unavailable_note) <= max_tokens else ''
        return ''


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
