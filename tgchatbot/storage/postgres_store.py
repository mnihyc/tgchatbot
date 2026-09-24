"""Conversation persistence: immutable originals and disposable, scoped projections.

Every ordinary read joins the session's current generation. Context reads additionally
match context_id. This is the reset boundary, including for unfinished background jobs.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence
import unicodedata
from urllib.parse import quote
import uuid

from psycopg import AsyncConnection, sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from tgchatbot.core.context_state import (CompactionWorkingSet, ImageRetirement, MemoryBlock, StoredConversationMessage,
    is_auto_note_message, matching_tool_result)
from tgchatbot.core.token_estimator import TokenEstimator
from tgchatbot.domain.models import (
    ChatMode, ConversationMessage, MessagePart, MessageRole, PartKind,
    ProcessVisibility, PromptInjectionMode, ResponseDelivery, SessionSettings,
    StickerMode, ToolHistoryMode,
)
from tgchatbot.operational import from_env
from tgchatbot.domain.provenance import AGENT_PRESENTATION_VERSION
from tgchatbot.domain.attachments import generated_attachment_reference
from tgchatbot.domain.identities import actor_observation

SCHEMA_VERSION = 3
EMBEDDING_DIMENSIONS = 1536
_BLOCK_SUMMARY_COLUMNS = ('b.id,b.sequence_no,b.summary_text,b.estimated_tokens,b.details,'
    'cardinality(b.source_ids) AS source_count')
_SCHEMA_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CJK = r"\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0002ffff"
_TERMS = re.compile(rf"[{_CJK}]+|[^\W_{_CJK}]+(?:[_-][^\W_{_CJK}]+)*", re.UNICODE)
_CANONICAL_COLUMNS = (
    'source', 'source_chat_id', 'source_message_id', 'actor_id', 'actor_kind',
    'actor_name', 'topic_id', 'reply_to_source_id',
)
_ENUM_SETTINGS = {
    'mode': ChatMode, 'process_visibility': ProcessVisibility,
    'response_delivery': ResponseDelivery, 'sticker_mode': StickerMode,
    'prompt_injection_mode': PromptInjectionMode, 'tool_history_mode': ToolHistoryMode,
}


@dataclass(frozen=True)
class DatabaseConfig:
    """Optional environment policy; SQL/schema validity is enforced separately."""
    pool_min_size: int | None = None
    pool_max_size: int | None = None
    pool_timeout_s: float | None = None
    statement_timeout_s: float | None = None
    lock_timeout_s: float | None = None
    initialization_timeout_s: float | None = None
    read_page_size: int = 100
    recent_page_size: int = 20
    history_page_size: int = 1000
    revision_page_size: int = 20
    memory_block_page_size: int = 128
    search_results: int = 20
    profile_results: int = 50
    job_claim_size: int = 1
    job_lease_seconds: float = 60.0
    job_max_attempts: int = 3
    job_retry_delay_seconds: float = 5.0
    job_cleanup_page_size: int = 1000
    job_retention_seconds: float = 86400.0
    retirement_page_size: int = 1000
    retired_batch_lease_seconds: float = 900.0
    retired_batch_poll_seconds: float = 60.0
    relationship_neighbors: int = 2
    relationship_read_limit: int = 20

    def __post_init__(self) -> None:
        zero_allowed = {'pool_min_size', 'statement_timeout_s', 'lock_timeout_s',
            'initialization_timeout_s', 'job_retry_delay_seconds', 'job_retention_seconds',
            'retired_batch_poll_seconds', 'relationship_neighbors'}
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            if isinstance(value, bool) or not math.isfinite(value) or value < 0 or (value == 0 and field.name not in zero_allowed):
                raise ValueError(f'MEMORY_DB_{field.name.upper()} must be finite and '
                    + ('nonnegative' if field.name in zero_allowed else 'positive'))


class StaleScopeError(ValueError):
    """A reset, rollback, source edit, or expired job invalidated a write."""


def lexical_terms(text: str) -> list[str]:
    """Case-fold words and CJK bigrams without crossing punctuation/language gaps."""
    terms: list[str] = []
    for match in _TERMS.finditer(unicodedata.normalize('NFKC', text).casefold()):
        term = match.group()
        if re.fullmatch(rf"[{_CJK}]+", term):
            terms.extend(term[index:index + 2] for index in range(len(term) - 1))
            if len(term) == 1:
                terms.append(term)
        else:
            terms.append(term)
            terms.extend(piece for piece in re.split('[_-]', term) if piece != term and piece)
    # PostgreSQL lexemes have a 2046-byte format limit. Hash only oversized
    # normalized terms into a namespace the tokenizer cannot otherwise emit.
    # Originals remain intact, and queries use the identical equality encoding.
    encoded = (term if len(term.encode('utf-8')) <= 2046 else
        '!sha256:' + hashlib.sha256(term.encode('utf-8')).hexdigest() for term in terms)
    return list(dict.fromkeys(encoded))


def message_body(message: ConversationMessage) -> str:
    """The exact canonical body to which excerpt character spans refer."""
    return '\n'.join(part.text for part in message.parts if part.text is not None)


def _timestamp(value: Any) -> datetime | None:
    if value is None or value == '':
        return None
    if isinstance(value, datetime):
        result = value
    elif isinstance(value, (float, int)):
        result = datetime.fromtimestamp(value, timezone.utc)
    else:
        result = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return result.replace(tzinfo=timezone.utc) if result.tzinfo is None else result.astimezone(timezone.utc)


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def _ids(values: Iterable[int]) -> list[int]:
    return sorted({int(value) for value in values if int(value) > 0})


def _limit(value: int | None, default: int) -> int:
    result = default if value is None else int(value)
    if result < 0:
        raise ValueError('Pagination length must be nonnegative')
    return result


def _duration(value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError('Duration must be finite and ' + ('positive' if positive else 'nonnegative'))
    return float(value)


def _vector(values: Sequence[float] | None) -> str | None:
    if values is None:
        return None
    if len(values) != EMBEDDING_DIMENSIONS:
        raise ValueError(f'embedding must have {EMBEDDING_DIMENSIONS} dimensions')
    values = [float(value) for value in values]
    if not all(math.isfinite(value) for value in values):
        raise ValueError('embedding values must be finite')
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        raise ValueError('embedding must not be zero')
    return '[' + ','.join(str(value / norm) for value in values) + ']'


class PostgresStore:
    def __init__(self, dsn: str, *, schema: str = 'public',
                 config: DatabaseConfig | None = None) -> None:
        if not _SCHEMA_NAME.fullmatch(schema):
            raise ValueError('invalid PostgreSQL schema name')
        self.dsn = dsn
        self.schema = schema
        self.config = config if config is not None else from_env(DatabaseConfig, 'MEMORY_DB')
        pool_options = {name: value for name, value in (
            ('min_size', self.config.pool_min_size), ('max_size', self.config.pool_max_size),
            ('timeout', self.config.pool_timeout_s)) if value is not None}
        self.pool = AsyncConnectionPool(
            dsn, open=False, kwargs={'row_factory': dict_row}, configure=self._configure, **pool_options,
        )
        self._initialized = False
        self._read_only = False
        self._initialize_lock = asyncio.Lock()

    async def _configure(self, conn: AsyncConnection) -> None:
        await conn.execute(sql.SQL('SET search_path TO {}, public').format(sql.Identifier(self.schema)))
        await conn.execute("SET TIME ZONE 'UTC'")
        for name, value in (('statement_timeout', self.config.statement_timeout_s),
                            ('lock_timeout', self.config.lock_timeout_s)):
            if value is not None:
                await conn.execute('SELECT set_config(%s,%s,false)', (name, f'{value * 1000:g}ms'))
        # Chat and actor cardinalities vary greatly. Reusing a generic prepared
        # plan can turn a selective lookup into a multi-second full scan.
        await conn.execute("SET plan_cache_mode = 'force_custom_plan'")
        # Inline vector scans and many tiny GIN probes have high estimated cost
        # but short execution time. LLVM compilation otherwise dominates them.
        await conn.execute('SET jit = off')
        if self._read_only:
            await conn.execute('SET default_transaction_read_only = on')
        await conn.commit()

    async def open_readonly(self) -> None:
        """Connect to an existing schema for operator inspection, without DDL."""
        self._read_only = True
        await self.pool.open(wait=True)
        async with self.pool.connection() as conn:
            await self._check_schema(conn)

    async def read_session_scope(self, conn, session_id):
        """Start a source read: live callers lock; operator callers pin a snapshot."""
        if self._read_only:
            await conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        lock = '' if self._read_only else ' FOR SHARE'
        return await (await conn.execute('SELECT * FROM sessions WHERE session_id=%s' + lock,
                                        (session_id,))).fetchone()

    async def initialize(self) -> None:
        async with self._initialize_lock:
            if self._initialized:
                return
            await self.pool.open(wait=True)
            async with self.pool.connection() as conn:
                if self.config.initialization_timeout_s is not None:
                    await conn.execute("SELECT set_config('statement_timeout',%s,true)",
                        (f'{self.config.initialization_timeout_s * 1000:g}ms',))
                await conn.execute('CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public')
                await conn.execute(sql.SQL('CREATE SCHEMA IF NOT EXISTS {}').format(sql.Identifier(self.schema)))
                # Serializes two starts without depending on an initially absent table.
                await conn.execute('SELECT pg_advisory_xact_lock(hashtext(%s))', (f'tgchatbot-schema:{self.schema}',))
                await self._check_schema(conn)
                await conn.execute(Path(__file__).with_name('postgres_schema.sql').read_text())
            self._initialized = True

    async def close(self) -> None:
        await self.pool.close()
        self._initialized = False

    async def _check_schema(self, conn: AsyncConnection) -> None:
        row = await (await conn.execute('SELECT to_regclass(%s) AS relation', (f'{self.schema}.schema_version',))).fetchone()
        if row['relation'] is None:
            existing = await (await conn.execute(
                'SELECT tablename FROM pg_tables WHERE schemaname=%s AND tablename IN (\'sessions\', \'messages\', \'message_revisions\')',
                (self.schema,),
            )).fetchone()
            if existing:
                raise RuntimeError('unversioned conversation schema; refusing automatic changes')
            return
        row = await (await conn.execute(sql.SQL('SELECT max(version) AS version FROM {}.schema_version').format(sql.Identifier(self.schema)))).fetchone()
        if row['version'] != SCHEMA_VERSION:
            raise RuntimeError(f'incompatible conversation schema version {row["version"]!r}; expected {SCHEMA_VERSION}')

    @staticmethod
    def _scope(row: Mapping[str, Any]) -> dict[str, int]:
        return {key: int(row[key]) for key in ('generation', 'context_id', 'revision')}

    @staticmethod
    def _check_scope(current: Mapping[str, Any], expected: Mapping[str, Any] | None, *, context: bool = True,
                      revision: bool = True) -> None:
        if expected is None:
            return
        keys = ('generation', 'context_id', 'revision') if context and revision else ('generation', 'context_id') if context else ('generation',)
        if any(key in expected and int(current[key]) != int(expected[key]) for key in keys):
            raise StaleScopeError('conversation scope changed')

    async def _session(self, conn: AsyncConnection, session_id: str, *, lock: bool = False) -> dict[str, Any]:
        await conn.execute('INSERT INTO sessions (session_id) VALUES (%s) ON CONFLICT DO NOTHING', (session_id,))
        return await (await conn.execute('SELECT * FROM sessions WHERE session_id=%s' + (' FOR UPDATE' if lock else ''), (session_id,))).fetchone()

    async def get_scope(self, session_id: str) -> dict[str, int]:
        async with self.pool.connection() as conn:
            return self._scope(await self._session(conn, session_id))

    async def get_existing_scope(self, session_id: str) -> dict[str, int] | None:
        """Read a conversation boundary without creating a missing session."""
        async with self.pool.connection() as conn:
            row = await (await conn.execute('SELECT generation,context_id,revision FROM sessions '
                'WHERE session_id=%s', (session_id,))).fetchone()
        return self._scope(row) if row is not None else None

    async def assert_scope(self, session_id: str, expected_scope: Mapping[str, Any], *,
                           generation_only: bool = False) -> dict[str, int]:
        scope = await self.get_existing_scope(session_id)
        if scope is None:
            raise StaleScopeError('conversation scope is unavailable')
        self._check_scope(scope, expected_scope, context=not generation_only)
        return scope

    async def get_or_create_session(self, session_id: str, defaults: SessionSettings) -> SessionSettings:
        async with self.pool.connection() as conn:
            row = await self._session(conn, session_id, lock=True)
            payload = row['settings']
            if not payload:
                payload = asdict(defaults)
                await conn.execute('UPDATE sessions SET settings=%s WHERE session_id=%s', (Jsonb(payload), session_id))
        return self.decode_settings(payload, defaults)

    @staticmethod
    def decode_settings(payload: dict, defaults: SessionSettings) -> SessionSettings:
        values = asdict(defaults)
        values.update({key: value for key, value in payload.items() if key in values})
        for key, enum in _ENUM_SETTINGS.items():
            try:
                values[key] = enum(values[key])
            except (ValueError, TypeError):
                values[key] = getattr(defaults, key)
        return SessionSettings(**values)

    async def save_session(self, session_id: str, settings: SessionSettings) -> None:
        async with self.pool.connection() as conn:
            await conn.execute('''INSERT INTO sessions (session_id,settings) VALUES (%s,%s)
                ON CONFLICT (session_id) DO UPDATE SET settings=excluded.settings,updated_at=now()''',
                (session_id, Jsonb(asdict(settings))))

    async def bind_message_source(self, session_id: str, message_id: int, *, source: str,
                                   source_chat_id: str, source_message_ids: Sequence[str],
                                   actor_id: str, actor_kind: str, actor_name: str,
                                   expected_scope: Mapping[str, Any] | None = None) -> None:
        from tgchatbot.storage.source_aliases import bind_message_source
        await bind_message_source(self, session_id, message_id, source=source,
            source_chat_id=source_chat_id, source_message_ids=source_message_ids,
            actor_id=actor_id, actor_kind=actor_kind, actor_name=actor_name, expected_scope=expected_scope)

    async def count_sessions(self) -> int:
        async with self.pool.connection() as conn:
            return int((await (await conn.execute('SELECT count(*) AS count FROM sessions')).fetchone())['count'])

    async def list_session_ids(self) -> list[str]:
        async with self.pool.connection() as conn:
            return [row['session_id'] for row in await (await conn.execute('SELECT session_id FROM sessions')).fetchall()]

    async def get_sticker_persona(self, session_id: str) -> dict[str, Any] | None:
        async with self.pool.connection() as conn:
            row = await (await conn.execute('SELECT sticker_persona FROM sessions WHERE session_id=%s', (session_id,))).fetchone()
            return row['sticker_persona'] if row else None

    async def save_sticker_persona(self, session_id: str, persona: dict[str, Any] | None, *,
                                   expected_scope: Mapping[str, Any] | None = None) -> None:
        async with self.pool.connection() as conn:
            scope = await self._session(conn, session_id, lock=True)
            self._check_scope(scope, expected_scope)
            await conn.execute('UPDATE sessions SET sticker_persona=%s WHERE session_id=%s', (Jsonb(persona) if persona else None, session_id))

    async def clear_sticker_persona(self, session_id: str, *, expected_scope: Mapping[str, Any] | None = None) -> None:
        await self.save_sticker_persona(session_id, None, expected_scope=expected_scope)

    async def reset_context(self, session_id: str) -> dict[str, int]:
        async with self.pool.connection() as conn:
            scope = await self._session(conn, session_id, lock=True)
            from tgchatbot.storage.profiles import request_catchup
            await request_catchup(conn, session_id=session_id, scope=scope)
            row = await (await conn.execute('''UPDATE sessions SET context_id=context_id+1,
                revision=revision+1,context_version=context_version+1,compaction_version=0,
                profile_refresh_version=0,updated_at=now() WHERE session_id=%s RETURNING *''', (session_id,))).fetchone()
            return self._scope(row)

    async def clear_messages(self, session_id: str) -> None:
        await self.reset_context(session_id)

    async def reset_full(self, session_id: str, defaults: SessionSettings) -> dict[str, int]:
        async with self.pool.connection() as conn:
            await self._session(conn, session_id, lock=True)
            await conn.execute('''INSERT INTO agent_generations
                (session_id,generation,context_id,scope_revision,settings,sticker_persona)
                SELECT session_id,generation,context_id,revision,settings,sticker_persona FROM sessions
                WHERE session_id=%s''', (session_id,))
            row = await (await conn.execute('''UPDATE sessions SET generation=generation+1,
                context_id=context_id+1,revision=revision+1,context_version=context_version+1,compaction_version=0,profile_refresh_version=0,settings=%s,sticker_persona=NULL,updated_at=now()
                WHERE session_id=%s RETURNING *''', (Jsonb(asdict(defaults)), session_id))).fetchone()
            # Reset stays constant-size. A current-generation job retires old
            # vectors in bounded transactions, including earlier unfinished resets.
            await conn.execute('''INSERT INTO jobs
                (session_id,generation,context_id,scope_revision,kind,policy,payload,dedupe_key)
                VALUES (%s,%s,%s,%s,'memory_retire','memory',%s,%s)''',
                (session_id, row['generation'], row['context_id'], row['revision'],
                Jsonb({'before_generation': row['generation']}), f"before:{row['generation']}"))
            return self._scope(row)

    async def _encode_message(self, session_id: str, message: ConversationMessage) -> tuple[str, list[dict], dict, str, dict[str, bytes]]:
        # Text resides once in body; parts refer to character offsets in that body.
        fragments: list[str] = []
        parts: list[dict] = []
        previews: dict[str, bytes] = {}
        offset = 0
        for part in message.parts:
            item = {key: value for key, value in asdict(part).items() if value is not None and key not in {'text', 'data_b64'}}
            item['kind'] = part.kind.value
            part_text = part.text
            if part_text is None and part.kind != PartKind.TEXT:
                part_text = generated_attachment_reference(part)
            if part_text is not None:
                if fragments:
                    fragments.append('\n')
                    offset += 1
                item['text_span'] = [offset, offset + len(part_text)]
                fragments.append(part_text)
                offset += len(part_text)
            if part.data_b64:
                if part.kind not in {PartKind.IMAGE, PartKind.STICKER}:
                    raise ValueError('Only compressed image previews belong in message storage')
                data = base64.b64decode(part.data_b64, validate=True)
                reference = hashlib.sha256(data).hexdigest()
                item['preview_ref'] = reference
                previews[reference] = data
            parts.append(item)
        metadata = dict(message.metadata or {})
        # Signed provider continuations belong to the working presentation,
        # never to canonical evidence or a capacity-evicted sidecar file.
        metadata.pop('provider_native', None)
        metadata.pop('provider_native_artifact', None)
        metadata.pop('presentation_version', None)
        body = ''.join(fragments)
        # Do not use artifact filenames in the source fingerprint: retrying an update
        # can create a new filename for the same bytes.
        fingerprint_parts = [{key: value for key, value in asdict(part).items()
            if key not in {'preview_ref', 'artifact_path', 'workspace_path'}} for part in message.parts]
        fingerprint_metadata = {key: value for key, value in (message.metadata or {}).items()
            if key not in {'provider_native', 'provider_native_artifact', 'source_revision', 'presentation_version'}}
        fingerprint = _json_hash({'role': message.role.value, 'name': message.name,
            'parts': fingerprint_parts, 'metadata': fingerprint_metadata})
        return body, parts, metadata, fingerprint, previews

    @staticmethod
    def _canonical(message: ConversationMessage) -> dict[str, Any]:
        metadata = message.metadata or {}
        source_id = metadata.get('source_message_id', metadata.get('telegram_message_id'))
        source = str(metadata.get('source') or ('telegram' if source_id is not None else 'agent' if message.role != MessageRole.USER else 'unknown'))
        return {
            'source': source,
            'source_chat_id': str(metadata['source_chat_id']) if metadata.get('source_chat_id') is not None else None,
            'source_message_id': str(source_id) if source_id is not None else None,
            'actor_id': str(metadata['actor_id']) if metadata.get('actor_id') else None,
            'actor_kind': str(metadata.get('actor_kind') or 'unknown'),
            'actor_name': metadata.get('actor_name') or message.name,
            'topic_id': str(metadata['topic_id']) if metadata.get('topic_id') is not None else None,
            'reply_to_source_id': str(metadata['reply_to_source_id']) if metadata.get('reply_to_source_id') is not None else None,
            'sent_at': _timestamp(metadata.get('sent_at')),
        }

    async def allocate_tool_call_id(self, name: str) -> str:
        """Allocate a compact application call reference from the DB sequence.

        Reserving a sequence value need not create a message row. Like rolled
        back inserts, it leaves an ordinary gap while preventing collisions
        across processes, restarts and context resets.
        """
        async with self.pool.connection() as conn:
            row = await (await conn.execute(
                "SELECT nextval(pg_get_serial_sequence('messages','id')) AS id")).fetchone()
        return f'{name}:{row["id"]}'

    async def append_message(self, session_id: str, message: ConversationMessage, estimated_tokens: int | None = None,
                             *, expected_scope: Mapping[str, Any] | None = None,
                             generation_only: bool = False, intake: bool = False,
                             context_owner_message_id: int | None = None) -> StoredConversationMessage:
        encoded = await self._encode_message(session_id, message)
        async with self.pool.connection() as conn:
            return await self._append_encoded_message(conn, session_id, message, encoded,
                estimated_tokens=estimated_tokens, expected_scope=expected_scope,
                generation_only=generation_only, intake=intake, context_owner_message_id=context_owner_message_id)

    async def append_messages(self, session_id: str, messages: Sequence[ConversationMessage], *,
                              expected_scope: Mapping[str, Any] | None = None) -> list[StoredConversationMessage]:
        """Persist a completed exchange together, using the normal append path."""
        # Encode before taking the session lock shared with intake and reset.
        prepared = [(message, await self._encode_message(session_id, message)) for message in messages]
        stored = []
        async with self.pool.connection() as conn:
            for message, encoded in prepared:
                stored.append(await self._append_encoded_message(conn, session_id, message, encoded,
                    expected_scope=expected_scope if not stored else None))
                # The first append guards the starting scope. Its session lock
                # lasts until the whole transaction commits, including own edits.
        return stored

    async def _append_encoded_message(self, conn: AsyncConnection, session_id: str, message: ConversationMessage,
                                      encoded: tuple, *, estimated_tokens: int | None = None,
                                      expected_scope: Mapping[str, Any] | None = None,
                                      generation_only: bool = False, intake: bool = False,
                                      context_owner_message_id: int | None = None) -> StoredConversationMessage:
        body, parts, metadata, fingerprint, previews = encoded
        canonical = self._canonical(message)
        if canonical['source_message_id'] is not None and canonical['source_chat_id'] is None:
            canonical['source_chat_id'] = session_id
        estimate = estimated_tokens if estimated_tokens is not None else TokenEstimator.estimate_message(message)
        scope = await self._session(conn, session_id, lock=True)
        self._check_scope(scope, expected_scope, context=not generation_only, revision=not intake)
        existing = None
        if canonical['source_message_id'] is not None:
            existing = await (await conn.execute('''SELECT m.*,r.fingerprint,r.edited_at FROM messages m
                JOIN message_revisions r ON (r.message_id,r.revision)=(m.id,m.source_revision)
                WHERE m.session_id=%s AND m.generation=%s AND m.source=%s AND m.source_chat_id=%s
                AND m.source_message_id=%s FOR UPDATE OF m''',
                (session_id, scope['generation'], canonical['source'], canonical['source_chat_id'], canonical['source_message_id']))).fetchone()
        if existing:
            edited_at = _timestamp(metadata.get('edited_at'))
            if existing['fingerprint'] == fingerprint or (existing['edited_at'] and (edited_at is None or edited_at < existing['edited_at'])):
                return (await self._read_ids(conn, session_id, [existing['id']], include_hidden=True))[0]
            message_id = existing['id']
            revision = existing['source_revision'] + 1
            await conn.execute('''UPDATE messages SET source_revision=%s,actor_id=%s,actor_kind=%s,
                actor_name=%s,topic_id=%s,reply_to_source_id=%s,presentation=%s WHERE id=%s''',
                (revision, canonical['actor_id'], canonical['actor_kind'], canonical['actor_name'], canonical['topic_id'], canonical['reply_to_source_id'],
                 Jsonb({'presentation_version': AGENT_PRESENTATION_VERSION}), message_id))
            await self._invalidate_sources(conn, session_id, scope['generation'], [message_id])
            await conn.execute('UPDATE sessions SET revision=revision+1 WHERE session_id=%s', (session_id,))
        else:
            row = await (await conn.execute('''INSERT INTO messages
                (session_id,generation,context_id,role,source,source_chat_id,source_message_id,actor_id,
                 actor_kind,actor_name,topic_id,reply_to_source_id,sent_at,presentation)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,COALESCE(%s,now()),%s) RETURNING id''',
                (session_id, scope['generation'], scope['context_id'], message.role.value,
                 *(canonical[key] for key in _CANONICAL_COLUMNS), canonical['sent_at'],
                 Jsonb({'presentation_version': AGENT_PRESENTATION_VERSION})))).fetchone()
            message_id, revision = row['id'], 1
        for reference, data in previews.items():
            await conn.execute('''INSERT INTO message_previews (session_id,reference,payload)
                VALUES (%s,%s,%s) ON CONFLICT DO NOTHING''', (session_id, reference, data))
        await conn.execute('''INSERT INTO message_revisions
            (message_id,revision,body,parts,metadata,estimated_tokens,fingerprint,edited_at)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s)''',
            (message_id, revision, body, Jsonb(parts), Jsonb(metadata), max(0, int(estimate)), fingerprint, _timestamp(metadata.get('edited_at'))))
        if message.metadata.get('provider_native'):
            await conn.execute("UPDATE messages SET presentation=COALESCE(presentation,'{}'::jsonb)||%s WHERE id=%s",
                (Jsonb({'provider_native': message.metadata['provider_native']}), message_id))
        if context_owner_message_id is not None:
            if message.role != MessageRole.ASSISTANT:
                raise ValueError('Only delivered assistant speech can share its model replay owner')
            from tgchatbot.storage.assistant_delivery import link_replay_owner
            await link_replay_owner(conn, session_id=session_id, scope=scope,
                message_id=message_id, owner_message_id=context_owner_message_id)
        searchable = []
        # Context controls and memory lookups remain replayable history, but
        # are derived from existing evidence and do not get their own search vote.
        memory_lookup = (message.role == MessageRole.TOOL
            and message.name in ('memory_search', 'memory_read', 'user_profile_fetch')
            and metadata.get('tool_phase') in ('call', 'result'))
        search_parts = [] if memory_lookup or metadata.get('synthetic_role') in {'reply_target', 'profile_refresh'} else parts
        for part in search_parts:
            if part.get('origin') in {'auto_note', 'provenance'}:
                continue
            if part.get('text_span'):
                start, end = part['text_span']
                searchable.append(body[start:end])
            searchable.extend(str(part[key]) for key in ('filename', 'mime_type', 'artifact_path') if part.get(key))
        await conn.execute('''INSERT INTO message_search (message_id,lexemes) VALUES (%s,array_to_tsvector(%s::text[]))
            ON CONFLICT (message_id) DO UPDATE SET lexemes=excluded.lexemes''', (message_id, lexical_terms('\n'.join(searchable))))
        from tgchatbot.storage.profiles import queue_input
        await queue_input(conn, session_id=session_id, generation=scope['generation'], message_id=message_id,
            revision=revision, message=message, body=body, parts=parts, canonical=canonical)
        if message.role in {MessageRole.USER, MessageRole.ASSISTANT} and not metadata.get('synthetic_role'):
            await conn.execute('''INSERT INTO jobs
                (session_id,generation,context_id,scope_revision,kind,policy,source_ids,source_revisions,payload,dedupe_key)
                VALUES (%s,%s,%s,%s,'memory_ingest','memory',%s,%s,%s,%s) ON CONFLICT DO NOTHING''',
                (session_id, scope['generation'], scope['context_id'], scope['revision'], [message_id],
                 Jsonb({str(message_id): revision}), Jsonb({'message_id': message_id}), f'{message_id}:{revision}'))
        if (metadata.get('synthetic_role') == 'profile_refresh' and metadata.get('tool_phase') == 'result'
                and metadata.get('compaction_version') is not None):
            await conn.execute('''UPDATE sessions SET profile_refresh_version=greatest(profile_refresh_version,
                least(compaction_version,%s)) WHERE session_id=%s''', (int(metadata['compaction_version']), session_id))
        await conn.execute('UPDATE sessions SET context_version=context_version+1 WHERE session_id=%s', (session_id,))
        return (await self._read_ids(conn, session_id, [message_id], include_hidden=True, presentation=True))[0]

    def _message(self, row: Mapping[str, Any], *, presentation: bool = False) -> StoredConversationMessage:
        body, part_data, metadata = row['body'], row['parts'], dict(row['metadata'])
        if row.get('context_owner_message_id') is not None and not row.get('context_replay_detached'):
            metadata['context_owner_message_id'] = row['context_owner_message_id']
        if presentation and row.get('presentation'):
            projection = row['presentation']
            if 'presentation_version' in projection:
                metadata['presentation_version'] = projection['presentation_version']
            body, part_data = projection.get('body', body), projection.get('parts', part_data)
            if projection.get('tool_evidence'):
                metadata['tool_evidence'] = True
            if projection.get('provider_native'):
                metadata['provider_native'] = projection['provider_native']
            for field in ('portable_tool_history', 'provider_native_skip_same_provider'):
                if field in projection:
                    metadata[field] = projection[field]
        parts = []
        for original in part_data:
            item = dict(original)
            span = item.pop('text_span', None)
            kind = PartKind(item.pop('kind'))
            if span is not None:
                item['text'] = body[span[0]:span[1]]
            parts.append(MessagePart(kind=kind, **item))
        # Preserve transport details while giving all consumers the same stable
        # header, including normalized timestamps for span packing and retrieval.
        metadata.update({key: row[key] for key in _CANONICAL_COLUMNS})
        metadata['sent_at'] = row['sent_at'].isoformat()
        metadata['source_revision'] = row['source_revision']
        message = ConversationMessage(role=MessageRole(row['role']), name=row['actor_name'], parts=parts, metadata=metadata)
        estimate = row['estimated_tokens']
        if presentation and row.get('presentation'):
            estimate = row['presentation'].get('estimated_tokens', estimate)
        return StoredConversationMessage(db_id=row['id'], message=message,
            estimated_tokens=estimate, created_at=int(row['sent_at'].timestamp()),
            context_version=int(row.get('db_context_version', 0)))

    @staticmethod
    def _select_message(*, include_content: bool = True) -> str:
        columns = ('m.*,s.context_version AS db_context_version,r.body,r.parts,r.metadata,r.estimated_tokens,r.edited_at,'
                   'replay.owner_message_id AS context_owner_message_id,replay.detached AS context_replay_detached'
                   if include_content else 'm.id,m.source_revision,m.generation')
        return f'''SELECT {columns}
            FROM messages m JOIN sessions s ON s.session_id=m.session_id AND s.generation=m.generation
            JOIN message_revisions r ON (r.message_id,r.revision)=(m.id,m.source_revision)
            LEFT JOIN message_replay_owners replay ON replay.message_id=m.id'''

    async def _read_ids(self, conn: AsyncConnection, session_id: str, ids: list[int], *,
                        current_context: bool = False, include_hidden: bool = False,
                        presentation: bool = False) -> list[StoredConversationMessage]:
        query = self._select_message() + ' WHERE m.session_id=%s AND m.id=ANY(%s)'
        if current_context:
            query += ' AND m.context_id=s.context_id'
        if not include_hidden:
            query += ' AND NOT m.hidden AND NOT m.deleted'
        rows = await (await conn.execute(query + ' ORDER BY m.id', (session_id, ids))).fetchall()
        return [self._message(row, presentation=presentation) for row in rows]

    async def read_messages(self, session_id: str, message_ids: Sequence[int], *, current_context: bool = False,
                            limit: int | None = None) -> list[StoredConversationMessage]:
        ids = _ids(message_ids)[:_limit(limit, self.config.read_page_size)]
        async with self.pool.connection() as conn:
            return await self._read_ids(conn, session_id, ids, current_context=current_context)

    async def read_message_by_source(self, session_id: str, *, source: str,
                                      source_chat_id: str, source_message_id: str,
                                      expected_scope: Mapping[str, Any] | None = None,
                                      generation_only: bool = False) -> StoredConversationMessage | None:
        """Read an active original by transport identity, with the caller's scope."""
        async with self.pool.connection() as conn:
            scope = await self.read_session_scope(conn, session_id)
            if scope is None:
                return None
            self._check_scope(scope, expected_scope, context=not generation_only)
            row = await (await conn.execute(self._select_message() + '''
                WHERE m.session_id=%s AND m.source=%s AND m.source_chat_id=%s
                    AND m.source_message_id=%s AND NOT m.hidden AND NOT m.deleted''',
                (session_id, source, str(source_chat_id), str(source_message_id)))).fetchone()
            return self._message(row) if row else None

    async def _recent(self, session_id: str, limit: int, *, before_message_id: int | None = None,
                       uncompacted: bool = False, current_context: bool = True) -> list[StoredConversationMessage]:
        query = self._select_message() + ''' WHERE m.session_id=%s AND NOT m.hidden AND NOT m.deleted'''
        if current_context:
            query += ' AND m.context_id=s.context_id'
        parameters: list[Any] = [session_id]
        if before_message_id is not None:
            query += ' AND m.id < %s'
            parameters.append(before_message_id)
        if uncompacted:
            query += ' AND m.compacted_by_block_id IS NULL AND (replay.message_id IS NULL OR replay.detached)'
        query += ' ORDER BY m.id DESC LIMIT %s'
        parameters.append(_limit(limit, self.config.read_page_size))
        async with self.pool.connection() as conn:
            rows = await (await conn.execute(query, parameters)).fetchall()
        return [self._message(row, presentation=True) for row in rows]

    async def list_recent_visible_messages(self, session_id: str, limit: int | None = None, *,
                                           before_message_id: int | None = None,
                                           current_context: bool = True) -> list[StoredConversationMessage]:
        return await self._recent(session_id, _limit(limit, self.config.recent_page_size),
            before_message_id=before_message_id, current_context=current_context)

    async def list_recent_participant_messages(self, session_id: str, *,
            expected_scope: Mapping[str, Any] | None = None) -> list[ConversationMessage]:
        """Recent user-original identity metadata, including compacted sources.

        The existing recent-page setting bounds the original-message window.
        Tool/assistant traffic and synthetic controls do not consume that window;
        the participant collector owns identity and direct-reply eligibility.
        """
        async with self.pool.connection() as conn:
            scope = await self.read_session_scope(conn, session_id)
            if scope is None:
                return []
            self._check_scope(scope, expected_scope)
            rows = await (await conn.execute('''SELECT jsonb_build_object(
                    'actor_id',m.actor_id,'actor_kind',m.actor_kind,
                    'source_chat_id',m.source_chat_id,'reply_to_source_id',m.reply_to_source_id,
                    'reply_to_source_chat_id',r.metadata->'reply_to_source_chat_id',
                    'reply_to_actor',r.metadata->'reply_to_actor') AS metadata
                FROM messages m JOIN message_revisions r
                    ON (r.message_id,r.revision)=(m.id,m.source_revision)
                WHERE m.session_id=%s AND m.generation=%s AND m.context_id=%s
                    AND NOT m.hidden AND NOT m.deleted AND m.role='user'
                    AND COALESCE(r.metadata->>'synthetic_role','')=''
                ORDER BY m.id DESC LIMIT %s''',
                (session_id, scope['generation'], scope['context_id'],
                 _limit(None, self.config.recent_page_size)))).fetchall()
        return [ConversationMessage(MessageRole.USER, parts=[], metadata=row['metadata']) for row in rows]

    async def list_canonical_messages(self, session_id: str, *, after_message_id: int = 0,
                                      limit: int | None = None, expected_scope: Mapping[str, Any] | None = None) -> list[StoredConversationMessage]:
        async with self.pool.connection() as conn:
            if not self._read_only:
                await self._session(conn, session_id)
            scope = await self.read_session_scope(conn, session_id)
            if scope is None:
                return []
            self._check_scope(scope, expected_scope, context=False)
            rows = await (await conn.execute(self._select_message() + '''
                WHERE m.session_id=%s AND m.generation=%s AND m.id>%s AND NOT m.hidden AND NOT m.deleted
                ORDER BY m.id LIMIT %s''',
                (session_id, scope['generation'], int(after_message_id), _limit(limit, self.config.read_page_size)))).fetchall()
            return [self._message(row) for row in rows]

    async def list_messages(self, session_id: str, limit: int | None = None) -> list[ConversationMessage]:
        return [row.message for row in reversed(await self._recent(session_id, _limit(limit, self.config.read_page_size)))]

    async def list_uncompacted_messages(self, session_id: str, limit: int | None = None, *,
                                        before_message_id: int | None = None) -> list[StoredConversationMessage]:
        return list(reversed(await self._recent(session_id, _limit(limit, self.config.history_page_size),
            uncompacted=True, before_message_id=before_message_id)))

    async def get_context_version(self, session_id: str) -> int:
        async with self.pool.connection() as conn:
            row = await (await conn.execute('SELECT context_version FROM sessions WHERE session_id=%s', (session_id,))).fetchone()
        return int(row['context_version']) if row else 0

    async def load_live_context(self, session_id: str) -> tuple[list[MemoryBlock], list[StoredConversationMessage]]:
        blocks, messages, _ = await self.load_live_context_versioned(session_id)
        return blocks, messages

    async def load_live_context_versioned(self, session_id: str) -> tuple[list[MemoryBlock], list[StoredConversationMessage], int]:
        """Load the complete compaction input; page sizes bound fetches, not history.

        Both projections share a snapshot so a concurrent compaction, edit, or
        reset cannot put originals and their replacement summaries out of sync.
        Ordinary list APIs remain limited reads for callers that need a page.
        """
        async with self.pool.connection() as conn:
            await conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            return await self.read_live_context(conn, session_id)

    async def read_live_context(self, conn, session_id):
        """Shared live projection inside a caller-owned read snapshot."""
        messages, blocks = [], []
        row = await (await conn.execute('SELECT context_version FROM sessions WHERE session_id=%s', (session_id,))).fetchone()
        version = int(row['context_version']) if row else 0
        async with conn.cursor(name='live_context_messages') as cursor:
            await cursor.execute(self._select_message() + '''
                WHERE m.session_id=%s AND m.context_id=s.context_id
                AND NOT m.hidden AND NOT m.deleted AND m.compacted_by_block_id IS NULL
                AND (replay.message_id IS NULL OR replay.detached)
                ORDER BY m.id''', (session_id,))
            while rows := await cursor.fetchmany(self.config.history_page_size):
                messages.extend(self._message(row, presentation=True) for row in rows)
        async with conn.cursor(name='live_context_blocks') as cursor:
            await cursor.execute(f'''SELECT {_BLOCK_SUMMARY_COLUMNS} FROM memory_blocks b JOIN sessions s
                ON s.session_id=b.session_id AND s.generation=b.generation AND s.context_id=b.context_id
                WHERE b.session_id=%s AND b.valid AND COALESCE(b.details->>'lifecycle','sealed')='sealed'
                ORDER BY b.sequence_no,b.id''', (session_id,))
            while rows := await cursor.fetchmany(self.config.memory_block_page_size):
                blocks.extend(self._block(row) for row in rows)
        return blocks, messages, version

    async def load_token_calibration(self, key: tuple[str, str, str]) -> float:
        async with self.pool.connection() as conn:
            row = await (await conn.execute('''SELECT multiplier FROM provider_token_calibration
                WHERE (provider,model,history_mode)=(%s,%s,%s)''', key)).fetchone()
        return float(row['multiplier']) if row else 1.0

    async def update_token_calibration(self, key: tuple[str, str, str], target: float) -> float:
        # Preserve the learned estimator's original upward-only smoothing.
        # Concurrent chats on the same route update one authoritative row.
        async with self.pool.connection() as conn:
            row = await (await conn.execute('''INSERT INTO provider_token_calibration
                (provider,model,history_mode,multiplier) VALUES (%s,%s,%s,0.75+%s*0.25)
                ON CONFLICT(provider,model,history_mode) DO UPDATE SET
                    multiplier=greatest(provider_token_calibration.multiplier,
                        provider_token_calibration.multiplier*0.75+%s*0.25)
                RETURNING multiplier''', (*key, target, target))).fetchone()
        return float(row['multiplier'])

    async def load_compaction_window(self, session_id: str, *, through_message_id: int | None = None) -> CompactionWorkingSet:
        """Oldest preparation input; page targets never split a generated batch."""
        async with self.pool.connection() as conn:
            await conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            session = await (await conn.execute('SELECT * FROM sessions WHERE session_id=%s', (session_id,))).fetchone()
            if session is None:
                raise ValueError('Session must exist before preparing its context')
            if through_message_id is None:
                newest = await (await conn.execute('''SELECT max(id) AS id FROM messages
                    WHERE session_id=%s AND generation=%s AND context_id=%s''',
                    (session_id, session['generation'], session['context_id']))).fetchone()
                through_message_id = int(newest['id'] or 0)
            state = CompactionWorkingSet(session_id=session_id, loaded=True,
                through_message_id=through_message_id, database_version=session['context_version'])
            raw_query = self._select_message() + ''' WHERE m.session_id=%s AND m.context_id=s.context_id
                AND NOT m.hidden AND NOT m.deleted AND m.compacted_by_block_id IS NULL AND m.id<=%s
                AND (replay.message_id IS NULL OR replay.detached)'''
            page = await (await conn.execute(raw_query + ' ORDER BY m.id LIMIT %s',
                (session_id, through_message_id, self.config.history_page_size + 1))).fetchall()
            rows = page[:self.config.history_page_size]
            # Calls of a generated batch are admitted together, but intake may
            # occur before its outcomes. Extend through the whole durable batch.
            checked_batches: set[str] = set()
            while rows:
                batches = {row['metadata'].get('tool_batch_id') for row in rows} - {None} - checked_batches
                if not batches:
                    break
                checked_batches.update(batches)
                end = await (await conn.execute('''SELECT max(m.id) AS id FROM messages m
                    JOIN message_revisions r ON r.message_id=m.id AND r.revision=m.source_revision
                    WHERE m.session_id=%s AND m.generation=%s AND m.context_id=%s
                    AND NOT m.hidden AND NOT m.deleted AND m.compacted_by_block_id IS NULL
                    AND m.id>%s AND m.id<=%s AND m.role='tool'
                    AND r.metadata->>'tool_batch_id'=ANY(%s)''',
                    (session_id, session['generation'], session['context_id'], rows[-1]['id'],
                     through_message_id, list(batches)))).fetchone()
                if end['id'] is not None:
                    rows.extend(await (await conn.execute(raw_query + ' AND m.id>%s AND m.id<=%s ORDER BY m.id',
                        (session_id, through_message_id, rows[-1]['id'], end['id']))).fetchall())
            if rows:
                following = await (await conn.execute(raw_query + ' AND m.id>%s ORDER BY m.id LIMIT 1',
                    (session_id, through_message_id, rows[-1]['id']))).fetchone()
                if following:
                    last_message = self._message(rows[-1], presentation=True).message
                    next_message = self._message(following, presentation=True).message
                    if (matching_tool_result(last_message, next_message) or
                            is_auto_note_message(last_message) and next_message.role == MessageRole.USER
                            and not is_auto_note_message(next_message)):
                        rows.append(following)
            state.raw_messages = [self._message(row, presentation=True) for row in rows]
            last_id = rows[-1]['id'] if rows else 0
            state.more_raw = bool((await (await conn.execute(raw_query.replace(
                self._select_message(), 'SELECT 1 FROM messages m JOIN sessions s ON s.session_id=m.session_id AND s.generation=m.generation '
                'LEFT JOIN message_replay_owners replay ON replay.message_id=m.id')
                + ' AND m.id>%s LIMIT 1', (session_id, through_message_id, last_id))).fetchone()))
            block_boundary = last_id if state.more_raw else through_message_id
            root_query = self._root_block_query()
            blocks = await (await conn.execute(root_query + ''' AND COALESCE((b.details->>'end_message_id')::bigint,0)<=%s
                ORDER BY b.sequence_no,b.id LIMIT %s''',
                (session_id, session_id, block_boundary, self.config.memory_block_page_size + 1))).fetchall()
            state.more_blocks = len(blocks) > self.config.memory_block_page_size
            state.blocks = [self._block(row) for row in blocks[:self.config.memory_block_page_size]]
        state.rebuild_estimate()
        return state

    @staticmethod
    def _root_block_query() -> str:
        return f'''WITH covered AS MATERIALIZED (
            SELECT jsonb_array_elements_text(parent.details->'parent_block_ids')::bigint AS id
            FROM memory_blocks parent JOIN sessions scope
                ON scope.session_id=parent.session_id AND scope.generation=parent.generation
                AND scope.context_id=parent.context_id
            WHERE parent.session_id=%s AND parent.valid AND parent.details->>'kind'='digest')
            SELECT {_BLOCK_SUMMARY_COLUMNS} FROM memory_blocks b JOIN sessions s
                ON s.session_id=b.session_id AND s.generation=b.generation AND s.context_id=b.context_id
            WHERE b.session_id=%s AND b.valid AND COALESCE(b.details->>'lifecycle','sealed')='sealed'
                AND NOT EXISTS (SELECT 1 FROM covered WHERE covered.id=b.id)'''

    async def context_preparation_counts(self, session_id: str, through_message_id: int) -> dict[str, int]:
        async with self.pool.connection() as conn:
            raw = await (await conn.execute('''SELECT count(*) AS total FROM messages m JOIN sessions s
                ON s.session_id=m.session_id AND s.generation=m.generation AND s.context_id=m.context_id
                LEFT JOIN message_replay_owners replay ON replay.message_id=m.id
                WHERE m.session_id=%s AND NOT m.hidden AND NOT m.deleted
                AND (replay.message_id IS NULL OR replay.detached)
                AND m.compacted_by_block_id IS NULL AND m.id<=%s''',
                (session_id, through_message_id))).fetchone()
            blocks = await (await conn.execute('SELECT count(*) AS total FROM (' + self._root_block_query()
                + " AND COALESCE((b.details->>'end_message_id')::bigint,0)<=%s) roots",
                (session_id, session_id, through_message_id))).fetchone()
        return {'remaining_raw_messages': raw['total'], 'remaining_root_blocks': blocks['total']}

    async def get_compaction_version(self, session_id: str) -> int:
        async with self.pool.connection() as conn:
            row = await (await conn.execute('SELECT compaction_version FROM sessions WHERE session_id=%s',
                (session_id,))).fetchone()
        return int(row['compaction_version']) if row else 0

    async def compaction_needs_profile_refresh(self, session_id: str) -> bool:
        async with self.pool.connection() as conn:
            row = await (await conn.execute('''SELECT compaction_version>profile_refresh_version AS needed
                FROM sessions WHERE session_id=%s''', (session_id,))).fetchone()
        return bool(row and row['needed'])

    async def list_preview_refs(self, session_id: str) -> set[str]:
        """List references owned by current working image presentations.

        Summary coverage can be invalidated or undone, so covered originals in
        the current context retain their unretired presentations. Canonical audit
        revisions independently retain compressed bytes after prompt retirement.
        """
        async with self.pool.connection() as conn:
            rows = await (await conn.execute('''SELECT DISTINCT part->>'preview_ref' AS reference
                FROM messages m JOIN sessions s ON s.session_id=m.session_id
                    AND s.generation=m.generation AND s.context_id=m.context_id
                JOIN message_revisions r ON r.message_id=m.id AND r.revision=m.source_revision
                CROSS JOIN LATERAL jsonb_array_elements(COALESCE(m.presentation->'parts', r.parts)) part
                WHERE m.session_id=%s AND NOT m.hidden AND NOT m.deleted
                    AND part->>'preview_ref' IS NOT NULL''', (session_id,))).fetchall()
        return {row['reference'] for row in rows}

    async def load_preview_data(self, session_id: str, references: Sequence[str]) -> dict[str, bytes]:
        if not references:
            return {}
        async with self.pool.connection() as conn:
            rows = await (await conn.execute('''SELECT reference,payload FROM message_previews
                WHERE session_id=%s AND reference=ANY(%s)''', (session_id, list(references)))).fetchall()
        return {row['reference']: bytes(row['payload']) for row in rows}

    async def describe_message_images(self, session_id: str, message_ids: Sequence[int], *,
                                      expected_scope: Mapping[str, Any] | None = None) -> dict[int, list[dict]]:
        from tgchatbot.storage.message_images import describe_message_images
        return await describe_message_images(self, session_id, _ids(message_ids), expected_scope=expected_scope)

    async def resolve_message_images(self, session_id: str, message_ids: Sequence[int], image_ids: Sequence[str], *,
                                    expected_scope: Mapping[str, Any] | None = None,
                                    timezone: str | None = 'UTC') -> dict[str, Any]:
        from tgchatbot.storage.message_images import resolve_message_images
        return await resolve_message_images(self, session_id, _ids(message_ids), image_ids,
            expected_scope=expected_scope, timezone=timezone)

    async def list_unfinished_tool_calls(self, session_id: str) -> list[StoredConversationMessage]:
        """Recover declared calls without rereading the whole conversation."""
        async with self.pool.connection() as conn:
            rows = await (await conn.execute(self._select_message() + '''
                WHERE m.session_id=%s AND m.context_id=s.context_id AND m.role='tool'
                    AND NOT m.hidden AND NOT m.deleted AND m.compacted_by_block_id IS NULL
                    AND r.metadata->>'tool_phase'='call'
                    AND NOT EXISTS (
                        SELECT 1 FROM messages done JOIN message_revisions outcome
                            ON outcome.message_id=done.id AND outcome.revision=done.source_revision
                        WHERE done.session_id=m.session_id AND done.generation=m.generation
                            AND done.context_id=m.context_id AND done.role='tool' AND done.id>m.id
                            AND NOT done.hidden AND NOT done.deleted
                            AND outcome.metadata->>'tool_phase'='result'
                            AND (outcome.metadata->>'tool_call_message_id'=m.id::text
                                OR (NOT outcome.metadata ? 'tool_call_message_id'
                                    AND done.actor_name IS NOT DISTINCT FROM m.actor_name
                                    AND outcome.metadata#>>'{tool_payload,call_id}'=r.metadata#>>'{tool_payload,call_id}'
                                    AND outcome.metadata->>'tool_batch_id' IS NOT DISTINCT FROM r.metadata->>'tool_batch_id')))
                ORDER BY m.id''', (session_id,))).fetchall()
        return [self._message(row, presentation=True) for row in rows]

    async def retire_context_images(self, session_id: str, *, target_images: int,
                                    protected_ids: Sequence[int] = (),
                                    through_message_id: int | None = None,
                                    expected_scope: Mapping[str, Any] | None = None) -> ImageRetirement:
        """Retire the oldest admitted images from authoritative presentations."""
        async with self.pool.connection() as conn:
            scope = await self._session(conn, session_id, lock=True)
            self._check_scope(scope, expected_scope)
            query = self._select_message() + '''
                WHERE m.session_id=%s AND m.context_id=s.context_id
                    AND NOT m.hidden AND NOT m.deleted AND m.compacted_by_block_id IS NULL'''
            parameters = [session_id]
            if through_message_id is not None:
                query += ' AND m.id<=%s'
                parameters.append(through_message_id)
            rows = await (await conn.execute(query + ' ORDER BY m.id', parameters)).fetchall()
            messages = [self._message(row, presentation=True) for row in rows]
            remaining = max(0, sum(item.image_count for item in messages) - target_images)
            removed = 0
            for item in messages:
                if not remaining:
                    break
                if item.db_id in protected_ids:
                    continue
                parts = []
                changed = False
                for part in item.message.parts:
                    if remaining and part.kind == PartKind.IMAGE and (part.data_b64 or part.preview_ref):
                        parts.append(MessagePart(PartKind.TEXT, text='[Image compacted]',
                            remote_sync=False, origin='image_compacted'))
                        remaining -= 1
                        removed += 1
                        changed = True
                    else:
                        parts.append(part)
                if changed:
                    message = replace(item.message, parts=parts)
                    body, encoded_parts, _, _, _ = await self._encode_message(session_id, message)
                    await conn.execute('UPDATE messages SET presentation=COALESCE(presentation,\'{}\'::jsonb)||%s WHERE id=%s',
                        (Jsonb({'body': body, 'parts': encoded_parts,
                            'estimated_tokens': TokenEstimator.estimate_message(message)}), item.db_id))
            if removed:
                committed = await (await conn.execute('''UPDATE sessions SET context_version=context_version+1,
                    compaction_version=compaction_version+1 WHERE session_id=%s RETURNING compaction_version''',
                    (session_id,))).fetchone()
                return ImageRetirement(removed, committed['compaction_version'])
            return ImageRetirement(0)

    async def mark_tool_evidence(self, session_id: str, message_id: int, *,
                                 expected_scope: Mapping[str, Any] | None = None) -> StoredConversationMessage:
        """Annotate the authoritative call without rewriting its source content."""
        async with self.pool.connection() as conn:
            scope = await self._session(conn, session_id, lock=True)
            self._check_scope(scope, expected_scope)
            row = await (await conn.execute('''UPDATE messages
                SET presentation=COALESCE(presentation,'{}'::jsonb)||'{"tool_evidence":true}'::jsonb
                WHERE id=%s AND session_id=%s AND generation=%s AND context_id=%s
                    AND NOT hidden AND NOT deleted RETURNING id''',
                (message_id, session_id, scope['generation'], scope['context_id']))).fetchone()
            if not row:
                raise StaleScopeError('tool call is no longer in the active context')
            await conn.execute('UPDATE sessions SET context_version=context_version+1 WHERE session_id=%s', (session_id,))
            return (await self._read_ids(conn, session_id, [row['id']], presentation=True))[0]

    async def _invalidate_sources(self, conn: AsyncConnection, session_id: str, generation: int, ids: list[int]) -> None:
        from tgchatbot.storage.assistant_delivery import detach_replay_owners
        ids = await detach_replay_owners(conn, session_id=session_id, generation=generation, source_ids=ids)
        # A multi-person excerpt or unfinished tail may contain unaffected
        # originals. Rebuild their projections rather than losing that coverage.
        survivors: set[int] = set()
        profile_survivors: set[int] = set()
        corrected_parents: set[int] = set()
        for table in ('excerpts', 'profile_facts', 'excerpt_tails'):
            valid = ' AND valid' if table != 'excerpt_tails' else ''
            columns = 'source_ids,supersedes' if table == 'profile_facts' else 'source_ids'
            affected = await (await conn.execute(sql.SQL('SELECT ' + columns + ' FROM {} WHERE session_id=%s AND generation=%s'
                + valid + ' AND source_ids && %s::bigint[]').format(sql.Identifier(table)), (session_id, generation, ids))).fetchall()
            target = profile_survivors if table == 'profile_facts' else survivors
            target.update(source for row in affected for source in row['source_ids'] if source not in ids)
            corrected_parents.update(row['supersedes'] for row in affected if row.get('supersedes') is not None)
        blocks = await (await conn.execute('''UPDATE memory_blocks SET valid=false WHERE session_id=%s AND generation=%s
            AND valid AND source_ids && %s::bigint[] RETURNING id''', (session_id, generation, ids))).fetchall()
        if blocks:
            await conn.execute('UPDATE messages SET compacted_by_block_id=NULL WHERE session_id=%s AND generation=%s AND compacted_by_block_id=ANY(%s)',
                (session_id, generation, [row['id'] for row in blocks]))
        for table in ('excerpts', 'profile_facts'):
            await conn.execute(sql.SQL('UPDATE {} SET valid=false WHERE session_id=%s AND generation=%s AND valid AND source_ids && %s::bigint[]').format(sql.Identifier(table)),
                (session_id, generation, ids))
        if corrected_parents:
            await self._refresh_profile_intervals(conn, list(corrected_parents))
            parents = await (await conn.execute('SELECT source_ids FROM profile_facts WHERE id=ANY(%s) AND valid',
                (list(corrected_parents),))).fetchall()
            profile_survivors.update(mid for row in parents for mid in row['source_ids'] if mid not in ids)
        retired = await (await conn.execute('''UPDATE profile_facts SET retired_at=NULL,retirement_sources=NULL
            WHERE session_id=%s AND generation=%s AND retired_at IS NOT NULL
            AND retirement_sources && %s::bigint[] RETURNING source_ids''', (session_id, generation, ids))).fetchall()
        profile_survivors.update(mid for row in retired for mid in row['source_ids'] if mid not in ids)
        await conn.execute('''UPDATE profile_current c SET fact_ids=ARRAY(
            SELECT f.id FROM unnest(c.fact_ids) member(id) JOIN profile_facts f ON f.id=member.id
            WHERE f.valid AND f.status IN ('active','superseded') AND (f.valid_to IS NULL OR f.valid_to>now()))
            WHERE c.session_id=%s AND c.generation=%s''', (session_id, generation))
        await conn.execute('DELETE FROM excerpt_tails WHERE session_id=%s AND generation=%s AND source_ids && %s::bigint[]',
            (session_id, generation, ids))
        retired_jobs = await (await conn.execute('''UPDATE jobs SET status='stale',finished_at=now(),lease_token=NULL
            WHERE session_id=%s AND generation=%s AND status IN ('pending','running') AND source_ids && %s::bigint[]
            RETURNING kind,source_ids''', (session_id, generation, ids))).fetchall()
        # Coalesced intake may not have produced an excerpt or tail yet. Its
        # untouched originals need the same reconstruction as existing excerpts.
        survivors.update(mid for job in retired_jobs if job['kind'] == 'memory_ingest'
            for mid in job['source_ids'] if mid not in ids)
        await conn.execute("UPDATE profile_inputs SET spans='[]',pending_bytes=0 WHERE message_id=ANY(%s)", (ids,))
        if profile_survivors:
            from tgchatbot.storage.profiles import reconcile_sources
            await reconcile_sources(conn, session_id, generation, profile_survivors)
        if survivors:
            # The session is already locked by the caller. Revision+1 is the
            # imminent edit/rollback epoch and makes this retry idempotent.
            await conn.execute('''INSERT INTO jobs
                (session_id,generation,context_id,scope_revision,kind,policy,source_ids,source_revisions,payload,dedupe_key)
                SELECT m.session_id,m.generation,s.context_id,s.revision+1,'memory_ingest','memory',ARRAY[m.id],
                jsonb_build_object(m.id::text,m.source_revision),jsonb_build_object('message_id',m.id),
                'rebuild:'||m.id||':'||m.source_revision||':'||(s.revision+1)
                FROM messages m JOIN sessions s ON s.session_id=m.session_id AND s.generation=m.generation
                JOIN message_revisions r ON (r.message_id,r.revision)=(m.id,m.source_revision)
                WHERE m.session_id=%s AND m.generation=%s AND m.id=ANY(%s)
                AND NOT m.hidden AND NOT m.deleted AND m.role IN ('user','assistant')
                AND NOT (r.metadata ? 'synthetic_role') ON CONFLICT DO NOTHING''',
                (session_id, generation, list(survivors)))

    async def _refresh_profile_intervals(self, conn: AsyncConnection, parent_ids: list[int]) -> None:
        await conn.execute('''UPDATE profile_facts f SET
            valid_to=LEAST(f.original_valid_to,(SELECT min(c.valid_from) FROM profile_facts c WHERE c.supersedes=f.id AND c.valid)),
            status=CASE WHEN EXISTS(SELECT 1 FROM profile_facts c WHERE c.supersedes=f.id AND c.valid)
                THEN 'superseded' ELSE CASE WHEN f.status='superseded' THEN 'active' ELSE f.status END END
            WHERE f.id=ANY(%s)''', (parent_ids,))

    async def _hide(self, session_id: str, *, ids: list[int] | None = None, since: int | None = None, deleted: bool = False) -> int:
        if ids == []:
            return 0
        async with self.pool.connection() as conn:
            scope = await self._session(conn, session_id, lock=True)
            clause = 'id=ANY(%s)' if ids is not None else 'id >= %s'
            rows = await (await conn.execute(f'''UPDATE messages SET hidden=true,deleted=deleted OR %s
                WHERE session_id=%s AND generation=%s AND context_id=%s AND NOT hidden AND {clause} RETURNING id''',
                (deleted, session_id, scope['generation'], scope['context_id'], ids if ids is not None else since))).fetchall()
            changed = [row['id'] for row in rows]
            if changed:
                await self._invalidate_sources(conn, session_id, scope['generation'], changed)
                await conn.execute('UPDATE sessions SET revision=revision+1 WHERE session_id=%s', (session_id,))
                await conn.execute('UPDATE sessions SET context_version=context_version+1 WHERE session_id=%s', (session_id,))
            return len(changed)

    async def hide_message_ids(self, session_id: str, message_ids: Sequence[int]) -> int:
        return await self._hide(session_id, ids=_ids(message_ids))

    async def hide_messages_since(self, session_id: str, message_id: int) -> int:
        return await self._hide(session_id, since=int(message_id))

    async def delete_message_ids(self, session_id: str, message_ids: Sequence[int]) -> int:
        return await self._hide(session_id, ids=_ids(message_ids), deleted=True)

    async def _sources(self, conn: AsyncConnection, session_id: str, scope: Mapping[str, Any], ids: list[int], *,
                        current_context: bool = False, expected_revisions: Mapping[str, Any] | None = None,
                        include_content: bool = True) -> list[dict]:
        if not ids:
            raise ValueError('original source messages are required')
        query = self._select_message(include_content=include_content) + ' WHERE m.session_id=%s AND m.id=ANY(%s) AND NOT m.hidden AND NOT m.deleted'
        if current_context:
            query += ' AND m.context_id=s.context_id'
        rows = await (await conn.execute(query + ' ORDER BY m.id', (session_id, ids))).fetchall()
        if len(rows) != len(ids) or any(row['generation'] != scope['generation'] for row in rows):
            raise StaleScopeError('a source is missing, hidden, or outside the active scope')
        if expected_revisions is not None:
            if any(int(expected_revisions.get(str(row['id']), expected_revisions.get(row['id'], -1))) != row['source_revision'] for row in rows):
                raise StaleScopeError('an original source was revised')
        return rows

    @staticmethod
    def _source_revisions(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return {str(row['id']): row['source_revision'] for row in rows}

    @staticmethod
    def _search_result(row: Mapping[str, Any]) -> dict[str, Any]:
        return {'id': row['id'], 'text': row['body'], 'source_ids': [row['id']], 'parts': row.get('parts', []),
            'fragments': [{'offset': 0, 'text': row['body']}], 'total_characters': len(row['body']),
            'kind': 'message', 'role': row['role'], 'generation': row['generation'],
            'actor_id': row['actor_id'], 'actor_name': row['actor_name'], 'actor_kind': row['actor_kind'],
            'source': row['source'], 'source_chat_id': row['source_chat_id'], 'source_message_id': row['source_message_id'],
            'topic_id': row['topic_id'], 'sent_at': row['sent_at'].isoformat(),
            'source_revision': row['source_revision'], 'metadata': row['metadata']}

    async def search_messages(self, session_id: str, query: str, *, actor_id: str | None = None,
                              before: Any = None, after: Any = None, topic_id: str | None = None,
                              limit: int | None = None) -> list[dict[str, Any]]:
        async with self.pool.connection() as conn:
            scope = await self.read_session_scope(conn, session_id)
            if scope is None:
                return []
            return await self._search_messages(conn, session_id, query, actor_id=actor_id,
                before=before, after=after, topic_id=topic_id, limit=limit)

    async def _search_messages(self, conn: AsyncConnection, session_id: str, query: str, *,
                               actor_id: str | None = None, before: Any = None, after: Any = None,
                               topic_id: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        limit = _limit(limit, self.config.search_results)
        terms = lexical_terms(query)
        if not terms or limit <= 0:
            return []
        # OR candidates preserve recall for mixed Chinese/English questions. Ranking
        # favors passages matching more terms; identity/time remain SQL constraints.
        tsquery = ' | '.join("'" + term.replace("'", "''") + "'" for term in terms)
        conjunction = ' & '.join("'" + term.replace("'", "''") + "'" for term in terms)
        # array_to_tsvector deliberately stores neither positions nor frequencies:
        # every row matching every term has the same maximal OR rank. Taking the
        # newest such rows first is exact and avoids ranking a huge common-word
        # posting list when it already fills the requested result set.
        all_statement = '''SELECT m.id FROM messages m
            JOIN sessions s ON s.session_id=m.session_id AND s.generation=m.generation
            JOIN message_search x ON x.message_id=m.id WHERE m.session_id=%s
            AND NOT m.hidden AND NOT m.deleted AND x.lexemes @@ %s::tsquery'''
        all_parameters: list[Any] = [session_id, conjunction]
        for column, value, operator in (('actor_id', actor_id, '='), ('topic_id', topic_id, '='),
                                        ('sent_at', _timestamp(before), '<'), ('sent_at', _timestamp(after), '>=')):
            if value is not None:
                all_statement += f' AND m.{column} {operator} %s'
                all_parameters.append(value)
        all_parameters.append(limit)
        complete = await (await conn.execute(all_statement + ' ORDER BY m.id DESC LIMIT %s', all_parameters)).fetchall()
        if len(complete) < limit and len(terms) > 1:
            # Unknown words in natural questions should not force ranking every
            # common-word match. Terms absent from the entire lexical index
            # cannot affect ordering: they only scale every OR rank equally.
            # EXISTS otherwise favors a sequential scan on the assumption a
            # match appears early, which is disastrous for an absent word.
            probes = ','.join('EXISTS(SELECT 1 FROM message_search WHERE lexemes @@ %s::tsquery)' for _ in terms)
            await conn.execute('SET LOCAL enable_seqscan = off')
            present = (await (await conn.execute('SELECT ARRAY[' + probes + '] AS present',
                ["'" + term.replace("'", "''") + "'" for term in terms])).fetchone())['present']
            await conn.execute('SET LOCAL enable_seqscan = on')
            available = [term for term, exists in zip(terms, present, strict=True) if exists]
            if not available:
                return []
            if len(available) != len(terms):
                tsquery = ' | '.join("'" + term.replace("'", "''") + "'" for term in available)
                all_parameters[1] = ' & '.join("'" + term.replace("'", "''") + "'" for term in available)
                complete = await (await conn.execute(all_statement + ' ORDER BY m.id DESC LIMIT %s', all_parameters)).fetchall()
        if len(complete) >= limit:
            rows = await (await conn.execute(self._select_message() + ' WHERE m.id=ANY(%s) AND NOT m.hidden AND NOT m.deleted ORDER BY m.id DESC',
                ([row['id'] for row in complete],))).fetchall()
            return [self._search_result(row) for row in rows]
        statement = '''WITH ranked AS MATERIALIZED (
            SELECT m.id,ts_rank(x.lexemes,%s::tsquery) AS rank FROM messages m
            JOIN sessions s ON s.session_id=m.session_id AND s.generation=m.generation
            JOIN message_search x ON x.message_id=m.id
            WHERE m.session_id=%s AND NOT m.hidden AND NOT m.deleted AND x.lexemes @@ %s::tsquery'''
        parameters: list[Any] = [tsquery, session_id, tsquery]
        for column, value, operator in (('actor_id', actor_id, '='), ('topic_id', topic_id, '='),
                                        ('sent_at', _timestamp(before), '<'), ('sent_at', _timestamp(after), '>=')):
            if value is not None:
                statement += f' AND m.{column} {operator} %s'
                parameters.append(value)
        statement += ' ORDER BY rank DESC,m.id DESC LIMIT %s) '
        parameters.append(limit)
        statement += self._select_message() + ' JOIN ranked ON ranked.id=m.id ORDER BY ranked.rank DESC,m.id DESC'
        rows = await (await conn.execute(statement, parameters)).fetchall()
        return [self._search_result(row) for row in rows]

    async def list_message_revisions(self, session_id: str, message_id: int, *, limit: int | None = None) -> list[dict[str, Any]]:
        """Explicit audit read; unlike ordinary search this may include old generations."""
        async with self.pool.connection() as conn:
            rows = await (await conn.execute('''SELECT r.* FROM message_revisions r JOIN messages m ON m.id=r.message_id
                WHERE m.session_id=%s AND m.id=%s ORDER BY r.revision DESC LIMIT %s''',
                (session_id, int(message_id), _limit(limit, self.config.revision_page_size)))).fetchall()
            return rows

    @staticmethod
    def _block(row: Mapping[str, Any]) -> MemoryBlock:
        details = dict(row['details'])
        allowed = {field.name for field in fields(MemoryBlock)} - {'block_id', 'sequence_no', 'summary_text', 'estimated_tokens', 'compaction_version'}
        details = {key: value for key, value in details.items() if key in allowed}
        for key in ('parent_block_ids', 'actor_labels', 'topic_labels', 'actor_identities'):
            if key in details and details[key] is not None:
                details[key] = tuple(details[key])
        if 'source_message_count' not in details:
            details['source_message_count'] = row['source_count'] if 'source_count' in row else len(row['source_ids'])
        return MemoryBlock(block_id=row['id'], sequence_no=row['sequence_no'], summary_text=row['summary_text'],
            estimated_tokens=row['estimated_tokens'], **details)

    async def list_memory_blocks(self, session_id: str, limit: int | None = None) -> list[MemoryBlock]:
        async with self.pool.connection() as conn:
            rows = await (await conn.execute(f'''SELECT {_BLOCK_SUMMARY_COLUMNS} FROM memory_blocks b JOIN sessions s
                ON s.session_id=b.session_id AND s.generation=b.generation AND s.context_id=b.context_id
                WHERE b.session_id=%s AND b.valid AND COALESCE(b.details->>'lifecycle','sealed')='sealed'
                ORDER BY b.sequence_no DESC LIMIT %s''', (session_id, _limit(limit, self.config.memory_block_page_size)))).fetchall()
            return [self._block(row) for row in reversed(rows)]

    async def memory_block_actor_identities(self, session_id: str, block_ids: Sequence[int]) -> list[dict]:
        """Resolve selected legacy parents once, without loading their original bodies.

        Newly compacted blocks carry these observations forward. Match the
        revisions the parent saw, rather than borrowing a later identity edit.
        """
        if not block_ids:
            return []
        async with self.pool.connection() as conn:
            rows = await (await conn.execute('''WITH source_refs AS MATERIALIZED (
                SELECT b.id AS block_id,b.generation,unnest(b.source_ids) AS message_id
                FROM memory_blocks b JOIN sessions s ON s.session_id=b.session_id
                    AND s.generation=b.generation AND s.context_id=b.context_id
                WHERE b.session_id=%s AND b.id=ANY(%s) AND b.valid),
                latest AS MATERIALIZED (
                SELECT DISTINCT ON (m.actor_id) source.block_id,
                    m.id,m.source_revision,m.actor_id,m.actor_kind,m.actor_name,m.sent_at
                FROM source_refs source JOIN LATERAL (
                    SELECT id,source_revision,actor_id,actor_kind,actor_name,sent_at,
                        session_id,generation,hidden,deleted
                    FROM messages WHERE id=source.message_id
                    -- Keep one primary-key lookup per source even before a
                    -- large fresh import has planner statistics. No rows are skipped.
                    OFFSET 0) m ON true
                WHERE m.session_id=%s AND m.generation=source.generation
                    AND NOT m.hidden AND NOT m.deleted AND m.actor_id IS NOT NULL AND m.actor_id<>'unknown'
                ORDER BY m.actor_id,m.sent_at DESC,m.id DESC)
                SELECT latest.*,r.metadata->>'actor_username' AS actor_username
                FROM latest JOIN memory_blocks b ON b.id=latest.block_id
                JOIN message_revisions r ON r.message_id=latest.id AND r.revision=latest.source_revision
                    AND r.revision=(b.source_revisions->>latest.id::text)::integer''',
                (session_id, list(block_ids), session_id))).fetchall()
        return [actor_observation(row, row['id']) for row in rows]

    async def create_memory_block(self, session_id: str, *, summary_text: str, estimated_tokens: int,
                                   source_message_ids: Sequence[int], level: int = 1, kind: str = 'episode',
                                   lifecycle: str = 'sealed', source_kind: str = 'raw',
                                   source_message_count: int | None = None, start_message_id: int | None = None,
                                   end_message_id: int | None = None, parent_block_ids: Sequence[int] | None = None,
                                   topic_labels: Sequence[str] | None = None, actor_labels: Sequence[str] | None = None,
                                   actor_identities: Sequence[dict] | None = None,
                                   time_start: str | None = None, time_end: str | None = None,
                                   retained_raw_excerpt_count: int = 0, validator_status: str | None = None,
                                   validator_score: float | None = None, structured_data: dict | None = None,
                                   expected_scope: Mapping[str, Any] | None = None,
                                   expected_source_revisions: Mapping[str, Any] | None = None,
                                   _replace_ids: Sequence[int] | None = None) -> MemoryBlock:
        ids, parent_ids = _ids(source_message_ids), _ids(parent_block_ids or [])
        replace_ids = _ids(_replace_ids or [])
        expected_revisions = (None if expected_source_revisions is None else
            {str(message_id): int(revision) for message_id, revision in expected_source_revisions.items()})
        async with self.pool.connection() as conn:
            scope = await self._session(conn, session_id, lock=True)
            self._check_scope(scope, expected_scope)
            parents = []
            all_parent_ids = _ids(parent_ids + replace_ids)
            if all_parent_ids:
                parents = await (await conn.execute('''SELECT * FROM memory_blocks WHERE session_id=%s AND generation=%s
                    AND context_id=%s AND id=ANY(%s) AND valid''',
                    (session_id, scope['generation'], scope['context_id'], all_parent_ids))).fetchall()
                if len(parents) != len(all_parent_ids):
                    raise StaleScopeError('a parent memory block is no longer valid')
                if expected_revisions is not None:
                    for parent in parents:
                        for message_id, revision in parent['source_revisions'].items():
                            # Parent summaries own the revisions they observed;
                            # current originals cannot substitute for that evidence.
                            observed = expected_revisions.setdefault(str(message_id), int(revision))
                            if observed != int(revision):
                                raise StaleScopeError('parent and raw evidence use conflicting source revisions')
                ids = _ids(ids + [source for parent in parents for source in parent['source_ids']])
            # Parent ancestry can outgrow the bounded summary text. Publishing
            # needs source identities/revisions, not every original body again.
            sources = await self._sources(conn, session_id, scope, ids, current_context=True,
                expected_revisions=expected_revisions, include_content=False)
            sequence = min((parent['sequence_no'] for parent in parents if parent['id'] in replace_ids), default=None)
            if sequence is None:
                sequence = (await (await conn.execute('''SELECT COALESCE(max(sequence_no),0)+1 AS sequence FROM memory_blocks
                    WHERE session_id=%s AND generation=%s AND context_id=%s''',
                    (session_id, scope['generation'], scope['context_id']))).fetchone())['sequence']
            details = dict(source_message_count=source_message_count if source_message_count is not None else len(ids),
                start_message_id=start_message_id if start_message_id is not None else min(ids),
                end_message_id=end_message_id if end_message_id is not None else max(ids), level=level, kind=kind,
                lifecycle=lifecycle, source_kind=source_kind, parent_block_ids=parent_ids or replace_ids,
                topic_labels=list(topic_labels or []), actor_labels=list(actor_labels or []), time_start=time_start,
                time_end=time_end, retained_raw_excerpt_count=retained_raw_excerpt_count,
                validator_status=validator_status, validator_score=validator_score, structured_data=structured_data or {},
                presentation_version=AGENT_PRESENTATION_VERSION)
            if actor_identities is not None:
                details['actor_identities'] = list(actor_identities)
            row = await (await conn.execute('''INSERT INTO memory_blocks
                (session_id,generation,context_id,sequence_no,summary_text,estimated_tokens,source_ids,source_revisions,details)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING *''',
                (session_id, scope['generation'], scope['context_id'], sequence, summary_text, estimated_tokens,
                 ids, Jsonb(self._source_revisions(sources)), Jsonb(details)))).fetchone()
            await conn.execute('UPDATE messages SET compacted_by_block_id=%s WHERE id=ANY(%s)', (row['id'], ids))
            if replace_ids:
                await conn.execute('UPDATE memory_blocks SET valid=false,superseded_by=%s WHERE id=ANY(%s)', (row['id'], replace_ids))
            committed = await (await conn.execute('''UPDATE sessions SET context_version=context_version+1,
                compaction_version=compaction_version+1 WHERE session_id=%s RETURNING compaction_version''',
                (session_id,))).fetchone()
            return replace(self._block(row), compaction_version=committed['compaction_version'])

    async def replace_memory_blocks(self, session_id: str, *, block_ids: Sequence[int],
                                     summary_text: str, estimated_tokens: int, source_message_count: int,
                                     start_message_id: int | None, end_message_id: int | None,
                                     source_message_ids: Sequence[int] | None = None, **kwargs: Any) -> MemoryBlock:
        if not block_ids:
            raise ValueError('block_ids cannot be empty')
        kwargs.setdefault('level', 2)
        kwargs.setdefault('kind', 'digest')
        kwargs.setdefault('source_kind', 'blocks')
        return await self.create_memory_block(session_id, summary_text=summary_text, estimated_tokens=estimated_tokens,
            source_message_count=source_message_count, start_message_id=start_message_id, end_message_id=end_message_id,
            source_message_ids=source_message_ids or [], _replace_ids=block_ids, **kwargs)

    async def create_excerpt(self, session_id: str, source_message_ids: Sequence[int], *,
                              embedding: Sequence[float] | None = None, model: str = '',
                              spans: Sequence[Mapping[str, int]] | None = None,
                              expected_scope: Mapping[str, Any] | None = None,
                              expected_source_revisions: Mapping[str, Any] | None = None) -> dict[str, Any]:
        ids = _ids(source_message_ids)
        encoded = _vector(embedding)
        if encoded is not None and not model:
            raise ValueError('embedding model identity is required')
        async with self.pool.connection() as conn:
            scope = await self._session(conn, session_id, lock=True)
            self._check_scope(scope, expected_scope, context=False)
            sources = await self._sources(conn, session_id, scope, ids, expected_revisions=expected_source_revisions)
            normalized_spans = [dict(span) for span in spans or []]
            by_id = {row['id']: row for row in sources}
            for span in normalized_spans:
                if set(span) != {'message_id', 'start', 'end'} or span['message_id'] not in by_id:
                    raise ValueError('excerpt span must reference one of its source messages')
                if not 0 <= span['start'] < span['end'] <= len(by_id[span['message_id']]['body']):
                    raise ValueError('excerpt span is outside its original message')
            if normalized_spans and {span['message_id'] for span in normalized_spans} != set(ids):
                raise ValueError('every excerpt source must have a span when spans are provided')
            revisions = self._source_revisions(sources)
            fingerprint = _json_hash([ids, revisions, normalized_spans])
            row = await (await conn.execute('''INSERT INTO excerpts
                (session_id,generation,source_ids,source_revisions,spans,fingerprint,model,embedding)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s::halfvec(1536))
                ON CONFLICT (session_id,generation,fingerprint,model) DO UPDATE
                SET embedding=COALESCE(excluded.embedding,excerpts.embedding),valid=true
                RETURNING id,source_ids,source_revisions,spans,model,embedding IS NOT NULL AS has_embedding''',
                (session_id, scope['generation'], ids, Jsonb(revisions), Jsonb(normalized_spans), fingerprint, model, encoded))).fetchone()
            return self._render_excerpt(row, sources)

    async def get_excerpt_tail(self, session_id: str, topic_id: str | None = None) -> dict[str, Any] | None:
        async with self.pool.connection() as conn:
            return await (await conn.execute('''SELECT t.* FROM excerpt_tails t JOIN sessions s
                ON s.session_id=t.session_id AND s.generation=t.generation
                WHERE t.session_id=%s AND t.topic_id=%s''', (session_id, topic_id or ''))).fetchone()

    async def save_excerpt_tail(self, session_id: str, topic_id: str | None, source_message_ids: Sequence[int], *,
                                spans: Sequence[Mapping[str, int]],
                                expected_scope: Mapping[str, Any] | None = None,
                                expected_source_revisions: Mapping[str, Any] | None = None) -> dict[str, Any]:
        ids = _ids(source_message_ids)
        normalized = [dict(span) for span in spans]
        async with self.pool.connection() as conn:
            scope = await self._session(conn, session_id, lock=True)
            self._check_scope(scope, expected_scope, context=False)
            sources = await self._sources(conn, session_id, scope, ids, expected_revisions=expected_source_revisions)
            by_id = {row['id']: row for row in sources}
            if {span.get('message_id') for span in normalized} != set(ids):
                raise ValueError('every tail source must have a span')
            for span in normalized:
                if set(span) != {'message_id', 'start', 'end'} or not 0 <= span['start'] < span['end'] <= len(by_id[span['message_id']]['body']):
                    raise ValueError('tail span is outside its original message')
            return await (await conn.execute('''INSERT INTO excerpt_tails
                (session_id,generation,topic_id,source_ids,source_revisions,spans)
                VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT (session_id,generation,topic_id)
                DO UPDATE SET source_ids=excluded.source_ids,source_revisions=excluded.source_revisions,
                spans=excluded.spans,updated_at=now() RETURNING *''',
                (session_id, scope['generation'], topic_id or '', ids, Jsonb(self._source_revisions(sources)), Jsonb(normalized)))).fetchone()

    async def clear_excerpt_tail(self, session_id: str, topic_id: str | None = None, *,
                                 expected_scope: Mapping[str, Any] | None = None) -> None:
        async with self.pool.connection() as conn:
            scope = await self._session(conn, session_id, lock=True)
            self._check_scope(scope, expected_scope, context=False)
            await conn.execute('DELETE FROM excerpt_tails WHERE session_id=%s AND generation=%s AND topic_id=%s',
                (session_id, scope['generation'], topic_id or ''))

    @staticmethod
    def _render_excerpt(candidate: Mapping[str, Any], sources: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        span_map: dict[int, list[dict]] = {}
        for span in candidate['spans']:
            span_map.setdefault(span['message_id'], []).append(span)
        texts, attributed_sources = [], []
        for row in sources:
            body = row['body']
            source = PostgresStore._search_result(row)
            if span_map:
                source['fragments'] = [{'offset': span['start'], 'text': body[span['start']:span['end']]}
                    for span in span_map.get(row['id'], [])]
                body = '\n'.join(fragment['text'] for fragment in source['fragments'])
            texts.append(f"[{row['sent_at'].isoformat()} {row['actor_id'] or 'unknown'} {row['actor_name'] or ''}] {body}")
            attributed_sources.append(source)
        return {**candidate, 'kind': 'excerpt', 'text': '\n'.join(texts),
            'sources': attributed_sources}

    async def get_excerpt(self, session_id: str, excerpt_id: int) -> dict[str, Any] | None:
        async with self.pool.connection() as conn:
            scope = await self.read_session_scope(conn, session_id)
            if scope is None:
                return None
            row = await (await conn.execute('''SELECT id,source_ids,source_revisions,spans,model,
                embedding IS NOT NULL AS has_embedding FROM excerpts
                WHERE session_id=%s AND generation=%s AND valid AND id=%s''',
                (session_id, scope['generation'], int(excerpt_id)))).fetchone()
            if row is None:
                return None
            sources = await self._sources(conn, session_id, scope, row['source_ids'], expected_revisions=row['source_revisions'])
            return self._render_excerpt(row, sources)

    async def _excerpt_predicate(self, conn: AsyncConnection, session_id: str, scope: Mapping[str, Any],
                                  model: str, *, actor_id: str | None = None, before: Any = None,
                                  after: Any = None, topic_id: str | None = None) -> tuple[str, list[Any]]:
        predicate = 'e.session_id=%s AND e.generation=%s AND e.valid AND e.embedding IS NOT NULL AND e.model=%s'
        parameters: list[Any] = [session_id, scope['generation'], model]
        filters, values = [], []
        for column, value, operator in (('actor_id', actor_id, '='), ('topic_id', topic_id, '='),
                                        ('sent_at', _timestamp(before), '<'), ('sent_at', _timestamp(after), '>=')):
            if value is not None:
                filters.append(f'm.{column} {operator} %s')
                values.append(value)
        if not filters:
            return predicate, parameters
        source_where = 'm.session_id=%s AND m.generation=%s AND NOT m.hidden AND NOT m.deleted AND ' + ' AND '.join(filters)
        sources = await (await conn.execute('SELECT m.id FROM messages m WHERE ' + source_where + ' LIMIT 10001',
            [session_id, scope['generation'], *values])).fetchall()
        if len(sources) <= 10000:
            # A single large overlap array makes PostgreSQL assume nearly every
            # row matches. Individual GIN probes retain the selective access
            # path; materializing their bounded IDs then enables primary lookup.
            matches = await (await conn.execute('''SELECT DISTINCT hit.id
                FROM unnest(%s::bigint[]) wanted(id) CROSS JOIN LATERAL (
                    SELECT indexed.id FROM excerpts indexed WHERE indexed.valid
                    AND indexed.source_ids @> ARRAY[wanted.id] AND indexed.session_id=%s
                    AND indexed.generation=%s AND indexed.model=%s AND indexed.embedding IS NOT NULL OFFSET 0
                ) hit LIMIT 10001''', ([row['id'] for row in sources], session_id, scope['generation'], model))).fetchall()
            if len(matches) <= 10000:
                predicate += ' AND e.id=ANY(%s::bigint[])'
                parameters.append([row['id'] for row in matches])
                return predicate, parameters
        predicate += ' AND EXISTS(SELECT 1 FROM messages m WHERE m.id=ANY(e.source_ids) AND ' + source_where + ')'
        parameters.extend([session_id, scope['generation'], *values])
        return predicate, parameters

    async def search_excerpts(self, session_id: str, query: str = '', *, embedding: Sequence[float] | None = None,
                               model: str | None = None, actor_id: str | None = None, before: Any = None,
                               after: Any = None, topic_id: str | None = None, limit: int | None = None) -> list[dict[str, Any]]:
        limit = _limit(limit, self.config.search_results)
        if limit == 0:
            return []
        encoded = _vector(embedding)
        results: list[dict] = []
        async with self.pool.connection() as conn:
            # Keep lexical candidates, source revisions, and vector candidates in
            # the same reset/edit scope while rendering the bounded response.
            scope = await self.read_session_scope(conn, session_id)
            if scope is None:
                return []
            lexical = await self._search_messages(conn, session_id, query, actor_id=actor_id,
                before=before, after=after, topic_id=topic_id, limit=limit) if query else []
            if encoded is not None:
                if not model:
                    raise ValueError('query embedding model identity is required')
                predicate, parameters = await self._excerpt_predicate(conn, session_id, scope, model,
                    actor_id=actor_id, before=before, after=after, topic_id=topic_id)
                # Scan inline full vectors exactly, sorting only IDs/distances.
                # Materialize the bounded winners, never all eligible vectors.
                # The +0 prevents an approximate index from changing recall if
                # an operator adds one experimentally to the database.
                statement = '''WITH winners AS MATERIALIZED (SELECT e.id,
                    e.embedding <=> %s::halfvec(1536) AS distance FROM excerpts e WHERE ''' + predicate
                statement += ''' ORDER BY (e.embedding <=> %s::halfvec(1536))+0,e.id LIMIT %s)
                    SELECT e.id,e.source_ids,e.source_revisions,e.spans,e.model,true AS has_embedding,w.distance
                    FROM winners w JOIN excerpts e ON e.id=w.id ORDER BY w.distance,e.id'''
                parameters = [encoded, *parameters, encoded, limit]
                candidates = await (await conn.execute(statement, parameters)).fetchall()
                for candidate in candidates:
                    sources = await self._sources(conn, session_id, scope,
                        candidate['source_ids'], expected_revisions=candidate['source_revisions'])
                    results.append(self._render_excerpt(candidate, sources))
        # Reciprocal rank fusion avoids comparing lexical scores with cosine scales.
        # 60 is the conventional smoothing constant; the candidate sets remain bounded.
        fused: dict[tuple[Any, ...], dict] = {}
        for ranking in (results, lexical):
            seen: set[tuple[Any, ...]] = set()
            for rank, result in enumerate(ranking, 1):
                spans, sources = result.get('spans'), result.get('sources', [])
                whole_original = not spans
                if spans and len(spans) == len(sources) == 1:
                    # An explicit span covering one complete original is the
                    # same evidence as its lexical/spanless representation.
                    # Partial passages and multi-source context remain distinct.
                    whole_original = spans[0] == {'message_id': sources[0]['id'],
                        'start': 0, 'end': len(sources[0]['text'])}
                key = ('sources', *result['source_ids']) if whole_original else ('excerpt', result['id'])
                if key in seen:
                    continue  # One relevance vote per evidence identity/channel.
                seen.add(key)
                if key not in fused:
                    fused[key] = {**result, 'score': 0.0}
                fused[key]['score'] += 1.0 / (60 + rank)
        return sorted(fused.values(), key=lambda item: item['score'], reverse=True)[:limit]

    async def save_profile_fact(self, session_id: str, *, subject_actor_id: str, asserted_by: str,
                                 claim: str, source_ids: Sequence[int], kind: str = 'explicit', status: str = 'active',
                                 valid_from: Any = None, valid_to: Any = None,
                                 claim_key: str | None = None, supersedes: int | None = None,
                                 expected_scope: Mapping[str, Any] | None = None,
                                 expected_source_revisions: Mapping[str, Any] | None = None) -> dict[str, Any]:
        async with self.pool.connection() as conn:
            scope = await self._session(conn, session_id, lock=True)
            self._check_scope(scope, expected_scope, context=False)
            row = await self._save_profile_fact(conn, scope, session_id, subject_actor_id=subject_actor_id,
                asserted_by=asserted_by, claim=claim, source_ids=source_ids, kind=kind, status=status,
                valid_from=valid_from, valid_to=valid_to, claim_key=claim_key, supersedes=supersedes,
                expected_source_revisions=expected_source_revisions)
            from tgchatbot.storage.profiles import publish_current
            await publish_current(self, conn, scope, session_id, subject_actor_id,
                add_ids=[row['id']], max_bytes=None)
            return row

    async def _save_profile_fact(self, conn, scope, session_id: str, *, subject_actor_id: str, asserted_by: str,
                                 claim: str, source_ids: Sequence[int], kind: str = 'explicit', status: str = 'active',
                                 valid_from: Any = None, valid_to: Any = None,
                                 claim_key: str | None = None, supersedes: int | None = None,
                                 expected_scope: Mapping[str, Any] | None = None,
                                 expected_source_revisions: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if not subject_actor_id or not asserted_by or not claim.strip():
            raise ValueError('a profile claim needs a subject, asserter, and text')
        if kind not in {'explicit', 'inferred'}:
            raise ValueError('profile kind must be explicit or inferred')
        if status not in {'active', 'disputed', 'retracted'}:
            raise ValueError('profile status must be active, disputed, or retracted')
        ids = _ids(source_ids)
        sources = await self._sources(conn, session_id, scope, ids, expected_revisions=expected_source_revisions)
        if not any(row['role'] == 'user' and row['actor_id'] == asserted_by and row['actor_kind'] != 'bot' and any(
                part['kind'] == 'text' and part.get('origin') not in
                {'auto_note', 'provenance', 'attachment_excerpt', 'attachment_reference', 'service_event'}
                and part.get('text_span') and row['body'][part['text_span'][0]:part['text_span'][1]].strip()
                for part in row['parts']) for row in sources):
            raise ValueError('profile evidence must include an original message from the named asserter')
        revisions = self._source_revisions(sources)
        start, end = _timestamp(valid_from), _timestamp(valid_to)
        fingerprint = _json_hash([subject_actor_id, asserted_by, claim.strip(), kind, status,
            start, end, ids, revisions, claim_key, supersedes])
        if supersedes is not None:
            previous = await (await conn.execute('''SELECT * FROM profile_facts WHERE id=%s
                AND session_id=%s AND generation=%s AND subject_actor_id=%s AND valid
                AND status IN ('active','superseded')''',
                (int(supersedes), session_id, scope['generation'], subject_actor_id))).fetchone()
            if previous is None or (claim_key is not None and previous['claim_key'] != claim_key):
                raise StaleScopeError('superseded fact does not belong to the active subject and claim')
            start = start or max(row['sent_at'] for row in sources)
            if previous['valid_from'] is not None and start < previous['valid_from']:
                raise ValueError('a correction cannot precede the fact it supersedes')
        row = await (await conn.execute('''INSERT INTO profile_facts
            (session_id,generation,subject_actor_id,asserted_by,claim,kind,status,valid_from,valid_to,source_ids,source_revisions,claim_key,fingerprint,supersedes,original_valid_to)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (session_id,generation,fingerprint) DO UPDATE SET fingerprint=excluded.fingerprint RETURNING *''',
            (session_id, scope['generation'], subject_actor_id, asserted_by, claim.strip(), kind, status,
             start, end, ids, Jsonb(revisions), claim_key, fingerprint, supersedes, end))).fetchone()
        if supersedes is not None:
            await self._refresh_profile_intervals(conn, [supersedes])
        return row

    async def get_profile(self, session_id: str, actor_id: str, *, at: Any = None, limit: int | None = None) -> list[dict[str, Any]]:
        when = _timestamp(at) or datetime.now(timezone.utc)
        query = self._current_profile_query() if at is None else self._historical_profile_query()
        async with self.pool.connection() as conn:
            return await (await conn.execute('SELECT f.* ' + query + ' ORDER BY f.id DESC LIMIT %s',
                (session_id, actor_id, when, when, _limit(limit, self.config.profile_results)))).fetchall()

    @staticmethod
    def _current_profile_query(*, include_future: bool = False) -> str:
        return '''FROM profile_current c JOIN sessions s
            ON s.session_id=c.session_id AND s.generation=c.generation
            CROSS JOIN LATERAL unnest(c.fact_ids) member(id)
            JOIN profile_facts f ON f.id=member.id AND f.session_id=c.session_id
                AND f.generation=c.generation AND f.subject_actor_id=c.actor_id
            WHERE c.session_id=%s AND c.actor_id=%s AND f.valid
            AND f.status IN ('active','superseded')''' + (
                '' if include_future else ' AND (f.valid_from IS NULL OR f.valid_from<=%s)') + (
                ' AND (f.valid_to IS NULL OR f.valid_to>%s)')

    @staticmethod
    def _historical_profile_query() -> str:
        return '''FROM profile_facts f JOIN sessions s
            ON s.session_id=f.session_id AND s.generation=f.generation
            WHERE f.session_id=%s AND f.subject_actor_id=%s AND f.valid
            AND f.status IN ('active','superseded')
            AND (f.valid_from IS NULL OR f.valid_from<=%s) AND (f.valid_to IS NULL OR f.valid_to>%s)'''

    async def fetch_profile_snapshot(self, session_id: str, actor_ids: Sequence[str], *,
                                     expected_scope: Mapping[str, Any] | None = None,
                                     max_bytes: int | None = None, for_learning: bool = False,
                                     include_pending: bool = False) -> dict[str, Any]:
        """Read selected profiles and their internal evidence from one committed snapshot."""
        async with self.pool.connection() as conn:
            await conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            result = await self.read_profile_snapshot(conn, session_id, actor_ids,
                expected_scope=expected_scope, for_learning=for_learning, include_pending=include_pending)
        if expected_scope is not None:
            await self.assert_scope(session_id, expected_scope)
        return result

    async def read_profile_snapshot(self, conn, session_id, actor_ids, *, expected_scope=None,
                                    for_learning=False, include_pending=False, as_of=None):
        """Profile facts and pending inputs in the caller's existing snapshot."""
        from psycopg import AsyncServerCursor
        from tgchatbot.domain.profiles import profile_document
        from tgchatbot.storage.profiles import _identity, add_source_dates
        scope = await (await conn.execute('SELECT generation,context_id,revision FROM sessions '
                                          'WHERE session_id=%s', (session_id,))).fetchone()
        if scope is None and expected_scope is not None:
            raise StaleScopeError('conversation scope is unavailable')
        if scope is not None:
            self._check_scope(scope, expected_scope)
        if as_of is None:
            as_of = (await (await conn.execute('SELECT clock_timestamp() AS at')).fetchone())['at']
        profiles, facts = [], []
        for actor_id in actor_ids:
            identity, accepted = None, []
            if scope is not None and actor_id != 'unknown':
                identity = await _identity(conn, session_id, scope['generation'], actor_id)
            if scope is not None and actor_id != 'unknown':
                async with AsyncServerCursor(conn, name='profile_snapshot') as cursor:
                    cursor.itersize = self.config.read_page_size
                    await cursor.execute('SELECT f.* ' + self._current_profile_query(include_future=for_learning) + ' ORDER BY f.id DESC',
                        (session_id, actor_id, as_of) if for_learning else (session_id, actor_id, as_of, as_of))
                    async for fact in cursor:
                        accepted.append(fact)
            document = profile_document(actor_id, identity, accepted, known_agent=scope is not None)
            profiles.append(document)
            facts.extend(accepted)
        if for_learning:
            await add_source_dates(conn, profiles)
        if include_pending:
            # Learning publishes facts and consumes source spans atomically.
            # Operator progress must describe that same committed snapshot.
            pending_material = {actor_id: {'sources': 0, 'bytes': 0,
                'first_message_id': None, 'last_message_id': None} for actor_id in actor_ids}
            if scope is not None:
                pending = await (await conn.execute('''SELECT actor_id,count(*) AS sources,
                    sum(pending_bytes) AS bytes,min(message_id) AS first_message_id,
                    max(message_id) AS last_message_id FROM profile_inputs
                    WHERE session_id=%s AND generation=%s AND actor_id=ANY(%s) AND pending_bytes>0
                    GROUP BY actor_id''', (session_id, scope['generation'], list(actor_ids)))).fetchall()
                pending_material.update({row['actor_id']: {key: int(value) for key, value in row.items()
                    if key != 'actor_id'} for row in pending})
        return {'scope': dict(scope) if scope is not None else None, 'as_of': as_of,
                'profiles': profiles, 'facts': facts,
                **({'pending_material': pending_material} if include_pending else {})}

    async def resolve_profile_fact_sources(self, session_id: str, fact_ids: Sequence[int], *,
                                            expected_scope: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Resolve profile references to intact originals without refreshing learning.

        A fact can remain inspectable after leaving the current profile. Changed
        or unavailable evidence, another chat, and earlier generations cannot.
        """
        ids = _ids(fact_ids)
        if not ids:
            return []
        async with self.pool.connection() as conn:
            await conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
            scope = await (await conn.execute('SELECT generation,context_id,revision FROM sessions '
                'WHERE session_id=%s', (session_id,))).fetchone()
            if scope is None:
                if expected_scope is not None:
                    raise StaleScopeError('conversation scope is unavailable')
                return []
            self._check_scope(scope, expected_scope)
            rows = await (await conn.execute('''SELECT f.id AS fact_id,f.subject_actor_id AS actor_id,
                evidence.source_ids,
                (COALESCE(f.id=ANY(c.fact_ids),false) AND f.status IN ('active','superseded')
                    AND (f.valid_from IS NULL OR f.valid_from<=now())
                    AND (f.valid_to IS NULL OR f.valid_to>now())) AS current
                FROM profile_facts f LEFT JOIN profile_current c
                    ON c.session_id=f.session_id AND c.generation=f.generation AND c.actor_id=f.subject_actor_id
                CROSS JOIN LATERAL (
                    SELECT array_agg(m.id ORDER BY m.sent_at,m.id) AS source_ids
                    FROM unnest(f.source_ids) source(id) JOIN messages m ON m.id=source.id
                    AND m.session_id=f.session_id AND m.generation=f.generation
                    AND NOT m.hidden AND NOT m.deleted
                    AND m.source_revision::text=f.source_revisions->>m.id::text
                ) evidence
                WHERE f.session_id=%s AND f.generation=%s AND f.id=ANY(%s) AND f.valid
                    AND cardinality(evidence.source_ids)=cardinality(f.source_ids)
                ORDER BY f.id''', (session_id, scope['generation'], ids))).fetchall()
        if expected_scope is not None:
            # Use a new read snapshot to detect an edit/reset during resolution;
            # scope discovery here must never create a session as a side effect.
            current = await self.get_existing_scope(session_id)
            if current is None:
                raise StaleScopeError('conversation scope is unavailable')
            self._check_scope(current, expected_scope)
        return rows

    async def claim_profile_batch(self, **kwargs):
        from tgchatbot.storage.profiles import claim_batch
        return await claim_batch(self, **kwargs)

    async def apply_profile_patch(self, job, additions, removals, *, max_bytes, expected_source_revisions):
        from tgchatbot.storage.profiles import apply_patch
        await apply_patch(self, job, additions, removals, max_bytes=max_bytes,
            expected_source_revisions=expected_source_revisions)

    async def enqueue_job(self, session_id: str, kind: str, *, source_ids: Sequence[int] | None = None,
                            source_message_ids: Sequence[int] | None = None, payload: dict | None = None,
                            policy: str = 'memory', dedupe_key: str | None = None,
                            expected_scope: Mapping[str, Any] | None = None) -> dict[str, Any]:
        if policy not in {'memory', 'context'}:
            raise ValueError('job policy must be memory or context')
        ids = _ids(source_ids if source_ids is not None else source_message_ids or [])
        async with self.pool.connection() as conn:
            scope = await self._session(conn, session_id, lock=True)
            self._check_scope(scope, expected_scope, context=policy == 'context')
            sources = await self._sources(conn, session_id, scope, ids, current_context=policy == 'context') if ids else []
            return await (await conn.execute('''INSERT INTO jobs
                (session_id,generation,context_id,scope_revision,kind,policy,source_ids,source_revisions,payload,dedupe_key)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (session_id,generation,kind,dedupe_key) DO UPDATE SET dedupe_key=excluded.dedupe_key RETURNING *''',
                (session_id, scope['generation'], scope['context_id'], scope['revision'], kind, policy, ids,
                 Jsonb(self._source_revisions(sources)), Jsonb(payload or {}), dedupe_key))).fetchone()

    async def coalesce_memory_jobs(self, session_id: str, *, source_ids: Sequence[int],
                                   expected_scope: Mapping[str, Any] | None = None) -> dict[str, Any] | None:
        """Replace a committed import page's singleton queue rows atomically."""
        ids = _ids(source_ids)
        if not ids:
            return None
        async with self.pool.connection() as conn:
            scope = await self._session(conn, session_id, lock=True)
            self._check_scope(scope, expected_scope, context=False)
            sources = await self._sources(conn, session_id, scope, ids)
            current_revisions = self._source_revisions(sources)
            removed = await (await conn.execute('''DELETE FROM jobs j
                WHERE session_id=%s AND generation=%s AND kind='memory_ingest' AND status='pending'
                AND cardinality(source_ids)=1 AND source_ids[1]=ANY(%s)
                AND source_revisions=jsonb_build_object(j.source_ids[1]::text,%s::jsonb -> j.source_ids[1]::text)
                RETURNING source_ids''',
                (session_id, scope['generation'], ids, Jsonb(current_revisions)))).fetchall()
            merged = _ids(source for row in removed for source in row['source_ids'])
            if not merged:
                return None
            revisions = {str(mid): current_revisions[str(mid)] for mid in merged}
            dedupe = 'coalesced:' + _json_hash([merged, revisions, scope['revision']])
            return await (await conn.execute('''INSERT INTO jobs
                (session_id,generation,context_id,scope_revision,kind,policy,source_ids,source_revisions,payload,dedupe_key)
                VALUES (%s,%s,%s,%s,'memory_ingest','memory',%s,%s,'{}',%s)
                ON CONFLICT(session_id,generation,kind,dedupe_key) DO UPDATE SET dedupe_key=excluded.dedupe_key RETURNING *''',
                (session_id, scope['generation'], scope['context_id'], scope['revision'], merged, Jsonb(revisions), dedupe))).fetchone()

    async def claim_jobs(self, limit: int | None = None, *, kind: str | None = None,
                         lease_seconds: float | None = None) -> list[dict[str, Any]]:
        lease_seconds = _duration(self.config.job_lease_seconds if lease_seconds is None else lease_seconds, positive=True)
        clause = ' AND j.kind=%s' if kind else ''
        parameters: list[Any] = [kind] if kind else []
        parameters.extend([_limit(limit, self.config.job_claim_size), str(uuid.uuid4()), lease_seconds])
        async with self.pool.connection() as conn:
            return await (await conn.execute('''WITH candidates AS (
                SELECT j.id FROM jobs j JOIN sessions s ON s.session_id=j.session_id AND s.generation=j.generation
                WHERE (j.status='pending' OR (j.status='running' AND j.lease_until<now())) AND j.available_at<=now()
                AND (j.policy='memory' OR (j.context_id=s.context_id AND j.scope_revision=s.revision))'''
                + clause + ''' ORDER BY j.available_at,j.id FOR UPDATE OF j SKIP LOCKED LIMIT %s)
                UPDATE jobs j SET status='running',attempts=attempts+1,lease_token=%s,
                lease_until=now()+(%s * interval '1 second') FROM candidates c WHERE j.id=c.id RETURNING j.*''',
                parameters)).fetchall()

    async def complete_job(self, job_id: int | Mapping[str, Any], *, lease_token: str | None = None,
                             error: str | None = None, retry: bool = False) -> bool:
        if isinstance(job_id, Mapping):
            lease_token, job_id = lease_token or job_id.get('lease_token'), int(job_id['id'])
        async with self.pool.connection() as conn:
            initial = await (await conn.execute('SELECT session_id FROM jobs WHERE id=%s', (job_id,))).fetchone()
            if not initial:
                return False
            scope = await self._session(conn, initial['session_id'], lock=True)
            job = await (await conn.execute('SELECT * FROM jobs WHERE id=%s FOR UPDATE', (job_id,))).fetchone()
            if job['status'] != 'running' or job['lease_token'] != lease_token or job['lease_until'] < datetime.now(timezone.utc):
                return False
            valid = job['generation'] == scope['generation'] and (job['policy'] == 'memory' or
                (job['context_id'] == scope['context_id'] and job['scope_revision'] == scope['revision']))
            if valid and job['source_ids']:
                try:
                    await self._sources(conn, job['session_id'], scope, job['source_ids'], expected_revisions=job['source_revisions'])
                except StaleScopeError:
                    valid = False
            status = 'stale' if not valid else 'pending' if retry and job['attempts'] < self.config.job_max_attempts else 'failed' if error or retry else 'done'
            if status == 'done':
                # Completed work is represented by source-backed projections,
                # not an ever-growing operational event ledger.
                await conn.execute('DELETE FROM jobs WHERE id=%s', (job_id,))
                return True
            await conn.execute('''UPDATE jobs SET status=%s,lease_token=NULL,lease_until=NULL,error=%s,
                finished_at=CASE WHEN %s='pending' THEN NULL ELSE now() END,
                available_at=now()+(%s * interval '1 second') WHERE id=%s''',
                (status, error, status, self.config.job_retry_delay_seconds, job_id))
            return valid and status == 'done'

    async def _locked_job(self, conn: AsyncConnection, requested: Mapping[str, Any]) -> dict[str, Any] | None:
        initial = await (await conn.execute('SELECT session_id FROM jobs WHERE id=%s', (int(requested['id']),))).fetchone()
        if initial is None:
            return None
        scope = await self._session(conn, initial['session_id'], lock=True)
        job = await (await conn.execute('SELECT * FROM jobs WHERE id=%s FOR UPDATE', (int(requested['id']),))).fetchone()
        if job is None or job['status'] != 'running' or job['lease_token'] != requested.get('lease_token') or job['lease_until'] < datetime.now(timezone.utc):
            return None
        valid = job['generation'] == scope['generation'] and (job['policy'] == 'memory' or
            (job['context_id'] == scope['context_id'] and job['scope_revision'] == scope['revision']))
        if valid and job['source_ids']:
            try:
                await self._sources(conn, job['session_id'], scope, job['source_ids'], expected_revisions=job['source_revisions'])
            except StaleScopeError:
                valid = False
        if not valid:
            await conn.execute("UPDATE jobs SET status='stale',finished_at=now(),lease_token=NULL,lease_until=NULL WHERE id=%s", (job['id'],))
            return None
        return job

    async def update_job_payload(self, job: Mapping[str, Any], payload: Mapping[str, Any]) -> bool:
        """Persist an external operation's intent before submitting it."""
        async with self.pool.connection() as conn:
            locked = await self._locked_job(conn, job)
            if locked is None:
                return False
            await conn.execute('UPDATE jobs SET payload=%s WHERE id=%s', (Jsonb(dict(payload)), locked['id']))
            return True

    async def defer_job(self, job: Mapping[str, Any], payload: Mapping[str, Any] | None, delay_seconds: float) -> bool:
        """Return owned work to pending; None preserves its latest durable payload."""
        delay_seconds = _duration(delay_seconds)
        async with self.pool.connection() as conn:
            locked = await self._locked_job(conn, job)
            if locked is None:
                return False
            await conn.execute('''UPDATE jobs SET payload=%s,status='pending',lease_token=NULL,lease_until=NULL,
                available_at=now()+(%s * interval '1 second'),attempts=GREATEST(attempts-1,0) WHERE id=%s''',
                (Jsonb(dict(payload) if payload is not None else locked['payload']), float(delay_seconds), locked['id']))
            return True

    async def renew_job(self, job: Mapping[str, Any], *, lease_seconds: float | None = None) -> bool:
        lease_seconds = _duration(self.config.job_lease_seconds if lease_seconds is None else lease_seconds, positive=True)
        async with self.pool.connection() as conn:
            locked = await self._locked_job(conn, job)
            if locked is None:
                return False
            await conn.execute("UPDATE jobs SET lease_until=now()+(%s * interval '1 second') WHERE id=%s",
                (lease_seconds, locked['id']))
            return True

    async def job_status(self, session_id: str | None = None) -> list[dict[str, Any]]:
        async with self.pool.connection() as conn:
            return await self.read_job_status(conn, session_id)

    async def read_job_status(self, conn, session_id=None):
        clause = ' AND j.session_id=%s' if session_id is not None else ''
        return await (await conn.execute('''SELECT j.kind,j.status,count(*) AS count,min(j.created_at) AS oldest_at
            FROM jobs j JOIN sessions s ON s.session_id=j.session_id AND s.generation=j.generation
            WHERE (j.policy='memory' OR (j.context_id=s.context_id AND j.scope_revision=s.revision))'''
            + clause + ' GROUP BY j.kind,j.status ORDER BY j.kind,j.status',
            (session_id,) if session_id is not None else ())).fetchall()

    async def cleanup_jobs(self, *, limit: int | None = None, older_than_seconds: float | None = None) -> int:
        """Bound the operational queue; originals and fact provenance are untouched."""
        older_than_seconds = _duration(self.config.job_retention_seconds if older_than_seconds is None else older_than_seconds)
        async with self.pool.connection() as conn:
            rows = await (await conn.execute('''WITH obsolete AS (
                SELECT j.id FROM jobs j JOIN sessions s ON s.session_id=j.session_id
                WHERE ((j.status IN ('done','failed','stale') AND j.finished_at<now()-(%s * interval '1 second'))
                    OR j.generation<>s.generation OR (j.policy='context' AND
                        (j.context_id<>s.context_id OR j.scope_revision<>s.revision)))
                AND NOT (j.kind='embedding_batch' AND (j.payload ? 'name' OR COALESCE(j.payload->>'phase' IN ('submitting','polling'),false)))
                ORDER BY j.id FOR UPDATE OF j SKIP LOCKED LIMIT %s)
                DELETE FROM jobs j USING obsolete o WHERE j.id=o.id RETURNING j.id''',
                (older_than_seconds, _limit(limit, self.config.job_cleanup_page_size)))).fetchall()
            return len(rows)


async def _main_check() -> None:
    # Deploy provides its existing .env through Docker. No session/model/token setup
    # is needed for this read-only preflight.
    from dotenv import load_dotenv
    load_dotenv()
    dsn = os.environ.get('DATABASE_URL')
    if not dsn:
        password = os.environ.get('POSTGRES_PASSWORD')
        if not password:
            raise SystemExit('DATABASE_URL or POSTGRES_PASSWORD is required')
        dsn = f'postgresql://tgchatbot:{quote(password, safe="")}@postgres:5432/tgchatbot'
    store = PostgresStore(dsn)
    async with await AsyncConnection.connect(dsn, row_factory=dict_row) as conn:
        await conn.execute('SET TRANSACTION READ ONLY')
        await store._check_schema(conn)
    print('Conversation schema is compatible.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Check conversation database compatibility without modifying it.')
    parser.add_argument('--check-schema', action='store_true', required=True)
    parser.parse_args()
    asyncio.run(_main_check())
