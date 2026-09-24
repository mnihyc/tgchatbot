"""Operator views of committed context and the agent's existing retrieval service."""
from __future__ import annotations

import base64
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

from tgchatbot.core.memory import MemoryService
from tgchatbot.domain.models import PartKind
from tgchatbot.domain.timestamps import format_timestamp_fields, resolve_timezone
from tgchatbot.embeddings import EmbeddingClient


async def prepared_inspection(store, config, session_id, *, topic='overview', limit=5, object_id=None):
    """Construct only local adapters/declarations, never SSH or API operations."""
    from contextlib import AsyncExitStack
    from tgchatbot.core.inspection import inspect_context
    from tgchatbot.core.memory import MemoryService
    from tgchatbot.core.runtime import AgentRuntime
    from tgchatbot.embeddings import EmbeddingConfig
    from tgchatbot.providers.factory import build_provider
    from tgchatbot.stickers.catalog import StickerCatalog
    from tgchatbot.storage.sticker_catalog import StickerCatalogStore
    from tgchatbot.tools.registry import ToolRegistry
    from tgchatbot.tools.remote_workspace import RemoteWorkspaceClient
    async with AsyncExitStack() as cleanup:
        names = {'openai', 'gemini', *(profile.name for profile in config.chat_completions)}
        providers = {}
        for name in names:
            provider = build_provider(config, name, require_credentials=False)
            cleanup.push_async_callback(provider.aclose)
            providers[name] = provider
        catalog = StickerCatalog(StickerCatalogStore(store), config.sticker_dir, persona_store=store)
        remote = SimpleNamespace(enabled=RemoteWorkspaceClient.configured(config), _master_started=False)
        tools = ToolRegistry(config, remote, catalog)
        memory = MemoryService(store, SimpleNamespace(enabled=EmbeddingConfig.from_env().enabled), config=config.memory)
        runtime = AgentRuntime(config=config, store=store, providers=providers, tool_registry=tools, memory=memory)
        return await inspect_context(runtime, session_id, topic, limit=limit, object_id=object_id)


async def context_records(store, session_id, *, options):
    # Counts and summaries share a snapshot. Raw originals stay out of this view;
    # the existing search/read/audit commands own inspecting their contents.
    async with store.pool.connection() as conn:
        await conn.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        scope = await (await conn.execute('''SELECT generation,context_id,revision,
            context_version,compaction_version,profile_refresh_version FROM sessions
            WHERE session_id=%s''', (session_id,))).fetchone()
        if scope is None:
            raise ValueError(f'Chat {session_id} does not exist; import or receive messages first')
        counts = await (await conn.execute('''SELECT count(*) AS visible_messages,
            count(*) FILTER (WHERE compacted_by_block_id IS NULL) AS uncompacted_messages,
            min(id) FILTER (WHERE compacted_by_block_id IS NULL) AS first_uncompacted_message_id,
            max(id) FILTER (WHERE compacted_by_block_id IS NULL) AS last_uncompacted_message_id
            FROM messages WHERE session_id=%s AND generation=%s AND context_id=%s
            AND NOT hidden AND NOT deleted''',
            (session_id, scope['generation'], scope['context_id']))).fetchone()
        # Reuse the storage owner's root selection, including covered episodes,
        # sealed/open lifecycle and reset visibility, instead of inventing a
        # second definition of what remains in the compaction pipeline.
        root_query = store._root_block_query()
        blocks = await (await conn.execute('''SELECT count(*) AS root_blocks,
            COALESCE(sum(estimated_tokens),0) AS root_block_estimated_tokens
            FROM (''' + root_query + ') roots', (session_id, session_id))).fetchone()
        yield {'type': 'context', 'session_id': session_id, **scope, **counts, **blocks}
        async with conn.cursor(name='operator_context_blocks') as cursor:
            cursor.itersize = options.page_size
            await cursor.execute(root_query + ' ORDER BY b.sequence_no,b.id', (session_id, session_id))
            async for row in cursor:
                yield {'type': 'context_block', 'session_id': session_id,
                       **format_timestamp_fields(asdict(store._block(row)), ('time_start', 'time_end'))}


async def query_memory(store, config, embedding_config, session_id, *, query=None,
                       message_ids=None, actor_id=None, before=None, after=None,
                       limit=None, offset=0, length=None, include_neighbors=False,
                       lexical_only=False):
    from tgchatbot.tools.memory import existing_scope
    scope = await existing_scope(store, session_id)
    client = None
    try:
        # Read and explicitly lexical search require no embedding credentials or
        # client. No worker is attached: inspection never learns a profile.
        embeddings = SimpleNamespace(enabled=False, space_id='')
        if query is not None and not lexical_only:
            client = EmbeddingClient(embedding_config)
            embeddings = client
        memory = MemoryService(store, embeddings, config=config.memory)
        if query is not None:
            result = await memory.search(session_id, query, scope=scope, actor_id=actor_id,
                before=before, after=after, limit=limit)
            if lexical_only:
                result['coverage'] = 'lexical only: explicitly requested by operator'
            return result
        return await memory.read(session_id, message_ids, scope=scope, offset=offset,
            length=length, include_neighbors=include_neighbors)
    finally:
        if client is not None:
            await client.aclose()


async def export_image(store, session_id, *, message_id, image_id, output: Path):
    from tgchatbot.tools.memory import existing_scope
    scope = await existing_scope(store, session_id)
    selected = await store.resolve_message_images(session_id, [message_id], [image_id],
        expected_scope=scope, timezone=resolve_timezone().key)
    result = selected['image_results'][0]
    if result['status'] != 'selected':
        raise ValueError(result['reason'])
    part = next(part for part in selected['evidence_parts'] if part.kind == PartKind.IMAGE)
    payload = base64.b64decode(part.data_b64)
    # The caller chooses the path; do not overwrite an unrelated operator file.
    with output.open('xb') as target:
        target.write(payload)
    return {'type': 'image_exported', 'session_id': session_id, 'message_id': message_id,
            'image_id': image_id, 'mime_type': part.mime_type, 'bytes': len(payload), 'output': str(output)}
