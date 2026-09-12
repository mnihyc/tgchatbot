from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv

from tgchatbot.config import load_config
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.core.memory import MemoryService
from tgchatbot.core.memory_worker import MemoryWorker
from tgchatbot.embeddings import EmbeddingClient, EmbeddingConfig
from tgchatbot.embeddings.config import sticker_embedding_config
from tgchatbot.storage.sticker_catalog import StickerCatalogStore
from tgchatbot.storage.sticker_delivery import StickerDeliveryStore
from tgchatbot.logging_config import configure_logging
from tgchatbot.healthcheck import health_path
from tgchatbot.providers.factory import build_providers
from tgchatbot.stickers.catalog import StickerCatalog
from tgchatbot.storage.artifacts import ArtifactStore
from tgchatbot.storage.presets import PresetStore
from tgchatbot.storage.postgres_store import PostgresStore
from tgchatbot.storage.previews import PreviewCache
from tgchatbot.tools.registry import ToolRegistry
from tgchatbot.tools.remote_workspace import RemoteWorkspaceClient
from tgchatbot.transports.telegram_adapter import TelegramBotApp

logger = logging.getLogger(__name__)


async def _cleanup(remote: RemoteWorkspaceClient, providers: dict[str, object]) -> None:
    for name, provider in providers.items():
        aclose = getattr(provider, 'aclose', None)
        if callable(aclose):
            try:
                await aclose()
            except Exception as exc:
                logger.warning('cleanup.provider_failed name=%s err=%s', name, exc.__class__.__name__)
    aclose_remote = getattr(remote, 'aclose', None)
    if callable(aclose_remote):
        try:
            await aclose_remote()
        except Exception as exc:
            logger.warning('cleanup.remote_failed err=%s', exc.__class__.__name__)


def _build_providers(config) -> dict[str, object]:
    return build_providers(config)


def main() -> None:
    load_dotenv()
    config = load_config()
    configure_logging(config.log_level, timezone_name=config.default_metadata_timezone)
    # Container recreation may reuse a PID while retaining /tmp. Readiness
    # belongs to this startup, not a recent marker from the previous process.
    marker = health_path()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text('{}')
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    store = embeddings = sticker_embeddings = previews = sticker_catalog = worker = remote = None
    providers = {}
    try:
        artifact_store = ArtifactStore(config.artifact_dir)
        preset_store = PresetStore(config.preset_dir)
        store = PostgresStore(config.database_url)
        previews = PreviewCache(store, max_bytes=config.memory.preview_cache_bytes)
        embeddings = EmbeddingClient(EmbeddingConfig.from_env())
        if embeddings.config.dimensions != 1536:
            raise ValueError('Conversation memory requires EMBEDDING_DIMENSIONS=1536; rebuild with a compatible schema to change it')
        if not embeddings.enabled:
            logger.warning('memory.embeddings_unconfigured semantic_search=unavailable lexical_search=available')
        loop.run_until_complete(store.initialize())
        persisted_sessions = loop.run_until_complete(store.count_sessions())
        remote = RemoteWorkspaceClient(config)
        catalog_store = StickerCatalogStore(store)
        loop.run_until_complete(catalog_store.initialize())
        delivery_store = StickerDeliveryStore(store)
        loop.run_until_complete(delivery_store.initialize(recover_interrupted=True))
        sticker_embeddings = EmbeddingClient(sticker_embedding_config())
        sticker_catalog = StickerCatalog(catalog_store, config.sticker_dir,
            persona_store=store, embedding_client=sticker_embeddings, delivery_store=delivery_store)
        loop.run_until_complete(sticker_catalog.aensure_loaded())
        providers = _build_providers(config)
        if config.default_provider not in providers:
            available = ', '.join(sorted(providers)) or '-'
            raise RuntimeError(f'DEFAULT_PROVIDER={config.default_provider!r} is not configured. Configured providers: {available}')
        logger.info('app.start provider_default=%s providers=%s delivery=%s remote=%s sessions_loaded=%s',
                    config.default_provider, ','.join(providers), config.default_response_delivery,
                    remote.enabled, persisted_sessions)
        logger.info('stickers.ready stats=%s', sticker_catalog.stats())
        if remote.enabled:
            loop.run_until_complete(remote.warmup())
        memory = MemoryService(store, embeddings, config=config.memory)
        worker = MemoryWorker(store=store, embeddings=embeddings, providers=providers, config=config)
        memory.worker = worker
        runtime = AgentRuntime(config=config, store=store,
            tool_registry=ToolRegistry(config, remote, sticker_catalog), providers=providers,
            memory=memory, preview_cache=previews, sticker_delivery=delivery_store)
        bot = TelegramBotApp(config=config, runtime=runtime, store=store,
            artifact_store=artifact_store, preset_store=preset_store, remote_workspace=remote)
        async def start_worker():
            worker.start()
        loop.run_until_complete(start_worker())
        bot.run_polling()
    except KeyboardInterrupt:
        logger.info('app.stop signal=keyboard_interrupt')
    finally:
        # Startup can fail after opening a database or HTTP client. The same
        # cleanup owns both partial startup and normal polling shutdown.
        if worker is not None:
            loop.run_until_complete(worker.close())
        loop.run_until_complete(_cleanup(remote, providers))
        if sticker_embeddings is not None:
            loop.run_until_complete(sticker_embeddings.aclose())
        if embeddings is not None:
            loop.run_until_complete(embeddings.aclose())
        if store is not None:
            loop.run_until_complete(store.close())
        if previews is not None:
            previews.close()
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
        asyncio.set_event_loop(None)
        logger.info('app.stop complete=1')


if __name__ == '__main__':
    main()
