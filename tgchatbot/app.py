from __future__ import annotations

import asyncio
import logging

from dotenv import load_dotenv

from tgchatbot.config import load_config
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.logging_config import configure_logging
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.providers.openai_responses import OpenAIResponsesProvider
from tgchatbot.stickers.catalog import StickerCatalog
from tgchatbot.storage.artifacts import ArtifactStore
from tgchatbot.storage.presets import PresetStore
from tgchatbot.storage.sqlite_store import SQLiteStore
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
    providers: dict[str, object] = {}
    if config.openai.api_key:
        providers['openai'] = OpenAIResponsesProvider(config.openai)
    if config.gemini.api_key:
        providers['gemini'] = GeminiProvider(config.gemini)
    if not providers:
        raise RuntimeError('At least one provider API key must be configured: OPENAI_API_KEY and/or GEMINI_API_KEY')
    return providers


def main() -> None:
    load_dotenv()
    config = load_config()
    configure_logging(config.log_level)

    artifact_store = ArtifactStore(config.artifact_dir)
    preset_store = PresetStore(config.preset_dir)
    store = SQLiteStore(config.db_path)
    persisted_sessions = store._count_sessions_sync()
    remote = RemoteWorkspaceClient(config)
    sticker_catalog = StickerCatalog(config.sticker_index_path, config.sticker_dir)
    sticker_catalog.load()
    sticker_stats = sticker_catalog.stats()

    providers = _build_providers(config)
    if config.default_provider not in providers:
        configured_providers = ', '.join(sorted(providers.keys())) or '-'
        raise RuntimeError(
            f'DEFAULT_PROVIDER={config.default_provider!r} is not configured. '
            f'Configured providers: {configured_providers}'
        )
    logger.info(
        'app.start provider_default=%s providers=%s delivery=%s remote=%s data_dir=%s sessions_loaded=%s',
        config.default_provider,
        ','.join(providers.keys()),
        config.default_response_delivery,
        config.ssh_exec.enabled and bool(config.ssh_exec.host),
        config.data_dir,
        persisted_sessions,
    )
    logger.info('stickers.ready loaded=%s count=%s packs=%s index=%s', sticker_stats.get('loaded'), sticker_stats.get('stickers'), sticker_stats.get('packs'), config.sticker_index_path)
    if remote.enabled:
        logger.info('remote.warmup.start host=%s port=%s', config.ssh_exec.host, config.ssh_exec.port)
        asyncio.run(remote.warmup())
        logger.info('remote.warmup.done ready=%s', getattr(remote, '_master_started', False))

    asyncio.set_event_loop(asyncio.new_event_loop())
    runtime = AgentRuntime(
        config=config,
        store=store,
        tool_registry=ToolRegistry(config, remote, sticker_catalog),
        providers=providers,
    )
    bot = TelegramBotApp(
        config=config,
        runtime=runtime,
        store=store,
        artifact_store=artifact_store,
        preset_store=preset_store,
        remote_workspace=remote,
    )
    try:
        bot.run_polling()
    except KeyboardInterrupt:
        logger.info('app.stop signal=keyboard_interrupt')
    finally:
        asyncio.run(_cleanup(remote, providers))
        logger.info('app.stop complete=1')


if __name__ == '__main__':
    main()
