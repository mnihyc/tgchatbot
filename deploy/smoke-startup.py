"""Container startup against disposable PostgreSQL, without Telegram/model calls."""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch
import uuid

from dotenv import dotenv_values, load_dotenv
import httpx
import psycopg
from psycopg import sql

import tgchatbot.app as app
from tgchatbot.healthcheck import healthy, start_heartbeat, stop_heartbeat
from tgchatbot.storage.previews import PreviewCache


dsn = os.environ['TEST_DATABASE_URL']
defaults = {key: value or '' for key, value in dotenv_values('/fixture/.env.example').items()}
real_store, real_catalog = app.PostgresStore, app.StickerCatalog


def check_startup(name: str, credentials: dict[str, str], *, fail_catalog=False,
                  operational_overrides: dict[str, str] | None = None) -> None:
    schema = 'smoke_startup_' + uuid.uuid4().hex
    stores, catalogs, loops = [], [], []

    def make_store(database_url, **kwargs):
        store = real_store(database_url, schema=schema, **kwargs)
        stores.append(store)
        return store

    def make_catalog(*args, **kwargs):
        catalog = real_catalog(*args, **kwargs)
        catalogs.append(catalog)
        return catalog

    def mock_polling(bot):
        loop = asyncio.get_event_loop()
        loops.append(loop)

        async def poll_once():
            assert await bot.store.count_sessions() == 0
            assert bot.config.default_provider == name
            assert bot.runtime.memory.store is bot.store
            assert bot.remote_workspace.enabled is False
            assert catalogs[-1].stats()['stickers'] == 0
            assert catalogs[-1].loaded
            assert catalogs[-1].store.store is bot.store
            assert catalogs[-1].delivery_store is bot.runtime.sticker_delivery
            assert bot.runtime.preview_cache.root.is_dir()
            assert bot.config.artifact_dir.parent == bot.config.temp_dir
            assert bot.runtime.memory.embeddings.config.model == 'gemini-embedding-2'
            if operational_overrides:
                from tgchatbot.operational import from_env
                from tgchatbot.tools.import_desktop import ImportConfig
                from tgchatbot.tools.memory import OperationsConfig
                assert bot.runtime.preview_cache.max_bytes == 4096
                assert bot.runtime.memory.config.read_messages == 7
                assert bot.runtime.memory.worker.limits.claim_jobs == 2
                assert bot.runtime.memory.worker.builder.token_limit == 128
                assert bot.runtime.memory.worker.limits.max_active_batches == 2
                assert bot.runtime.memory.embeddings.config.cache_entries == 0
                assert bot.store.config.read_page_size == 3
                assert bot.store.pool.min_size == 0 and bot.store.pool.max_size == 2
                async with bot.store.pool.connection() as connection:
                    timeout = await (await connection.execute('SHOW statement_timeout')).fetchone()
                assert timeout['statement_timeout'] == '1250ms'
                assert from_env(ImportConfig, 'IMPORT').batch_messages == 150
                assert from_env(OperationsConfig, 'MEMORY_OPERATIONS').page_size == 1100
            application = SimpleNamespace(running=True, updater=SimpleNamespace(running=True), bot_data={})
            assert not healthy()
            await start_heartbeat(application)
            await asyncio.sleep(0.01)
            assert healthy(), 'Running polling loop never became healthy'
            await stop_heartbeat(application)
            assert not healthy(), 'Stopped polling loop retained a healthy marker'

        loop.run_until_complete(poll_once())

    with tempfile.TemporaryDirectory(prefix='startup-smoke-', dir='/tmp') as directory:
        health_file = Path(directory) / 'tmp' / 'health.json'
        health_file.parent.mkdir()
        # Container recreation can reuse PID1 while its bound /tmp survives.
        # The new process must not borrow the previous process's readiness.
        health_file.write_text(json.dumps({'pid': os.getpid(), 'time': time.time()}))
        environment = {**defaults, **credentials,
            'PATH': os.environ.get('PATH', ''), 'HOME': os.environ.get('HOME', '/app/data/home'),
            'TGBOT_TOKEN': '123456:synthetic-test-only-token', 'DATABASE_URL': dsn,
            'APP_DATA_DIR': str(Path(directory) / 'data'), 'APP_TEMP_DIR': str(Path(directory) / 'tmp'),
            'APP_HEALTH_FILE': str(health_file), 'LOG_LEVEL': 'WARNING'}
        # Use the real dotenv parser on one ordinary .env. The startup hook
        # selects only this synthetic fixture's location inside the container.
        dotenv_path = Path(directory) / '.env'
        dotenv_path.write_text(''.join(f'{key}={value}\n' for key, value in (operational_overrides or {}).items()))
        try:
            with patch.dict(os.environ, environment, clear=True), \
                 patch.object(app, 'load_dotenv', lambda: load_dotenv(dotenv_path)), \
                 patch.object(app, 'PostgresStore', side_effect=make_store), \
                 patch.object(app, 'StickerCatalog', side_effect=make_catalog), \
                 patch.object(app.TelegramBotApp, 'run_polling', mock_polling), \
                 patch.object(httpx.AsyncClient, 'send', side_effect=AssertionError('External HTTP is forbidden during startup')), \
                 patch.object(httpx.Client, 'send', side_effect=AssertionError('External HTTP is forbidden during startup')):
                if fail_catalog:
                    with patch.object(real_catalog, 'aensure_loaded', side_effect=RuntimeError('synthetic catalog failure')):
                        try:
                            app.main()
                        except RuntimeError as exc:
                            assert str(exc) == 'synthetic catalog failure'
                        else:
                            raise AssertionError('Injected startup failure was swallowed')
                else:
                    app.main()
                assert not healthy(), 'Startup failure or shutdown retained a stale healthy marker'
                assert stores and all(store.pool.closed for store in stores), 'Startup left a PostgreSQL pool open'
                assert all(loop.is_closed() for loop in loops), 'Shutdown left its event loop open'
                assert not list((Path(directory) / 'tmp').glob('previews-*')), 'Shutdown retained temporary previews'
                # The failed-startup branch also must release the exclusive lock.
                cache = PreviewCache(Path(directory) / 'tmp')
                cache.close()
        finally:
            with psycopg.connect(dsn) as connection:
                connection.execute(sql.SQL('DROP SCHEMA IF EXISTS {} CASCADE').format(sql.Identifier(schema)))
    print(f'Fresh startup and cleanup passed: {name}' + (' (injected failure)' if fail_catalog else '')
          + (' (nondefault .env settings)' if operational_overrides else ''), flush=True)


for provider, credentials in (
    ('openai', {'OPENAI_API_KEY': 'synthetic-openai-key'}),
    ('gemini', {'GEMINI_API_KEY': 'synthetic-gemini-key'}),
    ('deepseek', {'DEEPSEEK_API_KEY': 'synthetic-deepseek-key'}),
    ('openrouter', {'OPENROUTER_API_KEY': 'synthetic-openrouter-key', 'OPENROUTER_MODEL': 'example/model'}),
    ('openai', {'OPENAI_API_KEY': 'synthetic-openai-key', 'GEMINI_API_KEY': 'synthetic-gemini-key', 'DEFAULT_PROVIDER': 'openai'}),
):
    check_startup(provider, credentials)
check_startup('openai', {'OPENAI_API_KEY': 'synthetic-openai-key'}, fail_catalog=True)
check_startup('openai', {'OPENAI_API_KEY': 'synthetic-openai-key'}, operational_overrides={
    'MEMORY_PREVIEW_CACHE_BYTES': '4096',
    'MEMORY_READ_MESSAGES': '7', 'MEMORY_WORKER_CLAIM_JOBS': '2',
    'MEMORY_WORKER_EXCERPT_TOKENS': '128', 'MEMORY_WORKER_MAX_ACTIVE_BATCHES': '2',
    'MEMORY_DB_READ_PAGE_SIZE': '3', 'MEMORY_DB_POOL_MIN_SIZE': '0',
    'MEMORY_DB_POOL_MAX_SIZE': '2', 'MEMORY_DB_STATEMENT_TIMEOUT_S': '1.25',
    'EMBEDDING_CACHE_ENTRIES': '0', 'IMPORT_BATCH_MESSAGES': '150',
    'MEMORY_OPERATIONS_PAGE_SIZE': '1100',
})
