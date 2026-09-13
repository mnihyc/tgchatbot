"""Operators can discover, inspect, correct and resume stickers without SQL."""
from __future__ import annotations

from contextlib import redirect_stdout
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import uuid

from PIL import Image
from psycopg import AsyncConnection, sql

from tests.test_sticker_build_workflow import FakeEmbeddings, FakeProvider
from tgchatbot.stickers.build import CatalogBuilder
from tgchatbot.stickers import manage
from tgchatbot.stickers.media import content_hash
from tgchatbot.storage.postgres_store import DatabaseConfig, PostgresStore
from tgchatbot.storage.sticker_catalog import StickerCatalogStore


class StickerCatalogOperationsTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.dsn = os.getenv('TEST_DATABASE_URL')
        if not self.dsn:
            self.skipTest('TEST_DATABASE_URL required')
        self.schema = 'sticker_operations_' + uuid.uuid4().hex
        self.store = PostgresStore(self.dsn, schema=self.schema, config=DatabaseConfig(read_page_size=1))
        await self.store.initialize()
        self.addAsyncCleanup(self.cleanup_store)
        self.catalog = StickerCatalogStore(self.store)
        await self.catalog.initialize()
        temporary = tempfile.TemporaryDirectory(prefix='fixture-sticker-operations-', dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.provider, self.embeddings = FakeProvider(), FakeEmbeddings()
        self.builder = CatalogBuilder(self.catalog, self.provider, self.embeddings)

    async def cleanup_store(self):
        await self.store.close()
        async with await AsyncConnection.connect(self.dsn, autocommit=True) as conn:
            await conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))

    def picture(self, relative_path, color):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (20, 20), color).save(path)
        return 'sha256:' + content_hash(path)

    async def cli(self, arguments):
        output = io.StringIO()
        with patch.object(manage, 'load_dotenv'), \
             patch.object(manage, 'load_config', return_value=SimpleNamespace(database_url=self.dsn)), \
             patch.object(manage, 'PostgresStore', side_effect=lambda dsn: PostgresStore(
                 dsn, schema=self.schema, config=DatabaseConfig(read_page_size=1))), redirect_stdout(output):
            self.assertEqual(await manage.run(manage.parse_args(arguments)), 0)
        return [json.loads(line) for line in output.getvalue().splitlines()]

    async def test_inspection_discovers_assets_and_exports_exact_saved_corrections(self):
        first = self.picture('series/volume1/one.png', 'red')
        second = self.picture('other/two.png', 'blue')
        initial = await self.builder.build(self.root)
        corrections = {'card': {'caption': 'Reviewed caption'}, 'style_tags': ['ink wash']}
        corrected = await self.builder.build(self.root, corrections={first: corrections})
        described = await self.catalog.update_pack_descriptions({'series': 'Dry, playful style'})
        calls = len(self.provider.calls), len(self.embeddings.calls)
        self.assertEqual(len(await self.cli(['revisions'])), 3)
        assets = await self.cli(['assets'])
        self.assertEqual({row['asset_id'] for row in assets}, {first, second})
        self.assertEqual((await self.cli(['assets', '--pack', 'series']))[0]['asset_id'], first)
        detail = (await self.cli(['asset', first]))[0]
        self.assertEqual(detail['generated_card']['caption'], 'Hello')
        self.assertEqual(detail['card']['caption'], 'Reviewed caption')
        self.assertEqual(detail['corrections'], corrections)
        self.assertEqual(detail['aliases'], [{'path': 'series/volume1/one.png', 'pack': 'series'}])
        self.assertEqual(detail['pack_descriptions'], {'series': 'Dry, playful style'})
        self.assertTrue(detail['has_image_vector'])
        self.assertTrue(detail['has_reading_vectors'])
        self.assertNotIn('image_vector', detail)
        self.assertNotIn('reading_vectors', detail)
        self.assertEqual((await self.cli(['asset', first, '--revision', initial.revision_id]))[0]['corrections'], {})
        exported = (await self.cli(['corrections']))[0]
        self.assertEqual(exported, {first: corrections, second: {}})
        self.assertEqual((await self.cli(['corrections', '--asset-id', first]))[0], {first: corrections})
        self.assertEqual((await self.cli(['corrections', '--pack', 'other']))[0], {second: {}})
        self.assertEqual(await self.catalog.active_revision_id(), described)
        self.assertEqual((len(self.provider.calls), len(self.embeddings.calls)), calls)

        # The exported shape is the exact existing mutation owner's input.
        reapplied = await self.builder.build(self.root, corrections=exported)
        self.assertTrue(reapplied.active)
        self.assertEqual((len(self.provider.calls), len(self.embeddings.calls)), calls)
        cleared = await self.builder.build(self.root, corrections={first: {}})
        self.assertTrue(cleared.active)
        self.assertEqual((await self.catalog.inspect_asset(first))['corrections'], {})
        self.assertEqual((await self.catalog.inspect_asset(first))['card']['caption'], 'Hello')
        self.assertEqual((await self.catalog.inspect_asset(first, corrected.revision_id))['corrections'], corrections)

    async def test_lost_build_output_can_be_recovered_and_saved_channels_resumed(self):
        self.picture('series/one.png', 'red')
        active = await self.builder.build(self.root)
        failed_id = self.picture('series/two.png', 'blue')
        self.embeddings.fail_images = True
        await self.builder.build(self.root)  # Simulate losing stdout/result identity.
        calls = len(self.provider.calls), len(self.embeddings.calls)
        revisions = await self.cli(['revisions'])
        staging = next(row for row in revisions if row['state'] == 'staging')
        self.assertTrue(staging['parent_is_current'])
        status = (await self.cli(['status', '--revision', staging['revision_id']]))[0]
        self.assertEqual(status['active_revision_id'], active.revision_id)
        self.assertEqual(status['source_root'], str(self.root))
        self.assertEqual(status['recipe'], self.builder.recipe)
        self.assertEqual(status['counts']['failed'], {
            'assets': 1, 'saved_annotations': 1, 'saved_reading_vectors': 1, 'saved_image_vectors': 0})
        self.assertEqual(status['failures'][0]['asset_id'], failed_id)
        self.assertIn('Synthetic image embedding outage', status['failures'][0]['error'])
        pending = await self.cli(['assets', '--revision', staging['revision_id'], '--state', 'failed'])
        self.assertEqual([row['asset_id'] for row in pending], [failed_id])
        self.assertEqual((len(self.provider.calls), len(self.embeddings.calls)), calls)
        self.embeddings.fail_images = False
        resumed = await self.builder.build(Path(status['source_root']), resume=staging['revision_id'])
        self.assertTrue(resumed.active)
        self.assertEqual(len(self.provider.calls), calls[0])
        self.assertEqual(len(self.embeddings.calls), calls[1] + 1)
        final = (await self.cli(['status']))[0]
        self.assertEqual(final['revision_id'], resumed.revision_id)
        self.assertEqual(final['failures'], [])
        self.assertEqual(final['counts']['ready']['assets'], 2)

    async def test_empty_unknown_and_filtered_catalogs_are_distinguishable(self):
        self.assertEqual(await self.cli(['revisions']), [])
        self.assertEqual(await self.cli(['assets']), [])
        self.assertEqual((await self.cli(['status']))[0]['revision_id'], None)
        self.assertEqual(await self.cli(['corrections']), [{}])
        with self.assertRaises(KeyError):
            await self.cli(['status', '--revision', 'missing'])
        with self.assertRaises(KeyError):
            await self.cli(['asset', 'missing'])
        known = self.picture('series/one.png', 'red')
        await self.builder.build(self.root)
        with self.assertRaises(KeyError):
            await self.cli(['corrections', '--asset-id', known, '--asset-id', 'missing'])
        self.assertEqual(await self.cli(['assets', '--pack', 'other']), [])

    async def test_dates_use_configured_timezone_without_rewriting_card_evidence(self):
        output = io.StringIO()
        with patch.dict(os.environ, {'DEFAULT_METADATA_TIMEZONE': 'Asia/Singapore'}), redirect_stdout(output):
            manage.emit({'created_at': datetime(2026, 1, 1, tzinfo=timezone.utc),
                         'card': {'caption': '2026-01-01T00:00:00Z'}})
        rendered = json.loads(output.getvalue())
        self.assertEqual(rendered['created_at'], '2026-01-01T08:00:00+08:00')
        self.assertEqual(rendered['card']['caption'], '2026-01-01T00:00:00Z')

    async def test_selected_asset_reads_release_a_single_connection_before_return(self):
        asset_id = self.picture('series/one.png', 'red')
        await self.builder.build(self.root)
        single = PostgresStore(self.dsn, schema=self.schema, config=DatabaseConfig(
            pool_min_size=0, pool_max_size=1, pool_timeout_s=1, read_page_size=1))
        await single.pool.open(wait=True)
        self.addAsyncCleanup(single.close)
        catalog = StickerCatalogStore(single)
        iterator = catalog.iter_assets
        retained = []

        def retain(*args, **kwargs):
            records = iterator(*args, **kwargs)
            retained.append(records)
            return records

        # Keep references to prevent garbage collection from concealing a cursor
        # left open by an early return. The next operation needs the same slot.
        try:
            with patch.object(catalog, 'iter_assets', side_effect=retain):
                for _ in range(3):
                    self.assertEqual((await catalog.inspect_asset(asset_id))['asset_id'], asset_id)
            self.assertEqual((await catalog.inspect_revision())['counts']['ready']['assets'], 1)
        finally:
            for records in retained:
                await records.aclose()
