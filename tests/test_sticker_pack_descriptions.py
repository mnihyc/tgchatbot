"""Manual pack context follows immutable catalog publication without model work."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
import uuid

import numpy as np
from PIL import Image
from psycopg import AsyncConnection, sql

from tests.test_sticker_build_workflow import FakeEmbeddings, FakeProvider
from tgchatbot.stickers.build import CatalogBuilder
from tgchatbot.stickers.media import content_hash
from tgchatbot.storage.postgres_store import PostgresStore
from tgchatbot.storage.sticker_catalog import CatalogConflict, StickerCatalogStore


class StickerPackDescriptionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.dsn = os.getenv('TEST_DATABASE_URL')
        if not self.dsn:
            self.skipTest('TEST_DATABASE_URL required')
        self.schema = 'sticker_packs_' + uuid.uuid4().hex
        self.store = PostgresStore(self.dsn, schema=self.schema)
        await self.store.initialize()
        self.addAsyncCleanup(self.cleanup_store)
        self.catalog = StickerCatalogStore(self.store)
        await self.catalog.initialize()

    async def cleanup_store(self):
        await self.store.close()
        async with await AsyncConnection.connect(self.dsn, autocommit=True) as conn:
            await conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))

    async def publish_fixture(self):
        revision = await self.catalog.begin_revision(source_root='/fixture/stickers', recipe={'fixture': True})
        for asset_id, aliases in [
            ('one', [{'path': 'series/volume1/one.webp', 'pack': 'series'},
                     {'path': 'other/one.webp', 'pack': 'other'}]),
            ('two', [{'path': 'series/volume2/two.webp', 'pack': 'series'},
                     {'path': 'two.webp', 'pack': ''}]),
        ]:
            await self.catalog.stage_asset(revision, asset_id=asset_id, content_hash=asset_id,
                aliases=aliases, media={'frames': ['fixture']}, generated_card={'caption': 'Hello'},
                corrections={'reviewed': True}, card={'caption': 'Hello'},
                provenance={'provider': 'fixture'}, dimensions=3,
                image_vector=np.asarray([1, 0, 0], dtype=np.float32),
                reading_vectors=np.asarray([[0, 1, 0]], dtype=np.float32), state='ready')
        await self.catalog.activate(revision)
        return revision

    async def rows(self, statement, parameters=()):
        async with self.store.pool.connection() as conn:
            return await (await conn.execute(statement, parameters)).fetchall()

    async def test_update_remove_preserves_original_catalog_and_shared_vector_references(self):
        original = await self.publish_fixture()
        before = await self.rows('SELECT * FROM sticker_catalog_items WHERE revision_id=%s ORDER BY asset_id', (original,))
        vectors = await self.rows('SELECT * FROM sticker_catalog_vectors ORDER BY id')
        self.assertEqual(await self.catalog.list_pack_descriptions(), {'other': None, 'series': None})

        described = await self.catalog.update_pack_descriptions({'series': 'Dry, playful exchanges.'})
        self.assertNotEqual(described, original)
        self.assertEqual(await self.catalog.list_pack_descriptions(), {'other': None, 'series': 'Dry, playful exchanges.'})
        after = await self.rows('SELECT * FROM sticker_catalog_items WHERE revision_id=%s ORDER BY asset_id', (described,))
        for old, new in zip(before, after, strict=True):
            self.assertEqual({key: value for key, value in old.items() if key != 'revision_id'},
                             {key: value for key, value in new.items() if key != 'revision_id'})
        self.assertEqual(await self.rows('SELECT * FROM sticker_catalog_vectors ORDER BY id'), vectors)
        self.assertEqual((await self.catalog.load_snapshot(original)).pack_descriptions, {})
        updated = await self.catalog.load_snapshot(described)
        self.assertEqual(updated.pack_descriptions, {'series': 'Dry, playful exchanges.'})
        self.assertEqual(updated.recipe, {'fixture': True})
        self.assertEqual(updated.source_root, '/fixture/stickers')

        removed = await self.catalog.update_pack_descriptions({'series': None})
        self.assertNotEqual(removed, described)
        self.assertEqual(await self.catalog.list_pack_descriptions(), {'other': None, 'series': None})
        self.assertEqual((await self.catalog.load_snapshot(described)).pack_descriptions,
                         {'series': 'Dry, playful exchanges.'})

    async def test_known_unconfigured_remove_and_identical_set_are_noops(self):
        original = await self.publish_fixture()
        self.assertEqual(await self.catalog.update_pack_descriptions({'series': None}), original)
        self.assertEqual(await self.catalog.update_pack_descriptions({}), original)
        described = await self.catalog.update_pack_descriptions({'series': 'Playful'})
        self.assertEqual(await self.catalog.update_pack_descriptions({'series': 'Playful'}), described)
        self.assertEqual(len(await self.rows('SELECT id FROM sticker_catalog_revisions')), 2)

    async def test_unknown_pack_and_blank_description_do_not_publish(self):
        original = await self.publish_fixture()
        for updates in [{'series/volume1': 'Nested folder'}, {'typo': None}, {'': 'Loose files'}]:
            with self.assertRaises(KeyError):
                await self.catalog.update_pack_descriptions(updates)
        with self.assertRaises(ValueError):
            await self.catalog.update_pack_descriptions({'series': ' \n '})
        self.assertEqual(await self.catalog.active_revision_id(), original)
        self.assertEqual(len(await self.rows('SELECT id FROM sticker_catalog_revisions')), 1)

    async def test_append_and_rebuild_inherit_pack_context_and_explicit_empty_clears(self):
        await self.publish_fixture()
        described = await self.catalog.update_pack_descriptions({'series': 'Playful'})
        appended = await self.catalog.begin_revision(source_root='/fixture/stickers', recipe={}, expected_parent=described)
        await self.catalog.stage_asset(appended, asset_id='three', content_hash='three',
            aliases=[{'path': 'series/volume3/three.webp', 'pack': 'series'}],
            media={}, generated_card={}, corrections={}, card={}, provenance={}, state='ready')
        await self.catalog.activate(appended)
        self.assertEqual((await self.catalog.load_snapshot()).pack_descriptions, {'series': 'Playful'})
        snapshot = await self.catalog.load_snapshot()
        rebuilt = await self.catalog.begin_revision(source_root=snapshot.source_root, recipe=snapshot.recipe,
            assets=list(snapshot.assets), dimensions=3, expected_parent=appended)
        await self.catalog.activate(rebuilt)
        self.assertEqual((await self.catalog.load_snapshot()).pack_descriptions, {'series': 'Playful'})
        cleared = await self.catalog.begin_revision(source_root=snapshot.source_root, recipe=snapshot.recipe,
            expected_parent=rebuilt, pack_descriptions={})
        await self.catalog.activate(cleared)
        self.assertEqual((await self.catalog.load_snapshot()).pack_descriptions, {})

    async def test_builder_append_regenerate_and_resume_keep_pack_description(self):
        temporary = tempfile.TemporaryDirectory(prefix='fixture-pack-context-', dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)

        def picture(relative_path, color):
            path = root / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new('RGB', (20, 20), color).save(path)
            return 'sha256:' + content_hash(path)

        target = picture('series/volume1/one.png', 'red')
        picture('other/one.png', 'blue')
        provider, embeddings = FakeProvider(), FakeEmbeddings()
        builder = CatalogBuilder(self.catalog, provider, embeddings)
        first = await builder.build(root)
        self.assertTrue(first.active)
        calls = len(provider.calls), len(embeddings.calls)
        described = await self.catalog.update_pack_descriptions({'series': 'Dry, playful exchanges.'})
        self.assertEqual((len(provider.calls), len(embeddings.calls)), calls)
        original_rows = await self.rows(
            'SELECT * FROM sticker_catalog_items WHERE revision_id=%s ORDER BY asset_id', (described,))

        added = picture('series/volume2/two.png', 'green')
        appended = await builder.build(root)
        self.assertTrue(appended.active)
        self.assertEqual(len(provider.calls), calls[0] + 1)
        snapshot = await self.catalog.load_snapshot()
        self.assertEqual(snapshot.pack_descriptions, {'series': 'Dry, playful exchanges.'})
        self.assertEqual(next(asset for asset in snapshot.assets if asset.asset_id == added).aliases[0].pack, 'series')
        self.assertEqual(await self.catalog.list_pack_descriptions(),
                         {'other': None, 'series': 'Dry, playful exchanges.'})
        appended_rows = {row['asset_id']: row for row in await self.rows(
            'SELECT * FROM sticker_catalog_items WHERE revision_id=%s', (appended.revision_id,))}
        for before in original_rows:
            after = appended_rows[before['asset_id']]
            self.assertEqual({key: value for key, value in before.items() if key != 'revision_id'},
                             {key: value for key, value in after.items() if key != 'revision_id'})

        provider.card['caption'] = 'Revised greeting'
        provider.card['readings'][0]['meaning'] = 'An enthusiastic greeting'
        before_generation = len(provider.calls)
        regenerated = await builder.build(root, regenerate_ids=[target])
        self.assertTrue(regenerated.active)
        self.assertEqual(len(provider.calls), before_generation + 1)
        self.assertEqual((await self.catalog.get_asset(target)).card['caption'], 'Revised greeting')
        self.assertEqual((await self.catalog.load_snapshot()).pack_descriptions,
                         {'series': 'Dry, playful exchanges.'})

        retry_target = picture('series/volume3/three.png', 'yellow')
        embeddings.fail_images = True
        failed = await builder.build(root)
        self.assertFalse(failed.active)
        self.assertEqual(await self.catalog.active_revision_id(), regenerated.revision_id)
        self.assertEqual((await self.catalog.load_snapshot(failed.revision_id)).pack_descriptions,
                         {'series': 'Dry, playful exchanges.'})
        self.assertIsNotNone((await self.catalog.get_asset(retry_target, failed.revision_id)).generated_card)
        before_resume_generation = len(provider.calls)
        embeddings.fail_images = False
        resumed = await builder.build(root, resume=failed.revision_id)
        self.assertTrue(resumed.active)
        self.assertEqual(resumed.revision_id, failed.revision_id)
        self.assertEqual(len(provider.calls), before_resume_generation)
        self.assertEqual((await self.catalog.load_snapshot()).pack_descriptions,
                         {'series': 'Dry, playful exchanges.'})

    async def test_stale_metadata_publication_cannot_overwrite_newer_description(self):
        original = await self.publish_fixture()
        staging = await self.catalog.begin_revision(source_root='/fixture/stickers', recipe={},
            expected_parent=original, pack_descriptions={'series': 'Stale'})
        current = await self.catalog.update_pack_descriptions({'series': 'Current'})
        with self.assertRaises(CatalogConflict):
            await self.catalog.activate(staging)
        with self.assertRaises(CatalogConflict):
            await self.catalog.begin_revision(source_root='/fixture/stickers', recipe={},
                expected_parent=original, pack_descriptions={'series': 'Stale'})
        self.assertEqual(await self.catalog.active_revision_id(), current)
        self.assertEqual(await self.catalog.list_pack_descriptions(), {'other': None, 'series': 'Current'})

    async def test_renamed_pack_keeps_manual_description_queryable_and_removable(self):
        temporary = tempfile.TemporaryDirectory(prefix='fixture-pack-rename-', dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        path = root / 'series' / 'one.png'
        path.parent.mkdir()
        Image.new('RGB', (20, 20), 'red').save(path)
        provider, embeddings = FakeProvider(), FakeEmbeddings()
        builder = CatalogBuilder(self.catalog, provider, embeddings)
        self.assertTrue((await builder.build(root)).active)
        described = await self.catalog.update_pack_descriptions({'series': 'Playful'})
        calls = len(provider.calls), len(embeddings.calls)

        (root / 'series').rename(root / 'renamed')
        rebuilt = await builder.build(root)
        self.assertTrue(rebuilt.active)
        self.assertEqual((len(provider.calls), len(embeddings.calls)), calls)
        self.assertEqual(await self.catalog.list_pack_descriptions(), {'renamed': None, 'series': 'Playful'})
        snapshot = await self.catalog.load_snapshot()
        self.assertEqual({alias.pack for asset in snapshot.assets for alias in asset.aliases}, {'renamed'})
        self.assertEqual(snapshot.pack_descriptions, {'series': 'Playful'})

        changed = await self.catalog.update_pack_descriptions({'series': 'Retained for reuse'})
        self.assertNotEqual(changed, rebuilt.revision_id)
        self.assertEqual(await self.catalog.list_pack_descriptions(),
                         {'renamed': None, 'series': 'Retained for reuse'})
        removed = await self.catalog.update_pack_descriptions({'series': None})
        self.assertNotEqual(removed, changed)
        self.assertEqual(await self.catalog.list_pack_descriptions(), {'renamed': None})
        current = await self.catalog.load_snapshot()
        self.assertEqual(current.pack_descriptions, {})
        self.assertEqual([(asset.asset_id, asset.aliases) for asset in current.assets],
                         [(asset.asset_id, asset.aliases) for asset in snapshot.assets])
        self.assertEqual((await self.catalog.load_snapshot(described)).pack_descriptions, {'series': 'Playful'})
        self.assertEqual((len(provider.calls), len(embeddings.calls)), calls)

    async def test_existing_schema_initialization_adds_empty_context_without_changing_assets(self):
        revision = await self.publish_fixture()
        before = await self.rows('SELECT * FROM sticker_catalog_items WHERE revision_id=%s ORDER BY asset_id', (revision,))
        async with self.store.pool.connection() as conn:
            await conn.execute('ALTER TABLE sticker_catalog_revisions DROP COLUMN pack_descriptions')
        await self.catalog.initialize()
        self.assertEqual((await self.catalog.load_snapshot()).pack_descriptions, {})
        self.assertEqual(await self.rows('SELECT * FROM sticker_catalog_items WHERE revision_id=%s ORDER BY asset_id', (revision,)), before)

    async def test_empty_catalog_lists_nothing_and_cannot_create_orphaned_description(self):
        self.assertEqual(await self.catalog.list_pack_descriptions(), {})
        self.assertEqual((await self.catalog.load_snapshot()).pack_descriptions, {})
        with self.assertRaises(CatalogConflict):
            await self.catalog.update_pack_descriptions({'absent': 'No members yet'})
        self.assertEqual(await self.rows('SELECT id FROM sticker_catalog_revisions'), [])
