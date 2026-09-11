"""Catalog publication and maintenance with original files, real PG and mock models."""
from __future__ import annotations
import copy
from dataclasses import replace
import json
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import uuid

import numpy as np
from PIL import Image
from psycopg import AsyncConnection, sql

from tgchatbot.domain.models import ProviderResponse
from tgchatbot.storage.postgres_store import DatabaseConfig, PostgresStore
from tgchatbot.storage.sticker_catalog import StickerCatalogStore, CatalogConflict
from tgchatbot.stickers.build import CatalogBuilder, BuildConfig
from tgchatbot.stickers.media import content_hash

CARD = {'caption': 'Hello', 'appearance': 'A round blue bird', 'action': 'A raised wing',
        'readings': [{'meaning': 'Greeting', 'context': 'Opening a friendly conversation'}],
        'uncertainty': '', 'compatibility': {'harshness_level': 0, 'intimacy_level': 0, 'meme_dependence_level': 0}}


class FakeProvider:
    def __init__(self):
        self.calls = []
        self.card = copy.deepcopy(CARD)
    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        return ProviderResponse(final_text=json.dumps(self.card))


class FakeEmbeddings:
    supports_media = True
    def __init__(self, space='fixture-space'):
        self.space_id = space
        self.config = SimpleNamespace(dimensions=3, space_spec={'dimensions': 3, 'model': space})
        self.calls = []
        self.fail_images = False
    async def embed_documents(self, documents, **kwargs):
        self.calls.append(documents)
        if self.fail_images and documents[0].media:
            raise RuntimeError('Synthetic image embedding outage')
        vectors = []
        for document in documents:
            payload = document.text + ''.join(media.data_b64 for media in document.media) + self.space_id
            raw = np.asarray(list(hashlib.sha256(payload.encode()).digest()[:3]), dtype=np.float32)
            vectors.append(raw / np.linalg.norm(raw))
        return vectors


class StickerBuildWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.dsn = os.getenv('TEST_DATABASE_URL')
        if not self.dsn:
            self.skipTest('TEST_DATABASE_URL required')
        self.schema = 'sticker_' + uuid.uuid4().hex
        self.temp = tempfile.TemporaryDirectory(prefix='fixture-sticker-', dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.store = PostgresStore(self.dsn, schema=self.schema)
        await self.store.initialize()
        self.addAsyncCleanup(self.cleanup_store)
        self.catalog = StickerCatalogStore(self.store)
        await self.catalog.initialize()
        self.provider, self.embeddings = FakeProvider(), FakeEmbeddings()
        self.builder = CatalogBuilder(self.catalog, self.provider, self.embeddings)

    async def cleanup_store(self):
        await self.store.close()
        async with await AsyncConnection.connect(self.dsn, autocommit=True) as conn:
            await conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))

    def picture(self, name, color):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (20, 20), color).save(path)
        return 'sha256:' + content_hash(path)

    async def test_append_same_bytes_and_aliases_do_not_rebuy_completed_work(self):
        self.assertEqual((await self.catalog.load_snapshot()).assets, ())
        first_id = self.picture('pack/one.png', 'red')
        first = await self.builder.build(self.root)
        self.assertTrue(first.active)
        self.assertEqual(len(self.provider.calls), 1)
        before = await self.catalog.load_snapshot()
        self.picture('other/alias.png', 'red')
        new_id = self.picture('other/two.png', 'blue')
        second = await self.builder.build(self.root)
        self.assertEqual(len(self.provider.calls), 2)
        snapshot = await self.catalog.load_snapshot()
        self.assertEqual({a.asset_id for a in snapshot.assets}, {first_id, new_id})
        first_asset = await self.catalog.get_asset(first_id)
        self.assertEqual({a.pack for a in first_asset.aliases}, {'pack', 'other'})
        self.assertEqual(first_asset.provenance, before.assets[0].provenance)
        self.assertEqual((await self.catalog.load_snapshot(first.revision_id)).assets[0].aliases[0].path, 'pack/one.png')
        self.assertNotEqual(first.revision_id, second.revision_id)

    async def test_embedding_failure_keeps_active_and_resume_reuses_successful_annotation(self):
        self.picture('pack/one.png', 'red')
        first = await self.builder.build(self.root)
        second_id = self.picture('pack/two.png', 'blue')
        self.embeddings.fail_images = True
        failed = await self.builder.build(self.root)
        self.assertFalse(failed.active)
        self.assertEqual(await self.catalog.active_revision_id(), first.revision_id)
        self.assertIsNone(await self.catalog.get_asset(second_id))
        staged = await self.catalog.get_asset(second_id, failed.revision_id)
        self.assertEqual(staged.generated_card['caption'], 'Hello')
        self.assertIsNotNone(staged.reading_vectors)
        self.assertEqual(len(self.provider.calls), 2)
        with self.assertRaises(CatalogConflict):
            await self.catalog.activate(failed.revision_id)
        self.embeddings.fail_images = False
        resumed = CatalogBuilder(self.catalog, self.provider, self.embeddings,
                                 config=replace(BuildConfig(), concurrency=2, request_timeout_s=900))
        result = await resumed.build(self.root, resume=failed.revision_id)
        self.assertTrue(result.active)
        self.assertEqual(len(self.provider.calls), 2)
        self.assertEqual(len([call for call in self.embeddings.calls if not call[0].media]), 2)

    async def test_selected_regeneration_preserves_deliberate_corrections_and_other_assets(self):
        target = self.picture('pack/one.png', 'red')
        other = self.picture('other/two.png', 'blue')
        await self.builder.build(self.root)
        correction = {'card': {'caption': 'Correct visible caption',
                      'readings': [{'meaning': 'Affectionate greeting', 'context': 'A close friend arrives'}]},
                      'family_ids': ['supported-blue-bird'], 'style_tags': ['flat drawing']}
        before_calls = len(self.embeddings.calls)
        await self.builder.build(self.root, corrections={target: correction})
        self.assertEqual(len(self.provider.calls), 2)
        self.assertEqual(len(self.embeddings.calls), before_calls + 1)
        before_other = await self.catalog.get_asset(other)
        self.provider.card['caption'] = 'New model guess'
        await self.builder.build(self.root, regenerate_files=['pack/one.png'])
        self.assertEqual(len(self.provider.calls), 3)
        selected = await self.catalog.get_asset(target)
        self.assertEqual(selected.generated_card['caption'], 'New model guess')
        self.assertEqual(selected.card['caption'], 'Correct visible caption')
        self.assertEqual(selected.family_ids, ('supported-blue-bird',))
        self.assertEqual((await self.catalog.get_asset(other)).provenance, before_other.provenance)
        self.assertEqual(selected.card['compatibility']['harshness_level'], 0)

    async def test_space_change_reembeds_every_asset_without_reannotation(self):
        self.picture('pack/one.png', 'red')
        self.picture('pack/two.png', 'blue')
        await self.builder.build(self.root)
        other_embeddings = FakeEmbeddings('next-space')
        await CatalogBuilder(self.catalog, self.provider, other_embeddings).build(self.root)
        self.assertEqual(len(self.provider.calls), 2)
        self.assertEqual(len(other_embeddings.calls), 4)
        snapshot = await self.catalog.load_snapshot()
        self.assertEqual(snapshot.recipe['embedding_space_id'], 'next-space')
        self.assertTrue(all(a.provenance['embedding_space_id'] == 'next-space' for a in snapshot.assets))

    async def test_conflicting_publication_cannot_overwrite_newer_catalog(self):
        self.picture('pack/one.png', 'red')
        await self.builder.build(self.root)
        snapshot = await self.catalog.load_snapshot()
        staged = await self.catalog.begin_revision(source_root=str(self.root), recipe=snapshot.recipe)
        await self.builder.build(self.root)
        current = await self.catalog.active_revision_id()
        with self.assertRaises(CatalogConflict):
            await self.catalog.activate(staged)
        self.assertEqual(await self.catalog.active_revision_id(), current)

    async def test_reused_filename_keeps_old_identity_without_pointing_it_at_new_bytes(self):
        old_id = self.picture('pack/one.png', 'red')
        await self.builder.build(self.root)
        new_id = self.picture('pack/one.png', 'blue')
        await self.builder.build(self.root)
        self.assertEqual((await self.catalog.get_asset(old_id)).aliases, ())
        self.assertEqual((await self.catalog.get_asset(new_id)).aliases[0].path, 'pack/one.png')
        self.assertEqual(content_hash(self.root/'pack/one.png'), new_id.removeprefix('sha256:'))

    async def test_text_only_route_keeps_reading_search_without_dropping_media_silently(self):
        self.picture('pack/one.png', 'red')
        self.embeddings.supports_media = False
        await self.builder.build(self.root)
        asset = (await self.catalog.load_snapshot()).assets[0]
        self.assertIsNone(asset.image_vector)
        self.assertEqual(asset.reading_vectors.shape, (1, 3))
        self.assertTrue(all(not document.media for call in self.embeddings.calls for document in call))
        self.assertTrue(any(part.data_b64 for part in self.provider.calls[0]['messages'][0].parts))

    async def vector_bytes(self):
        async with self.store.pool.connection() as conn:
            row = await (await conn.execute('SELECT count(*) AS count, COALESCE(sum(octet_length(payload)),0) AS bytes FROM sticker_catalog_vectors')).fetchone()
        return row['count'], row['bytes']

    async def test_revision_history_shares_vector_bytes_and_only_adds_changed_channels(self):
        target = self.picture('pack/one.png', 'red')
        self.picture('pack/two.png', 'blue')
        first = await self.builder.build(self.root)
        # Identical reading text shares one vector, two images have distinct bytes.
        self.assertEqual(await self.vector_bytes(), (3, 36))
        self.picture('pack/three.png', 'green')
        await self.builder.build(self.root)
        self.assertEqual(await self.vector_bytes(), (4, 48))
        self.provider.card['caption'] = 'Revised caption; supported meaning unchanged'
        await self.builder.build(self.root, regenerate_ids=[target])
        self.assertEqual(await self.vector_bytes(), (4, 48))
        await self.builder.build(self.root, corrections={target: {'card': {'readings': [
            {'meaning': 'A different supported reading', 'context': 'A particular exchange'}]}}})
        self.assertEqual(await self.vector_bytes(), (5, 60))
        self.assertEqual(len((await self.catalog.load_snapshot(first.revision_id)).assets), 2)

    async def test_publication_rejects_missing_channel_even_when_item_marked_ready(self):
        target = self.picture('pack/one.png', 'red')
        first = await self.builder.build(self.root)
        snapshot = await self.catalog.load_snapshot()
        staged = await self.catalog.begin_revision(source_root=str(self.root), recipe=snapshot.recipe)
        asset = await self.catalog.get_asset(target, staged)
        await self.builder._save(staged, asset, image_vector=None, state='ready')
        with self.assertRaises(CatalogConflict):
            await self.catalog.activate(staged)
        self.assertEqual(await self.catalog.active_revision_id(), first.revision_id)

    async def test_inventory_uses_expected_parent_instead_of_overwriting_later_changes(self):
        self.picture('pack/one.png', 'red')
        first = await self.builder.build(self.root)
        stale = await self.catalog.load_snapshot()
        self.picture('pack/two.png', 'blue')
        second = await self.builder.build(self.root)
        with self.assertRaises(CatalogConflict):
            await self.catalog.begin_revision(source_root=str(self.root), recipe=stale.recipe,
                assets=list(stale.assets), dimensions=3, expected_parent=first.revision_id)
        self.assertEqual(await self.catalog.active_revision_id(), second.revision_id)

    async def test_build_can_checkpoint_with_an_operator_selected_single_connection_pool(self):
        self.picture('pack/one.png', 'red')
        narrow_store = PostgresStore(self.dsn, schema=self.schema,
            config=DatabaseConfig(pool_min_size=1, pool_max_size=1, pool_timeout_s=0.2))
        await narrow_store.initialize()
        try:
            catalog = StickerCatalogStore(narrow_store)
            result = await CatalogBuilder(catalog, self.provider, self.embeddings).build(self.root)
            self.assertTrue(result.active)
            self.assertEqual(len((await catalog.load_snapshot()).assets), 1)
        finally:
            await narrow_store.close()
