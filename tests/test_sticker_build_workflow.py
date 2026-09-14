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
from unittest.mock import AsyncMock, patch
import uuid

import numpy as np
from PIL import Image
from psycopg import AsyncConnection, sql

from tgchatbot.domain.models import ProviderResponse
from tgchatbot.config import ChatCompletionsConfig
from tgchatbot.providers.base import ProviderCapabilities
from tgchatbot.providers.chat_completions import ChatCompletionsProvider
from tgchatbot.storage.postgres_store import DatabaseConfig, PostgresStore
from tgchatbot.storage.sticker_catalog import StickerCatalogStore, CatalogConflict
from tgchatbot.storage.sticker_delivery import StickerDeliveryStore
from tgchatbot.stickers.build import CatalogBuilder, BuildConfig
from tgchatbot.stickers.catalog import StickerCatalog
from tgchatbot.stickers.media import MediaConfig, PreparedMedia, content_hash, prepare_media
from tgchatbot.stickers.plan import StickerRetrievalPlan
from tgchatbot.tools.base import ToolContext
from tgchatbot.tools.sticker_send import StickerQueryTool

CARD = {'caption': 'Hello', 'appearance': 'A round blue bird', 'action': 'A raised wing',
        'readings': [{'meaning': 'Greeting', 'context': 'Opening a friendly conversation'}],
        'uncertainty': '', 'compatibility': {'harshness_level': 0, 'intimacy_level': 0, 'meme_dependence_level': 0}}


class FakeProvider:
    capabilities = ProviderCapabilities(multimodal_input=True)
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

    async def test_annotation_without_provider_vision_fails_before_generation_and_keeps_active_catalog(self):
        first_id = self.picture('pack/one.png', 'red')
        first = await self.builder.build(self.root)
        added_id = self.picture('pack/two.png', 'blue')
        provider = ChatCompletionsProvider(ChatCompletionsConfig(name='custom', api_key='synthetic',
            base_url='https://fixture.invalid/', model='fixture-model', multimodal_input=False))
        await provider.aclose()
        provider.generate = AsyncMock(return_value=ProviderResponse(final_text=json.dumps(CARD)))
        before_embeddings = len(self.embeddings.calls)
        result = await CatalogBuilder(self.catalog, provider, self.embeddings,
            config=replace(BuildConfig(), provider='custom', model='fixture-model', service_tier='')).build(self.root)
        self.assertFalse(result.active)
        self.assertEqual(await self.catalog.active_revision_id(), first.revision_id)
        provider.generate.assert_not_awaited()
        self.assertEqual(len(self.embeddings.calls), before_embeddings)
        self.assertEqual([failure['asset_id'] for failure in result.failed], [added_id])
        self.assertIn('image input', result.failed[0]['error'])
        failed = await self.catalog.get_asset(added_id, result.revision_id)
        self.assertEqual(failed.state, 'failed')
        self.assertIsNone(failed.generated_card)
        self.assertIsNotNone(await self.catalog.get_asset(first_id))

    async def test_empty_prepared_frames_fail_without_calling_generation_or_embedding(self):
        asset_id = self.picture('pack/one.png', 'red')
        empty = PreparedMedia(asset_id.removeprefix('sha256:'), {'supplied_frames': 0}, ())
        with patch('tgchatbot.stickers.build.prepare_media', return_value=empty):
            result = await self.builder.build(self.root)
        self.assertFalse(result.active)
        self.assertIsNone(await self.catalog.active_revision_id())
        self.assertIn('usable image frames', result.failed[0]['error'])
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(self.embeddings.calls, [])
        self.assertIsNone((await self.catalog.get_asset(asset_id, result.revision_id)).generated_card)

    async def test_invalid_source_fails_without_replacing_active_catalog_or_buying_annotation(self):
        self.picture('pack/one.png', 'red')
        first = await self.builder.build(self.root)
        bad = self.root / 'pack' / 'broken.png'
        original = b'This file is not a decodable image.'
        bad.write_bytes(original)
        before_calls = len(self.provider.calls), len(self.embeddings.calls)
        result = await self.builder.build(self.root)
        self.assertFalse(result.active)
        self.assertEqual(await self.catalog.active_revision_id(), first.revision_id)
        self.assertEqual((len(self.provider.calls), len(self.embeddings.calls)), before_calls)
        self.assertEqual(bad.read_bytes(), original)
        self.assertEqual(len(result.failed), 1)
        self.assertIsNone((await self.catalog.get_asset(result.failed[0]['asset_id'], result.revision_id)).generated_card)

    async def test_existing_cards_reembed_with_text_only_embeddings_without_requiring_generation_vision(self):
        asset_id = self.picture('pack/one.png', 'red')
        await self.builder.build(self.root)
        self.provider.capabilities = ProviderCapabilities(multimodal_input=False)
        embeddings = FakeEmbeddings('text-only-next-space')
        embeddings.supports_media = False
        result = await CatalogBuilder(self.catalog, self.provider, embeddings).build(self.root)
        self.assertTrue(result.active)
        self.assertEqual(len(self.provider.calls), 1, 'Reusing an existing visual card does not annotate again')
        current = await self.catalog.get_asset(asset_id)
        self.assertEqual(current.generated_card, CARD)
        self.assertEqual(current.provenance['visual_embedding_source'], 'description')
        self.assertTrue(all(not document.media for batch in embeddings.calls for document in batch))

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

    async def test_explicit_pruning_keeps_remaining_copies_and_historical_evidence_without_rebuying_work(self):
        source = self.root / 'originals'
        retained_id = self.picture('originals/pack/one.png', 'red')
        self.picture('originals/pack/copy.png', 'red')
        removed_id = self.picture('originals/pack/two.png', 'blue')
        first = await self.builder.build(source)
        before = await self.catalog.load_snapshot()
        calls = len(self.provider.calls), len(self.embeddings.calls)
        (source / 'pack/one.png').rename(self.root / 'removed-copy.png')
        (source / 'pack/two.png').rename(self.root / 'removed-original.png')

        await self.builder.build(source)
        self.assertIsNotNone(await self.catalog.get_asset(removed_id), 'Default builds retain missing evidence')
        pruned = await self.builder.build(source, prune_missing=True)
        self.assertTrue(pruned.active)
        self.assertEqual({a.asset_id for a in (await self.catalog.load_snapshot()).assets}, {retained_id})
        retained = await self.catalog.get_asset(retained_id)
        self.assertEqual([a.path for a in retained.aliases], ['pack/copy.png'])
        original = next(a for a in before.assets if a.asset_id == retained_id)
        self.assertEqual(retained.card, original.card)
        np.testing.assert_array_equal(retained.image_vector, original.image_vector)
        np.testing.assert_array_equal(retained.reading_vectors, original.reading_vectors)
        historical = await self.catalog.load_snapshot(first.revision_id)
        self.assertEqual({a.asset_id for a in historical.assets}, {retained_id, removed_id})
        self.assertEqual((len(self.provider.calls), len(self.embeddings.calls)), calls)

    async def test_explicit_pruning_can_publish_an_empty_catalog_without_erasing_history(self):
        source = self.root / 'originals'
        asset_id = self.picture('originals/pack/one.png', 'red')
        first = await self.builder.build(source)
        (source / 'pack').rename(self.root / 'removed-pack')
        result = await self.builder.build(source, prune_missing=True)
        self.assertTrue(result.active)
        self.assertEqual((await self.catalog.load_snapshot()).assets, ())
        self.assertIsNotNone(await self.catalog.get_asset(asset_id, first.revision_id))
        self.assertEqual(len(self.provider.calls), 1)

    async def test_imported_catalog_is_eligible_until_the_agent_requests_exclusions(self):
        calm_id = self.picture('pack/calm.png', 'red')
        intense_id = self.picture('pack/intense.png', 'blue')
        animation = self.root / 'pack/animated.gif'
        frames = [Image.new('RGB', (20, 20), color) for color in ('green', 'yellow')]
        try:
            frames[0].save(animation, save_all=True, append_images=frames[1:], duration=100, loop=0)
        finally:
            for frame in frames:
                frame.close()
        animated_id = 'sha256:' + content_hash(animation)
        built = await self.builder.build(self.root, corrections={intense_id: {'card': {'compatibility': {
            'harshness_level': 4, 'intimacy_level': 4, 'meme_dependence_level': 4}}}})
        self.assertTrue(built.active)
        self.assertTrue((await self.catalog.get_asset(animated_id)).media['animated'])
        calls = len(self.provider.calls), len(self.embeddings.calls)
        self.embeddings.enabled = True
        self.embeddings.config.space_id = self.embeddings.space_id
        self.embeddings.embed_query = AsyncMock(return_value=(await self.catalog.get_asset(calm_id)).reading_vectors[0])
        runtime = StickerCatalog(self.catalog, self.root, embedding_client=self.embeddings)
        tool = StickerQueryTool(runtime)
        context = ToolContext('fixture-chat', 'Participant')
        all_ids = {calm_id, intense_id, animated_id}
        direct = await runtime.achoose(plan=StickerRetrievalPlan(intent_core='A greeting', candidate_budget=3))
        self.assertEqual({match.entry.sticker_id for match in direct}, all_ids)
        nullable = {'max_harshness': None, 'max_intimacy': None,
                    'max_meme_dependence': None, 'allow_animation': None}
        cases = [
            ({}, all_ids),
            ({**nullable, 'advanced': {'intensity_limits': nullable}}, all_ids),
            ({'allow_animation': False}, {calm_id, intense_id}),
            ({'max_harshness': 3}, {calm_id, animated_id}),
            ({'advanced': {'intensity_limits': {'max_intimacy': 0}}}, {calm_id, animated_id}),
            ({'advanced': {'intensity_limits': {'max_meme_dependence': 0}}}, {calm_id, animated_id}),
            ({'allow_animation': False, 'max_harshness': 0,
              'advanced': {'intensity_limits': nullable}}, {calm_id}),
            ({'allow_animation': True, 'max_harshness': 4,
              'advanced': {'intensity_limits': {'allow_animation': False, 'max_harshness': 0}}}, {calm_id}),
            ({'intensity_limits': {'allow_animation': False, 'max_harshness': 0},
              'advanced': {'intensity_limits': nullable}}, {calm_id}),
            ({'safety_limits': {'allow_animation': False, 'max_harshness': 0},
              'advanced': {'intensity_limits': {'max_intimacy': 4}}}, {calm_id}),
            ({'advanced': {'intensity_limits': nullable,
                          'safety_limits': {'allow_animation': False, 'max_harshness': 0}}}, {calm_id}),
            ({'intensity_limits': {'allow_animation': False, 'max_harshness': 0},
              'advanced': {'intensity_limits': {'allow_animation': True, 'max_harshness': 4}}}, all_ids),
            ({'intensity_limits': {'allow_animation': False, 'max_harshness': 0},
              'advanced': {'intensity_limits': {'allow_animation': True}}}, {calm_id, animated_id}),
            ({'intensity_limits': {'allow_animation': False, 'max_harshness': 0},
              'advanced': {'intensity_limits': {'max_harshness': 4}}}, {calm_id, intense_id}),
        ]
        for controls, expected in cases:
            with self.subTest(controls=controls):
                result = await tool.run({'intent_core': 'A greeting', 'candidate_budget': 3, **controls}, context)
                self.assertTrue(result.output['ok'])
                self.assertEqual({item['sticker_id'] for item in result.output['candidates']}, expected)
                self.assertEqual(result.stickers, [])
        self.assertEqual((len(self.provider.calls), len(self.embeddings.calls)), calls)

    async def test_folder_aggregation_refreshes_delivery_continuity_without_rebuying_vectors(self):
        first_id = self.picture('birdPack/one.png', 'red')
        second_id = self.picture('birdPackV2/two.png', 'blue')
        standalone_id = self.picture('独立作品_standalonePack/three.png', 'green')
        first = await self.builder.build(self.root)
        before = await self.catalog.get_asset(first_id)
        vector_storage = await self.vector_bytes()
        calls = len(self.provider.calls), len(self.embeddings.calls)
        deliveries = StickerDeliveryStore(self.store)
        await deliveries.initialize()
        scope = await self.store.get_scope('chat')
        await deliveries.queue('chat', first_id, operation_id='sent-before-reorganization',
            expected_scope=scope, timing='send_now')
        await deliveries.begin('sent-before-reorganization')
        await deliveries.finish('sent-before-reorganization', 'sent', telegram_message_id=1)
        self.embeddings.enabled = True
        self.embeddings.config.space_id = self.embeddings.space_id
        self.embeddings.embed_query = AsyncMock(return_value=before.image_vector)
        runtime = StickerCatalog(self.catalog, self.root, delivery_store=deliveries,
            embedding_client=self.embeddings)
        plan = StickerRetrievalPlan.from_payload({'intent_core': 'A friendly greeting'})
        await runtime.achoose(plan=plan, session_id='chat')
        self.assertEqual((await runtime.adescribe_style_context('chat'))['recent_source_pack_ids'], ['birdPack'])

        group = self.root / '蓝鸟系列'
        group.mkdir()
        (self.root / 'birdPack').rename(group / '蓝鸟_birdPack')
        (self.root / 'birdPackV2').rename(group / '蓝鸟第二弹_birdPackV2')
        result = await self.builder.build(self.root)

        self.assertTrue(result.active)
        self.assertEqual(result.completed, 0)
        self.assertEqual((len(self.provider.calls), len(self.embeddings.calls)), calls)
        self.assertEqual(await self.vector_bytes(), vector_storage)
        current = await self.catalog.get_asset(first_id)
        self.assertEqual([(alias.path, alias.pack) for alias in current.aliases],
            [('蓝鸟系列/蓝鸟_birdPack/one.png', '蓝鸟系列')])
        self.assertEqual(current.generated_card, before.generated_card)
        self.assertEqual(current.provenance, before.provenance)
        np.testing.assert_array_equal(current.image_vector, before.image_vector)
        np.testing.assert_array_equal(current.reading_vectors, before.reading_vectors)
        self.assertEqual((await self.catalog.get_asset(first_id, first.revision_id)).aliases, before.aliases)
        self.assertEqual((await self.catalog.get_asset(standalone_id)).aliases[0].pack, '独立作品_standalonePack')

        matches = await runtime.achoose(plan=plan, session_id='chat')
        preferred = {match.entry.sticker_id for match in matches
            if 'preferred_family_or_pack' in match.channels}
        self.assertEqual(preferred, {first_id, second_id})
        context = await runtime.adescribe_style_context('chat')
        self.assertEqual(context['recent_source_pack_ids'], ['蓝鸟系列'])
        restarted = StickerCatalog(self.catalog, self.root, delivery_store=StickerDeliveryStore(self.store))
        await restarted.aensure_loaded()
        self.assertEqual(await restarted.adescribe_style_context('chat'), context)
        required = StickerRetrievalPlan.from_payload({'intent_core': 'Hello', 'required_pack': '蓝鸟系列'})
        self.assertEqual({match.entry.sticker_id for match in await restarted.achoose(plan=required)},
            {first_id, second_id})

    async def test_reorganization_keeps_real_copies_and_completely_missing_original_evidence(self):
        source = self.root / 'source'
        target = self.picture('source/pack/one.png', 'red')
        self.picture('source/other/alias.png', 'red')
        missing_id = self.picture('source/missing/two.png', 'blue')
        first = await self.builder.build(source)
        old_target = await self.catalog.get_asset(target)
        old_missing = await self.catalog.get_asset(missing_id)
        calls = len(self.provider.calls), len(self.embeddings.calls)
        group = source / '鸟系列'
        group.mkdir()
        (source / 'pack').rename(group / '小鸟_pack')
        (source / 'missing').rename(self.root / 'unavailable')

        result = await self.builder.build(source)

        self.assertTrue(result.active)
        self.assertEqual((len(self.provider.calls), len(self.embeddings.calls)), calls)
        self.assertEqual({(alias.path, alias.pack) for alias in (await self.catalog.get_asset(target)).aliases},
            {('鸟系列/小鸟_pack/one.png', '鸟系列'), ('other/alias.png', 'other')})
        self.assertEqual((await self.catalog.get_asset(target, first.revision_id)).aliases, old_target.aliases)
        missing = await self.catalog.get_asset(missing_id)
        self.assertEqual(missing.aliases, old_missing.aliases)
        self.assertEqual(missing.generated_card, old_missing.generated_card)
        np.testing.assert_array_equal(missing.image_vector, old_missing.image_vector)
        runtime = StickerCatalog(self.catalog, source)
        self.assertIsNone(await runtime.aget_available(missing_id))
        self.assertIsNotNone(await runtime.aget_available(target))

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
                                 config=replace(BuildConfig(), concurrency=2, request_timeout_s=900,
                                                max_output_tokens=16384))
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

    async def test_frame_allowance_change_reuses_static_images_and_preserves_reviewed_cards(self):
        static_id = self.picture('pack/static.png', 'red')
        animated_path = self.root / 'pack' / 'animated.webp'
        frames = [Image.new('RGB', (20, 20), color)
                  for color in ('red', 'green', 'blue', 'yellow', 'white')]
        try:
            frames[0].save(animated_path, save_all=True, append_images=frames[1:],
                           duration=100, loop=0, lossless=True)
        finally:
            for frame in frames:
                frame.close()
        animated_id = 'sha256:' + content_hash(animated_path)
        corrections = {identity: {'card': {'caption': 'Reviewed caption', 'readings': [
            {'meaning': 'Reviewed greeting', 'context': 'A familiar friend arrives'}]},
            'style_tags': ['reviewed drawing']} for identity in (static_id, animated_id)}
        first_builder = CatalogBuilder(self.catalog, self.provider, self.embeddings,
                                       media_config=MediaConfig(max_frames=3))
        await first_builder.build(self.root, corrections=corrections)
        before = {identity: await self.catalog.get_asset(identity) for identity in corrections}
        annotation_calls, embedding_calls = len(self.provider.calls), len(self.embeddings.calls)

        larger = CatalogBuilder(self.catalog, self.provider, self.embeddings,
                                media_config=MediaConfig(max_frames=5))
        result = await larger.build(self.root)

        self.assertTrue(result.active)
        self.assertEqual(result.completed, 1, 'Only animated visual evidence changes')
        self.assertEqual(len(self.provider.calls), annotation_calls)
        new_calls = self.embeddings.calls[embedding_calls:]
        self.assertEqual(len(new_calls), 1)
        self.assertEqual(new_calls[0][0].item_id, animated_id + ':image')
        self.assertEqual(len(new_calls[0][0].media), 5)
        for identity in corrections:
            current = await self.catalog.get_asset(identity)
            for field in ('aliases', 'generated_card', 'corrections', 'card'):
                self.assertEqual(getattr(current, field), getattr(before[identity], field), field)
            np.testing.assert_array_equal(current.reading_vectors, before[identity].reading_vectors)
        static = await self.catalog.get_asset(static_id)
        self.assertEqual(static.media, before[static_id].media)
        self.assertEqual(static.provenance, before[static_id].provenance)
        np.testing.assert_array_equal(static.image_vector, before[static_id].image_vector)
        animated = await self.catalog.get_asset(animated_id)
        self.assertEqual(animated.media['supplied_frames'], 5)
        self.assertFalse(np.array_equal(animated.image_vector, before[animated_id].image_vector))

        calls = len(self.provider.calls), len(self.embeddings.calls)
        self.assertEqual((await larger.build(self.root)).completed, 0)
        self.assertEqual((len(self.provider.calls), len(self.embeddings.calls)), calls)
        resized = CatalogBuilder(self.catalog, self.provider, self.embeddings,
                                 media_config=MediaConfig(max_frames=5, max_dimension=10))
        self.assertEqual((await resized.build(self.root)).completed, 2)
        self.assertEqual(len(self.provider.calls), calls[0])
        changed_images = self.embeddings.calls[calls[1]:]
        self.assertEqual({batch[0].item_id for batch in changed_images},
                         {static_id + ':image', animated_id + ':image'})
        self.assertTrue(all(document.media for batch in changed_images for document in batch))

    async def test_selected_regeneration_refreshes_changed_samples_and_resumes_image_failure(self):
        path = self.root / 'pack' / 'animated.webp'
        path.parent.mkdir()
        frames = [Image.new('RGB', (20, 20), color) for color in ('red', 'green', 'blue')]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=[50, 100, 50], lossless=True)
        target = 'sha256:' + content_hash(path)
        other = self.picture('other/two.png', 'yellow')
        prepared = prepare_media(path, self.builder.media_config)
        # The original and preparation settings stay identical across a decoder
        # fix, while the selected pixels and observed timing change.
        stale = replace(prepared, frames=prepared.frames[1:], facts={**prepared.facts,
            'supplied_frames': len(prepared.frames) - 1,
            'frame_times_s': [frame.timestamp_s for frame in prepared.frames[1:]],
            'duration_s': prepared.facts['duration_s'] - .05})
        def old_preparation(source, config):
            return stale if source == path else prepare_media(source, config)
        correction = {'card': {'caption': 'Deliberately corrected caption'}}
        with patch('tgchatbot.stickers.build.prepare_media', side_effect=old_preparation):
            baseline = await self.builder.build(self.root, corrections={target: correction})
        before_target = await self.catalog.get_asset(target)
        before_other = await self.catalog.get_asset(other)
        self.provider.card['action'] = 'The newly visible opening gesture'
        self.embeddings.fail_images = True

        failed = await self.builder.build(self.root, regenerate_ids=[target])

        self.assertFalse(failed.active, 'A stale image vector must not bypass the refreshed image request')
        self.assertEqual([failure['asset_id'] for failure in failed.failed], [target])
        self.assertEqual(await self.catalog.active_revision_id(), baseline.revision_id)
        staged = await self.catalog.get_asset(target, failed.revision_id)
        self.assertEqual(staged.media, prepared.facts)
        self.assertIsNone(staged.image_vector)
        self.assertIsNotNone(staged.reading_vectors)
        self.assertEqual(staged.generated_card['action'], self.provider.card['action'])
        self.assertEqual(staged.corrections, correction)
        self.assertEqual(staged.card['caption'], correction['card']['caption'])
        generated_images = [part.data_b64 for part in self.provider.calls[-1]['messages'][0].parts
                            if part.data_b64]
        self.assertEqual(generated_images, [frame.data_b64 for frame in prepared.frames])
        failed_image = self.embeddings.calls[-1][0]
        self.assertEqual(failed_image.item_id, target + ':image')
        self.assertEqual([media.data_b64 for media in failed_image.media], generated_images)
        annotation_calls = len(self.provider.calls)
        embedding_calls = len(self.embeddings.calls)
        self.embeddings.fail_images = False

        resumed = await self.builder.build(self.root, resume=failed.revision_id)

        self.assertTrue(resumed.active)
        self.assertEqual(len(self.provider.calls), annotation_calls, 'Keep the successful annotation on retry')
        self.assertEqual(len(self.embeddings.calls), embedding_calls + 1, 'Only the failed image request retries')
        self.assertEqual(self.embeddings.calls[-1][0], failed_image)
        current = await self.catalog.get_asset(target)
        self.assertEqual(current.media, prepared.facts)
        self.assertEqual(current.generated_card, staged.generated_card)
        np.testing.assert_array_equal(current.reading_vectors, staged.reading_vectors)
        self.assertFalse(np.array_equal(current.image_vector, before_target.image_vector))
        expected_image = (await FakeEmbeddings().embed_documents([failed_image]))[0]
        np.testing.assert_array_equal(current.image_vector, expected_image)
        unchanged = await self.catalog.get_asset(other)
        for field in ('aliases', 'media', 'generated_card', 'corrections', 'card', 'provenance', 'state', 'error'):
            self.assertEqual(getattr(unchanged, field), getattr(before_other, field), field)
        np.testing.assert_array_equal(unchanged.image_vector, before_other.image_vector)
        np.testing.assert_array_equal(unchanged.reading_vectors, before_other.reading_vectors)
        self.assertEqual(content_hash(path), target.removeprefix('sha256:'))

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

    async def test_text_only_route_uses_explicit_description_fallback_and_keeps_real_evidence(self):
        from tgchatbot.stickers.catalog import StickerCatalog
        from tgchatbot.stickers.plan import StickerRetrievalPlan
        from tgchatbot.domain.models import PartKind
        from tgchatbot.tools.sticker_send import _candidate_payload
        from unittest.mock import AsyncMock
        target = self.picture('pack/one.png', 'red')
        self.embeddings.supports_media = False
        await self.builder.build(self.root)
        asset = (await self.catalog.load_snapshot()).assets[0]
        self.assertIsNotNone(asset.image_vector)
        self.assertEqual(asset.provenance['visual_embedding_source'], 'description')
        self.assertEqual(asset.reading_vectors.shape, (1, 3))
        self.assertTrue(all(not document.media for call in self.embeddings.calls for document in call))
        self.assertTrue(any(part.data_b64 for part in self.provider.calls[0]['messages'][0].parts))
        self.assertIn(CARD['appearance'], self.embeddings.calls[-1][0].text)
        self.embeddings.enabled = True
        self.embeddings.config.space_id = self.embeddings.space_id
        self.embeddings.embed_query = AsyncMock(return_value=asset.image_vector)
        catalog = StickerCatalog(self.catalog, self.root, embedding_client=self.embeddings)
        plan = StickerRetrievalPlan.from_payload({'intent_core': 'A friendly greeting',
            'persona_mode': 'use_once', 'persona': {'visual_identity': {'rendering_style': 'simple drawing'}}})
        matches = await catalog.achoose(plan=plan)
        self.assertEqual(matches[0].entry.sticker_id, target)
        self.assertEqual(_candidate_payload(matches[0])['appearance_embedding_source'], 'description')
        self.assertTrue(any(part.kind == PartKind.IMAGE for part in await catalog.evidence(matches)))

        calls = len(self.embeddings.calls)
        await self.builder.build(self.root, corrections={target: {'card': {'appearance': 'A vivid blue bird'}}})
        self.assertEqual(len(self.embeddings.calls), calls + 1)
        self.assertEqual(len(self.provider.calls), 1)
        self.assertIn('A vivid blue bird', self.embeddings.calls[-1][0].text)

    async def test_disabled_visual_channel_and_blank_description_need_no_fallback_vector(self):
        self.picture('pack/one.png', 'red')
        self.embeddings.supports_media = False
        self.provider.card.update(appearance='\t', action='\n', caption='')
        await self.builder.build(self.root)
        self.assertIsNone((await self.catalog.load_snapshot()).assets[0].image_vector)
        disabled = CatalogBuilder(self.catalog, self.provider, self.embeddings,
                                  config=replace(BuildConfig(), image_embeddings=False))
        await disabled.build(self.root)
        self.assertIsNone((await self.catalog.load_snapshot()).assets[0].image_vector)

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

    async def test_obsolete_resume_rejects_before_purchasing_more_work(self):
        target = self.picture('pack/one.png', 'red')
        self.embeddings.fail_images = True
        failed = await self.builder.build(self.root)
        self.assertFalse(failed.active)
        self.embeddings.fail_images = False
        current = await self.builder.build(self.root)
        calls = len(self.embeddings.calls)
        with self.assertRaises(CatalogConflict):
            await self.builder.build(self.root, resume=failed.revision_id)
        self.assertEqual(len(self.embeddings.calls), calls)
        self.assertEqual(await self.catalog.active_revision_id(), current.revision_id)
        self.assertEqual((await self.catalog.get_asset(target)).state, 'ready')

    async def test_invalid_corrections_fail_before_annotation_and_staging(self):
        target = self.picture('pack/one.png', 'red')
        for card in ({'caption': []}, {'compatibility': {'harshness_level': 9}},
                     {'unrecognized_caption_field': 'hello'}, {'readings': ['not a reading']}):
            with self.subTest(card=card), self.assertRaises(ValueError):
                await self.builder.build(self.root, corrections={target: {'card': card}})
        self.assertEqual(self.provider.calls, [])
        self.assertEqual(self.embeddings.calls, [])
        async with self.store.pool.connection() as conn:
            row = await (await conn.execute('SELECT count(*) AS total FROM sticker_catalog_revisions')).fetchone()
        self.assertEqual(row['total'], 0)

    async def test_mixed_formats_report_unsupported_files_without_rejecting_valid_assets(self):
        target = self.picture('pack/one.png', 'red')
        (self.root / 'pack/animated.tgs').write_bytes(b'synthetic unsupported vector sticker')
        result = await self.builder.build(self.root)
        self.assertTrue(result.active)
        self.assertEqual(result.unsupported_files, ('pack/animated.tgs',))
        self.assertEqual({asset.asset_id for asset in (await self.catalog.load_snapshot()).assets}, {target})
        self.assertEqual(len(self.provider.calls), 1)

    async def test_unsupported_only_selection_does_not_append_unrelated_new_assets(self):
        self.picture('existing/one.png', 'red')
        current = await self.builder.build(self.root)
        self.picture('new/two.png', 'blue')
        (self.root / 'vectors').mkdir()
        (self.root / 'vectors/animated.tgs').write_bytes(b'synthetic unsupported vector sticker')
        for selection in ({'regenerate_files': ['vectors/animated.tgs']}, {'regenerate_packs': ['vectors']}):
            with self.subTest(selection=selection), self.assertRaisesRegex(ValueError, 'No processable'):
                await self.builder.build(self.root, **selection)
        self.assertEqual(await self.catalog.active_revision_id(), current.revision_id)
        self.assertEqual(len(self.provider.calls), 1)

    async def test_unsupported_only_empty_install_does_not_claim_catalog_activation(self):
        (self.root / 'animated.tgs').write_bytes(b'synthetic unsupported vector sticker')
        result = await self.builder.build(self.root)
        self.assertFalse(result.active)
        self.assertIsNone(result.revision_id)
        self.assertEqual(result.unsupported_files, ('animated.tgs',))
        self.assertEqual(len(self.provider.calls), 0)
