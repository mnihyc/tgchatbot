"""Conversation choices and media evidence, with real image files and fake APIs."""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

import numpy as np
from PIL import Image

from tgchatbot.domain.models import PartKind, StickerTiming
from tgchatbot.stickers.catalog import StickerCatalog
from tgchatbot.stickers.config import StickerConfig
from tgchatbot.stickers.plan import StickerRetrievalPlan
from tgchatbot.storage.sticker_catalog import CatalogAlias, CatalogAsset, CatalogSnapshot
from tgchatbot.tools.base import ToolContext
from tgchatbot.tools.sticker_send import StickerQueryTool, StickerSendSelectedTool


class StickerConversationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.assets = []
        self.store = Mock(active_revision_id=AsyncMock(return_value='r1'))
        self.store.load_snapshot = AsyncMock(side_effect=lambda _: CatalogSnapshot('r1', {'embedding_space_id': 'space'}, '', tuple(self.assets)))
        self.personas = Mock(get_sticker_persona=AsyncMock(return_value=None),
            save_sticker_persona=AsyncMock(), clear_sticker_persona=AsyncMock())
        self.deliveries = Mock(recent=AsyncMock(return_value=[]))
        self.embeddings = Mock(enabled=True, config=SimpleNamespace(space_id='space'),
            embed_query=AsyncMock(return_value=np.array([1., 0., 0.], dtype=np.float32)))
        self.catalog = StickerCatalog(self.store, self.root, persona_store=self.personas,
            embedding_client=self.embeddings, delivery_store=self.deliveries)
        self.ctx = ToolContext('chat-a', 'Participant')

    def asset(self, *, caption='', action='Offering a hug', readings=None, reading_vectors=None,
              image_vector=None, pack='pack-a', animated=False, harshness=0, family=None):
        position = len(self.assets)
        path = self.root / pack / f'{position}.png'
        path.parent.mkdir(exist_ok=True)
        Image.new('RGB', (16, 16), (position * 29 % 256, 120, 200)).save(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        readings = readings if readings is not None else [{'meaning': 'Offer care', 'context': 'Someone is tired'}]
        card = {'caption': caption, 'appearance': 'A simple drawn character', 'action': action,
            'readings': readings, 'uncertainty': '', 'compatibility': {'harshness_level': harshness,
                'intimacy_level': 0, 'meme_dependence_level': 0}}
        vectors = reading_vectors if reading_vectors is not None else [[1, 0, 0] for _ in readings]
        asset = CatalogAsset('sha256:' + digest, digest, (CatalogAlias(path.relative_to(self.root).as_posix(), pack),),
            {'animated': animated}, card, {'family_ids': family or []}, card, {},
            np.array(image_vector, dtype=np.float32) if image_vector is not None else None,
            np.array(vectors, dtype=np.float32))
        self.assets.append(asset)
        return asset

    async def query(self, **kwargs):
        return await StickerQueryTool(self.catalog).run({'intent_core': 'Offer a quiet hug after work', **kwargs}, self.ctx)

    async def test_each_reading_can_retrieve_one_asset_without_duplicate_votes(self):
        multi = self.asset(readings=[{'meaning': 'Ask for care', 'context': 'Sender seeks comfort'},
            {'meaning': 'Offer care', 'context': 'Recipient is tired'}], reading_vectors=[[0, 1, 0], [1, 0, 0]])
        other = self.asset(action='A small reassuring nod', reading_vectors=[[.9, .43589, 0]])
        result = await self.query(candidate_budget=2)
        ids = [c['sticker_id'] for c in result.output['candidates']]
        self.assertEqual(ids, [multi.asset_id, other.asset_id])
        self.assertEqual(result.output['candidates'][0]['matched_reading']['meaning'], 'Offer care')
        self.assertEqual(len(set(ids)), 2)

    async def test_query_returns_actual_attributed_images_and_never_sends(self):
        asset = self.asset(caption='抱抱', action='Offering a hug')
        result = await self.query()
        self.assertEqual(result.stickers, [])
        images = [part for part in result.evidence_parts if part.kind == PartKind.IMAGE]
        self.assertTrue(images)
        self.assertEqual(images[0].origin, 'sticker_candidate:' + asset.asset_id)
        self.assertTrue(images[0].data_b64)
        self.assertNotIn('path', result.output['candidates'][0])
        self.assertNotIn('fit_signals', result.output['candidates'][0])
        self.assertEqual(self.deliveries.recent.await_count, 1)

    async def test_model_interpretation_never_echoes_requested_warmth_as_candidate_fact(self):
        self.asset(caption='Go away', action='Pushing the recipient away',
            readings=[{'meaning': 'Dismiss someone', 'context': 'Sender wants distance'}])
        result = await self.query(reaction_tone='warm and loving')
        candidate = result.output['candidates'][0]
        self.assertEqual(candidate['caption'], 'Go away')
        self.assertEqual(candidate['readings'][0]['meaning'], 'Dismiss someone')
        self.assertNotIn('warm', str(candidate))

    async def test_excluded_animation_harshness_and_missing_media_do_not_consume_slots(self):
        self.asset(animated=True)
        self.asset(harshness=4)
        missing = self.asset()
        (self.root / missing.aliases[0].path).rename(self.root / 'unavailable.png')
        valid = self.asset(reading_vectors=[[.8, .6, 0]], harshness=0)
        result = await self.query(candidate_budget=1, advanced={'intensity_limits': {'max_harshness': 0}})
        self.assertEqual([x['sticker_id'] for x in result.output['candidates']], [valid.asset_id])

    async def test_pack_preference_keeps_global_choices_but_explicit_requirement_filters(self):
        global_asset = self.asset(pack='global', reading_vectors=[[1, 0, 0]])
        preferred = self.asset(pack='familiar', reading_vectors=[[.7, .71414, 0]])
        result = await self.query(preferred_pack='familiar', candidate_budget=2)
        self.assertEqual({x['sticker_id'] for x in result.output['candidates']}, {global_asset.asset_id, preferred.asset_id})
        required = await self.query(required_pack='familiar')
        self.assertEqual([x['sticker_id'] for x in required.output['candidates']], [preferred.asset_id])
        absent = await self.query(required_pack='absent')
        self.assertEqual(absent.output['status'], 'no_candidates')
        self.assertNotIn('does not exist', str(absent.output))

    async def test_known_family_can_span_packs_and_pack_is_not_identity(self):
        a = self.asset(pack='first', family=['round-cat'])
        b = self.asset(pack='second', family=['round-cat'])
        self.asset(pack='first', family=['dog'])
        result = await self.query(required_character_family='round-cat')
        self.assertEqual({x['sticker_id'] for x in result.output['candidates']}, {a.asset_id, b.asset_id})

    async def test_confirmed_repeat_stays_eligible_and_fresh_variant_is_offered(self):
        prior = self.asset(reading_vectors=[[1, 0, 0]])
        fresh = self.asset(reading_vectors=[[.99, .14106, 0]])
        self.deliveries.recent.return_value = [{'sticker_id': prior.asset_id}]
        result = await self.query(diversity_preference='prefer_fresh_variant', candidate_budget=2)
        candidates = result.output['candidates']
        by_id = {candidate['sticker_id']: candidate for candidate in candidates}
        self.assertEqual(set(by_id), {fresh.asset_id, prior.asset_id})
        self.assertFalse(by_id[fresh.asset_id]['recently_delivered'])
        self.assertTrue(by_id[prior.asset_id]['recently_delivered'])
        self.personas.save_sticker_persona.assert_not_awaited()

    async def test_negative_meanings_are_constraints_not_positive_embedding_text(self):
        self.asset()
        result = await self.query(advanced={'text_constraints': {'avoid_text_meanings': ['violent revenge']},
            'forbid': ['sexual threat']}, selection_lens={'avoid_misread_as': 'attention seeking'})
        sent_queries = ' '.join(call.args[0] for call in self.embeddings.embed_query.await_args_list)
        self.assertNotIn('violent revenge', sent_queries)
        self.assertNotIn('attention seeking', sent_queries)
        self.assertEqual(result.output['constraints']['avoid_text_meanings'], ['violent revenge'])

    async def test_skip_does_no_retrieval_or_persona_mutation(self):
        result = await self.query(send=False, persona_mode='clear_session_persona')
        self.assertTrue(result.output['skipped'])
        self.store.active_revision_id.assert_not_awaited()
        self.embeddings.embed_query.assert_not_awaited()
        self.personas.get_sticker_persona.assert_not_awaited()
        self.personas.clear_sticker_persona.assert_not_awaited()

    async def test_persona_remember_use_once_clear_and_failed_write(self):
        self.asset()
        await self.query(persona={'visual_identity': {'character_archetype': 'cat'}})
        self.personas.save_sticker_persona.assert_awaited_once_with('chat-a', {'visual_identity': {'character_archetype': 'cat'}})
        once = await self.query(persona_mode='use_once', persona={'visual_identity': {'character_archetype': 'penguin'}})
        self.assertEqual(once.output['persona']['visual_identity']['character_archetype'], 'penguin')
        state = self.catalog.style_memory.get('chat-a')
        self.assertEqual(state.session_persona['visual_identity']['character_archetype'], 'cat')
        self.personas.clear_sticker_persona.side_effect = RuntimeError('fixture DB failure')
        failed = await self.query(persona_mode='clear_session_persona')
        self.assertFalse(failed.output['ok'])
        self.assertEqual(state.session_persona['visual_identity']['character_archetype'], 'cat')
        self.personas.clear_sticker_persona.side_effect = None
        await self.query(persona_mode='clear_session_persona')
        self.assertIsNone(state.session_persona)

    async def test_selection_is_queued_and_retains_timing_aliases_without_recording_delivery(self):
        asset = self.asset()
        await self.catalog.aensure_loaded()
        tool = StickerSendSelectedTool(self.catalog)
        result = await tool.run({'sticker_id': asset.asset_id, 'timing': 'before_final'}, self.ctx)
        self.assertEqual(result.output['status'], 'queued')
        self.assertEqual(result.stickers[0].timing, StickerTiming.SEND_NOW)
        self.assertEqual(result.stickers[0].content_sha256, asset.content_hash)
        after = await tool.run({'selected_sticker_id': asset.asset_id}, self.ctx)
        self.assertEqual(after.stickers[0].timing, StickerTiming.AFTER_FINAL)
        self.deliveries.recent.assert_not_awaited()

    async def test_reused_filename_cannot_send_different_original(self):
        asset = self.asset()
        await self.catalog.aensure_loaded()
        (self.root / asset.aliases[0].path).write_bytes(b'replaced media')
        result = await StickerSendSelectedTool(self.catalog).run({'selected_sticker_id': asset.asset_id}, self.ctx)
        self.assertFalse(result.output['ok'])
        self.assertEqual(result.stickers, [])

    async def test_available_alias_preserves_content_identity_after_pack_copy(self):
        asset = self.asset()
        source = self.root / asset.aliases[0].path
        copy = self.root / 'copy.png'
        copy.write_bytes(source.read_bytes())
        self.assets[0] = replace(asset, aliases=asset.aliases + (CatalogAlias('copy.png', 'second'),))
        source.rename(self.root / 'moved.png')
        result = await StickerSendSelectedTool(self.catalog).run({'selected_sticker_id': asset.asset_id}, self.ctx)
        self.assertTrue(result.output['ok'])
        self.assertEqual(result.stickers[0].path, copy)

    async def test_configured_candidate_count_can_exceed_old_cap(self):
        for _ in range(10):
            self.asset()
        self.catalog.config = StickerConfig(candidate_count=9, max_candidates=12)
        result = await self.query()
        self.assertEqual(len(result.output['candidates']), 9)
        self.assertEqual(StickerRetrievalPlan.from_payload({'intent_core': 'hello'}).candidate_budget, 5)

    async def test_empty_install_needs_no_sticker_assets_or_embedding_key(self):
        self.embeddings.enabled = False
        result = await self.query()
        self.assertEqual(result.output['status'], 'no_candidates')
        self.embeddings.embed_query.assert_not_awaited()
        self.assertFalse((self.root / 'sticker_index.sqlite3').exists())

    async def test_embedding_space_mismatch_is_explicit_and_known_id_remains_sendable(self):
        asset = self.asset()
        self.embeddings.config.space_id = 'different-model'
        result = await self.query()
        self.assertFalse(result.output['ok'])
        self.assertIn('embedding space changed', result.output['error'])
        sent = await StickerSendSelectedTool(self.catalog).run({'selected_sticker_id': asset.asset_id}, self.ctx)
        self.assertTrue(sent.output['ok'])

    async def test_publication_during_search_keeps_original_caption_vector_and_id_together(self):
        import asyncio
        asset = self.asset(caption='A quiet hug')
        entered, release = asyncio.Event(), asyncio.Event()
        async def delayed_embedding(*args, **kwargs):
            entered.set()
            await release.wait()
            return np.array([1., 0., 0.], dtype=np.float32)
        self.embeddings.embed_query.side_effect = delayed_embedding
        pending = asyncio.create_task(self.query())
        await entered.wait()
        new_card = {**asset.card, 'caption': 'Corrected caption'}
        changed = replace(asset, card=new_card)
        self.store.active_revision_id.return_value = 'r2'
        self.store.load_snapshot.side_effect = lambda _: CatalogSnapshot('r2', {'embedding_space_id': 'space'}, '', (changed,))
        await self.catalog.aensure_loaded()
        release.set()
        prior = await pending
        self.assertEqual(prior.output['catalog_revision'], 'r1')
        self.assertEqual(prior.output['candidates'][0]['caption'], 'A quiet hug')
        current = await self.query()
        self.assertEqual(current.output['catalog_revision'], 'r2')
        self.assertEqual(current.output['candidates'][0]['caption'], 'Corrected caption')

    async def test_only_fitting_asset_is_not_removed_because_it_was_sent_before(self):
        fitting = self.asset(caption='抱抱')
        self.asset(harshness=4, action='Threatening the recipient')
        self.deliveries.recent.return_value = [{'sticker_id': fitting.asset_id}]
        result = await self.query(diversity_preference='prefer_fresh_variant')
        self.assertEqual([c['sticker_id'] for c in result.output['candidates']], [fitting.asset_id])
        self.assertTrue(result.output['candidates'][0]['recently_delivered'])
