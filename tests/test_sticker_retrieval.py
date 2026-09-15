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
        self.pack_descriptions = {}
        self.store = Mock(active_revision_id=AsyncMock(return_value='r1'))
        self.store.load_snapshot = AsyncMock(side_effect=lambda _: CatalogSnapshot('r1', {'embedding_space_id': 'space'}, '', tuple(self.assets),
            pack_descriptions=dict(self.pack_descriptions)))
        self.personas = Mock(get_sticker_persona=AsyncMock(return_value=None),
            save_sticker_persona=AsyncMock(), clear_sticker_persona=AsyncMock())
        async def remember(session_id, value, **kwargs):
            self.personas.get_sticker_persona.return_value = value
        async def clear(session_id, **kwargs):
            self.personas.get_sticker_persona.return_value = None
        self.personas.save_sticker_persona.side_effect = remember
        self.personas.clear_sticker_persona.side_effect = clear
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
            np.array(vectors, dtype=np.float32), sticker_number=position + 1)
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
        self.assertEqual(ids, [multi.agent_id, other.agent_id])
        self.assertEqual(next(reading['meaning'] for reading in result.output['candidates'][0]['readings'] if reading.get('retrieval_match')), 'Offer care')
        self.assertEqual(len(set(ids)), 2)

    async def test_query_returns_actual_attributed_images_and_never_sends(self):
        asset = self.asset(caption='抱抱', action='Offering a hug')
        result = await self.query()
        self.assertEqual(result.stickers, [])
        images = [part for part in result.evidence_parts if part.kind == PartKind.IMAGE]
        self.assertTrue(images)
        self.assertEqual(images[0].origin, 'sticker_candidate:' + asset.agent_id)
        self.assertTrue(images[0].data_b64)
        self.assertNotIn('path', result.output['candidates'][0])
        self.assertNotIn('fit_signals', result.output['candidates'][0])
        self.assertEqual(self.deliveries.recent.await_count, 1)

    async def test_static_candidate_has_identity_without_an_animation_sampling_warning(self):
        asset = self.asset(caption='抱抱')
        result = await self.query()
        labels = [part.text for part in result.evidence_parts if part.kind == PartKind.TEXT]
        self.assertEqual(len(labels), 1)
        self.assertIn(asset.agent_id, labels[0])
        self.assertIn('animated=False', labels[0])
        self.assertNotIn('sampled frame times', labels[0])
        self.assertNotIn('intermediate animation events', labels[0])
        self.assertEqual(sum(part.kind == PartKind.IMAGE for part in result.evidence_parts), 1)
        self.assertEqual(result.output['candidates'][0]['caption'], '抱抱')

    async def test_configured_pack_description_accompanies_every_matching_candidate_every_time(self):
        first = self.asset(pack='series')
        second = self.asset(pack='series')
        plain = self.asset(pack='unconfigured')
        description = 'Dry, playful reaction cartoons with a blunt expressive style.'
        self.pack_descriptions = {'series': description, 'not-in-shortlist': 'Unrelated collection'}
        for _ in range(2):
            result = await self.query(candidate_budget=3)
            candidates = {item['sticker_id']: item for item in result.output['candidates']}
            for asset in (first, second):
                self.assertEqual(candidates[asset.agent_id]['packs'], ['series'])
                self.assertEqual(candidates[asset.agent_id]['pack_descriptions'], {'series': description})
            self.assertNotIn('pack_descriptions', candidates[plain.agent_id])
            self.assertEqual(result.stickers, [])
        self.personas.save_sticker_persona.assert_not_awaited()

    async def test_shared_asset_retains_each_pack_description_and_removal_refreshes_next_query(self):
        asset = self.asset(pack='series')
        self.assets[0] = replace(asset, aliases=asset.aliases + (CatalogAlias('other/alias.png', 'other'),))
        self.pack_descriptions = {'series': 'Warm drawn scenes', 'other': 'Collected reaction images'}
        first = await self.query()
        self.assertEqual(first.output['candidates'][0]['pack_descriptions'], self.pack_descriptions)
        self.store.active_revision_id.return_value = 'r2'
        self.store.load_snapshot.side_effect = lambda _: CatalogSnapshot('r2', {'embedding_space_id': 'space'}, '',
            tuple(self.assets), pack_descriptions={'other': 'Collected reaction images'})
        revised = await self.query()
        self.assertEqual(revised.output['candidates'][0]['packs'], ['series', 'other'])
        self.assertEqual(revised.output['candidates'][0]['pack_descriptions'], {'other': 'Collected reaction images'})
        self.assertEqual(revised.output['candidates'][0].get('caption', ''), asset.card['caption'])

    async def test_animation_warning_survives_sampling_only_one_distinct_frame(self):
        asset = self.asset(animated=True)
        path = (self.root / asset.aliases[0].path).with_suffix('.gif')
        frames = [Image.new('RGB', (16, 16), color) for color in ('red', 'blue', 'red')]
        try:
            frames[0].save(path, save_all=True, append_images=frames[1:], duration=100, loop=0)
        finally:
            for frame in frames:
                frame.close()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        asset = replace(asset, asset_id='sha256:' + digest, content_hash=digest,
            aliases=(CatalogAlias(path.relative_to(self.root).as_posix(), 'pack-a'),))
        self.assets[0] = asset
        # Two endpoints are identical; three samples reveal blue and the return to red.
        # Sampling/deduplication runs normally, without a mocked media decoder.
        for max_frames, expected_images in ((2, 1), (3, 3)):
            with self.subTest(max_frames=max_frames):
                self.catalog.media_config = replace(self.catalog.media_config, max_frames=max_frames)
                result = await self.query(allow_animation=True)
                self.assertTrue(result.output['ok'])
                label = next(part.text for part in result.evidence_parts if part.kind == PartKind.TEXT)
                self.assertIn(asset.agent_id, label)
                self.assertIn('animated=True', label)
                self.assertIn('sampled frame times', label)
                self.assertIn('intermediate animation events may be omitted', label)
                self.assertEqual(sum(part.kind == PartKind.IMAGE for part in result.evidence_parts), expected_images)

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
        result = await self.query(candidate_budget=1,
            advanced={'intensity_limits': {'max_harshness': 0, 'allow_animation': False}})
        self.assertEqual([x['sticker_id'] for x in result.output['candidates']], [valid.agent_id])

    async def test_pack_preference_keeps_global_choices_but_explicit_requirement_filters(self):
        global_asset = self.asset(pack='global', reading_vectors=[[1, 0, 0]])
        preferred = self.asset(pack='familiar', reading_vectors=[[.7, .71414, 0]])
        result = await self.query(preferred_pack='familiar', candidate_budget=2)
        self.assertEqual({x['sticker_id'] for x in result.output['candidates']}, {global_asset.agent_id, preferred.agent_id})
        required = await self.query(required_pack='familiar')
        self.assertEqual([x['sticker_id'] for x in required.output['candidates']], [preferred.agent_id])
        absent = await self.query(required_pack='absent')
        self.assertEqual(absent.output['status'], 'no_candidates')
        self.assertNotIn('does not exist', str(absent.output))

    async def test_known_family_can_span_packs_and_pack_is_not_identity(self):
        a = self.asset(pack='first', family=['round-cat'])
        b = self.asset(pack='second', family=['round-cat'])
        self.asset(pack='first', family=['dog'])
        result = await self.query(required_character_family='round-cat')
        self.assertEqual({x['sticker_id'] for x in result.output['candidates']}, {a.agent_id, b.agent_id})

    async def test_confirmed_repeat_stays_eligible_and_fresh_variant_is_offered(self):
        prior = self.asset(reading_vectors=[[1, 0, 0]])
        fresh = self.asset(reading_vectors=[[.99, .14106, 0]])
        self.deliveries.recent.return_value = [{'sticker_id': prior.asset_id}]
        result = await self.query(diversity_preference='prefer_fresh_variant', candidate_budget=2)
        candidates = result.output['candidates']
        by_id = {candidate['sticker_id']: candidate for candidate in candidates}
        self.assertEqual(set(by_id), {fresh.agent_id, prior.agent_id})
        self.assertFalse(by_id[fresh.agent_id].get('recently_delivered', False))
        self.assertTrue(by_id[prior.agent_id].get('recently_delivered', False))
        self.personas.save_sticker_persona.assert_not_awaited()

    async def test_common_caption_still_responds_to_changed_conversational_intent(self):
        warm = self.asset(caption='好', readings=[{'meaning': 'Warm agreement', 'context': 'Accept kindly'}],
            reading_vectors=[[1, 0, 0]])
        reluctant = self.asset(caption='好', readings=[{'meaning': 'Reluctant agreement', 'context': 'Accept grudgingly'}],
            reading_vectors=[[0, 1, 0]])
        async def embed(text, **kwargs):
            return np.array([1, 0, 0] if text.startswith('Warm') else [0, 1, 0], dtype=np.float32)
        self.embeddings.embed_query.side_effect = embed
        for intent, expected in [('Warm agreement', warm), ('Reluctant agreement', reluctant)]:
            with self.subTest(intent=intent):
                result = await self.query(intent_core=intent, caption_meaning='好', candidate_budget=1)
                self.assertEqual([c['sticker_id'] for c in result.output['candidates']], [expected.agent_id])
                self.assertEqual(next(reading['meaning'] for reading in result.output['candidates'][0]['readings'] if reading.get('retrieval_match')), intent)

    async def test_caption_hint_keeps_contextual_alternatives_and_literal_companion(self):
        literal = self.asset(caption='好', reading_vectors=[[0, 1, 0]])
        fitting = self.asset(caption='抱抱', reading_vectors=[[1, 0, 0]])
        for budget in (1, 2):
            with self.subTest(budget=budget):
                result = await self.query(caption_meaning='好', candidate_budget=budget)
                self.assertEqual([c['sticker_id'] for c in result.output['candidates']],
                    [fitting.agent_id, literal.agent_id][:budget])

    async def test_id_lookup_and_offline_caption_lookup_retain_hard_requirements(self):
        literal = self.asset(caption='好', pack='literal')
        selected = self.asset(caption='抱抱', pack='selected')
        self.embeddings.enabled = False
        direct = await self.query(intent_core=selected.agent_id, caption_meaning='好')
        self.assertEqual([c['sticker_id'] for c in direct.output['candidates']], [selected.agent_id])
        self.assertEqual(direct.output['search_scope']['retrieval'], 'asset_id')
        caption = await self.query(caption_meaning='好')
        self.assertEqual([c['sticker_id'] for c in caption.output['candidates']], [literal.agent_id])
        self.assertEqual(caption.output['search_scope']['retrieval'], 'literal')
        required = await self.query(caption_meaning='好', required_pack='selected')
        self.assertFalse(required.output['ok'])
        self.embeddings.embed_query.assert_not_awaited()

    async def test_caption_companion_uses_intent_order_before_unranked_captions(self):
        strongest = self.asset(caption='抱抱', reading_vectors=[[1, 0, 0]])
        weaker = self.asset(caption='好', reading_vectors=[[.8, .6, 0]])
        better = self.asset(caption='好', reading_vectors=[[.9, .43589, 0]])
        unranked = self.asset(caption='好', readings=[], reading_vectors=[])
        result = await self.query(caption_meaning='好', candidate_budget=2)
        self.assertEqual([c['sticker_id'] for c in result.output['candidates']], [strongest.agent_id, better.agent_id])
        full = await self.query(caption_meaning='好', candidate_budget=4)
        self.assertEqual({c['sticker_id'] for c in full.output['candidates']},
            {strongest.agent_id, weaker.agent_id, better.agent_id, unranked.agent_id})

    async def test_caption_only_catalog_can_be_inspected_without_vectors(self):
        asset = self.asset(caption='好', readings=[], reading_vectors=[])
        result = await self.query(caption_meaning='好')
        self.assertEqual([c['sticker_id'] for c in result.output['candidates']], [asset.agent_id])
        self.assertEqual(result.output['search_scope']['retrieval'], 'literal')
        self.embeddings.embed_query.assert_not_awaited()

    async def test_known_id_cannot_substitute_caption_match_when_restricted_or_missing(self):
        selected = self.asset(caption='抱抱', pack='selected')
        self.asset(caption='好', pack='other')
        restricted = await self.query(intent_core=selected.agent_id, caption_meaning='好', required_pack='other')
        self.assertEqual(restricted.output['candidates'], [])
        (self.root / selected.aliases[0].path).rename(self.root / 'missing.png')
        missing = await self.query(intent_core=selected.agent_id, caption_meaning='好')
        self.assertEqual(missing.output['candidates'], [])
        self.embeddings.embed_query.assert_not_awaited()

    async def test_changed_embedding_space_keeps_literal_lookup_and_reports_provider_failures(self):
        asset = self.asset(caption='好')
        self.embeddings.config.space_id = 'different-model'
        mismatch = await self.query(caption_meaning='好')
        self.assertEqual([c['sticker_id'] for c in mismatch.output['candidates']], [asset.agent_id])
        self.assertEqual(mismatch.output['search_scope']['retrieval'], 'literal')
        self.embeddings.embed_query.assert_not_awaited()
        direct = await self.query(intent_core=asset.agent_id)
        self.assertEqual([c['sticker_id'] for c in direct.output['candidates']], [asset.agent_id])
        self.embeddings.config.space_id = 'space'
        self.embeddings.embed_query.side_effect = RuntimeError('fixture provider failure')
        failed = await self.query(caption_meaning='好')
        self.assertFalse(failed.output['ok'])
        self.assertIn('fixture provider failure', failed.output['error'])

    def continuity_assets(self):
        strongest = self.asset(pack='global', reading_vectors=[[1, 0, 0]])
        familiar = self.asset(pack='familiar', family=['familiar-character'], reading_vectors=[[.98, .199, 0]])
        fresh = self.asset(pack='new', reading_vectors=[[.96, .28, 0]])
        self.deliveries.recent.return_value = [{'sticker_id': familiar.asset_id}, {'sticker_id': strongest.asset_id}]
        return strongest, familiar, fresh

    async def test_explicit_freshness_precedes_inherited_continuity_but_keeps_best_match(self):
        strongest, familiar, fresh = self.continuity_assets()
        for saved in (None, {'visual_identity': {'prefer_pack': 'familiar'}}):
            self.personas.get_sticker_persona.return_value = saved
            for extra in ({}, {'advanced': {'style_focus': {'style_goal': None}}},
                          {'advanced': {'style_focus': {'style_goal': 'unknown'}}},
                          {'persona_mode': 'use_once', 'persona': {'affect_profile': {'default_tone': 'warm'}}}):
                with self.subTest(saved=saved, extra=extra):
                    result = await self.query(candidate_budget=2, diversity_preference='prefer_fresh_variant', **extra)
                    self.assertEqual([c['sticker_id'] for c in result.output['candidates']],
                        [strongest.agent_id, fresh.agent_id])
        one = await self.query(candidate_budget=1, diversity_preference='prefer_fresh_variant')
        self.assertEqual([c['sticker_id'] for c in one.output['candidates']], [strongest.agent_id])

    async def test_explicit_switch_precedes_saved_pack_but_does_not_exclude_it(self):
        strongest, familiar, fresh = self.continuity_assets()
        self.personas.get_sticker_persona.return_value = {'visual_identity': {'prefer_pack': 'familiar'}}
        for budget in (1, 2, 3):
            with self.subTest(budget=budget):
                result = await self.query(candidate_budget=budget,
                    advanced={'style_focus': {'style_goal': 'prefer_switch'}})
                self.assertEqual([c['sticker_id'] for c in result.output['candidates']],
                    [strongest.agent_id, fresh.agent_id, familiar.agent_id][:budget])

    async def test_explicit_continuity_conflicts_keep_existing_compromise(self):
        strongest, familiar, fresh = self.continuity_assets()
        for explicit in ({'preferred_pack': 'familiar'}, {'preferred_character_family': 'familiar-character'},
                         {'advanced': {'style_focus': {'style_goal': 'preserve'}}}, {'style_policy': 'continue'},
                         {'persona_mode': 'use_once', 'persona': {'visual_identity': {'preferred_pack': 'familiar'}}}):
            with self.subTest(explicit=explicit):
                result = await self.query(candidate_budget=2, diversity_preference='prefer_fresh_variant', **explicit)
                self.assertEqual([c['sticker_id'] for c in result.output['candidates']],
                    [strongest.agent_id, familiar.agent_id])
        default = await self.query(candidate_budget=2)
        self.assertEqual([c['sticker_id'] for c in default.output['candidates']], [strongest.agent_id, familiar.agent_id])

    async def test_explicit_family_does_not_turn_saved_pack_into_requested_identity(self):
        strongest, familiar, fresh = self.continuity_assets()
        requested = self.asset(pack='requested', family=['requested-character'], reading_vectors=[[.9, .43589, 0]])
        self.personas.get_sticker_persona.return_value = {'visual_identity': {'prefer_pack': 'familiar'}}
        result = await self.query(candidate_budget=2, preferred_character_family='requested-character',
            diversity_preference='prefer_fresh_variant')
        self.assertEqual([c['sticker_id'] for c in result.output['candidates']], [strongest.agent_id, requested.agent_id])
        absent = await self.query(candidate_budget=2, preferred_character_family='absent',
            diversity_preference='prefer_fresh_variant')
        self.assertEqual({c['sticker_id'] for c in absent.output['candidates']}, {strongest.agent_id, fresh.agent_id})
        explicit_both = await self.query(candidate_budget=2, preferred_character_family='requested-character',
            preferred_pack='familiar', diversity_preference='prefer_fresh_variant')
        self.assertEqual([c['sticker_id'] for c in explicit_both.output['candidates']], [strongest.agent_id, familiar.agent_id])
        self.personas.save_sticker_persona.assert_not_awaited()

    async def test_moving_inherited_pack_keeps_current_appearance_and_switch_order(self):
        strongest = self.asset(pack='global', reading_vectors=[[1, 0, 0]], image_vector=[1, 0, 0])
        familiar = self.asset(pack='familiar', reading_vectors=[[.98, .199, 0]], image_vector=[.98, .199, 0])
        appearance = self.asset(pack='familiar', reading_vectors=[[.5, .866, 0]], image_vector=[0, 1, 0])
        switched = self.asset(pack='different', reading_vectors=[[.96, .28, 0]], image_vector=[.96, .28, 0])
        self.deliveries.recent.return_value = [{'sticker_id': familiar.asset_id}, {'sticker_id': strongest.asset_id}]
        self.personas.get_sticker_persona.return_value = {'visual_identity': {'prefer_pack': 'familiar'}}
        async def embed(text, **kwargs):
            return np.array([0, 1, 0] if 'preferred expression/style:' in text else [1, 0, 0], dtype=np.float32)
        self.embeddings.embed_query.side_effect = embed
        for budget in (3, 4):
            with self.subTest(budget=budget):
                result = await self.query(candidate_budget=budget, persona_mode='use_once',
                    persona={'visual_identity': {'rendering_style': 'gentle line art'}},
                    advanced={'style_focus': {'style_goal': 'prefer_switch'}})
                self.assertEqual([c['sticker_id'] for c in result.output['candidates']],
                    [strongest.agent_id, appearance.agent_id, switched.agent_id, familiar.agent_id][:budget])

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
        result = await tool.run({'sticker_id': asset.agent_id, 'timing': 'before_final'}, self.ctx)
        self.assertEqual(result.output['status'], 'queued')
        self.assertEqual(result.stickers[0].timing, StickerTiming.SEND_NOW)
        self.assertEqual(result.stickers[0].content_sha256, asset.content_hash)
        after = await tool.run({'selected_sticker_id': asset.agent_id}, self.ctx)
        self.assertEqual(after.stickers[0].timing, StickerTiming.AFTER_FINAL)
        self.deliveries.recent.assert_not_awaited()

    async def test_reused_filename_cannot_send_different_original(self):
        asset = self.asset()
        await self.catalog.aensure_loaded()
        (self.root / asset.aliases[0].path).write_bytes(b'replaced media')
        result = await StickerSendSelectedTool(self.catalog).run({'selected_sticker_id': asset.agent_id}, self.ctx)
        self.assertFalse(result.output['ok'])
        self.assertEqual(result.stickers, [])

    async def test_available_alias_preserves_content_identity_after_pack_copy(self):
        asset = self.asset()
        source = self.root / asset.aliases[0].path
        copy = self.root / 'copy.png'
        copy.write_bytes(source.read_bytes())
        self.assets[0] = replace(asset, aliases=asset.aliases + (CatalogAlias('copy.png', 'second'),))
        source.rename(self.root / 'moved.png')
        result = await StickerSendSelectedTool(self.catalog).run({'selected_sticker_id': asset.agent_id}, self.ctx)
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
        sent = await StickerSendSelectedTool(self.catalog).run({'selected_sticker_id': asset.agent_id}, self.ctx)
        self.assertTrue(sent.output['ok'])

    async def test_publication_during_search_keeps_original_caption_vector_and_id_together(self):
        import asyncio
        asset = self.asset(caption='A quiet hug')
        self.pack_descriptions = {'pack-a': 'Original pack description'}
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
        self.store.load_snapshot.side_effect = lambda _: CatalogSnapshot('r2', {'embedding_space_id': 'space'}, '', (changed,),
            pack_descriptions={'pack-a': 'Updated pack description'})
        await self.catalog.aensure_loaded()
        release.set()
        prior = await pending
        self.assertNotIn('catalog_revision', prior.output)
        self.assertEqual(prior.output['candidates'][0]['caption'], 'A quiet hug')
        self.assertEqual(prior.output['candidates'][0]['pack_descriptions'], {'pack-a': 'Original pack description'})
        current = await self.query()
        self.assertNotIn('catalog_revision', current.output)
        self.assertEqual(current.output['candidates'][0]['caption'], 'Corrected caption')
        self.assertEqual(current.output['candidates'][0]['pack_descriptions'], {'pack-a': 'Updated pack description'})

    async def test_only_fitting_asset_is_not_removed_because_it_was_sent_before(self):
        fitting = self.asset(caption='抱抱')
        self.asset(harshness=4, action='Threatening the recipient')
        self.deliveries.recent.return_value = [{'sticker_id': fitting.asset_id}]
        result = await self.query(diversity_preference='prefer_fresh_variant', max_harshness=0)
        self.assertEqual([c['sticker_id'] for c in result.output['candidates']], [fitting.agent_id])
        self.assertTrue(result.output['candidates'][0].get('recently_delivered', False))

    async def test_known_empty_required_pack_needs_no_remote_embedding_request(self):
        self.asset(pack='available')
        self.embeddings.enabled = False
        result = await self.query(required_pack='different-pack')
        self.assertTrue(result.output['ok'])
        self.assertEqual(result.output['status'], 'no_candidates')
        self.embeddings.embed_query.assert_not_awaited()

    async def test_source_alias_changed_to_external_symlink_is_not_sendable(self):
        asset = self.asset()
        confined = self.root / 'catalog'
        alias = confined / asset.aliases[0].path
        alias.parent.mkdir(parents=True)
        alias.symlink_to(self.root / asset.aliases[0].path)
        self.catalog.sticker_root = confined
        result = await StickerSendSelectedTool(self.catalog).run(
            {'selected_sticker_id': asset.agent_id}, self.ctx)
        self.assertFalse(result.output['ok'])
        self.assertEqual(result.stickers, [])

    async def test_audio_only_media_keeps_description_and_other_candidate_images(self):
        import wave
        asset = self.asset()
        path = self.root / asset.aliases[0].path
        with wave.open(str(path), 'wb') as source:
            source.setnchannels(1)
            source.setsampwidth(2)
            source.setframerate(8000)
            source.writeframes(b'\x00\x00' * 800)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assets[0] = replace(asset, asset_id='sha256:' + digest, content_hash=digest)
        other = self.asset()
        result = await self.query()
        self.assertTrue(result.output['ok'])
        self.assertEqual(len(result.output['candidates']), 2)
        self.assertTrue(any('visual evidence unavailable' in (part.text or '') for part in result.evidence_parts))
        self.assertTrue(any(part.kind == PartKind.IMAGE and part.origin.endswith(other.agent_id)
                            for part in result.evidence_parts))

    async def test_inflight_query_retains_delivered_history_after_lru_eviction(self):
        self.catalog.style_memory.max_sessions = 1
        asset = self.asset()
        self.deliveries.recent.return_value = [{'sticker_id': asset.asset_id}]
        plan = StickerRetrievalPlan.from_payload({'intent_core': 'A familiar greeting'})
        state, persona = await self.catalog.aprepare_query_context(
            plan=plan, session_id='chat-a', persist_persona=False)
        self.catalog.style_memory.get('another-chat')
        matches = await self.catalog.achoose(plan=plan, session_id='chat-a',
                                            session_state=state, persona_context=persona)
        self.assertEqual([match.entry.sticker_id for match in matches], [asset.asset_id])
        self.assertTrue(matches[0].recently_delivered)

    async def test_similar_text_descriptions_do_not_claim_visual_near_duplicates(self):
        prior = self.asset(image_vector=[1, 0, 0])
        different = self.asset(image_vector=[1, 0, 0])
        self.store.load_snapshot.side_effect = lambda _: CatalogSnapshot('r1', {
            'embedding_space_id': 'space', 'visual_embedding_source': 'description'}, '', tuple(self.assets))
        self.deliveries.recent.return_value = [{'sticker_id': prior.asset_id}]
        result = await self.query()
        candidates = {item['sticker_id']: item for item in result.output['candidates']}
        self.assertTrue(candidates[prior.agent_id].get('recently_delivered', False))
        self.assertEqual(candidates[different.agent_id].get('visually_similar_deliveries', []), [])

    async def test_compact_readings_preserve_both_roles_and_card_without_mutating_it(self):
        import copy
        asset = self.asset(readings=[
            {'meaning': 'Offer a headpat', 'context': 'Recipient needs reassurance'},
            {'meaning': 'Enjoy a headpat', 'context': 'Sender has just received affection'},
            {'meaning': 'Ask for a headpat', 'context': 'Sender wants some affection'}],
            reading_vectors=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        before = copy.deepcopy(asset.card)
        result = await self.query()
        candidate = result.output['candidates'][0]
        self.assertEqual(candidate['sticker_id'], asset.agent_id)
        self.assertEqual([{key: value for key, value in reading.items() if key != 'retrieval_match'}
                          for reading in candidate['readings']], before['readings'])
        self.assertEqual(sum(bool(reading.get('retrieval_match')) for reading in candidate['readings']), 1)
        self.assertEqual(asset.card, before)
        self.assertNotIn('matched_reading', candidate)
        self.assertNotIn('appearance_embedding_source', candidate)
        self.assertNotIn('uncertainty', candidate)
        self.assertNotIn('guidance', result.output)
        self.assertNotIn('catalog_revision', result.output)
        self.assertNotIn('candidate_count', result.output)
        self.assertTrue(all(asset.asset_id not in (part.text or '') + (part.origin or '')
                            for part in result.evidence_parts))

    async def test_recent_history_guides_selection_without_listing_unrelated_hashes(self):
        prior = self.asset(pack='familiar', caption='Hello', image_vector=[1, 0, 0])
        other = self.asset(pack='other', caption='Goodbye', image_vector=[1, 0, 0])
        removed_hash = 'sha256:' + 'f' * 64
        self.deliveries.recent.return_value = [{'sticker_id': prior.asset_id}, {'sticker_id': removed_hash}]
        result = await self.query(required_pack='other')
        self.assertNotIn('recent_deliveries', result.output)
        self.assertEqual(result.output['candidates'][0]['sticker_id'], other.agent_id)
        self.assertEqual(result.output['candidates'][0]['visually_similar_deliveries'], [prior.agent_id])
        self.assertEqual(self.catalog.style_memory.get('chat-a').recent_sticker_ids, [prior.asset_id, removed_hash])
        repeat = await self.query(required_pack='familiar')
        self.assertTrue(repeat.output['candidates'][0]['recently_delivered'])
        empty = await self.query(required_pack='absent')
        self.assertEqual(empty.output['status'], 'no_candidates')
        self.assertNotIn('recent_deliveries', empty.output)
        # Querying never becomes a delivery or a learned preference.
        self.assertEqual([call[0] for call in self.deliveries.method_calls], ['recent'] * 3)
        self.personas.save_sticker_persona.assert_not_awaited()

    async def test_old_hash_call_and_short_reference_select_same_verified_original(self):
        asset = self.asset(caption='Hello')
        tool = StickerSendSelectedTool(self.catalog)
        for reference in (asset.asset_id, asset.agent_id):
            result = await tool.run({'selected_sticker_id': reference}, self.ctx)
            self.assertEqual(result.output['sticker_id'], asset.agent_id)
            self.assertEqual(result.stickers[0].source_id, asset.asset_id)
            self.assertEqual(result.stickers[0].content_sha256, asset.content_hash)
            self.assertNotIn('catalog_revision', result.output)
            self.assertNotIn('guidance', result.output)
        failed = await tool.run({'selected_sticker_id': 'sha256:' + 'f' * 64}, self.ctx)
        self.assertFalse(failed.output['ok'])
        self.assertEqual(failed.stickers, [])
        self.assertNotIn('sticker_id', failed.output)

    async def test_stale_or_unavailable_reference_never_becomes_a_semantic_substitute(self):
        removed = self.asset(caption='A hug')
        other = self.asset(caption='sid:999')
        await self.catalog.aensure_loaded()
        self.store.active_revision_id.return_value = 'r2'
        self.store.load_snapshot.side_effect = lambda _: CatalogSnapshot(
            'r2', {'embedding_space_id': 'space'}, '', (other,))
        for reference in (removed.agent_id, 'sid:999'):
            with self.subTest(reference=reference):
                result = await self.query(intent_core=reference)
                self.assertEqual(result.output['status'], 'no_candidates')
                self.assertEqual(result.output['candidates'], [])
                self.assertEqual(result.evidence_parts, [])
                self.embeddings.embed_query.assert_not_awaited()
                sent = await StickerSendSelectedTool(self.catalog).run(
                    {'selected_sticker_id': reference}, self.ctx)
                self.assertFalse(sent.output['ok'])
                self.assertEqual(sent.stickers, [])
        # A retained record whose original file is missing also stays exact.
        (self.root / removed.aliases[0].path).rename(self.root / 'moved.png')
        self.store.active_revision_id.return_value = 'r3'
        self.store.load_snapshot.side_effect = lambda _: CatalogSnapshot(
            'r3', {'embedding_space_id': 'space'}, '', (removed, other))
        unavailable = await self.query(intent_core=removed.agent_id)
        self.assertEqual(unavailable.output['candidates'], [])
        self.embeddings.embed_query.assert_not_awaited()
        natural = await self.query(intent_core='Offer a greeting, similar to sid:1')
        self.assertEqual([candidate['sticker_id'] for candidate in natural.output['candidates']], [other.agent_id])
        self.embeddings.embed_query.assert_awaited_once()
