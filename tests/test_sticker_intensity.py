"""Soft expression preferences preserve choices, intent and conversation continuity."""
from dataclasses import replace
import os
import unittest
from unittest.mock import patch

from tests import test_sticker_retrieval as fixtures
from tests import test_sticker_ranking_controls as ranking_fixtures
from tgchatbot.stickers.config import StickerConfig
from tgchatbot.storage.sticker_catalog import CatalogAlias
from tgchatbot.tools.sticker_send import StickerQueryTool


class StickerIntensityTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = fixtures.StickerConversationTests.asyncSetUp
    asset = fixtures.StickerConversationTests.asset
    query = fixtures.StickerConversationTests.query
    expression_choices = ranking_fixtures.StickerRankingControlTests.expression_choices

    def preference_choices(self):
        strongest = self.asset(pack='global', harshness=4, reading_vectors=[[1, 0, 0]])
        familiar = self.asset(pack='familiar', harshness=4, reading_vectors=[[.98, .199, 0]])
        fresh = self.asset(pack='other', harshness=0, reading_vectors=[[.96, .28, 0]])
        alternative = self.asset(pack='familiar', harshness=0, reading_vectors=[[.9, .43589, 0]])
        return strongest, familiar, fresh, alternative

    @staticmethod
    def ids(result):
        return [candidate['sticker_id'] for candidate in result.output['candidates']]

    async def test_nearby_preference_keeps_semantic_leader_and_all_candidates(self):
        strongest, familiar, fresh, alternative = self.preference_choices()
        baseline = await self.query(candidate_budget=4)
        result = await self.query(candidate_budget=4, intensity_preference={'harshness': 0})
        self.assertEqual(self.ids(baseline), [strongest.agent_id, familiar.agent_id, fresh.agent_id, alternative.agent_id])
        self.assertEqual(self.ids(result), [strongest.agent_id, fresh.agent_id, alternative.agent_id, familiar.agent_id])
        one = await self.query(candidate_budget=1, intensity_preference={'harshness': 0})
        self.assertEqual(self.ids(one), [strongest.agent_id])
        self.assertEqual(self.embeddings.embed_query.await_count, 3)
        self.assertEqual(result.stickers, [])
        self.personas.save_sticker_persona.assert_not_awaited()

    async def test_optional_nulls_are_neutral_but_zero_is_a_preference(self):
        self.preference_choices()
        baseline = await self.query(candidate_budget=4)
        for preference in (None, {}, {'harshness': None, 'intimacy': None, 'meme_dependence': None}):
            with self.subTest(preference=preference):
                result = await self.query(candidate_budget=4, intensity_preference=preference)
                self.assertEqual(result.output, baseline.output)
                self.assertEqual(result.evidence_parts, baseline.evidence_parts)
        zero = await self.query(candidate_budget=4, intensity_preference={'harshness': 0})
        nullable = await self.query(candidate_budget=4,
            intensity_preference={'harshness': 0, 'intimacy': None, 'meme_dependence': None})
        self.assertNotEqual(self.ids(zero), self.ids(baseline))
        self.assertEqual(nullable.output, zero.output)

    async def test_preference_is_symmetric_around_the_requested_level(self):
        strongest, familiar, fresh, alternative = self.preference_choices()
        # Equal relevance order; the next choice near 3 is level 2, not level 0.
        for position, level in enumerate((0, 0, 2, 4)):
            asset = self.assets[position]
            self.assets[position] = replace(asset, card={**asset.card,
                'compatibility': {**asset.card['compatibility'], 'intimacy_level': level}})
        result = await self.query(candidate_budget=4, intensity_preference={'intimacy': 3})
        self.assertEqual(self.ids(result), [strongest.agent_id, fresh.agent_id, familiar.agent_id, alternative.agent_id])
        self.assertIn(alternative.agent_id, self.ids(result), 'A value above the target remains eligible')

    async def test_environment_can_disable_ranking_without_changing_eligibility(self):
        self.expression_choices()
        baseline = await self.query(candidate_budget=6)
        with patch.dict(os.environ, {'STICKER_INTENSITY_RANK_WEIGHT': '0'}):
            self.catalog.config = StickerConfig.from_env()
        result = await self.query(candidate_budget=6,
            intensity_preference={'harshness': 4, 'intimacy': 4, 'meme_dependence': 4})
        self.assertEqual(result.output, baseline.output)

    async def test_visual_diversity_does_not_undo_requested_intensity(self):
        first, second, photo, similar, alternative, _ = self.expression_choices()
        asset = self.assets[4]
        self.assets[4] = replace(asset, card={**asset.card,
            'compatibility': {**asset.card['compatibility'], 'harshness_level': 4}})
        baseline = await self.query(candidate_budget=4)
        self.assertEqual(self.ids(baseline), [first.agent_id, photo.agent_id, second.agent_id, alternative.agent_id])
        preferred = await self.query(candidate_budget=4, intensity_preference={'harshness': 0})
        self.assertEqual(self.ids(preferred), [first.agent_id, photo.agent_id, second.agent_id, similar.agent_id])
        larger = await self.query(candidate_budget=8, intensity_preference={'harshness': 0})
        self.assertIn(alternative.agent_id, self.ids(larger))

    async def test_freshness_and_explicit_continuity_keep_their_existing_priority(self):
        strongest, familiar, fresh, alternative = self.preference_choices()
        self.personas.get_sticker_persona.return_value = {'visual_identity': {'prefer_pack': 'familiar'}}
        self.deliveries.recent.return_value = [{'sticker_id': a.asset_id} for a in (strongest, familiar)]
        arguments = {'candidate_budget': 2, 'intensity_preference': {'harshness': 0},
                     'diversity_preference': 'prefer_fresh_variant'}
        default = await self.query(**arguments)
        self.assertEqual(self.ids(default), [strongest.agent_id, fresh.agent_id])
        explicit = await self.query(**arguments, preferred_pack='familiar')
        self.assertEqual(self.ids(explicit), [strongest.agent_id, alternative.agent_id])
        switched = await self.query(candidate_budget=2, intensity_preference={'harshness': 0},
            advanced={'style_focus': {'style_goal': 'prefer_switch'}})
        self.assertEqual(self.ids(switched), [strongest.agent_id, fresh.agent_id])
        required = await self.query(required_pack='familiar', intensity_preference={'harshness': 0})
        self.assertEqual(set(self.ids(required)), {familiar.agent_id, alternative.agent_id})
        self.personas.save_sticker_persona.assert_not_awaited()

    async def test_exact_lookup_needs_no_embeddings_and_duplicate_aliases_are_one_candidate(self):
        selected = self.asset(caption='Hello', harshness=4)
        alias = CatalogAlias(selected.aliases[0].path, 'collected')
        self.assets[0] = replace(selected, aliases=selected.aliases + (alias,))
        self.embeddings.enabled = False
        for intent in (selected.agent_id, 'Hello'):
            result = await self.query(intent_core=intent, intensity_preference={'harshness': 0})
            self.assertEqual(self.ids(result), [selected.agent_id])
            self.assertEqual(result.output['candidates'][0]['packs'], ['pack-a', 'collected'])
        self.embeddings.embed_query.assert_not_awaited()

    async def test_literal_caption_alternatives_can_prefer_intensity_without_embeddings(self):
        sharp = self.asset(caption='Hello', harshness=4)
        gentle = self.asset(caption='Hello', harshness=0)
        self.embeddings.enabled = False
        result = await self.query(intent_core='Hello', intensity_preference={'harshness': 0})
        self.assertEqual(self.ids(result), [gentle.agent_id, sharp.agent_id])
        self.embeddings.embed_query.assert_not_awaited()

    async def test_preference_does_not_persist_to_later_queries(self):
        self.preference_choices()
        before = await self.query(candidate_budget=4)
        await self.query(candidate_budget=4, intensity_preference={'harshness': 0})
        after = await self.query(candidate_budget=4)
        self.assertEqual(after.output, before.output)
        self.personas.save_sticker_persona.assert_not_awaited()
        self.personas.clear_sticker_persona.assert_not_awaited()

    def test_tool_preference_is_optional_for_all_provider_declarations(self):
        spec = StickerQueryTool(self.catalog).spec
        for declaration in (spec.generic_function_declaration(), spec.gemini_function_declaration()):
            schema = declaration['parameters']
            preference = schema['properties']['intensity_preference']
            self.assertNotIn('intensity_preference', schema.get('required', []))
            self.assertFalse(preference.get('required'))
            for axis in ('harshness', 'intimacy', 'meme_dependence'):
                self.assertNotIn('default', preference['properties'][axis])
        strict = spec.openai_tool()['parameters']['properties']['intensity_preference']
        self.assertIn('null', strict['type'])
        for axis in ('harshness', 'intimacy', 'meme_dependence'):
            self.assertIn('null', strict['properties'][axis]['type'])


if __name__ == '__main__':
    unittest.main()
