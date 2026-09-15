"""Selection controls: retrieval preferences do not replace asset or delivery truth."""
from __future__ import annotations

import unittest
import json
from dataclasses import replace
from math import sqrt
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tests import test_sticker_retrieval as fixtures
from tests import test_sticker_runtime as runtime_fixtures
from tests.business_helpers import BusinessTestCase
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import ChatMode, ConversationMessage, ToolResult
from tgchatbot.tools.base import ToolSpec
from tgchatbot.tools.sticker_send import StickerQueryTool
from tgchatbot.storage.sticker_catalog import CatalogAlias, CatalogSnapshot


class StickerRankingControlTests(unittest.IsolatedAsyncioTestCase):
    # Reuse only fixture construction, without inheriting its test cases.
    asyncSetUp = fixtures.StickerConversationTests.asyncSetUp
    asset = fixtures.StickerConversationTests.asset
    query = fixtures.StickerConversationTests.query

    def expression_choices(self, *, include_alternative=True):
        first = self.asset(pack='drawn', caption='辛苦了', reading_vectors=[[1, 0, 0]],
                           image_vector=[0, 0, 1])
        second = self.asset(pack='drawn', reading_vectors=[[.99, sqrt(1-.99**2), 0]],
                            image_vector=[.1, 0, sqrt(.99)])
        photo = self.asset(pack='familiar', reading_vectors=[[.6, .8, 0]],
                           image_vector=[.8, .6, 0])
        similar = self.asset(pack='familiar', caption='抱抱', reading_vectors=[[.65, sqrt(1-.65**2), 0]],
                             image_vector=[.79, .35, sqrt(1-.79**2-.35**2)])
        alternative = (self.asset(pack='another', reading_vectors=[[.8, .6, 0]],
                       image_vector=[.7, -.6, sqrt(.15)]) if include_alternative else None)
        irrelevant = self.asset(pack='unrelated', caption='走开',
            readings=[{'meaning': 'Reject affection', 'context': 'Request distance'}],
            reading_vectors=[[.1, sqrt(.99), 0]], image_vector=[.69, -sqrt(1-.69**2), 0])
        return first, second, photo, similar, alternative, irrelevant

    async def test_visual_alternative_broadens_shortlist_without_weaker_meaning_or_duplicate_aliases(self):
        first, second, photo, similar, alternative, irrelevant = self.expression_choices()
        self.assets[4] = replace(alternative, aliases=alternative.aliases +
            (CatalogAlias(alternative.aliases[0].path, 'collected'),))

        result = await self.query(candidate_budget=4)

        candidates = result.output['candidates']
        self.assertEqual([item['sticker_id'] for item in candidates],
                         [first.agent_id, photo.agent_id, second.agent_id, alternative.agent_id])
        self.assertEqual(candidates[0]['caption'], '辛苦了')
        self.assertEqual(candidates[-1]['packs'], ['another', 'collected'])
        self.assertEqual(len({part.origin for part in result.evidence_parts}), 4)
        again = await self.query(candidate_budget=4)
        self.assertEqual(again.output['candidates'], candidates)

    async def test_visual_novelty_cannot_fill_shortlist_with_less_relevant_expression(self):
        first, second, photo, similar, _, irrelevant = self.expression_choices(include_alternative=False)

        result = await self.query(candidate_budget=4)

        self.assertEqual([item['sticker_id'] for item in result.output['candidates']],
                         [first.agent_id, photo.agent_id, second.agent_id, similar.agent_id])
        self.assertNotIn(irrelevant.agent_id, str(result.output['candidates']))

    async def test_visual_diversity_keeps_requested_continuity_and_small_pool(self):
        first, second, photo, similar, alternative, irrelevant = self.expression_choices()

        preferred = await self.query(preferred_pack='familiar', candidate_budget=2)
        self.assertEqual([item['sticker_id'] for item in preferred.output['candidates']],
                         [first.agent_id, similar.agent_id])
        required = await self.query(required_pack='familiar', candidate_budget=4)
        self.assertEqual({item['sticker_id'] for item in required.output['candidates']},
                         {photo.agent_id, similar.agent_id})
        self.assertEqual(next(item['caption'] for item in required.output['candidates']
                              if item['sticker_id'] == similar.agent_id), '抱抱')
        exact = await StickerQueryTool(self.catalog).run({'intent_core': similar.agent_id}, self.ctx)
        self.assertEqual([item['sticker_id'] for item in exact.output['candidates']], [similar.agent_id])

    async def test_identical_image_vectors_do_not_invent_diversity_or_discard_captions(self):
        for caption, score in [('抱抱', .9), ('辛苦了', .8), ('休息吧', .7)]:
            self.asset(caption=caption, reading_vectors=[[score, sqrt(1-score**2), 0]],
                       image_vector=[.8, .6, 0])
        first = await self.query(candidate_budget=8)
        second = await self.query(candidate_budget=8)
        self.assertEqual(first.output['candidates'], second.output['candidates'])
        self.assertEqual({item['caption'] for item in first.output['candidates']}, {'抱抱', '辛苦了', '休息吧'})
        self.assertEqual(len(first.output['candidates']), 3)

    async def test_stronger_visual_cue_and_caption_requirement_keep_their_best_match(self):
        first, _, photo, similar, _, _ = self.expression_choices()
        cue = 'the familiar photographed gesture'
        async def embed(text, **kwargs):
            return similar.image_vector if text.endswith(cue) else fixtures.np.array([1., 0., 0.])
        self.embeddings.embed_query.side_effect = embed

        visual = await self.query(expression_cue=cue, candidate_budget=2)
        self.assertEqual([item['sticker_id'] for item in visual.output['candidates']],
                         [first.agent_id, similar.agent_id])
        captioned = await self.query(candidate_budget=2,
            advanced={'text_constraints': {'text_priority': 'require'}})
        self.assertEqual([item['sticker_id'] for item in captioned.output['candidates']],
                         [first.agent_id, similar.agent_id])
        self.assertTrue(all(item['caption'] for item in captioned.output['candidates']))

    async def test_description_fallback_does_not_claim_visual_diversity(self):
        first, second, photo, similar, _, _ = self.expression_choices()
        self.store.load_snapshot.side_effect = lambda _: CatalogSnapshot('r1',
            {'embedding_space_id': 'space', 'visual_embedding_source': 'description'}, '', tuple(self.assets))

        result = await self.query(candidate_budget=4)

        self.assertEqual([item['sticker_id'] for item in result.output['candidates']],
                         [first.agent_id, photo.agent_id, second.agent_id, similar.agent_id])

    async def test_many_readings_and_image_match_do_not_crowd_out_second_asset(self):
        multi = self.asset(
            readings=[{'meaning': 'Ask for comfort'}, {'meaning': 'Offer comfort'},
                      {'meaning': 'Offer quiet reassurance'}],
            reading_vectors=[[0, 1, 0], [1, 0, 0], [.99, .14106, 0]],
            image_vector=[1, 0, 0])
        other = self.asset(action='A reassuring nod', reading_vectors=[[.8, .6, 0]],
                           image_vector=[.9, .43589, 0])

        result = await self.query(candidate_budget=2)

        candidates = result.output['candidates']
        self.assertEqual([item['sticker_id'] for item in candidates],
                         [multi.agent_id, other.agent_id])
        self.assertEqual(next(reading['meaning'] for reading in candidates[0]['readings'] if reading.get('retrieval_match')), 'Offer comfort')
        self.assertEqual(len({part.origin for part in result.evidence_parts}), 2)

    async def test_inspecting_again_neither_rotates_choices_nor_learns_a_delivery(self):
        prior = self.asset(pack='familiar', reading_vectors=[[1, 0, 0]])
        self.asset(pack='other', reading_vectors=[[.8, .6, 0]])
        self.deliveries.recent.return_value = [{'sticker_id': prior.asset_id}]
        self.personas.get_sticker_persona.return_value = {
            'visual_identity': {'prefer_pack': 'familiar'}}

        first = await self.query(candidate_budget=2)
        second = await self.query(candidate_budget=2)

        self.assertEqual(first.output['candidates'], second.output['candidates'])
        self.assertEqual(first.evidence_parts, second.evidence_parts)
        self.assertEqual(first.stickers + second.stickers, [])
        self.personas.save_sticker_persona.assert_not_awaited()
        self.personas.clear_sticker_persona.assert_not_awaited()
        self.assertEqual([call[0] for call in self.deliveries.method_calls],
                         ['recent', 'recent'])

    async def test_similar_artwork_does_not_erase_opposing_caption_and_meaning(self):
        # Controlled equal image vectors deliberately carry no evidence that
        # these opposite readings are interchangeable conversational acts.
        reassurance = self.asset(caption='没事的', action='Reassuring someone',
            readings=[{'meaning': 'Offer reassurance', 'context': 'Someone is worried'}],
            image_vector=[1, 0, 0])
        dismissal = self.asset(caption='别烦我', action='Dismissing someone',
            readings=[{'meaning': 'Request distance', 'context': 'Sender wants to be left alone'}],
            reading_vectors=[[.8, .6, 0]], image_vector=[1, 0, 0])
        self.deliveries.recent.return_value = [{'sticker_id': reassurance.asset_id}]

        result = await self.query(candidate_budget=2)

        by_id = {item['sticker_id']: item for item in result.output['candidates']}
        self.assertEqual(set(by_id), {reassurance.agent_id, dismissal.agent_id})
        self.assertEqual(by_id[dismissal.agent_id].get('visually_similar_deliveries', []),
                         [reassurance.agent_id])
        self.assertEqual(by_id[dismissal.agent_id]['readings'][0]['meaning'], 'Request distance')
        self.assertFalse(by_id[dismissal.agent_id].get('recently_delivered', False))
        self.assertEqual(result.stickers, [])

    async def test_freshness_does_not_override_required_pack_or_forbid_only_repeat(self):
        fitting = self.asset(pack='chosen', caption='抱抱')
        self.asset(pack='other', reading_vectors=[[1, 0, 0]])
        self.deliveries.recent.return_value = [{'sticker_id': fitting.asset_id}]

        result = await self.query(required_pack='chosen', candidate_budget=1,
                                  diversity_preference='prefer_fresh_variant')

        self.assertEqual([item['sticker_id'] for item in result.output['candidates']],
                         [fitting.agent_id])
        self.assertTrue(result.output['candidates'][0].get('recently_delivered', False))


class StickerImageAdmissionControlTests(BusinessTestCase):
    make_provider = runtime_fixtures.StickerEvidenceWorkflowTests.make_provider

    async def test_requested_fresh_choice_reaches_model_and_image_reduction_is_explicit(self):
        fixture = StickerRankingControlTests()
        await fixture.asyncSetUp()
        self.addCleanup(fixture.doCleanups)
        global_asset = fixture.asset(pack='global', reading_vectors=[[1, 0, 0]])
        familiar = fixture.asset(pack='familiar', reading_vectors=[[.9, .43589, 0]])
        fresh = fixture.asset(pack='other', reading_vectors=[[.8, .6, 0]])
        fixture.personas.get_sticker_persona.return_value = {
            'visual_identity': {'prefer_pack': 'familiar'}}
        fixture.deliveries.recent.return_value = [
            {'sticker_id': familiar.asset_id}, {'sticker_id': global_asset.asset_id}]
        arguments = {'intent_core': 'Offer a quiet hug after work',
                     'diversity_preference': 'prefer_fresh_variant', 'candidate_budget': 2}
        query = StickerQueryTool(fixture.catalog)
        query_results = []

        async def run_query(args, context):
            result = await query.run(args, context)
            query_results.append(result)
            return result

        self.tools.list_tools.return_value = [ToolSpec('sticker_query', 'Find expressions',
            {'type': 'object'}, SimpleNamespace(run=run_query))]
        for image_limit in (2, 1):
            with self.subTest(image_limit=image_limit):
                self.session = f'telegram:ranking-images-{image_limit}'
                await self.settings(mode=ChatMode.ASSIST, max_input_images=image_limit,
                    compact_target_images=image_limit, compact_trigger_tokens=100000,
                    max_interaction_rounds=1)
                provider = await self.make_provider([
                    {'status': 'completed', 'output': [{'type': 'function_call', 'name': 'sticker_query',
                                 'call_id': 'query', 'arguments': json.dumps(arguments)}]},
                    {'status': 'completed', 'output': [{'type': 'message', 'role': 'assistant', 'content': [
                        {'type': 'output_text', 'text': 'I can examine these candidates.'}]}]},
                ])
                runtime = AgentRuntime(config=self.config, store=self.store,
                    tool_registry=self.tools, providers={'openai': provider},
                    preview_cache=self.preview_cache)

                result = await runtime.run_turn(session_id=self.session, user_display_name='Human',
                    incoming_message=ConversationMessage.user_text('Find a fresh expression'))

                self.assertEqual([item['sticker_id']
                    for item in query_results[-1].output['candidates']],
                    [global_asset.agent_id, fresh.agent_id])
                continuation = next(item for item in self.wire[1]['input']
                                    if item.get('type') == 'function_call_output')
                output = json.loads(continuation['output'][0]['text'])
                expected = [global_asset.agent_id, fresh.agent_id][:image_limit]
                self.assertEqual([item['sticker_id'] for item in output['candidates']], expected)
                self.assertEqual(len(output['candidates']), image_limit)
                self.assertEqual(sum(part['type'] == 'input_image'
                    for part in continuation['output']), image_limit)
                if image_limit == 1:
                    self.assertIn('omitted candidates were not visually presented', output['evidence_notice'])
                else:
                    self.assertNotIn('evidence_notice', output)
                self.assertEqual(result.stickers, [])

    async def test_smaller_later_candidate_fits_without_partial_animation(self):
        await self.settings(mode=ChatMode.ASSIST, max_input_images=3,
            compact_target_images=3, compact_trigger_tokens=100000, max_interaction_rounds=1)
        provider = await self.make_provider([
            runtime_fixtures.function('sticker_query', 'query'),
            {'status': 'completed', 'output': [{'type': 'message', 'role': 'assistant', 'content': [
                {'type': 'output_text', 'text': 'I will inspect the supplied candidates.'}]}]},
        ])
        candidates = [{'sticker_id': name} for name in ('a', 'b', 'c')]
        parts = (runtime_fixtures.candidate_parts('a', 2)
                 + runtime_fixtures.candidate_parts('b', 2)
                 + runtime_fixtures.candidate_parts('c', 1))
        runner = SimpleNamespace(run=AsyncMock(return_value=ToolResult('', 'sticker_query',
            {'ok': True, 'candidates': candidates, 'candidate_count': 3}, evidence_parts=parts)))
        self.tools.list_tools.return_value = [ToolSpec(
            'sticker_query', 'Find expressions', {'type': 'object'}, runner)]
        runtime = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools,
                               providers={'openai': provider}, preview_cache=self.preview_cache)

        result = await runtime.run_turn(session_id=self.session, user_display_name='Human',
            incoming_message=ConversationMessage.user_text('Inspect candidates before choosing'))

        continuation = next(item for item in self.wire[1]['input']
                            if item.get('type') == 'function_call_output')
        output = json.loads(continuation['output'][0]['text'])
        self.assertEqual([item['sticker_id'] for item in output['candidates']], ['a', 'c'])
        self.assertEqual(output['candidate_count'], 2)
        self.assertEqual(sum(part['type'] == 'input_image'
                             for part in continuation['output']), 3)
        self.assertIn('omitted candidates were not visually presented', output['evidence_notice'])
        self.assertEqual(result.stickers, [])
