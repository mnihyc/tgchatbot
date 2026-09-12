"""Selection controls: retrieval preferences do not replace asset or delivery truth."""
from __future__ import annotations

import unittest
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

from tests import test_sticker_retrieval as fixtures
from tests import test_sticker_runtime as runtime_fixtures
from tests.business_helpers import BusinessTestCase
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import ChatMode, ConversationMessage, ToolResult
from tgchatbot.tools.base import ToolSpec
from tgchatbot.tools.sticker_send import StickerQueryTool


class StickerRankingControlTests(unittest.IsolatedAsyncioTestCase):
    # Reuse only fixture construction, without inheriting its test cases.
    asyncSetUp = fixtures.StickerConversationTests.asyncSetUp
    asset = fixtures.StickerConversationTests.asset
    query = fixtures.StickerConversationTests.query

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
                         [multi.asset_id, other.asset_id])
        self.assertEqual(candidates[0]['matched_reading']['meaning'], 'Offer comfort')
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
        self.assertEqual(set(by_id), {reassurance.asset_id, dismissal.asset_id})
        self.assertEqual(by_id[dismissal.asset_id]['visually_similar_deliveries'],
                         [reassurance.asset_id])
        self.assertEqual(by_id[dismissal.asset_id]['readings'][0]['meaning'], 'Request distance')
        self.assertFalse(by_id[dismissal.asset_id]['recently_delivered'])
        self.assertEqual(result.stickers, [])

    async def test_freshness_does_not_override_required_pack_or_forbid_only_repeat(self):
        fitting = self.asset(pack='chosen', caption='抱抱')
        self.asset(pack='other', reading_vectors=[[1, 0, 0]])
        self.deliveries.recent.return_value = [{'sticker_id': fitting.asset_id}]

        result = await self.query(required_pack='chosen', candidate_budget=1,
                                  diversity_preference='prefer_fresh_variant')

        self.assertEqual([item['sticker_id'] for item in result.output['candidates']],
                         [fitting.asset_id])
        self.assertTrue(result.output['candidates'][0]['recently_delivered'])


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
                    {'output': [{'type': 'function_call', 'name': 'sticker_query',
                                 'call_id': 'query', 'arguments': json.dumps(arguments)}]},
                    {'output': [{'type': 'message', 'role': 'assistant', 'content': [
                        {'type': 'output_text', 'text': 'I can examine these candidates.'}]}]},
                ])
                runtime = AgentRuntime(config=self.config, store=self.store,
                    tool_registry=self.tools, providers={'openai': provider},
                    preview_cache=self.preview_cache)

                result = await runtime.run_turn(session_id=self.session, user_display_name='Human',
                    incoming_message=ConversationMessage.user_text('Find a fresh expression'))

                self.assertEqual([item['sticker_id']
                    for item in query_results[-1].output['candidates']],
                    [global_asset.asset_id, fresh.asset_id])
                continuation = next(item for item in self.wire[1]['input']
                                    if item.get('type') == 'function_call_output')
                output = json.loads(continuation['output'][0]['text'])
                expected = [global_asset.asset_id, fresh.asset_id][:image_limit]
                self.assertEqual([item['sticker_id'] for item in output['candidates']], expected)
                self.assertEqual(output['candidate_count'], image_limit)
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
            {'output': [{'type': 'message', 'role': 'assistant', 'content': [
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
