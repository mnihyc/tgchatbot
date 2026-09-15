"""Independent regressions for literal lookup and explicit chooser controls."""
from __future__ import annotations
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock

import numpy as np
from PIL import Image
from tgchatbot.stickers.catalog import StickerCatalog
from tgchatbot.stickers.plan import StickerRetrievalPlan
from tgchatbot.storage.sticker_catalog import CatalogAlias, CatalogAsset, CatalogSnapshot
from tgchatbot.tools.base import ToolContext
from tgchatbot.tools.sticker_send import StickerQueryTool


class StickerReviewWorkflows(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.assets = []
        self.store = Mock(active_revision_id=AsyncMock(return_value='revision'))
        self.store.load_snapshot = AsyncMock(side_effect=lambda _: CatalogSnapshot('revision', {'embedding_space_id': 'original-space'}, '', tuple(self.assets)))
        self.embeddings = Mock(enabled=True, config=SimpleNamespace(space_id='original-space'),
            embed_query=AsyncMock(return_value=np.array([1., 0., 0.], dtype=np.float32)))
        self.catalog = StickerCatalog(self.store, self.root, embedding_client=self.embeddings)
        self.ctx = ToolContext('synthetic-chat', 'Participant')

    def asset(self, caption, reading, image, family=(), pack='pack', animated=False):
        position = len(self.assets)
        path = self.root / f'{position}.png'
        Image.new('RGB', (12, 12), (position * 40, 70, 100)).save(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        card = {'caption': caption, 'appearance': 'A synthetic character', 'action': caption,
                'readings': [{'meaning': caption, 'context': 'A fitting conversation'}], 'uncertainty': '',
                'compatibility': {'harshness_level': 0, 'intimacy_level': 0, 'meme_dependence_level': 0}}
        asset = CatalogAsset('sha256:' + digest, digest, (CatalogAlias(path.name, pack),),
            {'animated': animated}, card, {'family_ids': list(family)}, card, {},
            np.array(image, dtype=np.float32), np.array([reading], dtype=np.float32), sticker_number=position + 1)
        self.assets.append(asset)
        return asset

    async def test_exact_multiline_caption_survives_changed_or_failing_embedding_route(self):
        selected = self.asset('好好\n休息', [1, 0, 0], [1, 0, 0])
        self.embeddings.config.space_id = 'replacement-route-not-built'
        self.embeddings.embed_query.side_effect = RuntimeError('Embedding provider unavailable')
        result = await StickerQueryTool(self.catalog).run({'intent_core': '好好 休息'}, self.ctx)
        self.assertTrue(result.output['ok'])
        self.assertEqual([item['sticker_id'] for item in result.output['candidates']], [selected.agent_id])
        self.assertEqual(result.output['candidates'][0]['caption'], '好好\n休息')
        self.embeddings.embed_query.assert_not_awaited()
        self.assertTrue(result.evidence_parts)

    async def test_exact_content_identity_needs_no_embedding_request(self):
        selected = self.asset('Greeting', [1, 0, 0], [1, 0, 0])
        self.embeddings.enabled = False
        result = await StickerQueryTool(self.catalog).run({'intent_core': selected.agent_id}, self.ctx)
        self.assertTrue(result.output['ok'])
        self.assertEqual(result.output['candidates'][0]['sticker_id'], selected.agent_id)
        self.embeddings.embed_query.assert_not_awaited()

    async def test_two_candidate_budget_keeps_global_and_requested_family_with_two_vector_channels(self):
        global_reading = self.asset('Strong reading match', [1, 0, 0], [0, 1, 0])
        self.asset('Strong image match', [.8, .6, 0], [1, 0, 0])
        familiar = self.asset('Familiar fitting variant', [.6, .8, 0], [.6, .8, 0], family=['round-cat'])
        result = await StickerQueryTool(self.catalog).run({'intent_core': 'Express welcome',
            'preferred_character_family': 'round-cat', 'candidate_budget': 2}, self.ctx)
        self.assertEqual([item['sticker_id'] for item in result.output['candidates']], [global_reading.agent_id, familiar.agent_id])
        single = await StickerQueryTool(self.catalog).run({'intent_core': 'Express welcome',
            'preferred_character_family': 'round-cat', 'candidate_budget': 1}, self.ctx)
        self.assertEqual(single.output['candidates'][0]['sticker_id'], global_reading.agent_id)

    async def test_strict_nullable_advanced_controls_do_not_erase_explicit_animation_and_pack(self):
        selected = self.asset('Animated acknowledgement', [1, 0, 0], [1, 0, 0], pack='familiar', animated=True)
        payload = {'intent_core': 'Express approval', 'allow_animation': True, 'preferred_pack': 'familiar',
            'advanced': {'intensity_limits': {'allow_animation': None, 'max_harshness': None},
                         'style_focus': {'preferred_pack': None}}}
        plan = StickerRetrievalPlan.from_payload(payload)
        self.assertTrue(plan.allow_animation)
        self.assertEqual(plan.prefer_pack, 'familiar')
        result = await StickerQueryTool(self.catalog).run(payload, self.ctx)
        self.assertTrue(result.output['ok'])
        self.assertEqual(result.output['candidates'][0]['sticker_id'], selected.agent_id)

    async def test_null_advanced_groups_preserve_explicit_compatibility_exclusions(self):
        self.asset('A caption', [1, 0, 0], [1, 0, 0])
        result = await StickerQueryTool(self.catalog).run({'intent_core': 'A social acknowledgement',
            'forbid': ['humiliating the recipient'], 'text_constraints': {'avoid_text_meanings': ['asking for money']},
            'advanced': {'forbid': None, 'text_constraints': None}}, self.ctx)
        self.assertEqual(result.output['constraints']['forbid'], ['humiliating the recipient'])
        self.assertEqual(result.output['constraints']['avoid_text_meanings'], ['asking for money'])
        sent = ' '.join(call.args[0] for call in self.embeddings.embed_query.await_args_list)
        self.assertNotIn('humiliating', sent)
        self.assertNotIn('asking for money', sent)
