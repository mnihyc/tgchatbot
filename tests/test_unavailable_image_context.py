"""Unavailable visual references consume text context, never image slots."""
from __future__ import annotations

from tests.business_helpers import BusinessTestCase
from tests.test_message_images import pixels
from tgchatbot.core.token_estimator import TokenEstimator
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind


class UnavailableImageContextTests(BusinessTestCase):
    def missing(self, number):
        return ConversationMessage(MessageRole.USER, [MessagePart(PartKind.IMAGE,
            text=f'[Imported attachment {number}: photo unavailable]', filename=f'missing-{number}.png',
            mime_type='image/png', origin='attachment_reference', remote_sync=False)])

    def image(self, color):
        return ConversationMessage(MessageRole.USER, [MessagePart(PartKind.IMAGE,
            mime_type='image/png', data_b64=pixels(color), remote_sync=False)])

    async def test_references_beside_one_image_do_not_trigger_image_only_compaction(self):
        settings = await self.settings(max_input_images=1, compact_target_images=1,
            compact_trigger_tokens=100000)
        rows = await self.store.append_messages(self.session,
            [self.missing(1), self.missing(2), self.image('red')])
        state = await self.runtime._get_live_state(self.session)
        events = []
        async def emit(event):
            events.append(event)
        compacted = await self.runtime._compact_if_needed(session_id=self.session,
            state=state, settings=settings, provider=self.provider, instructions='', tools=[], emit=emit)
        self.assertFalse(compacted, 'Text-only unavailable references must not trigger image retirement')
        self.assertEqual(events, [])
        self.assertEqual(state.estimated_images, 1)
        self.assertEqual(await self.store.get_compaction_version(self.session), 0)
        _blocks, restored = await self.store.load_live_context(self.session)
        self.assertEqual([row.message for row in restored], [row.message for row in rows])
        self.assertEqual(self.provider.requests, [], 'Image-only accounting needs no model request')

    async def test_old_missing_references_cannot_consume_retirement_ahead_of_real_pixels(self):
        await self.settings()
        rows = await self.store.append_messages(self.session,
            [self.missing(1), self.missing(2), self.image('red'), self.image('blue')])
        retired = await self.store.retire_context_images(self.session, target_images=1)
        self.assertEqual(retired.removed_images, 1)
        _blocks, restored = await self.store.load_live_context(self.session)
        self.assertEqual([row.message for row in restored[:2]], [row.message for row in rows[:2]],
            'Unavailable references must keep their searchable description and occurrence')
        self.assertEqual(restored[2].message.parts[0].text, '[Image compacted]')
        self.assertTrue(restored[3].message.parts[0].preview_ref)
        self.assertEqual(sum(row.image_count for row in restored), 1)
        described = await self.store.describe_message_images(self.session, [row.db_id for row in rows])
        self.assertEqual([items[0]['available'] for items in described.values()], [False, False, True, True])

    async def test_unavailable_reference_estimate_tracks_its_text_without_visual_charge(self):
        await self.settings()
        short = self.missing(1)
        long = self.missing(1)
        long.parts[0].text += ' More unavailable attachment context.' * 80
        self.assertLess(TokenEstimator.estimate_message(short), TokenEstimator.IMAGE_TOKENS)
        self.assertGreater(TokenEstimator.estimate_message(long), TokenEstimator.estimate_message(short))
        rows = await self.store.append_messages(self.session, [short, self.image('red')])
        self.assertEqual(rows[0].image_count, 0)
        self.assertEqual(rows[1].image_count, 1, 'Database preview refs still represent real image admission')
        self.assertGreaterEqual(rows[1].estimated_tokens, TokenEstimator.IMAGE_TOKENS)
