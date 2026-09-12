"""Summary reads retain layered content and historical source-count fallback."""
from __future__ import annotations

from tests.business_helpers import BusinessTestCase
from tgchatbot.domain.models import ConversationMessage


class SummaryReconstructionTests(BusinessTestCase):
    async def test_context_page_and_preparation_preserve_layers_and_missing_count_fallback(self):
        await self.settings()
        rows = await self.store.append_messages(self.session,
            [ConversationMessage.user_text(f'Original conversation {number}.') for number in range(5)])
        first = await self.store.create_memory_block(self.session, source_message_ids=[row.db_id for row in rows[:2]],
            summary_text='First conversation episode.', estimated_tokens=10, actor_labels=['person-a'])
        second = await self.store.create_memory_block(self.session, source_message_ids=[row.db_id for row in rows[2:4]],
            summary_text='Second conversation episode.', estimated_tokens=11, topic_labels=['garden'])
        digest = await self.store.create_memory_block(self.session, source_message_ids=[],
            parent_block_ids=[first.block_id, second.block_id], kind='digest', level=2,
            summary_text='Both episodes together.', estimated_tokens=12)
        # Historical records can omit the denormalized count. Their canonical
        # source membership still supplies it without changing summary content.
        async with self.store.pool.connection() as conn:
            await conn.execute("UPDATE memory_blocks SET details=details-'source_message_count' WHERE id=ANY(%s)",
                ([second.block_id, digest.block_id],))
            original_rows = await (await conn.execute('SELECT * FROM memory_blocks ORDER BY sequence_no,id')).fetchall()
        expected = [first, second, digest]
        self.assertEqual([self.store._block(row) for row in original_rows], expected,
            'The existing full-row mapper remains compatible with historical records')
        blocks, raw, _version = await self.store.load_live_context_versioned(self.session)
        self.assertEqual(blocks, expected, 'Full context still includes every valid ancestry node')
        self.assertEqual([row.db_id for row in raw], [rows[-1].db_id])
        self.assertEqual(await self.store.list_memory_blocks(self.session, limit=10), expected)
        prepared = await self.store.load_compaction_window(self.session)
        self.assertEqual(prepared.blocks, [digest], 'Preparation keeps its existing root selection')
        self.assertEqual(prepared.raw_messages, raw)
        self.assertEqual([block.source_message_count for block in blocks], [2, 2, 4])
        self.assertEqual([block.render_as_message() for block in blocks],
            [block.render_as_message() for block in expected])
        counts = await self.store.context_preparation_counts(self.session, rows[-1].db_id)
        self.assertEqual(counts, {'remaining_raw_messages': 1, 'remaining_root_blocks': 1})
