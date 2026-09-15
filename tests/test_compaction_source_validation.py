"""Compaction publishes only from available, current original evidence."""
from __future__ import annotations

from tests.business_helpers import BusinessTestCase
from tgchatbot.domain.models import ConversationMessage, MessagePart, PartKind
from tgchatbot.storage.postgres_store import StaleScopeError


class CompactionSourceValidationTests(BusinessTestCase):
    async def original(self, session, number, text, **metadata):
        stored = await self.store.append_message(session, ConversationMessage.user_text(text, metadata={
            'source': 'telegram', 'source_chat_id': session.split(':', 1)[1],
            'source_message_id': str(number), 'actor_id': 'telegram:user:101',
            'actor_kind': 'user', 'actor_name': 'Participant', **metadata}))
        # Capture the canonical evidence before any attempted publication.
        return (await self.store.read_messages(session, [stored.db_id]))[0]

    async def test_unavailable_or_changed_evidence_cannot_partially_publish_a_summary(self):
        cases = ('missing_id', 'hidden', 'deleted', 'other_chat', 'old_context',
                 'old_generation', 'revised_source', 'missing_current_revision')
        defaults = self.config.default_session_settings()
        for number, case in enumerate(cases, 1000):
            with self.subTest(case=case):
                session = f'telegram:{number}'
                await self.store.get_or_create_session(session, defaults)
                bad = await self.original(session, 1, 'Earlier evidence before publication.')
                bad_id = bad.db_id
                if case == 'missing_id':
                    bad_id = 999999
                elif case == 'hidden':
                    await self.store.hide_message_ids(session, [bad_id])
                elif case == 'deleted':
                    await self.store.delete_message_ids(session, [bad_id])
                elif case == 'other_chat':
                    other = f'telegram:{number + 1000}'
                    await self.store.get_or_create_session(other, defaults)
                    bad_id = (await self.original(other, 1, 'Evidence from another conversation.')).db_id
                elif case == 'old_context':
                    await self.store.reset_context(session)
                elif case == 'old_generation':
                    await self.store.reset_full(session, defaults)
                elif case == 'revised_source':
                    await self.original(session, 1, 'The original evidence has been corrected.')
                elif case == 'missing_current_revision':
                    # A damaged current-revision pointer must not turn an ID
                    # without original evidence into an acceptable summary source.
                    async with self.store.pool.connection() as conn:
                        await conn.execute('UPDATE messages SET source_revision=2 WHERE id=%s', (bad_id,))
                good = await self.original(session, 2, 'Current evidence must remain uncompacted on failure.')
                scope = await self.store.get_scope(session)
                version = await self.store.get_compaction_version(session)
                with self.assertRaises(StaleScopeError):
                    await self.store.create_memory_block(session, summary_text='Unpublishable mixed evidence.',
                        estimated_tokens=10, source_message_ids=[good.db_id, bad_id], expected_scope=scope,
                        expected_source_revisions={str(good.db_id): 1,
                            str(bad_id): 2 if case == 'missing_current_revision' else 1})
                self.assertEqual(await self.store.get_compaction_version(session), version)
                async with self.store.pool.connection() as conn:
                    blocks = await (await conn.execute('SELECT count(*) AS n FROM memory_blocks WHERE session_id=%s',
                                                       (session,))).fetchone()
                    state = await (await conn.execute('SELECT compacted_by_block_id FROM messages WHERE id=%s',
                                                      (good.db_id,))).fetchone()
                self.assertEqual(blocks['n'], 0)
                self.assertIsNone(state['compacted_by_block_id'])
                saved = await self.store.read_messages(session, [good.db_id])
                self.assertEqual(saved[0].message, good.message)

    async def test_compaction_preserves_mixed_originals_for_reading_and_later_excerpts(self):
        await self.settings()
        message = ConversationMessage.user_text('红色杯子的位置在左边。', metadata={
            'source': 'telegram', 'source_chat_id': '100', 'source_message_id': '1',
            'actor_id': 'telegram:user:101', 'actor_kind': 'user', 'actor_name': 'Participant',
            'custom_evidence': {'quoted': False, 'reference': 'cup-layout'}})
        message.parts.extend([
            MessagePart(PartKind.IMAGE, mime_type='image/png',
                data_b64='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=',
                detail='Original image accompanying the location statement.'),
            MessagePart(PartKind.FILE, filename='notes.txt', mime_type='text/plain',
                detail='Original attached notes reference.', remote_sync=False),
        ])
        first = await self.store.append_message(self.session, message)
        second = await self.original(self.session, 2, 'Later clarification: leave enough room beside the cup.')
        ids = [first.db_id, second.db_id]
        before = await self.store.read_messages(self.session, ids)
        block = await self.store.create_memory_block(self.session,
            summary_text='The cup and its accompanying notes were discussed.', estimated_tokens=15,
            source_message_ids=ids, expected_source_revisions={str(mid): 1 for mid in ids})
        async with self.store.pool.connection() as conn:
            committed = await (await conn.execute('SELECT source_ids,source_revisions FROM memory_blocks WHERE id=%s',
                                                  (block.block_id,))).fetchone()
        self.assertEqual(committed['source_ids'], ids)
        self.assertEqual(committed['source_revisions'], {str(mid): 1 for mid in ids})
        self.assertEqual([row.message for row in await self.store.read_messages(self.session, ids)],
                         [row.message for row in before])
        excerpt = await self.store.create_excerpt(self.session, ids)
        self.assertIn('红色杯子的位置在左边。', excerpt['text'])
        self.assertIn('Later clarification', excerpt['text'])
        self.assertEqual((await self.store.get_excerpt(self.session, excerpt['id']))['text'], excerpt['text'])
