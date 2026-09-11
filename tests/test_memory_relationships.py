"""Retrieving an old conversation preserves chronological and source boundaries."""
from __future__ import annotations

import os
import unittest
import uuid

from psycopg import sql

from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind, SessionSettings
from tgchatbot.storage.postgres_store import PostgresStore, StaleScopeError
from tgchatbot.storage.relationships import expand_message_ids


@unittest.skipUnless(os.environ.get('TEST_DATABASE_URL'), 'requires disposable PostgreSQL/pgvector')
class ConversationRelationships(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.schema = 'test_relationships_' + uuid.uuid4().hex
        self.store = PostgresStore(os.environ['TEST_DATABASE_URL'], schema=self.schema)
        await self.store.initialize()
        self.session = 'telegram:-10042'
        await self.store.get_or_create_session(self.session, SessionSettings())

    async def asyncTearDown(self):
        async with self.store.pool.connection() as conn:
            await conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))
        await self.store.close()

    async def append(self, number, *, minute=0, topic='7', session=None, **metadata):
        session = session or self.session
        await self.store.get_or_create_session(session, SessionSettings())
        return await self.store.append_message(session, ConversationMessage.user_text(f'original {number}', metadata={
            'source': 'telegram', 'source_chat_id': session.removeprefix('telegram:'),
            'source_message_id': str(number), 'actor_id': 'telegram:user:11', 'actor_kind': 'user',
            'actor_name': 'Alex', 'topic_id': topic, 'sent_at': f'2025-01-01T00:{minute:02}:00Z', **metadata}))

    async def test_imported_history_uses_original_time_with_same_topic_neighbors(self):
        latest = await self.append(50, minute=50)
        other_topic = await self.append(31, minute=31, topic='8')
        before = await self.append(10, minute=10)
        seed = await self.append(30, minute=30)
        after = await self.append(40, minute=40)
        second_before = await self.append(20, minute=20)
        far = await self.append(1, minute=1)
        ids = await expand_message_ids(self.store, self.session, [seed.db_id])
        self.assertEqual(ids, [seed.db_id, second_before.db_id, before.db_id, after.db_id, latest.db_id])
        self.assertNotIn(other_topic.db_id, ids)
        self.assertNotIn(far.db_id, ids)

    async def test_reply_can_cross_topic_but_never_chat_or_full_reset(self):
        reply = await self.append(1, topic='8')
        foreign = await self.append(2, topic='8', session='telegram:-10043')
        seed = await self.append(3, reply_to_source_id='1')
        self.assertEqual(await expand_message_ids(self.store, self.session, [seed.db_id], neighbors=0), [seed.db_id, reply.db_id])
        external = await self.append(4, reply_to_source_id='1', reply_to_source_chat_id='-10043')
        self.assertEqual(await expand_message_ids(self.store, self.session, [external.db_id, foreign.db_id], neighbors=0), [external.db_id])
        scope = await self.store.get_scope(self.session)
        await self.store.reset_full(self.session, SessionSettings())
        fresh = await self.append(5, reply_to_source_id='1')
        self.assertEqual(await expand_message_ids(self.store, self.session, [fresh.db_id, seed.db_id]), [fresh.db_id])
        with self.assertRaises(StaleScopeError):
            await expand_message_ids(self.store, self.session, [fresh.db_id], expected_scope=scope)

    async def test_soft_reset_keeps_active_neighbors_while_undo_evidence_is_excluded(self):
        keep = await self.append(1, minute=1)
        abandoned = await self.append(2, minute=2)
        await self.store.hide_message_ids(self.session, [abandoned.db_id])
        await self.store.reset_context(self.session)
        seed = await self.append(3, minute=3, reply_to_source_id='2')
        self.assertEqual(await expand_message_ids(self.store, self.session, [seed.db_id]), [seed.db_id, keep.db_id])

    async def test_total_limit_and_tied_timestamps_are_deterministic(self):
        sources = [await self.append(number) for number in range(1, 26)]
        seed = sources[12]
        ids = await expand_message_ids(self.store, self.session, [seed.db_id])
        self.assertEqual(ids, [seed.db_id, sources[11].db_id, sources[10].db_id, sources[13].db_id, sources[14].db_id])
        selected = [source.db_id for source in sources[2:20]]
        result = await expand_message_ids(self.store, self.session, selected)
        self.assertEqual(result[:18], selected)
        self.assertEqual(len(result), 20)
        self.assertEqual(len(set(result)), 20)

    async def test_reply_to_later_delivered_answer_chunk_keeps_original_after_soft_reset(self):
        answer = await self.store.append_message(self.session, ConversationMessage(
            MessageRole.ASSISTANT, [MessagePart(PartKind.TEXT, text='One complete original answer across Telegram chunks.')],
            metadata={'source': 'agent', 'actor_id': 'agent', 'actor_kind': 'bot',
                      'topic_id': '8', 'sent_at': '2025-01-01T00:01:00Z'}))
        await self.store.bind_message_source(self.session, answer.db_id, source='telegram',
            source_chat_id='-10042', source_message_ids=['101', '102', '103'],
            actor_id='telegram:user:900', actor_kind='bot', actor_name='Chat bot')
        await self.store.reset_context(self.session)
        primary = await self.append(104, minute=4, reply_to_source_id='101')
        last_chunk = await self.append(105, minute=5, reply_to_source_id='103')
        for reply in (primary, last_chunk):
            self.assertEqual(await expand_message_ids(self.store, self.session, [reply.db_id], neighbors=0),
                             [reply.db_id, answer.db_id])
        foreign = await self.append(106, reply_to_source_id='103', reply_to_source_chat_id='-10043')
        self.assertEqual(await expand_message_ids(self.store, self.session, [foreign.db_id], neighbors=0), [foreign.db_id])
        await self.store.reset_full(self.session, SessionSettings())
        fresh = await self.append(107, reply_to_source_id='103')
        self.assertEqual(await expand_message_ids(self.store, self.session, [fresh.db_id], neighbors=0), [fresh.db_id])

    async def test_abandoned_delivered_answer_cannot_return_through_chunk_alias(self):
        answer = await self.store.append_message(self.session, ConversationMessage(
            MessageRole.ASSISTANT, [MessagePart(PartKind.TEXT, text='An answer later abandoned by undo.')],
            metadata={'source': 'agent', 'actor_id': 'agent', 'actor_kind': 'bot'}))
        await self.store.bind_message_source(self.session, answer.db_id, source='telegram',
            source_chat_id='-10042', source_message_ids=['101', '102'],
            actor_id='telegram:user:900', actor_kind='bot', actor_name='Chat bot')
        await self.store.hide_message_ids(self.session, [answer.db_id])
        reply = await self.append(103, minute=3, reply_to_source_id='102')
        self.assertEqual(await expand_message_ids(self.store, self.session, [reply.db_id], neighbors=0), [reply.db_id])


if __name__ == '__main__':
    unittest.main()
