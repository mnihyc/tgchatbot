"""Layered summaries retain the exact original revisions their parents used."""
from __future__ import annotations

import json
from unittest.mock import patch

from tests.business_helpers import BusinessTestCase
from tgchatbot.domain.models import ConversationMessage, ProviderResponse
from tgchatbot.storage.postgres_store import StaleScopeError


class CompactionSourceInheritanceTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.settings()

    async def original(self, number, text):
        return await self.store.append_message(self.session, ConversationMessage.user_text(text, metadata={
            'source': 'telegram', 'source_chat_id': '100', 'source_message_id': str(number),
            'actor_id': 'telegram:user:101', 'actor_kind': 'user', 'actor_name': 'Participant'}))

    async def earlier_toolspan(self):
        source = await self.original(1, 'The earlier lookup established that the cobalt key is in the drawer.')
        parent = await self.store.create_memory_block(self.session, summary_text='Earlier lookup: key in drawer.',
            estimated_tokens=20, source_message_ids=[source.db_id], level=0, kind='toolspan',
            expected_source_revisions={str(source.db_id): 1})
        return source, parent

    async def promote(self, parent, raw, expected):
        return await self.store.replace_memory_blocks(self.session, block_ids=[parent.block_id],
            parent_block_ids=[parent.block_id], source_message_ids=[item.db_id for item in raw],
            summary_text='Earlier lookup and later discussion form one episode.', estimated_tokens=30,
            source_message_count=1 + len(raw), start_message_id=None, end_message_id=None,
            level=1, kind='episode', source_kind='mixed' if raw else 'blocks',
            expected_source_revisions=expected)

    async def test_raw_discussion_and_prior_toolspan_share_one_episode_without_losing_originals(self):
        old, parent = await self.earlier_toolspan()
        raw = await self.original(2, 'The new discussion asks how to return that key.')
        episode = await self.promote(parent, [raw], {str(raw.db_id): 1})
        async with self.store.pool.connection() as conn:
            row = await (await conn.execute('SELECT * FROM memory_blocks WHERE id=%s', (episode.block_id,))).fetchone()
            retired = await (await conn.execute('SELECT valid,superseded_by FROM memory_blocks WHERE id=%s', (parent.block_id,))).fetchone()
        self.assertEqual(row['source_ids'], [old.db_id, raw.db_id])
        self.assertEqual(row['source_revisions'], {str(old.db_id): 1, str(raw.db_id): 1})
        self.assertEqual(retired, {'valid': False, 'superseded_by': episode.block_id})
        originals = await self.store.read_messages(self.session, [old.db_id, raw.db_id])
        self.assertEqual([item.message for item in originals], [old.message, raw.message])

    async def test_parent_only_promotion_accepts_empty_raw_revision_map(self):
        old, parent = await self.earlier_toolspan()
        episode = await self.promote(parent, [], {})
        self.assertEqual(episode.parent_block_ids, (parent.block_id,))
        originals = await self.store.read_messages(self.session, [old.db_id])
        self.assertEqual(originals[0].message, old.message)

    async def test_real_parent_source_edit_rejects_summary_without_publishing_a_partial_episode(self):
        old, parent = await self.earlier_toolspan()
        raw = await self.original(2, 'A later request depends on the old lookup.')
        revised = await self.original(1, 'Correction: the key moved to the blue bag.')
        before = await self.store.get_compaction_version(self.session)
        with self.assertRaises(StaleScopeError):
            await self.promote(parent, [raw], {str(raw.db_id): 1})
        self.assertEqual(await self.store.get_compaction_version(self.session), before)
        async with self.store.pool.connection() as conn:
            count = (await (await conn.execute('SELECT count(*) AS n FROM memory_blocks')).fetchone())['n']
        self.assertEqual(count, 1)
        self.assertEqual((await self.store.read_messages(self.session, [old.db_id]))[0].message, revised.message)
        self.assertEqual([row.db_id for row in await self.store.list_uncompacted_messages(self.session)], [old.db_id, raw.db_id])

    async def test_conflicting_raw_revision_cannot_be_overwritten_by_parent_revision(self):
        old, _old_parent = await self.earlier_toolspan()
        raw = await self.original(2, 'The continuation depends on the earlier source.')
        await self.original(1, 'Correction: the key is now in the blue bag.')
        parent = await self.store.create_memory_block(self.session,
            summary_text='Updated lookup: key in blue bag.', estimated_tokens=20,
            source_message_ids=[old.db_id], level=0, kind='toolspan',
            expected_source_revisions={str(old.db_id): 2})
        before = await self.store.get_compaction_version(self.session)
        # The raw slice was captured before the correction, while its parent
        # reflects the new evidence. Neither observation can overwrite the other.
        with self.assertRaises(StaleScopeError):
            await self.promote(parent, [old, raw], {str(old.db_id): 1, str(raw.db_id): 1})
        self.assertEqual(await self.store.get_compaction_version(self.session), before)
        self.assertEqual([block.block_id for block in await self.store.list_memory_blocks(self.session)], [parent.block_id])
        episode = await self.promote(parent, [old, raw], {old.db_id: 2, raw.db_id: 1})
        self.assertEqual(episode.kind, 'episode')

    async def test_callers_without_revision_expectations_keep_existing_append_semantics(self):
        old, parent = await self.earlier_toolspan()
        raw = await self.original(2, 'A caller can omit an optional revision expectation.')
        episode = await self.promote(parent, [raw], None)
        self.assertEqual(episode.source_message_count, 2)
        self.assertEqual(len(await self.store.read_messages(self.session, [old.db_id, raw.db_id])), 2)

    async def test_reply_continues_after_promoting_toolspan_with_recent_raw_discussion(self):
        await self.settings(compact_trigger_tokens=2000, compact_target_tokens=1400,
            compact_batch_tokens=1800, compact_min_messages=2, min_raw_messages_reserve=1,
            compact_keep_recent_ratio=.1)
        old, parent = await self.earlier_toolspan()
        originals = [old]
        for number in range(2, 9):
            originals.append(await self.original(number,
                f'Discussion {number}: ' + 'We considered where to leave the key after returning home. ' * 20))
        requests = []

        async def model(**request):
            requests.append(request)
            if not request.get('response_schema'):
                return ProviderResponse(final_text='You can return the key after the earlier lookup.')
            fields = request['response_schema']['properties']
            result = {key: [] for key in fields}
            result['scope'] = 'Earlier lookup and subsequent discussion about the key.'
            if 'interaction_mode' in fields:
                result['interaction_mode'] = 'chat_or_sharing'
            if 'interaction_modes_seen' in fields:
                result['interaction_modes_seen'] = ['chat_or_sharing']
            return ProviderResponse(final_text=json.dumps(result))

        with patch.object(self.provider, 'generate', side_effect=model):
            reply = await self.runtime.run_turn(session_id=self.session, user_display_name='Participant',
                incoming_message=ConversationMessage.user_text('helper, what did we decide about returning the key?'))
        self.assertEqual(reply.text, 'You can return the key after the earlier lookup.')
        self.assertTrue(any(request.get('response_schema_name') == 'episode_memory_block' for request in requests))
        async with self.store.pool.connection() as conn:
            row = await (await conn.execute('SELECT valid,superseded_by FROM memory_blocks WHERE id=%s', (parent.block_id,))).fetchone()
            promoted = await (await conn.execute('SELECT source_ids,source_revisions,details FROM memory_blocks WHERE id=%s', (row['superseded_by'],))).fetchone()
        self.assertFalse(row['valid'])
        self.assertEqual(promoted['details']['kind'], 'episode')
        self.assertIn(old.db_id, promoted['source_ids'])
        self.assertGreater(len(promoted['source_ids']), 1)
        self.assertTrue(all(revision == 1 for revision in promoted['source_revisions'].values()))
        stored = await self.store.read_messages(self.session, [item.db_id for item in originals])
        self.assertEqual([item.message for item in stored], [item.message for item in originals])
