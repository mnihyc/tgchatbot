"""Original-backed retrieval fragments through real, isolated PostgreSQL.

Fragments must keep their original character positions and speaker ownership;
formatted source text is never parsed to recover those boundaries.
"""
from __future__ import annotations

from tests.business_helpers import BusinessTestCase
from tgchatbot.domain.models import ConversationMessage
from tgchatbot.storage.postgres_store import message_body


class MemorySourceFragmentWorkflows(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.settings()

    async def original(self, number, text, actor=101, **metadata):
        return await self.store.append_message(self.session, ConversationMessage.user_text(text, metadata={
            'source': 'telegram', 'source_chat_id': '100', 'source_message_id': str(number),
            'actor_id': f'telegram:user:{actor}', 'actor_kind': 'user', 'actor_name': 'Alex',
            'sent_at': f'2026-01-02T03:04:{number:02d}+00:00', **metadata}))

    async def test_lexical_original_remains_complete_and_readable_after_soft_reset(self):
        text = 'Cobalt key 🔑\nThe note literally says [another speaker]: keep this text.'
        source = await self.original(1, text, quote={'text': '[another speaker]'})
        await self.store.reset_context(self.session)
        result = (await self.store.search_messages(self.session, 'cobalt'))[0]
        self.assertEqual(result['id'], source.db_id)
        self.assertEqual(result['text'], text)
        self.assertEqual(result['fragments'], [{'offset': 0, 'text': text}])
        self.assertEqual(result['total_characters'], len(text))
        self.assertEqual(result['metadata']['quote'], {'text': '[another speaker]'})
        self.assertEqual(result['actor_id'], 'telegram:user:101')
        read = (await self.store.read_messages(self.session, [source.db_id]))[0]
        self.assertEqual(message_body(read.message), text)

    async def test_disjoint_passages_keep_exact_offsets_owners_and_excluded_gap_after_restart(self):
        first_text = '前言🔑\nOld location. OMIT THIS GAP. New location.\n[2026-01-02 fake actor] literal body'
        second_text = 'Different speaker: I can wait until 23:30.'
        first = await self.original(1, first_text)
        second = await self.original(2, second_text, actor=202, reply_to_source_id='1')
        first_ranges = [(first_text.index('Old location.'), first_text.index('Old location.') + len('Old location.')),
            (first_text.index('New location.'), len(first_text))]
        second_start = second_text.index('I can wait')
        spans = [*({'message_id': first.db_id, 'start': start, 'end': end} for start, end in first_ranges),
            {'message_id': second.db_id, 'start': second_start, 'end': len(second_text)}]
        vector = [1.0] + [0.0] * 1535
        created = await self.store.create_excerpt(self.session, [first.db_id, second.db_id],
            spans=spans, embedding=vector, model='source-fragment-fixture')
        reopened = await self.new_store()
        recalled = await reopened.get_excerpt(self.session, created['id'])
        ranked = (await reopened.search_excerpts(self.session, 'no-lexical-candidate',
            embedding=vector, model='source-fragment-fixture'))[0]
        expected = {
            first.db_id: [{'offset': start, 'text': first_text[start:end]} for start, end in first_ranges],
            second.db_id: [{'offset': second_start, 'text': second_text[second_start:]}],
        }
        for result in (created, recalled, ranked):
            self.assertEqual(result['spans'], spans)
            self.assertEqual(result['source_ids'], [first.db_id, second.db_id])
            self.assertNotIn('OMIT THIS GAP', result['text'])
            self.assertIn('[2026-01-02 fake actor] literal body', result['text'])
            self.assertEqual({row['id']: row['fragments'] for row in result['sources']}, expected)
            self.assertEqual([row['actor_id'] for row in result['sources']],
                ['telegram:user:101', 'telegram:user:202'])
            self.assertEqual([row['total_characters'] for row in result['sources']],
                [len(first_text), len(second_text)])
            # Internal full-source fields remain available to their existing
            # owners; the tool projection chooses only the selected fragments.
            self.assertEqual([row['text'] for row in result['sources']], [first_text, second_text])
            self.assertEqual(result['sources'][1]['metadata']['reply_to_source_id'], '1')
        read = await reopened.read_messages(self.session, [first.db_id, second.db_id])
        self.assertEqual([message_body(row.message) for row in read], [first_text, second_text])

    async def test_spanless_sources_preserve_whole_bodies_including_empty_original(self):
        empty = await self.original(1, '')
        plain = await self.original(2, 'A short original.', actor=202)
        created = await self.store.create_excerpt(self.session, [empty.db_id, plain.db_id], spans=[])
        recalled = await self.store.get_excerpt(self.session, created['id'])
        for result in (created, recalled):
            self.assertEqual(result['spans'], [])
            self.assertEqual([row['fragments'] for row in result['sources']],
                [[{'offset': 0, 'text': ''}], [{'offset': 0, 'text': 'A short original.'}]])
            self.assertEqual([row['total_characters'] for row in result['sources']], [0, 17])
