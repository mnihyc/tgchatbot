"""Search evidence identity through real memory tools and PostgreSQL.

Only hosted embedding output is controlled. Different people, repeated events,
partial passages and conversational context are not interchangeable evidence.
"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.memory import MemoryService
from tgchatbot.domain.models import ConversationMessage
from tgchatbot.storage.postgres_store import message_body
from tgchatbot.tools.base import ToolContext


def vector(similarity=1.0):
    result = np.zeros(1536, dtype=np.float32)
    result[0], result[1] = similarity, np.sqrt(1 - similarity * similarity)
    return result


class MemoryEvidenceIdentityTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.settings()
        self.embeddings = SimpleNamespace(enabled=True, space_id='evidence-identity-fixture',
            embed_query=AsyncMock(return_value=vector()))
        self.memory = MemoryService(self.store, self.embeddings)
        self.context = ToolContext(self.session, 'Participant')

    async def original(self, text, number, actor=101):
        return await self.store.append_message(self.session, ConversationMessage.user_text(text, metadata={
            'source': 'telegram', 'source_chat_id': '100', 'source_message_id': str(number),
            'actor_id': f'telegram:user:{actor}', 'actor_kind': 'user', 'actor_name': 'Participant',
            'sent_at': f'2026-01-02T03:04:{number:02d}+00:00'}))

    @staticmethod
    def full_span(source):
        return {'message_id': source.db_id, 'start': 0, 'end': len(message_body(source.message))}

    async def index(self, sources, spans=None, similarity=1):
        return await self.store.create_excerpt(self.session, [source.db_id for source in sources],
            spans=spans, embedding=vector(similarity), model=self.embeddings.space_id)

    async def tool(self, name, arguments):
        tool = next(item for item in self.memory.tools if item.name == name)
        result = (await tool.runner.run(arguments, self.context)).output
        self.assertTrue(result['ok'])
        return result

    async def search(self, query):
        return await self.tool('memory_search', {'query': query})

    async def test_full_original_matches_once_and_keeps_semantic_citation_readable(self):
        source = await self.original('The cobalt key is in the kitchen drawer. 钥匙🔑', 1)
        span = self.full_span(source)
        excerpt = await self.index([source], [span])
        result = await self.search('cobalt')
        self.assertEqual(result['matches'], [{'message_ids': [source.db_id]}])
        self.assertEqual(excerpt['spans'], [span])
        self.assertEqual(len(result['messages']), 1)
        self.assertEqual(result['messages'][0]['speaker']['id'], 'person_id:101')
        self.assertEqual(result['messages'][0]['fragments'], [{'offset': 0, 'text': message_body(source.message)}])
        read = await self.tool('memory_read', result['matches'][0])
        self.assertEqual(read['messages'], result['messages'])

    async def test_partial_match_does_not_replace_complete_original_with_its_later_correction(self):
        text = 'The cobalt key is in the drawer. Correction: it is now in the blue bag.'
        source = await self.original(text, 1)
        span = {'message_id': source.db_id, 'start': 0, 'end': len('The cobalt key is in the drawer.')}
        await self.index([source], [span])
        rows = await self.store.search_excerpts(self.session, 'cobalt', embedding=vector(),
            model=self.embeddings.space_id)
        self.assertEqual(len(rows), 2)
        passage = next(row for row in rows if row.get('spans'))
        original = next(row for row in rows if row['kind'] == 'message')
        self.assertNotIn('Correction:', passage['text'])
        self.assertEqual(original['text'], text)
        self.assertEqual(original['source_ids'], passage['source_ids'])
        result = await self.search('cobalt')
        self.assertEqual(result['matches'], [{'message_ids': [source.db_id]}] * 2)
        self.assertEqual(len(result['messages']), 1)
        self.assertEqual(result['messages'][0]['fragments'], [{'offset': 0, 'text': text}])
        self.assertNotIn('partial', result['messages'][0])

    async def test_disjoint_passages_in_one_original_remain_separately_retrievable(self):
        first = '钥匙🔑 used to be in the drawer. '
        unseen = 'Unretrieved interlude. ' * 100
        last = 'It is now in the blue bag. 蓝色包里。'
        text = first + unseen + last
        source = await self.original(text, 1)
        spans = [{'message_id': source.db_id, 'start': start, 'end': end}
                 for start, end in ((0, len(first)), (len(first + unseen), len(text)))]
        for span in spans:
            await self.index([source], [span])
        rows = await self.store.search_excerpts(self.session, 'unmatchedlexicalcontrol',
            embedding=vector(), model=self.embeddings.space_id)
        self.assertEqual(len(rows), 2)
        self.assertCountEqual([row['spans'] for row in rows], [[span] for span in spans])
        result = await self.search('unmatchedlexicalcontrol')
        self.assertEqual(result['matches'], [{'message_ids': [source.db_id]}] * 2)
        self.assertEqual(len(result['messages']), 1)
        record = result['messages'][0]
        self.assertEqual(record['fragments'], [{'offset': 0, 'text': first},
            {'offset': len(first + unseen), 'text': last}])
        self.assertTrue(record['partial'])
        self.assertEqual(record['total_characters'], len(text))
        read = await self.tool('memory_read', {'message_ids': [source.db_id]})
        self.assertEqual(read['messages'][0]['fragments'], [{'offset': 0, 'text': text}])
        self.assertEqual(read['messages'][0]['speaker'], record['speaker'])
        self.assertEqual(read['messages'][0]['sent_at'], record['sent_at'])

    async def test_identical_words_keep_people_and_repeated_events_separate(self):
        sources = [await self.original('I prefer cobalt stationery.', number, actor)
                   for number, actor in ((1, 101), (2, 102), (3, 101))]
        for source in sources:
            await self.index([source], [self.full_span(source)])
        result = await self.search('cobalt')
        rows = result['matches']
        self.assertEqual(len(rows), 3)
        self.assertEqual({tuple(row['message_ids']) for row in rows}, {(source.db_id,) for source in sources})
        self.assertEqual([row['message_id'] for row in result['messages']], [source.db_id for source in sources])
        read = await self.tool('memory_read', {'message_ids': [source.db_id for source in sources]})
        self.assertEqual(read['messages'], result['messages'])
        identities = {row['message_id']: row['speaker']['id'] for row in read['messages']}
        self.assertEqual(identities, {sources[0].db_id: 'person_id:101',
            sources[1].db_id: 'person_id:102', sources[2].db_id: 'person_id:101'})

    async def test_question_answer_context_remains_distinct_from_lexical_question(self):
        question = await self.original('Where did the cobalt key go?', 1)
        answer = await self.original('It moved to the blue bag.', 2, actor=102)
        await self.index([question, answer], [self.full_span(question), self.full_span(answer)])
        result = await self.search('cobalt')
        rows = result['matches']
        self.assertEqual(len(rows), 2)
        self.assertIn({'message_ids': [question.db_id, answer.db_id]}, rows)
        self.assertIn({'message_ids': [question.db_id]}, rows)
        self.assertEqual([row['message_id'] for row in result['messages']], [question.db_id, answer.db_id])
        self.assertEqual([row['fragments'] for row in result['messages']],
            [[{'offset': 0, 'text': message_body(source.message)}] for source in (question, answer)])
        self.assertEqual([row['speaker']['id'] for row in result['messages']], ['person_id:101', 'person_id:102'])

    async def test_duplicate_semantic_representation_cannot_outvote_independent_lexical_support(self):
        repeated = await self.original('The key is in a drawer.', 1)
        supported = await self.original('The cobalt key is in a blue bag.', 2)
        await self.index([repeated])
        await self.index([repeated], [self.full_span(repeated)])
        await self.index([supported], [self.full_span(supported)], similarity=.8)
        rows = (await self.search('cobalt'))['matches']
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['message_ids'], [supported.db_id],
            'Two representations from one channel must not outrank evidence supported by both channels')
        self.assertEqual(rows[1]['message_ids'], [repeated.db_id])

    async def test_edit_and_reset_visibility_stay_owned_by_originals(self):
        source = await self.original('The cobalt key is in the drawer.', 1)
        await self.index([source], [self.full_span(source)])
        await self.store.reset_context(self.session)
        self.assertEqual(len((await self.search('cobalt'))['matches']), 1)
        revised = await self.original('Correction: the cobalt key is in the blue bag.', 1)
        self.assertEqual(source.db_id, revised.db_id)
        rows = await self.store.search_excerpts(self.session, 'cobalt', embedding=vector(),
            model=self.embeddings.space_id)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['kind'], 'message', 'An old vector cannot describe a revised original')
        self.assertEqual(rows[0]['source_revision'], 2)
        result = await self.search('cobalt')
        self.assertEqual(result['matches'], [{'message_ids': [source.db_id]}])
        self.assertEqual(result['messages'][0]['fragments'],
            [{'offset': 0, 'text': message_body(revised.message)}])
        await self.store.reset_full(self.session, self.config.default_session_settings())
        result = await self.search('cobalt')
        self.assertEqual(result['matches'], [])
        self.assertEqual(result['messages'], [])
        read = await self.tool('memory_read', {'message_ids': [source.db_id]})
        self.assertEqual(read['messages'], [])
        self.assertEqual(read['unavailable_ids'], [source.db_id])
