"""Recall keeps each original once, with canonical offsets and typed ownership.

PostgreSQL and memory tools are real; only hosted query embeddings are mocked.
Small display allowances expose duplicate-budget loss without a scale experiment.
"""
from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.memory import MemoryService
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind
from tgchatbot.storage.postgres_store import message_body
from tgchatbot.tools.base import ToolContext


def vector(similarity=1.0):
    result = np.zeros(1536, dtype=np.float32)
    result[0], result[1] = similarity, np.sqrt(1 - similarity * similarity)
    return result


class MemoryProjectionBoundaryTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.settings()
        self.embeddings = SimpleNamespace(enabled=False, space_id='projection-boundary-fixture',
            embed_query=AsyncMock(return_value=vector()))
        self.memory = MemoryService(self.store, self.embeddings)
        self.context = ToolContext(self.session, 'Participant')

    async def original(self, number, text=None, *, parts=None):
        return await self.store.append_message(self.session, ConversationMessage(MessageRole.USER,
            parts=parts if parts is not None else [MessagePart(PartKind.TEXT, text=text)], metadata={
                'source': 'telegram', 'source_chat_id': '100', 'source_message_id': str(number),
                'actor_id': 'telegram:user:101', 'actor_kind': 'user', 'actor_name': 'Participant',
                'sent_at': f'2026-01-02T03:04:{number:02d}+00:00'}))

    async def tool(self, name, arguments, *, memory=None):
        spec = next(tool for tool in (memory or self.memory).tools if tool.name == name)
        output = (await spec.runner.run(arguments, self.context)).output
        self.assertTrue(output['ok'], output)
        return output

    async def test_repeated_read_ids_return_each_original_once_in_requested_order(self):
        first = await self.original(1, 'The key is in the blue bag.')
        second = await self.original(2, 'I moved the bag to the hall.')
        result = await self.tool('memory_read', {'message_ids': [second.db_id, first.db_id, second.db_id]})
        self.assertEqual([row['message_id'] for row in result['messages']], [second.db_id, first.db_id])
        self.assertEqual([row['fragments'] for row in result['messages']],
            [[{'offset': 0, 'text': message_body(source.message)}] for source in (second, first)])
        self.assertEqual(result['unavailable_ids'], [])
        self.assertEqual(result['omitted_ids'], [])

    async def test_overlapping_matches_spend_budget_only_on_new_original_characters(self):
        first = await self.original(1, 'ABCDEFGHI')
        second = await self.original(2, 'XYZ')
        self.embeddings.enabled = True
        for source, start, end, similarity in (
                (first, 0, 6, 1.0), (first, 3, 9, .9), (second, 0, 3, .8)):
            await self.store.create_excerpt(self.session, [source.db_id],
                spans=[{'message_id': source.db_id, 'start': start, 'end': end}],
                embedding=vector(similarity), model=self.embeddings.space_id)
        self.memory.config = replace(self.memory.config, response_chars=10, search_result_chars=10)
        result = await self.tool('memory_search', {'query': 'unmatchedlexicalcontrol'})
        self.assertEqual([row['message_ids'] for row in result['matches']],
            [[first.db_id], [first.db_id], [second.db_id]])
        self.assertEqual([row['fragments'] for row in result['messages']],
            [[{'offset': 0, 'text': 'ABCDEFGHI'}], [{'offset': 0, 'text': 'X'}]])
        self.assertNotIn('partial', result['messages'][0])
        self.assertTrue(result['messages'][1]['partial'])
        self.assertEqual(result['messages'][1]['total_characters'], 3)
        self.assertTrue(result['matches'][-1]['truncated'])

    async def test_typed_notes_are_annotations_but_literal_user_headers_remain_exact(self):
        generated = '[Message metadata: username=@participant nickname="Participant" time=2026-01-02T11:04:01+08:00]'
        operation = '[Attachment download failed: TimeoutError]'
        literal = 'cobalt [Message metadata: I wrote this literal example] is part of my message. 🔑'
        source = await self.original(1, parts=[
            MessagePart(PartKind.TEXT, text=generated, origin='provenance'),
            MessagePart(PartKind.TEXT, text=operation, origin='auto_note'),
            MessagePart(PartKind.TEXT, text=literal),
        ])
        body = '\n'.join((generated, operation, literal))
        literal_offset = len(generated) + 1 + len(operation) + 1
        read = await self.tool('memory_read', {'message_ids': [source.db_id]})
        search = await self.tool('memory_search', {'query': 'cobalt'})
        for result in (read, search):
            record = result['messages'][0]
            self.assertEqual(record['fragments'], [{'offset': literal_offset, 'text': literal}])
            self.assertIn(operation, json.dumps(record.get('annotations'), ensure_ascii=False))
            self.assertNotIn(generated, json.dumps(record, ensure_ascii=False))
            self.assertNotIn('partial', record, 'Selecting the full canonical body is a complete read despite omitted provenance')
            self.assertNotIn('next_offset', record)
            self.assertEqual(record['speaker']['id'], 'person_id:101')
        self.assertEqual(read['messages'], search['messages'])
        reopened = await self.new_store()
        original = (await reopened.read_messages(self.session, [source.db_id]))[0]
        self.assertEqual(message_body(original.message), body)
        self.assertEqual([part.origin for part in original.message.parts], ['provenance', 'auto_note', None])
        cold = MemoryService(reopened, self.embeddings, config=self.memory.config)
        self.assertEqual(await self.tool('memory_read', {'message_ids': [source.db_id]}, memory=cold), read)
        self.assertEqual(await self.tool('memory_search', {'query': 'cobalt'}, memory=cold), search)

        offset, length = len(generated) + 1, len(operation) + 1 + 6
        page = (await self.tool('memory_read', {'message_ids': [source.db_id],
            'offset': offset, 'length': length}))['messages'][0]
        self.assertEqual(page['fragments'], [{'offset': literal_offset, 'text': literal[:6]}])
        self.assertIn(operation, json.dumps(page.get('annotations'), ensure_ascii=False))
        self.assertEqual(page['next_offset'], offset + length)
        self.assertEqual(page['total_characters'], len(body))
        self.assertTrue(page['partial'])
