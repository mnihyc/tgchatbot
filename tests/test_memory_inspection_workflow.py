"""Operator recall and compaction inspection using isolated PostgreSQL originals."""
from __future__ import annotations

import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import numpy as np

from tests.business_helpers import BusinessTestCase
from tests.test_memory_image_tools import PIXEL
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind
from tgchatbot.tools.memory import OperationsConfig, audit_records
from tgchatbot.tools.memory_inspection import context_records, export_image, query_memory


class MemoryInspectionWorkflowTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.settings()

    async def original(self, number, text, *, session=None, actor='telegram:user:7', image=False):
        session = session or self.session
        await self.store.get_or_create_session(session, self.config.default_session_settings())
        parts = [MessagePart(PartKind.TEXT, text=text)]
        if image:
            parts.append(MessagePart(PartKind.IMAGE, mime_type='image/png', data_b64=PIXEL))
        return await self.store.append_message(session, ConversationMessage(MessageRole.USER, parts, metadata={
            'source': 'telegram', 'source_chat_id': session.split(':')[-1], 'source_message_id': str(number),
            'actor_id': actor, 'actor_kind': 'user', 'actor_name': 'Participant',
            'sent_at': f'2026-09-01T00:00:{number:02d}+00:00'}))

    async def test_lexical_discovery_read_pagination_and_reset_audit_boundary(self):
        original = await self.original(1, 'Cobalt suitcase with a green handle.')
        other_actor = await self.original(2, 'Cobalt suitcase belongs to someone else.', actor='telegram:user:8')
        foreign = await self.original(1, 'Cobalt private suitcase.', session='telegram:200')
        await self.store.reset_context(self.session)
        with patch('tgchatbot.tools.memory_inspection.EmbeddingClient', side_effect=AssertionError('Unexpected API client')):
            found = await query_memory(self.store, self.config, None, self.session, query='Cobalt',
                actor_id='telegram:user:7', lexical_only=True, after='2026-09-01T07:59:00+08:00',
                before='2026-09-01T08:01:00+08:00')
            self.assertEqual([r['message_ids'] for r in found['matches']], [[original.db_id]])
            self.assertIn('explicitly requested', found['coverage'])
            first = await query_memory(self.store, self.config, None, self.session,
                message_ids=[original.db_id], length=7)
            next_page = await query_memory(self.store, self.config, None, self.session,
                message_ids=[original.db_id], offset=first['messages'][0]['next_offset'])
        self.assertEqual(first['messages'][0]['fragments'][0]['text'] +
            next_page['messages'][0]['fragments'][0]['text'], 'Cobalt suitcase with a green handle.')
        self.assertEqual(first['messages'][0]['speaker']['id'], 'telegram:user:7')
        await self.store.reset_full(self.session, self.config.default_session_settings())
        empty = await query_memory(self.store, self.config, None, self.session, query='Cobalt', lexical_only=True)
        self.assertEqual(empty['matches'], [])
        unavailable = await query_memory(self.store, self.config, None, self.session,
            message_ids=[original.db_id, other_actor.db_id, foreign.db_id])
        self.assertEqual(set(unavailable['unavailable_ids']), {original.db_id, other_actor.db_id, foreign.db_id})
        self.assertEqual(len([row async for row in audit_records(self.store, self.session,
            message_id=original.db_id)]), 1, 'Audit retains the retired original')

    async def test_semantic_operator_query_uses_existing_space_and_closes_client_on_failure(self):
        original = await self.original(1, 'A sapphire travel case.')
        vector = np.zeros(1536, dtype=np.float32)
        vector[0] = 1
        await self.store.create_excerpt(self.session, [original.db_id], embedding=vector, model='inspection-fixture')
        client = SimpleNamespace(enabled=True, space_id='inspection-fixture',
            embed_query=AsyncMock(return_value=vector), aclose=AsyncMock())
        with patch('tgchatbot.tools.memory_inspection.EmbeddingClient', return_value=client):
            found = await query_memory(self.store, self.config, object(), self.session, query='blue luggage')
        self.assertEqual(found['coverage'], 'lexical and semantic')
        self.assertEqual(found['matches'][0]['message_ids'], [original.db_id])
        client.embed_query.assert_awaited_once_with('blue luggage')
        client.aclose.assert_awaited_once()
        client.embed_query.side_effect = RuntimeError('Synthetic transient outage')
        client.aclose.reset_mock()
        with patch('tgchatbot.tools.memory_inspection.EmbeddingClient', return_value=client):
            fallback = await query_memory(self.store, self.config, object(), self.session, query='sapphire')
        self.assertIn('embedding query failed', fallback['coverage'])
        self.assertEqual(fallback['matches'][0]['message_ids'], [original.db_id])
        client.aclose.assert_awaited_once()

    async def test_context_reports_committed_roots_and_streams_all_pages_without_original_bodies(self):
        messages = [await self.original(i, f'Private original {i}') for i in range(1, 5)]
        first = await self.store.create_memory_block(self.session, summary_text='Episode one',
            estimated_tokens=8, source_message_ids=[messages[0].db_id], time_start='2026-09-01T00:00:01+00:00')
        second = await self.store.create_memory_block(self.session, summary_text='Episode two',
            estimated_tokens=9, source_message_ids=[messages[1].db_id])
        digest = await self.store.create_memory_block(self.session, summary_text='Combined digest',
            estimated_tokens=12, source_message_ids=[], parent_block_ids=[first.block_id, second.block_id],
            kind='digest', level=2, time_start='2026-09-01T00:00:01+00:00')
        third = await self.store.create_memory_block(self.session, summary_text='Episode three',
            estimated_tokens=7, source_message_ids=[messages[2].db_id])
        with patch.dict('os.environ', {'DEFAULT_METADATA_TIMEZONE': 'Asia/Singapore'}):
            records = [r async for r in context_records(self.store, self.session, options=OperationsConfig(page_size=1))]
        self.assertEqual((records[0]['visible_messages'], records[0]['uncompacted_messages'],
            records[0]['root_blocks'], records[0]['root_block_estimated_tokens']), (4, 1, 2, 19))
        self.assertEqual(records[0]['first_uncompacted_message_id'], messages[3].db_id)
        self.assertEqual({r['block_id'] for r in records[1:]}, {digest.block_id, third.block_id})
        self.assertEqual(records[1]['time_start'], '2026-09-01T08:00:01+08:00')
        self.assertNotIn('Private original', str(records))
        await self.store.reset_context(self.session)
        reset = [r async for r in context_records(self.store, self.session, options=OperationsConfig())]
        self.assertEqual(len(reset), 1)
        self.assertEqual((reset[0]['visible_messages'], reset[0]['root_blocks']), (0, 0))

    async def test_context_snapshot_stays_consistent_when_reset_occurs_between_records(self):
        original = await self.original(1, 'Archived source')
        block = await self.store.create_memory_block(self.session, summary_text='Archived summary',
            estimated_tokens=8, source_message_ids=[original.db_id])
        records = context_records(self.store, self.session, options=OperationsConfig(page_size=1))
        header = await anext(records)
        self.assertEqual(header['root_blocks'], 1)
        await self.store.reset_context(self.session)
        rest = [r async for r in records]
        self.assertEqual([r['block_id'] for r in rest], [block.block_id])

    async def test_image_discovery_export_and_full_reset_use_source_owned_visibility(self):
        original = await self.original(1, 'Here is the cobalt suitcase.', image=True)
        wrong = await self.original(2, 'Other source')
        await self.store.reset_context(self.session)
        found = await query_memory(self.store, self.config, None, self.session, query='cobalt', lexical_only=True)
        image_id = found['messages'][0]['images'][0]['image_id']
        output = self.path / 'retained.png'
        exported = await export_image(self.store, self.session, message_id=original.db_id,
            image_id=image_id, output=output)
        self.assertEqual(output.read_bytes(), base64.b64decode(PIXEL))
        self.assertEqual(exported['mime_type'], 'image/png')
        with self.assertRaises(FileExistsError):
            await export_image(self.store, self.session, message_id=original.db_id, image_id=image_id, output=output)
        with self.assertRaises(ValueError):
            await export_image(self.store, self.session, message_id=wrong.db_id,
                image_id=image_id, output=self.path / 'wrong.png')
        await self.store.reset_full(self.session, self.config.default_session_settings())
        with self.assertRaises(ValueError):
            await export_image(self.store, self.session, message_id=original.db_id,
                image_id=image_id, output=self.path / 'retired.png')
        self.assertFalse((self.path / 'retired.png').exists())
        self.assertEqual(output.read_bytes(), base64.b64decode(PIXEL))
