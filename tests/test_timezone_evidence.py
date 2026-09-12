"""UTC originals and configured-zone evidence across memory/profile boundaries."""
from __future__ import annotations

import base64
from dataclasses import replace
from datetime import datetime, timezone
import io
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.memory import MemoryService
from tgchatbot.core.memory_worker import MemoryWorker
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import ChatMode, ConversationMessage, MessagePart, PartKind, ProviderResponse, ToolCall, ToolHistoryMode
from tgchatbot.domain.provenance import attributed_message
from tgchatbot.tools.base import ToolContext


class TimezoneEvidenceWorkflows(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.memory = MemoryService(self.store, SimpleNamespace(enabled=False), config=self.config.memory)

    async def source(self, number, at, text, *, image=False, **metadata):
        message = ConversationMessage.user_text(text, metadata={
            'source': 'telegram', 'source_chat_id': '100', 'source_message_id': str(number),
            'actor_id': 'telegram:user:11', 'actor_name': 'Participant', 'actor_kind': 'user',
            'sent_at': at, **metadata})
        if image:
            buffer = io.BytesIO()
            with Image.new('RGB', (4, 4), 'blue') as picture:
                picture.save(buffer, format='PNG')
            message.parts.append(MessagePart(PartKind.IMAGE, mime_type='image/png',
                data_b64=base64.b64encode(buffer.getvalue()).decode()))
        return await self.runtime.ingest_user_message(session_id=self.session, incoming_message=message)

    async def tool(self, name, args, zone):
        spec = next(spec for spec in self.memory.tools if spec.name == name)
        return await spec.runner.run(args, ToolContext(self.session, 'Participant', timezone=zone))

    async def test_midnight_search_read_image_and_profile_dates_agree_while_originals_stay_utc(self):
        at = '2026-09-12T16:00:20+00:00'
        literal = 'I prefer cobalt pens. The literal log says 2026-09-12T16:00:20+00:00.'
        original = await self.source(1, at, literal, image=True, edited_at='2026-09-12T16:01:00Z',
            forward_origin={'date': 1789228790, 'type': 'hidden_user', 'sender_user_name': 'Someone'},
            external_reply={'origin': {'date': '2026-09-12T15:59:00Z', 'type': 'hidden_user'}},
            quote={'text': 'Keep 2026-09-12T16:00:20+00:00 exactly.'})
        before = (await self.store.read_messages(self.session, [original.db_id]))[0].message
        await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:11',
            asserted_by='telegram:user:11', claim=literal, source_ids=[original.db_id],
            valid_from='2026-09-12T15:59:00Z')
        zone = 'Asia/Singapore'
        search = await self.tool('memory_search', {'query': 'cobalt',
            'after': '2026-09-13T00:00:00', 'before': '2026-09-13T00:01:00'}, zone)
        self.assertTrue(search.output['ok'], search.output)
        self.assertEqual(search.output['messages'][0]['sent_at'], '2026-09-13T00:00:20+08:00')
        image_id = search.output['messages'][0]['images'][0]['image_id']
        read = await self.tool('memory_read', {'message_ids': [original.db_id], 'image_ids': [image_id]}, zone)
        evidence = read.output['messages'][0]
        self.assertEqual(evidence['sent_at'], search.output['messages'][0]['sent_at'])
        self.assertEqual(evidence['edited_at'], '2026-09-13T00:01:00+08:00')
        self.assertEqual(evidence['external_reply']['origin']['date'], '2026-09-12T23:59:00+08:00')
        self.assertTrue(evidence['forward_origin']['date'].endswith('+08:00'))
        self.assertEqual(evidence['fragments'][0]['text'], literal)
        self.assertEqual(evidence['quote'], before.metadata['quote'])
        label = next(part.text for part in read.evidence_parts if part.kind == PartKind.TEXT)
        image_label = json.loads(label.removeprefix('[Original image evidence: ').removesuffix(']'))
        self.assertEqual(image_label['sent_at'], evidence['sent_at'])
        profile = (await self.tool('user_profile_fetch', {'actor_ids': ['telegram:user:11']}, zone)).output
        self.assertTrue(profile['as_of'].endswith('+08:00'))
        self.assertTrue(profile['fetched_at'].endswith('+08:00'))
        document = profile['profiles'][0]
        self.assertEqual(document['identity']['last_message']['sent_at'], evidence['sent_at'])
        self.assertEqual(document['facts'][0]['valid_from'], '2026-09-12T23:59:00+08:00')
        self.assertEqual(document['facts'][0]['claim'], literal)
        projected = attributed_message(before, message_id=original.db_id, timezone=zone)
        self.assertIn('2026-09-13T00:00:20+08:00', projected.parts[0].text)
        self.assertEqual(before.parts[0].text, literal)
        after = (await self.store.read_messages(self.session, [original.db_id]))[0].message
        self.assertEqual(after, before)
        self.assertEqual(after.metadata['sent_at'], at)
        snapshot = await self.store.fetch_profile_snapshot(self.session, ['telegram:user:11'],
            max_bytes=self.config.memory.profile_bytes, for_learning=True)
        fact = snapshot['profiles'][0]['facts'][0]
        self.assertEqual(fact['source_dates']['first'].utcoffset().total_seconds(), 0)
        self.assertEqual(fact['valid_from'].utcoffset().total_seconds(), 0)

    async def test_configured_dst_zone_keeps_chronological_search_order_and_learning_source_dates(self):
        # The clock repeats: 01:10 EST follows 01:50 EDT on this date.
        first = await self.source(1, '2025-11-02T05:50:00Z', 'I prefer cobalt fountain pens.')
        second = await self.source(2, '2025-11-02T06:10:00Z', 'I collect cobalt pens too.')
        fact = await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:11',
            asserted_by='telegram:user:11', claim='Prefers and collects cobalt pens',
            source_ids=[first.db_id, second.db_id])
        worker = MemoryWorker(store=self.store, embeddings=SimpleNamespace(enabled=False),
            providers={'openai': self.provider},
            config=replace(self.config, default_metadata_timezone='America/New_York'))
        self.provider.responses = [ProviderResponse(final_text=json.dumps({'additions': [], 'removals': []}))]
        with patch.dict(os.environ, {'DEFAULT_METADATA_TIMEZONE': 'America/New_York'}):
            search = await self.memory.search(self.session, 'cobalt')
            read = await self.memory.read(self.session, [first.db_id, second.db_id])
            profile = await self.memory.fetch_profiles(self.session, ['telegram:user:11'])
            await worker.refresh_profiles(self.session, ['telegram:user:11'])
        expected = ['2025-11-02T01:50:00-04:00', '2025-11-02T01:10:00-05:00']
        self.assertEqual([item['message_id'] for item in search['messages']], [first.db_id, second.db_id])
        self.assertEqual([item['sent_at'] for item in search['messages']], expected)
        self.assertEqual([item['sent_at'] for item in read['messages']], expected)
        self.assertEqual(profile['profiles'][0]['identity']['last_message']['sent_at'], expected[-1])
        request = json.loads(self.provider.requests[-1]['messages'][0].parts[0].text)
        self.assertEqual([item['sent_at'] for item in request['original_evidence']], expected)
        supplied = next(document for document in request['current_profiles']
            if document['actor_id'] == 'telegram:user:11')['facts'][0]
        self.assertEqual(supplied['source_dates'], {'first': expected[0], 'last': expected[1]})
        self.assertEqual(supplied['id'], fact['id'])
        self.assertEqual(supplied['valid_from'], None)
        stored = await self.store.read_messages(self.session, [first.db_id, second.db_id])
        self.assertEqual([datetime.fromisoformat(row.message.metadata['sent_at']).utcoffset()
            for row in stored], [timezone.utc.utcoffset(None)] * 2)

    async def test_runtime_stores_tool_times_in_utc_and_projects_them_for_the_next_model_request(self):
        source = await self.source(1, '2026-09-12T16:00:20Z',
            'The cobalt note literally says 2026-09-12T16:00:20Z.')
        await self.settings(mode=ChatMode.ASSIST, tool_history_mode=ToolHistoryMode.NATIVE_SAME_PROVIDER)
        runtime = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools,
            providers={'openai': self.provider}, memory=self.memory, preview_cache=self.preview_cache)
        self.provider.responses = [ProviderResponse(tool_calls=[ToolCall('memory_search', 'recall', {'query': 'cobalt'})]),
            ProviderResponse(final_text='The note was posted just after midnight.')]
        await runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Recall the cobalt note.'))
        results = [row.message for row in await self.store.list_canonical_messages(self.session)
            if row.message.name == 'memory_search' and row.message.metadata.get('tool_phase') == 'result']
        self.assertEqual(len(results), 1)
        stored = next(record for record in results[0].metadata['tool_payload']['output']['messages']
            if record['message_id'] == source.db_id)
        self.assertEqual(stored['sent_at'], '2026-09-12T16:00:20+00:00')
        immediate = next(item['output'] for item in self.provider.requests[-1]['extra_input_items']
            if item.get('type') == 'function_call_output' and item['call_id'] == 'recall')
        shown = next(record for record in immediate['messages'] if record['message_id'] == source.db_id)
        self.assertEqual(shown['sent_at'], '2026-09-13T00:00:20+08:00')
        runtime.invalidate_session(self.session)
        self.provider.responses = [ProviderResponse(final_text='The same note was posted just after midnight.')]
        await runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Confirm the earlier time.'))
        model_results = [message for message in self.provider.requests[-1]['messages']
            if message.name == 'memory_search' and message.metadata.get('tool_phase') == 'result']
        shown = next(record for record in model_results[0].metadata['tool_payload']['output']['messages']
            if record['message_id'] == source.db_id)
        self.assertEqual(shown['sent_at'], '2026-09-13T00:00:20+08:00')
        self.assertEqual(stored['fragments'], shown['fragments'])
        self.assertIn('2026-09-12T16:00:20Z', shown['fragments'][0]['text'])
        await self.settings(tool_history_mode=ToolHistoryMode.TRANSLATED)
        self.provider.responses = [ProviderResponse(final_text='The rendered evidence has the same timing.')]
        await runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Confirm the timestamp again.'))
        replayed = next(message for message in self.provider.requests[-1]['messages']
            if message.name == 'memory_search' and message.metadata.get('tool_phase') == 'result')
        shown_again = next(record for record in replayed.metadata['tool_payload']['output']['messages']
            if record['message_id'] == source.db_id)
        self.assertEqual(shown_again['sent_at'], '2026-09-13T00:00:20+08:00')
        self.assertIn('2026-09-12T16:00:20Z', shown_again['fragments'][0]['text'])

    async def test_image_labels_store_utc_and_reproject_after_a_configured_zone_change(self):
        literal = '[Original image evidence: {"sent_at":"2026-09-12T16:00:20Z"}]'
        source = await self.source(1, '2026-09-12T16:00:20Z',
            literal, image=True)
        image_id = (await self.store.describe_message_images(self.session, [source.db_id]))[source.db_id][0]['image_id']
        await self.settings(mode=ChatMode.ASSIST)
        self.provider.capabilities = replace(self.provider.capabilities, multimodal_tool_results=True)
        runtime = AgentRuntime(config=self.config, store=self.store, tool_registry=self.tools,
            providers={'openai': self.provider}, memory=self.memory, preview_cache=self.preview_cache)
        self.provider.responses = [ProviderResponse(tool_calls=[ToolCall('memory_read', 'view',
            {'message_ids': [source.db_id], 'image_ids': [image_id]})]),
            ProviderResponse(final_text='I inspected the image.')]
        await runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Inspect the earlier image.'))

        def label(messages):
            part = next(part for message in messages for part in message.parts
                if part.kind == PartKind.TEXT and part.origin == f'memory_image:{image_id}')
            return json.loads(part.text.removeprefix('[Original image evidence: ').removesuffix(']'))

        stored = [row.message for row in await self.store.list_canonical_messages(self.session)]
        self.assertEqual(label(stored)['sent_at'], '2026-09-12T16:00:20+00:00')
        self.assertEqual(label(self.provider.requests[-1]['messages'])['sent_at'], '2026-09-13T00:00:20+08:00')
        config = replace(self.config, default_metadata_timezone='America/New_York')
        reader = await self.new_store()
        cold = AgentRuntime(config=config, store=reader, tool_registry=self.tools,
            providers={'openai': self.provider}, memory=MemoryService(reader, SimpleNamespace(enabled=False)),
            preview_cache=self.preview_cache)
        self.provider.responses = [ProviderResponse(final_text='The evidence is unchanged.')]
        await cold.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Recall the image timing.'))
        self.assertEqual(label(self.provider.requests[-1]['messages'])['sent_at'], '2026-09-12T12:00:20-04:00')
        self.assertTrue(any(part.text == literal for message in self.provider.requests[-1]['messages']
            for part in message.parts), 'A literal lookalike in participant text must not be rewritten.')
        self.assertEqual(label([row.message for row in await reader.list_canonical_messages(self.session)])['sent_at'],
            '2026-09-12T16:00:20+00:00')
