"""Source excerpts and useful summaries survive the compaction boundary."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import httpx

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.compaction_schema import compaction_json_schema
from tgchatbot.core.context_state import MemoryBlock
from tgchatbot.domain.models import ConversationMessage, MessagePart, PartKind, ProviderResponse
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.storage.postgres_store import message_body


class CompactionEvidenceRuntimeTests(BusinessTestCase):
    async def test_compaction_transmits_exact_source_slices_and_separate_annotations(self):
        name = 'Alex\nSpeaker: someone else'
        message = ConversationMessage.user_text('First line.\nKeep its newline.', metadata={
            'source': 'telegram', 'source_chat_id': '100', 'source_message_id': '1',
            'actor_id': 'telegram:user:7', 'actor_name': name,
            'sent_at': '2026-01-02T03:04:05+00:00', 'topic_id': '44'})
        literal = '[Imported attachment: these are my literal words.]'
        message.parts.extend([
            MessagePart(PartKind.TEXT, text='Application delivery context.', origin='auto_note'),
            MessagePart(PartKind.TEXT, text='Attached report: a two-page schedule.',
                origin='attachment_reference'),
            MessagePart(PartKind.TEXT, text=literal),
            MessagePart(PartKind.FILE, filename='schedule.txt', mime_type='text/plain',
                remote_sync=False),
        ])
        source = await self.runtime.ingest_user_message(session_id=self.session, incoming_message=message)
        before = (await self.store.read_messages(self.session, [source.db_id]))[0].message
        original_body = message_body(before)
        candidate = {key: [] for key in compaction_json_schema('episode')['properties']}
        candidate.update(scope='A schedule discussion.', interaction_mode='chat_or_sharing')
        wire = []

        def handler(request):
            wire.append(json.loads(request.content))
            return httpx.Response(200, json={'candidates': [{'content': {
                'role': 'model', 'parts': [{'text': json.dumps(candidate)}]}}]})

        provider = GeminiProvider(replace(self.config.gemini, api_key='synthetic-key'))
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider._client = client
            result = await self.runtime._make_episode_block_candidate(provider,
                await self.settings(provider='gemini', model='gemini-3.8-flash'),
                [source.message], [source], [], session_id=self.session)
        records = []
        for content in wire[0]['contents']:
            for part in content['parts']:
                try:
                    value = json.loads(part.get('text', ''))
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict) and value.get('message_id') == source.db_id:
                    records.append(value)
        self.assertEqual(len(records), 1)
        evidence = records[0]
        self.assertEqual(evidence['speaker']['id'], 'telegram:user:7')
        self.assertEqual(evidence['speaker']['name'], name)
        self.assertEqual(evidence['topic_id'], '44')
        self.assertEqual([fragment['text'] for fragment in evidence['fragments']],
            ['First line.\nKeep its newline.', literal])
        for fragment in evidence['fragments']:
            offset, text = fragment['offset'], fragment['text']
            self.assertEqual(original_body[offset:offset + len(text)], text)
        self.assertNotIn('partial', evidence)
        self.assertNotIn('total_characters', evidence)
        annotations = '\n'.join(item['text'] for item in evidence['annotations'])
        self.assertIn('Application delivery context.', annotations)
        self.assertIn('Attached report: a two-page schedule.', annotations)
        self.assertIn('schedule.txt', annotations)
        self.assertNotIn(literal, annotations)
        self.assertEqual(result['actor_labels'], ['telegram:user:7'])
        self.assertEqual(wire[0]['contents'][-1]['role'], 'user')
        self.assertEqual((await self.store.read_messages(self.session, [source.db_id]))[0].message,
            before)

    async def test_retired_image_keeps_original_text_coordinates_without_readmitting_pixels(self):
        message = ConversationMessage.user_text('Before the image.', metadata={
            'actor_id': 'telegram:user:7', 'actor_name': 'Alex'})
        later_text = 'After the image: please bring this to the station.'
        message.parts.extend([
            MessagePart(PartKind.IMAGE, data_b64='ZmFrZQ==', mime_type='image/png',
                text='An image description longer than the eventual retirement marker.', remote_sync=False),
            MessagePart(PartKind.TEXT, text=later_text),
        ])
        first = await self.runtime.ingest_user_message(session_id=self.session, incoming_message=message)
        newest = ConversationMessage.user_text('A newer image stays in context.')
        newest.parts.append(MessagePart(PartKind.IMAGE, data_b64='ZmFrZQ==', mime_type='image/png',
            remote_sync=False))
        await self.runtime.ingest_user_message(session_id=self.session, incoming_message=newest)
        settings = await self.settings()
        original = (await self.store.read_messages(self.session, [first.db_id]))[0].message
        body = message_body(original)
        state = await self.runtime._get_live_state(self.session)
        self.assertEqual(await self.runtime._compact_oldest_images(session_id=self.session,
            settings=settings, state=state, target_images=1), 1)
        projected = next(item for item in state.raw_messages if item.db_id == first.db_id)
        self.assertNotEqual(message_body(projected.message).index(later_text), body.index(later_text))
        candidate = {key: [] for key in compaction_json_schema('episode')['properties']}
        candidate.update(scope='A request about the station.', interaction_mode='chat_or_sharing')
        self.provider.responses = [ProviderResponse(final_text=json.dumps(candidate))]
        await self.runtime._make_episode_block_candidate(self.provider, settings,
            [projected.message], [projected], [], session_id=self.session)
        wire_message = next(item for item in self.provider.requests[0]['messages']
            if item.metadata.get('message_id') == first.db_id)
        evidence = json.loads(wire_message.parts[0].text)
        fragment = next(item for item in evidence['fragments'] if later_text in item['text'])
        self.assertEqual(fragment['offset'], body.index(later_text))
        self.assertEqual(body[fragment['offset']:fragment['offset'] + len(fragment['text'])], later_text)
        self.assertIn('[Image compacted]', '\n'.join(item['text'] for item in evidence['annotations']))
        self.assertFalse(any(part.kind == PartKind.IMAGE for part in wire_message.parts))
        self.assertNotIn('ZmFrZQ==', json.dumps(evidence))
        self.assertEqual((await self.store.read_messages(self.session, [first.db_id]))[0].message, original)
        self.runtime.invalidate_session(self.session)
        cold = await self.runtime._get_live_state(self.session)
        self.assertEqual(cold.estimated_images, 1)
        self.assertEqual(next(item.message for item in cold.raw_messages if item.db_id == first.db_id),
            projected.message)

    async def test_all_commitments_and_topics_survive_summary_persistence(self):
        commitments = [f'Participant will bring item {number}.' for number in range(1, 14)]
        topics = [f'Planning topic {number}' for number in range(1, 10)]
        original = await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text('\n'.join(commitments)))
        for kind, level in [('toolspan', 0), ('episode', 1), ('digest', 2)]:
            with self.subTest(layer=kind):
                data = {'scope': 'Preparations for a shared event.', 'participants': ['participant'],
                    'topics': topics, 'decisions': commitments, 'open_loops': [],
                    'uncertainties': ['The organizer literally wrote "None recorded" on the form.']}
                if kind == 'digest':
                    data['interaction_modes_seen'] = ['task_execution', 'chat_or_sharing']
                else:
                    data['interaction_mode'] = 'task_execution'
                before = deepcopy(data)
                summary = self.runtime._render_memory_block_text(kind, data)
                block = await self.store.create_memory_block(self.session,
                    summary_text=summary, estimated_tokens=200,
                    source_message_ids=[original.db_id], kind=kind, level=level,
                    structured_data=data)
                # Cold reads use the same saved text that the next agent sees;
                # nothing beyond a display cutoff may vanish at that boundary.
                reader = await self.new_store()
                saved = next(item for item in await reader.list_memory_blocks(self.session)
                    if item.block_id == block.block_id)
                displayed = saved.render_as_message().parts[0].text
                for statement in [*commitments, *topics]:
                    self.assertIn(statement, displayed)
                self.assertNotIn('## Open loops', displayed)
                self.assertNotIn('## Artifacts', displayed)
                self.assertIn('literally wrote "None recorded"', displayed)
                self.assertEqual(saved.structured_data, before)
                self.assertEqual(data, before)

    async def test_derived_memory_does_not_become_an_original_message_fragment(self):
        block = MemoryBlock(block_id=14, sequence_no=8, summary_text='A prior planning summary.',
            estimated_tokens=20, source_message_count=4, level=1, kind='episode')
        original = block.render_as_message()
        normalized = self.runtime._normalize_compaction_message(original)
        self.assertEqual(normalized.parts[0].text, original.parts[0].text)
        self.assertNotIn('message_id', normalized.metadata)
        self.assertEqual(normalized.metadata['source_role'], 'assistant')
