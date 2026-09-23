"""Source excerpts and useful summaries survive the compaction boundary."""
from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace

import httpx

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.compaction_schema import compaction_json_schema
from tgchatbot.core.context_state import MemoryBlock
from tgchatbot.core.runtime import AgentRuntime, CompactionModelRequestFailed
from tgchatbot.domain.models import ConversationMessage, MessagePart, PartKind, ProviderResponse
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.storage.postgres_store import message_body
from tgchatbot.storage.previews import PreviewCache


class CompactionEvidenceRuntimeTests(BusinessTestCase):
    async def prepare_retry_compaction(self, statuses):
        self.config = replace(self.config, context=replace(self.config.context, compact_retry_delay_s=0))
        self.runtime.config = self.config
        settings = await self.settings(provider='gemini', model='gemini-3.8-flash',
            provider_retry_count=1, min_raw_messages_reserve=1, service_tier='flex')
        earlier = await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text('The earlier plan is already settled.'))
        prior_block = await self.store.create_memory_block(self.session,
            summary_text='The earlier plan is settled.', estimated_tokens=12,
            source_message_ids=[earlier.db_id], kind='episode', level=1)
        sources = []
        for text in ('Bring the blue ticket.', 'I will bring that ticket.', 'Latest question stays raw.'):
            message = ConversationMessage.user_text(text, metadata={'actor_id': 'telegram:user:7'})
            if not sources:
                message.parts.append(MessagePart(PartKind.IMAGE, mime_type='image/png',
                    data_b64='dGlja2V0LWltYWdl', remote_sync=False))
            sources.append(await self.runtime.ingest_user_message(session_id=self.session, incoming_message=message))
        candidate = {key: [] for key in compaction_json_schema('episode')['properties']}
        candidate.update(scope='A ticket commitment.', interaction_mode='chat_or_sharing',
            decisions=['Participant will bring the blue ticket.'])
        wire = []

        def respond(request):
            wire.append(json.loads(request.content))
            status = statuses.pop(0)
            if status != 200:
                return httpx.Response(status, json={'error': {'message': 'Temporary Flex capacity failure'}})
            return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP',
                'content': {'role': 'model', 'parts': [{'text': json.dumps(candidate)}]}}]})

        provider = GeminiProvider(replace(self.config.gemini, api_key='synthetic-key'))
        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(provider.aclose)
        self.runtime.providers['gemini'] = provider
        self.runtime.invalidate_session(self.session)
        state = await self.runtime._get_live_state(self.session)
        return settings, provider, state, sources, prior_block, wire

    async def test_temporary_compaction_http_error_retries_and_commits_one_episode(self):
        settings, provider, state, sources, prior, wire = await self.prepare_retry_compaction([503, 200])
        with self.assertLogs('tgchatbot', level='WARNING'):
            changed = await self.runtime._compact_old_context(session_id=self.session,
                settings=settings, provider=provider, state=state, pressure=True)
        self.assertTrue(changed)
        self.assertEqual(len(wire), 2)
        self.assertEqual(wire[0], wire[1], 'Retry the same source batch, schema and service tier')
        self.assertEqual(wire[0]['service_tier'], 'flex')
        self.assertEqual(wire[0]['generationConfig']['responseJsonSchema'], compaction_json_schema('episode'))
        blocks = await self.store.list_memory_blocks(self.session)
        self.assertEqual(len(blocks), 2)
        self.assertIn(prior.block_id, [block.block_id for block in blocks])
        new = next(block for block in blocks if block.block_id != prior.block_id)
        self.assertEqual((new.kind, new.level, new.validator_status), ('episode', 1, 'passed'))
        self.assertEqual(new.structured_data['decisions'], ['Participant will bring the blue ticket.'])
        self.assertNotIn(sources[0].db_id, [row.db_id for row in state.raw_messages])
        self.assertEqual(state.raw_messages[-1].db_id, sources[-1].db_id)
        originals = await self.store.read_messages(self.session, [row.db_id for row in sources])
        self.assertEqual([message_body(row.message) for row in originals],
            [message_body(row.message) for row in sources])

    async def test_exhausted_compaction_retries_preserve_warm_and_rebuilt_context_then_recover(self):
        statuses = [503] * 4
        settings, provider, state, sources, prior, wire = await self.prepare_retry_compaction(statuses)
        originals = await self.store.read_messages(self.session, [row.db_id for row in sources])
        history = self.runtime._build_provider_history(state, settings=settings, provider_name='gemini')
        pixels = await self.preview_cache.materialize_many(self.session, history, vision=True)
        before = deepcopy(state)
        with self.assertLogs('tgchatbot', level='WARNING'):
            with self.assertRaises(CompactionModelRequestFailed):
                await self.runtime._compact_old_context(session_id=self.session,
                    settings=settings, provider=provider, state=state, pressure=True)
        self.assertEqual(len(wire), 4)
        self.assertTrue(all(request == wire[0] for request in wire))
        self.assertEqual(state, before)
        self.assertEqual(await self.store.read_messages(self.session, [row.db_id for row in sources]), originals)
        self.assertEqual([block.block_id for block in await self.store.list_memory_blocks(self.session)], [prior.block_id])
        reader = await self.new_store()
        cold_cache = PreviewCache(reader, max_bytes=0)
        restarted = AgentRuntime(config=self.config, store=reader, tool_registry=self.tools,
            providers={'gemini': provider}, preview_cache=cold_cache)
        rebuilt = await restarted._get_live_state(self.session)
        rebuilt_history = restarted._build_provider_history(rebuilt, settings=settings, provider_name='gemini')
        self.assertEqual(rebuilt_history, history)
        self.assertEqual(await cold_cache.materialize_many(self.session, rebuilt_history, vision=True), pixels)
        statuses.append(200)
        self.assertTrue(await self.runtime._compact_old_context(session_id=self.session,
            settings=settings, provider=provider, state=state, pressure=True))
        self.assertEqual(len(wire), 5)
        self.assertEqual(wire[4], wire[0])
        self.assertNotIn(sources[0].db_id, [row.db_id for row in state.raw_messages])
        self.assertEqual(state.raw_messages[-1].db_id, sources[-1].db_id)
        self.assertEqual(len(await self.store.list_memory_blocks(self.session)), 2)

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
            return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP', 'content': {
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
        self.assertEqual(evidence['speaker']['id'], 'person_id:7')
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
        self.assertEqual(result['actor_labels'], ['person_id:7'])
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
