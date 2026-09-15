"""Original-image discovery and explicit recall through real memory tools/DB.

Only hosted embedding output is controlled. Images are generated fixture bytes;
no deployment data or external provider is accessed.
"""
from __future__ import annotations

import base64
from dataclasses import replace
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.memory import MemoryService
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind
from tgchatbot.storage.postgres_store import message_body
from tgchatbot.storage.relationships import expand_message_ids
from tgchatbot.tools.base import ToolContext


PIXEL = base64.b64encode(bytes.fromhex(
    '89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489'
    '0000000b49444154789c636000020000050001a5f645400000000049454e44ae426082'
)).decode('ascii')


class MemoryImageToolTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.settings()
        self.embeddings = SimpleNamespace(enabled=False, space_id='image-memory-fixture', embed_query=AsyncMock())
        self.memory = MemoryService(self.store, self.embeddings)
        self.context = ToolContext(self.session, 'Participant')

    async def original(self, number, text='', *, actor=101, image=False, topic='7',
                       reply=None, role=MessageRole.USER, name=None, **metadata):
        parts = [MessagePart(PartKind.TEXT, text=text)] if text else []
        if image:
            parts.append(MessagePart(PartKind.IMAGE, mime_type='image/png', data_b64=PIXEL))
        values = {'source': 'telegram', 'source_chat_id': '100', 'source_message_id': str(number),
            'actor_id': f'telegram:user:{actor}', 'actor_kind': 'user', 'actor_name': 'Participant',
            'sent_at': f'2026-01-02T03:{number // 60:02d}:{number % 60:02d}+00:00', 'topic_id': topic,
            **metadata}
        if reply is not None:
            values['reply_to_source_id'] = str(reply)
        return await self.store.append_message(self.session,
            ConversationMessage(role, parts, name=name, metadata=values))

    async def tool(self, name, arguments):
        spec = next(tool for tool in self.memory.tools if tool.name == name)
        result = await spec.runner.run(arguments, self.context)
        self.assertTrue(result.output['ok'], result.output)
        return result

    async def test_imported_frames_keep_selectable_occurrences_without_repeating_synthetic_descriptors(self):
        message = ConversationMessage.user_text('This is the cobalt animation.', metadata={
            'source': 'telegram', 'source_chat_id': '100', 'source_message_id': '1',
            'actor_id': 'telegram:user:101', 'actor_kind': 'user', 'actor_name': 'Participant',
            'sent_at': '2026-01-02T03:00:01+00:00'})
        hint = 'Animation: cobalt.webm. Original export file unavailable; retained previews available.'
        message.parts.append(MessagePart(PartKind.TEXT, text=hint, origin='attachment_reference'))
        message.parts.extend(MessagePart(PartKind.IMAGE, mime_type='image/png', filename=f'cobalt-preview-{index}.png',
            data_b64=PIXEL, detail='auto') for index in range(5))
        message.parts.append(MessagePart(PartKind.IMAGE, mime_type='image/png', data_b64=PIXEL,
            text='A custom caption describing the final pose.'))
        source = await self.store.append_message(self.session, message)
        canonical_before = (await self.store.read_messages(self.session, [source.db_id]))[0].message
        read = await self.tool('memory_read', {'message_ids': [source.db_id]})
        record = read.output['messages'][0]
        self.assertEqual(record['annotations'], [
            {'kind': 'attachment', 'text': hint},
            {'kind': 'attachment', 'text': 'A custom caption describing the final pose.'}])
        self.assertEqual(len(record['images']), 6)
        self.assertEqual(len({image['image_id'] for image in record['images']}), 6)
        self.assertTrue(all(image['available'] for image in record['images']))
        selected = await self.tool('memory_read', {'message_ids': [source.db_id],
            'image_ids': [record['images'][2]['image_id']]})
        self.assertEqual(selected.output['image_results'][0]['status'], 'selected')
        self.assertEqual(len(selected.evidence_parts), 2)
        label = json.loads(selected.evidence_parts[0].text.removeprefix('[Original image evidence: ')[:-1])
        self.assertEqual(label['speaker']['id'], 'person_id:101')
        self.assertEqual(label['image_id'], record['images'][2]['image_id'])
        self.assertNotIn('source_revision', label)
        self.assertEqual((await self.store.read_messages(self.session, [source.db_id]))[0].message, canonical_before)

    async def test_captionless_nearby_image_has_its_own_identity_and_never_changes_match(self):
        photo = await self.original(1, image=True, actor=102)
        caption = await self.original(2, 'The cobalt suitcase is finally packed.', actor=101)
        unrelated = await self.original(3, image=True, actor=103, topic='8')
        baseline = await self.store.search_excerpts(self.session, 'cobalt', limit=20)
        result = await self.tool('memory_search', {'query': 'cobalt', 'actor_id': 'telegram:user:101'})
        output = result.output
        self.assertEqual([row['message_ids'] for row in output['matches']],
            [row['source_ids'] for row in baseline])
        records = {row['message_id']: row for row in output['messages']}
        self.assertEqual(records[caption.db_id]['fragments'], [{'offset': 0, 'text': baseline[0]['text']}])
        self.assertNotIn('images', records[caption.db_id])
        self.assertEqual(output['related_context'], [photo.db_id])
        neighbor = records[photo.db_id]
        self.assertEqual(neighbor['speaker']['id'], 'person_id:102')
        self.assertEqual(neighbor['fragments'], [], 'A captionless photo has no words spoken by its sender')
        self.assertNotIn('annotations', neighbor, 'The image reference already describes the synthetic attachment.')
        self.assertNotIn('partial', neighbor, 'Attachment annotations retain the complete selected evidence')
        self.assertTrue(neighbor['images'][0]['available'])
        self.assertNotIn(unrelated.db_id, output['related_context'])
        self.assertEqual([row['message_id'] for row in output['messages']], [photo.db_id, caption.db_id])
        self.assertFalse(result.evidence_parts)
        self.assertNotIn(PIXEL, json.dumps(output, default=str))

    async def test_multisource_excerpt_keeps_spans_and_images_under_the_actual_senders(self):
        question = await self.original(1, 'Which suitcase?', actor=101)
        answer = await self.original(2, 'This one.', image=True, actor=102,
            forward_origin={'type': 'user', 'sender_user': {'id': 900}})
        vector = np.zeros(1536, dtype=np.float32)
        vector[0] = 1
        self.embeddings.enabled = True
        self.embeddings.embed_query.return_value = vector
        spans = [{'message_id': item.db_id, 'start': 0, 'end': len(message_body(item.message))}
                 for item in (question, answer)]
        await self.store.create_excerpt(self.session, [question.db_id, answer.db_id], spans=spans,
            embedding=vector, model=self.embeddings.space_id)
        output = (await self.tool('memory_search', {'query': 'semantic-control'})).output
        self.assertEqual(len(output['matches']), 1)
        match = output['matches'][0]
        self.assertEqual(match['message_ids'], [question.db_id, answer.db_id])
        self.assertNotIn('images', match, 'An excerpt does not own its participants\' images')
        self.assertEqual([row['message_id'] for row in output['messages']], [question.db_id, answer.db_id])
        self.assertEqual([row['fragments'] for row in output['messages']],
            [[{'offset': 0, 'text': text}] for text in ('Which suitcase?', 'This one.')])
        self.assertNotIn('annotations', output['messages'][1])
        self.assertFalse(any(row.get('partial') for row in output['messages']))
        self.assertNotIn('images', output['messages'][0])
        self.assertTrue(output['messages'][1]['images'][0]['available'])
        self.assertEqual(output['messages'][1]['speaker']['id'], 'person_id:102')
        self.assertEqual(output['messages'][1]['forward_origin']['actor']['actor_id'], 'person_id:900')
        self.assertEqual(output.get('related_context', []), [])

    async def test_read_defaults_are_text_only_and_explicit_selection_has_attributed_evidence(self):
        photo = await self.original(1, 'My packed suitcase.', image=True)
        for optional in ({}, {'image_ids': []}, {'image_ids': None}):
            result = await self.tool('memory_read', {'message_ids': [photo.db_id], **optional})
            record = result.output['messages'][0]
            self.assertEqual(record['fragments'], [{'offset': 0, 'text': 'My packed suitcase.'}])
            self.assertNotIn('annotations', record)
            self.assertNotIn('partial', record)
            self.assertFalse(result.evidence_parts)
            self.assertNotIn('image_results', result.output)
        image_id = result.output['messages'][0]['images'][0]['image_id']
        selected = await self.tool('memory_read', {'message_ids': [photo.db_id], 'image_ids': [image_id]})
        self.assertEqual(selected.output['image_results'], [{'image_id': image_id, 'status': 'selected'}])
        self.assertEqual([part.kind for part in selected.evidence_parts], [PartKind.TEXT, PartKind.IMAGE])
        self.assertTrue(all(part.origin == 'memory_image:' + image_id for part in selected.evidence_parts))
        self.assertIn('person_id:101', selected.evidence_parts[0].text)
        self.assertIn(f'"message_id": {photo.db_id}', selected.evidence_parts[0].text)
        self.assertEqual(selected.evidence_parts[1].data_b64, PIXEL)
        self.assertNotIn(PIXEL, json.dumps(selected.output))
        self.assertFalse(selected.artifacts)
        self.assertFalse(selected.stickers)

    async def test_neighbor_discovery_does_not_grant_implicit_image_selection(self):
        photo = await self.original(1, image=True, actor=102)
        caption = await self.original(2, 'The cobalt suitcase.', reply=1)
        discovery = await self.tool('memory_read', {'message_ids': [caption.db_id], 'include_neighbors': True})
        image_id = next(row for row in discovery.output['messages']
                        if row['message_id'] == photo.db_id)['images'][0]['image_id']
        denied = await self.tool('memory_read', {'message_ids': [caption.db_id],
            'include_neighbors': True, 'image_ids': [image_id]})
        self.assertEqual(denied.output['image_results'][0]['status'], 'unavailable')
        self.assertFalse(denied.evidence_parts)
        allowed = await self.tool('memory_read', {'message_ids': [photo.db_id], 'image_ids': [image_id]})
        self.assertEqual(allowed.output['image_results'][0]['status'], 'selected')
        self.assertEqual(len(allowed.evidence_parts), 2)

    async def test_related_context_is_one_hop_and_reply_precedence_uses_shared_window(self):
        self.store.config = replace(self.store.config, relationship_neighbors=0)
        old = await self.original(1, image=True, actor=103, topic='old')
        reply = await self.original(2, image=True, actor=102, topic='other', reply=1)
        seed = await self.original(3, 'cobalt suitcase', reply=2)
        result = (await self.tool('memory_search', {'query': 'cobalt'})).output
        self.assertEqual(result['related_context'], [reply.db_id])
        self.assertNotIn(old.db_id, result['related_context'])
        self.assertEqual(await expand_message_ids(self.store, self.session, [seed.db_id]), [seed.db_id, reply.db_id])
        self.memory.config = replace(self.memory.config, read_messages=1)
        self.assertEqual((await self.tool('memory_search', {'query': 'cobalt'})).output['related_context'],
            [reply.db_id], 'A full ranked seed window must not consume its separate one-hop context window')

    async def test_related_context_deduplicates_originals_without_deduplicating_people(self):
        self.store.config = replace(self.store.config, relationship_neighbors=0)
        photo = await self.original(1, image=True, actor=102)
        first = await self.original(2, 'cobalt question', actor=101, reply=1)
        second = await self.original(3, 'cobalt question', actor=103, reply=1)
        output = (await self.tool('memory_search', {'query': 'cobalt'})).output
        self.assertEqual({row['message_ids'][0] for row in output['matches']}, {first.db_id, second.db_id})
        self.assertEqual(output['related_context'], [photo.db_id])
        self.assertEqual([row['message_id'] for row in output['messages']], [photo.db_id, first.db_id, second.db_id])
        self.assertEqual([row['speaker']['id'] for row in output['messages']],
            ['person_id:102', 'person_id:101', 'person_id:103'])

    async def test_small_context_window_follows_ranked_evidence_instead_of_import_order(self):
        self.store.config = replace(self.store.config, relationship_neighbors=0)
        old_photo = await self.original(1, image=True)
        old_caption = await self.original(2, 'cobalt old suitcase', reply=1)
        recent_photo = await self.original(3, image=True, actor=102)
        recent_caption = await self.original(4, 'cobalt replacement suitcase', reply=3, actor=102)
        vector = np.zeros(1536, dtype=np.float32)
        vector[0] = 1
        self.embeddings.enabled = True
        self.embeddings.embed_query.return_value = vector
        await self.store.create_excerpt(self.session, [recent_caption.db_id],
            embedding=vector, model=self.embeddings.space_id)
        self.memory.config = replace(self.memory.config, read_messages=1)
        output = (await self.tool('memory_search', {'query': 'cobalt'})).output
        self.assertEqual([row['message_ids'] for row in output['matches']],
            [[recent_caption.db_id], [old_caption.db_id]])
        self.assertEqual(output['related_context'], [recent_photo.db_id])
        self.assertNotIn(old_photo.db_id, output['related_context'])
        self.assertEqual([row['message_id'] for row in output['messages']],
            [old_caption.db_id, recent_photo.db_id, recent_caption.db_id])

    async def test_context_budget_charges_its_attribution_and_keeps_ranked_text_allowance(self):
        neighbor = await self.original(1, 'Quoted \\" caption 照片. ' * 100, image=True, actor=102)
        source = await self.original(2, 'cobalt match ' * 20)
        self.memory.config = replace(self.memory.config, response_chars=850, search_result_chars=300)
        output = (await self.tool('memory_search', {'query': 'cobalt'})).output
        records = {row['message_id']: row for row in output['messages']}
        self.assertEqual(records[source.db_id]['fragments'], [{'offset': 0, 'text': message_body(source.message)}])
        self.assertEqual(output['related_context'], [neighbor.db_id])
        related = [records[neighbor.db_id]]
        self.assertTrue(related[0]['partial'])
        self.assertEqual(related[0]['total_characters'], len(message_body(neighbor.message)))
        self.assertLess(len(related[0]['fragments'][0]['text']), len(message_body(neighbor.message)))
        ranked_text = len(message_body(source.message))
        descriptors = len(json.dumps(records[source.db_id].get('images', []), ensure_ascii=False))
        self.assertLessEqual(ranked_text + descriptors + len(json.dumps(related, ensure_ascii=False)),
            self.memory.config.response_chars)
        self.memory.config = replace(self.memory.config, response_chars=100)
        exhausted = (await self.tool('memory_search', {'query': 'cobalt'})).output
        self.assertEqual(exhausted['matches'], [{'message_ids': [source.db_id], 'truncated': True}])
        self.assertEqual(exhausted['messages'][0]['fragments'], [{'offset': 0, 'text': message_body(source.message)[:100]}])
        self.assertTrue(exhausted['messages'][0]['partial'])
        self.assertEqual(exhausted.get('related_context', []), [])

    async def test_recalled_copies_stay_out_of_discovery_but_external_observations_remain(self):
        photo = await self.original(1, image=True, actor=102)
        image_id = (await self.tool('memory_read', {'message_ids': [photo.db_id]})).output['messages'][0]['images'][0]['image_id']
        opened = await self.tool('memory_read', {'message_ids': [photo.db_id], 'image_ids': [image_id]})
        derived = await self.store.append_message(self.session, ConversationMessage(MessageRole.TOOL,
            [MessagePart(PartKind.TEXT, text='cobalt image recalled'), *opened.evidence_parts], name='memory_read',
            metadata={'tool_phase': 'result', 'topic_id': '7', 'sent_at': '2026-01-02T03:00:03+00:00'}))
        external = await self.original(4, 'Independent shell observation', image=True, role=MessageRole.TOOL,
            name='shell_exec', tool_phase='result')
        await self.original(2, 'cobalt suitcase')
        search = (await self.tool('memory_search', {'query': 'cobalt'})).output
        discovered = search['related_context']
        self.assertNotIn(derived.db_id, discovered)
        self.assertIn(photo.db_id, discovered)
        self.assertIn(external.db_id, discovered)
        copied = await self.tool('memory_read', {'message_ids': [derived.db_id], 'image_ids': [image_id]})
        self.assertEqual(copied.output['messages'][0].get('images', []), [])
        self.assertEqual(copied.output['image_results'][0]['status'], 'unavailable')
        self.assertFalse(copied.evidence_parts)

    async def test_soft_reset_keeps_image_recall_and_full_reset_removes_discovery_and_selection(self):
        source = await self.original(1, 'cobalt suitcase', image=True)
        descriptor = (await self.tool('memory_search', {'query': 'cobalt'})).output['messages'][0]['images'][0]
        await self.store.reset_context(self.session)
        retained = await self.tool('memory_read', {'message_ids': [source.db_id], 'image_ids': [descriptor['image_id']]})
        self.assertEqual(retained.output['image_results'][0]['status'], 'selected')
        await self.store.reset_full(self.session, self.config.default_session_settings())
        search = (await self.tool('memory_search', {'query': 'cobalt'})).output
        self.assertEqual(search['matches'], [])
        self.assertEqual(search['messages'], [])
        self.assertEqual(search.get('related_context', []), [])
        denied = await self.tool('memory_read', {'message_ids': [source.db_id], 'image_ids': [descriptor['image_id']]})
        self.assertEqual(denied.output['unavailable_ids'], [source.db_id])
        self.assertEqual(denied.output['image_results'][0]['status'], 'unavailable')
        self.assertFalse(denied.evidence_parts)
