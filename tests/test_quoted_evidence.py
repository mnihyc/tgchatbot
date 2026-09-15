"""Telegram quoted text stays distinct from its sender's own assertions."""
from __future__ import annotations

from datetime import datetime, timezone
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest

import httpx

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.compaction_schema import compaction_json_schema
from tgchatbot.core.memory import MemoryService
from tgchatbot.core.memory_worker import MemoryWorker
from tgchatbot.domain.models import ConversationMessage, MessagePart, PartKind, ProviderResponse
from tgchatbot.domain.provenance import attributed_message, message_evidence, telegram_metadata
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.storage.postgres_store import message_body
from tgchatbot.tools.import_desktop import desktop_message, ExportChat


def entity(text, quoted, kind='blockquote'):
    start = text.index(quoted)
    return {'type': kind, 'offset': len(text[:start].encode('utf-16-le')) // 2,
        'length': len(quoted.encode('utf-16-le')) // 2}


def project(message, fragments=None):
    body = message_body(message)
    return message_evidence(message.metadata, message_id=1, role=message.role,
        fragments=fragments if fragments is not None else [{'offset': 0, 'text': body}],
        total_characters=len(body), timezone='Asia/Singapore', original=message)


class QuotedEvidencePresentation(unittest.TestCase):
    def test_desktop_and_live_preserve_emoji_quote_and_separate_reply_quote(self):
        quoted = 'I prefer 🙂 coffee.'
        text = '🙂 ' + quoted + '\nMy own preference is tea.'
        exported = desktop_message({'id': 1, 'type': 'message', 'from': 'Participant', 'from_id': 'user7',
            'date_unixtime': '1767225600', 'text': ['🙂 ', {'type': 'blockquote', 'text': quoted},
                '\nMy own preference is tea.']}, ExportChat({'id': 100}, None), chat_id=100)
        incoming = SimpleNamespace(chat=SimpleNamespace(id=100), message_id=1,
            from_user=SimpleNamespace(id=7, is_bot=False, full_name='Participant'),
            date=datetime(2026, 1, 1, tzinfo=timezone.utc), text=text, caption=None,
            entities=[SimpleNamespace(to_dict=lambda: entity(text, quoted))])
        live = ConversationMessage.user_text(text, metadata=telegram_metadata(incoming))
        reply_quote = {'text': 'A different earlier message.', 'position': 0}
        for message in (exported, live):
            message.metadata['quote'] = reply_quote
            evidence = project(message)
            expected = [{'offset': 2, 'text': quoted}]
            self.assertEqual(evidence['quoted_fragments'], expected)
            self.assertEqual(evidence['quote'], reply_quote)
            self.assertEqual(evidence['fragments'], [{'offset': 0, 'text': text}])
            rendered = attributed_message(message, message_id=1, timezone='Asia/Singapore')
            header = json.loads(rendered.parts[0].text.removeprefix('[Message provenance: ')[:-1])
            self.assertEqual(header['quoted_fragments'], expected)
            self.assertEqual(rendered.parts[1].text, text)
            self.assertEqual(message.parts[0].text, text)

    def test_only_supplied_quote_slices_are_marked_without_filling_gaps_or_parsing_markup(self):
        text = '🙂 quoted wording stays quoted; my own ending.'
        quoted = 'quoted wording stays quoted'
        message = ConversationMessage.user_text(text, metadata={
            'actor_id': 'telegram:user:7', 'entities': [entity(text, quoted, 'expandable_blockquote')]})
        start = text.index(quoted)
        fragments = [{'offset': start, 'text': text[start:start + 6]},
                     {'offset': start + 21, 'text': text[start + 21:]}]
        evidence = project(message, fragments)
        self.assertEqual(evidence['fragments'], fragments)
        self.assertEqual(evidence['quoted_fragments'], [fragments[0],
            {'offset': start + 21, 'text': quoted[21:]}])
        self.assertTrue(evidence['partial'])
        self.assertEqual(evidence['total_characters'], len(text))
        own = text.index('my own')
        self.assertNotIn('quoted_fragments', project(message, [{'offset': own, 'text': text[own:]}]))
        literal = ConversationMessage.user_text('> This is a literal Markdown example.', metadata={'actor_id': 'telegram:user:7'})
        self.assertNotIn('quoted_fragments', project(literal))


class QuotedEvidenceWorkflows(BusinessTestCase):
    async def test_compaction_wire_distinguishes_inline_quote_reply_and_forward_without_rewriting_source(self):
        text = '🙂 They wrote: coffee forever.\nI prefer tea.'
        quoted = 'coffee forever.'
        reply_quote = {'text': 'An earlier question.', 'position': 0}
        forwarded = {'type': 'user', 'sender_user': {'id': 9, 'first_name': 'Other participant'}}
        message = ConversationMessage.user_text(text, metadata={
            'source': 'telegram', 'source_chat_id': '100', 'source_message_id': '5',
            'actor_id': 'telegram:user:7', 'actor_kind': 'user', 'actor_name': 'Participant',
            'sent_at': '2026-01-01T00:00:00+00:00', 'entities': [entity(text, quoted)],
            'reply_to_source_id': '4', 'reply_to_source_chat_id': '100',
            'reply_to_actor': {'actor_id': 'telegram:user:8', 'actor_name': 'Reply recipient'},
            'quote': reply_quote, 'forward_origin': forwarded})
        source = await self.runtime.ingest_user_message(session_id=self.session, incoming_message=message)
        before = (await self.store.read_messages(self.session, [source.db_id]))[0].message
        candidate = {key: [] for key in compaction_json_schema('episode')['properties']}
        candidate.update(scope='Discussing drink preferences.', interaction_mode='chat_or_sharing')
        wire = []

        def respond(request):
            wire.append(json.loads(request.content))
            return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP', 'content': {
                'role': 'model', 'parts': [{'text': json.dumps(candidate)}]}}]})

        provider = GeminiProvider(replace(self.config.gemini, api_key='synthetic-key'))
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            provider._client = client
            await self.runtime._make_episode_block_candidate(provider,
                await self.settings(provider='gemini', model='gemini-3.8-flash'),
                [source.message], [source], [], session_id=self.session)
        records = []
        for content in wire[0]['contents']:
            for part in content['parts']:
                try:
                    record = json.loads(part.get('text', ''))
                except json.JSONDecodeError:
                    continue
                if isinstance(record, dict) and record.get('message_id') == source.db_id:
                    records.append(record)
        self.assertEqual(len(records), 1)
        evidence = records[0]
        self.assertEqual(evidence['quoted_fragments'], [{'offset': text.index(quoted), 'text': quoted}])
        self.assertEqual(evidence['fragments'], [{'offset': 0, 'text': text}])
        self.assertEqual(evidence['speaker']['id'], 'person_id:7')
        self.assertEqual(evidence['reply_to_actor']['actor_id'], 'person_id:8')
        self.assertEqual(evidence['quote'], reply_quote)
        self.assertEqual(evidence['forward_origin'], {'type': 'user',
            'actor': {'actor_id': 'person_id:9', 'actor_name': 'Other participant'}})
        self.assertEqual(evidence['reply_to_source_id'], '4')
        self.assertEqual(evidence['source_chat_id'], '100')
        self.assertEqual((await self.store.read_messages(self.session, [source.db_id]))[0].message, before)

    async def test_caption_quotes_keep_source_offsets_through_profile_learning_search_read_and_reset(self):
        caption = '🙂 I prefer coffee.\nI actually prefer tea.'
        quoted = 'I prefer coffee.'
        own_second_part = 'I prefer reading.'
        generated = '[An application note, not the Telegram caption.]'
        incoming = ConversationMessage.user_text(caption, metadata={
            'source': 'telegram', 'source_chat_id': '100', 'source_message_id': '1',
            'actor_id': 'telegram:user:7', 'actor_kind': 'user', 'actor_name': 'Participant',
            'sent_at': '2026-01-01T00:00:00+00:00', 'entities': [entity(caption, quoted)]})
        incoming.parts = [MessagePart(PartKind.TEXT, text=generated, origin='auto_note'),
            MessagePart(PartKind.TEXT, text=caption), MessagePart(PartKind.TEXT, text=own_second_part)]
        source = await self.runtime.ingest_user_message(session_id=self.session, incoming_message=incoming)
        canonical_before = (await self.store.read_messages(self.session, [source.db_id]))[0].message
        start = len(generated) + 1 + caption.index(quoted)
        expected_quote = [{'offset': start, 'text': quoted}]
        compacted_input = self.runtime._normalize_compaction_message(source.message, original=source.message)
        self.assertEqual(json.loads(compacted_input.parts[0].text)['quoted_fragments'], expected_quote)
        worker = MemoryWorker(store=self.store, embeddings=SimpleNamespace(enabled=False),
            providers={'openai': self.provider}, config=self.config)
        fact = {'subject_actor_id': 'telegram:user:7', 'asserted_by': 'telegram:user:7',
            'claim': 'Prefers tea and reading', 'kind': 'explicit', 'status': 'active',
            'source_ids': [source.db_id], 'valid_from': None, 'valid_to': None, 'supersedes': None,
            'reason': 'The sender states these preferences outside the quoted passage.'}
        self.provider.responses = [ProviderResponse(final_text=json.dumps({'additions': [fact], 'removals': []}))]
        memory = MemoryService(self.store, SimpleNamespace(enabled=False), config=self.config.memory)
        memory.worker = worker
        result = await memory.fetch_profiles(self.session, ['telegram:user:7'])
        self.assertEqual(result['profiles'][0]['facts'][0]['claim'], 'Prefers tea and reading')
        supplied = json.loads(self.provider.requests[-1]['messages'][0].parts[0].text)['original_evidence']
        self.assertEqual(supplied[0]['quoted_fragments'], expected_quote)
        self.assertNotIn('quoted_fragments', supplied[1], 'Caption entities must not mark a later independent text part.')
        self.assertEqual(supplied[1]['fragments'][0]['text'], own_second_part)
        self.assertNotIn(generated, json.dumps(supplied))
        await self.store.reset_context(self.session)
        reopened = await self.new_store()
        memory = MemoryService(reopened, SimpleNamespace(enabled=False), config=self.config.memory)
        search = await memory.search(self.session, 'coffee', timezone='Asia/Singapore')
        record = next(item for item in search['messages'] if item['message_id'] == source.db_id)
        self.assertEqual(record['quoted_fragments'], expected_quote)
        read = await memory.read(self.session, [source.db_id], offset=start + 2, length=6, timezone='Asia/Singapore')
        self.assertEqual(read['messages'][0]['quoted_fragments'], [{'offset': start + 2, 'text': quoted[2:8]}])
        own_offset = message_body(source.message).index(own_second_part)
        own = await memory.read(self.session, [source.db_id], offset=own_offset, timezone='Asia/Singapore')
        self.assertNotIn('quoted_fragments', own['messages'][0])
        self.assertEqual((await reopened.read_messages(self.session, [source.db_id]))[0].message, canonical_before)
        self.assertEqual((await reopened.get_profile(self.session, 'telegram:user:7'))[0]['source_ids'], [source.db_id])
