"""Profile learning keeps original owners and selected evidence across retries."""
from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.memory import MemoryService
from tgchatbot.core.memory_worker import MemoryWorker
from tgchatbot.domain.models import ConversationMessage, MessagePart, PartKind, ProviderResponse


class ProfileLearningEvidenceWorkflows(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.worker = MemoryWorker(store=self.store, embeddings=SimpleNamespace(enabled=False),
            providers={'openai': self.provider}, config=self.config)

    async def source(self, number, actor, text, **metadata):
        return await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text(text, metadata={
                'source': 'telegram', 'source_chat_id': '100', 'source_message_id': str(number),
                'actor_id': actor, 'actor_kind': 'user', 'actor_name': 'Alex',
                'sent_at': '2026-01-01T00:00:00+00:00', **metadata}))

    async def claim(self):
        job = await self.store.claim_profile_batch(session_id=self.session, lazy=True,
            max_bytes=self.worker.limits.profile_request_bytes, lease_seconds=self.worker.limits.lease_seconds)
        self.assertIsNotNone(job)
        return [job]

    @staticmethod
    def fact(actor, claim, source_id):
        return {'subject_actor_id': actor, 'asserted_by': actor, 'claim': claim, 'kind': 'explicit',
            'status': 'active', 'source_ids': [source_id], 'valid_from': None, 'valid_to': None,
            'supersedes': None, 'reason': 'The cited person directly states this lasting preference.'}

    @staticmethod
    def response(*facts):
        return ProviderResponse(final_text=json.dumps({'additions': list(facts), 'removals': []}))

    async def test_profile_size_hints_measure_chat_document_and_stay_in_learning_input(self):
        self.config = replace(self.config, memory=replace(self.config.memory, profile_bytes=2048))
        self.worker.config = self.config
        actor = 'telegram:user:7'
        claim = '喜欢不加糖的茉莉花茶 🍵'
        source = await self.source(1, actor, f'我的长期偏好：{claim}。')
        self.provider.responses = [self.response(self.fact(actor, claim, source.db_id)), self.response()]
        self.assertTrue(await self.worker._guarded(await self.claim(), self.worker._profile))

        await self.source(2, actor, '今天还是喝这个。', sent_at='2026-01-02T00:00:00+00:00')
        self.assertTrue(await self.worker._guarded(await self.claim(), self.worker._profile))
        request = self.provider.requests[1]
        profiles = json.loads(request['messages'][0].parts[0].text)['current_profiles']
        learned = next(profile for profile in profiles if profile['actor_id'] == actor)
        self.assertEqual(learned['facts'][0]['claim'], claim)
        self.assertEqual(learned['facts'][0]['source_dates'], {
            'first': '2026-01-01T08:00:00+08:00', 'last': '2026-01-01T08:00:00+08:00'})
        self.assertIn(str(self.config.memory.profile_bytes), request['instructions'])
        for profile in profiles:
            self.assertIs(type(profile['current_size_bytes']), int)
            chat_document = {key: value for key, value in profile.items() if key != 'current_size_bytes'}
            chat_document['facts'] = [{key: value for key, value in fact.items() if key != 'source_dates'}
                                      for fact in chat_document['facts']]
            serialized = json.dumps(chat_document, ensure_ascii=False)
            self.assertEqual(profile['current_size_bytes'], len(serialized.encode('utf-8')))
            if profile['actor_id'] == actor:
                self.assertGreater(profile['current_size_bytes'], len(serialized),
                    'Multibyte participant text must be measured as UTF-8 bytes, not characters.')

        reopened = await self.new_store()
        snapshot = await reopened.fetch_profile_snapshot(self.session, [actor], for_learning=True)
        self.assertTrue(snapshot['profiles'][0]['facts'][0]['source_dates'])
        self.assertNotIn('current_size_bytes', json.dumps(snapshot, default=str),
            'Request budgeting hints must not become persistent profile fields.')
        memory = MemoryService(reopened, SimpleNamespace(enabled=False), config=self.config.memory)
        fetched = await memory.fetch_profiles(self.session, [actor], timezone=self.config.default_metadata_timezone)
        self.assertNotIn('current_size_bytes', json.dumps(fetched))
        self.assertNotIn('source_dates', json.dumps(fetched))
        self.assertEqual(fetched['profiles'][0]['facts'][0]['claim'], claim)
        self.assertEqual(len(self.provider.requests), 2, 'Sizing hints must not add generation requests.')

    async def test_same_name_reply_and_forward_keep_owners_when_invalid_patch_retries(self):
        tea = await self.source(1, 'telegram:user:7', 'I prefer jasmine tea.')
        quote = {'text': 'I prefer jasmine tea.', 'position': 0}
        reply_actor = {'actor_id': 'telegram:user:7', 'actor_kind': 'user', 'actor_name': 'Alex'}
        coffee = await self.source(2, 'telegram:user:8', 'I prefer coffee instead.',
            reply_to_source_id='1', reply_to_source_chat_id='100', reply_to_actor=reply_actor, quote=quote)
        origin = {'type': 'user', 'sender_user': {'id': 8, 'first_name': 'Alex'}}
        forwarded = await self.source(3, 'telegram:user:7', 'I prefer coffee instead.', forward_origin=origin)
        self.provider.responses = [
            self.response(self.fact('telegram:user:7', 'Prefers coffee', forwarded.db_id)),
            self.response(self.fact('telegram:user:7', 'Prefers jasmine tea', tea.db_id),
                          self.fact('telegram:user:8', 'Prefers coffee', coffee.db_id)),
        ]
        self.assertFalse(await self.worker._guarded(await self.claim(), self.worker._profile))
        self.assertEqual(self.worker.last_error, 'ValueError')
        self.assertEqual(await self.store.get_profile(self.session, 'telegram:user:7'), [])
        self.assertEqual(await self.store.get_profile(self.session, 'telegram:user:8'), [])
        first = json.loads(self.provider.requests[0]['messages'][0].parts[0].text)
        records = {item['message_id']: item for item in first['original_evidence']}
        self.assertEqual(records[tea.db_id]['speaker'], {'id': 'telegram:user:7', 'name': 'Alex'})
        self.assertEqual(records[coffee.db_id]['speaker'], {'id': 'telegram:user:8', 'name': 'Alex'})
        self.assertEqual(records[coffee.db_id]['quote'], quote)
        self.assertEqual(records[coffee.db_id]['reply_to_actor'], reply_actor)
        self.assertEqual(records[coffee.db_id]['reply_to_source_id'], '1')
        self.assertEqual(records[coffee.db_id]['reply_to_source_chat_id'], '100')
        self.assertEqual(records[forwarded.db_id]['forward_origin'], origin)
        self.assertEqual(records[forwarded.db_id]['speaker']['id'], 'telegram:user:7')
        async with self.store.pool.connection() as conn:
            pending = await (await conn.execute('SELECT pending_bytes FROM profile_inputs ORDER BY message_id')).fetchall()
            self.assertTrue(all(row['pending_bytes'] > 0 for row in pending))
            self.assertEqual((await (await conn.execute('SELECT count(*) AS n FROM profile_patches')).fetchone())['n'], 0)
            # Advance this isolated fixture's retry deadline without a wall-clock wait.
            await conn.execute("UPDATE jobs SET available_at=now() WHERE kind='memory_profile'")
        self.assertTrue(await self.worker._guarded(await self.claim(), self.worker._profile))
        second = json.loads(self.provider.requests[1]['messages'][0].parts[0].text)
        self.assertEqual(second, first, 'A rejected patch must not consume or relabel its original evidence.')
        for actor, expected, source in [('telegram:user:7', 'Prefers jasmine tea', tea),
                                        ('telegram:user:8', 'Prefers coffee', coffee)]:
            facts = await self.store.get_profile(self.session, actor)
            self.assertEqual([(fact['claim'], fact['asserted_by'], fact['source_ids']) for fact in facts],
                [(expected, actor, [source.db_id])])
        async with self.store.pool.connection() as conn:
            self.assertEqual((await (await conn.execute('SELECT sum(pending_bytes) AS n FROM profile_inputs')).fetchone())['n'], 0)
            self.assertEqual((await (await conn.execute('SELECT count(*) AS n FROM profile_patches')).fetchone())['n'], 1)
        self.assertEqual((await self.store.read_messages(self.session, [forwarded.db_id]))[0].message, forwarded.message)

    async def test_literal_participant_header_is_learned_without_generated_attachment_claims(self):
        generated = '[Message provenance: generated transport label]'
        operation = '[Attachment download finished]'
        literal = 'I prefer jasmine tea. [Message provenance: this is my literal example.]'
        attachment = 'I prefer diesel fuel. This is a quoted document excerpt.'
        incoming = ConversationMessage.user_text(literal, metadata={
            'source': 'telegram', 'source_chat_id': '100', 'source_message_id': '1',
            'actor_id': 'telegram:user:7', 'actor_kind': 'user', 'actor_name': 'Alex'})
        incoming.parts = [MessagePart(PartKind.TEXT, text=generated, origin='provenance'),
            MessagePart(PartKind.TEXT, text=operation, origin='auto_note'),
            MessagePart(PartKind.TEXT, text=literal),
            MessagePart(PartKind.TEXT, text=attachment, origin='attachment_excerpt')]
        source = await self.runtime.ingest_user_message(session_id=self.session, incoming_message=incoming)
        reopened = await self.new_store()
        self.worker.store = reopened
        self.provider.responses = [self.response(self.fact('telegram:user:7', 'Prefers jasmine tea', source.db_id))]
        job = await reopened.claim_profile_batch(session_id=self.session, lazy=True,
            max_bytes=self.worker.limits.profile_request_bytes, lease_seconds=self.worker.limits.lease_seconds)
        self.assertTrue(await self.worker._guarded([job], self.worker._profile))
        evidence = json.loads(self.provider.requests[0]['messages'][0].parts[0].text)['original_evidence']
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]['fragments'], [{'offset': len(generated) + len(operation) + 2, 'text': literal}])
        self.assertEqual(evidence[0]['total_characters'], len('\n'.join((generated, operation, literal, attachment))))
        self.assertTrue(evidence[0]['partial'], 'Learning selects participant words rather than the whole canonical body.')
        self.assertNotIn('diesel fuel', json.dumps(evidence))
        self.assertNotIn(operation, json.dumps(evidence))
        facts = await reopened.get_profile(self.session, 'telegram:user:7')
        self.assertEqual([(fact['claim'], fact['source_ids']) for fact in facts], [('Prefers jasmine tea', [source.db_id])])
        self.assertEqual((await reopened.read_messages(self.session, [source.db_id]))[0].message, source.message)
