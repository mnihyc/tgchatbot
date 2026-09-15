"""Profile batches preserve complete short utterances and bounded long sources."""
from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.memory_worker import MemoryWorker
from tgchatbot.domain.models import ConversationMessage, ProviderResponse


class ProfileBatchBoundaries(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.settings()
        self.worker = MemoryWorker(store=self.store, embeddings=SimpleNamespace(enabled=False),
            providers={'openai': self.provider}, config=self.config)
        self.evidence = []
        self.actor = 'telegram:user:7'

    async def source(self, number, text):
        stored = await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text(text, metadata={
                'source': 'telegram', 'source_chat_id': '100', 'source_message_id': str(number),
                'actor_id': self.actor, 'actor_kind': 'user', 'actor_name': 'Participant',
                'sent_at': f'2026-01-01T00:00:{number:02d}+00:00'}))
        # Learning must preserve canonical originals independently of the
        # versioned working presentation returned by live intake.
        return (await self.store.read_messages(self.session, [stored.db_id]))[0]

    def model(self, declarations):
        async def generate(**request):
            payload = json.loads(request['messages'][0].parts[0].text)
            self.evidence.append(payload['original_evidence'])
            current = {fact['claim']: fact for profile in payload['current_profiles']
                if profile['actor_id'] == self.actor for fact in profile['facts']}
            additions = []
            for item in payload['original_evidence']:
                declaration = declarations.get(''.join(fragment['text'] for fragment in item['fragments']))
                if declaration is None:
                    continue
                claim, replaces = declaration
                old = current.get(replaces)
                additions.append({'subject_actor_id': self.actor, 'asserted_by': self.actor,
                    'claim': claim, 'kind': 'explicit', 'status': 'active',
                    'source_ids': [item['message_id']], 'valid_from': None, 'valid_to': None,
                    'supersedes': old['id'] if old else None,
                    'reason': 'The complete direct utterance supports this preference or correction.'})
            return ProviderResponse(final_text=json.dumps({'additions': additions, 'removals': []}))
        self.provider.generate = generate

    async def background(self):
        job = await self.store.claim_profile_batch(session_id=self.session,
            max_bytes=self.worker.limits.profile_request_bytes,
            lease_seconds=self.worker.limits.lease_seconds)
        self.assertIsNotNone(job)
        self.assertTrue(await self.worker._guarded([job], self.worker._profile), self.worker.last_error)

    async def pending(self, source):
        async with self.store.pool.connection() as conn:
            return (await (await conn.execute('SELECT spans,pending_bytes FROM profile_inputs WHERE message_id=%s',
                (source.db_id,))).fetchone())

    async def test_complete_short_preference_waits_for_next_batch_and_is_learned_once(self):
        text = 'I prefer quiet evening walks.'
        self.worker.limits = replace(self.worker.limits, profile_request_bytes=len(text.encode()) + 5)
        self.model({text: ('Prefers quiet evening walks.', None)})
        await self.source(1, 'A normal short update.')
        declaration = await self.source(2, text)
        await self.background()
        self.assertEqual(await self.pending(declaration), {
            'spans': [{'start': 0, 'end': len(text)}], 'pending_bytes': len(text.encode())})
        self.assertEqual(await self.store.get_profile(self.session, self.actor), [])
        before = len(self.evidence)
        await self.worker.refresh_profiles(self.session, [self.actor, 'agent'])
        self.assertEqual(len(self.evidence), before + 1, 'Explicit refresh processes one batch')
        facts = await self.store.get_profile(self.session, self.actor)
        self.assertEqual([(fact['claim'], fact['source_ids']) for fact in facts],
            [('Prefers quiet evening walks.', [declaration.db_id])])
        self.assertEqual(self.evidence[-1][0]['fragments'], [{'offset': 0, 'text': text}])
        await self.worker.refresh_profiles(self.session, [self.actor, 'agent'])
        self.assertEqual(len(self.evidence), before + 1, 'Consumed evidence is not learned again')
        self.assertEqual((await self.store.read_messages(self.session, [declaration.db_id]))[0].message,
            declaration.message)

    async def test_complete_activity_correction_replaces_only_prior_activity_after_boundary(self):
        old = 'I prefer quiet evening walks.'
        tea = 'I prefer jasmine tea.'
        correction = 'I now prefer evening cycling instead of evening walks.'
        self.worker.limits = replace(self.worker.limits, profile_request_bytes=len(correction.encode()) + 5)
        self.model({old: ('Prefers quiet evening walks.', None), tea: ('Prefers jasmine tea.', None),
            correction: ('Prefers evening cycling.', 'Prefers quiet evening walks.')})
        old_source = await self.source(1, old)
        await self.source(2, tea)
        await self.worker.refresh_profiles(self.session, [self.actor, 'agent'])
        earlier = await self.store.get_profile(self.session, self.actor)
        self.assertEqual({fact['claim'] for fact in earlier},
            {'Prefers quiet evening walks.', 'Prefers jasmine tea.'})
        old_fact = next(fact for fact in earlier if fact['claim'] == 'Prefers quiet evening walks.')
        await self.source(3, 'A normal short update.')
        corrected = await self.source(4, correction)
        await self.background()
        self.assertEqual((await self.pending(corrected))['pending_bytes'], len(correction.encode()))
        self.assertEqual({fact['claim'] for fact in await self.store.get_profile(self.session, self.actor)},
            {'Prefers quiet evening walks.', 'Prefers jasmine tea.'})
        await self.worker.refresh_profiles(self.session, [self.actor, 'agent'])
        current = await self.store.get_profile(self.session, self.actor)
        self.assertEqual({fact['claim'] for fact in current},
            {'Prefers evening cycling.', 'Prefers jasmine tea.'})
        new = next(fact for fact in current if fact['claim'] == 'Prefers evening cycling.')
        self.assertEqual((new['supersedes'], new['source_ids']), (old_fact['id'], [corrected.db_id]))
        async with self.store.pool.connection() as conn:
            prior = await (await conn.execute('SELECT status,valid_to FROM profile_facts WHERE id=%s',
                (old_fact['id'],))).fetchone()
        self.assertEqual(prior['status'], 'superseded')
        self.assertIsNotNone(prior['valid_to'])
        self.assertEqual((await self.store.read_messages(self.session, [old_source.db_id]))[0].message,
            old_source.message)

    async def test_oversized_multibyte_original_still_fragments_with_exact_remaining_spans(self):
        text = '我喜欢安静的地方与有趣的企鹅🙂。' * 5
        maximum = 32
        self.worker.limits = replace(self.worker.limits, profile_request_bytes=maximum)
        self.model({})
        await self.source(1, 'Hello.')
        original = await self.source(2, text)
        await self.background()
        self.assertEqual((await self.pending(original))['pending_bytes'], len(text.encode()))
        while (await self.pending(original))['pending_bytes']:
            before = (await self.pending(original))['pending_bytes']
            await self.worker.refresh_profiles(self.session, [self.actor, 'agent'])
            self.assertLess((await self.pending(original))['pending_bytes'], before)
        fragments = [item for batch in self.evidence for item in batch if item['message_id'] == original.db_id]
        self.assertGreater(len(fragments), 1)
        self.assertEqual(''.join(fragment['text'] for item in fragments for fragment in item['fragments']), text)
        self.assertTrue(all(sum(len(fragment['text'].encode()) for item in batch
            for fragment in item['fragments']) <= maximum for batch in self.evidence))
        cursor = 0
        for item in fragments:
            self.assertTrue(item['partial'])
            self.assertEqual(item['total_characters'], len(text))
            for fragment in item['fragments']:
                self.assertEqual(fragment['offset'], cursor)
                cursor += len(fragment['text'])
        self.assertEqual(cursor, len(text))
        self.assertEqual((await self.store.read_messages(self.session, [original.db_id]))[0].message,
            original.message)
