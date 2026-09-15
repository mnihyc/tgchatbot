"""Older imports inform learning without becoming newer by ingestion order."""
from __future__ import annotations

from datetime import datetime
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.memory import MemoryService
from tgchatbot.core.memory_worker import MemoryWorker
from tgchatbot.domain.models import ProviderResponse
from tgchatbot.domain.profiles import profile_size
from tgchatbot.domain.identities import canonical_actor_id
from tgchatbot.tools.import_desktop import import_file


class ProfileSourceChronologyWorkflows(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.embeddings = SimpleNamespace(enabled=False)
        self.worker = MemoryWorker(store=self.store, embeddings=self.embeddings,
            providers={'openai': self.provider}, config=self.config)
        self.memory = MemoryService(self.store, self.embeddings, config=self.config.memory)
        self.memory.worker = self.worker

    async def import_messages(self, *messages):
        path = self.path / 'result.json'
        path.write_text(json.dumps({'name': 'Synthetic group', 'id': 100,
            'type': 'private_supergroup', 'messages': list(messages)}), encoding='utf-8')
        await import_file(self.store, path, chat_id=100, defaults=self.config.default_session_settings())
        return {int(row.message.metadata['source_message_id']): row.db_id
            for row in await self.store.list_canonical_messages(self.session)}

    @staticmethod
    def message(number, actor, date, text):
        return {'id': number, 'type': 'message', 'date': date,
            'date_unixtime': str(int(datetime.fromisoformat(date).timestamp())),
            'from': 'Alex', 'from_id': f'user{actor}', 'text': text}

    @staticmethod
    def fact(actor, claim, sources, *, valid_from=None):
        return {'subject_actor_id': f'telegram:user:{actor}', 'asserted_by': f'telegram:user:{actor}',
            'claim': claim, 'kind': 'explicit', 'status': 'active', 'source_ids': sources,
            'valid_from': valid_from, 'valid_to': None, 'supersedes': None,
            'reason': 'The named person explicitly states this enduring preference.'}

    @staticmethod
    def response(*additions):
        return ProviderResponse(final_text=json.dumps({'additions': list(additions), 'removals': []}))

    async def test_reverse_order_import_supplies_source_dates_and_keeps_actor_profiles_separate(self):
        sources = await self.import_messages(
            self.message(101, 11, '2026-09-01T08:00:00+08:00', 'I prefer jasmine tea now.'),
            self.message(102, 11, '2026-09-10T08:00:00+08:00', 'I always drink my jasmine tea unsweetened.'),
            self.message(103, 22, '2026-09-05T08:00:00+08:00', 'I prefer coffee now.'))
        self.provider.responses = [self.response(
            self.fact(11, 'Prefers unsweetened jasmine tea', [sources[101], sources[102]]),
            self.fact(22, 'Prefers coffee', [sources[103]]))]
        initial = await self.memory.fetch_profiles(self.session, ['telegram:user:11', 'telegram:user:22'])
        self.assertNotIn('refresh_error', initial)
        self.assertEqual(len(self.provider.requests), 1)
        initial_profiles = {canonical_actor_id(profile['actor_id']): profile for profile in initial['profiles']}

        # These older messages arrive after September has already been learned.
        # Both people share a display name and had different preferences then.
        sources = await self.import_messages(
            self.message(1, 11, '2026-01-01T08:00:00+08:00', 'I prefer coffee.'),
            self.message(2, 22, '2026-01-02T08:00:00+08:00', 'I prefer tea.'),
            self.message(3, 11, '2026-01-03T08:00:00+08:00', 'I collect fountain pens.'))

        async def learn_older_evidence(**kwargs):
            request = json.loads(kwargs['messages'][0].parts[0].text)
            profiles = {profile['actor_id']: profile for profile in request['current_profiles']}
            alice = profiles['telegram:user:11']['facts'][0]
            other = profiles['telegram:user:22']['facts'][0]
            self.assertEqual(alice['source_dates'], {
                'first': '2026-09-01T08:00:00+08:00', 'last': '2026-09-10T08:00:00+08:00'})
            self.assertEqual(other['source_dates'], {
                'first': '2026-09-05T08:00:00+08:00', 'last': '2026-09-05T08:00:00+08:00'})
            self.assertIsNone(alice['valid_from'], 'Supporting dates must not fabricate semantic validity.')
            self.assertEqual({item['speaker']['id'] for item in request['original_evidence']},
                {'telegram:user:11', 'telegram:user:22'})
            self.assertTrue(all(item['sent_at'].startswith('2026-01-')
                for item in request['original_evidence']))
            # The model boundary remains responsible for this semantic choice:
            # keep newer drink preferences, while learning independent old evidence.
            return self.response(self.fact(11, 'Collects fountain pens', [sources[3]]))

        with patch.object(self.provider, 'generate', AsyncMock(side_effect=learn_older_evidence)) as model:
            result = await self.memory.fetch_profiles(self.session, ['telegram:user:11', 'telegram:user:22'])
        model.assert_awaited_once()
        self.assertNotIn('refresh_error', result)
        profiles = {canonical_actor_id(profile['actor_id']): profile for profile in result['profiles']}
        self.assertEqual({fact['claim'] for fact in profiles['telegram:user:11']['facts']},
            {'Prefers unsweetened jasmine tea', 'Collects fountain pens'})
        self.assertEqual(profiles['telegram:user:22'], initial_profiles['telegram:user:22'])
        self.assertIn(initial_profiles['telegram:user:11']['facts'][0], profiles['telegram:user:11']['facts'])
        for profile in result['profiles']:
            self.assertLessEqual(profile_size(profile), self.config.memory.profile_bytes)
            for fact in profile['facts']:
                self.assertNotIn('source_dates', fact, 'Learning metadata must not expand the chat tool contract.')
        async with self.store.pool.connection() as conn:
            remaining = (await (await conn.execute('SELECT sum(pending_bytes) AS n FROM profile_inputs')).fetchone())['n']
        self.assertEqual(remaining, 0)

    async def test_source_dates_remain_separate_from_future_validity_and_survive_restart(self):
        next_year = datetime.now().year + 1
        sources = await self.import_messages(self.message(1, 11, '2026-01-01T08:00:00+08:00',
            f'From January {next_year}, use my work email for invitations.'))
        valid_from = f'{next_year}-01-01T00:00:00+08:00'
        self.provider.responses = [self.response(self.fact(11, 'Use work email for invitations',
            [sources[1]], valid_from=valid_from))]
        initial = await self.memory.fetch_profiles(self.session, ['telegram:user:11'])
        self.assertNotIn('refresh_error', initial)
        self.assertEqual(initial['profiles'][0]['facts'], [], 'A future preference is not yet active.')

        restarted = await self.new_store()
        snapshot = await restarted.fetch_profile_snapshot(self.session, ['telegram:user:11'],
            max_bytes=self.config.memory.profile_bytes, for_learning=True)
        fact = snapshot['profiles'][0]['facts'][0]
        self.assertEqual(fact['source_dates']['first'].isoformat(), '2026-01-01T00:00:00+00:00')
        self.assertEqual(fact['source_dates']['last'], fact['source_dates']['first'])
        self.assertEqual(fact['valid_from'], datetime.fromisoformat(valid_from))
        current = await restarted.fetch_profile_snapshot(self.session, ['telegram:user:11'],
            max_bytes=self.config.memory.profile_bytes)
        self.assertEqual(current['profiles'][0]['facts'], [])

        # A later, unrelated message affects the person's last-message date but
        # must not alter the dates of this fact's own supporting evidence.
        await self.import_messages(self.message(2, 11, '2026-09-01T08:00:00+08:00', 'Good morning!'))
        later = await restarted.fetch_profile_snapshot(self.session, ['telegram:user:11'],
            max_bytes=self.config.memory.profile_bytes, for_learning=True)
        self.assertEqual(later['profiles'][0]['facts'][0], fact)
        self.assertGreater(later['profiles'][0]['identity']['last_message']['sent_at'],
            fact['source_dates']['last'])
