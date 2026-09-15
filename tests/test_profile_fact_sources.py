"""Compact profile references open original evidence without learning or crossing scope."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from unittest.mock import patch

from tests.business_helpers import BusinessTestCase
from tgchatbot.domain.models import ConversationMessage
from tgchatbot.storage.postgres_store import StaleScopeError
from tgchatbot.tools.import_desktop import import_file


class ProfileFactSourceWorkflows(BusinessTestCase):
    async def source(self, number, text, *, actor='telegram:user:7', session=None):
        session = session or self.session
        return await self.store.append_message(session, ConversationMessage.user_text(text, metadata={
            'source': 'telegram', 'source_chat_id': session.removeprefix('telegram:'),
            'source_message_id': str(number), 'actor_id': actor, 'actor_name': 'Alex', 'actor_kind': 'user',
            'sent_at': '2025-01-01T00:00:00+00:00'}))

    async def fact(self, *sources, claim='Prefers jasmine tea', **fields):
        return await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7',
            asserted_by='telegram:user:7', claim=claim, source_ids=[source.db_id for source in sources], **fields)

    async def state(self):
        async with self.store.pool.connection() as conn:
            return {table: await (await conn.execute(f'SELECT * FROM {table} ORDER BY 1')).fetchall()
                for table in ('sessions', 'jobs', 'profile_inputs', 'profile_current', 'profile_facts')}

    async def test_current_and_retired_facts_open_evidence_without_consuming_learning(self):
        first = await self.source(1, 'I prefer jasmine tea.')
        old = await self.fact(first)
        second = await self.source(2, 'I collect fountain pens.')
        current = await self.fact(second, claim='Collects fountain pens')
        job = await self.store.claim_profile_batch(session_id=self.session, lazy=True,
            max_bytes=12000, lease_seconds=900)
        await self.store.apply_profile_patch(job, [], [{'fact_id': old['id'], 'reason': 'No longer useful in the compact profile.'}],
            max_bytes=self.config.memory.profile_bytes, expected_source_revisions=job['source_revisions'])
        await self.source(3, 'A new preference waiting for the ordinary learner.')
        before = await self.state()
        scope = await self.store.get_existing_scope(self.session)
        result = await self.store.resolve_profile_fact_sources(self.session,
            [current['id'], old['id'], current['id']], expected_scope=scope)
        self.assertEqual(result, [
            {'fact_id': old['id'], 'actor_id': 'telegram:user:7', 'source_ids': [first.db_id], 'current': False},
            {'fact_id': current['id'], 'actor_id': 'telegram:user:7', 'source_ids': [second.db_id], 'current': True}])
        originals = await self.store.read_messages(self.session, result[0]['source_ids'])
        self.assertEqual(originals[0].message.parts[0].text, 'I prefer jasmine tea.')
        self.assertEqual(await self.state(), before)
        self.assertEqual(self.provider.requests, [])

    async def test_superseded_expired_and_future_facts_remain_readable_without_becoming_current(self):
        now = datetime.now(timezone.utc)
        original = await self.source(1, 'I prefer tea for now.')
        prior = await self.fact(original, valid_from=now - timedelta(days=10))
        correction = await self.source(2, 'I now prefer coffee.')
        current = await self.fact(correction, claim='Prefers coffee', supersedes=prior['id'],
            valid_from=now - timedelta(days=1))
        expiry = await self.source(3, 'Use my temporary email until yesterday.')
        expired = await self.fact(expiry, claim='Use temporary email', valid_to=now - timedelta(days=1))
        planned = await self.source(4, 'From tomorrow, use my work email.')
        future = await self.fact(planned, claim='Use work email', valid_from=now + timedelta(days=1))
        facts = [prior, current, expired, future]
        result = await self.store.resolve_profile_fact_sources(self.session, [fact['id'] for fact in facts])
        self.assertEqual({row['fact_id']: row['current'] for row in result},
            {prior['id']: False, current['id']: True, expired['id']: False, future['id']: False})
        self.assertEqual({mid for row in result for mid in row['source_ids']},
            {original.db_id, correction.db_id, expiry.db_id, planned.db_id})

    async def test_fact_references_survive_soft_reset_but_never_cross_chats_or_full_reset(self):
        original = await self.source(1, 'I enjoy warm tea.')
        fact = await self.fact(original)
        previous = await self.store.get_existing_scope(self.session)
        self.assertEqual(previous, await self.store.get_scope(self.session))
        await self.store.reset_context(self.session)
        scope = await self.store.get_existing_scope(self.session)
        result = await self.store.resolve_profile_fact_sources(self.session, [fact['id']], expected_scope=scope)
        self.assertEqual(result[0]['source_ids'], [original.db_id])
        with self.assertRaises(StaleScopeError):
            await self.store.resolve_profile_fact_sources(self.session, [fact['id']], expected_scope=previous)
        await self.source(1, 'Unrelated conversation.', session='telegram:200')
        self.assertEqual(await self.store.resolve_profile_fact_sources('telegram:200', [fact['id']]), [])
        await self.store.reset_full(self.session, self.config.default_session_settings())
        self.assertEqual(await self.store.resolve_profile_fact_sources(self.session, [fact['id']]), [])
        with self.assertRaises(StaleScopeError):
            await self.store.resolve_profile_fact_sources(self.session, [fact['id']], expected_scope=scope)
        self.assertEqual(len(await self.store.list_message_revisions(self.session, original.db_id)), 1)

    async def test_missing_conversation_lookups_do_not_create_a_session(self):
        before = await self.store.count_sessions()
        self.assertIsNone(await self.store.get_existing_scope('telegram:404'))
        self.assertEqual(await self.store.resolve_profile_fact_sources('telegram:404', [123]), [])
        with self.assertRaises(StaleScopeError):
            await self.store.resolve_profile_fact_sources('telegram:404', [123], expected_scope={'generation': 1})
        with self.assertRaises(StaleScopeError):
            await self.store.assert_scope('telegram:404', {'generation': 1})
        self.assertEqual(await self.store.count_sessions(), before)

    async def test_any_revised_hidden_or_deleted_source_makes_the_fact_unavailable(self):
        for number, change in enumerate(('edit', 'hide', 'delete'), start=1):
            with self.subTest(change=change):
                source = await self.source(number * 10, 'I prefer tea.')
                companion = await self.source(number * 10 + 1, 'I drink it unsweetened.')
                fact = await self.fact(source, companion, claim=f'Tea preference {number}')
                self.assertEqual(len(await self.store.resolve_profile_fact_sources(self.session, [fact['id']])), 1)
                if change == 'edit':
                    await self.source(number * 10, 'Correction: I prefer coffee.')
                elif change == 'hide':
                    await self.store.hide_message_ids(self.session, [source.db_id])
                else:
                    await self.store.delete_message_ids(self.session, [source.db_id])
                self.assertEqual(await self.store.resolve_profile_fact_sources(self.session, [fact['id']]), [])
                # A stale validity flag cannot make changed/hidden originals
                # masquerade as the evidence supporting the old profile claim.
                async with self.store.pool.connection() as conn:
                    await conn.execute('UPDATE profile_facts SET valid=true WHERE id=%s', (fact['id'],))
                self.assertEqual(await self.store.resolve_profile_fact_sources(self.session, [fact['id']]), [])

    async def test_reverse_import_sorts_fact_sources_by_original_time_then_id(self):
        path = self.path / 'result.json'
        path.write_text(json.dumps({'name': 'Synthetic group', 'id': 100, 'type': 'private_supergroup',
            'messages': [{'id': number, 'type': 'message', 'from': 'Alex', 'from_id': 'user7',
                'date': date, 'text': text} for number, date, text in (
                (10, '2026-09-01T08:00:00+08:00', 'I still prefer tea.'),
                (11, '2020-01-01T08:00:00+08:00', 'I prefer tea.'),
                (12, '2020-01-01T08:00:00+08:00', 'I drink it unsweetened.'))]}))
        await import_file(self.store, path, chat_id=100)
        sources = await self.store.list_canonical_messages(self.session)
        fact = await self.fact(*sources)
        result = await self.store.resolve_profile_fact_sources(self.session, [fact['id']])
        self.assertEqual(result[0]['source_ids'], [sources[1].db_id, sources[2].db_id, sources[0].db_id])

    async def test_scope_change_during_resolution_is_detected_before_return(self):
        source = await self.source(1, 'I prefer tea.')
        fact = await self.fact(source)
        scope = await self.store.get_existing_scope(self.session)
        original_read = self.store.get_existing_scope
        async def reset_before_recheck(session_id):
            await self.store.reset_full(session_id, self.config.default_session_settings())
            return await original_read(session_id)
        with patch.object(self.store, 'get_existing_scope', side_effect=reset_before_recheck):
            with self.assertRaises(StaleScopeError):
                await self.store.resolve_profile_fact_sources(self.session, [fact['id']], expected_scope=scope)
