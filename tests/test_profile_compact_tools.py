"""Compact profiles defer evidence without losing ownership or replay fidelity."""
from __future__ import annotations

import copy
from dataclasses import replace
from datetime import datetime, timedelta
from types import SimpleNamespace

from tests.business_helpers import BusinessTestCase
from tests.test_memory_image_tools import PIXEL
from tgchatbot.core.memory import MemoryService
from tgchatbot.core.memory_worker import MemoryWorker
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import ConversationMessage, MessagePart, PartKind
from tgchatbot.domain.identities import actor_reference
from tgchatbot.tools.base import ToolContext


class CompactProfileToolWorkflows(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.settings()
        self.actor = 'telegram:user:101'
        self.other = 'telegram:user:102'
        self.embeddings = SimpleNamespace(enabled=False)
        self.memory = MemoryService(self.store, self.embeddings)
        self.runtime.memory = self.memory
        self.context = ToolContext(self.session, 'Participant', timezone='Asia/Singapore')

    async def source(self, number, text, *, actor=None, image=False):
        message = ConversationMessage.user_text(text, metadata={
            'source': 'telegram', 'source_chat_id': '100', 'source_message_id': str(number),
            'actor_id': actor or self.actor, 'actor_kind': 'user', 'actor_name': 'Alex',
            'sent_at': '2026-01-02T03:04:05+00:00'})
        if image:
            message.parts.append(MessagePart(PartKind.IMAGE, mime_type='image/png', data_b64=PIXEL))
        return await self.runtime.ingest_user_message(session_id=self.session, incoming_message=message)

    async def fact(self, claim, *sources, actor=None, asserted_by=None, **kwargs):
        return await self.store.save_profile_fact(self.session,
            subject_actor_id=actor or self.actor, asserted_by=asserted_by or self.actor,
            claim=claim, source_ids=[source.db_id for source in sources], **kwargs)

    async def tool_result(self, name, args, *, memory=None):
        spec = next(tool for tool in (memory or self.memory).tools if tool.name == name)
        result = await spec.runner.run(args, self.context)
        self.assertTrue(result.output['ok'], result.output)
        return result

    async def tool(self, name, args, **kwargs):
        return (await self.tool_result(name, args, **kwargs)).output

    async def pending_profile_evidence(self):
        async with self.store.pool.connection() as conn:
            return await (await conn.execute('SELECT message_id,spans,pending_bytes '
                'FROM profile_inputs ORDER BY message_id')).fetchall()

    async def test_compact_fetch_preserves_claims_qualifiers_dates_and_different_asserters(self):
        own = await self.source(1, 'I prefer jasmine tea. Please keep your replies concise.')
        observer = await self.source(2, 'Alex seems to enjoy quiet places.', actor=self.other)
        direct = await self.fact('Prefers jasmine tea.', own,
            valid_from='2024-01-01T00:00:00Z', valid_to='2099-01-01T00:00:00Z')
        inferred = await self.fact('May enjoy quiet places.', observer,
            asserted_by=self.other, kind='inferred')
        style = await self.fact('Keep replies concise.', own, actor='agent')
        before = await self.store.fetch_profile_snapshot(self.session, [self.actor, 'agent'], for_learning=True)

        output = await self.tool('user_profile_fetch', {'actor_ids': [self.actor]})
        profiles = {profile['actor_id']: profile for profile in output['profiles']}
        current = {fact['fact_id']: fact for profile in profiles.values() for fact in profile['facts']}
        self.assertEqual(set(current), {direct['id'], inferred['id'], style['id']})
        self.assertEqual(current[direct['id']]['claim'], direct['claim'])
        self.assertEqual(current[direct['id']]['kind'], 'explicit')
        self.assertNotIn('asserted_by', current[direct['id']], 'The containing person is the direct asserter.')
        self.assertEqual(current[inferred['id']]['asserted_by'], actor_reference(self.other))
        self.assertEqual(current[inferred['id']]['kind'], 'inferred')
        self.assertEqual(current[style['id']]['asserted_by'], actor_reference(self.actor),
            'An agent style preference must retain the person who requested it.')
        self.assertEqual(profiles['agent']['subject_kind'], 'agent_preferences')
        for key in ('valid_from', 'valid_to'):
            self.assertEqual(datetime.fromisoformat(current[direct['id']][key]).utcoffset(), timedelta(hours=8))
        for fact in current.values():
            self.assertNotIn('source_ids', fact)
            self.assertNotIn('id', fact)
            self.assertNotIn(None, fact.values(), 'Empty chronology adds no information to the runtime view.')

        after = await self.store.fetch_profile_snapshot(self.session, [self.actor, 'agent'], for_learning=True)
        self.assertEqual(after['profiles'], before['profiles'])
        self.assertEqual(after['facts'], before['facts'])
        learned = {fact['id']: fact for profile in after['profiles'] for fact in profile['facts']}
        self.assertEqual(learned[direct['id']]['source_ids'], [own.db_id])
        self.assertEqual(learned[inferred['id']]['source_ids'], [observer.db_id])
        self.assertTrue(learned[direct['id']]['source_dates'])
        self.assertEqual(self.provider.requests, [])

    async def test_fact_evidence_read_deduplicates_originals_without_consuming_lazy_learning(self):
        source = await self.source(1, 'I prefer jasmine tea. Please keep your replies concise.')
        tea = await self.fact('Prefers jasmine tea.', source)
        style = await self.fact('Keep replies concise.', source, actor='agent')
        await self.source(2, 'I now prefer coffee, so the pending batch will need to reconcile this.')
        self.memory.worker = MemoryWorker(store=self.store, embeddings=self.embeddings,
            providers={'openai': self.provider}, config=self.config)
        pending = await self.pending_profile_evidence()
        self.assertTrue(any(row['pending_bytes'] > 0 for row in pending))

        spec = next(tool for tool in self.memory.tools if tool.name == 'memory_read')
        args = {key: None for key in spec.openai_tool()['parameters']['required']}
        missing = max(tea['id'], style['id']) + 1000
        args['profile_fact_ids'] = [tea['id'], style['id'], tea['id'], missing]
        result = await self.tool('memory_read', args)
        self.assertEqual(result['profile_facts'], [
            {'fact_id': tea['id'], 'actor_id': actor_reference(self.actor), 'source_ids': [source.db_id], 'current': True},
            {'fact_id': style['id'], 'actor_id': 'agent', 'source_ids': [source.db_id], 'current': True}])
        self.assertEqual(result['unavailable_profile_fact_ids'], [missing])
        self.assertEqual([row['message_id'] for row in result['messages']], [source.db_id])
        self.assertEqual(result['messages'][0]['speaker']['id'], actor_reference(self.actor))
        self.assertEqual(result['messages'][0]['fragments'], [{'offset': 0, 'text': source.message.parts[0].text}])
        self.assertEqual(await self.pending_profile_evidence(), pending)
        self.assertEqual(self.provider.requests, [], 'Reading citations must not start a learning batch.')

    async def test_mixed_explicit_messages_and_fact_sources_offer_complete_followup_ids(self):
        sources = [await self.source(number, text) for number, text in enumerate(
            ('One dated statement.', 'Second dated statement.', 'Third dated statement.'), start=1)]
        fact = await self.fact('A preference supported over several occasions.', *sources)
        direct = await self.source(4, 'The immediate discussion has its own context.')
        # Deliberately small test-only windows exercise both ID and character continuation.
        self.memory.config = replace(self.memory.config, read_messages=2, read_chars=7)
        result = await self.tool('memory_read', {'message_ids': [direct.db_id], 'profile_fact_ids': [fact['id']]})
        self.assertEqual(result['profile_facts'][0]['source_ids'], [source.db_id for source in sources])
        self.assertEqual([row['message_id'] for row in result['messages']], [direct.db_id, sources[0].db_id])
        self.assertEqual(result['unavailable_ids'], [])
        self.assertEqual(result['omitted_ids'], [source.db_id for source in sources[1:]])
        self.assertTrue(all(row['next_offset'] == 7 for row in result['messages']))
        remaining = await self.tool('memory_read', {'message_ids': result['omitted_ids']})
        self.assertEqual([row['message_id'] for row in remaining['messages']], [source.db_id for source in sources[1:]])
        tail = await self.tool('memory_read', {'message_ids': [sources[0].db_id], 'offset': 7})
        self.assertEqual(tail['messages'][0]['fragments'], [{'offset': 7, 'text': 'ed stat'}])
        spec = next(tool for tool in self.memory.tools if tool.name == 'memory_read')
        invalid = await spec.runner.run({'message_ids': [source.db_id for source in sources]}, self.context)
        self.assertFalse(invalid.output['ok'], 'The existing explicit-message request limit remains unchanged.')

    async def test_unavailable_fact_only_read_is_explicit_without_invalid_empty_message_error(self):
        for neighbors in (False, True):
            output = await self.tool('memory_read', {'profile_fact_ids': [999999], 'include_neighbors': neighbors})
            self.assertEqual(output['profile_facts'], [])
            self.assertEqual(output['unavailable_profile_fact_ids'], [999999])
            self.assertEqual(output['messages'], [])
            self.assertEqual(output['unavailable_ids'], [])
            self.assertEqual(output['omitted_ids'], [])
        count = await self.store.count_sessions()
        self.context = replace(self.context, session_id='telegram:unknown')
        output = await self.tool('memory_read', {'profile_fact_ids': [999999], 'include_neighbors': True})
        self.assertEqual(output['unavailable_profile_fact_ids'], [999999])
        self.assertEqual(await self.store.count_sessions(), count)

    async def test_fact_discovery_exposes_images_without_implicitly_selecting_them(self):
        source = await self.source(1, 'This is my favorite blue teapot.', image=True)
        fact = await self.fact('Owns a favorite blue teapot.', source)
        discovered = await self.tool_result('memory_read', {'profile_fact_ids': [fact['id']]})
        image_id = discovered.output['messages'][0]['images'][0]['image_id']
        self.assertFalse(discovered.evidence_parts)
        implicit = await self.tool_result('memory_read', {'profile_fact_ids': [fact['id']], 'image_ids': [image_id]})
        self.assertEqual(implicit.output['image_results'][0]['status'], 'unavailable')
        self.assertFalse(implicit.evidence_parts)
        explicit = await self.tool_result('memory_read', {'message_ids': [source.db_id],
            'profile_fact_ids': [fact['id']], 'image_ids': [image_id]})
        self.assertEqual(explicit.output['image_results'][0]['status'], 'selected')
        self.assertEqual([part.kind for part in explicit.evidence_parts], [PartKind.TEXT, PartKind.IMAGE])

    async def test_typed_actor_references_reopen_profiles_and_search_without_crossing_reset(self):
        tea = await self.source(1, 'My preferred drink is jasmine tea.')
        coffee = await self.source(2, 'My preferred drink is black coffee.', actor=self.other)
        await self.fact('Prefers jasmine tea.', tea)
        await self.fact('Prefers black coffee.', coffee, actor=self.other, asserted_by=self.other)
        await self.store.reset_context(self.session)
        reopened = await self.new_store()
        memory = MemoryService(reopened, self.embeddings)
        result = await self.tool('user_profile_fetch', {'actor_ids': [actor_reference(self.actor), self.actor,
            actor_reference(self.other)], 'include_agent_preferences': False}, memory=memory)
        self.assertEqual([profile['actor_id'] for profile in result['profiles']],
                         [actor_reference(self.actor), actor_reference(self.other)])
        self.assertEqual([profile['facts'][0]['claim'] for profile in result['profiles']],
                         ['Prefers jasmine tea.', 'Prefers black coffee.'])
        self.assertEqual(set(result), {'ok', 'as_of', 'profiles'})
        self.assertNotIn('source_revision', result['profiles'][0]['identity']['last_message'])
        search = await self.tool('memory_search', {'query': 'preferred drink', 'actor_id': actor_reference(self.actor)}, memory=memory)
        self.assertEqual(search, await memory.search(self.session, 'preferred drink', actor_id=self.actor,
            timezone='Asia/Singapore'))
        self.assertIn(tea.db_id, [mid for match in search['matches'] for mid in match['message_ids']])
        self.assertEqual({message['speaker']['id'] for message in search['messages']},
                         {actor_reference(self.actor), actor_reference(self.other)})
        self.assertEqual((await reopened.read_messages(self.session, [tea.db_id]))[0].message.metadata['actor_id'], self.actor)
        await reopened.reset_full(self.session, self.config.default_session_settings())
        current = await self.tool('user_profile_fetch', {'actor_ids': [actor_reference(self.actor)],
            'include_agent_preferences': False}, memory=memory)
        self.assertEqual(current['profiles'][0]['status'], 'unknown_identity')
        self.assertEqual(current['profiles'][0]['facts'], [])
        self.assertEqual((await memory.search(self.session, 'preferred drink', actor_id=actor_reference(self.actor)))['messages'], [])

    async def test_restart_and_compaction_keep_recorded_full_payload_and_append_compact_refresh(self):
        source = await self.source(1, 'I prefer jasmine tea.')
        fact = await self.fact('Prefers jasmine tea.', source)
        full = await self.store.fetch_profile_snapshot(self.session, [self.actor])
        legacy = {'ok': True, 'profiles': copy.deepcopy(full['profiles'])}
        pair = []
        for phase, payload in (
            ('call', {'call_id': 'earlier-full-profile', 'arguments': {'actor_ids': [self.actor]}}),
            ('result', {'call_id': 'earlier-full-profile', 'output': legacy}),
        ):
            pair.append(await self.runtime.record_tool_observation(session_id=self.session,
                name='user_profile_fetch', phase=phase, payload=payload))
        canonical_pair = await self.store.read_messages(self.session, [row.db_id for row in pair])
        warm = copy.deepcopy((await self.runtime._get_live_state(self.session)).raw_messages)
        self.runtime.invalidate_session(self.session)
        self.assertEqual((await self.runtime._get_live_state(self.session)).raw_messages, warm)

        await self.store.create_memory_block(self.session, summary_text='Earlier tea preference.',
            estimated_tokens=5, source_message_ids=[source.db_id])
        await self.store.close()
        reopened = await self.new_store()
        memory = MemoryService(reopened, self.embeddings)
        runtime = AgentRuntime(config=self.config, store=reopened, tool_registry=self.tools,
            providers={'openai': self.provider}, memory=memory)
        await runtime.prepare_context(session_id=self.session)
        originals = await reopened.read_messages(self.session, [row.db_id for row in pair])
        self.assertEqual([row.message for row in originals], [row.message for row in canonical_pair])
        earlier_fact = originals[-1].message.metadata['tool_payload']['output']['profiles'][0]['facts'][0]
        self.assertEqual(earlier_fact['source_ids'], [source.db_id])
        self.assertEqual(earlier_fact['id'], fact['id'])
        rows = await reopened.list_uncompacted_messages(self.session)
        refreshed = [row for row in rows if row.message.metadata.get('synthetic_role') == 'profile_refresh']
        self.assertEqual(len(refreshed), 2)
        compact = refreshed[-1].message.metadata['tool_payload']['output']['profiles'][0]['facts'][0]
        self.assertEqual(compact['fact_id'], fact['id'])
        self.assertEqual(compact['claim'], fact['claim'])
        self.assertNotIn('source_ids', compact)
        evidence = await self.tool('memory_read', {'profile_fact_ids': [fact['id']]}, memory=memory)
        self.assertEqual(evidence['messages'][0]['message_id'], source.db_id)
        self.assertEqual(self.provider.requests, [])
