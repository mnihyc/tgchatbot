"""Memory lookups remain replayable history without becoming new evidence."""
from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.memory import MemoryService
from tgchatbot.domain.models import ConversationMessage, MessageRole
from tgchatbot.storage.postgres_store import message_body
from tgchatbot.tools.base import ToolContext


class MemoryToolEvidenceTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.settings()
        # This experiment isolates lexical evidence; no remote API is called.
        self.memory = MemoryService(self.store, SimpleNamespace(enabled=False))
        self.context = ToolContext(self.session, 'Participant')

    async def tool(self, name, arguments):
        tool = next(tool for tool in self.memory.tools if tool.name == name)
        result = (await tool.runner.run(arguments, self.context)).output
        self.assertTrue(result['ok'])
        return result

    async def search_ids(self, query='saffron', limit=20):
        self.memory.config = replace(self.memory.config, search_results=limit)
        result = await self.tool('memory_search', {'query': query})
        return [row['message_ids'][0] for row in result['matches']]

    async def test_strict_provider_null_read_options_return_the_same_original_as_omitted_options(self):
        original = await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text('Saffron delivery is postponed until Thursday.'))
        spec = next(tool for tool in self.memory.tools if tool.name == 'memory_read')
        schema = spec.openai_tool()['parameters']
        arguments = {key: None for key in schema['required'] if key != 'message_ids'}
        arguments['message_ids'] = [original.db_id]
        self.assertIn('null', schema['properties']['offset']['type'])
        omitted = await self.tool('memory_read', {'message_ids': [original.db_id]})
        nullable = await self.tool('memory_read', arguments)
        self.assertEqual(nullable, omitted)
        self.assertEqual(nullable['messages'][0]['fragments'], [
            {'offset': 0, 'text': 'Saffron delivery is postponed until Thursday.'}])
        paged = await self.tool('memory_read', {**arguments, 'offset': 8, 'length': 8})
        self.assertEqual(paged['messages'][0]['fragments'], [{'offset': 8, 'text': 'delivery'}])
        for invalid_offset in (-1, ''):
            invalid = await spec.runner.run({**arguments, 'offset': invalid_offset}, self.context)
            self.assertFalse(invalid.output['ok'], 'Only an omitted/null offset receives the default')

    async def test_repeated_lookups_do_not_displace_originals_or_external_observations(self):
        fact = await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text('Saffron delivery is postponed until Thursday.'))
        question = await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text('Any update on saffron?'))
        answer = await self.runtime.record_assistant_text(session_id=self.session,
            text='I will remember the saffron arrangement.')
        external = await self.runtime.record_tool_observation(session_id=self.session,
            name='shell_exec', phase='result', payload={'output': {
                'stdout': 'Saffron supplier status: awaiting pickup.'}})
        original_ids = [row.db_id for row in (fact, question, answer, external)]
        self.assertCountEqual(await self.search_ids(limit=4), original_ids)
        search_snapshot = await self.tool('memory_search', {'query': 'saffron'})
        read_snapshot = await self.tool('memory_read', {'message_ids': [fact.db_id]})

        derived = []
        for name, arguments, output in (
            ('memory_search', {'query': 'saffron'}, search_snapshot),
            ('memory_read', {'message_ids': [fact.db_id]}, read_snapshot),
            ('user_profile_fetch', {'actor_ids': ['telegram:user:101']}, {'ok': True, 'profiles': [{
                'actor_id': 'telegram:user:101', 'facts': [{'text': 'Prefers saffron dishes.',
                    'source_ids': [fact.db_id]}]}]}),
        ):
            derived.append(await self.runtime.record_tool_observation(session_id=self.session,
                name=name, phase='call', payload={'call_id': name, 'arguments': arguments}))
            derived.append(await self.runtime.record_tool_observation(session_id=self.session,
                name=name, phase='result', payload={'call_id': name, 'output': output}))

        self.assertCountEqual(await self.search_ids(limit=4), original_ids,
            'Remembering a lookup must not replace the evidence that the lookup returned')
        # Call records without the query term also must not become search evidence.
        for name in ('memory_search', 'memory_read', 'user_profile_fetch'):
            self.assertEqual(await self.search_ids(query=name), [])

        all_rows = [fact, question, answer, external, *derived]
        read = await self.tool('memory_read', {'message_ids': [row.db_id for row in all_rows]})
        self.assertEqual(read['unavailable_ids'], [])
        self.assertEqual({row['message_id']: row['fragments'] for row in read['messages']},
            {row.db_id: [{'offset': 0, 'text': message_body(row.message)}] for row in all_rows})
        # Search eligibility does not change append-only tool history or DB reconstruction.
        live = await self.runtime._get_live_state(self.session)
        hot = copy.deepcopy(live.raw_messages)
        self.runtime.invalidate_session(self.session)
        restored = await self.runtime._get_live_state(self.session)
        self.assertEqual(restored.raw_messages, hot)
        self.assertEqual([row.db_id for row in restored.raw_messages], [row.db_id for row in all_rows])
        self.assertEqual([row.message.metadata['tool_payload'] for row in restored.raw_messages[-6:]],
            [row.message.metadata['tool_payload'] for row in derived])

    async def test_tool_name_collision_cannot_hide_human_or_assistant_evidence(self):
        rows = []
        for role in (MessageRole.USER, MessageRole.ASSISTANT):
            for name in ('memory_search', 'memory_read', 'user_profile_fetch'):
                rows.append(await self.store.append_message(self.session, ConversationMessage.text(
                    role, 'Saffron is relevant to this conversation.', name=name,
                    metadata={'tool_phase': 'result'})))
        self.assertCountEqual(await self.search_ids(), [row.db_id for row in rows])

    async def test_external_tools_and_unclassified_phases_keep_their_evidence(self):
        rows = []
        for name, phase in (
            ('shell_exec', 'call'), ('shell_exec', 'result'), ('memory_search_external', 'result'),
            ('memory_search', 'delivery'), ('memory_read', None), ('user_profile_fetch', 'event'),
            ('memory_read', {'unexpected': 'phase'}),
        ):
            rows.append(await self.store.append_message(self.session, ConversationMessage.text(
                MessageRole.TOOL, 'Saffron status from an independent observation.', name=name,
                metadata={'tool_phase': phase})))
        self.assertCountEqual(await self.search_ids(), [row.db_id for row in rows])

    async def test_edited_observation_updates_search_without_losing_revision_history(self):
        metadata = {'source': 'fixture', 'source_chat_id': '100', 'source_message_id': '1'}

        async def revise(phase):
            return await self.store.append_message(self.session, ConversationMessage.text(
                MessageRole.TOOL, 'Saffron status is pending.', name='memory_read',
                metadata={**metadata, 'tool_phase': phase}))

        initial = await revise('event')
        self.assertEqual(await self.search_ids(), [initial.db_id])
        derived = await revise('result')
        self.assertEqual(derived.db_id, initial.db_id)
        self.assertEqual(await self.search_ids(), [], 'The previous lexical projection must be replaced')
        read = await self.tool('memory_read', {'message_ids': [initial.db_id]})
        self.assertEqual(read['messages'][0]['fragments'], [{'offset': 0, 'text': 'Saffron status is pending.'}])
        restored = await revise('event')
        self.assertEqual(restored.db_id, initial.db_id)
        self.assertEqual(await self.search_ids(), [initial.db_id])
        async with self.store.pool.connection() as conn:
            revisions = await (await conn.execute('''SELECT revision,body,metadata->>'tool_phase' AS phase
                FROM message_revisions WHERE message_id=%s ORDER BY revision''', (initial.db_id,))).fetchall()
        self.assertEqual(revisions, [{'revision': revision, 'body': 'Saffron status is pending.', 'phase': phase}
            for revision, phase in ((1, 'event'), (2, 'result'), (3, 'event'))])
