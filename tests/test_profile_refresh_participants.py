"""Compaction refresh discovers people from scoped originals, not cache survivors."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from psycopg.errors import QueryCanceled

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.memory import MemoryService
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.identities import canonical_actor_id
from tgchatbot.domain.models import ConversationMessage, MessageRole
from tgchatbot.storage.postgres_store import StaleScopeError


class ProfileRefreshParticipantTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        await self.settings()
        self.runtime.memory = MemoryService(self.store, SimpleNamespace(enabled=False))

    async def original(self, number, actor, *, name=None, role=MessageRole.USER, session=None, **metadata):
        return await self.store.append_message(session or self.session, ConversationMessage.text(
            role, f'Original statement {number}.', metadata={
                'source': 'telegram', 'source_chat_id': '100', 'source_message_id': str(number),
                'actor_id': actor, 'actor_kind': 'user', 'actor_name': name or actor,
                'sent_at': '2026-01-02T03:04:05+00:00', **metadata}))

    async def compact(self, sources, store=None):
        await (store or self.store).create_memory_block(self.session,
            summary_text='Earlier original statements remain available.', estimated_tokens=10,
            source_message_ids=[source.db_id for source in sources])

    async def latest_refresh(self, store=None):
        rows = await (store or self.store).list_uncompacted_messages(self.session)
        pair = [row.message for row in rows if row.message.metadata.get('synthetic_role') == 'profile_refresh'][-2:]
        self.assertEqual([message.metadata['tool_phase'] for message in pair], ['call', 'result'])
        self.assertEqual(pair[0].metadata['tool_payload']['call_id'], pair[1].metadata['tool_payload']['call_id'])
        self.assertTrue(all(message.role == MessageRole.TOOL for message in pair))
        return pair[0].metadata['tool_payload']['arguments'], pair[1].metadata['tool_payload']['output']

    async def test_prepare_after_all_humans_compacted_keeps_named_profiles_warm_and_cold(self):
        first = await self.original(1, 'telegram:user:101', name='Person A')
        second = await self.original(2, 'telegram:user:102', name='Person B')
        canonical_before = await self.store.read_messages(self.session, [first.db_id, second.db_id])
        fact_ids = {}
        for source, actor, claim in ((first, 'telegram:user:101', 'Prefers tea'),
                                     (second, 'telegram:user:102', 'Prefers coffee')):
            fact = await self.store.save_profile_fact(self.session, subject_actor_id=actor,
                asserted_by=actor, claim=claim, source_ids=[source.db_id])
            fact_ids[actor] = fact['id']
        await self.compact([first, second])
        tool = await self.runtime.record_tool_observation(session_id=self.session,
            name='shell_exec', phase='result', payload={'output': {'stdout': 'Task completed.'}})
        await self.runtime.record_assistant_text(session_id=self.session, text='Done.')
        await self.runtime.prepare_context(session_id=self.session)
        args, warm = await self.latest_refresh()
        self.assertEqual([canonical_actor_id(actor) for actor in args['actor_ids']], ['telegram:user:102', 'telegram:user:101'])
        self.assertEqual({canonical_actor_id(p['actor_id']): p['identity']['actor_name'] for p in warm['profiles']},
            {'telegram:user:101': 'Person A', 'telegram:user:102': 'Person B', 'agent': None})
        self.assertEqual({canonical_actor_id(p['actor_id']): [fact['fact_id'] for fact in p['facts']] for p in warm['profiles']},
            {'telegram:user:101': [fact_ids['telegram:user:101']],
             'telegram:user:102': [fact_ids['telegram:user:102']], 'agent': []})
        # A later tool-only compaction must discover the same original people
        # after reopening the database, without a surviving runtime roster.
        await self.compact([tool])
        await self.store.close()
        reopened = await self.new_store()
        runtime = AgentRuntime(config=self.config, store=reopened, tool_registry=self.tools,
            providers={'openai': self.provider}, memory=MemoryService(reopened, SimpleNamespace(enabled=False)))
        await runtime.prepare_context(session_id=self.session)
        cold_args, cold = await self.latest_refresh(reopened)
        self.assertEqual(cold_args, args)
        self.assertEqual(cold['profiles'], warm['profiles'])
        self.assertEqual([row.message for row in await reopened.read_messages(self.session,
            [first.db_id, second.db_id])], [row.message for row in canonical_before])
        self.assertEqual(self.provider.requests, [])

    async def test_recent_original_window_ignores_tool_tail_and_keeps_trigger_priority(self):
        self.store.config = replace(self.store.config, recent_page_size=2)
        older = await self.original(1, 'telegram:user:101')
        second = await self.original(2, 'telegram:user:102')
        newest = await self.original(3, 'telegram:user:103')
        await self.compact([older, second, newest])
        for number in range(10):
            await self.runtime.record_tool_observation(session_id=self.session,
                name='shell_exec', phase='result', payload={'output': {'stdout': str(number)}})
            await self.original(10 + number, 'telegram:user:999', synthetic_role='reply_target')
        await self.runtime.prepare_context(session_id=self.session)
        args, _ = await self.latest_refresh()
        self.assertEqual([canonical_actor_id(actor) for actor in args['actor_ids']], ['telegram:user:103', 'telegram:user:102'])
        state = await self.runtime._get_live_state(self.session)
        await self.runtime._refresh_profiles_after_compaction(session_id=self.session, state=state,
            settings=await self.settings(), provider=self.provider, instructions='', tools=[], emit=None, trigger=older)
        args, _ = await self.latest_refresh()
        self.assertEqual([canonical_actor_id(actor) for actor in args['actor_ids']], ['telegram:user:101', 'telegram:user:103', 'telegram:user:102'])

    async def test_reset_and_visibility_boundaries_preserve_explicit_same_chat_reply_only(self):
        await self.original(1, 'telegram:user:201', name='Reply partner')
        await self.original(2, 'telegram:user:202', name='Forwarded person')
        await self.store.reset_context(self.session)
        first = await self.original(3, 'telegram:user:101')
        second = await self.original(4, 'telegram:user:102',
            reply_to_source_id='1', reply_to_source_chat_id='100',
            reply_to_actor={'actor_id': 'telegram:user:201', 'actor_kind': 'user'},
            forward_origin={'type': 'user', 'sender_user': {'id': 202}})
        await self.original(5, 'telegram:user:301', actor_kind='bot')
        await self.original(6, 'unknown', actor_kind='unknown')
        hidden = await self.original(7, 'telegram:user:302')
        deleted = await self.original(8, 'telegram:user:303')
        await self.store.hide_message_ids(self.session, [hidden.db_id])
        await self.store.delete_message_ids(self.session, [deleted.db_id])
        await self.original(9, 'telegram:user:304', role=MessageRole.ASSISTANT)
        await self.original(10, 'telegram:user:305', session='telegram:200')
        await self.original(11, 'telegram:user:306', synthetic_role='reply_target')
        await self.compact([first, second])
        await self.runtime.prepare_context(session_id=self.session)
        args, output = await self.latest_refresh()
        self.assertEqual([canonical_actor_id(actor) for actor in args['actor_ids']], ['telegram:user:102', 'telegram:user:101', 'telegram:user:201'])
        self.assertEqual({canonical_actor_id(p['actor_id']) for p in output['profiles']}, {*(canonical_actor_id(actor) for actor in args['actor_ids']), 'agent'})
        old_scope = await self.store.get_scope(self.session)
        await self.store.reset_context(self.session)
        with self.assertRaises(StaleScopeError):
            await self.store.list_recent_participant_messages(self.session, expected_scope=old_scope)
        self.assertEqual(await self.store.list_recent_participant_messages(self.session), [])
        fresh = await self.original(12, 'telegram:user:401')
        await self.compact([fresh])
        await self.runtime.prepare_context(session_id=self.session)
        args, _ = await self.latest_refresh()
        self.assertEqual([canonical_actor_id(actor) for actor in args['actor_ids']], ['telegram:user:401'])
        old_scope = await self.store.get_scope(self.session)
        await self.store.reset_full(self.session, self.config.default_session_settings())
        with self.assertRaises(StaleScopeError):
            await self.store.list_recent_participant_messages(self.session, expected_scope=old_scope)
        self.assertEqual(await self.store.list_recent_participant_messages(self.session), [])

    async def test_optional_participant_query_timeout_keeps_explicit_unavailable_refresh(self):
        source = await self.original(1, 'telegram:user:101')
        await self.compact([source])
        with patch.object(self.store, 'list_recent_participant_messages',
                          AsyncMock(side_effect=QueryCanceled('synthetic deadline'))):
            await self.runtime.prepare_context(session_id=self.session)
        _, output = await self.latest_refresh()
        self.assertFalse(output['ok'])
        self.assertIn('unavailable', output['coverage'])
        self.assertEqual([row.db_id for row in await self.store.read_messages(self.session, [source.db_id])], [source.db_id])
