"""Conversation workflows against a real, isolated PostgreSQL schema.

Set TEST_DATABASE_URL to the disposable pgvector database used by CI/development.
No private export, deployment environment, or paid model is used here.
"""
from __future__ import annotations

import asyncio
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

from psycopg import sql
from psycopg.conninfo import make_conninfo
from psycopg.errors import QueryCanceled

from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind, SessionSettings
from tgchatbot.storage.postgres_store import DatabaseConfig, PostgresStore, StaleScopeError, message_body


def original(number: int, text: str, *, actor: str | None = 'telegram:user:1',
             name: str = 'Alex', at: str = '2025-01-01T12:00:00Z', **metadata) -> ConversationMessage:
    return ConversationMessage.text(MessageRole.USER, text, name=name, metadata={
        'source': 'telegram', 'source_chat_id': '-100', 'source_message_id': str(number),
        'actor_id': actor, 'actor_kind': 'user' if actor else 'unknown',
        'actor_name': name, 'sent_at': at, **metadata,
    })


def vector(axis: int = 0) -> list[float]:
    result = [0.0] * 1536
    result[axis] = 1.0
    return result


@unittest.skipUnless(os.environ.get('TEST_DATABASE_URL'), 'requires disposable PostgreSQL/pgvector')
class PostgresConversationWorkflows(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.schema = 'test_conversation_' + uuid.uuid4().hex
        self.temp = tempfile.TemporaryDirectory(prefix='pg-fixture-', dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.store = PostgresStore(os.environ['TEST_DATABASE_URL'], schema=self.schema)
        await self.store.initialize()
        self.session = 'telegram:-100'
        self.defaults = SessionSettings(provider='deepseek', model='chat')
        await self.store.get_or_create_session(self.session, self.defaults)

    async def asyncTearDown(self):
        async with self.store.pool.connection() as conn:
            await conn.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(self.schema)))
        await self.store.close()

    async def append(self, number: int, text: str, **metadata):
        return await self.store.append_message(self.session, original(number, text, **metadata))

    async def fact(self, source, claim='likes tea', **kwargs):
        return await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:1',
            asserted_by='telegram:user:1', claim=claim, source_ids=[source.db_id], **kwargs)

    async def test_same_names_stay_separate_and_reply_forward_provenance_survives_restart(self):
        first = await self.append(1, 'I prefer green tea. 我喜欢杭州龙井。',
            reply_to_source_id='91', topic_id='7', forward_origin={'type': 'hidden_user', 'sender_user_name': 'Quoted author'})
        second = await self.append(2, 'I prefer coffee. 我不喝龙井。', actor='telegram:user:2', at='2025-02-01T12:00:00Z')
        unknown = await self.append(3, '龙井很好', actor=None)
        await self.store.close()
        self.store = PostgresStore(os.environ['TEST_DATABASE_URL'], schema=self.schema)
        await self.store.initialize()
        rows = await self.store.search_messages(self.session, '龙井', actor_id='telegram:user:1')
        self.assertEqual([row['id'] for row in rows], [first.db_id])
        rows = await self.store.search_messages(self.session, '龙井', after='2025-02-01T00:00:00Z')
        self.assertEqual([row['id'] for row in rows], [second.db_id])
        saved = (await self.store.read_messages(self.session, [first.db_id]))[0].message
        self.assertEqual(saved.metadata['reply_to_source_id'], '91')
        self.assertEqual(saved.metadata['forward_origin']['type'], 'hidden_user')
        self.assertIsNone((await self.store.search_messages(self.session, '很好'))[0]['actor_id'])
        self.assertNotEqual(unknown.db_id, first.db_id)

    async def test_soft_reset_preserves_recall_but_full_reset_restores_defaults_and_hides_audit(self):
        source = await self.append(1, 'I like tea')
        await self.fact(source)
        await self.store.save_sticker_persona(self.session, {'style': 'quiet'})
        await self.store.save_session(self.session, replace(self.defaults, model='temporary'))
        scope = await self.store.get_scope(self.session)
        memory_job = await self.store.enqueue_job(self.session, 'recall-fixture', source_ids=[source.db_id])
        context_job = await self.store.enqueue_job(self.session, 'context-fixture', source_ids=[source.db_id], policy='context')
        memory_job = (await self.store.claim_jobs(kind=memory_job['kind']))[0]
        context_job = (await self.store.claim_jobs(kind=context_job['kind']))[0]
        await self.store.reset_context(self.session)
        self.assertEqual(await self.store.list_messages(self.session), [])
        self.assertTrue(await self.store.search_messages(self.session, 'tea'))
        self.assertTrue(await self.store.get_profile(self.session, 'telegram:user:1'))
        self.assertTrue(await self.store.complete_job(memory_job))
        self.assertFalse(await self.store.complete_job(context_job))
        await self.store.append_message(self.session, original(2, 'new context'), expected_scope=scope, generation_only=True)
        with self.assertRaises(StaleScopeError):
            await self.store.assert_scope(self.session, scope)
        await self.store.reset_full(self.session, self.defaults)
        self.assertEqual(await self.store.search_messages(self.session, 'tea'), [])
        self.assertEqual(await self.store.read_messages(self.session, [source.db_id]), [])
        self.assertEqual(await self.store.get_profile(self.session, 'telegram:user:1'), [])
        self.assertIsNone(await self.store.get_sticker_persona(self.session))
        self.assertEqual((await self.store.get_or_create_session(self.session, self.defaults)).model, 'chat')
        self.assertEqual(len(await self.store.list_message_revisions(self.session, source.db_id)), 1)
        with self.assertRaises(StaleScopeError):
            await self.store.append_message(self.session, original(3, 'stale import'), expected_scope=scope, generation_only=True)
        with self.assertRaises(StaleScopeError):
            await self.store.save_sticker_persona(self.session, {'style': 'old turn'}, expected_scope=scope)
        self.assertIsNone(await self.store.get_sticker_persona(self.session))

    async def test_edit_keeps_original_and_invalidates_every_source_backed_derivative(self):
        source = await self.append(1, 'I live in London')
        duplicate = await self.append(1, 'I live in London')
        self.assertEqual(duplicate.db_id, source.db_id)
        scope = await self.store.get_scope(self.session)
        excerpt = await self.store.create_excerpt(self.session, [source.db_id], embedding=vector(), model='fixture:v1')
        await self.fact(source, 'lives in London')
        await self.store.create_memory_block(self.session, summary_text='Alex lives in London', estimated_tokens=8, source_message_ids=[source.db_id])
        jobs = await self.store.claim_jobs(kind='memory_ingest')
        self.assertEqual(len(jobs), 1)
        edited = await self.append(1, 'Correction: I live in Taipei', edited_at='2025-03-01T00:00:00Z')
        self.assertEqual(edited.db_id, source.db_id)
        self.assertFalse(await self.store.complete_job(jobs[0]))
        self.assertIsNone(await self.store.get_excerpt(self.session, excerpt['id']))
        self.assertEqual(await self.store.get_profile(self.session, 'telegram:user:1'), [])
        self.assertEqual(await self.store.list_memory_blocks(self.session), [])
        self.assertEqual(await self.store.search_messages(self.session, 'London'), [])
        self.assertEqual((await self.store.search_messages(self.session, 'Taipei'))[0]['source_revision'], 2)
        revisions = await self.store.list_message_revisions(self.session, source.db_id)
        self.assertEqual([row['body'] for row in revisions], ['Correction: I live in Taipei', 'I live in London'])
        with self.assertRaises(StaleScopeError):
            await self.store.create_excerpt(self.session, [source.db_id], expected_scope=scope,
                expected_source_revisions={str(source.db_id): 1})
        stale_export = await self.append(1, 'I live in London')
        self.assertIn('Taipei', message_body(stale_export.message))

    async def test_rollback_invalidates_transitive_summaries_and_restores_unaffected_originals(self):
        first = await self.append(1, 'Can we meet Tuesday?')
        second = await self.append(2, 'Actually Wednesday works')
        one = await self.store.create_memory_block(self.session, summary_text='Tuesday proposal', estimated_tokens=2, source_message_ids=[first.db_id])
        two = await self.store.create_memory_block(self.session, summary_text='Wednesday correction', estimated_tokens=2, source_message_ids=[second.db_id])
        await self.store.replace_memory_blocks(self.session, block_ids=[one.block_id, two.block_id],
            summary_text='Meeting Wednesday', estimated_tokens=2, source_message_count=2,
            start_message_id=first.db_id, end_message_id=second.db_id)
        self.assertEqual(await self.store.list_uncompacted_messages(self.session), [])
        await self.store.hide_messages_since(self.session, second.db_id)
        self.assertEqual(await self.store.list_memory_blocks(self.session), [])
        visible = await self.store.list_uncompacted_messages(self.session)
        self.assertEqual([row.db_id for row in visible], [first.db_id])
        self.assertEqual(await self.store.search_messages(self.session, 'Wednesday'), [])
        await self.store.reset_context(self.session)
        self.assertEqual(await self.store.hide_message_ids(self.session, [first.db_id]), 0)
        self.assertTrue(await self.store.search_messages(self.session, 'Tuesday'))

    async def test_profile_evidence_corrections_are_idempotent_and_temporal(self):
        source = await self.append(1, 'I drink coffee')
        first = await self.fact(source, 'drinks coffee', claim_key='drink', valid_from='2025-01-01T00:00:00Z')
        self.assertEqual(first['id'], (await self.fact(source, 'drinks coffee', claim_key='drink', valid_from='2025-01-01T00:00:00Z'))['id'])
        correction = await self.append(2, 'I stopped coffee. Tea only now.', at='2025-03-01T00:00:00Z')
        second = await self.fact(correction, 'drinks tea', claim_key='drink', supersedes=first['id'])
        self.assertEqual(second['id'], (await self.fact(correction, 'drinks tea', claim_key='drink', supersedes=first['id']))['id'])
        self.assertEqual([row['claim'] for row in await self.store.get_profile(self.session, 'telegram:user:1')], ['drinks tea'])
        self.assertEqual([row['claim'] for row in await self.store.get_profile(self.session, 'telegram:user:1', at='2025-02-01T00:00:00Z')], ['drinks coffee'])
        await self.store.hide_message_ids(self.session, [correction.db_id])
        self.assertEqual(await self.store.get_profile(self.session, 'telegram:user:1'), [],
            'Surviving historical evidence is queued for bounded reconciliation, not inserted automatically')
        answer = await self.store.append_message(self.session, ConversationMessage.assistant_text('Alex is a doctor'))
        with self.assertRaises(ValueError):
            await self.fact(answer, 'is a doctor')
        with self.assertRaises(ValueError):
            await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:2', asserted_by='telegram:user:2', claim='likes tea', source_ids=[source.db_id])

    async def test_split_excerpt_retrieval_retains_both_spans_and_respects_model_and_actor(self):
        source = await self.append(1, 'alpha beta gamma delta')
        one = await self.store.create_excerpt(self.session, [source.db_id], spans=[{'message_id': source.db_id, 'start': 0, 'end': 10}], embedding=vector(), model='fixture:v1')
        two = await self.store.create_excerpt(self.session, [source.db_id], spans=[{'message_id': source.db_id, 'start': 11, 'end': 22}], embedding=vector(1), model='fixture:v1')
        self.assertTrue(one['has_embedding'])
        self.assertEqual((await self.store.get_excerpt(self.session, one['id']))['text'], one['text'])
        self.assertNotIn('gamma', one['text'])
        self.assertIn('gamma', two['text'])
        results = await self.store.search_excerpts(self.session, embedding=vector(), model='fixture:v1', actor_id='telegram:user:1')
        self.assertEqual([row['id'] for row in results], [one['id'], two['id']])
        self.assertEqual(await self.store.search_excerpts(self.session, embedding=vector(), model='other-model'), [])
        self.assertEqual(await self.store.search_excerpts(self.session, embedding=vector(), model='fixture:v1', actor_id='telegram:user:2'), [])
        await self.store.reset_full(self.session, self.defaults)
        self.assertEqual(await self.store.search_excerpts(self.session, 'alpha', embedding=vector(), model='fixture:v1'), [])

    async def test_job_lease_payload_restart_and_cleanup_do_not_remove_originals(self):
        source = await self.append(1, 'Remember the hotel name')
        for index in range(4):
            await self.store.enqueue_job(self.session, 'batch-fixture', source_ids=[source.db_id], dedupe_key=str(index))
        left, right = await asyncio.gather(self.store.claim_jobs(2, kind='batch-fixture'), self.store.claim_jobs(2, kind='batch-fixture'))
        self.assertEqual(len({job['id'] for job in left + right}), 4)
        job = left[0]
        self.assertFalse(await self.store.update_job_payload({**job, 'lease_token': 'wrong'}, {'state': 'submitting'}))
        self.assertTrue(await self.store.update_job_payload(job, {'state': 'submitting', 'name': 'idempotent-remote-operation'}))
        self.assertTrue(await self.store.renew_job(job))
        self.assertTrue(await self.store.defer_job(job, {'state': 'polling', 'operation': 'remote-42'}, 0))
        self.assertFalse(await self.store.complete_job(job))
        reclaimed = (await self.store.claim_jobs(1, kind='batch-fixture'))[0]
        self.assertEqual(reclaimed['payload']['operation'], 'remote-42')
        self.assertNotEqual(reclaimed['lease_token'], job['lease_token'])
        self.assertTrue(await self.store.complete_job(reclaimed))
        self.assertEqual(await self.store.cleanup_jobs(older_than_seconds=0), 0)
        self.assertEqual(len(await self.store.read_messages(self.session, [source.db_id])), 1)
        stale = right[0]
        await self.store.reset_full(self.session, self.defaults)
        self.assertFalse(await self.store.update_job_payload(stale, {'state': 'done'}))

    async def test_working_preview_and_native_replay_do_not_mutate_portable_original(self):
        message = original(1, 'Here is the picture')
        message.parts.append(MessagePart(kind=PartKind.IMAGE, data_b64='YWJj', detail='a red cup'))
        stored = await self.store.append_message(self.session, message)
        await self.store.retire_context_images(self.session, target_images=0)
        self.assertIn('[Image compacted]', message_body((await self.store.list_recent_visible_messages(self.session))[0].message))
        original_saved = (await self.store.read_messages(self.session, [stored.db_id]))[0]
        self.assertEqual(original_saved.message.parts[0].text, 'Here is the picture')
        self.assertIn('a red cup', message_body(original_saved.message))
        self.assertIsNone(original_saved.message.parts[1].data_b64)
        self.assertIsNotNone(original_saved.message.parts[1].preview_ref)
        reply = await self.store.append_message(self.session, ConversationMessage.assistant_text('A red cup', metadata={'provider_native': {'provider': 'fixture', 'items': [{'text': 'A red cup'}]}}))
        recent = (await self.store.list_recent_visible_messages(self.session))[0]
        self.assertIn('provider_native', recent.message.metadata)
        portable = (await self.store.read_messages(self.session, [reply.db_id]))[0].message
        self.assertNotIn('provider_native', portable.metadata)
        self.assertEqual(message_body(portable), 'A red cup')
        audit = await self.store.list_message_revisions(self.session, reply.db_id)
        self.assertNotIn('provider_native', audit[0]['metadata'])

    async def test_tail_and_canonical_pagination_follow_source_and_reset_boundaries(self):
        first = await self.append(1, 'first')
        second = await self.append(2, 'second')
        scope = await self.store.get_scope(self.session)
        tail = await self.store.save_excerpt_tail(self.session, '7', [first.db_id],
            spans=[{'message_id': first.db_id, 'start': 0, 'end': 5}], expected_scope=scope)
        self.assertEqual(tail['source_revisions'], {str(first.db_id): 1})
        page = await self.store.list_canonical_messages(self.session, limit=1, expected_scope=scope)
        self.assertEqual(page[0].db_id, first.db_id)
        await self.store.reset_context(self.session)
        page = await self.store.list_canonical_messages(self.session, after_message_id=first.db_id, expected_scope=scope)
        self.assertEqual([row.db_id for row in page], [second.db_id])
        self.assertIsNotNone(await self.store.get_excerpt_tail(self.session, '7'))
        await self.append(1, 'edited', edited_at='2025-03-01T00:00:00Z')
        self.assertIsNone(await self.store.get_excerpt_tail(self.session, '7'))
        await self.store.reset_full(self.session, self.defaults)
        with self.assertRaises(StaleScopeError):
            await self.store.list_canonical_messages(self.session, expected_scope=scope)

    async def test_editing_one_person_requeues_other_people_from_invalidated_excerpt_and_tail(self):
        first = await self.append(1, 'I like tea')
        second = await self.append(2, 'I cycle to work', actor='telegram:user:2')
        for job in await self.store.claim_jobs(16, kind='memory_ingest'):
            self.assertTrue(await self.store.complete_job(job))
        await self.store.create_excerpt(self.session, [first.db_id, second.db_id], embedding=vector(), model='fixture:v1')
        await self.store.save_excerpt_tail(self.session, None, [first.db_id, second.db_id], spans=[
            {'message_id': first.db_id, 'start': 0, 'end': 10},
            {'message_id': second.db_id, 'start': 0, 'end': 15}])
        await self.append(1, 'I stopped tea', edited_at='2025-03-01T00:00:00Z')
        jobs = await self.store.claim_jobs(16, kind='memory_ingest')
        self.assertEqual({job['payload']['message_id'] for job in jobs}, {first.db_id, second.db_id})
        self.assertIsNone(await self.store.get_excerpt_tail(self.session))
        self.assertEqual((await self.store.search_messages(self.session, 'cycle', actor_id='telegram:user:2'))[0]['id'], second.db_id)

    async def test_rollback_of_correction_preserves_explicit_original_expiry(self):
        first = await self.append(1, 'I will work in London until July')
        fact = await self.fact(first, 'works in London', valid_from='2025-01-01T00:00:00Z', valid_to='2025-07-01T00:00:00Z')
        correction = await self.append(2, 'I left London in March', at='2025-03-01T00:00:00Z')
        await self.fact(correction, 'left London', supersedes=fact['id'])
        await self.store.hide_message_ids(self.session, [correction.db_id])
        before = await self.store.get_profile(self.session, 'telegram:user:1', at='2025-05-01T00:00:00Z')
        self.assertEqual([row['claim'] for row in before], ['works in London'])
        self.assertEqual(await self.store.get_profile(self.session, 'telegram:user:1', at='2025-08-01T00:00:00Z'), [])

    async def test_import_queue_coalesces_without_losing_running_or_original_messages(self):
        rows = [await self.append(index, f'Original imported message {index}') for index in range(1, 8)]
        running = (await self.store.claim_jobs(1, kind='memory_ingest'))[0]
        scope = await self.store.get_scope(self.session)
        await self.store.reset_context(self.session)
        batch = await self.store.coalesce_memory_jobs(self.session, source_ids=[row.db_id for row in rows], expected_scope=scope)
        self.assertEqual(len(batch['source_ids']), 6)
        self.assertNotIn(running['source_ids'][0], batch['source_ids'])
        self.assertIsNone(await self.store.coalesce_memory_jobs(self.session, source_ids=[row.db_id for row in rows], expected_scope=scope))
        claimed = await self.store.claim_jobs(16, kind='memory_ingest')
        self.assertEqual(len(claimed), 1)
        self.assertEqual(claimed[0]['source_ids'], batch['source_ids'])
        self.assertTrue(await self.store.complete_job(running))
        self.assertTrue(await self.store.complete_job(claimed[0]))
        self.assertEqual(await self.store.job_status(self.session), [])
        self.assertEqual(len(await self.store.read_messages(self.session, [row.db_id for row in rows])), 7)

    async def test_attachment_only_message_is_searchable_without_becoming_a_human_claim(self):
        message = original(1, 'My boarding pass')
        message.parts = [MessagePart(kind=PartKind.FILE, filename='holiday-tickets.pdf',
            mime_type='application/pdf', size_bytes=1234, artifact_path='remote:uploads/holiday-tickets.pdf')]
        saved = await self.store.append_message(self.session, message)
        row = (await self.store.read_messages(self.session, [saved.db_id]))[0]
        self.assertEqual(row.message.parts[0].kind, PartKind.FILE)
        self.assertIn('Attachment reference:', row.message.parts[0].text)
        self.assertIn('holiday-tickets.pdf', row.message.parts[0].text)
        self.assertEqual((await self.store.search_messages(self.session, 'holiday-tickets'))[0]['id'], saved.db_id)
        with self.assertRaises(ValueError):
            await self.fact(saved, 'likes holiday travel')

    async def test_decorative_notes_are_not_lexical_evidence_but_tool_results_are(self):
        message = original(1, 'Plain greeting')
        message.parts.append(MessagePart(kind=PartKind.TEXT, text='ornamentalmarker', origin='auto_note'))
        await self.store.append_message(self.session, message)
        tool = await self.store.append_message(self.session, ConversationMessage.text(MessageRole.TOOL, 'The train booking reference is ZXQ739'))
        self.assertEqual(await self.store.search_messages(self.session, 'ornamentalmarker'), [])
        result = (await self.store.search_messages(self.session, 'ZXQ739'))[0]
        self.assertEqual(result['id'], tool.db_id)
        self.assertEqual(result['role'], 'tool')

    async def test_concurrent_intake_survives_unrelated_edit_but_not_a_context_reset(self):
        await self.append(1, 'Earlier original')
        scope = await self.store.get_scope(self.session)
        await self.append(1, 'Corrected original', edited_at='2025-03-01T00:00:00Z')
        fresh = await self.store.append_message(self.session, original(2, 'Second person finished uploading'), expected_scope=scope, intake=True)
        self.assertEqual(message_body(fresh.message), 'Second person finished uploading')
        with self.assertRaises(StaleScopeError):
            await self.store.append_message(self.session, ConversationMessage.assistant_text('Stale response'), expected_scope=scope)
        await self.store.reset_context(self.session)
        with self.assertRaises(StaleScopeError):
            await self.store.append_message(self.session, original(3, 'Upload from old context'), expected_scope=scope, intake=True)

    async def test_failed_paid_batch_name_survives_cleanup_and_full_reset_for_reconciliation(self):
        queued = await self.store.enqueue_job(self.session, 'embedding_batch', payload={'phase': 'polling', 'name': 'batches/synthetic-paid-operation'})
        job = (await self.store.claim_jobs(kind='embedding_batch'))[0]
        self.assertFalse(await self.store.complete_job(job, error='temporary remote outage'))
        await self.store.enqueue_job(self.session, 'embedding_batch', payload={'phase': 'prepared'})
        unpaid = (await self.store.claim_jobs(kind='embedding_batch'))[0]
        self.assertFalse(await self.store.complete_job(unpaid, error='invalid input'))
        self.assertEqual(await self.store.cleanup_jobs(older_than_seconds=0), 1)
        await self.store.reset_full(self.session, self.defaults)
        await self.store.cleanup_jobs(older_than_seconds=0)
        async with self.store.pool.connection() as conn:
            row = await (await conn.execute('SELECT payload FROM jobs WHERE id=%s', (queued['id'],))).fetchone()
        self.assertEqual(row['payload']['name'], 'batches/synthetic-paid-operation')

    async def test_lexical_recall_keeps_best_full_matches_ahead_of_newer_partial_matches(self):
        complete = [await self.append(i, 'green tea and garden') for i in range(1, 5)]
        partial = await self.append(5, 'green coffee')
        top = await self.store.search_messages(self.session, 'green tea', limit=2)
        self.assertEqual([row['id'] for row in top], [complete[-1].db_id, complete[-2].db_id])
        with_unknown_word = await self.store.search_messages(self.session, 'green tea nonexistentword', limit=2)
        self.assertEqual([row['id'] for row in with_unknown_word], [row['id'] for row in top])
        all_results = await self.store.search_messages(self.session, 'green tea', limit=10)
        self.assertEqual(all_results[-1]['id'], partial.db_id)

    async def test_long_unbroken_words_are_preserved_and_exactly_searchable(self):
        word = 'x' * 13000
        saved = await self.append(1, 'saffron ' + word)
        near = await self.append(2, 'saffron ' + word[:-1] + 'y')
        self.assertEqual(message_body((await self.store.read_messages(self.session, [saved.db_id]))[0].message),
            'saffron ' + word)
        self.assertEqual([row['id'] for row in await self.store.search_messages(self.session, word)], [saved.db_id])
        self.assertEqual([row['id'] for row in await self.store.search_messages(self.session, word[:-1] + 'y')], [near.db_id])
        multibyte_word = 'é' * 1500
        multibyte = await self.append(3, multibyte_word)
        self.assertEqual([row['id'] for row in await self.store.search_messages(self.session, multibyte_word)], [multibyte.db_id])
        # Query processing must not silently ignore evidence after its64th term.
        query = ' '.join(f'absent{i}' for i in range(70)) + ' saffron'
        self.assertEqual(len(await self.store.search_messages(self.session, query)), 2)

    async def test_database_environment_overrides_and_unset_timeouts_respect_the_server(self):
        inherited_dsn = make_conninfo(os.environ['TEST_DATABASE_URL'],
            options='-c statement_timeout=1234ms -c lock_timeout=2345ms')
        inherited = PostgresStore(inherited_dsn, schema=self.schema, config=DatabaseConfig())
        await inherited.pool.open(wait=True)
        try:
            async with inherited.pool.connection() as conn:
                row = await (await conn.execute("SELECT current_setting('statement_timeout') AS statement, current_setting('lock_timeout') AS lock")).fetchone()
                self.assertEqual(row, {'statement': '1234ms', 'lock': '2345ms'})
        finally:
            await inherited.close()
        with patch.dict(os.environ, {'MEMORY_DB_POOL_MIN_SIZE': '1', 'MEMORY_DB_POOL_MAX_SIZE': '2',
                'MEMORY_DB_POOL_TIMEOUT_S': '1.5', 'MEMORY_DB_STATEMENT_TIMEOUT_S': '0.025',
                'MEMORY_DB_LOCK_TIMEOUT_S': '0', 'MEMORY_DB_INITIALIZATION_TIMEOUT_S': '10'}):
            configured = PostgresStore(inherited_dsn, schema=self.schema)
        await configured.initialize()
        try:
            async with configured.pool.connection() as conn:
                row = await (await conn.execute("SELECT current_setting('statement_timeout') AS statement, current_setting('lock_timeout') AS lock")).fetchone()
                self.assertEqual(row, {'statement': '25ms', 'lock': '0'})
            with self.assertRaises(QueryCanceled):
                async with configured.pool.connection() as conn:
                    await conn.execute('SELECT pg_sleep(0.1)')
            self.assertEqual(configured.pool.get_stats()['pool_max'], 2)
        finally:
            await configured.close()

    async def test_large_import_pages_and_derivatives_never_drop_requested_sources(self):
        first = await self.append(1, 'saffron imported original')
        count = 10005
        # Bulk fixture setup models an already persisted import, with canonical
        # originals and singleton ingress jobs. Exercise only public workflow
        # operations below; no artificial model calls or private export is used.
        async with self.store.pool.connection() as conn:
            await conn.execute('''INSERT INTO messages(session_id,generation,context_id,source,source_chat_id,
                source_message_id,actor_id,actor_kind,actor_name,role,sent_at)
                SELECT %s,1,1,'telegram','-100',n::text,'telegram:user:1','user','Alex','user',
                '2025-01-01T12:00:00Z'::timestamptz+n*interval '1 second' FROM generate_series(2,%s) n''',
                (self.session, count))
            await conn.execute('''INSERT INTO message_revisions(message_id,revision,body,parts,metadata,estimated_tokens,fingerprint)
                SELECT m.id,1,r.body,r.parts,'{}',r.estimated_tokens,md5(m.id::text)
                FROM messages m CROSS JOIN message_revisions r WHERE r.message_id=%s AND m.id<>%s''',
                (first.db_id, first.db_id))
            await conn.execute('''INSERT INTO message_search(message_id,lexemes)
                SELECT id,array_to_tsvector(ARRAY['saffron','imported','original']) FROM messages WHERE id<>%s''', (first.db_id,))
            await conn.execute('''INSERT INTO jobs(session_id,generation,context_id,scope_revision,kind,policy,source_ids,source_revisions,payload,dedupe_key)
                SELECT session_id,1,1,1,'memory_ingest','memory',ARRAY[id],jsonb_build_object(id::text,1),
                jsonb_build_object('message_id',id),'import-fixture:'||id FROM messages WHERE id<>%s''', (first.db_id,))
        self.store.config = replace(self.store.config, read_page_size=count, search_results=150, profile_results=150)
        rows = await self.store.list_canonical_messages(self.session, limit=count)
        ids = [row.db_id for row in rows]
        self.assertEqual(len(ids), count)
        self.assertEqual(len(await self.store.read_messages(self.session, ids)), count)
        self.assertEqual(len(await self.store.list_uncompacted_messages(self.session, limit=count)), count)
        self.assertEqual(len(await self.store.search_messages(self.session, 'saffron')), 150)
        merged = await self.store.coalesce_memory_jobs(self.session, source_ids=ids)
        self.assertEqual(merged['source_ids'], ids)
        claimed = (await self.store.claim_jobs(kind='memory_ingest'))[0]
        self.assertEqual(claimed['source_ids'], ids)
        self.assertTrue(await self.store.complete_job(claimed))
        block = await self.store.create_memory_block(self.session, source_message_ids=ids,
            summary_text='Imported saffron discussion', estimated_tokens=5)
        self.assertEqual(block.source_message_count, count)
        excerpt = await self.store.create_excerpt(self.session, ids)
        self.assertEqual(excerpt['source_ids'], ids)
        with patch.dict(os.environ, {'MEMORY_PROFILE_BYTES': '50000'}):
            for index in range(120):
                await self.fact(first, f'Distinct durable preference {index}')
        self.assertEqual(len(await self.store.get_profile(self.session, 'telegram:user:1')), 120)

    async def test_configured_job_retries_claim_pages_and_leases_retain_work(self):
        self.store.config = replace(self.store.config, job_claim_size=20, job_max_attempts=5,
            job_retry_delay_seconds=0, job_lease_seconds=7200, job_retention_seconds=0)
        for index in range(20):
            await self.store.enqueue_job(self.session, 'retry-fixture', payload={'original_request': index})
        jobs = await self.store.claim_jobs(kind='retry-fixture')
        self.assertEqual(len(jobs), 20)
        job = jobs[0]
        self.assertGreater((job['lease_until'] - job['created_at']).total_seconds(), 7190)
        self.assertTrue(await self.store.renew_job(job, lease_seconds=10800))
        for attempt in range(1, 5):
            self.assertEqual(job['attempts'], attempt)
            await self.store.complete_job(job, error='temporary service outage', retry=True)
            job = (await self.store.claim_jobs(kind='retry-fixture', limit=1))[0]
        self.assertEqual(job['attempts'], 5)
        self.assertTrue(await self.store.complete_job(job))
        await self.store.defer_job(jobs[1], jobs[1]['payload'], 8 * 86400)
        self.assertFalse(await self.store.complete_job(jobs[1]))
        self.assertEqual(await self.store.claim_jobs(kind='retry-fixture'), [])

    async def test_configured_expansion_and_large_delivered_answer_keep_all_identities(self):
        from tgchatbot.storage.relationships import expand_message_ids
        self.store.config = replace(self.store.config, relationship_neighbors=3, relationship_read_limit=30)
        rows = [await self.append(index, f'Conversation event {index}',
            at=f'2025-01-01T12:00:{index:02d}Z') for index in range(1, 40)]
        selected = [row.db_id for row in rows[9:34]]
        expanded = await expand_message_ids(self.store, self.session, selected, limit=31)
        self.assertEqual(len(expanded), 31)
        self.assertEqual(expanded[:25], selected)
        default = await expand_message_ids(self.store, self.session, [rows[19].db_id])
        self.assertEqual(len(default), 7)
        answer = await self.store.append_message(self.session, ConversationMessage.assistant_text('A long delivered answer'))
        await self.store.bind_message_source(self.session, answer.db_id, source='telegram', source_chat_id='-100',
            source_message_ids=[str(number) for number in range(10000, 11002)], actor_id='telegram:user:999',
            actor_kind='bot', actor_name='Assistant')
        reply = await self.append(40, 'Regarding the final answer chunk', reply_to_source_id='11001')
        linked = await expand_message_ids(self.store, self.session, [reply.db_id], neighbors=0, limit=2)
        self.assertEqual(linked, [reply.db_id, answer.db_id])

    async def test_configured_retirement_and_paid_reconciliation_preserve_originals(self):
        from tgchatbot.storage.retirement import retire_excerpt_chunk
        from tgchatbot.storage.retired_batches import claim_retired_batch, save_retired_batch
        self.store.config = replace(self.store.config, retirement_page_size=1200,
            retired_batch_lease_seconds=7200, retired_batch_poll_seconds=180)
        original = await self.append(1, 'An original that remains auditable after reset')
        excerpt = await self.store.create_excerpt(self.session, [original.db_id], embedding=vector(), model='fixture')
        async with self.store.pool.connection() as conn:
            await conn.execute('''INSERT INTO excerpts(session_id,generation,source_ids,source_revisions,spans,fingerprint,model,embedding)
                SELECT session_id,generation,source_ids,source_revisions,spans,'retirement:'||n,model,embedding
                FROM excerpts CROSS JOIN generate_series(1,1500) n WHERE id=%s''', (excerpt['id'],))
        await self.store.enqueue_job(self.session, 'embedding_batch', source_ids=[original.db_id],
            payload={'phase': 'polling', 'name': 'batches/already-paid'})
        await self.store.reset_full(self.session, self.defaults)
        job = (await self.store.claim_jobs(kind='memory_retire'))[0]
        self.assertEqual(await retire_excerpt_chunk(self.store, job), 1200)
        self.assertTrue(await self.store.defer_job(job, job['payload'], 0))
        job = (await self.store.claim_jobs(kind='memory_retire'))[0]
        self.assertEqual(await retire_excerpt_chunk(self.store, job, limit=2000), 301)
        self.assertTrue(await self.store.complete_job(job))
        paid = await claim_retired_batch(self.store)
        self.assertEqual(paid['payload']['name'], 'batches/already-paid')
        self.assertGreater((paid['lease_until'] - paid['created_at']).total_seconds(), 7190)
        self.assertTrue(await save_retired_batch(self.store, paid, paid['payload']))
        async with self.store.pool.connection() as conn:
            pending = await (await conn.execute('SELECT extract(epoch FROM available_at-now()) AS delay FROM jobs WHERE id=%s', (paid['id'],))).fetchone()
        self.assertGreater(pending['delay'], 179)
        self.assertEqual((await self.store.list_message_revisions(self.session, original.db_id))[0]['body'],
            'An original that remains auditable after reset')


if __name__ == '__main__':
    unittest.main()
