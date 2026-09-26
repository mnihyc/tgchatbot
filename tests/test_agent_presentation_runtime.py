"""Agent presentation changes preserve source ownership and historical replay."""
from __future__ import annotations

import json
import asyncio
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx

from tests.business_helpers import BusinessTestCase, ScriptedProvider
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.core.memory import MemoryService
from tgchatbot.domain.attachments import generated_attachment_reference
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind, ProviderResponse
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.storage.postgres_store import message_body


def peer(text, source_id, actor=7):
    return ConversationMessage.user_text(text, metadata={
        'source': 'telegram', 'source_chat_id': '100', 'source_message_id': str(source_id),
        'actor_id': f'telegram:user:{actor}', 'actor_kind': 'user',
        'actor_name': f'Participant {actor}', 'sent_at': '2026-09-15T01:00:00+00:00'})


class AgentPresentationRuntimeTests(BusinessTestCase):
    async def test_delivery_reference_failure_keeps_outcome_and_canonical_receipt(self):
        settings = await self.settings()
        canonical_id = 'sha256:' + 'a' * 64
        catalog = SimpleNamespace(aagent_sticker_id=AsyncMock(return_value='sid:456'))
        self.tools.sticker_catalog = catalog
        saved = []
        expected = []
        for failed_reference in (False, True):
            catalog.aagent_sticker_id.side_effect = RuntimeError('Catalog temporarily unavailable') if failed_reference else None
            receipt = {'sticker_id': canonical_id, 'sticker_label': 'A reassuring hug',
                'delivery_operation_id': 'synthetic-private-operation', 'telegram_message_id': 9001,
                'delivery_timing': 'after_final', 'delivery_state': 'sent', 'sent': True}
            row = await self.runtime.record_tool_observation(session_id=self.session,
                name='sticker_send', phase='delivery', payload=receipt)
            saved.append(row)
            expected.append(receipt)
            text = message_body(row.message)
            self.assertIn('state=sent', text)
            self.assertIn('A reassuring hug', text)
            self.assertIn('delivery_timing=after_text', text)
            self.assertNotIn('after_final', text)
            self.assertEqual('sid:456' in text, not failed_reference)
            self.assertNotIn(canonical_id, text)
            self.assertNotIn('synthetic-private-operation', text)
            self.assertNotIn('telegram_message_id', text)
        state = await self.runtime._get_live_state(self.session)
        before = self.runtime._build_provider_history(state, settings=settings, provider_name='openai')
        reopened = await self.new_store()
        original = await reopened.read_messages(self.session, [row.db_id for row in saved])
        self.assertEqual([row.message.metadata['tool_payload'] for row in original], expected)
        restarted = AgentRuntime(config=self.config, store=reopened, tool_registry=self.tools,
            providers={'openai': self.provider})
        state = await restarted._get_live_state(self.session)
        self.assertEqual(restarted._build_provider_history(state, settings=settings, provider_name='openai'), before)
        summaries = restarted._normalize_compaction_messages([row.message for row in state.raw_messages])
        self.assertEqual([message_body(message) for message in summaries],
            ['Agent-side tool event:\n' + message_body(row.message) for row in saved])

    async def test_sticker_compaction_keeps_selected_reading_after_duplicate_removal(self):
        await self.settings()
        reading = {'meaning': 'Offer comfort', 'context': 'The recipient had a difficult day'}
        output = {'ok': True, 'candidates': [{'sticker_id': 'sid:456', 'caption': '抱抱',
            'action': 'Offering a hug', 'readings': [
                {'meaning': 'Ask for comfort', 'context': 'The sender wants reassurance'},
                {**reading, 'retrieval_match': True}]}]}
        call = await self.runtime.record_tool_observation(session_id=self.session, name='sticker_query',
            phase='call', payload={'call_id': 'selection:1', 'arguments': {'intent_core': 'Comfort the recipient'}})
        result = await self.runtime.record_tool_observation(session_id=self.session, name='sticker_query',
            phase='result', payload={'call_id': 'selection:1', 'output': output})
        reopened = await self.new_store()
        stored = await reopened.list_uncompacted_messages(self.session)
        summary = '\n'.join(message_body(message) for message in self.runtime._normalize_compaction_messages(
            [row.message for row in stored]))
        self.assertIn('sid:456', summary)
        self.assertIn(reading['meaning'], summary)
        self.assertIn(reading['context'], summary)
        self.assertNotIn('retrieval_match', summary)
        canonical = await reopened.read_messages(self.session, [call.db_id, result.db_id])
        self.assertEqual(canonical[1].message.metadata['tool_payload']['output'], output,
            'Both conditional readings remain in the original query result.')

    async def test_framework_call_references_remain_unique_across_writers_restart_and_resets(self):
        await self.settings()
        second = await self.new_store()
        references = await asyncio.gather(*(store.allocate_tool_call_id('profile')
            for store in (self.store, second, self.store, second)))
        await second.close()
        reopened = await self.new_store()
        references.append(await reopened.allocate_tool_call_id('profile'))
        await reopened.reset_context(self.session)
        references.append(await reopened.allocate_tool_call_id('profile'))
        await reopened.reset_full(self.session, self.config.default_session_settings())
        references.append(await reopened.allocate_tool_call_id('profile'))
        self.assertEqual(len(set(references)), len(references))
        for reference in references:
            self.assertRegex(reference, r'^profile:[0-9]+$')
        # Allocation shares PostgreSQL's existing message sequence. Gaps are
        # ordinary identifiers, not missing messages or a continuity counter.
        saved = await reopened.append_message(self.session, peer('A later source is still stored normally.', 70))
        restored = await reopened.read_messages(self.session, [saved.db_id])
        self.assertEqual(message_body(restored[0].message), 'A later source is still stored normally.')

    async def test_automatic_profile_refresh_uses_one_short_persistent_call_reference(self):
        settings = await self.settings()
        source = await self.store.append_message(self.session, peer('I prefer tea.', 71))
        await self.store.save_profile_fact(self.session, subject_actor_id='telegram:user:7',
            asserted_by='telegram:user:7', claim='Prefers tea.', source_ids=[source.db_id])
        await self.store.create_memory_block(self.session, summary_text='A drink preference was shared.',
            estimated_tokens=12, source_message_ids=[source.db_id])
        self.runtime.memory = MemoryService(self.store, SimpleNamespace(enabled=False))
        await self.runtime.prepare_context(session_id=self.session)
        rows = await self.store.list_uncompacted_messages(self.session)
        pair = [row.message for row in rows if row.message.metadata.get('synthetic_role') == 'profile_refresh']
        self.assertEqual([message.metadata['tool_phase'] for message in pair], ['call', 'result'])
        reference = pair[0].metadata['tool_payload']['call_id']
        self.assertRegex(reference, r'^profile:[0-9]+$')
        self.assertEqual(pair[1].metadata['tool_payload']['call_id'], reference)
        self.assertTrue(all(message.metadata['tool_batch_id'] == reference for message in pair))
        self.assertTrue(all(message.role == MessageRole.TOOL for message in pair))
        self.assertEqual(pair[0].metadata['tool_payload']['arguments']['actor_ids'], ['person_id:7'])
        self.assertIn('Prefers tea.', json.dumps(pair[1].metadata['tool_payload']['output']))
        await self.store.close()
        reopened = await self.new_store()
        restarted = AgentRuntime(config=self.config, store=reopened, tool_registry=self.tools,
            providers={'openai': self.provider}, memory=MemoryService(reopened, SimpleNamespace(enabled=False)))
        await restarted.prepare_context(session_id=self.session)
        cold = [row.message for row in await reopened.list_uncompacted_messages(self.session)
            if row.message.metadata.get('synthetic_role') == 'profile_refresh']
        self.assertEqual(cold, pair, 'Restart replays the completed pair instead of regenerating its reference')
        state = await restarted._get_live_state(self.session)
        history = restarted._build_provider_history(state, settings=settings, provider_name='openai')
        replayed = [message for message in history if message.metadata.get('synthetic_role') == 'profile_refresh']
        self.assertEqual([message.metadata['tool_payload']['call_id'] for message in replayed], [reference, reference])
        self.assertEqual(self.provider.requests, [])

    async def test_legacy_labels_and_native_exchange_survive_restart_beside_new_references(self):
        settings = await self.settings(provider='gemini', model='gemini-3.8-flash')
        legacy = await self.store.append_message(self.session, peer('Earlier source remains exact.', 1))
        native = [{'role': 'model', 'parts': [{'text': 'Earlier answer.', 'thoughtSignature': 'synthetic-native-signature'}]}]
        own = await self.runtime.record_assistant_text(session_id=self.session, text='Earlier answer.', metadata={
            'provider_native': {'provider': 'gemini', 'model': settings.model, 'items': native}})
        async with self.store.pool.connection() as conn:
            await conn.execute("UPDATE messages SET presentation=presentation-'presentation_version' WHERE id=ANY(%s)",
                ([legacy.db_id, own.db_id],))
        self.runtime.invalidate_session(self.session)
        latest = await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=peer('New speaker stays distinct.', 2, actor=8))
        wire = []

        def reply(request):
            wire.append(json.loads(request.content))
            return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP',
                'content': {'role': 'model', 'parts': [{'text': 'Understood.'}]}}]})

        provider = GeminiProvider(replace(self.config.gemini, api_key='synthetic-key'))
        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(reply))
        self.addAsyncCleanup(provider.aclose)
        self.runtime.providers['gemini'] = provider
        await self.runtime.run_turn_from_stored(session_id=self.session,
            user_display_name='Participant', trigger_message_id=latest.db_id)
        first = wire[0]['contents']
        headers = {}
        for item in first:
            texts = [part.get('text', '') for part in item.get('parts', [])]
            for text in texts:
                if text.startswith('[Message provenance: '):
                    record = json.loads(text.removeprefix('[Message provenance: ').removesuffix(']'))
                    headers[record['message_id']] = record
        self.assertEqual(headers[legacy.db_id]['speaker']['id'], 'telegram:user:7')
        self.assertEqual(headers[latest.db_id]['speaker']['id'], 'person_id:8')
        self.assertEqual([item for item in first if item['role'] == 'model'], native)
        await self.store.close()
        reader = await self.new_store()
        restarted = AgentRuntime(config=self.config, store=reader, tool_registry=self.tools,
            providers={'gemini': provider})
        await restarted.run_turn_from_stored(session_id=self.session,
            user_display_name='Participant', trigger_message_id=latest.db_id)
        self.assertEqual(wire[1]['contents'], first)
        canonical = await reader.read_messages(self.session, [legacy.db_id, latest.db_id])
        self.assertEqual([message_body(item.message) for item in canonical],
            ['Earlier source remains exact.', 'New speaker stays distinct.'])
        self.assertEqual([item.message.metadata['actor_id'] for item in canonical],
            ['telegram:user:7', 'telegram:user:8'])

    async def test_relative_path_and_presentation_version_do_not_create_source_revisions(self):
        original = peer('Keep my /literal/path and caption exactly.', 20)
        file = MessagePart(PartKind.FILE, filename='报告.pdf', mime_type='application/pdf', size_bytes=1234,
            artifact_path='/srv/session/2026-09-15/report_a1b2.pdf', workspace_path='2026-09-15/report_a1b2.pdf')
        original.parts.append(file)
        saved = await self.store.append_message(self.session, original)
        async with self.store.pool.connection() as conn:
            before = await (await conn.execute('SELECT body,fingerprint FROM message_revisions WHERE message_id=%s',
                (saved.db_id,))).fetchone()
        expected_body = original.parts[0].text + '\n' + generated_attachment_reference(file)
        self.assertEqual(before['body'], expected_body)
        redelivered = replace(original, parts=[original.parts[0], replace(file,
            artifact_path='/another/session/2026-09-15/report_a1b2.pdf')],
            metadata=dict(original.metadata, presentation_version=1))
        again = await self.store.append_message(self.session, redelivered)
        self.assertEqual((again.db_id, again.message.metadata['source_revision']), (saved.db_id, 1))
        async with self.store.pool.connection() as conn:
            rows = await (await conn.execute('SELECT body,fingerprint FROM message_revisions WHERE message_id=%s',
                (saved.db_id,))).fetchall()
        self.assertEqual(rows, [before])
        reader = await self.new_store()
        restored = next(item for item in await reader.list_messages(self.session)
            if item.metadata.get('source_message_id') == '20')
        attachment = next(part for part in restored.parts if part.kind == PartKind.FILE)
        self.assertEqual(attachment.workspace_path, '2026-09-15/report_a1b2.pdf')
        self.assertEqual(restored.metadata['presentation_version'], 2)
        self.assertEqual(message_body(restored), expected_body)

    async def test_compaction_receives_complete_execution_evidence_and_preserves_originals(self):
        settings = await self.settings(min_raw_messages_reserve=1, compact_trigger_tokens=100000,
            compact_target_tokens=50000, compact_batch_tokens=20000,
            compact_tool_min_tokens=1, compact_tool_ratio_threshold=0)
        await self.store.append_message(self.session, peer('Inspect both jobs before deciding whether they succeeded.', 30))
        expected = []
        source_ids = []
        for number, (name, key) in enumerate((('shell_exec', 'command'), ('python_exec', 'code')), 1):
            argument = ('# Preserve inspected command context\n' * 35) + f'final_check_{number}()'
            stdout = ('Progress line\n' * 160) + f'Summary: job {number} failed; do not deploy.'
            output = {'ok': False, 'returncode': 1, 'stdout': stdout,
                'stderr': 'Additional log was omitted.', 'stderr_truncated': True}
            for phase, payload in (('call', {'arguments': {key: argument, 'timeout_s': 45}}),
                                   ('result', {'output': output})):
                stored = await self.store.append_message(self.session, ConversationMessage.text(MessageRole.TOOL,
                    json.dumps(payload, ensure_ascii=False), name=name, metadata={
                        'tool_phase': phase, 'tool_provider': 'openai',
                        'tool_payload': {'call_id': f'call-{number}', **payload}}))
                source_ids.append(stored.db_id)
            expected.append((argument, stdout))
        await self.store.append_message(self.session, peer('Keep this latest question raw.', 31))
        originals = await self.store.read_messages(self.session, source_ids)

        class Summarizer(ScriptedProvider):
            async def generate(inner, **kwargs):
                properties = kwargs['response_schema']['properties']
                candidate = {key: [] for key in properties}
                candidate.update(scope='Inspection of two failed jobs.', interaction_mode='task_execution',
                    decisions=['Do not deploy either failed job.'])
                inner.responses.append(ProviderResponse(final_text=json.dumps(candidate)))
                return await super().generate(**kwargs)

        provider = Summarizer()
        self.runtime.invalidate_session(self.session)
        state = await self.runtime._get_live_state(self.session)
        # Keep the normal recent-history reserve. Each older execution becomes
        # eligible in turn, so exercise both preparation batches.
        for _ in expected:
            self.assertTrue(await self.runtime._compact_old_context(session_id=self.session,
                settings=settings, provider=provider, state=state, pressure=True))
        self.assertEqual(len(provider.requests), 2)
        shown = '\n'.join(part.text or '' for request in provider.requests
            for message in request['messages'] for part in message.parts)
        for argument, stdout in expected:
            self.assertIn(json.dumps(argument, ensure_ascii=False), shown)
            self.assertIn(json.dumps(stdout, ensure_ascii=False), shown)
        self.assertIn('"stderr_truncated": true', shown)
        for request in provider.requests:
            self.assertEqual(request['tools'], [])
            self.assertEqual(request['settings'].native_web_search_mode, 'off')
            estimate = provider.estimate_request_tokens(settings=request['settings'], messages=request['messages'],
                instructions=request['instructions'], tools=[])
            self.assertLess(estimate.total_tokens, settings.compact_trigger_tokens)
        self.assertEqual(await self.store.read_messages(self.session, source_ids), originals)
        self.assertIn('Do not deploy either failed job.', (await self.store.list_memory_blocks(self.session))[0].summary_text)

    async def test_new_block_scope_is_shown_once_and_legacy_rendering_survives_restart(self):
        start = '2026-09-15T01:00:00+00:00'
        shown_time = '2026-09-15T09:00:00+08:00'
        summary = ('## Scope\n- Episode summary\n- Time span: ' + start + '\n'
            '- Participants: Participant 7, Participant 8\n- Topics: travel, tickets\n'
            '## Decisions\n- Bring the blue ticket.')
        blocks = []
        for number in (40, 41):
            source = await self.store.append_message(self.session, peer(f'Ticket discussion {number}.', number))
            blocks.append(await self.store.create_memory_block(self.session, summary_text=summary, estimated_tokens=90,
                source_message_ids=[source.db_id], actor_labels=['Participant 7', 'Participant 8'],
                topic_labels=['travel', 'tickets'], time_start=start, time_end=start))
        async with self.store.pool.connection() as conn:
            await conn.execute("UPDATE memory_blocks SET details=details-'presentation_version' WHERE id=%s",
                (blocks[0].block_id,))
        reader = await self.new_store()
        old, new = await reader.list_memory_blocks(self.session)
        old_text = message_body(old.render_as_message(timezone=self.config.default_metadata_timezone))
        new_text = message_body(new.render_as_message(timezone=self.config.default_metadata_timezone))
        expected_header = (f'[Memory episode block L1 #{old.sequence_no}; covers 1 earlier messages; '
            f'time={shown_time}; actors=Participant 7, Participant 8; topics=travel, tickets]')
        self.assertEqual(old_text, expected_header + '\n' + summary.replace(start, shown_time))
        self.assertEqual(new_text.count(shown_time), 1)
        self.assertEqual(new_text.count('Participant 7'), 1)
        self.assertEqual(new_text.count('travel, tickets'), 1)
        self.assertIn('Bring the blue ticket.', new_text)
        again = await (await self.new_store()).list_memory_blocks(self.session)
        self.assertEqual([message_body(block.render_as_message(timezone=self.config.default_metadata_timezone)) for block in again],
            [old_text, new_text])

    async def test_reply_target_stays_with_trigger_after_later_people_speak(self):
        first = await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=peer('Please inspect my report.', 50, actor=7))
        later = await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=peer('I am leaving for lunch.', 51, actor=8))
        self.provider.responses.append(ProviderResponse(final_text='I will inspect your report.'))
        result = await self.runtime.run_turn_from_stored(session_id=self.session,
            user_display_name='Participant', trigger_message_id=first.db_id)
        target = next(message for message in self.provider.requests[-1]['messages']
            if message.metadata.get('synthetic_role') == 'reply_target')
        self.assertEqual(target.metadata['reply_target']['message_id'], first.db_id)
        self.assertEqual(target.metadata['reply_target']['speaker']['id'], 'person_id:7')
        self.assertEqual(result.reply_target['message_id'], first.db_id)
        shown = '\n'.join(part.text or '' for message in self.provider.requests[-1]['messages'] for part in message.parts)
        self.assertEqual(shown.count('Please inspect my report.'), 1)
        self.assertIn('I am leaving for lunch.', shown)
        self.assertNotEqual(first.db_id, later.db_id)

    async def test_compacted_trigger_retains_exact_plain_request_after_control_metadata(self):
        original = peer('> 这是同事说的。\n请只检查附件，不要替我同意这句话。', 60)
        original.metadata['entities'] = [{'type': 'blockquote', 'offset': 0, 'length': 9}]
        trigger = await self.runtime.ingest_user_message(session_id=self.session, incoming_message=original)
        await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=peer('I will come back later.', 61, actor=8))
        changed = False

        async def compact_once(**kwargs):
            nonlocal changed
            if changed:
                return False
            changed = True
            await self.store.create_memory_block(self.session, summary_text='A report request was received.',
                estimated_tokens=12, source_message_ids=[trigger.db_id])
            await self.runtime._reload_live_state(kwargs['state'])
            return True

        self.provider.responses.append(ProviderResponse(final_text='I will inspect the attachment.'))
        with patch.object(self.runtime, '_compact_if_needed', side_effect=compact_once):
            await self.runtime.run_turn_from_stored(session_id=self.session,
                user_display_name='Participant', trigger_message_id=trigger.db_id)
        target = next(message for message in self.provider.requests[-1]['messages']
            if message.metadata.get('synthetic_role') == 'reply_target')
        self.assertTrue(target.parts[0].text.startswith('[Application reply target: '),
            [part.text for part in target.parts])
        self.assertEqual(target.parts[-1].text, original.parts[0].text)
        self.assertEqual(target.metadata['reply_target']['message_id'], trigger.db_id)
        self.assertEqual(target.metadata['reply_target']['speaker']['id'], 'person_id:7')
        self.assertEqual(message_body((await self.store.read_messages(self.session, [trigger.db_id]))[0].message),
            original.parts[0].text)
