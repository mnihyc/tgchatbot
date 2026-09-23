"""Excess tool calls are refused without losing results or changing request prefixes."""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from unittest.mock import AsyncMock

import httpx

from tests.business_helpers import BusinessTestCase
from tgchatbot.config import ChatCompletionsConfig
from tgchatbot.core.runtime import AgentRuntime, ContextLimitExceeded
from tgchatbot.core.compaction_schema import compaction_json_schema
from tgchatbot.domain.models import (ChatMode, ConversationMessage, PromptInjectionMode,
    ToolHistoryMode, ToolResult)
from tgchatbot.providers.chat_completions import ChatCompletionsProvider
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.providers.openai_responses import OpenAIResponsesProvider
from tgchatbot.storage.previews import PreviewCache


class ToolRoundLimitTests(BusinessTestCase):
    def response(self, name, *, calls=(), text='Finished.'):
        if name == 'gemini':
            parts = [{'functionCall': {'name': 'shell_exec', 'id': call_id, 'args': {}}}
                for call_id in calls] if calls else [{'text': text}]
            parts[0]['thoughtSignature'] = 'signature-' + (calls[0] if calls else text)
            return {'candidates': [{'finishReason': 'STOP', 'content': {'role': 'model', 'parts': parts}}]}
        if name == 'openai':
            items = [{'type': 'reasoning', 'id': 'reasoning-' + calls[0],
                'encrypted_content': 'opaque-' + calls[0], 'summary': []},
                *[{'type': 'function_call', 'id': 'item-' + call_id, 'call_id': call_id,
                    'name': 'shell_exec', 'arguments': '{}'} for call_id in calls]] if calls else [
                {'type': 'message', 'id': 'answer-' + text, 'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': text}]}]
            return {'status': 'completed', 'output': items}
        message = {'role': 'assistant', 'content': None if calls else text,
            'reasoning_content': 'opaque-' + (calls[0] if calls else text)}
        if calls:
            message['tool_calls'] = [{'id': call_id, 'type': 'function',
                'function': {'name': 'shell_exec', 'arguments': '{}'}} for call_id in calls]
        return {'choices': [{'finish_reason': 'tool_calls' if calls else 'stop', 'message': message}]}

    async def make_provider(self, name, scripts, wire):
        if name == 'gemini':
            provider = GeminiProvider(replace(self.config.gemini, api_key='mock', model='gemini-3.8-flash'))
        elif name == 'openai':
            provider = OpenAIResponsesProvider(replace(self.config.openai, api_key='mock'))
        else:
            provider = ChatCompletionsProvider(ChatCompletionsConfig(name=name, api_key='mock',
                base_url='https://fixture.invalid/v1', model='fixture'))
            self.config = replace(self.config, chat_completions=(provider.config,))
            self.runtime.config = self.config
        await provider.aclose()

        def handle(request):
            wire.append(json.loads(request.content))
            self.assertTrue(scripts, 'Unexpected request after the scripted conversation finished')
            return httpx.Response(200, json=copy.deepcopy(scripts.pop(0)))

        provider._client = httpx.AsyncClient(base_url='https://fixture.invalid/v1/',
            transport=httpx.MockTransport(handle))
        self.addAsyncCleanup(provider.aclose)
        self.runtime.providers = {name: provider}
        return provider

    @staticmethod
    def results(name, request):
        if name == 'gemini':
            return {part['functionResponse']['id']: part['functionResponse']['response']['result']
                for item in request['contents'] for part in item['parts'] if 'functionResponse' in part}
        if name == 'openai':
            return {item['call_id']: json.loads(item['output'])
                for item in request['input'] if item.get('type') == 'function_call_output'}
        return {item['tool_call_id']: json.loads(item['content'])
            for item in request['messages'] if item.get('role') == 'tool'}

    async def exercise_continuation(self, name):
        wire = []
        scripts = [self.response(name, calls=('first', 'last')),
            self.response(name, calls=('blocked-1', 'blocked-2')),
            self.response(name, calls=('retry',)), self.response(name, text='First completed.'),
            self.response(name, calls=('new-turn',)), self.response(name, text='Next completed.'),
            self.response(name), self.response(name)]
        native_responses = copy.deepcopy(scripts)
        provider = await self.make_provider(name, scripts, wire)
        settings = await self.settings(provider=name, model=provider.config.model, mode=ChatMode.ASSIST,
            system_prompt='Keep this exact preset.', prompt_injection_mode=PromptInjectionMode.EXACT,
            native_web_search_mode='on', tool_history_mode=ToolHistoryMode.NATIVE_SAME_PROVIDER,
            max_interaction_rounds=1, compact_trigger_tokens=100000)
        self.tools.runner.run.side_effect = lambda *args: ToolResult('', 'shell_exec', {'ok': True, 'value': 'Checked.'})
        result = await self.runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Check both notes and resolve differences.'), emit=AsyncMock())
        self.assertEqual(result.text, 'First completed.')
        self.assertEqual(self.tools.runner.run.await_count, 2)
        first_results, all_results = self.results(name, wire[1]), self.results(name, wire[2])
        self.assertEqual(first_results, {call_id: {'ok': True, 'value': 'Checked.'}
            for call_id in ('first', 'last')})
        self.assertEqual(first_results['last'], all_results['last'])
        refusal = {'ok': False, 'application_note':
            'Tool-call limit reached for this turn. This call was not executed. Finish the reply using the available results.'}
        self.assertEqual(all_results['blocked-1'], refusal)
        self.assertEqual(all_results['blocked-2'], refusal)
        self.assertEqual(self.results(name, wire[3])['retry'], refusal)

        async def record(runtime, completed):
            await runtime.record_assistant_text(session_id=self.session, text=completed.text, metadata={
                'provider_native': {'provider': name, 'model': settings.model,
                    'items': completed.provider_history_items}})

        await record(self.runtime, result)
        second = await self.runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Now check another note.'), emit=AsyncMock())
        self.assertEqual(second.text, 'Next completed.')
        self.assertEqual(self.tools.runner.run.await_count, 3)
        self.assertEqual(self.results(name, wire[5])['new-turn'], {'ok': True, 'value': 'Checked.'})
        await record(self.runtime, second)
        followup = await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text('Are the checks complete?'))
        await self.runtime.run_turn_from_stored(session_id=self.session,
            user_display_name='Participant', trigger_message_id=followup.db_id)

        await self.store.close()
        restored_store = await self.new_store()
        cache = PreviewCache(restored_store, max_bytes=0)
        self.addCleanup(cache.close)
        restored = AgentRuntime(config=self.config, store=restored_store, tool_registry=self.tools,
            providers={name: provider}, preview_cache=cache)
        await restored.run_turn_from_stored(session_id=self.session,
            user_display_name='Participant', trigger_message_id=followup.db_id)
        self.assertEqual(wire[-1], wire[-2])
        self.assertEqual(self.tools.runner.run.await_count, 3)
        self.assertFalse(scripts)

        history_key = {'gemini': 'contents', 'openai': 'input', 'compatible': 'messages'}[name]
        for previous, current in zip(wire[:-1], wire[1:]):
            self.assertEqual(current[history_key][:len(previous[history_key])], previous[history_key])
        invariant = {key: value for key, value in wire[0].items() if key != history_key}
        for request in wire:
            self.assertEqual({key: value for key, value in request.items() if key != history_key}, invariant)
        if name == 'gemini':
            self.assertEqual(wire[0]['systemInstruction'], {'parts': [{'text': settings.system_prompt}]})
            original = [body['candidates'][0]['content'] for body in native_responses[:-2]]
        elif name == 'openai':
            self.assertEqual(wire[0]['instructions'], settings.system_prompt)
            original = [item for body in native_responses[:-2] for item in body['output']]
        else:
            self.assertEqual(wire[0]['messages'][0], {'role': 'system', 'content': settings.system_prompt})
            original = [body['choices'][0]['message'] for body in native_responses[:-2]]
        for item in original:
            self.assertEqual(wire[-1][history_key].count(item), 1)

    async def test_gemini_blocks_excess_calls_and_restarts_identically(self):
        await self.exercise_continuation('gemini')

    async def test_responses_blocks_excess_calls_and_restarts_identically(self):
        await self.exercise_continuation('openai')

    async def test_compatible_blocks_excess_calls_and_restarts_identically(self):
        await self.exercise_continuation('compatible')

    async def test_immediate_compaction_keeps_the_unseen_refusal(self):
        wire = []
        blocked = self.response('gemini', calls=('blocked',))
        # The first result fits. The next (refused) command crosses the ceiling,
        # so the already-seen result can compact while the refusal stays intact.
        blocked['candidates'][0]['content']['parts'][0]['functionCall']['args'] = {
            'command': 'never execute this command; '*5000}
        summary = {key: [] for key in compaction_json_schema('toolspan')['properties']}
        summary.update(scope='A shell check returned diagnostic details.', interaction_mode='task_execution')
        scripts = [self.response('gemini', calls=('last',)),
            blocked, self.response('gemini', text=json.dumps(summary)), self.response('gemini')]
        provider = await self.make_provider('gemini', scripts, wire)
        await self.settings(provider='gemini', model=provider.config.model, mode=ChatMode.ASSIST,
            max_interaction_rounds=1, compact_trigger_tokens=100000,
            tool_history_mode=ToolHistoryMode.NATIVE_SAME_PROVIDER)
        await self.runtime.ingest_user_message(session_id=self.session,
            incoming_message=ConversationMessage.user_text('Earlier discussion to summarize.'))
        self.tools.runner.run.side_effect = lambda *args: ToolResult('', 'shell_exec',
            {'ok': True, 'stdout': 'diagnostic detail ' * 18000})
        result = await self.runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Finish the current check.'), emit=AsyncMock())
        self.assertEqual(result.text, 'Finished.')
        self.assertEqual(len(await self.store.list_memory_blocks(self.session)), 1)
        self.tools.runner.run.assert_awaited_once()
        result_output = self.results('gemini', wire[-1])['blocked']
        self.assertFalse(result_output['ok'])
        self.assertIn('application_note', result_output)
        stored = await self.store.list_uncompacted_messages(self.session)
        self.assertEqual(sum(row.message.metadata.get('tool_payload', {}).get('call_id') == 'blocked'
            for row in stored), 2)
        self.assertEqual(wire[0]['systemInstruction'], wire[-1]['systemInstruction'])
        self.assertEqual(wire[0]['tools'], wire[-1]['tools'])

    async def test_oversized_unseen_result_blocks_next_request_and_remains_stored(self):
        wire = []
        summary = {key: [] for key in compaction_json_schema('episode')['properties']}
        summary.update(scope='The user requested a diagnostic check.', interaction_mode='task_execution')
        provider = await self.make_provider('gemini', [self.response('gemini', calls=('read',)),
            self.response('gemini', text=json.dumps(summary))], wire)
        await self.settings(provider='gemini', model=provider.config.model, mode=ChatMode.ASSIST,
            compact_trigger_tokens=100000, tool_history_mode=ToolHistoryMode.NATIVE_SAME_PROVIDER)
        output = 'diagnostic detail '*50000
        self.tools.runner.run.return_value = ToolResult('', 'shell_exec', {'ok':True,'stdout':output})
        with self.assertRaises(ContextLimitExceeded):
            await self.runtime.run_turn(session_id=self.session, user_display_name='Participant',
                incoming_message=ConversationMessage.user_text('Run the check.'))
        self.assertEqual(len(wire),2)
        self.assertIn('responseJsonSchema',wire[-1]['generationConfig'])
        rows = await self.store.list_uncompacted_messages(self.session)
        result = next(row for row in rows if row.message.metadata.get('tool_phase')=='result')
        self.assertEqual(result.message.metadata['tool_payload']['output']['stdout'],output)
