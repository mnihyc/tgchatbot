"""Provider outcomes through real runtime retries and isolated PostgreSQL.

Only hosted HTTP and the selected-sticker tool result are controlled. Failed
fulfillment retries without replaying tools; completed silence remains valid.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx

from tests.business_helpers import BusinessTestCase
from tgchatbot.domain.models import ChatMode, ConversationMessage, MessageRole, OutboundSticker, ToolResult
from tgchatbot.providers.openai_responses import OpenAIResponsesProvider
from tgchatbot.storage.postgres_store import message_body
from tgchatbot.tools.base import ToolSpec


class OpenAIResponseOutcomeTests(BusinessTestCase):
    async def use_provider(self, responses):
        provider = OpenAIResponsesProvider(self.config.openai)
        await provider.aclose()
        captured = []

        def respond(request):
            captured.append(json.loads(request.content))
            return httpx.Response(200, json=responses.pop(0))

        provider._client = httpx.AsyncClient(base_url='https://provider.invalid/',
            transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(provider.aclose)
        self.runtime.providers = {'openai': provider}
        return captured

    async def test_explicit_failed_response_retries_without_becoming_an_empty_answer(self):
        await self.settings(provider_retry_count=1)
        failed_variants = [
            {'status': 'failed', 'error': {'code': 'server_error', 'message': 'Temporary model failure'}, 'output': []},
            {'status': 'failed', 'error': None, 'output': []},
            {'error': {'code': 'server_error', 'message': 'Temporary model failure'}, 'output': []},
            {'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'},
             'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'Wrong partial answer'}]}]},
            {'status': 'completed', 'output': [{'type': 'function_call', 'status': 'incomplete',
             'name': 'shell_exec', 'call_id': 'never-run', 'arguments': '{}'}]},
            {'status': 'completed', 'output': [{'type': 'message', 'content': [
             {'type': 'refusal', 'refusal': 'I cannot provide that requested content.'}]}]},
        ]
        for failed in failed_variants:
            with self.subTest(outcome=failed):
                captured = await self.use_provider([failed, {'status': 'completed', 'error': None,
                    'output': [{'type': 'message', 'role': 'assistant',
                        'content': [{'type': 'output_text', 'text': 'The keys are in the blue bag.'}]}]}])
                with self.assertLogs('tgchatbot.core.runtime', level='WARNING'):
                    result = await self.runtime.run_turn(session_id=self.session,
                        user_display_name='Participant', incoming_message=ConversationMessage.user_text('Where are the keys?'))
                self.assertEqual(result.text, 'The keys are in the blue bag.')
                self.assertEqual(len(captured), 2)
                self.assertEqual(captured[0], captured[1], 'Retry the same request without fabricating an intermediate answer')
                self.tools.runner.run.assert_not_awaited()
                originals = await self.store.list_canonical_messages(self.session)
                self.assertFalse(any(row.message.role in {MessageRole.ASSISTANT, MessageRole.TOOL}
                                     for row in originals))

    async def test_exhausted_explicit_failure_retains_original_for_a_later_turn(self):
        await self.settings(provider_retry_count=1)
        failed = {'status': 'failed', 'error': {'code': 'server_error'}, 'output': []}
        captured = await self.use_provider([failed, failed])
        text = 'The spare keys are in the blue bag; please remember.'
        with self.assertLogs('tgchatbot.core.runtime', level='WARNING'):
            with self.assertRaisesRegex(RuntimeError, 'API error'):
                await self.runtime.run_turn(session_id=self.session, user_display_name='Participant',
                    incoming_message=ConversationMessage.user_text(text))
        self.assertEqual(len(captured), 2)
        originals = await self.store.list_canonical_messages(self.session)
        self.assertEqual([message_body(row.message) for row in originals
                          if not row.message.metadata.get('synthetic_role')], [text])

    async def test_refusal_words_in_ordinary_text_are_not_a_provider_block_signal(self):
        await self.settings(provider_retry_count=1)
        text = 'The quoted message was “I cannot provide that requested content.”'
        captured = await self.use_provider([{'status': 'completed', 'error': None,
            'output': [{'type': 'message', 'role': 'assistant',
                'content': [{'type': 'output_text', 'text': text}]}]}])
        result = await self.runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Repeat the quoted sentence.'))
        self.assertEqual(result.text, text)
        self.assertEqual(len(captured), 1)
        self.assertFalse(result.stickers)

    async def test_retry_after_selected_sticker_does_not_repeat_the_tool_and_accepts_completed_silence(self):
        await self.settings(mode=ChatMode.ASSIST, provider_retry_count=1, max_interaction_rounds=1)
        asset = self.path / 'selected.webp'
        asset.write_bytes(b'fixture selected sticker; no transport call in this test')
        runner = SimpleNamespace(run=AsyncMock(return_value=ToolResult('', 'sticker_send_selected',
            {'ok': True, 'status': 'queued'}, stickers=[OutboundSticker(asset, source_id='fixture-selected')])))
        self.tools.list_tools.return_value = [ToolSpec('sticker_send_selected', 'Select the known sticker',
            {'type': 'object', 'properties': {}}, runner)]
        captured = await self.use_provider([
            {'status': 'completed', 'error': None, 'output': [{'type': 'function_call',
                'call_id': 'selected', 'name': 'sticker_send_selected', 'arguments': '{}'}]},
            {'status': 'incomplete', 'incomplete_details': {'reason': 'max_output_tokens'},
                'output': [{'type': 'function_call', 'call_id': 'duplicate',
                    'name': 'sticker_send_selected', 'arguments': '{}'}]},
            {'status': 'completed', 'error': None, 'output': []},
        ])
        result = await self.runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Only send the sticker, without text.'))
        self.assertEqual(result.text, '')
        self.assertEqual([sticker.source_id for sticker in result.stickers], ['fixture-selected'])
        runner.run.assert_awaited_once()
        self.assertEqual(len(captured), 3, 'Only the unfinished continuation is retried')
        self.assertEqual(captured[1], captured[2])
