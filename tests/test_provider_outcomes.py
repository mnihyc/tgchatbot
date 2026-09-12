"""Incomplete provider output must never become text, executable tools or learned facts."""
from __future__ import annotations

from copy import deepcopy
import unittest

from tgchatbot.providers.base import ProviderOutcomeError
from tgchatbot.providers.chat_completions import ChatCompletionsProvider
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.providers.openai_responses import OpenAIResponsesProvider


class ProviderCompletionTests(unittest.TestCase):
    def fixtures(self):
        gemini = object.__new__(GeminiProvider)
        responses = object.__new__(OpenAIResponsesProvider)
        chat = object.__new__(ChatCompletionsProvider)
        chat.name = 'compatible'
        text = '{"additions":[],"removals":[]}'
        return (
            (gemini, {'candidates': [{'finishReason': 'STOP', 'content': {'role': 'model', 'parts': [
                {'text': text}, {'functionCall': {'name': 'send_sticker', 'args': {}, 'id': 'once'}}]}}],
                'usageMetadata': {'promptTokenCount': 100, 'candidatesTokenCount': 20,
                    'totalTokenCount': 120, 'cachedContentTokenCount': 80, 'serviceTier': 'flex'}}),
            (responses, {'status': 'completed', 'output': [
                {'type': 'message', 'status': 'completed', 'content': [{'type': 'output_text', 'text': text}]},
                {'type': 'function_call', 'status': 'completed', 'name': 'send_sticker',
                    'arguments': '{}', 'call_id': 'once'}], 'usage': {'input_tokens': 100, 'output_tokens': 20,
                    'total_tokens': 120, 'input_tokens_details': {'cached_tokens': 80}}, 'service_tier': 'flex'}),
            (chat, {'choices': [{'finish_reason': 'tool_calls', 'message': {'role': 'assistant',
                'content': text, 'refusal': None, 'tool_calls': [{'id': 'once', 'type': 'function',
                    'function': {'name': 'send_sticker', 'arguments': '{}'}}]}}],
                'usage': {'prompt_tokens': 100, 'completion_tokens': 20, 'total_tokens': 120,
                    'prompt_tokens_details': {'cached_tokens': 80}}, 'service_tier': 'flex'}),
        )

    def rejected(self, provider, body):
        with self.assertRaises(ProviderOutcomeError) as failed:
            provider._parse_response(body)
        error = failed.exception
        self.assertEqual((error.usage.input_tokens, error.usage.output_tokens, error.usage.total_tokens), (100, 20, 120))
        self.assertEqual(error.usage.cached_input_tokens, 80)
        self.assertEqual(error.usage.service_tier, 'flex')
        self.assertTrue(error.reason)

    def test_complete_text_and_function_calls_remain_usable(self):
        for provider, body in self.fixtures():
            with self.subTest(provider=provider.name):
                parsed = provider._parse_response(body)
                self.assertEqual(parsed.final_text, '{"additions":[],"removals":[]}')
                self.assertEqual([(call.name, call.call_id, call.arguments) for call in parsed.tool_calls],
                    [('send_sticker', 'once', {})])

    def test_truncation_blocks_even_parseable_json_and_plausible_tool_calls(self):
        for provider, complete in self.fixtures():
            if provider.name == 'gemini':
                reasons = ('MAX_TOKENS', 'SAFETY', 'RECITATION', 'OTHER', 'BLOCKLIST',
                           'PROHIBITED_CONTENT', 'SPII', 'MALFORMED_FUNCTION_CALL', None)
            elif provider.name == 'openai':
                reasons = ('incomplete', 'failed', 'cancelled', 'queued', 'in_progress', None)
            else:
                reasons = ('length', 'content_filter', 'function_call', 'unknown', None)
            for reason in reasons:
                with self.subTest(provider=provider.name, reason=reason):
                    body = deepcopy(complete)
                    if provider.name == 'gemini':
                        body['candidates'][0]['finishReason'] = reason
                    elif provider.name == 'openai':
                        body['status'] = reason
                    else:
                        body['choices'][0]['finish_reason'] = reason
                    self.rejected(provider, body)

    def test_typed_blocks_override_apparently_completed_output(self):
        gemini, responses, chat = self.fixtures()
        variants = []
        for feedback in ({'blockReason': 'SAFETY'}, {'blockReason': 'OTHER'},
                         {'safetyRatings': [{'blocked': True}]}):
            variants.append((gemini[0], deepcopy(gemini[1]) | {'promptFeedback': feedback}))
        candidate_blocked = deepcopy(gemini[1])
        candidate_blocked['candidates'][0]['safetyRatings'] = [{'blocked': True}]
        variants.append((gemini[0], candidate_blocked))
        refusal = deepcopy(responses[1])
        refusal['output'][0]['content'] = [{'type': 'refusal', 'refusal': 'Declined'}]
        variants.append((responses[0], refusal))
        incomplete_tool = deepcopy(responses[1])
        incomplete_tool['output'][1]['status'] = 'incomplete'
        variants.append((responses[0], incomplete_tool))
        refusal = deepcopy(chat[1])
        refusal['choices'][0]['message']['refusal'] = 'Declined'
        variants.append((chat[0], refusal))
        for provider, body in variants:
            with self.subTest(provider=provider.name, body=body):
                self.rejected(provider, body)

    def test_missing_results_are_failures_but_completed_empty_responses_are_valid(self):
        for provider, full in self.fixtures():
            empty = deepcopy(full)
            if provider.name == 'gemini':
                empty['candidates'][0]['content']['parts'] = []
            elif provider.name == 'openai':
                empty['output'] = []
            else:
                empty['choices'][0] = {'finish_reason': 'stop', 'message': {'content': None}}
            with self.subTest(provider=provider.name):
                response = provider._parse_response(empty)
                self.assertEqual(response.final_text, '')
                self.assertEqual(response.tool_calls, [])
                if provider.name == 'gemini':
                    empty['candidates'] = []
                elif provider.name == 'openai':
                    del empty['status']
                else:
                    empty['choices'][0]['finish_reason'] = 'tool_calls'
                self.rejected(provider, empty)

    def test_ordinary_refusal_language_does_not_trigger_a_content_heuristic(self):
        text = 'The sender wrote: I cannot help with that.'
        for provider, body in self.fixtures():
            if provider.name == 'gemini':
                body['candidates'][0]['content']['parts'] = [{'text': text}]
            elif provider.name == 'openai':
                body['output'] = [{'type': 'message', 'content': [{'type': 'output_text', 'text': text}]}]
            else:
                body['choices'][0] = {'finish_reason': 'stop', 'message': {'content': text}}
            with self.subTest(provider=provider.name):
                self.assertEqual(provider._parse_response(body).final_text, text)
