from __future__ import annotations

import copy
import json

import httpx

from tgchatbot.config import ChatCompletionsConfig
from tgchatbot.core.token_estimator import TokenEstimator
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, SessionSettings, ToolCall
from tgchatbot.logging_config import dump_llm_exchange
from tgchatbot.providers.base import ProviderCapabilities, RequestTokenEstimate, estimate_json_schema_tokens, pending_image_tokens
from tgchatbot.providers.chat_completions import ChatCompletionsProvider
from tgchatbot.providers.openai_responses import OpenAIResponsesProvider
from tgchatbot.tools.base import ToolSpec


class DeepSeekResponsesProvider(OpenAIResponsesProvider):
    """DeepSeek's Responses endpoint supports framework-inserted tool pairs.

    Retain the existing provider configuration and controls; share Responses
    parsing, continuation and accounting without OpenAI-specific request options.
    """

    name = 'deepseek'
    input_image_roles = (MessageRole.USER,)

    def __init__(self, config: ChatCompletionsConfig) -> None:
        self.config = config
        self._client = None
        self.capabilities = ProviderCapabilities(multimodal_input=config.multimodal_input,
            function_tools=config.function_tools, structured_output=True)

    def _ensure_client(self) -> httpx.AsyncClient:
        if not self.config.api_key:
            raise RuntimeError('DEEPSEEK_API_KEY not set')
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self.config.base_url.rstrip('/') + '/',
                headers={'Authorization': f'Bearer {self.config.api_key}', 'Content-Type': 'application/json'},
                timeout=httpx.Timeout(self.config.request_timeout_s, connect=self.config.connect_timeout_s))
        return self._client

    def describe_controls(self, settings: SessionSettings):
        return ChatCompletionsProvider.describe_controls(self, settings)

    def _tool_defs_for_request(self, settings: SessionSettings, tools: list[ToolSpec]) -> list[dict]:
        return ([{'type': 'function', 'strict': False, **tool.generic_function_declaration()} for tool in tools]
                if self.capabilities.function_tools else [])

    async def generate(self, *, settings, messages, instructions, tools,
                       extra_input_items=None, response_schema=None, response_schema_name=None):
        if response_schema and self.config.structured_output in {'json_object', 'prompt'}:
            instructions += '\n\nReturn only a JSON object matching this JSON Schema:\n' + json.dumps(response_schema, ensure_ascii=False)
        payload = copy.deepcopy(self.config.extra_body)
        thinking = payload.pop('thinking', {})
        effort = payload.pop('reasoning_effort', None)
        if thinking.get('type') == 'disabled':
            payload['reasoning'] = {'effort': 'none'}
        elif effort is not None:
            payload['reasoning'] = {'effort': effort}
        # Absent an override, leave DeepSeek's own thinking default in effect.
        tool_defs = self._tool_defs_for_request(settings, tools)
        payload.update(model=settings.model, instructions=instructions,
            input=[item for message in messages for item in self._message_to_input_items(message)]
                  + copy.deepcopy(extra_input_items or []),
            tools=tool_defs, tool_choice='auto' if tool_defs else 'none',
            max_output_tokens=settings.max_output_tokens if settings.max_output_tokens is not None else self.config.max_output_tokens)
        for name in ('temperature', 'top_p'):
            value = getattr(settings, name)
            if value is None:
                value = getattr(self.config, name)
            if value is not None:
                payload[name] = value
        if response_schema:
            if self.config.structured_output == 'json_schema':
                payload['text'] = {'format': {'type': 'json_schema', 'name': response_schema_name or 'structured_output',
                    'schema': response_schema, 'strict': True}}
            elif self.config.structured_output == 'json_object':
                payload['text'] = {'format': {'type': 'json_object'}}
        response = await self._ensure_client().post('responses', json=payload)
        dump_llm_exchange(provider=self.name, model=settings.model, url='responses', payload=payload, response=response)
        response.raise_for_status()
        return self._parse_response(response.json())

    @staticmethod
    def _responses_items(items: list[dict]) -> list[dict]:
        """Convert Chat Completions-shaped items only at the wire boundary."""
        result = []
        for item in items:
            if item.get('type'):
                result.append(copy.deepcopy(item))
                continue
            if item.get('role') == 'tool':
                result.append({'type': 'function_call_output', 'call_id': item['tool_call_id'], 'output': item['content']})
                continue
            if item.get('reasoning_content') is not None:
                result.append({'type': 'reasoning', 'summary': [],
                    'content': [{'type': 'reasoning_text', 'text': item['reasoning_content']}]})
            content = item.get('content')
            if isinstance(content, list):
                content = [({'type': 'input_image', 'image_url': part['image_url']['url'],
                             **({'detail': part['image_url']['detail']} if 'detail' in part['image_url'] else {})}
                            if part.get('type') == 'image_url' else
                            {'type': 'output_text' if item['role'] == 'assistant' else 'input_text', 'text': part['text']})
                           for part in content]
            if content:
                result.append({'role': item['role'], 'content': content})
            for call in item.get('tool_calls') or []:
                result.append({'type': 'function_call', 'call_id': call['id'],
                    'name': call['function']['name'], 'arguments': call['function']['arguments']})
        return result

    def _message_to_input_items(self, message: ConversationMessage) -> list[dict]:
        # The existing serializer also preserves role/image restrictions and
        # recognizes framework refreshes. Responses-native snapshots pass intact.
        return self._responses_items(ChatCompletionsProvider._message_to_input_items(self, message))

    def make_tool_result_items(self, tool_call: ToolCall, tool_output: dict,
                               evidence_parts: list[MessagePart] | None = None) -> list[dict]:
        return self._responses_items(ChatCompletionsProvider.make_tool_result_items(self, tool_call, tool_output, evidence_parts))

    def _estimate_input_item_tokens(self, item: dict) -> int:
        if item.get('type') == 'reasoning':
            return 16 + sum(TokenEstimator.estimate_text(part.get('text')) for part in item.get('content') or [])
        if 'role' in item and isinstance(item.get('content'), str):
            return 8 + TokenEstimator.estimate_text(item['role']) + TokenEstimator.estimate_text(item['content'])
        return super()._estimate_input_item_tokens(item)

    def estimate_request_tokens(self, *, settings, messages, instructions, tools,
                                extra_input_items=None, response_schema=None, response_schema_name=None,
                                history_tokens_override=None):
        history = history_tokens_override
        if history is None:
            history = sum(self._estimate_input_item_tokens(item) for message in messages
                          for item in self._message_to_input_items(message))
            if self.capabilities.multimodal_input:
                history += pending_image_tokens([m for m in messages if m.role == MessageRole.USER])
        return RequestTokenEstimate.compose(history_tokens=history,
            instructions_tokens=TokenEstimator.estimate_text(instructions),
            tools_tokens=sum(self._estimate_tool_definition_tokens(item) for item in self._tool_defs_for_request(settings, tools))
                         + estimate_json_schema_tokens(response_schema, name=response_schema_name),
            extra_input_tokens=sum(self._estimate_input_item_tokens(item) for item in extra_input_items or []),
            framing_tokens=48 + 8 * (len(messages) + len(extra_input_items or [])))
