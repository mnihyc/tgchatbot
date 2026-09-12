from __future__ import annotations

import copy
import json
from typing import Any

import httpx

from tgchatbot.config import ChatCompletionsConfig
from tgchatbot.core.token_estimator import TokenEstimator
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind, ProviderResponse, SessionSettings, ToolCall, UsageInfo
from tgchatbot.logging_config import dump_llm_exchange
from tgchatbot.providers.base import (ControlDescriptor, ProviderCapabilities, RequestTokenEstimate,
    estimate_json_schema_tokens, evidence_text, pending_image_tokens, tool_message_evidence)
from tgchatbot.tools.base import ToolSpec


class ChatCompletionsProvider:
    """Stateless Chat Completions adapter for DeepSeek, OpenRouter and custom endpoints."""

    def __init__(self, config: ChatCompletionsConfig) -> None:
        self.config = config
        self.name = config.name
        self.capabilities = ProviderCapabilities(
            multimodal_input=config.multimodal_input,
            function_tools=config.function_tools,
            structured_output=True,
        )
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip('/') + '/',
            headers={'Authorization': f'Bearer {config.api_key}', 'Content-Type': 'application/json'},
            timeout=httpx.Timeout(config.request_timeout_s, connect=config.connect_timeout_s),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def supports_tool_evidence(self, settings: SessionSettings) -> bool:
        # Standard Chat Completions role=tool accepts text, regardless of
        # whether this model can read images on ordinary user messages.
        return False

    def describe_controls(self, settings: SessionSettings) -> dict[str, ControlDescriptor]:
        controls = {'native_web_search': ControlDescriptor(False, 'n/a', 'n/a')}
        for name in ('temperature', 'top_p', 'max_output_tokens'):
            value = getattr(settings, name)
            source = 'session' if value is not None else 'default'
            if value is None:
                value = getattr(self.config, name)
            controls[name] = ControlDescriptor(True, str(value) if value is not None else 'provider default', source)
        return controls

    def _messages_for_request(self, messages: list[ConversationMessage]) -> list[dict[str, Any]]:
        result = []
        for message in messages:
            for item in self._message_to_input_items(message):
                if (result and item.get('tool_calls') and result[-1].get('tool_calls')
                        and not item.get('content') and not result[-1].get('content')):
                    result[-1]['tool_calls'].extend(item['tool_calls'])
                else:
                    result.append(item)
        return result

    def _message_to_input_items(self, message: ConversationMessage) -> list[dict[str, Any]]:
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        native = metadata.get('provider_native')
        if isinstance(native, dict) and native.get('provider') == self.name and isinstance(native.get('items'), list):
            return copy.deepcopy([item for item in native['items'] if isinstance(item, dict)])
        payload = metadata.get('tool_payload')
        framework_refresh = metadata.get('synthetic_role') == 'profile_refresh'
        if message.role == MessageRole.TOOL and (metadata.get('tool_provider') == self.name or framework_refresh or metadata.get('tool_evidence') or metadata.get('portable_tool_history')) and isinstance(payload, dict):
            call_id = payload.get('call_id')
            if call_id and metadata.get('tool_phase') == 'call' and message.name:
                return [{'role': 'assistant', 'content': None, 'tool_calls': [{
                    'id': str(call_id), 'type': 'function',
                    'function': {'name': message.name, 'arguments': json.dumps(payload.get('arguments') or {}, ensure_ascii=False)},
                }]}]
            if call_id and metadata.get('tool_phase') == 'result':
                return self.make_tool_result_items(ToolCall(message.name or '', str(call_id), {}),
                    payload.get('output') or {}, tool_message_evidence(message))
        role = 'user' if message.role == MessageRole.TOOL else message.role.value
        parts: list[dict[str, Any]] = []
        for part in message.parts:
            if part.kind == PartKind.TEXT and part.text:
                parts.append({'type': 'text', 'text': part.text})
            elif part.kind == PartKind.IMAGE:
                if self.capabilities.multimodal_input and role == 'user' and part.data_b64 and part.mime_type:
                    parts.append({'type': 'image_url', 'image_url': {'url': f'data:{part.mime_type};base64,{part.data_b64}'}})
                else:
                    parts.append({'type': 'text', 'text': '[Image attached; image input is unavailable for this provider profile or message role.]'})
            elif part.kind == PartKind.FILE:
                descriptor = f'[Attached file: {part.filename or "file"}'
                if part.mime_type:
                    descriptor += f', {part.mime_type}'
                if part.size_bytes is not None:
                    descriptor += f', {part.size_bytes} bytes'
                if part.artifact_path:
                    descriptor += f', remote_path={part.artifact_path}'
                parts.append({'type': 'text', 'text': descriptor + ']'})
            elif part.kind == PartKind.STICKER:
                parts.append({'type': 'text', 'text': part.text or '[Sticker]'})
        if not parts:
            return []
        content: str | list[dict[str, Any]] = parts
        if all(part['type'] == 'text' for part in parts):
            content = '\n'.join(part['text'] for part in parts)
        return [{'role': role, 'content': content}]

    async def generate(
        self,
        *,
        settings: SessionSettings,
        messages: list[ConversationMessage],
        instructions: str,
        tools: list[ToolSpec],
        extra_input_items: list[dict] | None = None,
        response_schema: dict[str, Any] | None = None,
        response_schema_name: str | None = None,
    ) -> ProviderResponse:
        if response_schema and self.config.structured_output in {'json_object', 'prompt'}:
            instructions += '\n\nReturn only a JSON object matching this JSON Schema:\n' + json.dumps(response_schema, ensure_ascii=False)
        wire_messages = [{'role': 'system', 'content': instructions}] if instructions else []
        wire_messages.extend(self._messages_for_request(messages))
        wire_messages.extend(copy.deepcopy(extra_input_items or []))
        payload: dict[str, Any] = {
            **copy.deepcopy(self.config.extra_body),
            'model': settings.model,
            'messages': wire_messages,
            'stream': False,
            self.config.token_limit_parameter: settings.max_output_tokens if settings.max_output_tokens is not None else self.config.max_output_tokens,
        }
        for name in ('temperature', 'top_p'):
            value = getattr(settings, name)
            if value is None:
                value = getattr(self.config, name)
            if value is not None:
                payload[name] = value
        tier = settings.service_tier or getattr(self.config, 'service_tier', None)
        if tier:
            payload['service_tier'] = tier
        if tools and self.capabilities.function_tools:
            payload['tools'] = [{'type': 'function', 'function': tool.generic_function_declaration()} for tool in tools]
            payload['tool_choice'] = 'auto'
        if response_schema:
            if self.config.structured_output == 'json_schema':
                payload['response_format'] = {'type': 'json_schema', 'json_schema': {
                    'name': response_schema_name or 'structured_output', 'strict': True, 'schema': response_schema,
                }}
            elif self.config.structured_output == 'json_object':
                payload['response_format'] = {'type': 'json_object'}
        response = await self._client.post('chat/completions', json=payload)
        dump_llm_exchange(provider=self.name, model=settings.model, url='chat/completions', payload=payload, response=response)
        response.raise_for_status()
        return self._parse_response(response.json())

    def _parse_response(self, body: dict[str, Any]) -> ProviderResponse:
        if body.get('error'):
            raise RuntimeError(f'{self.name} returned an API error')
        choices = body.get('choices')
        if not isinstance(choices, list) or not choices or not isinstance(choices[0].get('message'), dict):
            raise ValueError(f'{self.name} returned no completion message')
        message = choices[0]['message']
        # Keep provider reasoning fields intact for signed/opaque tool continuations.
        # Visible response text is deliberately read only from content/refusal.
        native = {key: copy.deepcopy(message[key]) for key in ('role', 'content', 'tool_calls', 'reasoning_content', 'reasoning', 'reasoning_details') if key in message}
        native['role'] = 'assistant'
        native.setdefault('content', None)
        content = message.get('content')
        if isinstance(content, list):
            text = ''.join(str(part.get('text') or '') for part in content if isinstance(part, dict) and part.get('type') == 'text')
        else:
            text = str(content or message.get('refusal') or '')
        tool_calls = []
        for call in message.get('tool_calls') or []:
            function = call.get('function') or {}
            arguments = function.get('arguments', '{}')
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict) or not call.get('id') or not function.get('name'):
                raise ValueError(f'{self.name} returned an invalid function call')
            tool_calls.append(ToolCall(name=function['name'], call_id=call['id'], arguments=arguments))
        usage = body.get('usage') or {}
        input_tokens, output_tokens = usage.get('prompt_tokens'), usage.get('completion_tokens')
        cached_input_tokens = (usage.get('prompt_tokens_details') or {}).get('cached_tokens')
        if cached_input_tokens is None:
            cached_input_tokens = usage.get('prompt_cache_hit_tokens')
        total_tokens = usage.get('total_tokens')
        if total_tokens is None and (input_tokens is not None or output_tokens is not None):
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
        return ProviderResponse(
            final_text=text.strip(), tool_calls=tool_calls, continuation_items=[native],
            usage=UsageInfo(
                input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens,
                cached_input_tokens=cached_input_tokens,
                service_tier=body.get('service_tier'),
            ), raw=body,
        )

    def make_tool_result_items(self, tool_call: ToolCall, tool_output: dict,
                               evidence_parts: list[MessagePart] | None = None) -> list[dict]:
        content = json.dumps(tool_output, ensure_ascii=False)
        labels = [evidence_text(part) for part in evidence_parts or []]
        if any(labels):
            content += '\n\nTool evidence:\n' + '\n'.join(text for text in labels if text)
        return [{'role': 'tool', 'tool_call_id': tool_call.call_id, 'content': content}]

    def persistent_history_items(self, response: ProviderResponse) -> list[dict]:
        return copy.deepcopy(response.continuation_items)

    def estimate_request_tokens(
        self,
        *,
        settings: SessionSettings,
        messages: list[ConversationMessage],
        instructions: str,
        tools: list[ToolSpec],
        extra_input_items: list[dict] | None = None,
        response_schema: dict[str, Any] | None = None,
        response_schema_name: str | None = None,
        history_tokens_override: int | None = None,
    ) -> RequestTokenEstimate:
        def estimate(value: Any) -> int:
            if isinstance(value, dict):
                if value.get('type') == 'image_url':
                    return TokenEstimator.IMAGE_TOKENS
                return sum(TokenEstimator.estimate_text(str(key)) + estimate(item) for key, item in value.items())
            if isinstance(value, list):
                return sum(estimate(item) for item in value)
            return TokenEstimator.estimate_text(str(value)) if value is not None else 0

        # Match the legacy estimator's framing allowance. The runtime calibrates
        # these semantic estimates against observed token usage per provider/model.
        history_tokens = history_tokens_override if history_tokens_override is not None else estimate(self._messages_for_request(messages))
        if history_tokens_override is None and self.capabilities.multimodal_input:
            history_tokens += pending_image_tokens([message for message in messages if message.role == MessageRole.USER])
        return RequestTokenEstimate.compose(
            history_tokens=history_tokens,
            instructions_tokens=TokenEstimator.estimate_text(instructions),
            tools_tokens=estimate([tool.generic_function_declaration() for tool in tools]) + estimate_json_schema_tokens(response_schema, name=response_schema_name),
            extra_input_tokens=estimate(extra_input_items or []),
            framing_tokens=48 + 8 * (len(messages) + len(extra_input_items or [])),
        )
