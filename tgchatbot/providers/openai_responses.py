from __future__ import annotations

import json
from typing import Any

import httpx
import logging

from tgchatbot.config import OpenAIConfig
from tgchatbot.domain.models import ChatMode, ConversationMessage, MessageRole, PartKind, ProviderResponse, SessionSettings, ToolCall, UsageInfo
from tgchatbot.providers.base import ControlDescriptor, ProviderCapabilities
from tgchatbot.settings_schema import NATIVE_WEB_SEARCH_MAX_MAX, effective_optional_disabled_int, effective_reasoning_summary
from tgchatbot.tools.base import ToolSpec
from tgchatbot.logging_config import dump_llm_exchange

logger = logging.getLogger(__name__)

class OpenAIResponsesProvider:
    name = 'openai'
    capabilities = ProviderCapabilities(multimodal_input=True, function_tools=True, native_web_search=True, server_state=False)

    def __init__(self, config: OpenAIConfig) -> None:
        if not config.api_key:
            raise RuntimeError('OPENAI_API_KEY not set')
        self.config = config
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            headers={'Authorization': f'Bearer {config.api_key}', 'Content-Type': 'application/json'},
            timeout=httpx.Timeout(config.request_timeout_s, connect=config.connect_timeout_s),
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def describe_controls(self, settings: SessionSettings) -> dict[str, ControlDescriptor]:
        native_web = settings.native_web_search_mode if settings.native_web_search_mode != 'default' else ('on' if self.config.enable_native_web_search else 'off')
        reasoning_summary = effective_reasoning_summary(settings.reasoning_summary or self.config.reasoning_summary, provider='openai', default='off')
        return {
            'reasoning_effort': ControlDescriptor(True, settings.reasoning_effort or self.config.reasoning_effort, 'session' if settings.reasoning_effort else 'default'),
            'reasoning_summary': ControlDescriptor(True, reasoning_summary, 'session' if settings.reasoning_summary else 'default', 'OpenAI Responses reasoning.summary; "on" is normalized to "auto".'),
            'text_verbosity': ControlDescriptor(True, settings.text_verbosity or self.config.text_verbosity, 'session' if settings.text_verbosity else 'default'),
            'native_web_search': ControlDescriptor(True, native_web, 'session' if settings.native_web_search_mode != 'default' else 'default', 'Only used outside pure chat mode.'),
            'max_output_tokens': ControlDescriptor(True, str(settings.max_output_tokens if settings.max_output_tokens is not None else self.config.max_output_tokens), 'session' if settings.max_output_tokens is not None else 'default'),
        }

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
        input_items = [item for message in messages for item in self._message_to_input_items(message)]
        if extra_input_items:
            input_items.extend(extra_input_items)

        tool_defs = [tool.openai_tool() for tool in tools]
        native_web_search_enabled = self.config.enable_native_web_search
        if settings.native_web_search_mode == 'on':
            native_web_search_enabled = True
        elif settings.native_web_search_mode == 'off':
            native_web_search_enabled = False
        if native_web_search_enabled and settings.mode != ChatMode.CHAT:
            tool_defs.append({'type': 'web_search', 'search_context_size': 'low', 'user_location': None})

        include: list[str] = ['reasoning.encrypted_content']
        if native_web_search_enabled and settings.mode != ChatMode.CHAT:
            include.append('web_search_call.action.sources')

        reasoning: dict[str, Any] = {
            'effort': settings.reasoning_effort or self.config.reasoning_effort,
        }
        reasoning_summary = effective_reasoning_summary(settings.reasoning_summary or self.config.reasoning_summary, provider='openai', default='off')
        if reasoning_summary != 'off':
            reasoning['summary'] = reasoning_summary

        text_config: dict[str, Any] = {'verbosity': settings.text_verbosity or self.config.text_verbosity}
        if response_schema:
            text_config['format'] = {
                'type': 'json_schema',
                'name': response_schema_name or str(response_schema.get('title') or 'structured_output'),
                'schema': response_schema,
                'strict': True,
            }

        payload: dict[str, Any] = {
            'model': settings.model,
            'instructions': instructions,
            'input': input_items,
            'store': False,
            'include': include,
            'reasoning': reasoning,
            'text': text_config,
            'tools': tool_defs,
            'tool_choice': 'auto' if tool_defs else 'none',
            'max_output_tokens': settings.max_output_tokens if settings.max_output_tokens is not None else self.config.max_output_tokens,
        }
        effective_native_web_search_max = effective_optional_disabled_int(
            settings.native_web_search_max,
            self.config.native_web_search_max,
            maximum=NATIVE_WEB_SEARCH_MAX_MAX,
        )
        if native_web_search_enabled and settings.mode != ChatMode.CHAT and effective_native_web_search_max is not None:
            # Responses API applies max_tool_calls only to built-in tools. In this adapter the
            # only built-in tool we register is web_search, so this env knob works as a simple
            # cap on native web searches without limiting function tool calls.
            payload['max_tool_calls'] = effective_native_web_search_max
        response = await self._client.post('responses', json=payload)
        dump_llm_exchange(provider=self.name, model=settings.model, url='responses', payload=payload, response=response)
        if response.is_error:
            logger.error("OpenAI Responses error %s: %s", response.status_code, response.text[:4000])
        response.raise_for_status()
        return self._parse_response(response.json())

    def make_tool_result_items(self, tool_call: ToolCall, tool_output: dict) -> list[dict]:
        return [{'type': 'function_call_output', 'call_id': tool_call.call_id, 'output': json.dumps(tool_output, ensure_ascii=False)}]

    def _parse_response(self, body: dict[str, Any]) -> ProviderResponse:
        output = body.get('output', [])
        tool_calls: list[ToolCall] = []
        native_tool_calls: list[dict[str, Any]] = []
        final_text_parts: list[str] = []
        reasoning_summaries: list[str] = []
        continuation_items: list[dict[str, Any]] = []
        for item in output:
            item_type = item.get('type')
            continuation_items.append(item)
            if item_type == 'function_call':
                args_raw = item.get('arguments', '{}')
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCall(name=item.get('name', ''), call_id=item.get('call_id') or item.get('id') or '', arguments=args))
            elif item_type == 'web_search_call':
                native_tool_calls.append({
                    'name': 'web_search',
                    'type': 'web_search_call',
                    'id': item.get('id'),
                    'status': item.get('status'),
                    'action': item.get('action') or {},
                    'raw_item': item,
                })
            elif item_type == "reasoning":
                for summary_item in item.get("summary") or []:
                    if summary_item.get("type") == "summary_text":
                        text = summary_item.get("text")
                        if text:
                            reasoning_summaries.append(text)
            elif item_type == 'message':
                for content in item.get('content', []):
                    if content.get('type') == 'output_text':
                        final_text_parts.append(content.get('text', ''))
        usage_block = body.get('usage', {}) or {}
        input_tokens = usage_block.get('input_tokens')
        output_tokens = usage_block.get('output_tokens')
        total_tokens = usage_block.get('total_tokens')
        if total_tokens is None and (input_tokens is not None or output_tokens is not None):
            total_tokens = (input_tokens or 0) + (output_tokens or 0)
        return ProviderResponse(
            final_text=''.join(final_text_parts).strip(),
            reasoning_summaries=reasoning_summaries,
            tool_calls=tool_calls,
            native_tool_calls=native_tool_calls,
            continuation_items=continuation_items,
            usage=UsageInfo(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens),
            raw=body,
        )

    def persistent_history_items(self, response: ProviderResponse) -> list[dict]:
        items = response.continuation_items or []
        sanitized: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            # process any sanitization here if any
            cleaned = item
            if cleaned is not None:
                sanitized.append(cleaned)
        return sanitized

    def _message_to_input_items(self, message: ConversationMessage) -> list[dict[str, Any]]:
        if isinstance(message.metadata, dict):
            provider_native = message.metadata.get('provider_native') if isinstance(message.metadata.get('provider_native'), dict) else None
            if provider_native and str(provider_native.get('provider') or '').strip().lower() == self.name:
                items = provider_native.get('items')
                if isinstance(items, list):
                    return [item for item in items if isinstance(item, dict)]
        if message.role == MessageRole.TOOL and isinstance(message.metadata, dict):
            phase = str(message.metadata.get('tool_phase') or '').strip().lower()
            provider_name = str(message.metadata.get('tool_provider') or '').strip().lower()
            payload = message.metadata.get('tool_payload') if isinstance(message.metadata.get('tool_payload'), dict) else {}
            if provider_name == self.name and phase == 'call' and payload.get('call_id') and message.name:
                return [{'type': 'function_call', 'call_id': str(payload['call_id']), 'name': message.name, 'arguments': json.dumps(payload.get('arguments') or {}, ensure_ascii=False)}]
            if provider_name == self.name and phase == 'result' and payload.get('call_id'):
                return [{'type': 'function_call_output', 'call_id': str(payload['call_id']), 'output': json.dumps(payload.get('output') or {}, ensure_ascii=False)}]
        role = MessageRole.ASSISTANT.value if message.role == MessageRole.TOOL else message.role.value
        text_item_type = 'output_text' if role == MessageRole.ASSISTANT.value else 'input_text'
        content_items: list[dict[str, Any]] = []
        for part in message.parts:
            if part.kind == PartKind.TEXT and part.text:
                content_items.append({'type': text_item_type, 'text': part.text})
            elif part.kind == PartKind.IMAGE:
                encoded = part.data_b64
                if encoded and part.mime_type:
                    content_items.append({'type': 'input_image', 'image_url': f'data:{part.mime_type};base64,{encoded}', 'detail': part.detail or 'auto'})
            elif part.kind == PartKind.FILE:
                descriptor = f"[Attached file: {part.filename or 'file'}"
                if part.mime_type:
                    descriptor += f', {part.mime_type}'
                if part.size_bytes is not None:
                    descriptor += f', {part.size_bytes} bytes'
                if part.artifact_path:
                    descriptor += f', remote_path={part.artifact_path}'
                descriptor += ']'
                content_items.append({'type': text_item_type, 'text': descriptor})
            elif part.kind == PartKind.STICKER:
                content_items.append({'type': text_item_type, 'text': part.text or f"[Sticker: {part.filename or 'sticker'}]"})
        if not content_items:
            return []
        return [{'role': role, 'content': content_items}]
