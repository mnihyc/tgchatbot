from __future__ import annotations

import copy
import logging
from typing import Any

import httpx

from tgchatbot.config import GeminiConfig
from tgchatbot.domain.models import ConversationMessage, MessageRole, PartKind, ProviderResponse, SessionSettings, ToolCall, UsageInfo
from tgchatbot.providers.base import ControlDescriptor, ProviderCapabilities
from tgchatbot.settings_schema import (
    GEMINI_THINKING_BUDGET_MAX,
    GEMINI_THINKING_BUDGET_MIN,
    gemini_allowed_thinking_levels,
    gemini_supports_native_web_search,
    gemini_supports_thinking,
    gemini_supports_tool_combination,
    normalize_optional_bounded_int,
)
from tgchatbot.tools.base import ToolSpec
from tgchatbot.logging_config import dump_llm_exchange

logger = logging.getLogger(__name__)


class GeminiProvider:
    name = 'gemini'
    capabilities = ProviderCapabilities(multimodal_input=True, function_tools=True, native_web_search=True)

    def __init__(self, config: GeminiConfig) -> None:
        self.config = config
        self._client: httpx.AsyncClient | None = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if not self.config.api_key:
            raise RuntimeError('GEMINI_API_KEY not set')
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.request_timeout_s, connect=self.config.connect_timeout_s),
                headers={'x-goog-api-key': self.config.api_key},
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def describe_controls(self, settings: SessionSettings) -> dict[str, ControlDescriptor]:
        include_thoughts = 'on' if self._include_thoughts_enabled(settings) else 'off'
        thinking_budget = self._thinking_budget_for_request(settings)
        thinking_level = self._thinking_level_for_request(settings)
        native_web = settings.native_web_search_mode if settings.native_web_search_mode != 'default' else ('on' if self.config.enable_native_web_search else 'off')
        native_web_supported = gemini_supports_native_web_search(settings.model)
        native_web_note = 'Uses Gemini google_search grounding.'
        if native_web_supported and gemini_supports_tool_combination(settings.model):
            native_web_note += ' Gemini 3 can combine it with custom function tools when includeServerSideToolInvocations=true.'
        elif native_web_supported:
            native_web_note += ' On pre-Gemini-3 models it is only sent when no custom function tools are attached in the same request.'
        else:
            native_web_note = 'This model family does not advertise Gemini google_search support in this adapter.'
        thinking_level_values = gemini_allowed_thinking_levels(settings.model)
        thinking_level_note = None
        if thinking_level_values:
            thinking_level_note = (
                'Gemini thinkingConfig.thinkingLevel. '
                f"Allowed values for this model: {', '.join(thinking_level_values)}."
            )
        else:
            thinking_level_note = 'This model family does not support thinkingConfig.thinkingLevel.'
        thinking_budget_note = self._thinking_budget_note(settings.model)
        if settings.model.startswith('gemini-3'):
            thinking_budget_note += ' On Gemini 3 it is a legacy fallback and is ignored when thinking_level is set.'
        return {
            'include_thoughts': ControlDescriptor(
                gemini_supports_thinking(settings.model),
                include_thoughts,
                'session' if settings.include_thoughts is not None else 'default',
                'Gemini thinkingConfig.includeThoughts; if false or unset, thought text is not requested.',
            ),
            'thinking_budget': ControlDescriptor(
                gemini_supports_thinking(settings.model),
                str(thinking_budget) if thinking_budget is not None else 'default',
                'session' if settings.thinking_budget is not None else 'default',
                thinking_budget_note,
            ),
            'thinking_level': ControlDescriptor(
                bool(thinking_level_values),
                thinking_level or 'default',
                'session' if settings.thinking_level is not None else 'default',
                thinking_level_note,
            ),
            'native_web_search': ControlDescriptor(
                native_web_supported,
                native_web,
                'session' if settings.native_web_search_mode != 'default' else 'default',
                native_web_note,
            ),
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
        contents = [item for message in messages for item in self._message_to_contents(message)]
        if extra_input_items:
            contents.extend(item for item in extra_input_items if isinstance(item, dict))
        contents = self._prepare_contents_for_request(contents)

        generation_config: dict[str, Any] = {
            'temperature': settings.temperature if settings.temperature is not None else self.config.temperature,
            'topP': settings.top_p if settings.top_p is not None else self.config.top_p,
            'topK': settings.top_k if settings.top_k is not None else self.config.top_k,
            'maxOutputTokens': settings.max_output_tokens if settings.max_output_tokens is not None else self.config.max_output_tokens,
        }

        thinking = self._thinking_config_for_model(settings)
        if thinking:
            generation_config['thinkingConfig'] = thinking
        if response_schema:
            generation_config['responseMimeType'] = 'application/json'
            generation_config['responseJsonSchema'] = response_schema

        effective_native_web = self._native_web_search_enabled(settings)
        can_combine_native_and_custom_tools = gemini_supports_tool_combination(settings.model)
        tool_declarations = [tool.gemini_function_declaration() for tool in tools]
        request_tools: list[dict[str, Any]] = []
        server_side_tool_enabled = effective_native_web and (not tool_declarations or can_combine_native_and_custom_tools)
        if server_side_tool_enabled:
            request_tools.append({'googleSearch': {}})
        if tool_declarations:
            request_tools.append({'functionDeclarations': tool_declarations})

        tool_config: dict[str, Any] = {}
        if tool_declarations:
            mode = 'VALIDATED' if server_side_tool_enabled else 'AUTO'
            tool_config['functionCallingConfig'] = {'mode': mode}
        if server_side_tool_enabled:
            tool_config['includeServerSideToolInvocations'] = True

        payload: dict[str, Any] = {
            'systemInstruction': {'parts': [{'text': instructions}]},
            'contents': contents,
            'generationConfig': generation_config,
        }
        if request_tools:
            payload['tools'] = request_tools
        if tool_config:
            payload['toolConfig'] = tool_config

        client = self._ensure_client()
        response = await client.post(
            f"{self.config.base_url}/models/{settings.model}:generateContent",
            json=payload,
        )
        dump_llm_exchange(
            provider=self.name,
            model=settings.model,
            url=f"{self.config.base_url}/models/{settings.model}:generateContent",
            payload=payload,
            response=response,
        )
        if response.is_error:
            logger.error('Gemini generateContent error %s: %s', response.status_code, response.text[:4000])
        response.raise_for_status()
        return self._parse_response(response.json())

    def make_tool_result_items(self, tool_call: ToolCall, tool_output: dict) -> list[dict]:
        return [{
            'role': 'user',
            'parts': [{
                'functionResponse': {
                    'name': tool_call.name,
                    'id': tool_call.call_id,
                    'response': {'result': tool_output},
                }
            }],
        }]

    def _thinking_config_for_model(self, settings: SessionSettings) -> dict[str, Any] | None:
        if not gemini_supports_thinking(settings.model):
            return None
        config: dict[str, Any] = {}
        if self._include_thoughts_enabled(settings):
            config['includeThoughts'] = True
        thinking_level = self._thinking_level_for_request(settings)
        thinking_budget = self._thinking_budget_for_request(settings)
        if settings.model.startswith('gemini-2.5'):
            if thinking_budget is not None:
                config['thinkingBudget'] = thinking_budget
            return config or None
        if settings.model.startswith('gemini-3'):
            if thinking_level is not None:
                config['thinkingLevel'] = thinking_level
            elif thinking_budget is not None:
                config['thinkingBudget'] = thinking_budget
            return config or None
        return config or None

    def _native_web_search_enabled(self, settings: SessionSettings) -> bool:
        if settings.native_web_search_mode == 'on':
            return gemini_supports_native_web_search(settings.model)
        if settings.native_web_search_mode == 'off':
            return False
        return self.config.enable_native_web_search and gemini_supports_native_web_search(settings.model)

    def _include_thoughts_enabled(self, settings: SessionSettings) -> bool:
        if settings.include_thoughts is not None:
            return bool(settings.include_thoughts)
        return bool(self.config.include_thoughts)

    def _effective_thinking_budget(self, settings: SessionSettings) -> int | None:
        value = settings.thinking_budget if settings.thinking_budget is not None else self.config.thinking_budget
        return normalize_optional_bounded_int(value, minimum=GEMINI_THINKING_BUDGET_MIN, maximum=GEMINI_THINKING_BUDGET_MAX)

    def _effective_thinking_level(self, settings: SessionSettings) -> str | None:
        raw = settings.thinking_level if settings.thinking_level is not None else self.config.thinking_level
        if raw is None:
            return None
        normalized = raw.strip().lower()
        return normalized if normalized in gemini_allowed_thinking_levels(settings.model) else None

    def _thinking_budget_for_request(self, settings: SessionSettings) -> int | None:
        value = self._effective_thinking_budget(settings)
        if value is None:
            return None
        if settings.model.startswith('gemini-2.5'):
            if 'flash-lite' in settings.model:
                return value if value in {-1, 0} or 512 <= value <= 24576 else None
            if 'pro' in settings.model:
                return value if value == -1 or 128 <= value <= 32768 else None
            return value if value == -1 or 0 <= value <= 24576 else None
        if settings.model.startswith('gemini-3'):
            return value
        return None

    def _thinking_level_for_request(self, settings: SessionSettings) -> str | None:
        if not settings.model.startswith('gemini-3'):
            return None
        return self._effective_thinking_level(settings)

    @staticmethod
    def _thinking_budget_note(model: str) -> str:
        normalized = (model or '').strip().lower()
        if normalized.startswith('gemini-2.5'):
            if 'flash-lite' in normalized:
                return 'Gemini 2.5 Flash-Lite accepts thinkingBudget=-1, 0, or 512..24576.'
            if 'pro' in normalized:
                return 'Gemini 2.5 Pro accepts thinkingBudget=-1 or 128..32768. It cannot fully disable thinking with 0.'
            return 'Gemini 2.5 Flash accepts thinkingBudget=-1 or 0..24576.'
        if normalized.startswith('gemini-3'):
            return 'Gemini 3 prefers thinkingLevel over the legacy thinkingBudget field.'
        return 'This model family does not support Gemini thinking controls.'

    @staticmethod
    def _prepare_contents_for_request(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [copy.deepcopy(item) for item in contents if isinstance(item, dict)]

    def _parse_response(self, body: dict[str, Any]) -> ProviderResponse:
        candidates = body.get('candidates', [])
        if not candidates:
            return ProviderResponse(raw=body)
        candidate = candidates[0] or {}
        content = candidate.get('content', {}) or {}
        parts = content.get('parts', []) or []
        text_parts: list[str] = []
        reasoning_summaries: list[str] = []
        tool_calls: list[ToolCall] = []
        native_tool_calls: list[dict[str, Any]] = []
        continuation_items: list[dict[str, Any]] = [content] if parts else []
        for index, part in enumerate(parts):
            if part.get('thought') is True:
                text = part.get('text')
                if text:
                    reasoning_summaries.append(text)
                continue
            if 'text' in part:
                text_parts.append(part.get('text', ''))
            function_call = part.get('functionCall')
            if function_call:
                args = function_call.get('args') or {}
                tool_calls.append(ToolCall(name=function_call.get('name', ''), call_id=function_call.get('id') or f'gemini-call-{index}', arguments=args))
            tool_call = part.get('toolCall')
            if tool_call:
                native_tool_calls.append({
                    'name': str(tool_call.get('toolType') or 'google_search').lower(),
                    'type': 'tool_call',
                    'id': tool_call.get('id'),
                    'action': tool_call.get('args') or {},
                    'raw_item': part,
                })
            tool_response = part.get('toolResponse')
            if tool_response:
                native_tool_calls.append({
                    'name': str(tool_response.get('toolType') or 'google_search').lower(),
                    'type': 'tool_response',
                    'id': tool_response.get('id'),
                    'status': 'completed',
                    'action': tool_response.get('response') or {},
                    'raw_item': part,
                })

        grounding = candidate.get('groundingMetadata') or {}
        if grounding:
            native_tool_calls.append({
                'name': 'google_search',
                'type': 'google_search',
                'status': 'completed',
                'action': {
                    'queries': grounding.get('webSearchQueries') or [],
                    'chunks': grounding.get('groundingChunks') or [],
                    'supports': grounding.get('groundingSupports') or [],
                },
                'raw_item': grounding,
            })

        usage = body.get('usageMetadata', {}) or {}
        return ProviderResponse(
            final_text=''.join(text_parts).strip(),
            reasoning_summaries=reasoning_summaries,
            tool_calls=tool_calls,
            native_tool_calls=native_tool_calls,
            continuation_items=continuation_items,
            usage=UsageInfo(
                input_tokens=usage.get('promptTokenCount'),
                output_tokens=usage.get('candidatesTokenCount'),
                total_tokens=usage.get('totalTokenCount'),
            ),
            raw=body,
        )

    def persistent_history_items(self, response: ProviderResponse) -> list[dict]:
        items = response.continuation_items or []
        sanitized: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            cleaned = self._sanitize_history_content(item)
            if cleaned is not None:
                sanitized.append(cleaned)
        return sanitized

    @staticmethod
    def _sanitize_history_content(item: dict[str, Any]) -> dict[str, Any] | None:
        return copy.deepcopy(item)

    def _message_to_contents(self, message: ConversationMessage) -> list[dict[str, Any]]:
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
                return [{'role': 'model', 'parts': [{'functionCall': {'name': message.name, 'args': payload.get('arguments') or {}, 'id': str(payload['call_id'])}}]}]
            if provider_name == self.name and phase == 'result' and message.name:
                response_payload = {'name': message.name, 'response': {'result': payload.get('output') or {}}}
                if payload.get('call_id'):
                    response_payload['id'] = str(payload['call_id'])
                return [{'role': 'user', 'parts': [{'functionResponse': response_payload}]}]
        role = 'model' if message.role in {MessageRole.ASSISTANT, MessageRole.TOOL} else 'user'
        parts: list[dict[str, Any]] = []
        for part in message.parts:
            if part.kind == PartKind.TEXT and part.text:
                parts.append({'text': part.text})
            elif part.kind == PartKind.IMAGE:
                encoded = part.data_b64
                if encoded and part.mime_type:
                    parts.append({'inlineData': {'mimeType': part.mime_type, 'data': encoded}})
            elif part.kind == PartKind.FILE:
                descriptor = f"[Attached file: {part.filename or 'file'}"
                if part.mime_type:
                    descriptor += f', {part.mime_type}'
                if part.size_bytes is not None:
                    descriptor += f', {part.size_bytes} bytes'
                if part.artifact_path:
                    descriptor += f', remote_path={part.artifact_path}'
                descriptor += ']'
                parts.append({'text': descriptor})
            elif part.kind == PartKind.STICKER:
                parts.append({'text': part.text or f"[Sticker: {part.filename or 'sticker'}]"})
        if not parts:
            return []
        return [{'role': role, 'parts': parts}]
