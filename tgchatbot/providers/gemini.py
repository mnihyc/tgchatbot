from __future__ import annotations

import copy
import logging
from typing import Any

import httpx

from tgchatbot.config import GeminiConfig
from tgchatbot.core.token_estimator import TokenEstimator
from tgchatbot.domain.attachments import attachment_description
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind, ProviderResponse, SessionSettings, ToolCall, UsageInfo
from tgchatbot.providers.base import (ControlDescriptor, ProviderCapabilities, ProviderOutcomeError, RequestTokenEstimate,
    estimate_json_schema_tokens, evidence_text, pending_image_tokens, tool_message_evidence)
from tgchatbot.settings_schema import (
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
    inspection_format = 'gemini'
    name = 'gemini'
    capabilities = ProviderCapabilities(multimodal_input=True, function_tools=True, native_web_search=True, multimodal_tool_results=True)

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

    def supports_tool_evidence(self, settings: SessionSettings) -> bool:
        # Gemini 3 introduced nested function-response media. Unknown newer
        # model IDs use that dialect; do not create a fixed model allowlist.
        return not settings.model.removeprefix('models/').startswith(('gemini-1', 'gemini-2'))

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
            **{name: ControlDescriptor(True, str(getattr(settings, name) if getattr(settings, name) is not None else getattr(self.config, name)), 'session' if getattr(settings, name) is not None else 'default') for name in ('temperature', 'top_p', 'top_k')},
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

    def _build_request_tools(self, settings: SessionSettings, tools: list[ToolSpec]) -> tuple[list[dict[str, Any]], dict[str, Any], bool]:
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
            # Constrain function syntax while retaining the choice to answer in text.
            tool_config['functionCallingConfig'] = {'mode': 'VALIDATED'}
        if server_side_tool_enabled:
            tool_config['includeServerSideToolInvocations'] = True
        return request_tools, tool_config, server_side_tool_enabled

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
        history_tokens = int(history_tokens_override) if history_tokens_override is not None else sum(
            self._estimate_content_tokens(content)
            for message in messages
            for content in self._message_to_contents(message, tool_images=self.supports_tool_evidence(settings))
        )
        if history_tokens_override is None:
            history_tokens += pending_image_tokens(messages, tool_images=self.supports_tool_evidence(settings))
        instructions_tokens = TokenEstimator.estimate_text(instructions)
        request_tools, tool_config, server_side_tool_enabled = self._build_request_tools(settings, tools)
        tools_tokens = sum(self._estimate_request_tool_tokens(tool_entry) for tool_entry in request_tools)
        if tool_config:
            tools_tokens += self._estimate_semantic_value_tokens(tool_config)
        if response_schema:
            tools_tokens += estimate_json_schema_tokens(response_schema, name=response_schema_name or 'structured_output')
        extra_input_tokens = sum(
            self._estimate_content_tokens(item)
            for item in (extra_input_items or [])
            if isinstance(item, dict)
        )
        thinking = self._thinking_config_for_model(settings)
        framing_tokens = 40
        framing_tokens += TokenEstimator.estimate_text(str(settings.temperature if settings.temperature is not None else self.config.temperature))
        framing_tokens += TokenEstimator.estimate_text(str(settings.top_p if settings.top_p is not None else self.config.top_p))
        framing_tokens += TokenEstimator.estimate_text(str(settings.top_k if settings.top_k is not None else self.config.top_k))
        if thinking:
            framing_tokens += self._estimate_semantic_value_tokens(thinking)
        if server_side_tool_enabled:
            framing_tokens += 12
        return RequestTokenEstimate.compose(
            history_tokens=history_tokens,
            instructions_tokens=instructions_tokens,
            tools_tokens=tools_tokens,
            extra_input_tokens=extra_input_tokens,
            framing_tokens=framing_tokens,
        )

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
        contents = [item for message in messages for item in self._message_to_contents(message, tool_images=self.supports_tool_evidence(settings))]
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

        request_tools, tool_config, _server_side_tool_enabled = self._build_request_tools(settings, tools)

        payload: dict[str, Any] = {
            'systemInstruction': {'parts': [{'text': instructions}]},
            'contents': contents,
            'generationConfig': generation_config,
            'safetySettings': [
                {'category': category, 'threshold': 'OFF'}
                for category in ('HARM_CATEGORY_HARASSMENT', 'HARM_CATEGORY_HATE_SPEECH',
                                 'HARM_CATEGORY_SEXUALLY_EXPLICIT', 'HARM_CATEGORY_DANGEROUS_CONTENT')
            ],
        }
        if request_tools:
            payload['tools'] = request_tools
        if tool_config:
            payload['toolConfig'] = tool_config
        tier = settings.service_tier or getattr(self.config, 'service_tier', None)
        if tier:
            # Native REST field documented by Gemini's Flex inference guide.
            # A failed Flex request is never retried as Standard here.
            payload['service_tier'] = tier

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

    def make_tool_result_items(self, tool_call: ToolCall, tool_output: dict,
                               evidence_parts: list[MessagePart] | None = None) -> list[dict]:
        return self._tool_result_items(tool_call, tool_output, evidence_parts or [])

    def _tool_result_items(self, tool_call: ToolCall, tool_output: dict,
                           evidence_parts: list[MessagePart], *, allow_images: bool = True) -> list[dict]:
        response: dict[str, Any] = {'result': tool_output}
        media: list[dict[str, Any]] = []
        evidence: list[dict[str, Any]] = []
        for part in evidence_parts:
            if (allow_images and part.kind in {PartKind.IMAGE, PartKind.FILE} and part.data_b64
                    and part.mime_type in {'image/png', 'image/jpeg', 'image/webp', 'application/pdf', 'text/plain'}):
                media.append({'inlineData': {'mimeType': part.mime_type, 'data': part.data_b64}})
                if part.text:
                    evidence.append({'text': part.text})
                # The live Gemini API rejects the guide's optional displayName/
                # $ref form. Native inline parts work without those references.
                # Keep frame attribution explicit in the structured result.
                evidence.append({'image_part': len(media)})
            else:
                text = evidence_text(part)
                if text:
                    evidence.append({'text': text})
        if evidence:
            response['evidence'] = evidence
        function_response: dict[str, Any] = {
            'name': tool_call.name, 'id': tool_call.call_id, 'response': response,
        }
        if media:
            function_response['parts'] = media
            response['media_order'] = 'image_part is the 1-based position in functionResponse.parts; labels describe the following image part.'
        return [{
            'role': 'user',
            'parts': [{'functionResponse': function_response}],
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
        return normalize_optional_bounded_int(value, minimum=GEMINI_THINKING_BUDGET_MIN)

    def _effective_thinking_level(self, settings: SessionSettings) -> str | None:
        raw = settings.thinking_level if settings.thinking_level is not None else self.config.thinking_level
        if raw is None:
            return None
        normalized = raw.strip().lower()
        return normalized if normalized in gemini_allowed_thinking_levels(settings.model) else None

    def _thinking_budget_for_request(self, settings: SessionSettings) -> int | None:
        return self._effective_thinking_budget(settings) if gemini_supports_thinking(settings.model) else None

    def _thinking_level_for_request(self, settings: SessionSettings) -> str | None:
        if not settings.model.startswith('gemini-3'):
            return None
        return self._effective_thinking_level(settings)

    @staticmethod
    def _thinking_budget_note(model: str) -> str:
        if gemini_supports_thinking(model):
            return 'Thinking-token allowance: -1 is automatic; 0 requests disabled thinking. The model API validates its capacity.'
        return 'This model family does not support Gemini thinking controls.'

    @staticmethod
    def _prepare_contents_for_request(contents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for original in contents:
            if not isinstance(original, dict):
                continue
            item = copy.deepcopy(original)
            if (result and item.get('role') == result[-1].get('role') == 'model'
                    and all('functionCall' in part for part in item.get('parts', []))
                    and all('functionCall' in part for part in result[-1].get('parts', []))):
                result[-1]['parts'].extend(item.get('parts', []))
            else:
                result.append(item)
        return result

    def _parse_response(self, body: dict[str, Any]) -> ProviderResponse:
        usage = body.get('usageMetadata') or {}
        usage_info = UsageInfo(
            input_tokens=usage.get('promptTokenCount'),
            output_tokens=usage.get('candidatesTokenCount'),
            total_tokens=usage.get('totalTokenCount'),
            cached_input_tokens=usage.get('cachedContentTokenCount'),
            service_tier=usage.get('serviceTier'),
        )
        if body.get('error'):
            raise ProviderOutcomeError(self.name, 'API error', usage_info)
        feedback = body.get('promptFeedback') or {}
        prompt_block = feedback.get('blockReason')
        if prompt_block and prompt_block != 'BLOCK_REASON_UNSPECIFIED':
            raise ProviderOutcomeError(self.name, f'prompt blocked: {prompt_block}', usage_info)
        if any(rating.get('blocked') for rating in feedback.get('safetyRatings') or []):
            raise ProviderOutcomeError(self.name, 'prompt blocked', usage_info)
        candidates = body.get('candidates', [])
        if not candidates:
            raise ProviderOutcomeError(self.name, 'no completion candidate', usage_info)
        candidate = candidates[0] or {}
        finish_reason = candidate.get('finishReason')
        if finish_reason != 'STOP':
            raise ProviderOutcomeError(self.name, f'finish reason: {finish_reason or "missing"}', usage_info)
        if any(rating.get('blocked') for rating in candidate.get('safetyRatings') or []):
            raise ProviderOutcomeError(self.name, 'candidate blocked', usage_info)
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
            if 'functionCall' in part:
                function_call = part['functionCall']
                if not isinstance(function_call, dict):
                    raise ProviderOutcomeError(self.name, 'function call must be an object', usage_info)
                args = function_call.get('args', {})
                if not isinstance(args, dict):
                    raise ProviderOutcomeError(self.name, 'function call arguments must be an object', usage_info)
                name = function_call.get('name')
                if not isinstance(name, str) or not name.strip():
                    raise ProviderOutcomeError(self.name, 'function call requires a name', usage_info)
                tool_calls.append(ToolCall(name=name, call_id=function_call.get('id') or f'gemini-call-{index}', arguments=args))
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

        return ProviderResponse(
            final_text=''.join(text_parts).strip(),
            reasoning_summaries=reasoning_summaries,
            tool_calls=tool_calls,
            native_tool_calls=native_tool_calls,
            continuation_items=continuation_items,
            usage=usage_info,
            raw=body,
        )

    def _estimate_request_tool_tokens(self, tool_entry: dict[str, Any]) -> int:
        if not isinstance(tool_entry, dict):
            return 0
        if 'googleSearch' in tool_entry:
            return 20
        declarations = tool_entry.get('functionDeclarations')
        if isinstance(declarations, list):
            return 16 + sum(self._estimate_function_declaration_tokens(declaration) for declaration in declarations if isinstance(declaration, dict))
        return 8 + self._estimate_semantic_value_tokens(tool_entry)

    def _estimate_function_declaration_tokens(self, declaration: dict[str, Any]) -> int:
        return (
            20
            + TokenEstimator.estimate_text(declaration.get('name'))
            + TokenEstimator.estimate_text(declaration.get('description'))
            + estimate_json_schema_tokens(declaration.get('parameters') if isinstance(declaration.get('parameters'), dict) else None)
        )

    def _estimate_content_tokens(self, content: dict[str, Any]) -> int:
        if not isinstance(content, dict):
            return 0
        total = 8 + TokenEstimator.estimate_text(content.get('role'))
        for part in content.get('parts', []) or []:
            if not isinstance(part, dict):
                continue
            total += self._estimate_part_tokens(part)
        return total

    def _estimate_part_tokens(self, part: dict[str, Any]) -> int:
        if 'text' in part:
            return 4 + TokenEstimator.estimate_text(part.get('text'))
        if 'inlineData' in part and isinstance(part.get('inlineData'), dict):
            inline = part['inlineData']
            return TokenEstimator.IMAGE_TOKENS + 8 + TokenEstimator.estimate_text(inline.get('mimeType'))
        if 'functionCall' in part and isinstance(part.get('functionCall'), dict):
            call = part['functionCall']
            return 24 + TokenEstimator.estimate_text(call.get('name')) + TokenEstimator.estimate_text(call.get('id')) + self._estimate_semantic_value_tokens(call.get('args') or {})
        if 'functionResponse' in part and isinstance(part.get('functionResponse'), dict):
            response = part['functionResponse']
            return (24 + TokenEstimator.estimate_text(response.get('name')) + TokenEstimator.estimate_text(response.get('id'))
                    + self._estimate_semantic_value_tokens(response.get('response') or {})
                    + sum(self._estimate_part_tokens(item) for item in response.get('parts', []) if isinstance(item, dict)))
        if part.get('thought') is True:
            total = 12 + TokenEstimator.estimate_text(part.get('text'))
            if part.get('thoughtSignature'):
                total += min(TokenEstimator.estimate_text(part.get('thoughtSignature')), 128)
            return total
        if 'toolCall' in part or 'toolResponse' in part:
            return 20 + self._estimate_semantic_value_tokens(part)
        return 4 + self._estimate_semantic_value_tokens(part)

    def _estimate_semantic_value_tokens(self, value: Any) -> int:
        if value is None:
            return 0
        if isinstance(value, str):
            return TokenEstimator.estimate_text(value)
        if isinstance(value, (int, float, bool)):
            return TokenEstimator.estimate_text(str(value))
        if isinstance(value, list):
            return 2 + sum(self._estimate_semantic_value_tokens(item) for item in value)
        if isinstance(value, dict):
            total = 4
            for key, item in value.items():
                key_text = str(key)
                if key_text == 'data':
                    total += TokenEstimator.IMAGE_TOKENS
                    continue
                total += TokenEstimator.estimate_text(key_text)
                total += self._estimate_semantic_value_tokens(item)
            return total
        return TokenEstimator.estimate_text(str(value))

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

    def _message_to_contents(self, message: ConversationMessage, *, tool_images: bool = True) -> list[dict[str, Any]]:
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
            framework_refresh = message.metadata.get('synthetic_role') == 'profile_refresh'
            portable_evidence = bool(message.metadata.get('tool_evidence') or message.metadata.get('portable_tool_history'))
            if (provider_name == self.name or framework_refresh or portable_evidence) and phase == 'call' and payload.get('call_id') and message.name:
                part = {'functionCall': {'name': message.name, 'args': payload.get('arguments') or {}, 'id': str(payload['call_id'])}}
                if framework_refresh or portable_evidence:
                    # Google's documented marker for deterministic client-executed
                    # tool history; this is not a model-generated thought signature.
                    # https://ai.google.dev/gemini-api/docs/generate-content/thought-signatures
                    part['thoughtSignature'] = 'skip_thought_signature_validator'
                return [{'role': 'model', 'parts': [part]}]
            if (provider_name == self.name or framework_refresh or portable_evidence) and phase == 'result' and message.name:
                return self._tool_result_items(ToolCall(message.name, str(payload.get('call_id') or ''), {}),
                    payload.get('output') or {}, tool_message_evidence(message, images=tool_images), allow_images=tool_images)
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
                descriptor = attachment_description(part,
                    presentation_version=int(message.metadata.get('presentation_version', 1)))
                parts.append({'text': descriptor})
            elif part.kind == PartKind.STICKER:
                parts.append({'text': part.text or '[Sticker]'})
        if not parts:
            return []
        return [{'role': role, 'parts': parts}]
