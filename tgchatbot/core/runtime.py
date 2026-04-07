from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import Any

from tgchatbot.config import AppConfig
from tgchatbot.core.context_state import LiveConversationState, MemoryBlock, StoredConversationMessage
from tgchatbot.core.events import RuntimeEvent
from tgchatbot.core.policy import policy_for_mode
from tgchatbot.core.compaction_schema import compaction_json_schema, compaction_schema_name, parse_structured_candidate
from tgchatbot.core.prompting import build_compaction_prompt, build_system_prompt
from tgchatbot.core.token_estimator import TokenEstimator
from tgchatbot.domain.models import (
    ChatMode,
    ConversationMessage,
    MessagePart,
    MessageRole,
    OutboundSticker,
    PartKind,
    ProcessVisibility,
    ResponseDelivery,
    SessionSettings,
    StickerMode,
    ToolHistoryMode,
    StickerTiming,
    TurnResult,
)
from tgchatbot.logging_config import clip_for_log
from tgchatbot.providers.base import ModelProvider, RequestTokenEstimate
from tgchatbot.settings_schema import (
    COMPACT_KEEP_RECENT_RATIO_MAX,
    COMPACT_KEEP_RECENT_RATIO_MIN,
    COMPACT_MIN_MESSAGES_MIN,
    COMPACT_TOKEN_MIN,
    COMPACT_TOOL_RATIO_THRESHOLD_MIN,
    GROUP_SPONTANEOUS_REPLY_DELAY_MAX_S,
    IMAGE_LIMIT_MAX,
    MAX_INTERACTION_ROUNDS_MAX,
    MAX_INTERACTION_ROUNDS_MIN,
    NATIVE_WEB_SEARCH_MAX_MAX,
    PROVIDER_RETRY_COUNT_MAX,
    PROVIDER_RETRY_COUNT_MIN,
    REPLY_DELAY_MAX_S,
    SPONTANEOUS_REPLY_CHANCE_MAX,
    SPONTANEOUS_REPLY_CHANCE_MIN,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    TOP_K_MAX,
    TOP_K_MIN,
    TOP_P_MAX,
    TOP_P_MIN,
    clamp_float,
    clamp_int,
    effective_optional_disabled_int,
    format_optional_disabled_int,
)
from tgchatbot.storage.sqlite_store import SQLiteStore
from tgchatbot.tools.base import ToolContext, ToolSpec
from tgchatbot.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)
EventCallback = Callable[[RuntimeEvent], Awaitable[None]]


class CompactionModelRequestFailed(RuntimeError):
    def __init__(self, *, provider_name: str, mode: str) -> None:
        super().__init__(f'Compaction model request failed for provider={provider_name} mode={mode}')
        self.provider_name = provider_name
        self.mode = mode


class AgentRuntime:
    def __init__(
        self,
        *,
        config: AppConfig,
        store: SQLiteStore,
        tool_registry: ToolRegistry,
        providers: dict[str, ModelProvider],
    ) -> None:
        self.config = config
        self.store = store
        self.tool_registry = tool_registry
        self.providers = providers
        self._live_sessions: dict[str, LiveConversationState] = {}
        self._request_estimate_bias: dict[tuple[str, str, str], float] = {}

    @staticmethod
    def _session_log_id(session_id: str) -> str:
        return clip_for_log(session_id, limit=48)

    @staticmethod
    def _usage_log_text(usage: Any) -> str:
        if usage is None:
            return '-'
        return f"in={getattr(usage, 'input_tokens', None)} out={getattr(usage, 'output_tokens', None)} total={getattr(usage, 'total_tokens', None)}"

    @staticmethod
    def _message_text_preview(message: ConversationMessage, *, limit: int = 120) -> str:
        text_parts = [part.text.strip() for part in message.parts if part.kind == PartKind.TEXT and part.text and part.text.strip()]
        return clip_for_log(' | '.join(text_parts), limit=limit, rlimit=limit)

    @staticmethod
    def _message_part_counts(message: ConversationMessage) -> str:
        counts: dict[str, int] = {}
        for part in message.parts:
            counts[part.kind.value] = counts.get(part.kind.value, 0) + 1
        return ','.join(f'{kind}:{counts[kind]}' for kind in sorted(counts)) or 'none'

    @staticmethod
    def _tool_argument_summary(arguments: dict[str, Any]) -> str:
        keys = sorted(str(key) for key in arguments.keys())
        return ','.join(keys[:8]) + (',...' if len(keys) > 8 else '') if keys else '-'

    def _require_provider(self, provider_name: str) -> ModelProvider:
        provider = self.providers.get(provider_name)
        if provider is None:
            available = ', '.join(sorted(self.providers.keys())) or '-'
            raise RuntimeError(f'Provider {provider_name!r} is not configured. Available providers: {available}')
        return provider


    async def ingest_user_message(self, *, session_id: str, incoming_message: ConversationMessage) -> StoredConversationMessage:
        state = await self._get_live_state(session_id)
        stored_incoming = await self.store.append_message(session_id, incoming_message)
        self._append_live_message(state, stored_incoming)
        logger.info(
            'msg.ingest sid=%s msg=%s role=%s parts=%s preview=%s',
            self._session_log_id(session_id),
            stored_incoming.db_id,
            incoming_message.role.value,
            self._message_part_counts(incoming_message),
            self._message_text_preview(incoming_message),
        )
        return stored_incoming

    async def record_tool_observation(self, *, session_id: str, name: str, payload: dict[str, Any], phase: str, summary_text: str | None = None, provider_name: str | None = None, metadata_update: dict[str, Any] | None = None) -> None:
        state = await self._get_live_state(session_id)
        summary = summary_text or self._tool_observation_summary(name=name, phase=phase, payload=payload)
        logger.info('tool.obs sid=%s name=%s phase=%s provider=%s payload=%s', self._session_log_id(session_id), name, phase, provider_name or '-', self._compact_json(payload, limit=220))
        metadata = {'tool_phase': phase, 'tool_payload': payload, 'tool_provider': provider_name}
        if metadata_update:
            metadata.update(metadata_update)
        message = ConversationMessage(
            role=MessageRole.TOOL,
            name=name,
            parts=[MessagePart(kind=PartKind.TEXT, text=summary, remote_sync=False, origin='tool')],
            metadata=metadata,
        )
        stored = await self.store.append_message(session_id, message, estimated_tokens=TokenEstimator.estimate_message(message))
        self._append_live_message(state, stored)

    async def record_auto_user_note(
        self,
        *,
        session_id: str,
        parts: list[MessagePart],
        metadata: dict[str, Any] | None = None,
    ) -> StoredConversationMessage:
        state = await self._get_live_state(session_id)

        note_metadata = dict(metadata or {})
        note_metadata.setdefault('synthetic_role', 'auto_user_note')

        prefix = '[Auto-generated bot message, do not reply.]'

        normalized_parts: list[MessagePart] = [
            MessagePart(
                kind=PartKind.TEXT,
                text=prefix,
                remote_sync=False,
                origin='auto_note',
            )
        ]

        for part in parts:
            if part.kind == PartKind.TEXT and not (part.text or '').strip():
                continue

            normalized_parts.append(
                MessagePart(
                    kind=part.kind,
                    text=part.text,
                    mime_type=part.mime_type,
                    filename=part.filename,
                    data_b64=part.data_b64,
                    artifact_path=part.artifact_path,
                    size_bytes=part.size_bytes,
                    detail=part.detail,
                    remote_sync=part.remote_sync,
                    origin='auto_note',
                )
            )

        message = ConversationMessage(
            role=MessageRole.USER,
            parts=normalized_parts,
            metadata=note_metadata,
        )
        stored = await self.store.append_message(session_id, message)
        self._append_live_message(state, stored)
        return stored

    async def record_assistant_text(self, *, session_id: str, text: str, metadata: dict[str, Any] | None = None) -> StoredConversationMessage:
        state = await self._get_live_state(session_id)
        message = ConversationMessage.assistant_text(text, metadata=metadata)
        stored = await self.store.append_message(session_id, message)
        self._append_live_message(state, stored)
        return stored

    async def run_turn(
        self,
        *,
        session_id: str,
        user_display_name: str,
        incoming_message: ConversationMessage,
        emit: EventCallback | None = None,
    ) -> TurnResult:
        stored = await self.ingest_user_message(session_id=session_id, incoming_message=incoming_message)
        return await self.run_turn_from_stored(
            session_id=session_id,
            user_display_name=user_display_name,
            trigger_message_id=stored.db_id,
            emit=emit,
        )

    async def run_turn_from_stored(
        self,
        *,
        session_id: str,
        user_display_name: str,
        trigger_message_id: int,
        emit: EventCallback | None = None,
    ) -> TurnResult:
        settings = await self.store.get_or_create_session(session_id, self.config.default_session_settings())
        provider = self._require_provider(settings.provider)
        state = await self._get_live_state(session_id)

        policy = policy_for_mode(settings.mode)
        max_tool_rounds = self._effective_max_interaction_rounds(settings)

        instructions = build_system_prompt(settings)
        tools = (
            self.tool_registry.list_tools(
                allow_python_exec=policy.allow_python_exec,
                allow_stickers=(settings.sticker_mode == StickerMode.AUTO),
            )
            if policy.allow_tools
            else []
        )
        latest_preview = self._message_text_preview(state.raw_messages[-1].message) if state.raw_messages else ''
        est_ctx_tokens = self._estimate_request_breakdown(
            state=state,
            settings=settings,
            provider=provider,
            instructions='',
            tools=[],
        ).history_tokens
        est_req_tokens = self._estimate_request_breakdown(
            state=state,
            settings=settings,
            provider=provider,
            instructions=instructions,
            tools=tools,
        ).total_tokens
        logger.info(
            'turn.start sid=%s trigger=%s provider=%s/%s mode=%s rounds=%s raw=%s blocks=%s est_ctx=%s est_req=%s preview=%s',
            self._session_log_id(session_id),
            trigger_message_id,
            settings.provider,
            settings.model,
            settings.mode.value,
            max_tool_rounds,
            len(state.raw_messages),
            len(state.blocks),
            est_ctx_tokens,
            est_req_tokens,
            latest_preview,
        )
        await self._compact_if_needed(
            session_id=session_id,
            settings=settings,
            provider=provider,
            state=state,
            instructions=instructions,
            tools=tools,
            emit=emit,
        )

        history = self._build_provider_history(state, settings=settings, provider_name=provider.name)
        logger.debug('turn.history sid=%s provider=%s messages=%s cached=%s', self._session_log_id(session_id), provider.name, len(history), not state.provider_history_dirty)
        accumulated_items: list[dict[str, Any]] = []
        collected_artifacts = []
        collected_stickers: list[OutboundSticker] = []
        last_usage = None
        deferred_tool_texts: list[str] = []
        emitted_tool_text = False

        total_steps = max_tool_rounds + 1
        for iteration in range(total_steps):
            final_iteration = iteration == max_tool_rounds
            current_tools = [] if final_iteration else tools
            iteration_instructions = instructions
            if final_iteration and tools:
                # The last round intentionally disables tools. This keeps multi-step turns bounded and,
                # per the runtime audit notes, forces a best-effort final reply instead of letting the
                # model extend the same user turn forever with more tool calls.
                iteration_instructions = instructions + "\n\n[Internal control note]: this is the final interaction round for this turn. Keep the established personality and preset voice exactly as intended by the system prompt. This note is not a user message. Do not call any more tools. Use the conversation so far and give the best final reply now."
            raw_request_estimate = provider.estimate_request_tokens(
                settings=settings,
                messages=history,
                instructions=iteration_instructions,
                tools=current_tools,
                extra_input_items=accumulated_items or None,
                history_tokens_override=self._estimate_history_tokens_for_messages(
                    state=state,
                    provider=provider,
                    settings=settings,
                    selected_blocks=self._select_blocks_for_prompt(state, settings=settings),
                    messages=history,
                ),
            )
            adjusted_request_estimate = self._apply_request_estimate_bias(
                raw_request_estimate,
                provider_name=provider.name,
                model=settings.model,
                tool_history_mode=settings.tool_history_mode,
            )
            logger.info(
                'turn.iter sid=%s step=%s/%s final=%s tools=%s extra=%s est_req=%s raw_req=%s bias=%.3f',
                self._session_log_id(session_id),
                iteration + 1,
                total_steps,
                int(final_iteration),
                len(current_tools),
                len(accumulated_items),
                adjusted_request_estimate.total_tokens,
                raw_request_estimate.total_tokens,
                self._request_estimate_bias_for(provider_name=provider.name, model=settings.model, tool_history_mode=settings.tool_history_mode),
            )
            if emit and settings.process_visibility != ProcessVisibility.OFF:
                await emit(RuntimeEvent(kind='phase', title='Step', detail=f'iteration {iteration + 1}/{total_steps}'))

            response = await self._generate_with_retries(
                provider=provider,
                settings=settings,
                messages=history,
                instructions=iteration_instructions,
                tools=current_tools,
                extra_input_items=accumulated_items or None,
            )
            last_usage = response.usage
            self._update_request_estimate_bias(
                provider_name=provider.name,
                model=settings.model,
                tool_history_mode=settings.tool_history_mode,
                raw_estimate_total=raw_request_estimate.total_tokens,
                actual_input_tokens=getattr(last_usage, 'input_tokens', None),
            )
            logger.info('turn.model sid=%s step=%s/%s tool_calls=%s native_calls=%s continue=%s text_chars=%s usage=%s', self._session_log_id(session_id), iteration + 1, total_steps, len(response.tool_calls), len(getattr(response, 'native_tool_calls', []) or []), len(response.continuation_items), len(response.final_text or ''), self._usage_log_text(last_usage))
            persistent_history_items = provider.persistent_history_items(response) if hasattr(provider, 'persistent_history_items') else []

            if emit and settings.process_visibility in {ProcessVisibility.VERBOSE, ProcessVisibility.FULL}:
                summary = clip_for_log(' | '.join(getattr(response, "reasoning_summaries", [])), limit=220, rlimit=80)
                if summary:
                    logger.info('turn.reason sid=%s step=%s/%s token=%s preview=%s', self._session_log_id(session_id), iteration + 1, total_steps, TokenEstimator.estimate_text(summary), summary)
                    for summary in getattr(response, 'reasoning_summaries', [])[:2]:
                        await emit(RuntimeEvent(
                            kind='thinking',
                            title='Reasoning summary',
                            detail=summary,
                        ))

            for native_call in getattr(response, 'native_tool_calls', []):
                name = str(native_call.get('name') or native_call.get('type') or 'native_tool')
                action = native_call.get('action') or {}
                raw_item = native_call.get('raw_item') if isinstance(native_call.get('raw_item'), dict) else native_call
                logger.info('tool.native sid=%s name=%s phase=call provider=%s payload=%s', self._session_log_id(session_id), name, provider.name, self._compact_json(action, limit=220))
                logger.info('tool.native sid=%s name=%s phase=result provider=%s payload=%s', self._session_log_id(session_id), name, provider.name, self._compact_json(raw_item, limit=220))
                if emit and settings.process_visibility != ProcessVisibility.OFF:
                    await emit(
                        RuntimeEvent(
                            kind='tool_call',
                            title=f'{provider.name} {name}',
                            detail=self._compact_json(action or {'status': native_call.get('status')}, limit=220, rlimit=120),
                            payload=raw_item if isinstance(raw_item, dict) else native_call,
                        )
                    )
                    if settings.process_visibility in {ProcessVisibility.VERBOSE, ProcessVisibility.FULL}:
                        await emit(
                            RuntimeEvent(
                                kind='tool_result',
                                title=f'{provider.name} {name}',
                                detail=self._compact_json(raw_item, limit=220, rlimit=160),
                                payload=raw_item if isinstance(raw_item, dict) else native_call,
                            )
                        )

            if response.tool_calls:
                tool_turn_text = (response.final_text or self._provider_visible_text_from_items(persistent_history_items)).strip()
                if tool_turn_text:
                    logger.info(
                        'turn.tool_text sid=%s step=%s/%s chars=%s',
                        self._session_log_id(session_id),
                        iteration + 1,
                        total_steps,
                        len(tool_turn_text),
                    )
                    if emit is not None:
                        await emit(
                            RuntimeEvent(
                                kind='assistant_text',
                                title='Assistant',
                                detail=tool_turn_text,
                                payload={'text': tool_turn_text},
                            )
                        )
                        emitted_tool_text = True
                    else:
                        deferred_tool_texts.append(tool_turn_text)
                accumulated_items.extend(response.continuation_items)
                native_items = persistent_history_items
                for tool_index, tool_call in enumerate(response.tool_calls):
                    metadata_update = None
                    if native_items and tool_index == 0:
                        metadata_update = {'provider_native': {'provider': provider.name, 'items': native_items}}
                    elif native_items:
                        metadata_update = {'provider_native_skip_same_provider': True}
                    await self.record_tool_observation(
                        session_id=session_id,
                        name=tool_call.name,
                        phase='call',
                        payload={'call_id': tool_call.call_id, 'arguments': tool_call.arguments},
                        provider_name=provider.name,
                        metadata_update=metadata_update,
                    )
                    logger.info('tool.call sid=%s name=%s call=%s args=%s', self._session_log_id(session_id), tool_call.name, clip_for_log(tool_call.call_id, limit=32), self._tool_argument_summary(tool_call.arguments))
                    spec = self.tool_registry.get(tool_call.name)
                    if spec is None:
                        logger.warning('tool.unknown sid=%s name=%s', self._session_log_id(session_id), tool_call.name)
                        tool_output = {'ok': False, 'error': f'Unknown tool: {tool_call.name}'}
                    else:
                        if emit and settings.process_visibility != ProcessVisibility.OFF:
                            await emit(
                                RuntimeEvent(
                                    kind='tool_call',
                                    title=f'Executing {tool_call.name}',
                                    detail=self._compact_json(tool_call.arguments, limit=220, rlimit=80) or 'running tool',
                                    payload=tool_call.arguments,
                                )
                            )
                        result = await spec.runner.run(
                            tool_call.arguments,
                            ToolContext(
                                session_id=session_id,
                                user_display_name=user_display_name,
                            ),
                        )
                        result.call_id = tool_call.call_id
                        collected_artifacts.extend(result.artifacts)
                        tool_output = result.output
                        logger.info('tool.result sid=%s name=%s ok=%s artifacts=%s stickers=%s output=%s', self._session_log_id(session_id), tool_call.name, bool(result.output.get('ok')), len(result.artifacts), len(result.stickers), self._compact_json(result.output, limit=220))
                        if result.stickers:
                            collected_stickers.extend(result.stickers)
                            for sticker in result.stickers:
                                if emit and sticker.timing == StickerTiming.SEND_NOW:
                                    await emit(
                                        RuntimeEvent(
                                            kind='sticker',
                                            title='Sticker',
                                            detail='sending sticker',
                                            payload={
                                                'path': str(sticker.path),
                                                'emoji': sticker.emoji,
                                                'timing': sticker.timing.value,
                                                'label': sticker.label,
                                                'source_id': sticker.source_id,
                                            },
                                        )
                                    )
                    await self.record_tool_observation(
                        session_id=session_id,
                        name=tool_call.name,
                        phase='result',
                        payload={'call_id': tool_call.call_id, 'output': tool_output},
                        provider_name=provider.name,
                    )
                    if emit and settings.process_visibility in {ProcessVisibility.VERBOSE, ProcessVisibility.FULL}:
                        await emit(
                            RuntimeEvent(
                                kind='tool_result',
                                title=f'Tool {tool_call.name}',
                                detail=self._compact_json(tool_output, limit=220, rlimit=120),
                                payload=tool_output,
                            )
                        )
                    accumulated_items.extend(provider.make_tool_result_items(tool_call, tool_output))
                continue

            final_text = (response.final_text or self._provider_visible_text_from_items(persistent_history_items)).strip()
            if deferred_tool_texts:
                segments = [*deferred_tool_texts]
                if final_text and (not segments or segments[-1] != final_text):
                    segments.append(final_text)
                final_text = '\n\n'.join(segment for segment in segments if segment).strip()
            elif not final_text and emitted_tool_text:
                final_text = ''
            elif not final_text:
                final_text = '(empty response)'
            logger.info('turn.done sid=%s trigger=%s text_chars=%s artifacts=%s stickers=%s usage=%s', self._session_log_id(session_id), trigger_message_id, len(final_text), len(collected_artifacts), len(collected_stickers), self._usage_log_text(last_usage))
            if emit:
                await emit(RuntimeEvent(kind='final', title='Done', detail=f'trigger message #{trigger_message_id}'))
            return TurnResult(
                text=final_text,
                artifacts=collected_artifacts,
                usage=last_usage,
                stickers=collected_stickers,
                provider_name=provider.name,
                provider_history_items=persistent_history_items,
            )

        logger.warning('turn.limit sid=%s trigger=%s rounds=%s usage=%s', self._session_log_id(session_id), trigger_message_id, max_tool_rounds, self._usage_log_text(last_usage))
        raise RuntimeError('Interaction-round limit reached without a final response')

    async def describe_session(self, session_id: str) -> dict[str, Any]:
        state = await self._get_live_state(session_id)
        settings = await self.store.get_or_create_session(session_id, self.config.default_session_settings())
        provider = self._require_provider(settings.provider)
        l0_blocks = sum(1 for block in state.blocks if block.level == 0)
        l1_blocks = sum(1 for block in state.blocks if block.level == 1)
        l2_blocks = sum(1 for block in state.blocks if block.level == 2)
        tools = self.tool_registry.list_tools(
            allow_python_exec=policy_for_mode(settings.mode).allow_python_exec,
            allow_stickers=(settings.sticker_mode == StickerMode.AUTO),
        )
        instructions = build_system_prompt(settings)
        sticker_stats = self.tool_registry.sticker_catalog.stats()
        remote = self.tool_registry.remote_workspace
        controls = provider.describe_controls(settings) if hasattr(provider, 'describe_controls') else {}
        temperature = self._effective_temperature(settings)
        top_p = self._effective_top_p(settings)
        top_k = self._effective_top_k(settings)
        native_web_search_max = self._effective_native_web_search_max(settings)
        max_input_images = self._effective_max_input_images(provider, settings)
        compact_target_images = self._effective_compact_target_images(provider, settings)
        request_estimate = self._estimate_request_breakdown(
            state=state,
            settings=settings,
            provider=provider,
            instructions=instructions,
            tools=tools,
        )
        return {
            'provider': settings.provider,
            'model': settings.model,
            'mode': settings.mode.value,
            'process': settings.process_visibility.value,
            'delivery': settings.response_delivery.value,
            'stickers': settings.sticker_mode.value,
            'reasoning_effort': controls.get('reasoning_effort').effective_value if controls.get('reasoning_effort') else 'n/a',
            'reasoning_effort_source': controls.get('reasoning_effort').source if controls.get('reasoning_effort') else 'n/a',
            'reasoning_effort_supported': controls.get('reasoning_effort').supported if controls.get('reasoning_effort') else False,
            'reasoning_effort_note': controls.get('reasoning_effort').note if controls.get('reasoning_effort') else None,
            'reasoning_summary': controls.get('reasoning_summary').effective_value if controls.get('reasoning_summary') else 'n/a',
            'reasoning_summary_source': controls.get('reasoning_summary').source if controls.get('reasoning_summary') else 'n/a',
            'reasoning_summary_supported': controls.get('reasoning_summary').supported if controls.get('reasoning_summary') else False,
            'reasoning_summary_note': controls.get('reasoning_summary').note if controls.get('reasoning_summary') else None,
            'text_verbosity': controls.get('text_verbosity').effective_value if controls.get('text_verbosity') else 'n/a',
            'text_verbosity_source': controls.get('text_verbosity').source if controls.get('text_verbosity') else 'n/a',
            'text_verbosity_supported': controls.get('text_verbosity').supported if controls.get('text_verbosity') else False,
            'text_verbosity_note': controls.get('text_verbosity').note if controls.get('text_verbosity') else None,
            'include_thoughts': controls.get('include_thoughts').effective_value if controls.get('include_thoughts') else 'n/a',
            'include_thoughts_source': controls.get('include_thoughts').source if controls.get('include_thoughts') else 'n/a',
            'include_thoughts_supported': controls.get('include_thoughts').supported if controls.get('include_thoughts') else False,
            'include_thoughts_note': controls.get('include_thoughts').note if controls.get('include_thoughts') else None,
            'thinking_budget': controls.get('thinking_budget').effective_value if controls.get('thinking_budget') else 'n/a',
            'thinking_budget_source': controls.get('thinking_budget').source if controls.get('thinking_budget') else 'n/a',
            'thinking_budget_supported': controls.get('thinking_budget').supported if controls.get('thinking_budget') else False,
            'thinking_budget_note': controls.get('thinking_budget').note if controls.get('thinking_budget') else None,
            'thinking_level': controls.get('thinking_level').effective_value if controls.get('thinking_level') else 'n/a',
            'thinking_level_source': controls.get('thinking_level').source if controls.get('thinking_level') else 'n/a',
            'thinking_level_supported': controls.get('thinking_level').supported if controls.get('thinking_level') else False,
            'thinking_level_note': controls.get('thinking_level').note if controls.get('thinking_level') else None,
            'native_web_search': controls.get('native_web_search').effective_value if controls.get('native_web_search') else 'n/a',
            'native_web_search_source': controls.get('native_web_search').source if controls.get('native_web_search') else 'n/a',
            'native_web_search_supported': controls.get('native_web_search').supported if controls.get('native_web_search') else False,
            'native_web_search_note': controls.get('native_web_search').note if controls.get('native_web_search') else None,
            'max_output_tokens': controls.get('max_output_tokens').effective_value if controls.get('max_output_tokens') else 'n/a',
            'max_output_tokens_source': controls.get('max_output_tokens').source if controls.get('max_output_tokens') else 'n/a',
            'max_output_tokens_supported': controls.get('max_output_tokens').supported if controls.get('max_output_tokens') else False,
            'max_output_tokens_note': controls.get('max_output_tokens').note if controls.get('max_output_tokens') else None,
            'temperature': f"{temperature:g}",
            'temperature_source': 'session' if settings.temperature is not None else 'default',
            'temperature_supported': settings.provider == 'gemini',
            'temperature_note': 'Gemini generationConfig.temperature; retained across provider switches.',
            'top_p': f"{top_p:g}",
            'top_p_source': 'session' if settings.top_p is not None else 'default',
            'top_p_supported': settings.provider == 'gemini',
            'top_p_note': 'Gemini generationConfig.topP; retained across provider switches.',
            'top_k': str(top_k),
            'top_k_source': 'session' if settings.top_k is not None else 'default',
            'top_k_supported': settings.provider == 'gemini',
            'top_k_note': 'Gemini generationConfig.topK; retained across provider switches.',
            'native_web_search_max': format_optional_disabled_int(native_web_search_max, disabled_label='unlimited'),
            'native_web_search_max_source': 'session' if settings.native_web_search_max is not None else 'default',
            'native_web_search_max_supported': settings.provider == 'openai',
            'native_web_search_max_note': 'OpenAI-only cap for built-in web_search tool calls. 0 disables the explicit cap.',
            'prompt_injection_mode': settings.prompt_injection_mode.value,
            'tool_history_mode': settings.tool_history_mode.value,
            'link_prefetch_mode': settings.link_prefetch_mode if settings.link_prefetch_mode != 'default' else self.config.default_link_prefetch_mode,
            'link_prefetch_mode_source': 'session' if settings.link_prefetch_mode != 'default' else 'default',
            'max_interaction_rounds': self._effective_max_interaction_rounds(settings),
            'max_interaction_rounds_source': 'session' if settings.max_interaction_rounds is not None else 'default',
            'spontaneous_reply_chance': self._effective_spontaneous_reply_chance(settings),
            'spontaneous_reply_chance_source': 'session' if settings.spontaneous_reply_chance is not None else 'default',
            'group_spontaneous_reply_delay_s': self._effective_group_spontaneous_reply_delay_s(settings),
            'group_spontaneous_reply_delay_s_source': 'session' if (settings.group_spontaneous_reply_delay_s is not None or settings.spontaneous_reply_idle_s is not None) else 'default',
            'provider_retry_count': self._effective_provider_retry_count(settings),
            'provider_retry_count_source': 'session' if settings.provider_retry_count is not None else 'default',
            'private_reply_delay_s': self._effective_private_reply_delay_s(settings),
            'private_reply_delay_s_source': 'session' if (settings.private_reply_delay_s is not None or settings.reply_delay_s is not None) else 'default',
            'group_reply_delay_s': self._effective_group_reply_delay_s(settings),
            'group_reply_delay_s_source': 'session' if (settings.group_reply_delay_s is not None or settings.reply_delay_s is not None) else 'default',
            'metadata_injection_mode': settings.metadata_injection_mode or 'on',
            'metadata_injection_mode_source': 'session' if (settings.metadata_injection_mode or 'on') != self.config.default_metadata_injection_mode else 'default',
            'metadata_timezone': settings.metadata_timezone or 'UTC',
            'metadata_timezone_source': 'session' if (settings.metadata_timezone or 'UTC') != self.config.default_metadata_timezone else 'default',
            'system_prompt_chars': len(settings.system_prompt or ''),
            'raw_messages': len(state.raw_messages),
            'tool_history_messages': sum(1 for item in state.raw_messages if item.message.role == MessageRole.TOOL),
            'memory_blocks': len(state.blocks),
            'l0_blocks': l0_blocks,
            'l1_blocks': l1_blocks,
            'l2_blocks': l2_blocks,
            'estimated_history_tokens': request_estimate.history_tokens,
            'estimated_request_tokens': request_estimate.total_tokens,
            'estimated_request_images': self._estimate_request_images(state),
            'max_input_images': format_optional_disabled_int(max_input_images, disabled_label='unlimited'),
            'max_input_images_source': 'session' if settings.max_input_images is not None else 'default',
            'compact_target_images': format_optional_disabled_int(compact_target_images, disabled_label='disabled'),
            'compact_target_images_source': 'session' if settings.compact_target_images is not None else 'default',
            'compact_trigger_tokens': self._effective_compact_trigger_tokens(settings),
            'compact_trigger_tokens_source': 'session' if settings.compact_trigger_tokens is not None else 'default',
            'compact_target_tokens': self._effective_compact_target_tokens(settings),
            'compact_target_tokens_source': 'session' if settings.compact_target_tokens is not None else 'default',
            'compact_batch_tokens': self._effective_compact_batch_tokens(settings),
            'compact_batch_tokens_source': 'session' if settings.compact_batch_tokens is not None else 'default',
            'compact_keep_recent_ratio': f"{self._effective_compact_keep_recent_ratio(settings):.2f}",
            'compact_keep_recent_ratio_source': 'session' if settings.compact_keep_recent_ratio is not None else 'default',
            'compact_tool_ratio_threshold': f"{self._effective_compact_tool_ratio_threshold(settings):.2f}",
            'compact_tool_ratio_threshold_source': 'session' if settings.compact_tool_ratio_threshold is not None else 'default',
            'compact_tool_min_tokens': self._effective_compact_tool_min_tokens(settings),
            'compact_tool_min_tokens_source': 'session' if settings.compact_tool_min_tokens is not None else 'default',
            'compact_min_messages': self._effective_compact_min_messages(settings),
            'compact_min_messages_source': 'session' if settings.compact_min_messages is not None else 'default',
            'min_raw_messages_reserve': self._effective_min_raw_messages_reserve(settings),
            'min_raw_messages_reserve_source': 'session' if settings.min_raw_messages_reserve is not None else 'default',
            'provider_history_messages': len(self._build_provider_history(state, settings=settings, provider_name=provider.name)),
            'loaded_in_memory': state.loaded,
            'sticker_index_loaded': sticker_stats['loaded'],
            'sticker_index_count': sticker_stats['stickers'],
            'sticker_pack_count': sticker_stats['packs'],
            'remote_enabled': bool(remote and remote.enabled),
            'remote_master_ready': bool(remote and remote.enabled and getattr(remote, '_master_started', False)),
        }

    def _effective_max_interaction_rounds(self, settings: SessionSettings) -> int:
        if settings.max_interaction_rounds is not None:
            return clamp_int(settings.max_interaction_rounds, minimum=MAX_INTERACTION_ROUNDS_MIN, maximum=MAX_INTERACTION_ROUNDS_MAX, default=MAX_INTERACTION_ROUNDS_MIN)
        if settings.mode == ChatMode.CHAT:
            return clamp_int(self.config.default_chat_max_rounds, minimum=MAX_INTERACTION_ROUNDS_MIN, maximum=MAX_INTERACTION_ROUNDS_MAX, default=MAX_INTERACTION_ROUNDS_MIN)
        if settings.mode == ChatMode.ASSIST:
            return clamp_int(self.config.default_assist_max_rounds, minimum=MAX_INTERACTION_ROUNDS_MIN, maximum=MAX_INTERACTION_ROUNDS_MAX, default=MAX_INTERACTION_ROUNDS_MIN)
        return clamp_int(self.config.default_agent_max_rounds, minimum=MAX_INTERACTION_ROUNDS_MIN, maximum=MAX_INTERACTION_ROUNDS_MAX, default=MAX_INTERACTION_ROUNDS_MIN)

    def _effective_spontaneous_reply_chance(self, settings: SessionSettings) -> int:
        if settings.spontaneous_reply_chance is not None:
            return clamp_int(settings.spontaneous_reply_chance, minimum=SPONTANEOUS_REPLY_CHANCE_MIN, maximum=SPONTANEOUS_REPLY_CHANCE_MAX, default=SPONTANEOUS_REPLY_CHANCE_MIN)
        return clamp_int(self.config.default_group_spontaneous_reply_chance, minimum=SPONTANEOUS_REPLY_CHANCE_MIN, maximum=SPONTANEOUS_REPLY_CHANCE_MAX, default=SPONTANEOUS_REPLY_CHANCE_MIN)

    def _effective_group_spontaneous_reply_delay_s(self, settings: SessionSettings) -> float:
        if settings.group_spontaneous_reply_delay_s is not None:
            return clamp_float(settings.group_spontaneous_reply_delay_s, minimum=0.0, maximum=GROUP_SPONTANEOUS_REPLY_DELAY_MAX_S, default=0.0)
        if settings.spontaneous_reply_idle_s is not None:
            return clamp_float(settings.spontaneous_reply_idle_s, minimum=0.0, maximum=GROUP_SPONTANEOUS_REPLY_DELAY_MAX_S, default=0.0)
        return clamp_float(self.config.default_group_spontaneous_reply_delay_s, minimum=0.0, maximum=GROUP_SPONTANEOUS_REPLY_DELAY_MAX_S, default=0.0)

    def _effective_provider_retry_count(self, settings: SessionSettings) -> int:
        if settings.provider_retry_count is not None:
            return clamp_int(settings.provider_retry_count, minimum=PROVIDER_RETRY_COUNT_MIN, maximum=PROVIDER_RETRY_COUNT_MAX, default=PROVIDER_RETRY_COUNT_MIN)
        return clamp_int(self.config.default_provider_retry_count, minimum=PROVIDER_RETRY_COUNT_MIN, maximum=PROVIDER_RETRY_COUNT_MAX, default=PROVIDER_RETRY_COUNT_MIN)

    def _effective_private_reply_delay_s(self, settings: SessionSettings) -> float:
        if settings.private_reply_delay_s is not None:
            return clamp_float(settings.private_reply_delay_s, minimum=0.0, maximum=REPLY_DELAY_MAX_S, default=0.0)
        if settings.reply_delay_s is not None:
            return clamp_float(settings.reply_delay_s, minimum=0.0, maximum=REPLY_DELAY_MAX_S, default=0.0)
        return clamp_float(self.config.default_private_reply_delay_s, minimum=0.0, maximum=REPLY_DELAY_MAX_S, default=0.0)

    def _effective_group_reply_delay_s(self, settings: SessionSettings) -> float:
        if settings.group_reply_delay_s is not None:
            return clamp_float(settings.group_reply_delay_s, minimum=0.0, maximum=REPLY_DELAY_MAX_S, default=0.0)
        if settings.reply_delay_s is not None:
            return clamp_float(settings.reply_delay_s, minimum=0.0, maximum=REPLY_DELAY_MAX_S, default=0.0)
        return clamp_float(self.config.default_group_reply_delay_s, minimum=0.0, maximum=REPLY_DELAY_MAX_S, default=0.0)

    def _effective_native_web_search_max(self, settings: SessionSettings) -> int | None:
        return effective_optional_disabled_int(
            settings.native_web_search_max,
            self.config.openai.native_web_search_max,
            maximum=NATIVE_WEB_SEARCH_MAX_MAX,
        )

    def _effective_temperature(self, settings: SessionSettings) -> float:
        if settings.temperature is not None:
            return clamp_float(settings.temperature, minimum=TEMPERATURE_MIN, maximum=TEMPERATURE_MAX, default=TEMPERATURE_MIN)
        return clamp_float(self.config.gemini.temperature, minimum=TEMPERATURE_MIN, maximum=TEMPERATURE_MAX, default=TEMPERATURE_MIN)

    def _effective_top_p(self, settings: SessionSettings) -> float:
        if settings.top_p is not None:
            return clamp_float(settings.top_p, minimum=TOP_P_MIN, maximum=TOP_P_MAX, default=TOP_P_MIN)
        return clamp_float(self.config.gemini.top_p, minimum=TOP_P_MIN, maximum=TOP_P_MAX, default=TOP_P_MIN)

    def _effective_top_k(self, settings: SessionSettings) -> int:
        if settings.top_k is not None:
            return clamp_int(settings.top_k, minimum=TOP_K_MIN, maximum=TOP_K_MAX, default=TOP_K_MIN)
        return clamp_int(self.config.gemini.top_k, minimum=TOP_K_MIN, maximum=TOP_K_MAX, default=TOP_K_MIN)

    def _effective_compact_trigger_tokens(self, settings: SessionSettings) -> int:
        if settings.compact_trigger_tokens is not None:
            return max(256, int(settings.compact_trigger_tokens))
        return max(256, int(self.config.context.compact_trigger_tokens))

    def _effective_compact_target_tokens(self, settings: SessionSettings) -> int:
        trigger = self._effective_compact_trigger_tokens(settings)
        if settings.compact_target_tokens is not None:
            return max(256, min(trigger, int(settings.compact_target_tokens)))
        return max(256, min(trigger, int(self.config.context.compact_target_tokens)))

    def _effective_compact_batch_tokens(self, settings: SessionSettings) -> int:
        target = self._effective_compact_target_tokens(settings)
        if settings.compact_batch_tokens is not None:
            return max(256, min(target, int(settings.compact_batch_tokens)))
        return max(256, min(target, int(self.config.context.compact_batch_tokens)))

    def _configured_compact_batch_tokens(self, settings: SessionSettings) -> int:
        if settings.compact_batch_tokens is not None:
            return max(256, int(settings.compact_batch_tokens))
        return max(256, int(self.config.context.compact_batch_tokens))

    def _effective_compact_keep_recent_ratio(self, settings: SessionSettings) -> float:
        if settings.compact_keep_recent_ratio is not None:
            ratio = float(settings.compact_keep_recent_ratio)
        else:
            ratio = float(self.config.context.compact_keep_recent_ratio)
        return min(COMPACT_KEEP_RECENT_RATIO_MAX, max(COMPACT_KEEP_RECENT_RATIO_MIN, ratio))

    def _effective_compact_tool_ratio_threshold(self, settings: SessionSettings) -> float:
        if settings.compact_tool_ratio_threshold is not None:
            ratio = float(settings.compact_tool_ratio_threshold)
        else:
            ratio = float(self.config.context.compact_tool_ratio_threshold)
        return max(COMPACT_TOOL_RATIO_THRESHOLD_MIN, ratio)

    def _effective_compact_tool_min_tokens(self, settings: SessionSettings) -> int:
        if settings.compact_tool_min_tokens is not None:
            value = int(settings.compact_tool_min_tokens)
        else:
            value = int(self.config.context.compact_tool_min_tokens)
        return max(COMPACT_TOKEN_MIN, value)

    def _effective_compact_min_messages(self, settings: SessionSettings) -> int:
        if settings.compact_min_messages is not None:
            return max(COMPACT_MIN_MESSAGES_MIN, int(settings.compact_min_messages))
        return max(COMPACT_MIN_MESSAGES_MIN, int(self.config.context.compact_min_messages))

    def _effective_min_raw_messages_reserve(self, settings: SessionSettings) -> int:
        max_reasonable = max(0, self._effective_compact_min_messages(settings) - 1)
        if settings.min_raw_messages_reserve is not None:
            return max(0, min(max_reasonable, int(settings.min_raw_messages_reserve)))
        return max(0, min(max_reasonable, int(self.config.context.min_raw_messages_reserve)))

    @staticmethod
    def _compact_json(value: Any, *, limit: int = 900, rlimit: int = 0) -> str:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            text = str(value)
        limit = max(0, limit)
        rlimit = max(0, rlimit)
        marker = '...[truncated]...'

        if len(text) <= limit + rlimit:
            return text

        if limit == 0 and rlimit == 0:
            return marker
        if rlimit == 0:
            return text[:limit] + marker
        if limit == 0:
            return marker + text[-rlimit:]
        return text[:limit] + marker + text[-rlimit:]

    def _tool_observation_summary(self, *, name: str, phase: str, payload: dict[str, Any]) -> str:
        if phase == 'call':
            return self._describe_tool_call(name, payload)
        if phase == 'result':
            return self._describe_tool_result(name, payload)
        if phase == 'delivery':
            return self._describe_tool_delivery(name, payload)
        payload_text = self._compact_json(payload)
        return f'[Tool event {name}: {payload_text}]'

    async def _generate_with_retries(self, *, provider, settings: SessionSettings, messages: list[ConversationMessage], instructions: str, tools, extra_input_items):
        retries = self._effective_provider_retry_count(settings)
        last_exc = None
        for attempt in range(retries + 1):
            try:
                return await provider.generate(settings=settings, messages=messages, instructions=instructions, tools=tools, extra_input_items=extra_input_items)
            except Exception as exc:
                last_exc = exc
                if attempt >= retries:
                    raise
                logger.warning('provider.retry provider=%s model=%s attempt=%s/%s err=%s', settings.provider, settings.model, attempt + 1, retries + 1, exc.__class__.__name__)
        raise last_exc or RuntimeError('Provider call failed')

    def invalidate_session(self, session_id: str) -> None:
        self._live_sessions.pop(session_id, None)

    async def _get_live_state(self, session_id: str) -> LiveConversationState:
        state = self._live_sessions.get(session_id)
        if state and state.loaded:
            return state
        blocks = await self.store.list_memory_blocks(session_id)
        messages = await self.store.list_uncompacted_messages(session_id)
        state = LiveConversationState(session_id=session_id, blocks=blocks, raw_messages=messages, loaded=True)
        state.rebuild_estimate()
        self._live_sessions[session_id] = state
        logger.debug('state.load sid=%s raw=%s blocks=%s est_tokens=%s est_images=%s', self._session_log_id(session_id), len(messages), len(blocks), state.estimated_tokens, state.estimated_images)
        return state

    @staticmethod
    def _append_live_message(state: LiveConversationState, stored_message: StoredConversationMessage) -> None:
        state.raw_messages.append(stored_message)
        state.estimated_tokens += stored_message.estimated_tokens
        state.estimated_images += stored_message.image_count
        state.provider_history_dirty = True
        state.provider_history_cache_key = None
        state.provider_history_token_cache.clear()

    @staticmethod
    def _clone_message(message: ConversationMessage, *, metadata: dict[str, Any] | None = None) -> ConversationMessage:
        return ConversationMessage(role=message.role, parts=list(message.parts), name=message.name, metadata=metadata if metadata is not None else dict(message.metadata or {}))

    @staticmethod
    def _history_cache_key(
        *,
        provider_name: str,
        model: str,
        tool_history_mode: ToolHistoryMode,
        selected_block_ids: tuple[int, ...],
        latest_id: int,
    ) -> tuple[str, str, str, tuple[int, ...], int]:
        return (provider_name, model, tool_history_mode.value, selected_block_ids, latest_id)

    @staticmethod
    def _request_estimate_bias_key(*, provider_name: str, model: str, tool_history_mode: ToolHistoryMode) -> tuple[str, str, str]:
        return (provider_name, model, tool_history_mode.value)

    def _request_estimate_bias_for(self, *, provider_name: str, model: str, tool_history_mode: ToolHistoryMode) -> float:
        key = self._request_estimate_bias_key(provider_name=provider_name, model=model, tool_history_mode=tool_history_mode)
        return max(1.0, float(self._request_estimate_bias.get(key, 1.0)))

    def _apply_request_estimate_bias(
        self,
        estimate: RequestTokenEstimate,
        *,
        provider_name: str,
        model: str,
        tool_history_mode: ToolHistoryMode,
    ) -> RequestTokenEstimate:
        multiplier = self._request_estimate_bias_for(provider_name=provider_name, model=model, tool_history_mode=tool_history_mode)
        return estimate.scaled(multiplier) if multiplier > 1.0 else estimate

    def _update_request_estimate_bias(
        self,
        *,
        provider_name: str,
        model: str,
        tool_history_mode: ToolHistoryMode,
        raw_estimate_total: int,
        actual_input_tokens: int | None,
    ) -> None:
        actual = int(actual_input_tokens or 0)
        if actual <= 0 or raw_estimate_total <= 0:
            return
        key = self._request_estimate_bias_key(provider_name=provider_name, model=model, tool_history_mode=tool_history_mode)
        current = max(1.0, float(self._request_estimate_bias.get(key, 1.0)))
        target = min(4.0, max(1.0, (float(actual) / float(max(1, raw_estimate_total))) * 1.05))
        if target <= current:
            return
        updated = max(current, min(4.0, (current * 0.75) + (target * 0.25)))
        self._request_estimate_bias[key] = updated
        logger.debug(
            'estimate.bias provider=%s model=%s mode=%s raw=%s actual=%s bias=%.3f->%.3f',
            provider_name,
            model,
            tool_history_mode.value,
            raw_estimate_total,
            actual,
            current,
            updated,
        )

    def _estimate_history_tokens_for_messages(
        self,
        *,
        state: LiveConversationState,
        provider: ModelProvider,
        settings: SessionSettings,
        selected_blocks: list[MemoryBlock],
        messages: list[ConversationMessage],
    ) -> int:
        latest_id = state.raw_messages[-1].db_id if state.raw_messages else 0
        cache_key = self._history_cache_key(
            provider_name=provider.name,
            model=settings.model,
            tool_history_mode=settings.tool_history_mode,
            selected_block_ids=tuple(block.block_id for block in selected_blocks),
            latest_id=latest_id,
        )
        cached = state.provider_history_token_cache.get(cache_key)
        if cached is not None:
            return int(cached)
        estimate = provider.estimate_request_tokens(
            settings=settings,
            messages=messages,
            instructions='',
            tools=[],
        )
        state.provider_history_token_cache[cache_key] = int(estimate.history_tokens)
        return int(estimate.history_tokens)

    def _estimate_request_breakdown(
        self,
        *,
        state: LiveConversationState,
        settings: SessionSettings,
        provider: ModelProvider,
        instructions: str,
        tools: list[ToolSpec],
        extra_input_items: list[dict] | None = None,
    ) -> RequestTokenEstimate:
        history = self._build_provider_history(state, settings=settings, provider_name=provider.name)
        selected_blocks = self._select_blocks_for_prompt(state, settings=settings)
        history_tokens = self._estimate_history_tokens_for_messages(
            state=state,
            provider=provider,
            settings=settings,
            selected_blocks=selected_blocks,
            messages=history,
        )
        raw_estimate = provider.estimate_request_tokens(
            settings=settings,
            messages=history,
            instructions=instructions,
            tools=tools,
            extra_input_items=extra_input_items,
            history_tokens_override=history_tokens,
        )
        return self._apply_request_estimate_bias(
            raw_estimate,
            provider_name=provider.name,
            model=settings.model,
            tool_history_mode=settings.tool_history_mode,
        )

    def _estimate_raw_history_tokens(self, state: LiveConversationState, *, settings: SessionSettings, provider: ModelProvider) -> int:
        latest_id = state.raw_messages[-1].db_id if state.raw_messages else 0
        cache_key = self._history_cache_key(
            provider_name=provider.name,
            model=settings.model,
            tool_history_mode=settings.tool_history_mode,
            selected_block_ids=(),
            latest_id=latest_id,
        )
        cached = state.provider_history_token_cache.get(cache_key)
        if cached is not None:
            return int(cached)
        raw_messages: list[ConversationMessage] = []
        for item in state.raw_messages:
            mapped = self._history_message_for_provider(settings=settings, provider_name=provider.name, message=item.message)
            if mapped is not None:
                raw_messages.append(mapped)
        estimate = provider.estimate_request_tokens(
            settings=settings,
            messages=raw_messages,
            instructions='',
            tools=[],
        )
        state.provider_history_token_cache[cache_key] = int(estimate.history_tokens)
        return int(estimate.history_tokens)

    def _estimate_stored_messages_prompt_tokens(
        self,
        messages: list[StoredConversationMessage],
        *,
        settings: SessionSettings,
        provider: ModelProvider,
        sequence_token_cache: dict[tuple[int, ...], int] | None = None,
    ) -> int:
        if not messages:
            return 0
        cache_key = tuple(int(item.db_id) for item in messages)
        if sequence_token_cache is not None:
            cached = sequence_token_cache.get(cache_key)
            if cached is not None:
                return int(cached)
        mapped_messages = [
            mapped
            for item in messages
            if (mapped := self._history_message_for_provider(settings=settings, provider_name=provider.name, message=item.message)) is not None
        ]
        estimate = provider.estimate_request_tokens(
            settings=settings,
            messages=mapped_messages,
            instructions='',
            tools=[],
        )
        total = int(estimate.history_tokens)
        if sequence_token_cache is not None:
            sequence_token_cache[cache_key] = total
        return total

    @staticmethod
    def _history_position_for_block(block: MemoryBlock) -> tuple[int, int]:
        if block.end_message_id is not None:
            return (int(block.end_message_id), 0)
        if block.start_message_id is not None:
            return (int(block.start_message_id), 0)
        return (10**12 + int(block.sequence_no), 0)

    def _sanitize_provider_native_items_for_history(self, provider_name: str, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        # if any
        return [item for item in items if isinstance(item, dict)]

    def _provider_visible_text_from_items(self, items: list[dict[str, Any]]) -> str:
        texts: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get('type') or '').strip().lower()
            if item_type == 'message':
                content_items = item.get('content') if isinstance(item.get('content'), list) else []
                for content in content_items:
                    if not isinstance(content, dict) or str(content.get('type') or '').strip().lower() != 'output_text':
                        continue
                    text = str(content.get('text') or '').strip()
                    if text:
                        texts.append(text)
                continue
            parts = item.get('parts') if isinstance(item.get('parts'), list) else []
            for part in parts:
                if not isinstance(part, dict) or part.get('thought') is True:
                    continue
                text = str(part.get('text') or '').strip()
                if text:
                    texts.append(text)
        return '\n\n'.join(texts).strip()

    def _provider_native_visible_text(self, message: ConversationMessage) -> str:
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        provider_native = metadata.get('provider_native') if isinstance(metadata.get('provider_native'), dict) else {}
        items = provider_native.get('items') if isinstance(provider_native.get('items'), list) else []
        return self._provider_visible_text_from_items(items)

    def _history_message_for_provider(self, *, settings: SessionSettings, provider_name: str, message: ConversationMessage) -> ConversationMessage | None:
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        base_metadata = {key: value for key, value in metadata.items() if key not in {'provider_native', 'provider_native_skip_same_provider'}}
        provider_native = metadata.get('provider_native') if isinstance(metadata.get('provider_native'), dict) else None
        native_provider = str(provider_native.get('provider') or '').strip().lower() if provider_native else ''
        same_provider_native = settings.tool_history_mode == ToolHistoryMode.NATIVE_SAME_PROVIDER and native_provider == provider_name
        if same_provider_native and provider_native:
            items = provider_native.get('items') if isinstance(provider_native.get('items'), list) else []
            base_metadata['provider_native'] = {
                'provider': provider_name,
                'items': self._sanitize_provider_native_items_for_history(provider_name, items),
            }
        prepared = self._clone_message(message, metadata=base_metadata)
        if message.role != MessageRole.TOOL:
            return prepared
        phase = str(metadata.get('tool_phase') or '').strip().lower()
        origin_provider = str(metadata.get('tool_provider') or '').strip().lower()
        if (
            settings.tool_history_mode == ToolHistoryMode.NATIVE_SAME_PROVIDER
            and origin_provider == provider_name
            and bool(metadata.get('provider_native_skip_same_provider'))
            and phase in {'call', 'result'}
        ):
            return None
        if settings.tool_history_mode == ToolHistoryMode.NATIVE_SAME_PROVIDER and origin_provider == provider_name and phase in {'call', 'result'}:
            return prepared
        texts = [part.text.strip() for part in message.parts if part.text and part.text.strip()]
        provider_visible_text = self._provider_native_visible_text(message)
        if not texts:
            name = message.name or 'tool'
            payload = metadata.get('tool_payload')
            texts = [self._tool_observation_summary(name=name, phase='event', payload=payload if isinstance(payload, dict) else {})]
        translated_metadata = {'source_role': 'tool', 'tool_name': message.name, 'tool_phase': phase or 'event'}
        translated_text = '\n'.join(texts)
        if phase == 'call':
            if provider_visible_text and provider_visible_text not in translated_text:
                translated_text = f'{provider_visible_text}\n\n{translated_text}' if translated_text else provider_visible_text
            return ConversationMessage.assistant_text(translated_text, metadata=translated_metadata)
        return ConversationMessage.user_text(translated_text, metadata=translated_metadata)

    def _build_provider_history(self, state: LiveConversationState, *, settings: SessionSettings, provider_name: str) -> list[ConversationMessage]:
        latest_id = state.raw_messages[-1].db_id if state.raw_messages else 0
        selected_blocks = self._select_blocks_for_prompt(state, settings=settings)
        cache_key = self._history_cache_key(
            provider_name=provider_name,
            model=settings.model,
            tool_history_mode=settings.tool_history_mode,
            selected_block_ids=tuple(block.block_id for block in selected_blocks),
            latest_id=latest_id,
        )
        if not state.provider_history_dirty and state.provider_history_cache and state.provider_history_cache_key == cache_key:
            return state.provider_history_cache
        entries: list[tuple[tuple[int, int], ConversationMessage]] = []
        for block in selected_blocks:
            entries.append((self._history_position_for_block(block), block.render_as_message()))
        for item in state.raw_messages:
            mapped = self._history_message_for_provider(settings=settings, provider_name=provider_name, message=item.message)
            if mapped is not None:
                entries.append(((int(item.db_id), 1), mapped))
        messages = [message for _key, message in sorted(entries, key=lambda item: item[0])]
        state.provider_history_cache = messages
        state.provider_history_cache_key = cache_key
        state.provider_history_dirty = False
        return messages

    def _select_blocks_for_prompt(self, state: LiveConversationState, *, settings: SessionSettings) -> list[MemoryBlock]:
        blocks = [block for block in state.blocks if block.lifecycle == 'sealed']
        if not blocks:
            return []
        target_tokens = self._effective_compact_target_tokens(settings)
        provider = self._require_provider(settings.provider)
        raw_tokens = self._estimate_raw_history_tokens(state, settings=settings, provider=provider)
        if raw_tokens > target_tokens * 0.75:
            budget_fraction = 0.08
        elif raw_tokens > target_tokens * 0.5:
            budget_fraction = 0.12
        elif raw_tokens > target_tokens * 0.25:
            budget_fraction = 0.16
        else:
            budget_fraction = 0.22
        budget = int(target_tokens * budget_fraction)
        budget_floor = max(256, min(1024, int(target_tokens * 0.05)))
        budget = max(budget_floor, min(16000, budget))
        selected: list[MemoryBlock] = []
        selected_ids: set[int] = set()
        covered_parent_ids: set[int] = set()
        used_tokens = 0
        digests = [block for block in blocks if block.kind == 'digest']
        episodes = [block for block in blocks if block.kind != 'digest']

        for block in sorted(digests, key=lambda item: item.sequence_no, reverse=True):
            if block.block_id in covered_parent_ids or block.block_id in selected_ids:
                continue
            if used_tokens and used_tokens + block.estimated_tokens > budget:
                continue
            selected.append(block)
            selected_ids.add(block.block_id)
            covered_parent_ids.update(int(parent_id) for parent_id in block.parent_block_ids)
            used_tokens += block.estimated_tokens
            if len(selected) >= 2 and used_tokens >= int(budget * 0.6):
                break

        recent_episode_allowance = 1 if state.raw_messages else 2
        for block in sorted(episodes, key=lambda item: item.sequence_no, reverse=True)[:recent_episode_allowance]:
            if block.block_id in selected_ids or block.block_id in covered_parent_ids:
                continue
            if used_tokens and used_tokens + block.estimated_tokens > int(budget * 1.05):
                continue
            selected.append(block)
            selected_ids.add(block.block_id)
            used_tokens += block.estimated_tokens

        for block in sorted(episodes, key=lambda item: item.sequence_no, reverse=True):
            if block.block_id in selected_ids or block.block_id in covered_parent_ids:
                continue
            if used_tokens + block.estimated_tokens > budget:
                continue
            selected.append(block)
            selected_ids.add(block.block_id)
            used_tokens += block.estimated_tokens
            if len(selected) >= 6:
                break

        return sorted(selected, key=lambda item: item.sequence_no)

    @staticmethod
    def _block_time_bounds(blocks: list[MemoryBlock]) -> tuple[str | None, str | None]:
        values = sorted(
            value
            for block in blocks
            for value in (block.time_start, block.time_end)
            if value
        )
        if not values:
            return None, None
        return values[0], values[-1]

    async def _compact_if_needed(
        self,
        *,
        session_id: str,
        settings: SessionSettings,
        provider: ModelProvider,
        state: LiveConversationState,
        instructions: str,
        tools: list[ToolSpec] | None = None,
        emit: EventCallback | None,
    ) -> None:
        tools = tools or []
        total_estimate = self._estimate_request_tokens(
            state,
            settings=settings,
            provider=provider,
            instructions=instructions,
            tools=tools,
        )
        image_limit = self._effective_max_input_images(provider, settings)
        image_count = self._estimate_request_images(state)
        compact_trigger_tokens = self._effective_compact_trigger_tokens(settings)
        compact_target_tokens = self._effective_compact_target_tokens(settings)
        image_target = self._effective_compact_target_images(provider, settings)
        token_overflow = total_estimate > compact_trigger_tokens
        image_overflow = image_limit is not None and image_count > image_limit
        if not token_overflow and not image_overflow:
            return
        logger.info('compact.start sid=%s est_tokens=%s est_images=%s image_limit=%s', self._session_log_id(session_id), total_estimate, image_count, image_limit)

        if emit and settings.process_visibility != ProcessVisibility.OFF:
            phase_title = 'Compacting context' if token_overflow else 'Compacting images'
            detail = f'estimated prompt {total_estimate:,} tokens; target {compact_target_tokens:,}'
            if image_limit is not None:
                detail += f'; images {image_count}/{image_limit}'
            await emit(RuntimeEvent(kind='phase', title=phase_title, detail=detail))

        try:
            if token_overflow:
                while total_estimate > compact_target_tokens:
                    changed = await self._compact_old_context(session_id=session_id, settings=settings, provider=provider, state=state, pressure=False, emit=emit)
                    if not changed and total_estimate > compact_target_tokens:
                        changed = await self._compact_old_context(session_id=session_id, settings=settings, provider=provider, state=state, pressure=True, emit=emit)
                    if not changed:
                        logger.warning('Unable to compact session %s below target; remaining estimate=%s images=%s', session_id, total_estimate, image_count)
                        break
                    total_estimate = self._estimate_request_tokens(
                        state,
                        settings=settings,
                        provider=provider,
                        instructions=instructions,
                        tools=tools,
                    )
                    image_count = self._estimate_request_images(state)

            image_count = self._estimate_request_images(state)
            if image_limit is not None and image_count > image_limit:
                image_target_limit = image_target if image_target is not None else image_limit
                removed_images = await self._compact_oldest_images(
                    session_id=session_id,
                    settings=settings,
                    state=state,
                    target_images=image_target_limit,
                    emit=emit,
                )
                if not removed_images and image_count > image_limit:
                    logger.warning('Unable to compact images for session %s below limit; remaining images=%s limit=%s target=%s', session_id, image_count, image_limit, image_target_limit)
        except CompactionModelRequestFailed as exc:
            logger.warning(
                'compact.halt sid=%s provider=%s mode=%s reason=model_request_failed',
                self._session_log_id(session_id),
                exc.provider_name,
                exc.mode,
            )
            return

    def _estimate_request_tokens(
        self,
        state: LiveConversationState,
        *,
        settings: SessionSettings,
        provider: ModelProvider,
        instructions: str,
        tools: list[ToolSpec],
        extra_input_items: list[dict] | None = None,
    ) -> int:
        return self._estimate_request_breakdown(
            state=state,
            settings=settings,
            provider=provider,
            instructions=instructions,
            tools=tools,
            extra_input_items=extra_input_items,
        ).total_tokens

    def _estimate_request_images(self, state: LiveConversationState) -> int:
        return state.estimated_images

    def _digest_needed(self, state: LiveConversationState, settings: SessionSettings, *, total_estimate: int | None = None, pressure: bool = False) -> bool:
        shard = self._select_digest_shard(state.blocks, settings=settings, pressure=pressure)
        if not shard:
            return False
        episodes = [block for block in state.blocks if block.kind == 'episode' and block.lifecycle == 'sealed']
        selected_block_tokens = sum(block.estimated_tokens for block in self._select_blocks_for_prompt(state, settings=settings))
        target_tokens = self._effective_compact_target_tokens(settings)
        prompt_pressure = total_estimate is not None and total_estimate > target_tokens
        return prompt_pressure or len(episodes) >= 6 or selected_block_tokens > int(target_tokens * 0.45)

    def _effective_max_input_images(self, provider: ModelProvider, settings: SessionSettings) -> int | None:
        name = getattr(provider, 'name', '')
        default = self.config.gemini.max_input_images if name == 'gemini' else self.config.openai.max_input_images if name == 'openai' else 0
        return effective_optional_disabled_int(settings.max_input_images, default, maximum=IMAGE_LIMIT_MAX)

    def _configured_compact_target_images(self, provider: ModelProvider, settings: SessionSettings) -> int | None:
        name = getattr(provider, 'name', '')
        default = self.config.gemini.compact_target_images if name == 'gemini' else self.config.openai.compact_target_images if name == 'openai' else 0
        return effective_optional_disabled_int(settings.compact_target_images, default, maximum=IMAGE_LIMIT_MAX)

    def _effective_compact_target_images(self, provider: ModelProvider, settings: SessionSettings) -> int | None:
        configured_target = self._configured_compact_target_images(provider, settings)
        if configured_target is not None:
            return configured_target
        return self._effective_max_input_images(provider, settings)

    async def _compact_oldest_images(
        self,
        *,
        session_id: str,
        settings: SessionSettings,
        state: LiveConversationState,
        target_images: int,
        emit: EventCallback | None = None,
    ) -> int:
        images_to_remove = max(0, int(state.estimated_images) - max(0, int(target_images)))
        if images_to_remove <= 0:
            return 0

        removed_images = 0
        touched_messages = 0
        for index, stored in enumerate(state.raw_messages):
            if removed_images >= images_to_remove:
                break
            updated_parts: list[MessagePart] = []
            changed = False
            for part in stored.message.parts:
                if removed_images < images_to_remove and part.kind == PartKind.IMAGE:
                    updated_parts.append(MessagePart(kind=PartKind.TEXT, text='[Image compacted]', remote_sync=False, origin='image_compacted'))
                    removed_images += 1
                    changed = True
                    continue
                updated_parts.append(part)
            if not changed:
                continue
            updated_message = replace(stored.message, parts=updated_parts)
            updated_stored = replace(stored, message=updated_message, estimated_tokens=TokenEstimator.estimate_message(updated_message))
            state.raw_messages[index] = await self.store.update_message(session_id, updated_stored)
            touched_messages += 1

        if removed_images > 0:
            state.rebuild_estimate()
            logger.info(
                'compact.images sid=%s removed=%s touched_messages=%s remaining_images=%s target=%s',
                self._session_log_id(session_id),
                removed_images,
                touched_messages,
                state.estimated_images,
                target_images,
            )
            if emit and settings.process_visibility != ProcessVisibility.OFF:
                await emit(RuntimeEvent(kind='phase', title='Compacting images', detail=f'compacted {removed_images} earliest images; remaining {state.estimated_images}/{target_images}'))
        return removed_images

    async def _compact_old_context(
        self,
        *,
        session_id: str,
        settings: SessionSettings,
        provider: ModelProvider,
        state: LiveConversationState,
        pressure: bool = False,
        emit: EventCallback | None = None,
    ) -> bool:
        skipped_message_ids: set[int] = set()
        skipped_block_ids: set[int] = set()
        skipped_digest_shards: set[tuple[int, ...]] = set()
        sequence_token_cache: dict[tuple[int, ...], int] = {}
        max_attempts = 24
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            tool_slice = self._select_tool_heavy_raw_slice(
                state.raw_messages,
                settings=settings,
                provider=provider,
                pressure=pressure,
                excluded_message_ids=skipped_message_ids,
                sequence_token_cache=sequence_token_cache,
            )
            if tool_slice:
                candidate = await self._make_toolspan_block_candidate(provider, settings, tool_slice)
                if candidate is None:
                    skipped_ids = {item.db_id for item in tool_slice}
                    skipped_message_ids.update(skipped_ids)
                    logger.info('compact.l0.skip sid=%s raw_messages=%s skipped=%s', self._session_log_id(session_id), len(tool_slice), len(skipped_ids))
                    continue
                block_text = self._render_memory_block_text('toolspan', candidate['data'], time_start=candidate['time_start'], time_end=candidate['time_end'])
                block = await self.store.create_memory_block(
                    session_id,
                    summary_text=block_text,
                    estimated_tokens=TokenEstimator.estimate_text(block_text) + 32,
                    source_message_ids=[item.db_id for item in tool_slice],
                    level=0,
                    kind='toolspan',
                    lifecycle='sealed',
                    source_kind='raw',
                    source_message_count=len(tool_slice),
                    parent_block_ids=[],
                    topic_labels=candidate['topic_labels'],
                    actor_labels=candidate['actor_labels'],
                    time_start=candidate['time_start'],
                    time_end=candidate['time_end'],
                    retained_raw_excerpt_count=len(candidate['data'].get('retained_raw_excerpts', [])),
                    validator_status='passed',
                    validator_score=candidate['score'],
                    structured_data=candidate['data'],
                )
                consumed_ids = {item.db_id for item in tool_slice}
                state.blocks.append(block)
                state.raw_messages = [item for item in state.raw_messages if item.db_id not in consumed_ids]
                state.rebuild_estimate()
                tool_slice_tokens = self._estimate_stored_messages_prompt_tokens(
                    tool_slice,
                    settings=settings,
                    provider=provider,
                    sequence_token_cache=sequence_token_cache,
                )
                logger.info(
                    'compact.l0 sid=%s raw_messages=%s raw_tokens=%s->%s tool_user_ratio_threshold=%.2f',
                    self._session_log_id(session_id),
                    len(tool_slice),
                    tool_slice_tokens,
                    block.estimated_tokens,
                    self._effective_compact_tool_ratio_threshold(settings),
                )
                if emit and settings.process_visibility != ProcessVisibility.OFF:
                    await emit(RuntimeEvent(kind='phase', title='Compacting context',
                            detail=f'compact.l0 raw_messages={len(tool_slice)} raw_tokens={tool_slice_tokens}->{block.estimated_tokens}'))
                return True

            history_slice = self._select_oldest_history_slice(
                state,
                settings=settings,
                provider=provider,
                pressure=pressure,
                excluded_message_ids=skipped_message_ids,
                excluded_block_ids=skipped_block_ids,
                sequence_token_cache=sequence_token_cache,
            )
            if history_slice:
                candidate = await self._make_episode_block_candidate(provider, settings, history_slice['source_messages'], history_slice['raw_messages'], history_slice['parent_blocks'])
                if candidate is None:
                    skipped_ids = {item.db_id for item in history_slice['raw_messages']}
                    skipped_blocks = {block.block_id for block in history_slice['parent_blocks']}
                    skipped_message_ids.update(skipped_ids)
                    skipped_block_ids.update(skipped_blocks)
                    logger.info('compact.episode.skip sid=%s raw_messages=%s parent_blocks=%s', self._session_log_id(session_id), len(skipped_ids), len(skipped_blocks))
                    continue
                block_text = self._render_memory_block_text('episode', candidate['data'], time_start=candidate['time_start'], time_end=candidate['time_end'])
                estimate = TokenEstimator.estimate_text(block_text) + 32
                raw_ids = [item.db_id for item in history_slice['raw_messages']]
                parent_ids = [block.block_id for block in history_slice['parent_blocks']]
                if parent_ids:
                    block = await self.store.replace_memory_blocks(
                        session_id,
                        block_ids=parent_ids,
                        summary_text=block_text,
                        estimated_tokens=estimate,
                        source_message_count=history_slice['source_message_count'],
                        start_message_id=history_slice['start_message_id'],
                        end_message_id=history_slice['end_message_id'],
                        level=1,
                        kind='episode',
                        lifecycle='sealed',
                        source_kind='mixed' if raw_ids else 'blocks',
                        parent_block_ids=parent_ids,
                        topic_labels=candidate['topic_labels'],
                        actor_labels=candidate['actor_labels'],
                        time_start=candidate['time_start'],
                        time_end=candidate['time_end'],
                        retained_raw_excerpt_count=len(candidate['data'].get('retained_raw_excerpts', [])),
                        validator_status='passed',
                        validator_score=candidate['score'],
                        structured_data=candidate['data'],
                        source_message_ids=raw_ids,
                    )
                    state.blocks = await self.store.list_memory_blocks(session_id)
                else:
                    block = await self.store.create_memory_block(
                        session_id,
                        summary_text=block_text,
                        estimated_tokens=estimate,
                        source_message_ids=raw_ids,
                        level=1,
                        kind='episode',
                        lifecycle='sealed',
                        source_kind='raw',
                        source_message_count=history_slice['source_message_count'],
                        start_message_id=history_slice['start_message_id'],
                        end_message_id=history_slice['end_message_id'],
                        parent_block_ids=[],
                        topic_labels=candidate['topic_labels'],
                        actor_labels=candidate['actor_labels'],
                        time_start=candidate['time_start'],
                        time_end=candidate['time_end'],
                        retained_raw_excerpt_count=len(candidate['data'].get('retained_raw_excerpts', [])),
                        validator_status='passed',
                        validator_score=candidate['score'],
                        structured_data=candidate['data'],
                    )
                    state.blocks.append(block)
                consumed_ids = set(raw_ids)
                state.raw_messages = [item for item in state.raw_messages if item.db_id not in consumed_ids]
                state.rebuild_estimate()
                raw_history_tokens = self._estimate_stored_messages_prompt_tokens(
                    history_slice['raw_messages'],
                    settings=settings,
                    provider=provider,
                    sequence_token_cache=sequence_token_cache,
                ) + sum(parent.estimated_tokens for parent in history_slice['parent_blocks'])
                logger.info(
                    'compact.episode sid=%s raw_messages=%s parent_blocks=%s raw_tokens=%s->%s keep_ratio=%.2f',
                    self._session_log_id(session_id),
                    len(raw_ids),
                    len(parent_ids),
                    raw_history_tokens,
                    block.estimated_tokens,
                    self._effective_compact_keep_recent_ratio(settings),
                )
                if emit and settings.process_visibility != ProcessVisibility.OFF:
                    await emit(RuntimeEvent(kind='phase', title='Compacting context',
                            detail=f'compact.episode raw_messages={len(raw_ids)} parent_blocks={len(parent_ids)} raw_tokens={raw_history_tokens}->{block.estimated_tokens}'))
                return True

            if not self._digest_needed(state, settings, pressure=pressure):
                return False
            shard = self._select_digest_shard(
                state.blocks,
                settings=settings,
                pressure=pressure,
                excluded_parent_signatures=skipped_digest_shards,
            )
            if shard:
                candidate = await self._make_digest_block_candidate(provider, settings, shard)
                if candidate is None:
                    signature = tuple(block.block_id for block in shard)
                    skipped_digest_shards.add(signature)
                    logger.info('compact.digest.skip sid=%s parent_blocks=%s signature=%s', self._session_log_id(session_id), len(shard), signature)
                    continue
                block_text = self._render_memory_block_text('digest', candidate['data'], time_start=candidate['time_start'], time_end=candidate['time_end'])
                parent_ids = [block.block_id for block in shard]
                old_digest_ids = [
                    block.block_id
                    for block in state.blocks
                    if block.kind == 'digest' and tuple(block.parent_block_ids) == tuple(parent_ids)
                ]
                source_message_count = sum(block.source_message_count for block in shard)
                start_message_id = min((block.start_message_id for block in shard if block.start_message_id is not None), default=None)
                end_message_id = max((block.end_message_id for block in shard if block.end_message_id is not None), default=None)
                estimate = TokenEstimator.estimate_text(block_text) + 32
                if old_digest_ids:
                    await self.store.replace_memory_blocks(
                        session_id,
                        block_ids=old_digest_ids,
                        summary_text=block_text,
                        estimated_tokens=estimate,
                        source_message_count=source_message_count,
                        start_message_id=start_message_id,
                        end_message_id=end_message_id,
                        level=2,
                        kind='digest',
                        lifecycle='sealed',
                        source_kind='blocks',
                        parent_block_ids=parent_ids,
                        topic_labels=candidate['topic_labels'],
                        actor_labels=candidate['actor_labels'],
                        time_start=candidate['time_start'],
                        time_end=candidate['time_end'],
                        retained_raw_excerpt_count=0,
                        validator_status='passed',
                        validator_score=candidate['score'],
                        structured_data=candidate['data'],
                    )
                else:
                    await self.store.create_memory_block(
                        session_id,
                        summary_text=block_text,
                        estimated_tokens=estimate,
                        source_message_ids=[],
                        level=2,
                        kind='digest',
                        lifecycle='sealed',
                        source_kind='blocks',
                        source_message_count=source_message_count,
                        start_message_id=start_message_id,
                        end_message_id=end_message_id,
                        parent_block_ids=parent_ids,
                        topic_labels=candidate['topic_labels'],
                        actor_labels=candidate['actor_labels'],
                        time_start=candidate['time_start'],
                        time_end=candidate['time_end'],
                        retained_raw_excerpt_count=0,
                        validator_status='passed',
                        validator_score=candidate['score'],
                        structured_data=candidate['data'],
                    )
                state.blocks = await self.store.list_memory_blocks(session_id)
                state.rebuild_estimate()
                logger.info('compact.digest sid=%s parent_blocks=%s visible_blocks=%s', self._session_log_id(session_id), len(shard), len(state.blocks))
                if emit and settings.process_visibility != ProcessVisibility.OFF:
                    await emit(RuntimeEvent(kind='phase', title='Compacting context',
                            detail=f'compact.digest parent_blocks={len(shard)} visible_blocks={len(state.blocks)}'))
                return True

            return False
        logger.warning('compact.skip_exhausted sid=%s pressure=%s', self._session_log_id(session_id), int(pressure))
        return False

    def _protected_recent_raw_units(
        self,
        messages: list[StoredConversationMessage],
        *,
        settings: SessionSettings,
        provider: ModelProvider,
        pressure: bool = False,
        sequence_token_cache: dict[tuple[int, ...], int] | None = None,
    ) -> tuple[list[list[StoredConversationMessage]], list[list[StoredConversationMessage]]]:
        if not messages:
            return [], []
        units = self._group_raw_compaction_units(messages)
        if len(units) < 2:
            return units, []
        total_raw_tokens = sum(
            self._estimate_stored_messages_prompt_tokens(
                unit,
                settings=settings,
                provider=provider,
                sequence_token_cache=sequence_token_cache,
            )
            for unit in units
        )
        keep_ratio = self._effective_compact_keep_recent_ratio(settings)
        if pressure:
            keep_ratio = min(keep_ratio, 0.20)
        retain_budget = max(0, int(math.ceil(total_raw_tokens * keep_ratio)))
        if pressure:
            retain_budget = min(retain_budget, max(0, self._effective_compact_batch_tokens(settings) // 3))
        keep_units = 0
        kept_tokens = 0
        minimum_keep_units = max(1, self._effective_min_raw_messages_reserve(settings))
        if pressure:
            minimum_keep_units = 1
        max_keep_units = len(units) - 1
        if max_keep_units <= 0:
            return units, []
        for unit in reversed(units):
            if keep_units >= max_keep_units:
                break
            unit_tokens = self._estimate_stored_messages_prompt_tokens(
                unit,
                settings=settings,
                provider=provider,
                sequence_token_cache=sequence_token_cache,
            )
            if keep_units < minimum_keep_units or keep_units == 0 or kept_tokens < retain_budget:
                keep_units += 1
                kept_tokens += unit_tokens
            else:
                break
        if keep_units >= len(units):
            keep_units = len(units) - 1
        if keep_units <= 0:
            return units, []
        return units[:-keep_units], units[-keep_units:]

    def _eligible_compaction_units(
        self,
        messages: list[StoredConversationMessage],
        *,
        settings: SessionSettings,
        provider: ModelProvider,
        pressure: bool = False,
        excluded_message_ids: set[int] | None = None,
        sequence_token_cache: dict[tuple[int, ...], int] | None = None,
    ) -> list[list[StoredConversationMessage]]:
        eligible_units, _protected_units = self._protected_recent_raw_units(
            messages,
            settings=settings,
            provider=provider,
            pressure=pressure,
            sequence_token_cache=sequence_token_cache,
        )
        if not eligible_units:
            return []
        if not excluded_message_ids:
            return eligible_units
        runs = self._split_units_by_excluded_messages(eligible_units, excluded_message_ids)
        return runs[0] if runs else []

    @staticmethod
    def _split_units_by_excluded_messages(units: list[list[StoredConversationMessage]], excluded_message_ids: set[int]) -> list[list[list[StoredConversationMessage]]]:
        runs: list[list[list[StoredConversationMessage]]] = []
        current: list[list[StoredConversationMessage]] = []
        for unit in units:
            if any(item.db_id in excluded_message_ids for item in unit):
                if current:
                    runs.append(current)
                    current = []
                continue
            current.append(unit)
        if current:
            runs.append(current)
        return runs

    def _select_tool_heavy_raw_slice(
        self,
        messages: list[StoredConversationMessage],
        *,
        settings: SessionSettings,
        provider: ModelProvider,
        pressure: bool = False,
        excluded_message_ids: set[int] | None = None,
        sequence_token_cache: dict[tuple[int, ...], int] | None = None,
    ) -> list[StoredConversationMessage]:
        if not messages:
            return []
        older_units, _protected_units = self._protected_recent_raw_units(
            messages,
            settings=settings,
            provider=provider,
            pressure=pressure,
            sequence_token_cache=sequence_token_cache,
        )
        if len(older_units) < 3:
            return []
        scan_end = len(older_units)
        while scan_end > 0 and not self._toolspan_unit_stats(
            older_units[scan_end - 1],
            settings=settings,
            provider=provider,
            sequence_token_cache=sequence_token_cache,
        )['has_tool']:
            scan_end -= 1
        scan_units = older_units[:scan_end] if scan_end > 0 else []
        if len(scan_units) < 2:
            return []
        runs = self._split_units_by_excluded_messages(scan_units, excluded_message_ids or set()) if excluded_message_ids else [scan_units]
        min_tokens = self._effective_compact_tool_min_tokens(settings)
        ratio_threshold = self._effective_compact_tool_ratio_threshold(settings)
        max_gap_units = 3
        max_gap_user_messages = 3
        max_span_tokens = max(min_tokens, self._configured_compact_batch_tokens(settings) * (6 if pressure else 4))
        for run_units in runs:
            if len(run_units) < max_gap_units:
                continue
            stats = [
                self._toolspan_unit_stats(
                    unit,
                    settings=settings,
                    provider=provider,
                    sequence_token_cache=sequence_token_cache,
                )
                for unit in run_units
            ]
            start = None
            last_tool = None
            gap_units = 0
            gap_user_messages = 0
            for idx, stat in enumerate(stats):
                if stat['has_tool']:
                    if start is None:
                        start = idx
                    last_tool = idx
                    gap_units = 0
                    gap_user_messages = 0
                    continue
                if start is None:
                    continue
                gap_units += 1
                gap_user_messages += self._user_messages_for_unit(
                    run_units[idx],
                    settings=settings,
                    provider=provider,
                    sequence_token_cache=sequence_token_cache,
                )
                if (gap_units > max_gap_units or gap_user_messages > max_gap_user_messages) and last_tool is not None:
                    candidate = self._toolspan_slice_from_unit_range(run_units, stats, start, last_tool, max_tokens=max_span_tokens, min_tokens=min_tokens, ratio_threshold=ratio_threshold)
                    if candidate:
                        return candidate
                    start = None
                    last_tool = None
                    gap_units = 0
                    gap_user_messages = 0
            if start is not None and last_tool is not None:
                candidate = self._toolspan_slice_from_unit_range(run_units, stats, start, last_tool, max_tokens=max_span_tokens, min_tokens=min_tokens, ratio_threshold=ratio_threshold)
                if candidate:
                    return candidate
        return []

    def _user_messages_for_unit(
        self,
        unit: list[StoredConversationMessage],
        *,
        settings: SessionSettings,
        provider: ModelProvider,
        sequence_token_cache: dict[tuple[int, ...], int] | None = None,
    ) -> int:
        if any(
            self._stored_message_tool_profile(
                item,
                settings=settings,
                provider=provider,
                sequence_token_cache=sequence_token_cache,
            )['has_tool']
            for item in unit
        ):
            return 0
        return sum(1 for item in unit if item.message.role == MessageRole.USER and not self._is_auto_note_message(item.message))

    def _toolspan_slice_from_unit_range(self, units: list[list[StoredConversationMessage]], stats: list[dict[str, float | int | bool]], start_idx: int, end_idx: int, *, max_tokens: int, min_tokens: int, ratio_threshold: float) -> list[StoredConversationMessage]:
        if start_idx > end_idx:
            return []
        chosen_units = units[start_idx:end_idx + 1]
        chosen_stats = stats[start_idx:end_idx + 1]
        total_tokens = sum(int(item['total_tokens']) for item in chosen_stats)
        if total_tokens <= 0:
            return []
        while len(chosen_units) > 1 and total_tokens > max_tokens:
            last = chosen_units.pop()
            last_stat = chosen_stats.pop()
            if any(bool(s['has_tool']) for s in chosen_stats):
                total_tokens -= int(last_stat['total_tokens'])
            else:
                chosen_units.append(last)
                chosen_stats.append(last_stat)
                break
        tool_tokens = sum(int(item['tool_tokens']) for item in chosen_stats)
        user_tokens = sum(int(item['user_tokens']) for item in chosen_stats)
        total_tokens = sum(int(item['total_tokens']) for item in chosen_stats)
        if total_tokens < min_tokens:
            return []
        tool_ratio = float(tool_tokens) / float(max(1, user_tokens))
        if tool_ratio < ratio_threshold:
            return []
        flattened = [item for unit in chosen_units for item in unit]
        while flattened and flattened[0].message.role == MessageRole.USER and not self._is_tool_call_message(flattened[0].message):
            flattened.pop(0)
        while flattened and flattened[-1].message.role == MessageRole.USER and not self._is_tool_call_message(flattened[-1].message):
            flattened.pop()
        return flattened
    def _toolspan_unit_stats(
        self,
        unit: list[StoredConversationMessage],
        *,
        settings: SessionSettings,
        provider: ModelProvider,
        sequence_token_cache: dict[tuple[int, ...], int] | None = None,
    ) -> dict[str, float | int | bool]:
        total_tokens = 0
        tool_tokens = 0
        user_tokens = 0
        assistant_tokens = 0
        other_tokens = 0
        has_tool = False
        for item in unit:
            profile = self._stored_message_tool_profile(
                item,
                settings=settings,
                provider=provider,
                sequence_token_cache=sequence_token_cache,
            )
            total_tokens += int(profile['total_tokens'])
            tool_tokens += int(profile['tool_tokens'])
            user_tokens += int(profile['user_tokens'])
            assistant_tokens += int(profile['assistant_tokens'])
            other_tokens += int(profile['other_tokens'])
            has_tool = has_tool or bool(profile['has_tool'])
        return {
            'total_tokens': total_tokens,
            'tool_tokens': tool_tokens,
            'user_tokens': user_tokens,
            'assistant_tokens': assistant_tokens,
            'other_tokens': other_tokens,
            'has_tool': has_tool,
        }
    def _stored_message_tool_profile(
        self,
        item: StoredConversationMessage,
        *,
        settings: SessionSettings,
        provider: ModelProvider,
        sequence_token_cache: dict[tuple[int, ...], int] | None = None,
    ) -> dict[str, int | bool]:
        total_tokens = self._estimate_stored_messages_prompt_tokens(
            [item],
            settings=settings,
            provider=provider,
            sequence_token_cache=sequence_token_cache,
        )
        has_tool = item.message.role == MessageRole.TOOL or self._message_has_tool_context(item.message)
        role = item.message.role
        tool_tokens = total_tokens if has_tool else 0
        user_tokens = total_tokens if role == MessageRole.USER and not has_tool else 0
        assistant_tokens = total_tokens if role == MessageRole.ASSISTANT and not has_tool else 0
        other_tokens = total_tokens if role not in {MessageRole.USER, MessageRole.ASSISTANT} and not has_tool else 0
        return {
            'total_tokens': total_tokens,
            'tool_tokens': tool_tokens,
            'user_tokens': user_tokens,
            'assistant_tokens': assistant_tokens,
            'other_tokens': other_tokens,
            'has_tool': has_tool,
        }
    def _history_entry_for_raw_unit(
        self,
        unit: list[StoredConversationMessage],
        excluded_message_ids: set[int],
        *,
        settings: SessionSettings,
        provider: ModelProvider,
        sequence_token_cache: dict[tuple[int, ...], int] | None = None,
    ) -> dict[str, Any]:
        return {
            'kind': 'raw',
            'messages': unit,
            'block': None,
            'sort_key': (int(unit[-1].db_id), 1),
            'estimated_tokens': self._estimate_stored_messages_prompt_tokens(
                unit,
                settings=settings,
                provider=provider,
                sequence_token_cache=sequence_token_cache,
            ),
            'source_message_count': len(unit),
            'excluded': any(item.db_id in excluded_message_ids for item in unit),
        }

    def _history_entry_for_l0_block(self, block: MemoryBlock, excluded_block_ids: set[int]) -> dict[str, Any]:
        return {
            'kind': 'l0',
            'messages': [],
            'block': block,
            'sort_key': self._history_position_for_block(block),
            'estimated_tokens': int(block.estimated_tokens),
            'source_message_count': int(block.source_message_count),
            'excluded': block.block_id in excluded_block_ids,
        }

    def _eligible_history_entries(
        self,
        state: LiveConversationState,
        *,
        settings: SessionSettings,
        provider: ModelProvider,
        pressure: bool = False,
        excluded_message_ids: set[int] | None = None,
        excluded_block_ids: set[int] | None = None,
        sequence_token_cache: dict[tuple[int, ...], int] | None = None,
    ) -> list[dict[str, Any]]:
        excluded_message_ids = excluded_message_ids or set()
        excluded_block_ids = excluded_block_ids or set()
        older_units, protected_units = self._protected_recent_raw_units(
            state.raw_messages,
            settings=settings,
            provider=provider,
            pressure=pressure,
            sequence_token_cache=sequence_token_cache,
        )
        protected_start_id = protected_units[0][0].db_id if protected_units else None
        entries: list[dict[str, Any]] = [
            self._history_entry_for_raw_unit(
                unit,
                excluded_message_ids,
                settings=settings,
                provider=provider,
                sequence_token_cache=sequence_token_cache,
            )
            for unit in older_units
        ]
        for block in state.blocks:
            if block.lifecycle != 'sealed' or block.kind != 'toolspan' or block.level != 0:
                continue
            if protected_start_id is not None and block.end_message_id is not None and block.end_message_id >= protected_start_id:
                continue
            entries.append(self._history_entry_for_l0_block(block, excluded_block_ids))
        return sorted(entries, key=lambda item: item['sort_key'])

    @staticmethod
    def _split_history_entries_by_exclusions(entries: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        runs: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        for entry in entries:
            if entry.get('excluded'):
                if current:
                    runs.append(current)
                    current = []
                continue
            current.append(entry)
        if current:
            runs.append(current)
        return runs

    def _select_oldest_history_slice(
        self,
        state: LiveConversationState,
        *,
        settings: SessionSettings,
        provider: ModelProvider,
        pressure: bool = False,
        excluded_message_ids: set[int] | None = None,
        excluded_block_ids: set[int] | None = None,
        sequence_token_cache: dict[tuple[int, ...], int] | None = None,
    ) -> dict[str, Any] | None:
        entries = self._eligible_history_entries(
            state,
            settings=settings,
            provider=provider,
            pressure=pressure,
            excluded_message_ids=excluded_message_ids,
            excluded_block_ids=excluded_block_ids,
            sequence_token_cache=sequence_token_cache,
        )
        if not entries:
            return None
        min_messages = self._effective_compact_min_messages(settings)
        total_source_messages = sum(int(entry['source_message_count']) for entry in entries)
        if not pressure and total_source_messages < min_messages:
            return None
        batch_limit = self._effective_compact_batch_tokens(settings)
        if pressure:
            batch_limit = int(batch_limit * 2.0)
        runs = self._split_history_entries_by_exclusions(entries)
        for run in runs:
            if not run:
                continue
            selected: list[dict[str, Any]] = []
            selected_tokens = 0
            for entry in run:
                unit_tokens = int(entry['estimated_tokens'])
                if selected and selected_tokens + unit_tokens > batch_limit:
                    break
                selected.append(entry)
                selected_tokens += unit_tokens
                if selected_tokens >= batch_limit:
                    break
            if not selected and pressure:
                selected = [run[0]]
            if not selected:
                continue
            raw_messages = [item for entry in selected if entry['kind'] == 'raw' for item in entry['messages']]
            parent_blocks = [entry['block'] for entry in selected if entry['kind'] == 'l0' and entry['block'] is not None]
            source_messages: list[ConversationMessage] = []
            for entry in selected:
                if entry['kind'] == 'raw':
                    source_messages.extend(item.message for item in entry['messages'])
                elif entry['block'] is not None:
                    source_messages.append(entry['block'].render_as_message())
            if not source_messages:
                continue
            start_candidates = [item.db_id for item in raw_messages]
            start_candidates.extend(int(block.start_message_id) for block in parent_blocks if block.start_message_id is not None)
            end_candidates = [item.db_id for item in raw_messages]
            end_candidates.extend(int(block.end_message_id) for block in parent_blocks if block.end_message_id is not None)
            return {
                'source_messages': source_messages,
                'raw_messages': raw_messages,
                'parent_blocks': parent_blocks,
                'source_message_count': sum(int(entry['source_message_count']) for entry in selected),
                'start_message_id': min(start_candidates) if start_candidates else None,
                'end_message_id': max(end_candidates) if end_candidates else None,
            }
        return None

    def _group_raw_compaction_units(self, messages: list[StoredConversationMessage]) -> list[list[StoredConversationMessage]]:
        units: list[list[StoredConversationMessage]] = []
        index = 0
        while index < len(messages):
            current = messages[index]
            if self._is_tool_call_message(current.message) and index + 1 < len(messages) and self._is_matching_tool_result(current.message, messages[index + 1].message):
                units.append([current, messages[index + 1]])
                index += 2
                continue
            if self._is_auto_note_message(current.message) and index + 1 < len(messages) and messages[index + 1].message.role == MessageRole.USER and not self._is_auto_note_message(messages[index + 1].message):
                units.append([current, messages[index + 1]])
                index += 2
                continue
            units.append([current])
            index += 1
        return units

    @staticmethod
    def _is_auto_note_message(message: ConversationMessage) -> bool:
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        synthetic_role = str(metadata.get('synthetic_role') or '').strip().lower()
        return synthetic_role == 'auto_user_note' or any((part.origin or '') == 'auto_note' for part in message.parts)

    def _select_digest_shard(self, blocks: list[MemoryBlock], *, settings: SessionSettings, pressure: bool = False, excluded_parent_signatures: set[tuple[int, ...]] | None = None) -> list[MemoryBlock]:
        episodes = [
            block for block in blocks
            if block.lifecycle == 'sealed'
            and (
                (block.kind == 'episode' and block.level == 1)
                or (block.kind == 'digest' and block.level == 2)
            )
        ] # this could compact L2 + L1
        minimum_size = 3 if pressure else 4
        if len(episodes) < minimum_size:
            return []
        candidate_episodes = episodes[:-2] if len(episodes) > 5 else episodes[:-1]
        if len(candidate_episodes) < minimum_size:
            return []
        covered_ids: set[int] = set()
        for block in blocks:
            if block.kind == 'digest' and block.lifecycle == 'sealed':
                covered_ids.update(int(parent) for parent in block.parent_block_ids)
        if len(candidate_episodes) >= 12:
            shard_size = 8
        elif len(candidate_episodes) >= 8:
            shard_size = 6
        elif len(candidate_episodes) >= 5:
            shard_size = 4
        else:
            shard_size = minimum_size
        excluded_parent_signatures = excluded_parent_signatures or set()
        run: list[MemoryBlock] = []
        runs: list[list[MemoryBlock]] = []
        for episode in candidate_episodes:
            if episode.block_id in covered_ids:
                if run:
                    runs.append(run)
                    run = []
                continue
            run.append(episode)
        if run:
            runs.append(run)
        for run in runs:
            max_size = min(shard_size, len(run))
            min_size = minimum_size if not pressure else max(2, minimum_size)
            for size in range(max_size, min_size - 1, -1):
                for start in range(0, len(run) - size + 1):
                    shard = run[start:start + size]
                    signature = tuple(block.block_id for block in shard)
                    if signature in excluded_parent_signatures:
                        continue
                    return shard
        return []

    async def _make_toolspan_block_candidate(self, provider: ModelProvider, settings: SessionSettings, raw_messages: list[StoredConversationMessage]) -> dict[str, Any] | None:
        source_messages = [item.message for item in raw_messages]
        normalized = self._normalize_compaction_messages(source_messages)
        time_start, time_end = self._raw_message_time_bounds(raw_messages, settings.metadata_timezone)
        metadata_message = self._compaction_metadata_message(
            mode='toolspan',
            raw_messages=raw_messages,
            parent_blocks=[],
            time_start=time_start,
            time_end=time_end,
        )
        model_messages = [metadata_message, *normalized] if metadata_message is not None else normalized
        candidate = await self._generate_structured_candidate(provider, settings, model_messages, mode='toolspan')
        if candidate is None:
            return None
        return {
            'data': candidate,
            'score': 1.0,
            'topic_labels': self._candidate_topic_labels(candidate),
            'actor_labels': self._candidate_actor_labels(candidate),
            'time_start': time_start,
            'time_end': time_end,
        }

    async def _make_episode_block_candidate(
        self,
        provider: ModelProvider,
        settings: SessionSettings,
        source_messages: list[ConversationMessage],
        raw_messages: list[StoredConversationMessage],
        parent_blocks: list[MemoryBlock],
    ) -> dict[str, Any] | None:
        normalized = self._normalize_compaction_messages(source_messages)
        raw_start, raw_end = self._raw_message_time_bounds(raw_messages, settings.metadata_timezone)
        block_start, block_end = self._block_time_bounds(parent_blocks)
        time_start = min((value for value in (raw_start, block_start) if value), default=None)
        time_end = max((value for value in (raw_end, block_end) if value), default=None)
        metadata_message = self._compaction_metadata_message(
            mode='episode',
            raw_messages=raw_messages,
            parent_blocks=parent_blocks,
            time_start=time_start,
            time_end=time_end,
        )
        model_messages = [metadata_message, *normalized] if metadata_message is not None else normalized
        candidate = await self._generate_structured_candidate(provider, settings, model_messages, mode='episode')
        if candidate is None:
            return None
        #candidate.setdefault('parent_l0_refs', [f'L0#{block.sequence_no}' for block in parent_blocks if block.kind == 'toolspan' and block.level == 0])
        return {
            'data': candidate,
            'score': 1.0,
            'topic_labels': self._candidate_topic_labels(candidate),
            'actor_labels': self._candidate_actor_labels(candidate),
            'time_start': time_start,
            'time_end': time_end,
        }

    async def _make_digest_block_candidate(self, provider: ModelProvider, settings: SessionSettings, blocks: list[MemoryBlock]) -> dict[str, Any] | None:
        source_messages = [block.render_as_message() for block in blocks]
        time_start, time_end = self._block_time_bounds(blocks)
        metadata_message = self._compaction_metadata_message(
            mode='digest',
            raw_messages=[],
            parent_blocks=blocks,
            time_start=time_start,
            time_end=time_end,
        )
        model_messages = [metadata_message, *source_messages] if metadata_message is not None else source_messages
        candidate = await self._generate_structured_candidate(provider, settings, model_messages, mode='digest')
        if candidate is None:
            return None
        candidate.setdefault('parent_refs', [f'L{block.level}#{block.sequence_no}' for block in blocks])
        return {
            'data': candidate,
            'score': 1.0,
            'topic_labels': self._candidate_topic_labels(candidate),
            'actor_labels': self._candidate_actor_labels(candidate),
            'time_start': time_start,
            'time_end': time_end,
        }

    async def _generate_structured_candidate(self, provider: ModelProvider, settings: SessionSettings, messages: list[ConversationMessage], *, mode: str) -> dict[str, Any] | None:
        compaction_settings = replace(settings, mode=ChatMode.CHAT)
        try:
            response = await provider.generate(
                settings=compaction_settings,
                messages=messages,
                instructions=build_compaction_prompt(mode=mode),
                tools=[],
                extra_input_items=None,
                response_schema=compaction_json_schema(mode),
                response_schema_name=compaction_schema_name(mode),
            )
            return self._parse_candidate_json(response.final_text or '', mode=mode)
        except Exception as exc:
            logger.exception('compact.model_failed provider=%s mode=%s err=%s', provider.name, mode, exc.__class__.__name__)
            raise CompactionModelRequestFailed(provider_name=provider.name, mode=mode) from exc

    @staticmethod
    def _raw_message_time_bounds(messages: list[StoredConversationMessage], timezone: str) -> tuple[str | None, str | None]:
        values = sorted(item.created_at for item in messages if item.created_at)
        if not values:
            return None, None
        try:
            tz = ZoneInfo(timezone or "UTC")
        except ZoneInfoNotFoundError:
            tz = ZoneInfo('UTC')
        start = datetime.fromtimestamp(values[0], tz=tz).isoformat()
        end = datetime.fromtimestamp(values[-1], tz=tz).isoformat()
        return start, end

    def _compaction_metadata_message(
        self,
        *,
        mode: str,
        raw_messages: list[StoredConversationMessage],
        parent_blocks: list[MemoryBlock],
        time_start: str | None,
        time_end: str | None,
    ) -> ConversationMessage:
        participants: list[str] = []
        role_names: dict[MessageRole, str] = {
            MessageRole.USER: 'user',
            MessageRole.ASSISTANT: 'assistant',
            MessageRole.TOOL: 'tool',
            MessageRole.SYSTEM: 'system',
        }
        for item in raw_messages:
            label = role_names.get(item.message.role, item.message.role.value)
            if label not in participants:
                participants.append(label)
        for block in parent_blocks:
            for label in block.actor_labels:
                label_text = str(label).strip()
                if label_text and label_text not in participants:
                    participants.append(label_text)
        lines = ['[Compaction source metadata]']
        lines.append(f'- mode: {mode}')
        lines.append(f'- raw_message_count: {len(raw_messages)}')
        lines.append(f'- parent_block_count: {len(parent_blocks)}')
        if time_start and time_end and time_start != time_end:
            lines.append(f'- time_span: {time_start} .. {time_end}')
        elif time_start or time_end:
            lines.append(f'- time_span: {time_start or time_end}')
        if participants:
            lines.append('- participants: ' + ', '.join(participants[:8]))
        if mode == 'toolspan':
            lines.append('- preserve request context, assistant strategy, ordered tool actions, outcomes, and remaining open loops')
        elif mode == 'episode':
            lines.append('- preserve request context, meaningful tool usage, chronology, outcomes, decisions, open loops, and durable user profile')
            #parent_l0_refs = [f'L0#{block.sequence_no}' for block in parent_blocks if block.kind == 'toolspan' and block.level == 0]
            #if parent_l0_refs:
            #    lines.append('- parent_l0_refs: ' + ', '.join(parent_l0_refs))
            #else:
            #    lines.append('- parent_l0_refs: none')
        else:
            parent_refs = [f'L{block.level}#{block.sequence_no}' for block in parent_blocks]
            if parent_refs:
                lines.append('- parent_refs: ' + ', '.join(parent_refs[:12]))
            lines.append('- reconcile repeated goals, durable state, important changes, decisions, and still-open loops')
        return ConversationMessage.assistant_text('\n'.join(lines), metadata={'source_role': 'compaction_metadata'})

    @staticmethod
    def _parse_candidate_json(text: str, *, mode: str) -> dict[str, Any] | None:
        stripped = text.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except Exception:
            return None
        if not isinstance(parsed, dict):
            return None
        try:
            return parse_structured_candidate(mode, parsed)
        except Exception:
            return None

    @staticmethod
    def _candidate_topic_labels(candidate: dict[str, Any], key: str = 'topics') -> list[str]:
        value = candidate.get(key, [])
        if not isinstance(value, list):
            return []
        labels: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in labels:
                labels.append(text[:48])
            if len(labels) >= 8:
                break
        return labels
    @staticmethod
    def _candidate_actor_labels(candidate: dict[str, Any]) -> list[str]:
        value = candidate.get('participants', [])
        if not isinstance(value, list):
            return []
        labels: list[str] = []
        for item in value:
            text = str(item).strip()
            if text and text not in labels:
                labels.append(text[:48])
            if len(labels) >= 8:
                break
        return labels
    def _render_memory_block_text(self, kind: str, data: dict[str, Any], *, time_start: str | None = None, time_end: str | None = None) -> str:
        def emit_list(title: str, items: Any, default: str | None = None) -> list[str]:
            lines = ['', title]
            values = items if isinstance(items, list) else []
            if values:
                for item in values[:12]:
                    lines.append(f'- {str(item).strip()}')
            elif default is not None:
                lines.append(f'- {default}')
            return lines

        def time_line(start: str | None, end: str | None) -> str | None:
            if start and end and start != end:
                return f'- Time span: {start} .. {end}'
            if start or end:
                return f'- Time span: {start or end}'
            return None

        lines: list[str] = []
        if kind == 'toolspan':
            lines.append('## Scope')
            lines.append(f'- {data.get("scope") or "Tool-heavy interaction span"}')
            span = time_line(time_start, time_end)
            if span:
                lines.append(span)
            if data.get('interaction_mode'):
                lines.append('- Interaction mode: ' + str(data.get('interaction_mode')).strip())
            if data.get('participants'):
                lines.append('- Participants: ' + ', '.join(str(item).strip() for item in data.get('participants', [])[:8]))
            if data.get('topics'):
                lines.append('- Topics: ' + ', '.join(str(item).strip() for item in data.get('topics', [])[:8]))
            lines.extend(emit_list('## User profile', data.get('user_profile', []), 'None recorded'))
            lines.extend(emit_list('## User intent / shared context', data.get('user_intent_or_shared_context', []), 'None recorded'))
            lines.extend(emit_list('## Why it mattered', data.get('why_it_mattered', []), 'None recorded'))
            lines.extend(emit_list('## Assistant strategy', data.get('assistant_strategy', []), 'None recorded'))
            lines.extend(emit_list('## Tool timeline', data.get('tool_timeline', []), 'None recorded'))
            lines.extend(emit_list('## Results / takeaways', data.get('results_or_takeaways', []), 'None recorded'))
            lines.extend(emit_list('## Decisions', data.get('decisions', []), 'None recorded'))
            lines.extend(emit_list('## Open loops', data.get('open_loops', []), 'None recorded'))
            lines.extend(emit_list('## Artifacts', data.get('artifacts', []), 'None recorded'))
            lines.extend(emit_list('## Uncertainties', data.get('uncertainties', []), 'None recorded'))
            excerpts = data.get('retained_raw_excerpts', []) if isinstance(data.get('retained_raw_excerpts', []), list) else []
            if excerpts:
                lines.extend(emit_list('## Retained raw excerpts', excerpts))
        elif kind == 'episode':
            lines.append('## Scope')
            lines.append(f'- {data.get("scope") or "Episode summary"}')
            span = time_line(time_start, time_end)
            if span:
                lines.append(span)
            if data.get('interaction_mode'):
                lines.append('- Interaction mode: ' + str(data.get('interaction_mode')).strip())
            if data.get('participants'):
                lines.append('- Participants: ' + ', '.join(str(item).strip() for item in data.get('participants', [])[:8]))
            if data.get('topics'):
                lines.append('- Topics: ' + ', '.join(str(item).strip() for item in data.get('topics', [])[:8]))
            lines.extend(emit_list('## User profile', data.get('user_profile', []), 'None recorded'))
            lines.extend(emit_list('## User intent / shared context', data.get('user_intent_or_shared_context', []), 'None recorded'))
            lines.extend(emit_list('## Why it mattered', data.get('why_it_mattered', []), 'None recorded'))
            lines.extend(emit_list('## Tool usage', data.get('tool_usage', []), 'None recorded'))
            lines.extend(emit_list('## Interaction timeline', data.get('interaction_timeline', []), 'None recorded'))
            lines.extend(emit_list('## Results / takeaways', data.get('results_or_takeaways', []), 'None recorded'))
            lines.extend(emit_list('## Decisions', data.get('decisions', []), 'None recorded'))
            lines.extend(emit_list('## Open loops', data.get('open_loops', []), 'None recorded'))
            lines.extend(emit_list('## Artifacts', data.get('artifacts', []), 'None recorded'))
            lines.extend(emit_list('## Uncertainties', data.get('uncertainties', []), 'None recorded'))
            excerpts = data.get('retained_raw_excerpts', []) if isinstance(data.get('retained_raw_excerpts', []), list) else []
            if excerpts:
                lines.extend(emit_list('## Retained raw excerpts', excerpts))
        else:
            lines.append('## Scope')
            lines.append(f'- {data.get("scope") or "Digest summary"}')
            span = time_line(time_start, time_end)
            if span:
                lines.append(span)
            if data.get('interaction_modes_seen'):
                lines.append('- Interaction modes seen: ' + ', '.join(str(item).strip() for item in data.get('interaction_modes_seen', [])[:8]))
            if data.get('participants'):
                lines.append('- Participants: ' + ', '.join(str(item).strip() for item in data.get('participants', [])[:8]))
            if data.get('topics'):
                lines.append('- Topics: ' + ', '.join(str(item).strip() for item in data.get('topics', [])[:8]))
            lines.extend(emit_list('## User profile', data.get('user_profile', []), 'None recorded'))
            lines.extend(emit_list('## Recurring requests / shared threads', data.get('recurring_requests_or_shared_threads', []), 'None recorded'))
            lines.extend(emit_list('## Why history matters now', data.get('why_history_matters_now', []), 'None recorded'))
            lines.extend(emit_list('## Durable state', data.get('durable_state', []), 'None recorded'))
            lines.extend(emit_list('## Important changes', data.get('important_changes', []), 'None recorded'))
            lines.extend(emit_list('## Decisions', data.get('decisions', []), 'None recorded'))
            lines.extend(emit_list('## Open loops', data.get('open_loops', []), 'None recorded'))
            lines.extend(emit_list('## Artifacts', data.get('artifacts', []), 'None recorded'))
            lines.extend(emit_list('## Uncertainties', data.get('uncertainties', []), 'None recorded'))
            lines.extend(emit_list('## Parent refs', data.get('parent_refs', []), 'None recorded'))
        return '\n'.join(line for line in lines if line is not None).strip()
    async def _summarize_messages(
        self,
        provider: ModelProvider,
        settings: SessionSettings,
        messages: list[ConversationMessage],
        *,
        already_compacted: bool = False,
    ) -> str:
        mode = 'digest' if already_compacted else 'episode'
        normalized_messages = self._normalize_compaction_messages(messages)
        candidate = await self._generate_structured_candidate(provider, settings, normalized_messages, mode=mode)
        if isinstance(candidate, dict):
            return self._render_memory_block_text(mode, candidate)
        raise RuntimeError(f'Compaction summary generation failed for mode={mode}')
    def _normalize_compaction_messages(self, messages: list[ConversationMessage]) -> list[ConversationMessage]:
        normalized: list[ConversationMessage] = []
        index = 0
        while index < len(messages):
            message = messages[index]
            if self._is_tool_call_message(message) and index + 1 < len(messages):
                next_message = messages[index + 1]
                if self._is_matching_tool_result(message, next_message):
                    text = self._normalize_tool_pair(message, next_message)
                    if text:
                        normalized.append(ConversationMessage.assistant_text(text, metadata={'source_role': 'tool'}))
                    index += 2
                    continue
            single = self._normalize_compaction_message(message)
            if single is not None:
                normalized.append(single)
            index += 1
        return normalized

    @staticmethod
    def _is_tool_call_message(message: ConversationMessage) -> bool:
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        return message.role == MessageRole.TOOL and str(metadata.get('tool_phase') or '').strip().lower() == 'call'

    @staticmethod
    def _is_matching_tool_result(call_message: ConversationMessage, result_message: ConversationMessage) -> bool:
        call_meta = call_message.metadata if isinstance(call_message.metadata, dict) else {}
        result_meta = result_message.metadata if isinstance(result_message.metadata, dict) else {}
        if result_message.role != MessageRole.TOOL or str(result_meta.get('tool_phase') or '').strip().lower() != 'result':
            return False
        call_payload = call_meta.get('tool_payload') if isinstance(call_meta.get('tool_payload'), dict) else {}
        result_payload = result_meta.get('tool_payload') if isinstance(result_meta.get('tool_payload'), dict) else {}
        call_id = str(call_payload.get('call_id') or '')
        result_id = str(result_payload.get('call_id') or '')
        if call_id and result_id:
            return call_id == result_id
        return (call_message.name or '') == (result_message.name or '')

    def _normalize_compaction_message(self, message: ConversationMessage) -> ConversationMessage | None:
        if message.role == MessageRole.TOOL:
            return self._normalize_single_tool_message(message)
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        synthetic_role = str(metadata.get('synthetic_role') or '').strip().lower()
        is_auto_note = synthetic_role == 'auto_user_note'
        if is_auto_note:
            text = self._normalize_auto_note_message(message)
            return ConversationMessage.assistant_text(text, metadata={'source_role': 'transport'}) if text else None
        text = self._normalize_regular_message_text(message)
        if not text:
            return None
        if message.role == MessageRole.USER:
            return ConversationMessage.user_text(text, metadata={'source_role': 'user'})
        return ConversationMessage.assistant_text(text, metadata={'source_role': message.role.value})

    def _normalize_regular_message_text(self, message: ConversationMessage) -> str:
        text_parts: list[str] = []
        attachment_parts: list[str] = []
        auto_note_parts: list[MessagePart] = []
        for part in message.parts:
            if (part.origin or '').strip().lower() == 'auto_note':
                auto_note_parts.append(part)
                continue
            if part.kind == PartKind.TEXT:
                text = (part.text or '').strip()
                if text:
                    #if text.startswith('[Compacted memory block #') or text.startswith('[Memory '):
                    #    _, _, remainder = text.partition("\n")
                    #    text = remainder.strip() or text
                    text_parts.append(text)
            else:
                attachment_parts.append(self._describe_attachment_part(part))
        reasoning_summaries = self._message_reasoning_summaries(message)
        lines: list[str] = []
        transport_text = self._normalize_auto_note_parts(auto_note_parts)
        if transport_text:
            lines.append(transport_text)
        if text_parts:
            lines.append(' '.join(text_parts))
        for summary in reasoning_summaries[:2]:
            lines.append('Reasoning summary: ' + self._clip_inline(summary, 260))
        if attachment_parts:
            lines.append('Attachments: ' + '; '.join(attachment_parts))
        return "\n".join(line for line in lines if line).strip()

    def _message_reasoning_summaries(self, message: ConversationMessage) -> list[str]:
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        provider_native = metadata.get('provider_native') if isinstance(metadata.get('provider_native'), dict) else {}
        items = provider_native.get('items') if isinstance(provider_native.get('items'), list) else []
        summaries: list[str] = []
        for item in items:
            if not isinstance(item, dict) or str(item.get('type') or '').strip().lower() != 'reasoning':
                continue
            summary_items = item.get('summary') if isinstance(item.get('summary'), list) else []
            for summary_item in summary_items:
                if not isinstance(summary_item, dict):
                    continue
                text = str(summary_item.get('text') or '').strip()
                if text:
                    summaries.append(text)
        return summaries

    def _message_has_tool_context(self, message: ConversationMessage) -> bool:
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        if str(metadata.get('tool_phase') or '').strip().lower() in {'call', 'result'}:
            return True
        provider_native = metadata.get('provider_native') if isinstance(metadata.get('provider_native'), dict) else {}
        items = provider_native.get('items') if isinstance(provider_native.get('items'), list) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get('type') or '').strip().lower()
            if item_type in {'function_call', 'function_call_output'}:
                return True
        return False

    def _normalize_auto_note_message(self, message: ConversationMessage) -> str:
        return self._normalize_auto_note_parts(message.parts)

    def _normalize_auto_note_parts(self, parts: list[MessagePart]) -> str:
        texts: list[str] = []
        for part in parts:
            if part.kind != PartKind.TEXT:
                continue
            text = (part.text or '').strip()
            if not text or text == '[Auto-generated bot message, do not reply.]':
                continue
            texts.append(text)
        return '\n'.join(texts).strip()

    def _normalize_tool_pair(self, call_message: ConversationMessage, result_message: ConversationMessage) -> str:
        call_meta = call_message.metadata if isinstance(call_message.metadata, dict) else {}
        result_meta = result_message.metadata if isinstance(result_message.metadata, dict) else {}
        call_payload = call_meta.get('tool_payload') if isinstance(call_meta.get('tool_payload'), dict) else {}
        result_payload = result_meta.get('tool_payload') if isinstance(result_meta.get('tool_payload'), dict) else {}
        name = call_message.name or result_message.name or 'tool'
        visible_text = self._provider_native_visible_text(call_message)
        action = self._describe_tool_call(name, call_payload)
        outcome = self._describe_tool_result(name, result_payload)
        summary = ''
        if action and outcome:
            summary = f'{action}. Result: {outcome}'
        else:
            summary = action or outcome
        if visible_text and summary and visible_text not in summary:
            return f'{visible_text}\n\n{summary}'
        return visible_text or summary

    def _normalize_single_tool_message(self, message: ConversationMessage) -> ConversationMessage | None:
        metadata = message.metadata if isinstance(message.metadata, dict) else {}
        payload = metadata.get('tool_payload') if isinstance(metadata.get('tool_payload'), dict) else {}
        phase = str(metadata.get('tool_phase') or '').strip().lower()
        name = message.name or 'tool'
        visible_text = self._provider_native_visible_text(message)
        if phase == 'call':
            text = self._describe_tool_call(name, payload)
        elif phase == 'result':
            text = self._describe_tool_result(name, payload)
        elif phase == 'delivery':
            text = self._describe_tool_delivery(name, payload)
        else:
            text = self._normalize_regular_message_text(message)
        if visible_text and text and visible_text not in text:
            text = f'{visible_text}\n\n{text}'
        elif visible_text and not text:
            text = visible_text
        return ConversationMessage.assistant_text(text, metadata={'source_role': 'tool'}) if text else None

    def _describe_tool_call(self, name: str, payload: dict[str, Any]) -> str:
        arguments = payload.get('arguments') if isinstance(payload.get('arguments'), dict) else {}
        details: list[str] = []
        if name == 'shell_exec':
            command = self._clip_inline(arguments.get('command'), 220)
            if command:
                details.append(f'ran shell command {command!r}')
        elif name == 'python_exec':
            code = self._clip_inline(arguments.get('code'), 180)
            if code:
                details.append(f'ran Python code {code!r}')
        elif name == 'sticker_query':
            details.extend(self._describe_sticker_query_call(arguments))
        elif name == 'sticker_send_selected':
            details.extend(self._describe_sticker_send_call(arguments))
        elif arguments:
            details.append(f'called with arguments {self._clip_inline(self._compact_json(arguments, limit=180), 180)!r}')
        cwd = arguments.get('cwd_subdir')
        timeout_s = arguments.get('timeout_s')
        if cwd:
            details.append(f'cwd_subdir={cwd}')
        if timeout_s:
            details.append(f'timeout={timeout_s}s')
        prefix = f'Tool {name}'
        return prefix + (': ' + '; '.join(details) if details else ' invoked')

    def _describe_tool_result(self, name: str, payload: dict[str, Any]) -> str:
        output = payload.get('output') if isinstance(payload.get('output'), dict) else {}
        details: list[str] = []
        if output:
            if name == 'sticker_query':
                details.extend(self._describe_sticker_query_result(output))
            elif name == 'sticker_send_selected':
                details.extend(self._describe_sticker_send_result(output))
            else:
                if 'ok' in output:
                    details.append(f"ok={bool(output.get('ok'))}")
                if output.get('returncode') is not None:
                    details.append(f"returncode={output.get('returncode')}")
                stdout = self._clip_inline(output.get('stdout'), 180)
                stderr = self._clip_inline(output.get('stderr'), 180)
                if stdout:
                    details.append(f'stdout={stdout!r}')
                if stderr:
                    details.append(f'stderr={stderr!r}')
        elif payload:
            details.append(self._clip_inline(self._compact_json(payload, limit=220), 220))
        prefix = f'Tool {name} result'
        return prefix + (': ' + '; '.join(details) if details else ' recorded')

    def _describe_tool_delivery(self, name: str, payload: dict[str, Any]) -> str:
        if name == 'sticker_send':
            details = self._describe_sticker_delivery(payload)
            prefix = f'Tool {name} delivery'
            return prefix + (': ' + '; '.join(details) if details else ' recorded')
        payload_text = self._clip_inline(self._compact_json(payload, limit=220), 220)
        prefix = f'Tool {name} delivery'
        return prefix + (': ' + payload_text if payload_text else ' recorded')

    def _describe_sticker_query_call(self, arguments: dict[str, Any]) -> list[str]:
        details: list[str] = []
        intent = self._clip_inline(arguments.get('intent_core'), 96)
        if intent:
            details.append(f'query stickers for intent={intent!r}')
        tone = self._clip_inline(arguments.get('reaction_tone', arguments.get('emotion_tone')), 60)
        if tone:
            details.append(f'tone={tone!r}')
        social = self._clip_inline(arguments.get('social_intent', arguments.get('social_goal')), 60)
        if social:
            details.append(f'social={social!r}')
        expression = self._clip_inline(arguments.get('expression_cue', arguments.get('visual_hint')), 60)
        if expression:
            details.append(f'expression={expression!r}')
        caption = self._clip_inline(arguments.get('caption_meaning', arguments.get('text_hint')), 60)
        if caption:
            details.append(f'caption={caption!r}')
        preferred_pack = self._clip_inline(arguments.get('preferred_pack', arguments.get('prefer_pack')), 48)
        if preferred_pack:
            details.append(f'pack={preferred_pack!r}')
        preferred_cluster = self._clip_inline(arguments.get('preferred_style_cluster', arguments.get('prefer_cluster')), 48)
        if preferred_cluster:
            details.append(f'cluster={preferred_cluster!r}')
        candidate_budget = arguments.get('candidate_budget')
        if candidate_budget is not None:
            details.append(f'candidate_budget={candidate_budget}')
        return details

    def _describe_sticker_send_call(self, arguments: dict[str, Any]) -> list[str]:
        details: list[str] = []
        sticker_id = self._clip_inline(arguments.get('selected_sticker_id', arguments.get('sticker_id')), 48)
        if sticker_id:
            details.append(f'send sticker_id={sticker_id!r}')
        delivery_timing = self._clip_inline(arguments.get('delivery_timing', arguments.get('timing')), 32)
        if delivery_timing:
            details.append(f'delivery_timing={delivery_timing}')
        return details

    def _describe_sticker_query_result(self, output: dict[str, Any]) -> list[str]:
        details: list[str] = []
        details.append(f"ok={bool(output.get('ok'))}")
        if not output.get('ok'):
            error = self._clip_inline(output.get('error'), 120)
            if error:
                details.append(f'error={error!r}')
            culprit_fields = output.get('likely_culprit_fields') if isinstance(output.get('likely_culprit_fields'), list) else []
            if culprit_fields:
                details.append('likely_culprit_fields=' + ', '.join(str(item) for item in culprit_fields[:4]))
            return details
        candidates = output.get('candidates') if isinstance(output.get('candidates'), list) else []
        details.append(f'candidate_count={len(candidates)}')
        if output.get('field_warnings'):
            warnings = output.get('field_warnings') if isinstance(output.get('field_warnings'), list) else []
            if warnings:
                details.append(f'field_warning={self._clip_inline(warnings[0], 100)!r}')
        top_candidates: list[str] = []
        for candidate in candidates[:2]:
            if not isinstance(candidate, dict):
                continue
            top_candidates.append(self._describe_sticker_candidate(candidate))
        if top_candidates:
            details.append('top_candidates=' + ' | '.join(top_candidates))
        return details

    def _describe_sticker_send_result(self, output: dict[str, Any]) -> list[str]:
        details: list[str] = []
        details.append(f"ok={bool(output.get('ok'))}")
        if not output.get('ok'):
            error = self._clip_inline(output.get('error'), 120)
            if error:
                details.append(f'error={error!r}')
            return details
        sticker_id = self._clip_inline(output.get('sticker_id'), 48)
        if sticker_id:
            details.append(f'sticker_id={sticker_id!r}')
        status = self._clip_inline(output.get('status'), 32)
        if status:
            details.append(f'status={status}')
        delivery_timing = self._clip_inline(output.get('delivery_timing', output.get('timing')), 32)
        if delivery_timing:
            details.append(f'delivery_timing={delivery_timing}')
        pack_id = self._clip_inline(output.get('source_pack_id'), 48)
        if pack_id:
            details.append(f'pack={pack_id!r}')
        cluster_id = self._clip_inline(output.get('style_cluster'), 48)
        if cluster_id:
            details.append(f'cluster={cluster_id!r}')
        style_summary = self._clip_inline(output.get('style_summary'), 80)
        if style_summary:
            details.append(f'style={style_summary!r}')
        semantic_summary = self._clip_inline(output.get('semantic_summary', output.get('social_read')), 80)
        if semantic_summary:
            details.append(f'semantic={semantic_summary!r}')
        return details

    def _describe_sticker_delivery(self, payload: dict[str, Any]) -> list[str]:
        details: list[str] = []
        sticker_id = self._clip_inline(payload.get('sticker_id'), 48)
        if sticker_id:
            details.append(f'sticker_id={sticker_id!r}')
        label = self._clip_inline(payload.get('sticker_label'), 60)
        if label:
            details.append(f'label={label!r}')
        delivery_timing = self._clip_inline(payload.get('delivery_timing'), 32)
        if delivery_timing:
            details.append(f'delivery_timing={delivery_timing}')
        delivery_state = self._clip_inline(payload.get('delivery_state'), 32)
        if delivery_state:
            details.append(f'state={delivery_state}')
        if payload.get('sent') is not None:
            details.append(f"sent={bool(payload.get('sent'))}")
        telegram_message_id = payload.get('telegram_message_id')
        if telegram_message_id is not None:
            details.append(f'telegram_message_id={telegram_message_id}')
        error = self._clip_inline(payload.get('error'), 60)
        if error:
            details.append(f'error={error!r}')
        return details

    def _describe_sticker_candidate(self, candidate: dict[str, Any]) -> str:
        sticker_id = self._clip_inline(candidate.get('sticker_id'), 40) or '?'
        pack_id = self._clip_inline(candidate.get('source_pack_id'), 32) or '-'
        cluster_id = self._clip_inline(candidate.get('style_cluster'), 32) or '-'
        style_summary = self._clip_inline(candidate.get('style_summary'), 56) or '-'
        semantic_summary = self._clip_inline(candidate.get('semantic_summary', candidate.get('social_read') or candidate.get('summary')), 72) or '-'
        return f'{sticker_id} pack={pack_id} cluster={cluster_id} style={style_summary!r} semantic={semantic_summary!r}'

    @staticmethod
    def _describe_attachment_part(part: MessagePart) -> str:
        bits: list[str] = []
        if part.filename:
            bits.append(part.filename)
        elif part.kind != PartKind.TEXT:
            bits.append(part.kind.value)
        if part.mime_type:
            bits.append(part.mime_type)
        if part.size_bytes is not None:
            bits.append(f'{part.size_bytes} bytes')
        if part.artifact_path:
            bits.append(f'path={part.artifact_path}')
        return ' '.join(bits).strip() or part.kind.value

    @staticmethod
    def _clip_inline(value: Any, limit: int) -> str:
        text = str(value or '').replace('\n', ' ').strip()
        if len(text) > limit:
            return text[: limit - 3] + '...'
        return text




