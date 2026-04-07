from __future__ import annotations

import asyncio
import logging
import random
import contextlib
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pathlib import Path

from telegram import Chat, Message, Update
from telegram.constants import ChatAction, ChatType
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from tgchatbot.config import AppConfig
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import (
    ChatMode,
    ConversationMessage,
    MessagePart,
    MessageRole,
    PartKind,
    ProcessVisibility,
    PromptInjectionMode,
    ResponseDelivery,
    SessionSettings,
    StickerMode,
    StickerTiming,
    ToolHistoryMode,
    TurnResult,
    OutboundSticker,
)
from tgchatbot.logging_config import clip_for_log
from tgchatbot.media.ingest import extract_message_parts
from tgchatbot.media.link_prefetch import fetch_link_previews, previews_to_parts
from tgchatbot.storage.artifacts import ArtifactStore
from tgchatbot.storage.sqlite_store import SQLiteStore
from tgchatbot.storage.presets import PresetStore
from tgchatbot.tools.remote_workspace import RemoteWorkspaceClient
from tgchatbot.settings_schema import (
    COMPACT_TOOL_RATIO_THRESHOLD_MAX,
    COMPACT_TOOL_RATIO_THRESHOLD_MIN,
    GEMINI_THINKING_BUDGET_MAX,
    GEMINI_THINKING_BUDGET_MIN,
    GROUP_SPONTANEOUS_REPLY_DELAY_MAX_S,
    IMAGE_LIMIT_MAX,
    MAX_INTERACTION_ROUNDS_MAX,
    MAX_INTERACTION_ROUNDS_MIN,
    MAX_OUTPUT_TOKENS_MAX,
    MAX_OUTPUT_TOKENS_MIN,
    NATIVE_WEB_SEARCH_MAX_MAX,
    NATIVE_WEB_SEARCH_MAX_MIN,
    PROVIDER_RETRY_COUNT_MAX,
    PROVIDER_RETRY_COUNT_MIN,
    REPLY_DELAY_MAX_S,
    REASONING_SUMMARY_VALUES,
    SPONTANEOUS_REPLY_CHANCE_MAX,
    SPONTANEOUS_REPLY_CHANCE_MIN,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    TOP_K_MAX,
    TOP_K_MIN,
    TOP_P_MAX,
    TOP_P_MIN,
    effective_reasoning_summary,
    format_optional_disabled_int,
    gemini_allowed_thinking_levels,
    gemini_supports_thinking,
    gemini_thinking_budget_is_valid,
    gemini_thinking_budget_usage,
    normalize_optional_disabled_int,
)
from tgchatbot.transports.telegram_render import MAX_TELEGRAM_TEXT_CHARS, TelegramMessageRenderer, _chunk_text_for_telegram, bot_message_safe

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ReplyCandidate:
    stored_message_id: int
    user_display_name: str
    source_message: Message
    spontaneous: bool = False


@dataclass(slots=True)
class GroupReplyPlan:
    should_reply: bool
    explicit: bool


@dataclass(slots=True)
class ChatFlowState:
    mutex: asyncio.Lock = field(default_factory=asyncio.Lock)
    ingest_idle: asyncio.Event = field(default_factory=asyncio.Event)
    ingest_inflight: int = 0
    latest_reply_token: int = 0
    latest_reply_candidate: ReplyCandidate | None = None
    last_replied_message_id: int = 0
    reply_task: asyncio.Task[None] | None = None


class TelegramBotApp:
    @staticmethod
    def _chat_log_id(chat_id: int) -> str:
        return str(chat_id)

    @staticmethod
    def _message_preview(text: str, *, limit: int = 120) -> str:
        return clip_for_log(text, limit=limit)

    @staticmethod
    def _log_parts_preview(parts: list[MessagePart], limit: int = 160) -> str:
        text_bits: list[str] = []
        file_bits: list[str] = []

        for part in parts:
            if part.kind == PartKind.TEXT:
                text = (part.text or "").strip()
                if text:
                    text_bits.append(text)
            elif part.kind == PartKind.FILE:
                file_bits.append(part.filename or "unnamed-file")

        preview = " | ".join(text_bits)
        if file_bits:
            files = ",".join(file_bits[:3])
            if len(file_bits) > 3:
                files += f"+{len(file_bits) - 3}"
            preview = f"{preview} [files={files}]" if preview else f"[files={files}]"

        if not preview:
            return "<empty>"
        return clip_for_log(preview, limit=limit, rlimit=limit)

    @staticmethod
    def _parts_summary(parts: list[MessagePart]) -> str:
        counts: dict[str, int] = {}
        for part in parts:
            counts[part.kind.value] = counts.get(part.kind.value, 0) + 1
        return ','.join(f'{kind}:{counts[kind]}' for kind in sorted(counts)) or 'none'

    @staticmethod
    def _safe_user_error_text(stage: str, exc: Exception) -> str:
        raw = f'{exc.__class__.__name__}: {exc}'
        text = clip_for_log(raw, limit=220)
        lowered = text.lower()
        if 'api_key' in lowered or 'authorization' in lowered or 'bearer ' in lowered:
            text = exc.__class__.__name__
        return f'⚠️ {stage} failed.\n{text}'

    def __init__(self, *, config: AppConfig, runtime: AgentRuntime, store: SQLiteStore, artifact_store: ArtifactStore, preset_store: PresetStore | None = None, remote_workspace: RemoteWorkspaceClient | None = None) -> None:
        self.config = config
        self.runtime = runtime
        self.store = store
        self.artifact_store = artifact_store
        self.preset_store = preset_store or PresetStore(config.preset_dir)
        self.remote_workspace = remote_workspace
        self.application = Application.builder().token(config.telegram.token).build()
        self._chat_states: dict[int, ChatFlowState] = {}
        self._register_handlers()

    def _provider_reasoning_effort_default(self, provider_name: str) -> str:
        return self.config.openai.reasoning_effort

    def _provider_reasoning_summary_default(self, provider_name: str) -> str:
        return self.config.openai.reasoning_summary

    def _provider_text_verbosity_default(self, provider_name: str) -> str:
        return self.config.openai.text_verbosity

    def _gemini_include_thoughts_default(self) -> bool:
        return bool(self.config.gemini.include_thoughts)

    def _gemini_thinking_budget_default(self) -> int | None:
        return self.config.gemini.thinking_budget

    def _gemini_thinking_level_default(self) -> str | None:
        return self.config.gemini.thinking_level

    def _provider_native_web_search_default(self, provider_name: str) -> str:
        if provider_name == 'openai':
            return 'on' if self.config.openai.enable_native_web_search else 'off'
        return 'on' if self.config.gemini.enable_native_web_search else 'off'

    @staticmethod
    def _display_optional_disabled_int(value: int | None, *, disabled_label: str, maximum: int) -> str:
        normalized = normalize_optional_disabled_int(value, maximum=maximum)
        if normalized in {None, 0}:
            return disabled_label
        return format_optional_disabled_int(normalized, disabled_label=disabled_label)

    def _stored_native_web_search_max_default(self) -> int:
        return int(self.config.openai.native_web_search_max)

    def _stored_temperature_default(self) -> float:
        return float(self.config.gemini.temperature)

    def _stored_top_p_default(self) -> float:
        return float(self.config.gemini.top_p)

    def _stored_top_k_default(self) -> int:
        return int(self.config.gemini.top_k)

    def _compact_trigger_tokens_default(self) -> int:
        return int(self.config.context.compact_trigger_tokens)

    def _compact_target_tokens_default(self) -> int:
        return int(self.config.context.compact_target_tokens)

    def _compact_batch_tokens_default(self) -> int:
        return int(self.config.context.compact_batch_tokens)

    def _compact_keep_recent_ratio_default(self) -> float:
        return float(self.config.context.compact_keep_recent_ratio)

    def _compact_tool_ratio_threshold_default(self) -> float:
        return float(self.config.context.compact_tool_ratio_threshold)

    def _compact_tool_min_tokens_default(self) -> int:
        return int(self.config.context.compact_tool_min_tokens)

    @staticmethod
    def _parse_ratio_value(value: str) -> float | None:
        raw = value.strip().lower()
        try:
            if raw.endswith('%'):
                return float(raw[:-1]) / 100.0
            parsed = float(raw)
        except ValueError:
            return None
        if parsed > 1.0:
            parsed = parsed / 100.0
        if parsed < 0.0 or parsed > 0.95:
            return None
        return parsed

    def _compact_min_messages_default(self) -> int:
        return int(self.config.context.compact_min_messages)

    def _min_raw_messages_reserve_default(self) -> int:
        return int(self.config.context.min_raw_messages_reserve)

    def _param_usage_lines(self, settings: SessionSettings) -> list[str]:
        lines = ['Usage: /param <name> <value|default>']
        if settings.provider == 'openai':
            lines.extend([
                'reasoning_effort <none|minimal|low|medium|high|xhigh|default>',
                'reasoning_summary <off|on|auto|detailed|concise|default>',
                'text_verbosity <low|medium|high|default>',
                'native_web_search <on|off|default>',
                f'native_web_search_max <{NATIVE_WEB_SEARCH_MAX_MIN}..{NATIVE_WEB_SEARCH_MAX_MAX}|default>  (0 disables the explicit cap)',
            ])
        else:
            lines.extend([
                'include_thoughts <on|off|default>',
                'native_web_search <on|off|default>',
                f'temperature <{TEMPERATURE_MIN:g}..{TEMPERATURE_MAX:g}|default>',
                f'top_p <{TOP_P_MIN:g}..{TOP_P_MAX:g}|default>',
                f'top_k <{TOP_K_MIN}..{TOP_K_MAX}|default>',
            ])
            allowed_thinking_levels = gemini_allowed_thinking_levels(settings.model)
            if allowed_thinking_levels:
                lines.append(f"thinking_level <{'|'.join(allowed_thinking_levels)}|default>")
                lines.append(
                    f'thinking_budget <{GEMINI_THINKING_BUDGET_MIN}..{GEMINI_THINKING_BUDGET_MAX}|default>  '
                    '(legacy Gemini 3 fallback; ignored when thinking_level is set)'
                )
            elif gemini_supports_thinking(settings.model):
                lines.append(f'thinking_budget <{gemini_thinking_budget_usage(settings.model)}|default>')
        lines.extend([
            'link_prefetch <off|title|snippet|default>',
            f'max_output_tokens <{MAX_OUTPUT_TOKENS_MIN}..{MAX_OUTPUT_TOKENS_MAX}|default>',
            f'max_input_images <0..{IMAGE_LIMIT_MAX}|default>  (0 disables the image-count cap)',
            f'compact_target_images <0..{IMAGE_LIMIT_MAX}|default>  (0 disables the separate image compaction target)',
            'compact_trigger_tokens <256..10000000|default>',
            'compact_target_tokens <256..10000000|default>',
            'compact_batch_tokens <256..10000000|default>',
            'compact_keep_recent_ratio <0..0.95 | 50% | default>',
            'compact_tool_ratio_threshold <1..100 | default>',
            'compact_tool_min_tokens <256..10000000|default>',
            'compact_min_messages <2..1000|default>',
            'min_raw_messages_reserve <0..1000|default>',
            f'max_interaction_rounds <{MAX_INTERACTION_ROUNDS_MIN}..{MAX_INTERACTION_ROUNDS_MAX}|default>',
            f'spontaneous_reply_chance <{SPONTANEOUS_REPLY_CHANCE_MIN}..{SPONTANEOUS_REPLY_CHANCE_MAX}|default>',
            f'group_spontaneous_reply_delay_s <0..{int(GROUP_SPONTANEOUS_REPLY_DELAY_MAX_S)}|default>',
            f'private_reply_delay_s <0..{int(REPLY_DELAY_MAX_S)}|default>',
            f'group_reply_delay_s <0..{int(REPLY_DELAY_MAX_S)}|default>',
            f'provider_retry_count <{PROVIDER_RETRY_COUNT_MIN}..{PROVIDER_RETRY_COUNT_MAX}|default>',
            'metadata <on|off|default>',
            'metadata_timezone <IANA TZ like UTC or Asia/Tokyo|default>',
            'tool_history_mode <translated|native_same_provider|default>',
        ])
        return lines

    def _param_lines(self, session_status: dict[str, object], *, include_help: bool) -> list[str]:
        lines = ['Session parameters']
        if session_status['provider'] == 'openai':
            lines.extend([
                f"reasoning_effort={session_status['reasoning_effort']} ({session_status['reasoning_effort_source']}) supported={session_status['reasoning_effort_supported']}",
                f"reasoning_summary={session_status['reasoning_summary']} ({session_status['reasoning_summary_source']}) supported={session_status['reasoning_summary_supported']}",
                f"text_verbosity={session_status['text_verbosity']} ({session_status['text_verbosity_source']}) supported={session_status['text_verbosity_supported']}",
                f"native_web_search={session_status['native_web_search']} ({session_status['native_web_search_source']}) supported={session_status['native_web_search_supported']}",
                f"native_web_search_max={session_status['native_web_search_max']} ({session_status['native_web_search_max_source']}) supported={session_status['native_web_search_max_supported']}",
            ])
        else:
            lines.extend([
                f"include_thoughts={session_status['include_thoughts']} ({session_status['include_thoughts_source']}) supported={session_status['include_thoughts_supported']}",
                f"thinking_budget={session_status['thinking_budget']} ({session_status['thinking_budget_source']}) supported={session_status['thinking_budget_supported']}",
                f"thinking_level={session_status['thinking_level']} ({session_status['thinking_level_source']}) supported={session_status['thinking_level_supported']}",
                f"native_web_search={session_status['native_web_search']} ({session_status['native_web_search_source']}) supported={session_status['native_web_search_supported']}",
            ])
        lines.extend([
            f"temperature={session_status['temperature']} ({session_status['temperature_source']}) supported={session_status['temperature_supported']}",
            f"top_p={session_status['top_p']} ({session_status['top_p_source']}) supported={session_status['top_p_supported']}",
            f"top_k={session_status['top_k']} ({session_status['top_k_source']}) supported={session_status['top_k_supported']}",
            f"link_prefetch={session_status['link_prefetch_mode']} ({session_status['link_prefetch_mode_source']})",
            f"max_output_tokens={session_status['max_output_tokens']} ({session_status['max_output_tokens_source']}) supported={session_status['max_output_tokens_supported']}",
            f"max_input_images={session_status['max_input_images']} ({session_status['max_input_images_source']})",
            f"compact_target_images={session_status['compact_target_images']} ({session_status['compact_target_images_source']})",
            f"compact_trigger_tokens={session_status['compact_trigger_tokens']} ({session_status['compact_trigger_tokens_source']})",
            f"compact_target_tokens={session_status['compact_target_tokens']} ({session_status['compact_target_tokens_source']})",
            f"compact_batch_tokens={session_status['compact_batch_tokens']} ({session_status['compact_batch_tokens_source']})",
            f"compact_keep_recent_ratio={session_status['compact_keep_recent_ratio']} ({session_status['compact_keep_recent_ratio_source']})",
            f"compact_tool_ratio_threshold={session_status['compact_tool_ratio_threshold']} ({session_status['compact_tool_ratio_threshold_source']})",
            f"compact_tool_min_tokens={session_status['compact_tool_min_tokens']} ({session_status['compact_tool_min_tokens_source']})",
            f"(legacy) compact_min_messages={session_status['compact_min_messages']} ({session_status['compact_min_messages_source']}) min_raw_messages_reserve={session_status['min_raw_messages_reserve']} ({session_status['min_raw_messages_reserve_source']})",
            f"max_interaction_rounds={session_status['max_interaction_rounds']} ({session_status['max_interaction_rounds_source']})",
            f"spontaneous_reply_chance={session_status['spontaneous_reply_chance']}% ({session_status['spontaneous_reply_chance_source']})",
            f"group_spontaneous_reply_delay_s={session_status['group_spontaneous_reply_delay_s']} ({session_status['group_spontaneous_reply_delay_s_source']})",
            f"provider_retry_count={session_status['provider_retry_count']} ({session_status['provider_retry_count_source']})",
            f"private_reply_delay_s={session_status['private_reply_delay_s']} ({session_status['private_reply_delay_s_source']}) group_reply_delay_s={session_status['group_reply_delay_s']} ({session_status['group_reply_delay_s_source']})",
            f"metadata={session_status['metadata_injection_mode']} ({session_status['metadata_injection_mode_source']})",
            f"metadata_timezone={session_status['metadata_timezone']} ({session_status['metadata_timezone_source']})",
            f"prompt_injection={session_status['prompt_injection_mode']}",
            f"tool_history_mode={session_status['tool_history_mode']}",
        ])
        if include_help:
            lines.append('')
            lines.append('Set with /param <name> <value|default>')
            #lines.extend(self._param_usage_lines()[1:])
        return lines

    def _status_lines(self, session_status: dict[str, object], flow: dict[str, object], *, full: bool, include_param_help: bool) -> list[str]:
        lines = [
            'Session',
            f"provider={session_status['provider']} model={session_status['model']}",
            f"mode={session_status['mode']} process={session_status['process']} delivery={session_status['delivery']} stickers={session_status['stickers']}",
            f"raw_messages={session_status['raw_messages']} memory_blocks={session_status['memory_blocks']} l0_blocks={session_status['l0_blocks']} l1_blocks={session_status['l1_blocks']} l2_blocks={session_status['l2_blocks']} provider_history_messages={session_status['provider_history_messages']}",
            f"estimated_request_tokens={session_status['estimated_request_tokens']} estimated_request_images={session_status['estimated_request_images']}",
            f"compact_trigger_tokens={session_status['compact_trigger_tokens']} compact_target_tokens={session_status['compact_target_tokens']}",
            f"reply_running={flow['reply_running']} ingest_inflight={flow['ingest_inflight']}",
            f"prompt_chars={session_status['system_prompt_chars']}",
        ]
        if not full:
            lines.append('')
            lines.append('Use [/status full] for the original detailed status and current /params values.')
            return lines
        lines.extend([
            '',
            'Context',
            f"raw_messages={session_status['raw_messages']} tool_history_messages={session_status['tool_history_messages']} memory_blocks={session_status['memory_blocks']} l0_blocks={session_status['l0_blocks']} l1_blocks={session_status['l1_blocks']} l2_blocks={session_status['l2_blocks']} provider_history_messages={session_status['provider_history_messages']}",
            f"estimated_history_tokens={session_status['estimated_history_tokens']} estimated_request_tokens={session_status['estimated_request_tokens']}",
            f"estimated_request_images={session_status['estimated_request_images']} max_input_images={session_status['max_input_images']} compact_target_images={session_status['compact_target_images']}",
            f"compact_trigger_tokens={session_status['compact_trigger_tokens']} compact_target_tokens={session_status['compact_target_tokens']} compact_batch_tokens={session_status['compact_batch_tokens']} compact_keep_recent_ratio={session_status['compact_keep_recent_ratio']} compact_tool_ratio_threshold={session_status['compact_tool_ratio_threshold']} compact_tool_min_tokens={session_status['compact_tool_min_tokens']} compact_min_messages={session_status['compact_min_messages']} min_raw_messages_reserve={session_status['min_raw_messages_reserve']}",
            f"loaded_in_memory={session_status['loaded_in_memory']}",
            '',
            'Sticker index',
            f"loaded={session_status['sticker_index_loaded']} stickers={session_status['sticker_index_count']} packs={session_status['sticker_pack_count']}",
            '',
            'Remote + queue',
            f"remote_enabled={session_status['remote_enabled']} remote_master_ready={session_status['remote_master_ready']}",
            f"ingest_inflight={flow['ingest_inflight']} reply_running={flow['reply_running']}",
            f"latest_pending_reply_message_id={flow['latest_pending_reply_message_id']} last_replied_message_id={flow['last_replied_message_id']}",
            '',
        ])
        lines.extend(self._param_lines(session_status, include_help=include_param_help))
        return lines

    def _register_handlers(self) -> None:
        self.application.add_handler(CommandHandler('start', self.start_command))
        self.application.add_handler(CommandHandler('help', self.help_command))
        self.application.add_handler(CommandHandler('reset', self.reset_command))
        self.application.add_handler(CommandHandler('mode', self.mode_command))
        self.application.add_handler(CommandHandler('process', self.process_command))
        self.application.add_handler(CommandHandler('delivery', self.delivery_command))
        self.application.add_handler(CommandHandler('stickers', self.stickers_command))
        self.application.add_handler(CommandHandler('status', self.status_command))
        self.application.add_handler(CommandHandler('presets', self.presets_command))
        self.application.add_handler(CommandHandler('preset', self.preset_command))
        self.application.add_handler(CommandHandler('prompt', self.prompt_command))
        self.application.add_handler(CommandHandler('provider', self.provider_command))
        self.application.add_handler(CommandHandler('model', self.model_command))
        self.application.add_handler(CommandHandler('params', self.params_command))
        self.application.add_handler(CommandHandler('param', self.param_command))
        self.application.add_handler(CommandHandler('retry', self.retry_command))
        self.application.add_handler(CommandHandler('rollback', self.rollback_command))
        all_messages = (filters.TEXT | filters.PHOTO | filters.Sticker.ALL | filters.Document.ALL | filters.VIDEO | filters.ANIMATION | filters.CAPTION) & ~filters.COMMAND
        fresh_messages = filters.UpdateType.MESSAGE & all_messages
        edited_messages = filters.UpdateType.EDITED_MESSAGE & all_messages
        self.application.add_handler(MessageHandler(fresh_messages & filters.ChatType.PRIVATE, self.private_message))
        self.application.add_handler(MessageHandler(fresh_messages & filters.ChatType.GROUPS, self.group_message))
        self.application.add_handler(MessageHandler(edited_messages & filters.ChatType.PRIVATE, self.private_edited_message))
        self.application.add_handler(MessageHandler(edited_messages & filters.ChatType.GROUPS, self.group_edited_message))

    def run_polling(self) -> None:
        logger.info('telegram.polling.start whitelist=%s keywords=%s', len(self.config.telegram.whitelist), len(self.config.telegram.keywords))
        self.application.run_polling()

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        if not chat:
            return
        if not self._allowed(chat):
            await update.effective_message.reply_text('[BOT] Whitelist restricted.')
            return
        settings = await self.store.get_or_create_session(self._session_id(chat), self._default_settings())
        await update.effective_message.reply_text(
            'Started.\n'
            f'provider={settings.provider} model={settings.model}\n'
            f'mode={settings.mode.value} process={settings.process_visibility.value} delivery={settings.response_delivery.value} stickers={settings.sticker_mode.value}\n\n'
            'Use /help to see commands and controls.'
        )


    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        if not chat or not self._allowed(chat):
            return
        settings = await self.store.get_or_create_session(self._session_id(chat), self._default_settings())
        lines = [
            'Commands',
            '/status [full] - concise session/context view by default; use full for the detailed parameter block',
            f'/mode chat|assist|agent - current: {settings.mode.value}',
            f'/process off|minimal|status|verbose|full - current: {settings.process_visibility.value}',
            f'/delivery edit|final_new - current: {settings.response_delivery.value}',
            f'/stickers off|auto - current: {settings.sticker_mode.value}',
            f"/provider {'|'.join(sorted(self.runtime.providers.keys()))} - current: {settings.provider}",
            '/model <name>|default - set the exact model string for the current provider',
            '/params - show session tuning, defaults, and provider applicability',
            '/param <name> <value|default> - trusted users only; use /params for the full parameter list and value semantics',
            '/prompt - show prompt controls, including augment vs exact preset mode',
            '/preset <name> [augment|exact]|clear and /presets - manage prompt presets',
            '/reset [history|session|all] - hide history, reset session settings, or both',
            '/retry - regenerate from the latest visible user message after hiding newer assistant/tool output',
            '/rollback [count] - hide the last visible consecutive user/bot block(s) from session history',
            '',
       ]
        await update.effective_message.reply_text('\n'.join(lines))

    async def reset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        if not chat or not self._allowed(chat):
            return
        mode = (context.args[0].strip().lower() if context.args else '')
        if mode not in {'history', 'session', 'all'}:
            await update.effective_message.reply_text('Usage: /reset [history|session|all]\n[history] hides chat history only\n[session] resets session settings only\n[all] hides history and resets session settings to defaults')
            return
        if mode in {'history', 'all'}:
            await self.store.clear_messages(self._session_id(chat))
        if mode in {'session', 'all'}:
            await self.store.save_session(self._session_id(chat), self._default_settings())
        if mode in {'history', 'all'}:
            self.runtime.invalidate_session(self._session_id(chat))
            state = self._flow_state(chat.id)
            task: asyncio.Task[None] | None = None
            async with state.mutex:
                state.latest_reply_candidate = None
                state.last_replied_message_id = 0
                task = state.reply_task
                state.reply_task = None
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        note = 'Conversation history hidden.' if mode == 'history' else 'Session settings reset.' if mode == 'session' else 'Conversation history hidden and session settings reset to defaults.'
        await update.effective_message.reply_text(note)

    async def retry_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        message = update.effective_message
        if not chat or not message or not self._allowed(chat):
            return
        session_id = self._session_id(chat)
        recent = await self.store.list_recent_visible_messages(session_id, limit=40)
        trigger = next((item for item in recent if item.message.role == MessageRole.USER), None)
        if trigger is None:
            await message.reply_text('Nothing to retry: no visible user message found in this session.')
            return
        hidden = 0
        if recent and recent[0].db_id > trigger.db_id:
            hidden = await self.store.hide_messages_since(session_id, trigger.db_id + 1)
        self.runtime.invalidate_session(session_id)
        await self._cancel_pending_reply(chat.id)
        candidate = ReplyCandidate(
            stored_message_id=trigger.db_id,
            user_display_name=update.effective_user.username or update.effective_user.full_name or 'user',
            source_message=message,
        )
        await self._reply_to_candidate(candidate)
        if hidden > 0:
            await message.reply_text(f'Retried from message #{trigger.db_id}; hid {hidden} newer stored message(s) first.')

    async def rollback_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        message = update.effective_message
        if not chat or not message or not self._allowed(chat):
            return

        count_arg = context.args[0].strip() if context.args else '1'
        try:
            count = int(count_arg)
        except Exception:
            count = 0

        if count <= 0:
            await message.reply_text(
                'Usage: /rollback [count]\n'
                'Hides the last visible consecutive user/bot block(s). Example: /rollback 2'
            )
            return

        session_id = self._session_id(chat)
        target_ids = await self._collect_rollback_message_ids(session_id, count)
        if not target_ids:
            await message.reply_text('Nothing to roll back: session history is already empty.')
            return

        target_ids = sorted(set(target_ids))

        visible = await self.store.list_uncompacted_messages(session_id)
        by_id = {item.db_id: item for item in visible}
        targets = [by_id[mid] for mid in target_ids if mid in by_id]

        preview_lines: list[str] = []
        for item in targets[-10:]:
            preview_lines.append(
                f'- #{item.db_id} {self._rollback_item_type(item)}: '
                f'{self._rollback_item_preview(item)}'
            )

        hidden = await self.store.hide_message_ids(session_id, target_ids)
        self.runtime.invalidate_session(session_id)
        await self._cancel_pending_reply(chat.id)

        logger.info(
            'rollback.done sid=%s requested_blocks=%s hidden=%s ids=%s details=%s',
            session_id,
            count,
            hidden,
            clip_for_log(','.join(str(x) for x in target_ids), limit=180, rlimit=60),
            clip_for_log(' || '.join(preview_lines), limit=400, rlimit=120),
        )

        lines = [
            'Rollback complete.',
            f'- requested blocks: {count}',
            f'- hidden visible messages: {hidden}',
            f'- stored ids: {", ".join(str(x) for x in target_ids[:12])}'
            + (f' ... (+{len(target_ids) - 12} more)' if len(target_ids) > 12 else ''),
        ]
        if preview_lines:
            lines.append('')
            lines.append('Hidden messages:')
            lines.extend(preview_lines)

        await message.reply_text('\n'.join(lines))

    async def mode_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._command_allowed(update):
            return
        await self._update_session_enum(update, context, 'mode')

    async def process_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._command_allowed(update):
            return
        await self._update_session_enum(update, context, 'process')

    async def delivery_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._command_allowed(update):
            return
        await self._update_session_enum(update, context, 'delivery')

    async def stickers_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._command_allowed(update):
            return
        await self._update_session_enum(update, context, 'stickers')

    async def provider_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        if not chat or not self._allowed(chat):
            return
        settings = await self.store.get_or_create_session(self._session_id(chat), self._default_settings())
        value = (context.args[0].strip().lower() if context.args else '')
        available = set(self.runtime.providers.keys())
        if value not in available:
            available_text = '|'.join(sorted(available))
            await update.effective_message.reply_text(
                f'Usage: /provider {available_text}\n'
                f'Current provider: {settings.provider} (model={settings.model})'
            )
            return
        settings.provider = value
        settings.model = self.config.default_model_for_provider(value)
        await self.store.save_session(self._session_id(chat), settings)
        await update.effective_message.reply_text(f'Provider set to {value}.')

    async def model_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        if not chat or not self._allowed(chat):
            return
        settings = await self.store.get_or_create_session(self._session_id(chat), self._default_settings())
        value = (" ".join(context.args).strip() if context.args else '')
        if not value:
            await update.effective_message.reply_text(
                'Usage: /model <name>|default\n'
                f'Current provider/model: {settings.provider}/{settings.model}'
            )
            return
        if value.lower() == 'default':
            settings.model = self.config.default_model_for_provider(settings.provider)
        else:
            settings.model = value
        await self.store.save_session(self._session_id(chat), settings)
        await update.effective_message.reply_text(f'Model set to {settings.model} for provider {settings.provider}.')


    async def status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        if not chat or not self._allowed(chat):
            return
        session_status = await self.runtime.describe_session(self._session_id(chat))
        flow = await self._flow_snapshot(chat.id)
        full = bool(context.args and context.args[0].strip().lower() == 'full')
        lines = self._status_lines(session_status, flow, full=full, include_param_help=full)
        await update.effective_message.reply_text('\n'.join(lines))


    async def params_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        if not chat or not self._allowed(chat):
            return
        session_status = await self.runtime.describe_session(self._session_id(chat))
        lines = self._param_lines(session_status, include_help=True)
        if not self._advanced_allowed(update):
            lines.append('This sender is not allowed to change advanced session parameters.')
        await update.effective_message.reply_text('\n'.join(lines))

    async def param_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        if not chat or not self._allowed(chat):
            return
        if not self._advanced_allowed(update):
            await update.effective_message.reply_text('This sender is not allowed to change advanced session parameters.')
            return
        settings = await self.store.get_or_create_session(self._session_id(chat), self._default_settings())
        if len(context.args) < 2:
            await update.effective_message.reply_text('\n'.join(self._param_usage_lines(settings)))
            return
        name = context.args[0].strip().lower()
        value = context.args[1].strip().lower()
        if name == 'reasoning_effort':
            if settings.provider != 'openai':
                await update.effective_message.reply_text('reasoning_effort is only supported by the OpenAI provider in this app.')
                return
            allowed = {'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'default'}
            if value not in allowed:
                await update.effective_message.reply_text(f'Invalid reasoning_effort. Current effective value: {settings.reasoning_effort or self._provider_reasoning_effort_default(settings.provider)}')
                return
            settings.reasoning_effort = None if value == 'default' else value
        elif name == 'reasoning_summary':
            if settings.provider != 'openai':
                await update.effective_message.reply_text('reasoning_summary is only supported by the OpenAI provider in this app.')
                return
            allowed = set(REASONING_SUMMARY_VALUES) | {'default'}
            if value not in allowed:
                current = effective_reasoning_summary(
                    settings.reasoning_summary or self._provider_reasoning_summary_default(settings.provider),
                    provider=settings.provider,
                    default=self._provider_reasoning_summary_default(settings.provider),
                )
                await update.effective_message.reply_text(f'Invalid reasoning_summary. Current effective value: {current}')
                return
            settings.reasoning_summary = None if value == 'default' else value
        elif name == 'text_verbosity':
            if settings.provider != 'openai':
                await update.effective_message.reply_text('text_verbosity is only supported by the OpenAI provider in this app.')
                return
            allowed = {'low', 'medium', 'high', 'default'}
            if value not in allowed:
                await update.effective_message.reply_text(f'Invalid text_verbosity. Current effective value: {settings.text_verbosity or self._provider_text_verbosity_default(settings.provider)}')
                return
            settings.text_verbosity = None if value == 'default' else value
        elif name == 'include_thoughts':
            if settings.provider != 'gemini':
                await update.effective_message.reply_text('include_thoughts is only supported by the Gemini provider in this app.')
                return
            allowed = {'on', 'off', 'default'}
            if value not in allowed:
                current = 'on' if (settings.include_thoughts if settings.include_thoughts is not None else self._gemini_include_thoughts_default()) else 'off'
                await update.effective_message.reply_text(f'Invalid include_thoughts. Current effective value: {current}')
                return
            settings.include_thoughts = None if value == 'default' else (value == 'on')
        elif name == 'thinking_budget':
            if settings.provider != 'gemini':
                await update.effective_message.reply_text('thinking_budget is only supported by the Gemini provider in this app.')
                return
            if not gemini_supports_thinking(settings.model):
                await update.effective_message.reply_text(f'{settings.model} does not support Gemini thinking controls.')
                return
            if value == 'default':
                settings.thinking_budget = None
            else:
                try:
                    budget = int(value)
                except ValueError:
                    budget = GEMINI_THINKING_BUDGET_MIN - 1
                if not gemini_thinking_budget_is_valid(settings.model, budget):
                    current = settings.thinking_budget if settings.thinking_budget is not None else self._gemini_thinking_budget_default()
                    current_text = 'default' if current is None else str(current)
                    await update.effective_message.reply_text(
                        f'Invalid thinking_budget for {settings.model}. '
                        f'Allowed values: {gemini_thinking_budget_usage(settings.model)}. '
                        f'Current effective value: {current_text}'
                    )
                    return
                settings.thinking_budget = budget
                if settings.model.startswith('gemini-3'):
                    settings.thinking_level = None
        elif name == 'thinking_level':
            if settings.provider != 'gemini':
                await update.effective_message.reply_text('thinking_level is only supported by the Gemini provider in this app.')
                return
            allowed = gemini_allowed_thinking_levels(settings.model)
            if not allowed:
                await update.effective_message.reply_text(f'{settings.model} does not support thinking_level. Use thinking_budget instead.')
                return
            if value == 'default':
                settings.thinking_level = None
            elif value not in allowed:
                current = settings.thinking_level or self._gemini_thinking_level_default() or 'default'
                await update.effective_message.reply_text(
                    f"Invalid thinking_level for {settings.model}. Allowed values: {'|'.join(allowed)}. "
                    f'Current effective value: {current}'
                )
                return
            else:
                settings.thinking_level = value
                settings.thinking_budget = None
        elif name == 'native_web_search':
            allowed = {'on', 'off', 'default'}
            if value not in allowed:
                current = settings.native_web_search_mode if settings.native_web_search_mode != 'default' else self._provider_native_web_search_default(settings.provider)
                await update.effective_message.reply_text(f'Invalid native_web_search. Current effective value: {current}')
                return
            settings.native_web_search_mode = value
        elif name == 'native_web_search_max':
            if settings.provider != 'openai':
                await update.effective_message.reply_text('native_web_search_max is only supported by the OpenAI provider in this app.')
                return
            if value == 'default':
                settings.native_web_search_max = None
            else:
                try:
                    cap = int(value)
                except ValueError:
                    cap = -1
                if cap < NATIVE_WEB_SEARCH_MAX_MIN or cap > NATIVE_WEB_SEARCH_MAX_MAX:
                    current = settings.native_web_search_max if settings.native_web_search_max is not None else self._stored_native_web_search_max_default()
                    current_text = self._display_optional_disabled_int(current, disabled_label='unlimited', maximum=NATIVE_WEB_SEARCH_MAX_MAX)
                    await update.effective_message.reply_text(f'Invalid native_web_search_max. Current stored value: {current_text}')
                    return
                settings.native_web_search_max = cap
        elif name == 'temperature':
            if settings.provider != 'gemini':
                await update.effective_message.reply_text('temperature is only supported by the Gemini provider in this app.')
                return
            if value == 'default':
                settings.temperature = None
            else:
                try:
                    temp = float(value)
                except ValueError:
                    temp = -1.0
                if temp < TEMPERATURE_MIN or temp > TEMPERATURE_MAX:
                    current = settings.temperature if settings.temperature is not None else self._stored_temperature_default()
                    await update.effective_message.reply_text(f'Invalid temperature. Current stored value: {current}')
                    return
                settings.temperature = temp
        elif name == 'top_p':
            if settings.provider != 'gemini':
                await update.effective_message.reply_text('top_p is only supported by the Gemini provider in this app.')
                return
            if value == 'default':
                settings.top_p = None
            else:
                try:
                    top_p = float(value)
                except ValueError:
                    top_p = -1.0
                if top_p < TOP_P_MIN or top_p > TOP_P_MAX:
                    current = settings.top_p if settings.top_p is not None else self._stored_top_p_default()
                    await update.effective_message.reply_text(f'Invalid top_p. Current stored value: {current}')
                    return
                settings.top_p = top_p
        elif name == 'top_k':
            if settings.provider != 'gemini':
                await update.effective_message.reply_text('top_k is only supported by the Gemini provider in this app.')
                return
            if value == 'default':
                settings.top_k = None
            else:
                try:
                    top_k = int(value)
                except ValueError:
                    top_k = 0
                if top_k < TOP_K_MIN or top_k > TOP_K_MAX:
                    current = settings.top_k if settings.top_k is not None else self._stored_top_k_default()
                    await update.effective_message.reply_text(f'Invalid top_k. Current stored value: {current}')
                    return
                settings.top_k = top_k
        elif name == 'link_prefetch':
            allowed = {'off', 'title', 'snippet', 'default'}
            if value not in allowed:
                current = settings.link_prefetch_mode if settings.link_prefetch_mode != 'default' else self.config.default_link_prefetch_mode
                await update.effective_message.reply_text(f'Invalid link_prefetch. Current effective value: {current}')
                return
            settings.link_prefetch_mode = value
        elif name == 'max_output_tokens':
            if value == 'default':
                settings.max_output_tokens = None
            else:
                try:
                    max_tokens = int(value)
                except ValueError:
                    max_tokens = 0
                if max_tokens < MAX_OUTPUT_TOKENS_MIN or max_tokens > MAX_OUTPUT_TOKENS_MAX:
                    current = settings.max_output_tokens if settings.max_output_tokens is not None else (self.config.openai.max_output_tokens if settings.provider == 'openai' else self.config.gemini.max_output_tokens)
                    await update.effective_message.reply_text(f'Invalid max_output_tokens. Current effective value: {current}')
                    return
                settings.max_output_tokens = max_tokens
        elif name == 'max_input_images':
            if value == 'default':
                settings.max_input_images = None
            else:
                try:
                    image_limit = int(value)
                except ValueError:
                    image_limit = -1
                if image_limit < 0 or image_limit > IMAGE_LIMIT_MAX:
                    default_limit = self.config.openai.max_input_images if settings.provider == 'openai' else self.config.gemini.max_input_images
                    current = settings.max_input_images if settings.max_input_images is not None else default_limit
                    current_text = self._display_optional_disabled_int(current, disabled_label='unlimited', maximum=IMAGE_LIMIT_MAX)
                    await update.effective_message.reply_text(f'Invalid max_input_images. Current effective value: {current_text}')
                    return
                settings.max_input_images = image_limit
        elif name == 'compact_target_images':
            if value == 'default':
                settings.compact_target_images = None
            else:
                try:
                    target = int(value)
                except ValueError:
                    target = -1
                if target < 0 or target > IMAGE_LIMIT_MAX:
                    default_target = self.config.openai.compact_target_images if settings.provider == 'openai' else self.config.gemini.compact_target_images
                    current = settings.compact_target_images if settings.compact_target_images is not None else default_target
                    current_text = self._display_optional_disabled_int(current, disabled_label='disabled', maximum=IMAGE_LIMIT_MAX)
                    await update.effective_message.reply_text(f'Invalid compact_target_images. Current effective value: {current_text}')
                    return
                settings.compact_target_images = target
        elif name == 'compact_trigger_tokens':
            if value == 'default':
                settings.compact_trigger_tokens = None
            else:
                try:
                    target = int(value)
                except ValueError:
                    target = 0
                if target < 256 or target > 10000000:
                    current = settings.compact_trigger_tokens if settings.compact_trigger_tokens is not None else self._compact_trigger_tokens_default()
                    await update.effective_message.reply_text(f'Invalid compact_trigger_tokens. Current effective value: {current}')
                    return
                effective_target = settings.compact_target_tokens if settings.compact_target_tokens is not None else self._compact_target_tokens_default()
                if target < effective_target:
                    await update.effective_message.reply_text(f'compact_trigger_tokens must be >= compact_target_tokens ({effective_target}).')
                    return
                settings.compact_trigger_tokens = target
        elif name == 'compact_target_tokens':
            if value == 'default':
                settings.compact_target_tokens = None
            else:
                try:
                    target = int(value)
                except ValueError:
                    target = 0
                if target < 256 or target > 10000000:
                    current = settings.compact_target_tokens if settings.compact_target_tokens is not None else self._compact_target_tokens_default()
                    await update.effective_message.reply_text(f'Invalid compact_target_tokens. Current effective value: {current}')
                    return
                effective_trigger = settings.compact_trigger_tokens if settings.compact_trigger_tokens is not None else self._compact_trigger_tokens_default()
                if target > effective_trigger:
                    await update.effective_message.reply_text(f'compact_target_tokens must be <= compact_trigger_tokens ({effective_trigger}).')
                    return
                settings.compact_target_tokens = target
        elif name == 'compact_batch_tokens':
            if value == 'default':
                settings.compact_batch_tokens = None
            else:
                try:
                    target = int(value)
                except ValueError:
                    target = 0
                if target < 256 or target > 10000000:
                    current = settings.compact_batch_tokens if settings.compact_batch_tokens is not None else self._compact_batch_tokens_default()
                    await update.effective_message.reply_text(f'Invalid compact_batch_tokens. Current effective value: {current}')
                    return
                effective_target = settings.compact_target_tokens if settings.compact_target_tokens is not None else self._compact_target_tokens_default()
                if target > effective_target:
                    await update.effective_message.reply_text(f'compact_batch_tokens must be <= compact_target_tokens ({effective_target}).')
                    return
                settings.compact_batch_tokens = target
        elif name == 'compact_keep_recent_ratio':
            if value == 'default':
                settings.compact_keep_recent_ratio = None
            else:
                ratio = self._parse_ratio_value(value)
                if ratio is None:
                    current = settings.compact_keep_recent_ratio if settings.compact_keep_recent_ratio is not None else self._compact_keep_recent_ratio_default()
                    await update.effective_message.reply_text(f'Invalid compact_keep_recent_ratio. Current effective value: {current:.2f}')
                    return
                settings.compact_keep_recent_ratio = ratio
        elif name == 'compact_tool_ratio_threshold':
            if value == 'default':
                settings.compact_tool_ratio_threshold = None
            else:
                try:
                    ratio = float(value)
                except ValueError:
                    ratio = 0.0
                if ratio < 1.0 or ratio > 100.0:
                    current = settings.compact_tool_ratio_threshold if settings.compact_tool_ratio_threshold is not None else self._compact_tool_ratio_threshold_default()
                    await update.effective_message.reply_text(f'Invalid compact_tool_ratio_threshold. Current effective value: {current:.2f}')
                    return
                settings.compact_tool_ratio_threshold = ratio
        elif name == 'compact_tool_min_tokens':
            if value == 'default':
                settings.compact_tool_min_tokens = None
            else:
                try:
                    target = int(value)
                except ValueError:
                    target = 0
                if target < 256 or target > 10000000:
                    current = settings.compact_tool_min_tokens if settings.compact_tool_min_tokens is not None else self._compact_tool_min_tokens_default()
                    await update.effective_message.reply_text(f'Invalid compact_tool_min_tokens. Current effective value: {current}')
                    return
                settings.compact_tool_min_tokens = target
        elif name == 'compact_min_messages':
            if value == 'default':
                settings.compact_min_messages = None
            else:
                try:
                    count = int(value)
                except ValueError:
                    count = 0
                if count < 2 or count > 1000:
                    current = settings.compact_min_messages if settings.compact_min_messages is not None else self._compact_min_messages_default()
                    await update.effective_message.reply_text(f'Invalid compact_min_messages. Current effective value: {current}')
                    return
                reserve = settings.min_raw_messages_reserve if settings.min_raw_messages_reserve is not None else self._min_raw_messages_reserve_default()
                if reserve >= count:
                    await update.effective_message.reply_text(f'compact_min_messages must be > min_raw_messages_reserve ({reserve}).')
                    return
                settings.compact_min_messages = count
        elif name == 'min_raw_messages_reserve':
            if value == 'default':
                settings.min_raw_messages_reserve = None
            else:
                try:
                    count = int(value)
                except ValueError:
                    count = -1
                if count < 0 or count > 1000:
                    current = settings.min_raw_messages_reserve if settings.min_raw_messages_reserve is not None else self._min_raw_messages_reserve_default()
                    await update.effective_message.reply_text(f'Invalid min_raw_messages_reserve. Current effective value: {current}')
                    return
                compact_min_messages = settings.compact_min_messages if settings.compact_min_messages is not None else self._compact_min_messages_default()
                if count >= compact_min_messages:
                    await update.effective_message.reply_text(f'min_raw_messages_reserve must be < compact_min_messages ({compact_min_messages}).')
                    return
                settings.min_raw_messages_reserve = count
        elif name == 'max_interaction_rounds':
            if value == 'default':
                settings.max_interaction_rounds = None
            else:
                try:
                    rounds = int(value)
                except ValueError:
                    rounds = 0
                if rounds < MAX_INTERACTION_ROUNDS_MIN or rounds > MAX_INTERACTION_ROUNDS_MAX:
                    await update.effective_message.reply_text(f'Invalid max_interaction_rounds. Current effective value: {settings.max_interaction_rounds or self._default_max_rounds_for_mode(settings.mode)}')
                    return
                settings.max_interaction_rounds = rounds
        elif name == 'spontaneous_reply_chance':
            if value == 'default':
                settings.spontaneous_reply_chance = None
            else:
                try:
                    chance = int(value)
                except ValueError:
                    chance = -1
                if chance < SPONTANEOUS_REPLY_CHANCE_MIN or chance > SPONTANEOUS_REPLY_CHANCE_MAX:
                    await update.effective_message.reply_text(f'Invalid spontaneous_reply_chance. Current effective value: {settings.spontaneous_reply_chance if settings.spontaneous_reply_chance is not None else self.config.default_group_spontaneous_reply_chance}')
                    return
                settings.spontaneous_reply_chance = chance
        elif name == 'group_spontaneous_reply_delay_s':
            if value == 'default':
                settings.group_spontaneous_reply_delay_s = None
            else:
                try:
                    idle_s = float(value)
                except ValueError:
                    idle_s = -1.0
                if idle_s < 0 or idle_s > GROUP_SPONTANEOUS_REPLY_DELAY_MAX_S:
                    await update.effective_message.reply_text(f'Invalid group_spontaneous_reply_delay_s. Current effective value: {settings.group_spontaneous_reply_delay_s if settings.group_spontaneous_reply_delay_s is not None else self.config.default_group_spontaneous_reply_delay_s}')
                    return
                settings.group_spontaneous_reply_delay_s = idle_s
        elif name == 'provider_retry_count':
            if value == 'default':
                settings.provider_retry_count = None
            else:
                try:
                    retry_count = int(value)
                except ValueError:
                    retry_count = -1
                if retry_count < PROVIDER_RETRY_COUNT_MIN or retry_count > PROVIDER_RETRY_COUNT_MAX:
                    await update.effective_message.reply_text(f'Invalid provider_retry_count. Current effective value: {settings.provider_retry_count if settings.provider_retry_count is not None else self.config.default_provider_retry_count}')
                    return
                settings.provider_retry_count = retry_count
        elif name == 'private_reply_delay_s':
            if value == 'default':
                settings.private_reply_delay_s = None
            else:
                try:
                    delay_s = float(value)
                except ValueError:
                    delay_s = -1.0
                if delay_s < 0.0 or delay_s > REPLY_DELAY_MAX_S:
                    current = settings.private_reply_delay_s if settings.private_reply_delay_s is not None else self.config.default_private_reply_delay_s
                    await update.effective_message.reply_text(f'Invalid private_reply_delay_s. Current effective value: {current}')
                    return
                settings.private_reply_delay_s = delay_s
        elif name == 'group_reply_delay_s':
            if value == 'default':
                settings.group_reply_delay_s = None
            else:
                try:
                    delay_s = float(value)
                except ValueError:
                    delay_s = -1.0
                if delay_s < 0.0 or delay_s > REPLY_DELAY_MAX_S:
                    current = settings.group_reply_delay_s if settings.group_reply_delay_s is not None else self.config.default_group_reply_delay_s
                    await update.effective_message.reply_text(f'Invalid group_reply_delay_s. Current effective value: {current}')
                    return
                settings.group_reply_delay_s = delay_s
        elif name == 'reply_delay_s':
            if value == 'default':
                settings.private_reply_delay_s = None
                settings.group_reply_delay_s = None
            else:
                try:
                    delay_s = float(value)
                except ValueError:
                    delay_s = -1.0
                if delay_s < 0.0 or delay_s > REPLY_DELAY_MAX_S:
                    current_private = settings.private_reply_delay_s if settings.private_reply_delay_s is not None else self.config.default_private_reply_delay_s
                    current_group = settings.group_reply_delay_s if settings.group_reply_delay_s is not None else self.config.default_group_reply_delay_s
                    await update.effective_message.reply_text(f'Invalid legacy reply_delay_s. Current effective private/group values: {current_private}/{current_group}')
                    return
                settings.private_reply_delay_s = delay_s
                settings.group_reply_delay_s = delay_s
        elif name == 'metadata':
            allowed = {'on', 'off', 'default'}
            if value not in allowed:
                current = settings.metadata_injection_mode or 'on'
                await update.effective_message.reply_text(f'Invalid metadata. Current effective value: {current}')
                return
            settings.metadata_injection_mode = self.config.default_metadata_injection_mode if value == 'default' else value
        elif name == 'metadata_timezone':
            raw_value = context.args[1].strip()
            if raw_value.lower() == 'default':
                settings.metadata_timezone = self.config.default_metadata_timezone
            else:
                try:
                    ZoneInfo(raw_value)
                except ZoneInfoNotFoundError:
                    current = settings.metadata_timezone or 'UTC'
                    await update.effective_message.reply_text(f'Invalid metadata_timezone. Current effective value: {current}')
                    return
                settings.metadata_timezone = raw_value
        elif name == 'tool_history_mode':
            allowed = {'translated', 'native_same_provider', 'default'}
            if value not in allowed:
                current = settings.tool_history_mode.value if settings.tool_history_mode is not None else self.config.default_tool_history_mode
                await update.effective_message.reply_text(f'Invalid tool_history_mode. Current effective value: {current}')
                return
            settings.tool_history_mode = ToolHistoryMode(self.config.default_tool_history_mode) if value == 'default' else ToolHistoryMode(value)
        else:
            await update.effective_message.reply_text('Unknown parameter. Use /params to inspect supported names.')
            return
        await self.store.save_session(self._session_id(chat), settings)
        await update.effective_message.reply_text('Session parameter updated. Use /params or [/status full] to inspect the effective values.')

    async def presets_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not self._command_allowed(update):
            return
        names = self.preset_store.list_names()
        if not names:
            await update.effective_message.reply_text('No presets found in data/presets.')
            return
        await update.effective_message.reply_text('Available presets: ' + ', '.join(names))

    async def preset_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        if not chat or not self._allowed(chat):
            return
        settings = await self.store.get_or_create_session(self._session_id(chat), self._default_settings())
        if not context.args:
            current = settings.system_prompt.strip().splitlines()[0][:120] if settings.system_prompt.strip() else '(empty)'
            await update.effective_message.reply_text(
                'Usage: /preset <name> [augment|exact]|clear\n'
                f'Current prompt preview: {current}'
            )
            return
        name = context.args[0].strip()
        if name.lower() == 'clear':
            settings.system_prompt = self._default_settings().system_prompt
            await self.store.save_session(self._session_id(chat), settings)
            await update.effective_message.reply_text('Session prompt reset to default.')
            return
        text = self.preset_store.get_text(name)
        if text is None:
            await update.effective_message.reply_text(f'Preset not found: {name}')
            return
        if len(context.args) > 1:
            mode_value = context.args[1].strip().lower()
            try:
                settings.prompt_injection_mode = PromptInjectionMode(mode_value)
            except Exception:
                await update.effective_message.reply_text(f'Invalid prompt mode: {mode_value}. Use augment or exact.')
                return
        settings.system_prompt = text
        await self.store.save_session(self._session_id(chat), settings)
        await update.effective_message.reply_text(f'Preset loaded: {name} (prompt_injection={settings.prompt_injection_mode.value}).')

    async def prompt_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        if not chat or not self._allowed(chat):
            return
        settings = await self.store.get_or_create_session(self._session_id(chat), self._default_settings())
        if not context.args:
            preview = settings.system_prompt.strip() or '(empty)'
            if len(preview) > 700:
                preview = preview[:700] + '\n[truncated]'
            await update.effective_message.reply_text(
                'Usage:\n'
                '/prompt show\n'
                '/prompt reset\n'
                '/prompt set <text>\n'
                '/prompt append <text>\n'
                '/prompt mode <augment|exact>\n\n'
                f'Current prompt mode: {settings.prompt_injection_mode.value}\n\n'
                f'Current prompt:\n{preview}'
            )
            return
        sub = context.args[0].strip().lower()
        rest = ' '.join(context.args[1:]).strip()
        if sub == 'show':
            preview = settings.system_prompt.strip() or '(empty)'
            if len(preview) > 3500:
                preview = preview[:3500] + '\n[truncated]'
            await update.effective_message.reply_text(f'Prompt mode: {settings.prompt_injection_mode.value}\n\n{preview}')
            return
        if sub == 'reset':
            settings.system_prompt = self._default_settings().system_prompt
            await self.store.save_session(self._session_id(chat), settings)
            await update.effective_message.reply_text('System prompt reset to default.')
            return
        if sub == 'set':
            if not rest:
                await update.effective_message.reply_text('Usage: /prompt set <text>')
                return
            settings.system_prompt = rest
            await self.store.save_session(self._session_id(chat), settings)
            await update.effective_message.reply_text('System prompt replaced for this chat.')
            return
        if sub == 'append':
            if not rest:
                await update.effective_message.reply_text('Usage: /prompt append <text>')
                return
            base = settings.system_prompt.rstrip()
            settings.system_prompt = (base + '\n\n' + rest).strip()
            await self.store.save_session(self._session_id(chat), settings)
            await update.effective_message.reply_text('Text appended to the session prompt.')
            return
        if sub == 'mode':
            if not rest:
                await update.effective_message.reply_text(
                    f'Usage: /prompt mode <augment|exact>\nCurrent prompt mode: {settings.prompt_injection_mode.value}'
                )
                return
            mode_value = rest.split()[0].strip().lower()
            try:
                settings.prompt_injection_mode = PromptInjectionMode(mode_value)
            except Exception:
                await update.effective_message.reply_text(f'Invalid prompt mode: {mode_value}. Use augment or exact.')
                return
            await self.store.save_session(self._session_id(chat), settings)
            await update.effective_message.reply_text(f'Prompt injection mode set to {settings.prompt_injection_mode.value}.')
            return
        await update.effective_message.reply_text('Usage: /prompt show|reset|set <text>|append <text>|mode <augment|exact>')

    async def private_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message = update.effective_message
        should_reply = bool(message and self._is_plain_text_reply_trigger(message))
        await self._ingest_update(update, should_reply=should_reply, is_group=False, is_edit=False)

    async def private_edited_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._ingest_update(update, should_reply=False, is_group=False, is_edit=True)

    async def group_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        message = update.effective_message
        if not chat or not message:
            return
        plan = await self._compute_group_reply_plan(update, context)
        await self._ingest_update(update, should_reply=plan.should_reply, is_group=True, is_edit=False, group_explicit_reply=plan.explicit)

    async def group_edited_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await self._ingest_update(update, should_reply=False, is_group=True, is_edit=True)

    @staticmethod
    def _has_link_entities(message: Message) -> bool:
        for entity in (message.entities or ()):
            entity_type = str(getattr(entity, 'type', '')).lower()
            if entity_type in {'url', 'text_link'}:
                return True
        return False

    def _is_plain_text_reply_trigger(self, message: Message) -> bool:
        text = (message.text or '').strip()
        if not text or message.caption:
            return False
        media_attrs = (
            'photo', 'sticker', 'document', 'video', 'animation', 'audio', 'voice', 'video_note',
            'contact', 'location', 'venue', 'poll', 'dice', 'game', 'invoice', 'successful_payment',
        )
        if any(getattr(message, attr, None) for attr in media_attrs):
            return False
        return True

    async def _compute_group_reply_plan(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> GroupReplyPlan:
        chat = update.effective_chat
        message = update.effective_message
        if not chat or not message or not self._is_plain_text_reply_trigger(message):
            return GroupReplyPlan(should_reply=False, explicit=False)
        text = message.text or ''
        lowered = text.lower()
        has_ignore = any(bad.lower() in lowered for bad in self.config.telegram.ignore_keywords)
        explicit = any(k.lower() in lowered for k in self.config.telegram.keywords)
        explicit = explicit or bool(message.reply_to_message and message.reply_to_message.from_user and message.reply_to_message.from_user.id == context.bot.id)
        if explicit and not has_ignore:
            return GroupReplyPlan(should_reply=True, explicit=True)
        if has_ignore:
            return GroupReplyPlan(should_reply=False, explicit=False)
        settings = await self.store.get_or_create_session(self._session_id(chat), self._default_settings())
        chance = self.runtime._effective_spontaneous_reply_chance(settings)
        return GroupReplyPlan(should_reply=chance > 0, explicit=False)

    async def _ingest_update(self, update: Update, *, should_reply: bool, is_group: bool, is_edit: bool, group_explicit_reply: bool = False) -> None:
        chat = update.effective_chat
        message = update.effective_message
        if not chat or not message or not self._allowed(chat):
            return
        state = self._flow_state(chat.id)
        await self._mark_ingest_started(state)
        try:
            session_id = self._session_id(chat)
            logger.info('tg.ingest.start chat=%s msg=%s group=%s edit=%s reply=%s', self._chat_log_id(chat.id), message.message_id, int(is_group), int(is_edit), int(should_reply))
            settings = await self.store.get_or_create_session(session_id, self._default_settings())
            user_name = update.effective_user.username or update.effective_user.full_name or 'user'
            parts = await self._safe_extract_parts(message, session_id)
            logger.info('tg.ingest.parts chat=%s msg=%s sender=%s parts=%s preview=%s', self._chat_log_id(chat.id), message.message_id, user_name, self._parts_summary(parts), self._log_parts_preview(parts))#self._message_preview(message.text or message.caption or '')
            link_mode = settings.link_prefetch_mode if settings.link_prefetch_mode != 'default' else self.config.default_link_prefetch_mode
            message_text = message.text or message.caption or ''
            if message_text and link_mode != 'off':
                try:
                    previews = await fetch_link_previews(message_text, mode=link_mode, telegram=self.config.telegram)
                except Exception:
                    previews = []
                if previews:
                    preview_parts = previews_to_parts(previews, mode=link_mode)
                    parts.extend(preview_parts)
                    logger.info('tg.ingest.links chat=%s msg=%s urls=%s mode=%s chars=%s', self._chat_log_id(chat.id), message.message_id, len(previews), link_mode, sum(len(part.text or '') for part in preview_parts))
            if parts:
                parts = await self._sync_parts_to_remote(session_id, parts)
            if not parts:
                parts = [MessagePart(kind=PartKind.TEXT, text='')]

            auto_note_parts: list[MessagePart] = []
            if settings.metadata_injection_mode != 'off':
                event_time = message.edit_date or message.date or datetime.now(timezone.utc)
                try:
                    zone = ZoneInfo(settings.metadata_timezone or 'UTC')
                except ZoneInfoNotFoundError:
                    zone = ZoneInfo('UTC')
                local_time = event_time.astimezone(zone).isoformat(timespec='seconds')
                username = f'@{message.from_user.username}' if message.from_user and message.from_user.username else '-'
                nickname = f'{message.from_user.full_name}'.replace('"', "'") if message.from_user.full_name else ''
                auto_note_parts.append(
                    MessagePart(
                        kind=PartKind.TEXT,
                        text=f'[Message metadata: username={username} nickname="{nickname}" time={local_time}]',
                        remote_sync=False,
                        origin='auto_note',
                    )
                )
            user_parts = [part for part in parts if part.origin != 'auto_note']
            auto_note_parts.extend(part for part in parts if part.origin == 'auto_note')
            if not user_parts:
                user_parts = [MessagePart(kind=PartKind.TEXT, text='')]
            envelope = ConversationMessage(
                role=MessageRole.USER,
                parts=[*auto_note_parts, *user_parts],
                metadata={'telegram_message_id': message.message_id, 'is_edit': is_edit, 'media_group_id': message.media_group_id},
            )
            stored = await self.runtime.ingest_user_message(session_id=session_id, incoming_message=envelope)
            candidate = ReplyCandidate(
                stored_message_id=stored.db_id,
                user_display_name=user_name,
                source_message=message,
                spontaneous=bool(is_group and not group_explicit_reply),
            )
        finally:
            await self._mark_ingest_finished(state)
            logger.info('tg.ingest.done chat=%s msg=%s inflight=%s', self._chat_log_id(chat.id), message.message_id, state.ingest_inflight)

        if is_group:
            await self._clear_pending_spontaneous_candidate(chat.id, newer_message_id=candidate.stored_message_id)

        if should_reply:
            if is_group:
                settings = await self.store.get_or_create_session(session_id, self._default_settings())
                delay_s = self.runtime._effective_group_reply_delay_s(settings) if group_explicit_reply else self.runtime._effective_group_spontaneous_reply_delay_s(settings)
                delay_param = 'group_reply_delay_s' if group_explicit_reply else 'group_spontaneous_reply_delay_s'
                token = await self._next_reply_token(state)
                logger.info('tg.reply.schedule chat=%s msg=%s token=%s delay_s=%.2f delay_param=%s group=1 explicit=%s', self._chat_log_id(chat.id), candidate.stored_message_id, token, delay_s, delay_param, int(bool(group_explicit_reply)))
                if group_explicit_reply:
                    asyncio.create_task(self._promote_candidate_after_delay(chat.id, token, candidate, delay_s))
                else:
                    asyncio.create_task(self._promote_spontaneous_candidate_after_delay(chat.id, session_id, token, candidate, delay_s))
            else:
                delay_s = self.runtime._effective_private_reply_delay_s(settings)
                token = await self._next_reply_token(state)
                logger.info('tg.reply.schedule chat=%s msg=%s token=%s delay_s=%.2f delay_param=%s group=0', self._chat_log_id(chat.id), candidate.stored_message_id, token, delay_s, 'private_reply_delay_s')
                asyncio.create_task(self._promote_candidate_after_delay(chat.id, token, candidate, delay_s))
        else:
            if is_group:
                await self._next_reply_token(state)
            await self._ensure_reply_worker(chat.id)

    def _reply_to_message_id(self, message: Message) -> int | None:
        if not self.config.telegram.reply_to_user_message:
            return None
        return message.message_id

    async def _send_text_message(self, source_message: Message, text: str) -> Message:
        bot = source_message.get_bot()
        reply_to_message_id = self._reply_to_message_id(source_message)
        last_message: Message | None = None
        for chunk in _chunk_text_for_telegram(text, limit=MAX_TELEGRAM_TEXT_CHARS):
            last_message = await bot_message_safe(
                bot, 'send_message',
                chat_id=source_message.chat.id,
                text=(chunk or '...'),
                parse_mode='MarkdownV2',
                disable_web_page_preview=True,
                reply_to_message_id=reply_to_message_id,
            )
        if last_message is None:
            raise RuntimeError('Telegram text delivery produced no message')
        return last_message

    async def _record_delivered_assistant_text(self, *, session_id: str, result: TurnResult) -> None:
        text = (result.text or '').strip()
        if not text:
            return
        assistant_metadata = {'provider_native': {'provider': result.provider_name, 'items': result.provider_history_items}} if result.provider_name and result.provider_history_items else None
        await self.runtime.record_assistant_text(session_id=session_id, text=text, metadata=assistant_metadata)

    async def _notify_user_error(self, source_message: Message, text: str, *, renderer: TelegramMessageRenderer | None = None) -> None:
        if renderer is not None and renderer.message is not None:
            try:
                await renderer._edit_text(text, force=True)
                return
            except Exception:
                logger.exception('tg.error_notice.edit_failed chat=%s msg=%s', self._chat_log_id(source_message.chat.id), getattr(source_message, 'message_id', '-'))
        try:
            await self._send_text_message(source_message, text)
        except Exception:
            logger.exception('tg.error_notice.send_failed chat=%s msg=%s', self._chat_log_id(source_message.chat.id), getattr(source_message, 'message_id', '-'))

    async def _safe_extract_parts(self, message: Message, session_id: str) -> list[MessagePart]:
        try:
            return await extract_message_parts(message, self.artifact_store, session_id, self.config.telegram)
        except Exception as exc:
            logger.exception('Failed to ingest Telegram attachments for chat session %s', session_id)
            fallback_text = f'[Attachment download failed: {exc.__class__.__name__}]'
            text = message.text or message.caption
            parts = []
            if text:
                parts.append(MessagePart(kind=PartKind.TEXT, text=text))
            parts.append(MessagePart(kind=PartKind.TEXT, text=fallback_text, origin='auto_note'))
            return parts


    async def _sync_parts_to_remote(self, session_id: str, parts: list[MessagePart]) -> list[MessagePart]:
        syncable_parts = [part for part in parts if part.artifact_path and part.remote_sync]
        local_paths = [part.artifact_path for part in syncable_parts if part.artifact_path]
        if not local_paths:
            return parts
        if not self.remote_workspace or not self.remote_workspace.enabled:
            local_path_objs = tuple(Path(value) for value in local_paths)
            for path in local_path_objs:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    logger.warning('tg.remote_sync.cleanup_failed sid=%s file=%s', clip_for_log(session_id, limit=48), path.name)
            return [replace(part, artifact_path=None, remote_sync=False) if part.artifact_path and part.remote_sync else part for part in parts]
        local_path_objs = tuple(Path(value) for value in local_paths)
        predicted_remote = {str(path.resolve()): f"{self.remote_workspace.session_paths(session_id).inputs.rstrip('/')}/{path.name}" for path in local_path_objs}
        surviving_remote: set[str] = set()
        rotated_remote: list[str] = []
        try:
            sync_result = await self.remote_workspace.sync_inputs(session_id, local_path_objs)
            surviving_remote = set(sync_result.kept_paths)
            rotated_remote = list(sync_result.rotated_paths)
            logger.info('tg.remote_sync.ok sid=%s requested=%s kept=%s rotated=%s', clip_for_log(session_id, limit=48), len(local_path_objs), len(surviving_remote), len(rotated_remote))
        except Exception as exc:
            logger.exception('tg.remote_sync.failed sid=%s files=%s err=%s', clip_for_log(session_id, limit=48), len(local_path_objs), exc.__class__.__name__)
        finally:
            for path in local_path_objs:
                try:
                    path.unlink(missing_ok=True)
                except Exception:
                    logger.warning('tg.remote_sync.cleanup_failed sid=%s file=%s', clip_for_log(session_id, limit=48), path.name)
        updated_parts: list[MessagePart] = []
        synced_entries: list[str] = []
        for part in parts:
            if not (part.artifact_path and part.remote_sync):
                updated_parts.append(part)
                continue
            local_key = str(Path(part.artifact_path).resolve())
            remote_path = predicted_remote.get(local_key)
            if remote_path and remote_path in surviving_remote:
                synced_entries.append(f"{part.filename or Path(remote_path).name} -> {remote_path}")
                updated_parts.append(replace(part, artifact_path=remote_path, remote_sync=True))
                continue
            filename = part.filename or 'file'
            updated_parts.append(
                MessagePart(
                    kind=PartKind.TEXT,
                    text=f'[Attachment sync failed for remote use: {filename}]',
                    remote_sync=False,
                    origin='auto_note',
                )
            )
        note_parts: list[MessagePart] = []
        if rotated_remote:
            rotated_list = ', '.join(Path(path).name or path for path in rotated_remote)
            note_parts.append(MessagePart(kind=PartKind.TEXT, text=f'[Remote attachment rotation removed: {rotated_list}]', remote_sync=False, origin='auto_note'))
        if synced_entries:
            note_parts.append(MessagePart(kind=PartKind.TEXT, text='[Attachment synced to remote for tool use: ' + '; '.join(synced_entries) + ']', remote_sync=False, origin='auto_note'))
        return [*note_parts, *updated_parts]

    async def _promote_candidate_after_delay(self, chat_id: int, token: int, candidate: ReplyCandidate, delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        state = self._flow_state(chat_id)
        async with state.mutex:
            # The latest token wins. This debounce lets rapid multi-part uploads or message edits settle
            # so the worker answers only the newest candidate instead of racing every intermediate update.
            if state.latest_reply_token != token:
                logger.debug('tg.reply.drop chat=%s token=%s latest=%s', self._chat_log_id(chat_id), token, state.latest_reply_token)
                return
        logger.info('tg.reply.promote chat=%s msg=%s token=%s', self._chat_log_id(chat_id), candidate.stored_message_id, token)
        await self._set_latest_reply_candidate(chat_id, candidate)

    async def _promote_spontaneous_candidate_after_delay(self, chat_id: int, session_id: str, token: int, candidate: ReplyCandidate, delay_s: float) -> None:
        await asyncio.sleep(delay_s)
        state = self._flow_state(chat_id)
        async with state.mutex:
            if state.latest_reply_token != token:
                logger.debug('tg.reply.drop chat=%s token=%s latest=%s', self._chat_log_id(chat_id), token, state.latest_reply_token)
                return
        settings = await self.store.get_or_create_session(session_id, self._default_settings())
        chance = self.runtime._effective_spontaneous_reply_chance(settings)
        roll = random.random() * 100.0
        if chance <= 0 or roll >= chance:
            logger.info('tg.reply.spontaneous.skip chat=%s msg=%s token=%s roll=%.2f chance=%s', self._chat_log_id(chat_id), candidate.stored_message_id, token, roll, chance)
            return
        logger.info('tg.reply.spontaneous.hit chat=%s msg=%s token=%s roll=%.2f chance=%s', self._chat_log_id(chat_id), candidate.stored_message_id, token, roll, chance)
        await self._set_latest_reply_candidate(chat_id, candidate)

    async def _set_latest_reply_candidate(self, chat_id: int, candidate: ReplyCandidate) -> None:
        state = self._flow_state(chat_id)
        accepted = False
        async with state.mutex:
            if state.latest_reply_candidate is None or candidate.stored_message_id >= state.latest_reply_candidate.stored_message_id:
                state.latest_reply_candidate = candidate
                accepted = True
        logger.info('tg.reply.candidate chat=%s msg=%s accepted=%s', self._chat_log_id(chat_id), candidate.stored_message_id, int(accepted))
        await self._ensure_reply_worker(chat_id)

    async def _ensure_reply_worker(self, chat_id: int) -> None:
        state = self._flow_state(chat_id)
        async with state.mutex:
            if state.reply_task and not state.reply_task.done():
                return
            logger.info('tg.reply.worker_start chat=%s', self._chat_log_id(chat_id))
            state.reply_task = asyncio.create_task(self._reply_worker(chat_id))

    async def _reply_worker(self, chat_id: int) -> None:
        state = self._flow_state(chat_id)
        try:
            while True:
                await state.ingest_idle.wait()
                async with state.mutex:
                    candidate = state.latest_reply_candidate
                    if candidate is None or candidate.stored_message_id <= state.last_replied_message_id:
                        logger.info('tg.reply.worker_idle chat=%s last=%s', self._chat_log_id(chat_id), state.last_replied_message_id)
                        state.reply_task = None
                        return
                try:
                    logger.info('tg.reply.run chat=%s msg=%s', self._chat_log_id(chat_id), candidate.stored_message_id)
                    await self._reply_to_candidate(candidate)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.exception('Reply failed for chat %s message %s', chat_id, candidate.stored_message_id)
                async with state.mutex:
                    state.last_replied_message_id = max(state.last_replied_message_id, candidate.stored_message_id)
                    logger.info('tg.reply.done chat=%s msg=%s last=%s', self._chat_log_id(chat_id), candidate.stored_message_id, state.last_replied_message_id)
                    if state.latest_reply_candidate and state.latest_reply_candidate.stored_message_id <= state.last_replied_message_id:
                        state.latest_reply_candidate = None
        except asyncio.CancelledError:
            async with state.mutex:
                state.reply_task = None
            raise
        except Exception:
            logger.exception('Reply worker failed for chat %s', chat_id)
            async with state.mutex:
                state.reply_task = None

    async def _reply_to_candidate(self, candidate: ReplyCandidate) -> None:
        message = candidate.source_message
        chat = message.chat
        session_id = self._session_id(chat)
        settings = await self.store.get_or_create_session(session_id, self._default_settings())
        logger.info('tg.reply.prepare chat=%s msg=%s delivery=%s process=%s', self._chat_log_id(chat.id), candidate.stored_message_id, settings.response_delivery.value, settings.process_visibility.value)
        await message.get_bot().send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)

        placeholder: Message | None = None
        should_show_status = settings.process_visibility in {ProcessVisibility.STATUS, ProcessVisibility.VERBOSE, ProcessVisibility.FULL}
        should_send_minimal_ack = settings.process_visibility == ProcessVisibility.MINIMAL
        if should_show_status or should_send_minimal_ack:
            placeholder = await self._send_text_message(message, '...')

        renderer = TelegramMessageRenderer(
            placeholder if should_show_status else None,
            response_delivery=settings.response_delivery,
            min_edit_interval_s=self.config.telegram.min_edit_interval_s,
            source_message=message,
            reply_to_source_message=self.config.telegram.reply_to_user_message,
            process_visibility=settings.process_visibility,
        )
        if should_show_status:
            await renderer.begin()

        sent_before_receipts: list[dict[str, object]] = []

        async def emit_event(event):
            nonlocal sent_before_receipts
            if event.kind == 'sticker':
                sticker = self._sticker_from_event(event)
                if sticker is not None:
                    receipts = await self._send_stickers_direct(message, [sticker])
                    sent_before_receipts.extend(receipts)
                    for receipt in receipts:
                        await self.runtime.record_tool_observation(
                            session_id=session_id,
                            name='sticker_send',
                            phase='delivery',
                            payload=receipt,
                        )
                return
            if event.kind == 'assistant_text':
                text = (event.detail or str(event.payload.get('text') or '')).strip()
                if text:
                    await renderer.send_text(text)
                return
            if should_show_status:
                await renderer.emit(event)

        async def _typing_heartbeat() -> None:
            while True:
                try:
                    await message.get_bot().send_chat_action(chat_id=chat.id, action=ChatAction.TYPING)
                except Exception:
                    logger.debug("tg.typing.failed", exc_info=True)
                await asyncio.sleep(4.0)

        typing_task: asyncio.Task[None] | None = asyncio.create_task(_typing_heartbeat())
        
        # The stored message id identifies the debounce winner for bookkeeping and delivery flow,
        # while generation still uses the newest live session context at fire time.
        try:
            result = await self.runtime.run_turn_from_stored(
                session_id=session_id,
                user_display_name=candidate.user_display_name,
                trigger_message_id=candidate.stored_message_id,
                emit=emit_event,
            )
        except Exception as exc:
            logger.exception('tg.reply.failed chat=%s msg=%s', self._chat_log_id(chat.id), candidate.stored_message_id)
            try:
                await self._notify_user_error(
                    message,
                    self._safe_user_error_text('Reply generation', exc),
                    renderer=renderer if should_show_status else None,
                )
            except Exception:
                logger.exception('tg.reply.error_notice_failed chat=%s msg=%s', self._chat_log_id(chat.id), candidate.stored_message_id)
                if should_show_status:
                    try:
                        await renderer.abort()
                    except Exception:
                        logger.exception('tg.reply.abort_failed chat=%s msg=%s', self._chat_log_id(chat.id), candidate.stored_message_id)
            return
        finally:
            if typing_task:
                typing_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await typing_task

        logger.info('tg.reply.result chat=%s msg=%s text_chars=%s artifacts=%s stickers=%s usage=in=%s out=%s total=%s', self._chat_log_id(chat.id), candidate.stored_message_id, len(result.text), len(result.artifacts), len(result.stickers), result.usage.input_tokens, result.usage.output_tokens, result.usage.total_tokens)
        try:
            await self._deliver_result(message, renderer, settings, result, sent_before_receipts=sent_before_receipts)
            await self._record_delivered_assistant_text(session_id=session_id, result=result)
        except Exception as exc:
            logger.exception('tg.deliver.failed chat=%s msg=%s', self._chat_log_id(chat.id), candidate.stored_message_id)
            try:
                await self._notify_user_error(
                    message,
                    self._safe_user_error_text('Reply delivery', exc),
                    renderer=renderer if should_show_status else None,
                )
            except Exception:
                logger.exception('tg.deliver.error_notice_failed chat=%s msg=%s', self._chat_log_id(chat.id), candidate.stored_message_id)
                if should_show_status:
                    try:
                        await renderer.abort()
                    except Exception:
                        logger.exception('tg.deliver.abort_failed chat=%s msg=%s', self._chat_log_id(chat.id), candidate.stored_message_id)

    async def _deliver_result(
        self,
        source_message: Message,
        renderer: TelegramMessageRenderer,
        settings: SessionSettings,
        result: TurnResult,
        *,
        sent_before_receipts: list[dict[str, object]],
    ) -> None:
        after_stickers = [sticker for sticker in result.stickers if sticker.timing == StickerTiming.AFTER_FINAL]
        logger.info('tg.deliver.start chat=%s process=%s text_chars=%s artifacts=%s after_stickers=%s usage=in=%s out=%s total=%s', self._chat_log_id(source_message.chat.id), settings.process_visibility.value, len(result.text), len(result.artifacts), len(after_stickers), result.usage.input_tokens, result.usage.output_tokens, result.usage.total_tokens)
        if (result.text or '').strip():
            if settings.process_visibility in {ProcessVisibility.OFF, ProcessVisibility.MINIMAL}:
                await self._send_text_message(source_message, result.text)
            else:
                await renderer.finalize(result.text)
        elif settings.process_visibility not in {ProcessVisibility.OFF}:
            await renderer.complete_without_answer()
        if result.artifacts:
            if settings.process_visibility in {ProcessVisibility.OFF, ProcessVisibility.MINIMAL}:
                for artifact in result.artifacts:
                    try:
                        suffix = artifact.path.suffix.lower()
                        if suffix in {'.png', '.jpg', '.jpeg', '.webp'}:
                            with artifact.path.open('rb') as fh:
                                await source_message.get_bot().send_photo(chat_id=source_message.chat.id, photo=fh, caption=artifact.caption, reply_to_message_id=self._reply_to_message_id(source_message))
                        else:
                            from telegram import InputFile
                            with artifact.path.open('rb') as fh:
                                await source_message.get_bot().send_document(chat_id=source_message.chat.id, document=InputFile(fh, filename=artifact.filename), caption=artifact.caption, reply_to_message_id=self._reply_to_message_id(source_message))
                    except Exception:
                        logger.exception('Failed to send artifact %s after final delivery', artifact.path)
            else:
                await renderer.send_artifacts(result.artifacts)
        if settings.process_visibility in {ProcessVisibility.OFF, ProcessVisibility.MINIMAL}:
            try:
                sent_after_receipts = await self._send_stickers_direct(source_message, after_stickers)
            except Exception:
                logger.exception('Failed to send after-final stickers in direct mode')
                sent_after_receipts = []
        else:
            sent_after_receipts = await renderer.send_stickers(after_stickers)
        logger.info('tg.deliver.done chat=%s after_stickers=%s before_receipts=%s', self._chat_log_id(source_message.chat.id), len(sent_after_receipts), len(sent_before_receipts))
        for receipt in sent_after_receipts:
            await self.runtime.record_tool_observation(
                session_id=self._session_id(source_message.chat),
                name='sticker_send',
                phase='delivery',
                payload=receipt,
            )

    async def _send_stickers_direct(self, source_message: Message, stickers: list[OutboundSticker]) -> list[dict[str, object]]:
        receipts: list[dict[str, object]] = []
        for sticker in stickers:
            receipt: dict[str, object] = sticker.delivery_receipt()
            if not sticker.path.exists():
                receipt['error'] = 'missing_file'
                receipts.append(receipt)
                continue
            try:
                with sticker.path.open('rb') as fh:
                    sent_message = await source_message.get_bot().send_sticker(chat_id=source_message.chat.id, sticker=fh, emoji=sticker.emoji, reply_to_message_id=self._reply_to_message_id(source_message))
                receipt['delivery_state'] = 'sent'
                logger.info('tg.sticker.sent chat=%s sticker=%s timing=%s', self._chat_log_id(source_message.chat.id), clip_for_log(sticker.display_reference(), limit=48), sticker.timing.value)
                receipt['sent'] = True
                receipt['telegram_message_id'] = getattr(sent_message, 'message_id', None)
            except Exception as exc:
                logger.exception('Failed to send sticker %s', sticker.display_reference())
                receipt['error'] = exc.__class__.__name__
            receipts.append(receipt)
        return receipts

    @staticmethod
    def _sticker_from_event(event) -> OutboundSticker | None:
        payload = event.payload or {}
        path = payload.get('path')
        timing = payload.get('timing') or 'after_final'
        if not path:
            return None
        try:
            return OutboundSticker(
                path=Path(path),
                emoji=payload.get('emoji'),
                timing=StickerTiming.parse(timing),
                label=payload.get('label'),
                source_id=payload.get('source_id'),
            )
        except Exception:
            return None

    async def _mark_ingest_started(self, state: ChatFlowState) -> None:
        async with state.mutex:
            if state.ingest_inflight == 0:
                state.ingest_idle.clear()
            state.ingest_inflight += 1

    async def _mark_ingest_finished(self, state: ChatFlowState) -> None:
        async with state.mutex:
            state.ingest_inflight = max(0, state.ingest_inflight - 1)
            if state.ingest_inflight == 0:
                state.ingest_idle.set()

    async def _next_reply_token(self, state: ChatFlowState) -> int:
        async with state.mutex:
            state.latest_reply_token += 1
            return state.latest_reply_token

    async def _clear_pending_spontaneous_candidate(self, chat_id: int, *, newer_message_id: int) -> None:
        state = self._flow_state(chat_id)
        cleared_message_id: int | None = None
        async with state.mutex:
            pending = state.latest_reply_candidate
            if pending and pending.spontaneous and pending.stored_message_id < newer_message_id:
                cleared_message_id = pending.stored_message_id
                state.latest_reply_candidate = None
        if cleared_message_id is not None:
            logger.info(
                'tg.reply.spontaneous.cancel chat=%s pending_msg=%s newer_msg=%s',
                self._chat_log_id(chat_id),
                cleared_message_id,
                newer_message_id,
            )

    async def _cancel_pending_reply(self, chat_id: int) -> None:
        state = self._flow_state(chat_id)
        task: asyncio.Task[None] | None = None
        async with state.mutex:
            state.latest_reply_candidate = None
            task = state.reply_task
            state.reply_task = None
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    @staticmethod
    def _rollback_side(role: MessageRole) -> str:
        return 'user' if role == MessageRole.USER else 'bot'

    async def _collect_rollback_message_ids(self, session_id: str, block_count: int) -> list[int]:
        limit = max(40, min(400, block_count * 40))
        recent = []
        while True:
            recent = await self.store.list_recent_visible_messages(session_id, limit=limit)
            if not recent:
                return []
            groups = 0
            prev_side: str | None = None
            for item in recent:
                side = self._rollback_side(item.message.role)
                if side != prev_side:
                    groups += 1
                    prev_side = side
                if groups >= block_count:
                    break
            if groups >= block_count or len(recent) < limit or limit >= 5000:
                break
            limit = min(limit * 2, 5000)
        target_ids: list[int] = []
        groups = 0
        prev_side: str | None = None
        for item in recent:
            side = self._rollback_side(item.message.role)
            if side != prev_side:
                groups += 1
                if groups > block_count:
                    break
                prev_side = side
            target_ids.append(item.db_id)
        return target_ids

    @staticmethod
    def _rollback_item_type(item: StoredConversationMessage) -> str:
        message = item.message
        role = message.role.value
        if message.role == MessageRole.TOOL:
            tool_name = message.name or 'tool'
            phase = str((message.metadata or {}).get('tool_phase') or '').strip()
            return f'{role}:{tool_name}' if not phase else f'{role}:{tool_name}:{phase}'
        return role

    @classmethod
    def _rollback_item_preview(cls, item: StoredConversationMessage, *, limit: int = 40) -> str:
        texts: list[str] = []
        files: list[str] = []

        for part in item.message.parts:
            if part.kind == PartKind.TEXT:
                text = (part.text or '').strip()
                if text:
                    texts.append(text)
            elif part.filename:
                files.append(part.filename)
            else:
                files.append(part.kind.value)

        preview = ' | '.join(texts)
        if files:
            file_hint = ','.join(files[:3])
            if len(files) > 3:
                file_hint += f'+{len(files) - 3}'
            preview = f'{preview} [files={file_hint}]' if preview else f'[files={file_hint}]'

        if not preview:
            preview = f'[{cls._parts_summary(item.message.parts)}]'

        return clip_for_log(preview, limit=limit, rlimit=40)

    async def _flow_snapshot(self, chat_id: int) -> dict[str, object]:
        state = self._flow_state(chat_id)
        async with state.mutex:
            return {
                'ingest_inflight': state.ingest_inflight,
                'reply_running': bool(state.reply_task and not state.reply_task.done()),
                'latest_pending_reply_message_id': state.latest_reply_candidate.stored_message_id if state.latest_reply_candidate else None,
                'last_replied_message_id': state.last_replied_message_id,
            }

    async def _update_session_enum(self, update: Update, context: ContextTypes.DEFAULT_TYPE, target: str) -> None:
        chat = update.effective_chat
        if not chat:
            return
        settings = await self.store.get_or_create_session(self._session_id(chat), self._default_settings())
        value = (context.args[0].strip().lower() if context.args else '')
        if target == 'mode':
            try:
                settings.mode = ChatMode(value)
            except Exception:
                await update.effective_message.reply_text(f'Usage: /mode chat|assist|agent\nCurrent mode: {settings.mode.value}')
                return
            await self.store.save_session(self._session_id(chat), settings)
            await update.effective_message.reply_text(f'Mode set to {settings.mode.value}.')
            return
        if target == 'process':
            try:
                settings.process_visibility = ProcessVisibility(value)
            except Exception:
                await update.effective_message.reply_text(f'Usage: /process off|minimal|status|verbose|full\nCurrent process visibility: {settings.process_visibility.value}')
                return
            await self.store.save_session(self._session_id(chat), settings)
            await update.effective_message.reply_text(f'Process visibility set to {settings.process_visibility.value}.')
            return
        if target == 'stickers':
            try:
                settings.sticker_mode = StickerMode(value)
            except Exception:
                await update.effective_message.reply_text(f'Usage: /stickers off|auto\nCurrent sticker mode: {settings.sticker_mode.value}')
                return
            await self.store.save_session(self._session_id(chat), settings)
            await update.effective_message.reply_text(f'Sticker mode set to {settings.sticker_mode.value}.')
            return
        try:
            settings.response_delivery = ResponseDelivery(value)
        except Exception:
            await update.effective_message.reply_text(f'Usage: /delivery edit|final_new\nCurrent delivery mode: {settings.response_delivery.value}')
            return
        await self.store.save_session(self._session_id(chat), settings)
        await update.effective_message.reply_text(f'Response delivery set to {settings.response_delivery.value}.')

    def _command_allowed(self, update: Update) -> bool:
        chat = update.effective_chat
        return bool(chat and self._allowed(chat))

    def _advanced_allowed(self, update: Update) -> bool:
        if not self._command_allowed(update):
            return False
        control_uids = self.config.telegram.control_uids
        if not control_uids:
            return True
        user = update.effective_user
        return bool(user and str(user.id) in control_uids)

    def _allowed(self, chat: Chat) -> bool:
        whitelist = self.config.telegram.whitelist
        if not whitelist:
            return True
        return str(chat.id) in whitelist

    def _default_max_rounds_for_mode(self, mode: ChatMode) -> int:
        if mode == ChatMode.CHAT:
            return max(1, self.config.default_chat_max_rounds)
        if mode == ChatMode.ASSIST:
            return max(1, self.config.default_assist_max_rounds)
        return max(1, self.config.default_agent_max_rounds)

    def _default_settings(self) -> SessionSettings:
        return self.config.default_session_settings()

    def _flow_state(self, chat_id: int) -> ChatFlowState:
        state = self._chat_states.get(chat_id)
        if state is None:
            state = ChatFlowState()
            state.ingest_idle.set()
            self._chat_states[chat_id] = state
        return state

    @staticmethod
    def _session_id(chat: Chat) -> str:
        return f'telegram:{chat.id}'
