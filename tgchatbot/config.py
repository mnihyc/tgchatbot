from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from tgchatbot.domain.models import (
    ChatMode,
    ProcessVisibility,
    PromptInjectionMode,
    ResponseDelivery,
    SessionSettings,
    StickerMode,
    ToolHistoryMode,
    default_system_prompt,
)
from tgchatbot.settings_schema import (
    COMPACT_KEEP_RECENT_RATIO_MAX,
    COMPACT_KEEP_RECENT_RATIO_MIN,
    COMPACT_MIN_MESSAGES_MAX,
    COMPACT_MIN_MESSAGES_MIN,
    COMPACT_TOKEN_MAX,
    COMPACT_TOKEN_MIN,
    COMPACT_TOOL_RATIO_THRESHOLD_MAX,
    COMPACT_TOOL_RATIO_THRESHOLD_MIN,
    GEMINI_THINKING_BUDGET_MAX,
    GEMINI_THINKING_BUDGET_MIN,
    GEMINI_THINKING_LEVEL_VALUES,
    IMAGE_LIMIT_DISABLED,
    IMAGE_LIMIT_MAX,
    MAX_INTERACTION_ROUNDS_MAX,
    MAX_INTERACTION_ROUNDS_MIN,
    MAX_OUTPUT_TOKENS_MAX,
    MAX_OUTPUT_TOKENS_MIN,
    MIN_RAW_MESSAGES_RESERVE_MAX,
    MIN_RAW_MESSAGES_RESERVE_MIN,
    NATIVE_WEB_SEARCH_MAX_MAX,
    NATIVE_WEB_SEARCH_MAX_MIN,
    PROVIDER_RETRY_COUNT_MAX,
    PROVIDER_RETRY_COUNT_MIN,
    REASONING_EFFORT_VALUES,
    SPONTANEOUS_REPLY_CHANCE_MAX,
    SPONTANEOUS_REPLY_CHANCE_MIN,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    TEXT_VERBOSITY_VALUES,
    TOP_K_MAX,
    TOP_K_MIN,
    TOP_P_MAX,
    TOP_P_MIN,
    effective_reasoning_summary,
    normalize_choice as _choice,
    normalize_optional_choice,
    parse_optional_bounded_int_env,
    parse_bounded_float_env,
    parse_bounded_int_env,
    parse_optional_disabled_int_env,
)


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    base_url: str
    model: str
    reasoning_effort: str
    reasoning_summary: str
    text_verbosity: str
    max_output_tokens: int
    max_input_images: int
    compact_target_images: int
    enable_native_web_search: bool
    native_web_search_max: int
    request_timeout_s: float
    connect_timeout_s: float


@dataclass(frozen=True)
class GeminiConfig:
    api_key: str
    base_url: str
    model: str
    temperature: float
    top_p: float
    top_k: int
    include_thoughts: bool
    thinking_budget: int | None
    thinking_level: str | None
    enable_native_web_search: bool
    max_output_tokens: int
    max_input_images: int
    compact_target_images: int
    request_timeout_s: float
    connect_timeout_s: float


@dataclass(frozen=True)
class SSHExecConfig:
    enabled: bool
    host: str
    port: int
    workdir: str
    identity_file: str | None
    connect_timeout_s: int
    default_timeout_s: int
    max_tool_timeout_s: int
    max_stdout_chars: int
    max_stderr_chars: int
    max_input_files: int
    max_input_file_bytes: int
    max_output_files: int
    max_output_file_bytes: int
    server_alive_interval_s: int
    server_alive_count_max: int
    control_persist_s: int


@dataclass(frozen=True)
class TelegramConfig:
    token: str
    whitelist: tuple[str, ...]
    control_uids: tuple[str, ...]
    keywords: tuple[str, ...]
    ignore_keywords: tuple[str, ...]
    reply_to_user_message: bool
    min_edit_interval_s: float
    max_photo_bytes: int
    max_document_bytes: int
    max_sticker_bytes: int
    max_sticker_frames: int
    max_visual_file_frames: int
    max_inline_text_chars: int
    link_prefetch_timeout_s: float
    link_prefetch_max_urls: int
    link_prefetch_max_chars: int


@dataclass(frozen=True)
class ContextConfig:
    compact_trigger_tokens: int
    compact_target_tokens: int
    compact_batch_tokens: int
    compact_keep_recent_ratio: float
    compact_tool_ratio_threshold: float
    compact_tool_min_tokens: int
    compact_min_messages: int
    min_raw_messages_reserve: int


@dataclass(frozen=True)
class AppConfig:
    data_dir: Path
    log_level: str
    default_provider: str
    default_chat_mode: str
    default_process_visibility: str
    default_response_delivery: str
    default_sticker_mode: str
    default_prompt_injection_mode: str
    default_tool_history_mode: str
    default_link_prefetch_mode: str
    default_chat_max_rounds: int
    default_assist_max_rounds: int
    default_agent_max_rounds: int
    default_group_spontaneous_reply_chance: int
    default_group_spontaneous_reply_delay_s: float
    default_private_reply_delay_s: float
    default_group_reply_delay_s: float
    default_provider_retry_count: int
    default_metadata_injection_mode: str
    default_metadata_timezone: str
    default_system_prompt: str
    telegram: TelegramConfig
    openai: OpenAIConfig
    gemini: GeminiConfig
    ssh_exec: SSHExecConfig
    context: ContextConfig

    @property
    def db_path(self) -> Path:
        return self.data_dir / "tgchatbot.sqlite3"

    @property
    def artifact_dir(self) -> Path:
        return self.data_dir / "artifacts"

    @property
    def sticker_dir(self) -> Path:
        return self.data_dir / "stickers"

    @property
    def preset_dir(self) -> Path:
        return self.data_dir / "presets"

    @property
    def sticker_index_path(self) -> Path:
        return self.data_dir / "sticker_index.sqlite3"

    def default_model_for_provider(self, provider: str) -> str:
        if provider == 'gemini':
            return self.gemini.model
        return self.openai.model

    def default_session_settings(self) -> SessionSettings:
        provider = (self.default_provider or 'gemini').strip().lower() or 'gemini'
        return SessionSettings(
            provider=provider,
            model=self.default_model_for_provider(provider),
            mode=ChatMode(self.default_chat_mode),
            process_visibility=ProcessVisibility(self.default_process_visibility),
            response_delivery=ResponseDelivery(self.default_response_delivery),
            sticker_mode=StickerMode(self.default_sticker_mode),
            prompt_injection_mode=PromptInjectionMode(self.default_prompt_injection_mode),
            tool_history_mode=ToolHistoryMode(self.default_tool_history_mode),
            reasoning_effort=None,
            reasoning_summary=None,
            text_verbosity=None,
            include_thoughts=None,
            thinking_budget=None,
            thinking_level=None,
            native_web_search_mode='default',
            native_web_search_max=None,
            temperature=None,
            top_p=None,
            top_k=None,
            link_prefetch_mode='default',
            max_output_tokens=None,
            max_input_images=None,
            compact_target_images=None,
            compact_trigger_tokens=None,
            compact_target_tokens=None,
            compact_batch_tokens=None,
            compact_keep_recent_ratio=None,
            compact_tool_ratio_threshold=None,
            compact_tool_min_tokens=None,
            compact_min_messages=None,
            min_raw_messages_reserve=None,
            max_interaction_rounds=None,
            spontaneous_reply_chance=None,
            spontaneous_reply_idle_s=None,
            provider_retry_count=None,
            private_reply_delay_s=None,
            group_reply_delay_s=None,
            group_spontaneous_reply_delay_s=None,
            reply_delay_s=None,
            metadata_injection_mode=self.default_metadata_injection_mode,
            metadata_timezone=self.default_metadata_timezone,
            system_prompt=self.default_system_prompt,
        )


def load_config() -> AppConfig:
    data_dir = Path(os.getenv("APP_DATA_DIR", "./data")).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)

    token = os.getenv("TGBOT_TOKEN", "").strip()
    if not token:
        raise RuntimeError("TGBOT_TOKEN not set")

    default_system_prompt_value = os.getenv('DEFAULT_SYSTEM_PROMPT', '').strip() or default_system_prompt()

    return AppConfig(
        data_dir=data_dir,
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        default_provider=_choice(os.getenv("DEFAULT_PROVIDER", "gemini"), "gemini", {"openai", "gemini"}),
        default_chat_mode=_choice(os.getenv('DEFAULT_CHAT_MODE', 'chat'), 'chat', {mode.value for mode in ChatMode}),
        default_process_visibility=_choice(os.getenv('DEFAULT_PROCESS_VISIBILITY', 'status'), 'status', {value.value for value in ProcessVisibility}),
        default_response_delivery=_choice(os.getenv("DEFAULT_RESPONSE_DELIVERY", "edit"), "edit", {value.value for value in ResponseDelivery}),
        default_sticker_mode=_choice(os.getenv("DEFAULT_STICKER_MODE", "off"), "off", {value.value for value in StickerMode}),
        default_prompt_injection_mode=_choice(os.getenv("DEFAULT_PROMPT_INJECTION_MODE", "augment"), "augment", {value.value for value in PromptInjectionMode}),
        default_tool_history_mode=_choice(os.getenv("DEFAULT_TOOL_HISTORY_MODE", "translated"), "translated", {value.value for value in ToolHistoryMode}),
        default_link_prefetch_mode=_choice(os.getenv("DEFAULT_LINK_PREFETCH_MODE", "off"), "off", {"off", "title", "snippet"}),
        default_chat_max_rounds=parse_bounded_int_env(os.getenv("DEFAULT_CHAT_MAX_ROUNDS"), default=1, minimum=MAX_INTERACTION_ROUNDS_MIN, maximum=MAX_INTERACTION_ROUNDS_MAX),
        default_assist_max_rounds=parse_bounded_int_env(os.getenv("DEFAULT_ASSIST_MAX_ROUNDS"), default=4, minimum=MAX_INTERACTION_ROUNDS_MIN, maximum=MAX_INTERACTION_ROUNDS_MAX),
        default_agent_max_rounds=parse_bounded_int_env(os.getenv("DEFAULT_AGENT_MAX_ROUNDS"), default=6, minimum=MAX_INTERACTION_ROUNDS_MIN, maximum=MAX_INTERACTION_ROUNDS_MAX),
        default_group_spontaneous_reply_chance=parse_bounded_int_env(os.getenv("DEFAULT_GROUP_SPONTANEOUS_REPLY_CHANCE"), default=0, minimum=SPONTANEOUS_REPLY_CHANCE_MIN, maximum=SPONTANEOUS_REPLY_CHANCE_MAX),
        default_group_spontaneous_reply_delay_s=parse_bounded_float_env(os.getenv('DEFAULT_GROUP_SPONTANEOUS_REPLY_DELAY_S', os.getenv('DEFAULT_GROUP_SPONTANEOUS_REPLY_IDLE_S')), default=1200.0, minimum=0.0, maximum=86400.0),
        default_private_reply_delay_s=parse_bounded_float_env(os.getenv('DEFAULT_PRIVATE_REPLY_DELAY_S'), default=0.0, minimum=0.0, maximum=600.0),
        default_group_reply_delay_s=parse_bounded_float_env(os.getenv('DEFAULT_GROUP_REPLY_DELAY_S', os.getenv('TGBOT_GROUP_REPLY_DELAY_S')), default=5.0, minimum=0.0, maximum=600.0),
        default_provider_retry_count=parse_bounded_int_env(os.getenv("DEFAULT_PROVIDER_RETRY_COUNT"), default=1, minimum=PROVIDER_RETRY_COUNT_MIN, maximum=PROVIDER_RETRY_COUNT_MAX),
        default_metadata_injection_mode=_choice(os.getenv('DEFAULT_METADATA_INJECTION_MODE', 'on'), 'on', {'on', 'off'}),
        default_metadata_timezone=os.getenv('DEFAULT_METADATA_TIMEZONE', 'UTC').strip() or 'UTC',
        default_system_prompt=default_system_prompt_value,
        telegram=TelegramConfig(
            token=token,
            whitelist=tuple(_split_csv(os.getenv("TGBOT_WHITELIST"))),
            control_uids=tuple(_split_csv(os.getenv("TGBOT_CONTROL_UID_WHITELIST"))),
            keywords=tuple(_split_csv(os.getenv("TGBOT_KEYWORDS", "bot,reply"))),
            ignore_keywords=tuple(_split_csv(os.getenv("TGBOT_IGNORE_KEYWORDS", ""))),
            reply_to_user_message=_bool(os.getenv("TGBOT_REPLY_TO_USER_MESSAGE"), False),
            min_edit_interval_s=float(os.getenv("TGBOT_MIN_EDIT_INTERVAL_S", "1.25")),
            max_photo_bytes=int(os.getenv("TGBOT_MAX_PHOTO_BYTES", str(3 * 1024 * 1024))),
            max_document_bytes=int(os.getenv("TGBOT_MAX_DOCUMENT_BYTES", str(12 * 1024 * 1024))),
            max_sticker_bytes=int(os.getenv("TGBOT_MAX_STICKER_BYTES", str(3 * 1024 * 1024))),
            max_sticker_frames=int(os.getenv("TGBOT_MAX_STICKER_FRAMES", "4")),
            max_visual_file_frames=int(os.getenv("TGBOT_MAX_VISUAL_FILE_FRAMES", "10")),
            max_inline_text_chars=int(os.getenv("TGBOT_MAX_INLINE_TEXT_CHARS", "8000")),
            link_prefetch_timeout_s=float(os.getenv("TGBOT_LINK_PREFETCH_TIMEOUT_S", "4.0")),
            link_prefetch_max_urls=int(os.getenv("TGBOT_LINK_PREFETCH_MAX_URLS", "2")),
            link_prefetch_max_chars=int(os.getenv("TGBOT_LINK_PREFETCH_MAX_CHARS", "1200")),
        ),
        openai=OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
            model=os.getenv("OPENAI_MODEL", "gpt-5.4-nano").strip() or "gpt-5.4-nano",
            reasoning_effort=normalize_optional_choice(os.getenv("OPENAI_REASONING_EFFORT"), REASONING_EFFORT_VALUES) or "none",
            reasoning_summary=effective_reasoning_summary(os.getenv("OPENAI_REASONING_SUMMARY"), provider="openai", default="off"),
            text_verbosity=normalize_optional_choice(os.getenv("OPENAI_TEXT_VERBOSITY"), TEXT_VERBOSITY_VALUES) or "low",
            max_output_tokens=parse_bounded_int_env(os.getenv("OPENAI_MAX_OUTPUT_TOKENS"), default=4096, minimum=MAX_OUTPUT_TOKENS_MIN, maximum=MAX_OUTPUT_TOKENS_MAX),
            max_input_images=parse_optional_disabled_int_env(os.getenv("OPENAI_MAX_INPUT_IMAGES"), default=IMAGE_LIMIT_DISABLED, maximum=IMAGE_LIMIT_MAX),
            compact_target_images=parse_optional_disabled_int_env(os.getenv("OPENAI_COMPACT_TARGET_IMAGES"), default=IMAGE_LIMIT_DISABLED, maximum=IMAGE_LIMIT_MAX),
            enable_native_web_search=_bool(os.getenv("OPENAI_ENABLE_NATIVE_WEB_SEARCH"), False),
            native_web_search_max=parse_optional_disabled_int_env(os.getenv("OPENAI_NATIVE_WEB_SEARCH_MAX"), default=1, maximum=NATIVE_WEB_SEARCH_MAX_MAX),
            request_timeout_s=parse_bounded_float_env(os.getenv('OPENAI_REQUEST_TIMEOUT_S'), default=60.0, minimum=0.1, maximum=3600.0),
            connect_timeout_s=parse_bounded_float_env(os.getenv('OPENAI_CONNECT_TIMEOUT_S'), default=15.0, minimum=0.1, maximum=3600.0),
        ),
        gemini=GeminiConfig(
            api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            base_url=os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta").rstrip("/"),
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash",
            temperature=parse_bounded_float_env(os.getenv("GEMINI_TEMPERATURE"), default=1.0, minimum=TEMPERATURE_MIN, maximum=TEMPERATURE_MAX),
            top_p=parse_bounded_float_env(os.getenv("GEMINI_TOP_P"), default=0.95, minimum=TOP_P_MIN, maximum=TOP_P_MAX),
            top_k=parse_bounded_int_env(os.getenv("GEMINI_TOP_K"), default=40, minimum=TOP_K_MIN, maximum=TOP_K_MAX),
            include_thoughts=_bool(os.getenv("GEMINI_INCLUDE_THOUGHTS"), False),
            thinking_budget=parse_optional_bounded_int_env(os.getenv("GEMINI_THINKING_BUDGET"), minimum=GEMINI_THINKING_BUDGET_MIN, maximum=GEMINI_THINKING_BUDGET_MAX),
            thinking_level=normalize_optional_choice(os.getenv("GEMINI_THINKING_LEVEL"), GEMINI_THINKING_LEVEL_VALUES),
            enable_native_web_search=_bool(os.getenv("GEMINI_ENABLE_NATIVE_WEB_SEARCH"), False),
            max_output_tokens=parse_bounded_int_env(os.getenv("GEMINI_MAX_OUTPUT_TOKENS"), default=8192, minimum=MAX_OUTPUT_TOKENS_MIN, maximum=MAX_OUTPUT_TOKENS_MAX),
            max_input_images=parse_optional_disabled_int_env(os.getenv("GEMINI_MAX_INPUT_IMAGES"), default=3600, maximum=IMAGE_LIMIT_MAX),
            compact_target_images=parse_optional_disabled_int_env(os.getenv("GEMINI_COMPACT_TARGET_IMAGES"), default=3000, maximum=IMAGE_LIMIT_MAX),
            request_timeout_s=parse_bounded_float_env(os.getenv('GEMINI_REQUEST_TIMEOUT_S'), default=60.0, minimum=0.1, maximum=3600.0),
            connect_timeout_s=parse_bounded_float_env(os.getenv('GEMINI_CONNECT_TIMEOUT_S'), default=15.0, minimum=0.1, maximum=3600.0),
        ),
        ssh_exec=SSHExecConfig(
            enabled=_bool(os.getenv("SSH_EXEC_ENABLED"), True),
            host=os.getenv("SSH_EXEC_HOST", "").strip(),
            port=int(os.getenv("SSH_EXEC_PORT", "22")),
            workdir=os.getenv("SSH_EXEC_WORKDIR", "/tmp/tgchatbot").strip(),
            identity_file=(os.getenv("SSH_EXEC_IDENTITY_FILE", "").strip() or None),
            connect_timeout_s=int(os.getenv("SSH_EXEC_CONNECT_TIMEOUT_S", "10")),
            default_timeout_s=int(os.getenv("SSH_EXEC_DEFAULT_TIMEOUT_S", "20")),
            max_tool_timeout_s=int(os.getenv('SSH_EXEC_MAX_TOOL_TIMEOUT_S', '120')),
            max_stdout_chars=int(os.getenv("SSH_EXEC_MAX_STDOUT_CHARS", "16000")),
            max_stderr_chars=int(os.getenv("SSH_EXEC_MAX_STDERR_CHARS", "8000")),
            max_input_files=int(os.getenv("SSH_EXEC_MAX_INPUT_FILES", "6")),
            max_input_file_bytes=int(os.getenv("SSH_EXEC_MAX_INPUT_FILE_BYTES", str(8 * 1024 * 1024))),
            max_output_files=int(os.getenv("SSH_EXEC_MAX_OUTPUT_FILES", "5")),
            max_output_file_bytes=int(os.getenv("SSH_EXEC_MAX_OUTPUT_FILE_BYTES", str(8 * 1024 * 1024))),
            server_alive_interval_s=int(os.getenv('SSH_EXEC_SERVER_ALIVE_INTERVAL_S', '30')),
            server_alive_count_max=int(os.getenv('SSH_EXEC_SERVER_ALIVE_COUNT_MAX', '3')),
            control_persist_s=int(os.getenv('SSH_EXEC_CONTROL_PERSIST_S', '600')),
        ),
        context=ContextConfig(
            compact_trigger_tokens=parse_bounded_int_env(os.getenv("CONTEXT_COMPACT_TRIGGER_TOKENS"), default=300000, minimum=COMPACT_TOKEN_MIN, maximum=COMPACT_TOKEN_MAX),
            compact_target_tokens=parse_bounded_int_env(os.getenv("CONTEXT_COMPACT_TARGET_TOKENS"), default=100000, minimum=COMPACT_TOKEN_MIN, maximum=COMPACT_TOKEN_MAX),
            compact_batch_tokens=parse_bounded_int_env(os.getenv("CONTEXT_COMPACT_BATCH_TOKENS"), default=40000, minimum=COMPACT_TOKEN_MIN, maximum=COMPACT_TOKEN_MAX),
            compact_keep_recent_ratio=parse_bounded_float_env(os.getenv("CONTEXT_COMPACT_KEEP_RECENT_RAW_TOKEN_RATIO"), default=0.5, minimum=COMPACT_KEEP_RECENT_RATIO_MIN, maximum=COMPACT_KEEP_RECENT_RATIO_MAX),
            compact_tool_ratio_threshold=parse_bounded_float_env(os.getenv("CONTEXT_COMPACT_TOOL_RATIO_THRESHOLD"), default=10.0, minimum=COMPACT_TOOL_RATIO_THRESHOLD_MIN, maximum=COMPACT_TOOL_RATIO_THRESHOLD_MAX),
            compact_tool_min_tokens=parse_bounded_int_env(os.getenv("CONTEXT_COMPACT_TOOL_MIN_TOKENS"), default=10000, minimum=COMPACT_TOKEN_MIN, maximum=COMPACT_TOKEN_MAX),
            compact_min_messages=parse_bounded_int_env(os.getenv("CONTEXT_COMPACT_MIN_MESSAGES"), default=24, minimum=COMPACT_MIN_MESSAGES_MIN, maximum=COMPACT_MIN_MESSAGES_MAX),
            min_raw_messages_reserve=parse_bounded_int_env(os.getenv("CONTEXT_MIN_RAW_MESSAGES_RESERVE"), default=8, minimum=MIN_RAW_MESSAGES_RESERVE_MIN, maximum=MIN_RAW_MESSAGES_RESERVE_MAX),
        ),
    )
