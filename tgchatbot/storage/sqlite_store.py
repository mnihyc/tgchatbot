from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
UTC = ZoneInfo('UTC')

from tgchatbot.core.context_state import MemoryBlock, StoredConversationMessage
from tgchatbot.core.token_estimator import TokenEstimator
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
    ToolHistoryMode,
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
    GROUP_SPONTANEOUS_REPLY_DELAY_MAX_S,
    IMAGE_LIMIT_MAX,
    MAX_INTERACTION_ROUNDS_MAX,
    MAX_INTERACTION_ROUNDS_MIN,
    MAX_OUTPUT_TOKENS_MAX,
    MAX_OUTPUT_TOKENS_MIN,
    MIN_RAW_MESSAGES_RESERVE_MAX,
    MIN_RAW_MESSAGES_RESERVE_MIN,
    NATIVE_WEB_SEARCH_MAX_MAX,
    PROVIDER_RETRY_COUNT_MAX,
    PROVIDER_RETRY_COUNT_MIN,
    REASONING_EFFORT_VALUES,
    REASONING_SUMMARY_VALUES,
    REPLY_DELAY_MAX_S,
    SPONTANEOUS_REPLY_CHANCE_MAX,
    SPONTANEOUS_REPLY_CHANCE_MIN,
    TEMPERATURE_MAX,
    TEMPERATURE_MIN,
    TEXT_VERBOSITY_VALUES,
    TOP_K_MAX,
    TOP_K_MIN,
    TOP_P_MAX,
    TOP_P_MIN,
    clamp_float,
    clamp_int,
    normalize_choice,
    normalize_optional_bounded_int,
    normalize_optional_choice,
    normalize_optional_disabled_int,
    normalize_reasoning_summary_value,
)


class SQLiteStore:
    SESSION_COLUMNS = (
        'provider',
        'model',
        'mode',
        'process_visibility',
        'response_delivery',
        'sticker_mode',
        'prompt_injection_mode',
        'tool_history_mode',
        'reasoning_effort',
        'reasoning_summary',
        'text_verbosity',
        'include_thoughts',
        'thinking_budget',
        'thinking_level',
        'native_web_search_mode',
        'native_web_search_max',
        'temperature',
        'top_p',
        'top_k',
        'link_prefetch_mode',
        'max_output_tokens',
        'max_input_images',
        'compact_target_images',
        'compact_trigger_tokens',
        'compact_target_tokens',
        'compact_batch_tokens',
        'compact_keep_recent_ratio',
        'compact_tool_ratio_threshold',
        'compact_tool_min_tokens',
        'compact_min_messages',
        'min_raw_messages_reserve',
        'max_interaction_rounds',
        'spontaneous_reply_chance',
        'spontaneous_reply_idle_s',
        'provider_retry_count',
        'private_reply_delay_s',
        'group_reply_delay_s',
        'group_spontaneous_reply_delay_s',
        'reply_delay_s',
        'metadata_injection_mode',
        'metadata_timezone',
        'system_prompt',
    )

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('PRAGMA synchronous=NORMAL')
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    process_visibility TEXT NOT NULL,
                    response_delivery TEXT NOT NULL DEFAULT 'edit',
                    sticker_mode TEXT NOT NULL DEFAULT 'off',
                    prompt_injection_mode TEXT NOT NULL DEFAULT 'augment',
                    tool_history_mode TEXT NOT NULL DEFAULT 'translated',
                    reasoning_effort TEXT,
                    reasoning_summary TEXT,
                    text_verbosity TEXT,
                    include_thoughts INTEGER,
                    thinking_budget INTEGER,
                    thinking_level TEXT,
                    native_web_search_mode TEXT NOT NULL DEFAULT 'default',
                    native_web_search_max INTEGER,
                    temperature REAL,
                    top_p REAL,
                    top_k INTEGER,
                    link_prefetch_mode TEXT NOT NULL DEFAULT 'default',
                    max_output_tokens INTEGER,
                    max_input_images INTEGER,
                    compact_target_images INTEGER,
                    compact_trigger_tokens INTEGER,
                    compact_target_tokens INTEGER,
                    compact_batch_tokens INTEGER,
                    compact_keep_recent_ratio REAL,
                    compact_tool_ratio_threshold REAL,
                    compact_tool_min_tokens INTEGER,
                    compact_min_messages INTEGER,
                    min_raw_messages_reserve INTEGER,
                    max_interaction_rounds INTEGER,
                    spontaneous_reply_chance INTEGER,
                    spontaneous_reply_idle_s INTEGER,
                    provider_retry_count INTEGER,
                    private_reply_delay_s REAL,
                    group_reply_delay_s REAL,
                    group_spontaneous_reply_delay_s REAL,
                    reply_delay_s REAL,
                    metadata_injection_mode TEXT NOT NULL DEFAULT 'on',
                    metadata_timezone TEXT NOT NULL DEFAULT 'UTC',
                    system_prompt TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    name TEXT,
                    payload_json TEXT NOT NULL,
                    estimated_tokens INTEGER NOT NULL DEFAULT 0,
                    compacted INTEGER NOT NULL DEFAULT 0,
                    compacted_level INTEGER,
                    compacted_by_block_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id, id);
                CREATE INDEX IF NOT EXISTS idx_messages_session_compacted ON messages(session_id, compacted, id);
                CREATE INDEX IF NOT EXISTS idx_messages_session_compacted_block ON messages(session_id, compacted_by_block_id, id);

                CREATE TABLE IF NOT EXISTS compaction_blocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    sequence_no INTEGER NOT NULL,
                    summary_text TEXT NOT NULL,
                    estimated_tokens INTEGER NOT NULL,
                    source_message_count INTEGER NOT NULL,
                    start_message_id INTEGER,
                    end_message_id INTEGER,
                    level INTEGER NOT NULL DEFAULT 1,
                    kind TEXT NOT NULL DEFAULT 'episode',
                    lifecycle TEXT NOT NULL DEFAULT 'sealed',
                    source_kind TEXT NOT NULL DEFAULT 'raw',
                    parent_block_ids_json TEXT NOT NULL DEFAULT '[]',
                    topic_labels_json TEXT NOT NULL DEFAULT '[]',
                    actor_labels_json TEXT NOT NULL DEFAULT '[]',
                    time_start TEXT,
                    time_end TEXT,
                    retained_raw_excerpt_count INTEGER NOT NULL DEFAULT 0,
                    validator_status TEXT,
                    validator_score REAL,
                    structured_data_json TEXT NOT NULL DEFAULT '{}',
                    hidden INTEGER NOT NULL DEFAULT 0,
                    superseded_at TIMESTAMP,
                    superseded_by_block_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS idx_compaction_blocks_session_id ON compaction_blocks(session_id, sequence_no);
                """
            )
            session_columns = {row['name'] for row in conn.execute("PRAGMA table_info(sessions)")}
            if 'response_delivery' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN response_delivery TEXT NOT NULL DEFAULT 'edit'")
            if 'sticker_mode' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN sticker_mode TEXT NOT NULL DEFAULT 'off'")
            if 'prompt_injection_mode' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN prompt_injection_mode TEXT NOT NULL DEFAULT 'augment'")
            if 'tool_history_mode' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN tool_history_mode TEXT NOT NULL DEFAULT 'translated'")
            if 'reasoning_effort' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN reasoning_effort TEXT")
            if 'reasoning_summary' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN reasoning_summary TEXT")
            if 'text_verbosity' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN text_verbosity TEXT")
            if 'include_thoughts' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN include_thoughts INTEGER")
            if 'thinking_budget' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN thinking_budget INTEGER")
            if 'thinking_level' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN thinking_level TEXT")
            if 'native_web_search_mode' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN native_web_search_mode TEXT NOT NULL DEFAULT 'default'")
            if 'native_web_search_max' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN native_web_search_max INTEGER")
            if 'temperature' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN temperature REAL")
            if 'top_p' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN top_p REAL")
            if 'top_k' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN top_k INTEGER")
            if 'link_prefetch_mode' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN link_prefetch_mode TEXT NOT NULL DEFAULT 'default'")
            if 'max_output_tokens' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN max_output_tokens INTEGER")
            if 'max_input_images' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN max_input_images INTEGER")
            if 'compact_target_images' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN compact_target_images INTEGER")
            if 'compact_trigger_tokens' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN compact_trigger_tokens INTEGER")
            if 'compact_target_tokens' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN compact_target_tokens INTEGER")
            if 'compact_batch_tokens' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN compact_batch_tokens INTEGER")
            if 'compact_keep_recent_ratio' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN compact_keep_recent_ratio REAL")
            if 'compact_tool_ratio_threshold' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN compact_tool_ratio_threshold REAL")
            if 'compact_tool_min_tokens' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN compact_tool_min_tokens INTEGER")
            if 'compact_min_messages' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN compact_min_messages INTEGER")
            if 'min_raw_messages_reserve' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN min_raw_messages_reserve INTEGER")
            if 'max_interaction_rounds' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN max_interaction_rounds INTEGER")
            if 'spontaneous_reply_chance' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN spontaneous_reply_chance INTEGER")
            if 'spontaneous_reply_idle_s' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN spontaneous_reply_idle_s INTEGER")
            if 'provider_retry_count' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN provider_retry_count INTEGER")
            if 'private_reply_delay_s' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN private_reply_delay_s REAL")
            if 'group_reply_delay_s' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN group_reply_delay_s REAL")
            if 'group_spontaneous_reply_delay_s' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN group_spontaneous_reply_delay_s REAL")
            if 'reply_delay_s' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN reply_delay_s REAL")
            if 'metadata_injection_mode' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN metadata_injection_mode TEXT NOT NULL DEFAULT 'on'")
            if 'metadata_timezone' not in session_columns:
                conn.execute("ALTER TABLE sessions ADD COLUMN metadata_timezone TEXT NOT NULL DEFAULT 'UTC'")

            message_columns = {row['name'] for row in conn.execute("PRAGMA table_info(messages)")}
            if 'estimated_tokens' not in message_columns:
                conn.execute("ALTER TABLE messages ADD COLUMN estimated_tokens INTEGER NOT NULL DEFAULT 0")
            if 'compacted' not in message_columns:
                conn.execute("ALTER TABLE messages ADD COLUMN compacted INTEGER NOT NULL DEFAULT 0")
            if 'compacted_level' not in message_columns:
                conn.execute("ALTER TABLE messages ADD COLUMN compacted_level INTEGER")
            if 'compacted_by_block_id' not in message_columns:
                conn.execute("ALTER TABLE messages ADD COLUMN compacted_by_block_id INTEGER")
            if 'hidden' not in message_columns:
                conn.execute("ALTER TABLE messages ADD COLUMN hidden INTEGER NOT NULL DEFAULT 0")
            block_columns = {row['name'] for row in conn.execute("PRAGMA table_info(compaction_blocks)")}
            if 'hidden' not in block_columns:
                conn.execute("ALTER TABLE compaction_blocks ADD COLUMN hidden INTEGER NOT NULL DEFAULT 0")
            if 'superseded_at' not in block_columns:
                conn.execute("ALTER TABLE compaction_blocks ADD COLUMN superseded_at TIMESTAMP")
            if 'superseded_by_block_id' not in block_columns:
                conn.execute("ALTER TABLE compaction_blocks ADD COLUMN superseded_by_block_id INTEGER")
            if 'level' not in block_columns:
                conn.execute("ALTER TABLE compaction_blocks ADD COLUMN level INTEGER NOT NULL DEFAULT 1")
            if 'kind' not in block_columns:
                conn.execute("ALTER TABLE compaction_blocks ADD COLUMN kind TEXT NOT NULL DEFAULT 'episode'")
            if 'lifecycle' not in block_columns:
                conn.execute("ALTER TABLE compaction_blocks ADD COLUMN lifecycle TEXT NOT NULL DEFAULT 'sealed'")
            if 'source_kind' not in block_columns:
                conn.execute("ALTER TABLE compaction_blocks ADD COLUMN source_kind TEXT NOT NULL DEFAULT 'raw'")
            if 'parent_block_ids_json' not in block_columns:
                conn.execute("ALTER TABLE compaction_blocks ADD COLUMN parent_block_ids_json TEXT NOT NULL DEFAULT '[]'")
            if 'topic_labels_json' not in block_columns:
                conn.execute("ALTER TABLE compaction_blocks ADD COLUMN topic_labels_json TEXT NOT NULL DEFAULT '[]'")
            if 'actor_labels_json' not in block_columns:
                conn.execute("ALTER TABLE compaction_blocks ADD COLUMN actor_labels_json TEXT NOT NULL DEFAULT '[]'")
            if 'time_start' not in block_columns:
                conn.execute("ALTER TABLE compaction_blocks ADD COLUMN time_start TEXT")
            if 'time_end' not in block_columns:
                conn.execute("ALTER TABLE compaction_blocks ADD COLUMN time_end TEXT")
            if 'retained_raw_excerpt_count' not in block_columns:
                conn.execute("ALTER TABLE compaction_blocks ADD COLUMN retained_raw_excerpt_count INTEGER NOT NULL DEFAULT 0")
            if 'validator_status' not in block_columns:
                conn.execute("ALTER TABLE compaction_blocks ADD COLUMN validator_status TEXT")
            if 'validator_score' not in block_columns:
                conn.execute("ALTER TABLE compaction_blocks ADD COLUMN validator_score REAL")
            if 'structured_data_json' not in block_columns:
                conn.execute("ALTER TABLE compaction_blocks ADD COLUMN structured_data_json TEXT NOT NULL DEFAULT '{}'")

    @staticmethod
    def _safe_enum(enum_cls, value: str | None, default):
        try:
            return enum_cls(value) if value is not None else default
        except Exception:
            return default

    def _session_values(self, settings: SessionSettings) -> tuple:
        return (
            settings.provider,
            settings.model,
            settings.mode.value,
            settings.process_visibility.value,
            settings.response_delivery.value,
            settings.sticker_mode.value,
            settings.prompt_injection_mode.value,
            settings.tool_history_mode.value,
            settings.reasoning_effort,
            settings.reasoning_summary,
            settings.text_verbosity,
            (1 if settings.include_thoughts else 0) if settings.include_thoughts is not None else None,
            settings.thinking_budget,
            settings.thinking_level,
            settings.native_web_search_mode,
            settings.native_web_search_max,
            settings.temperature,
            settings.top_p,
            settings.top_k,
            settings.link_prefetch_mode,
            settings.max_output_tokens,
            settings.max_input_images,
            settings.compact_target_images,
            settings.compact_trigger_tokens,
            settings.compact_target_tokens,
            settings.compact_batch_tokens,
            settings.compact_keep_recent_ratio,
            settings.compact_tool_ratio_threshold,
            settings.compact_tool_min_tokens,
            settings.compact_min_messages,
            settings.min_raw_messages_reserve,
            settings.max_interaction_rounds,
            settings.spontaneous_reply_chance,
            settings.spontaneous_reply_idle_s,
            settings.provider_retry_count,
            settings.private_reply_delay_s,
            settings.group_reply_delay_s,
            settings.group_spontaneous_reply_delay_s,
            settings.reply_delay_s,
            settings.metadata_injection_mode,
            settings.metadata_timezone,
            settings.system_prompt,
        )

    def _row_to_session_settings(self, row: sqlite3.Row, defaults: SessionSettings) -> SessionSettings:
        provider = (str(row['provider'] or '').strip().lower() or defaults.provider)
        model = str(row['model'] or '').strip() or defaults.model
        return SessionSettings(
            provider=provider,
            model=model,
            mode=self._safe_enum(ChatMode, row['mode'], defaults.mode),
            process_visibility=self._safe_enum(ProcessVisibility, row['process_visibility'], defaults.process_visibility),
            response_delivery=self._safe_enum(ResponseDelivery, row['response_delivery'], defaults.response_delivery),
            sticker_mode=self._safe_enum(StickerMode, row['sticker_mode'], defaults.sticker_mode),
            prompt_injection_mode=self._safe_enum(PromptInjectionMode, row['prompt_injection_mode'], defaults.prompt_injection_mode),
            tool_history_mode=self._safe_enum(ToolHistoryMode, row['tool_history_mode'], defaults.tool_history_mode),
            reasoning_effort=normalize_optional_choice(row['reasoning_effort'], REASONING_EFFORT_VALUES),
            reasoning_summary=normalize_reasoning_summary_value(row['reasoning_summary']),
            text_verbosity=normalize_optional_choice(row['text_verbosity'], TEXT_VERBOSITY_VALUES),
            include_thoughts=(bool(int(row['include_thoughts'])) if row['include_thoughts'] is not None else None),
            thinking_budget=normalize_optional_bounded_int(row['thinking_budget'], minimum=GEMINI_THINKING_BUDGET_MIN, maximum=GEMINI_THINKING_BUDGET_MAX),
            thinking_level=normalize_optional_choice(row['thinking_level'], GEMINI_THINKING_LEVEL_VALUES),
            native_web_search_mode=normalize_choice(row['native_web_search_mode'], 'default', {'default', 'on', 'off'}),
            native_web_search_max=normalize_optional_disabled_int(row['native_web_search_max'], maximum=NATIVE_WEB_SEARCH_MAX_MAX),
            temperature=(clamp_float(float(row['temperature']), minimum=TEMPERATURE_MIN, maximum=TEMPERATURE_MAX, default=TEMPERATURE_MIN) if row['temperature'] is not None else None),
            top_p=(clamp_float(float(row['top_p']), minimum=TOP_P_MIN, maximum=TOP_P_MAX, default=TOP_P_MIN) if row['top_p'] is not None else None),
            top_k=(clamp_int(int(row['top_k']), minimum=TOP_K_MIN, maximum=TOP_K_MAX, default=TOP_K_MIN) if row['top_k'] is not None else None),
            link_prefetch_mode=normalize_choice(row['link_prefetch_mode'], 'default', {'default', 'off', 'title', 'snippet'}),
            max_output_tokens=(clamp_int(int(row['max_output_tokens']), minimum=MAX_OUTPUT_TOKENS_MIN, maximum=MAX_OUTPUT_TOKENS_MAX, default=MAX_OUTPUT_TOKENS_MIN) if row['max_output_tokens'] is not None else None),
            max_input_images=normalize_optional_disabled_int(row['max_input_images'], maximum=IMAGE_LIMIT_MAX),
            compact_target_images=normalize_optional_disabled_int(row['compact_target_images'], maximum=IMAGE_LIMIT_MAX),
            compact_trigger_tokens=(clamp_int(int(row['compact_trigger_tokens']), minimum=COMPACT_TOKEN_MIN, maximum=COMPACT_TOKEN_MAX, default=COMPACT_TOKEN_MIN) if row['compact_trigger_tokens'] is not None else None),
            compact_target_tokens=(clamp_int(int(row['compact_target_tokens']), minimum=COMPACT_TOKEN_MIN, maximum=COMPACT_TOKEN_MAX, default=COMPACT_TOKEN_MIN) if row['compact_target_tokens'] is not None else None),
            compact_batch_tokens=(clamp_int(int(row['compact_batch_tokens']), minimum=COMPACT_TOKEN_MIN, maximum=COMPACT_TOKEN_MAX, default=COMPACT_TOKEN_MIN) if row['compact_batch_tokens'] is not None else None),
            compact_keep_recent_ratio=(clamp_float(float(row['compact_keep_recent_ratio']), minimum=COMPACT_KEEP_RECENT_RATIO_MIN, maximum=COMPACT_KEEP_RECENT_RATIO_MAX, default=COMPACT_KEEP_RECENT_RATIO_MIN) if row['compact_keep_recent_ratio'] is not None else None),
            compact_tool_ratio_threshold=(clamp_float(float(row['compact_tool_ratio_threshold']), minimum=COMPACT_TOOL_RATIO_THRESHOLD_MIN, maximum=COMPACT_TOOL_RATIO_THRESHOLD_MAX, default=COMPACT_TOOL_RATIO_THRESHOLD_MIN) if row['compact_tool_ratio_threshold'] is not None else None),
            compact_tool_min_tokens=(clamp_int(int(row['compact_tool_min_tokens']), minimum=COMPACT_TOKEN_MIN, maximum=COMPACT_TOKEN_MAX, default=COMPACT_TOKEN_MIN) if row['compact_tool_min_tokens'] is not None else None),
            compact_min_messages=(clamp_int(int(row['compact_min_messages']), minimum=COMPACT_MIN_MESSAGES_MIN, maximum=COMPACT_MIN_MESSAGES_MAX, default=COMPACT_MIN_MESSAGES_MIN) if row['compact_min_messages'] is not None else None),
            min_raw_messages_reserve=(clamp_int(int(row['min_raw_messages_reserve']), minimum=MIN_RAW_MESSAGES_RESERVE_MIN, maximum=MIN_RAW_MESSAGES_RESERVE_MAX, default=MIN_RAW_MESSAGES_RESERVE_MIN) if row['min_raw_messages_reserve'] is not None else None),
            max_interaction_rounds=(clamp_int(int(row['max_interaction_rounds']), minimum=MAX_INTERACTION_ROUNDS_MIN, maximum=MAX_INTERACTION_ROUNDS_MAX, default=MAX_INTERACTION_ROUNDS_MIN) if row['max_interaction_rounds'] is not None else None),
            spontaneous_reply_chance=(clamp_int(int(row['spontaneous_reply_chance']), minimum=SPONTANEOUS_REPLY_CHANCE_MIN, maximum=SPONTANEOUS_REPLY_CHANCE_MAX, default=SPONTANEOUS_REPLY_CHANCE_MIN) if row['spontaneous_reply_chance'] is not None else None),
            spontaneous_reply_idle_s=(clamp_int(int(row['spontaneous_reply_idle_s']), minimum=0, maximum=int(GROUP_SPONTANEOUS_REPLY_DELAY_MAX_S), default=0) if row['spontaneous_reply_idle_s'] is not None else None),
            provider_retry_count=(clamp_int(int(row['provider_retry_count']), minimum=PROVIDER_RETRY_COUNT_MIN, maximum=PROVIDER_RETRY_COUNT_MAX, default=PROVIDER_RETRY_COUNT_MIN) if row['provider_retry_count'] is not None else None),
            private_reply_delay_s=(clamp_float(float(row['private_reply_delay_s']), minimum=0.0, maximum=REPLY_DELAY_MAX_S, default=0.0) if row['private_reply_delay_s'] is not None else None),
            group_reply_delay_s=(clamp_float(float(row['group_reply_delay_s']), minimum=0.0, maximum=REPLY_DELAY_MAX_S, default=0.0) if row['group_reply_delay_s'] is not None else None),
            group_spontaneous_reply_delay_s=(clamp_float(float(row['group_spontaneous_reply_delay_s']), minimum=0.0, maximum=GROUP_SPONTANEOUS_REPLY_DELAY_MAX_S, default=0.0) if row['group_spontaneous_reply_delay_s'] is not None else None),
            reply_delay_s=(clamp_float(float(row['reply_delay_s']), minimum=0.0, maximum=REPLY_DELAY_MAX_S, default=0.0) if row['reply_delay_s'] is not None else None),
            metadata_injection_mode=normalize_choice(row['metadata_injection_mode'], defaults.metadata_injection_mode, {'on', 'off'}),
            metadata_timezone=row['metadata_timezone'] or defaults.metadata_timezone,
            system_prompt=row['system_prompt'] or defaults.system_prompt,
        )

    async def get_or_create_session(self, session_id: str, defaults: SessionSettings) -> SessionSettings:
        return await asyncio.to_thread(self._get_or_create_session_sync, session_id, defaults)

    def _get_or_create_session_sync(self, session_id: str, defaults: SessionSettings) -> SessionSettings:
        select_columns = ', '.join(self.SESSION_COLUMNS)
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {select_columns} FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if row:
                return self._row_to_session_settings(row, defaults)
            placeholders = ', '.join(['?'] * (1 + len(self.SESSION_COLUMNS)))
            conn.execute(
                f"INSERT INTO sessions (session_id, {select_columns}) VALUES ({placeholders})",
                (session_id, *self._session_values(defaults)),
            )
            return defaults

    async def save_session(self, session_id: str, settings: SessionSettings) -> None:
        await asyncio.to_thread(self._save_session_sync, session_id, settings)

    def _save_session_sync(self, session_id: str, settings: SessionSettings) -> None:
        columns = ', '.join(self.SESSION_COLUMNS)
        placeholders = ', '.join(['?'] * (1 + len(self.SESSION_COLUMNS)))
        update_clause = ',\n                    '.join(f"{column} = excluded.{column}" for column in self.SESSION_COLUMNS)
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO sessions (session_id, {columns})
                VALUES ({placeholders})
                ON CONFLICT(session_id) DO UPDATE SET
                    {update_clause},
                    updated_at = CURRENT_TIMESTAMP
                """,
                (session_id, *self._session_values(settings)),
            )

    async def append_message(self, session_id: str, message: ConversationMessage, estimated_tokens: int | None = None) -> StoredConversationMessage:
        return await asyncio.to_thread(self._append_message_sync, session_id, message, estimated_tokens)

    def _append_message_sync(self, session_id: str, message: ConversationMessage, estimated_tokens: int | None) -> StoredConversationMessage:
        payload = {
            'parts': [self._serialize_part(part) for part in message.parts],
            'metadata': message.metadata,
        }
        estimate = estimated_tokens if estimated_tokens is not None else TokenEstimator.estimate_message(message)
        payload_json = json.dumps(payload, ensure_ascii=False)
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO messages (session_id, role, name, payload_json, estimated_tokens) VALUES (?, ?, ?, ?, ?)",
                (session_id, message.role.value, message.name, payload_json, estimate),
            )
            message_id = int(cursor.lastrowid)
            row = conn.execute(
                "SELECT created_at FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()
            created_at = int(datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()) if row and row['created_at'] is not None else None
        return StoredConversationMessage(db_id=message_id, message=message, estimated_tokens=estimate, created_at=created_at)

    async def update_message(self, session_id: str, stored_message: StoredConversationMessage) -> StoredConversationMessage:
        return await asyncio.to_thread(self._update_message_sync, session_id, stored_message)

    def _update_message_sync(self, session_id: str, stored_message: StoredConversationMessage) -> StoredConversationMessage:
        payload = {
            'parts': [self._serialize_part(part) for part in stored_message.message.parts],
            'metadata': stored_message.message.metadata,
        }
        estimate = TokenEstimator.estimate_message(stored_message.message)
        payload_json = json.dumps(payload, ensure_ascii=False)
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE messages SET role = ?, name = ?, payload_json = ?, estimated_tokens = ? WHERE session_id = ? AND id = ?",
                (stored_message.message.role.value, stored_message.message.name, payload_json, estimate, session_id, stored_message.db_id),
            )
            if int(cursor.rowcount or 0) <= 0:
                raise KeyError(f"message {stored_message.db_id} not found for session {session_id!r}")
            row = conn.execute(
                "SELECT created_at FROM messages WHERE id = ?",
                (stored_message.db_id,),
            ).fetchone()
            created_at = int(datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()) if row and row['created_at'] is not None else stored_message.created_at
        return StoredConversationMessage(db_id=stored_message.db_id, message=stored_message.message, estimated_tokens=estimate, created_at=created_at)

    async def count_sessions(self) -> int:
        return await asyncio.to_thread(self._count_sessions_sync)

    def _count_sessions_sync(self) -> int:
        with self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) AS c FROM sessions").fetchone()
            return int(row['c'] or 0)

    @staticmethod
    def _serialize_part(part: MessagePart) -> dict:
        data = {
            'kind': part.kind.value,
            'text': part.text,
            'mime_type': part.mime_type,
            'filename': part.filename,
            'data_b64': part.data_b64,
            'artifact_path': part.artifact_path,
            'size_bytes': part.size_bytes,
            'detail': part.detail,
            'remote_sync': part.remote_sync,
            'origin': part.origin,
        }
        return {key: value for key, value in data.items() if value is not None}

    @staticmethod
    def _deserialize_part(part: dict) -> MessagePart:
        payload = dict(part)
        kind = payload.pop('kind')
        return MessagePart(kind=PartKind(kind), **payload)

    @staticmethod
    def _row_to_message(row: sqlite3.Row) -> ConversationMessage:
        payload = json.loads(row['payload_json'])
        parts = [SQLiteStore._deserialize_part(part) for part in payload.get('parts', [])]
        return ConversationMessage(
            role=MessageRole(row['role']),
            name=row['name'],
            parts=parts,
            metadata=payload.get('metadata', {}),
        )

    @staticmethod
    def _decode_json_list(value: str | None) -> tuple:
        if not value:
            return ()
        try:
            parsed = json.loads(value)
        except Exception:
            return ()
        if isinstance(parsed, list):
            return tuple(item for item in parsed if isinstance(item, (str, int)))
        return ()

    @staticmethod
    def _decode_json_dict(value: str | None) -> dict:
        if not value:
            return {}
        try:
            parsed = json.loads(value)
        except Exception:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _hydrate_memory_block(self, row: sqlite3.Row) -> MemoryBlock:
        return MemoryBlock(
            block_id=int(row['id']),
            sequence_no=int(row['sequence_no']),
            summary_text=row['summary_text'],
            estimated_tokens=int(row['estimated_tokens']),
            source_message_count=int(row['source_message_count']),
            start_message_id=row['start_message_id'],
            end_message_id=row['end_message_id'],
            level=int(row['level']) if row['level'] is not None else 1,
            kind=str(row['kind'] or 'episode'),
            lifecycle=str(row['lifecycle'] or 'sealed'),
            source_kind=str(row['source_kind'] or 'raw'),
            parent_block_ids=tuple(int(item) for item in self._decode_json_list(row['parent_block_ids_json'])),
            topic_labels=tuple(str(item) for item in self._decode_json_list(row['topic_labels_json'])),
            actor_labels=tuple(str(item) for item in self._decode_json_list(row['actor_labels_json'])),
            time_start=row['time_start'],
            time_end=row['time_end'],
            retained_raw_excerpt_count=int(row['retained_raw_excerpt_count'] or 0),
            validator_status=row['validator_status'],
            validator_score=(float(row['validator_score']) if row['validator_score'] is not None else None),
            structured_data=self._decode_json_dict(row['structured_data_json']),
        )

    async def list_messages(self, session_id: str, limit: int = 100) -> list[ConversationMessage]:
        return await asyncio.to_thread(self._list_messages_sync, session_id, limit)

    def _list_messages_sync(self, session_id: str, limit: int) -> list[ConversationMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, name, payload_json FROM messages WHERE session_id = ? AND hidden = 0 ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        rows = list(reversed(rows))
        return [self._row_to_message(row) for row in rows]

    async def list_recent_visible_messages(self, session_id: str, limit: int = 20) -> list[StoredConversationMessage]:
        return await asyncio.to_thread(self._list_recent_visible_messages_sync, session_id, limit)

    def _list_recent_visible_messages_sync(self, session_id: str, limit: int) -> list[StoredConversationMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, role, name, payload_json, estimated_tokens, created_at FROM messages WHERE session_id = ? AND hidden = 0 ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        out: list[StoredConversationMessage] = []
        for row in rows:
            message = self._row_to_message(row)
            estimate = int(row['estimated_tokens'] or 0)
            if estimate <= 0:
                estimate = TokenEstimator.estimate_message(message)
            out.append(StoredConversationMessage(db_id=int(row['id']), message=message, estimated_tokens=estimate,
                        created_at=(int(datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()) if row['created_at'] is not None else None)))
        return out

    async def hide_messages_since(self, session_id: str, message_id: int) -> int:
        return await asyncio.to_thread(self._hide_messages_since_sync, session_id, message_id)

    def _hide_messages_since_sync(self, session_id: str, message_id: int) -> int:
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE messages SET hidden = 1 WHERE session_id = ? AND id >= ? AND hidden = 0",
                (session_id, message_id),
            )
            return int(cursor.rowcount or 0)

    async def hide_message_ids(self, session_id: str, message_ids: list[int]) -> int:
        return await asyncio.to_thread(self._hide_message_ids_sync, session_id, message_ids)

    def _hide_message_ids_sync(self, session_id: str, message_ids: list[int]) -> int:
        ids = [int(message_id) for message_id in message_ids if int(message_id) > 0]
        if not ids:
            return 0
        placeholders = ','.join('?' for _ in ids)
        with self._connect() as conn:
            cursor = conn.execute(
                f"UPDATE messages SET hidden = 1 WHERE session_id = ? AND id IN ({placeholders}) AND hidden = 0",
                (session_id, *ids),
            )
            return int(cursor.rowcount or 0)

    async def list_uncompacted_messages(self, session_id: str) -> list[StoredConversationMessage]:
        return await asyncio.to_thread(self._list_uncompacted_messages_sync, session_id)

    def _list_uncompacted_messages_sync(self, session_id: str) -> list[StoredConversationMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, role, name, payload_json, estimated_tokens, created_at FROM messages WHERE session_id = ? AND compacted = 0 AND hidden = 0 ORDER BY id ASC",
                (session_id,),
            ).fetchall()
        out: list[StoredConversationMessage] = []
        for row in rows:
            message = self._row_to_message(row)
            estimate = int(row['estimated_tokens'] or 0)
            if estimate <= 0:
                estimate = TokenEstimator.estimate_message(message)
            out.append(StoredConversationMessage(db_id=int(row['id']), message=message, estimated_tokens=estimate,
                        created_at=(int(datetime.strptime(row["created_at"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=UTC).timestamp()) if row['created_at'] is not None else None)))
        return out

    async def list_memory_blocks(self, session_id: str) -> list[MemoryBlock]:
        return await asyncio.to_thread(self._list_memory_blocks_sync, session_id)

    def _list_memory_blocks_sync(self, session_id: str) -> list[MemoryBlock]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT id, sequence_no, summary_text, estimated_tokens, source_message_count, start_message_id, end_message_id,
                       level, kind, lifecycle, source_kind, parent_block_ids_json, topic_labels_json, actor_labels_json,
                       time_start, time_end, retained_raw_excerpt_count, validator_status, validator_score, structured_data_json
                FROM compaction_blocks
                WHERE session_id = ? AND hidden = 0 AND lifecycle = 'sealed'
                ORDER BY sequence_no ASC
                """,
                (session_id,),
            ).fetchall()
        return [self._hydrate_memory_block(row) for row in rows]


    async def create_memory_block(
        self,
        session_id: str,
        *,
        summary_text: str,
        estimated_tokens: int,
        source_message_ids: list[int],
        level: int = 1,
        kind: str = 'episode',
        lifecycle: str = 'sealed',
        source_kind: str = 'raw',
        source_message_count: int | None = None,
        start_message_id: int | None = None,
        end_message_id: int | None = None,
        parent_block_ids: list[int] | tuple[int, ...] | None = None,
        topic_labels: list[str] | tuple[str, ...] | None = None,
        actor_labels: list[str] | tuple[str, ...] | None = None,
        time_start: str | None = None,
        time_end: str | None = None,
        retained_raw_excerpt_count: int = 0,
        validator_status: str | None = None,
        validator_score: float | None = None,
        structured_data: dict | None = None,
    ) -> MemoryBlock:
        return await asyncio.to_thread(
            self._create_memory_block_sync,
            session_id,
            summary_text,
            estimated_tokens,
            source_message_ids,
            level,
            kind,
            lifecycle,
            source_kind,
            source_message_count,
            start_message_id,
            end_message_id,
            list(parent_block_ids or []),
            list(topic_labels or []),
            list(actor_labels or []),
            time_start,
            time_end,
            retained_raw_excerpt_count,
            validator_status,
            validator_score,
            structured_data or {},
        )

    def _create_memory_block_sync(
        self,
        session_id: str,
        summary_text: str,
        estimated_tokens: int,
        source_message_ids: list[int],
        level: int,
        kind: str,
        lifecycle: str,
        source_kind: str,
        source_message_count: int | None,
        start_message_id: int | None,
        end_message_id: int | None,
        parent_block_ids: list[int],
        topic_labels: list[str],
        actor_labels: list[str],
        time_start: str | None,
        time_end: str | None,
        retained_raw_excerpt_count: int,
        validator_status: str | None,
        validator_score: float | None,
        structured_data: dict,
    ) -> MemoryBlock:
        if not source_message_ids and source_kind == 'raw':
            raise ValueError('source_message_ids cannot be empty for raw memory blocks')
        if source_message_ids:
            start_message_id = min(source_message_ids) if start_message_id is None else start_message_id
            end_message_id = max(source_message_ids) if end_message_id is None else end_message_id
        if source_message_count is None:
            source_message_count = len(source_message_ids)
        with self._connect() as conn:
            sequence_no = int(
                conn.execute(
                    "SELECT COALESCE(MAX(sequence_no), 0) + 1 AS next_seq FROM compaction_blocks WHERE session_id = ? AND hidden = 0",
                    (session_id,),
                ).fetchone()['next_seq']
            )
            cursor = conn.execute(
                """
                INSERT INTO compaction_blocks (
                    session_id, sequence_no, summary_text, estimated_tokens, source_message_count, start_message_id, end_message_id,
                    level, kind, lifecycle, source_kind, parent_block_ids_json, topic_labels_json, actor_labels_json, time_start, time_end,
                    retained_raw_excerpt_count, validator_status, validator_score, structured_data_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id, sequence_no, summary_text, estimated_tokens, int(source_message_count), start_message_id, end_message_id,
                    int(level), kind, lifecycle, source_kind, json.dumps(parent_block_ids, ensure_ascii=False),
                    json.dumps(topic_labels, ensure_ascii=False), json.dumps(actor_labels, ensure_ascii=False),
                    time_start, time_end, int(retained_raw_excerpt_count), validator_status, validator_score,
                    json.dumps(structured_data, ensure_ascii=False),
                ),
            )
            new_block_id = int(cursor.lastrowid)
            if source_message_ids:
                conn.executemany(
                    "UPDATE messages SET compacted = 1, compacted_level = ?, compacted_by_block_id = ? WHERE id = ? AND session_id = ?",
                    [(int(level), new_block_id, message_id, session_id) for message_id in source_message_ids],
                )
            row = conn.execute(
                """
                SELECT id, sequence_no, summary_text, estimated_tokens, source_message_count, start_message_id, end_message_id,
                       level, kind, lifecycle, source_kind, parent_block_ids_json, topic_labels_json, actor_labels_json,
                       time_start, time_end, retained_raw_excerpt_count, validator_status, validator_score, structured_data_json
                FROM compaction_blocks WHERE id = ?
                """,
                (new_block_id,),
            ).fetchone()
        return self._hydrate_memory_block(row)

    async def replace_memory_blocks(
        self,
        session_id: str,
        *,
        block_ids: list[int],
        summary_text: str,
        estimated_tokens: int,
        source_message_count: int,
        start_message_id: int | None,
        end_message_id: int | None,
        level: int = 2,
        kind: str = 'digest',
        lifecycle: str = 'sealed',
        source_kind: str = 'blocks',
        parent_block_ids: list[int] | tuple[int, ...] | None = None,
        topic_labels: list[str] | tuple[str, ...] | None = None,
        actor_labels: list[str] | tuple[str, ...] | None = None,
        time_start: str | None = None,
        time_end: str | None = None,
        retained_raw_excerpt_count: int = 0,
        validator_status: str | None = None,
        validator_score: float | None = None,
        structured_data: dict | None = None,
        source_message_ids: list[int] | None = None,
    ) -> MemoryBlock:
        return await asyncio.to_thread(
            self._replace_memory_blocks_sync,
            session_id,
            [int(block_id) for block_id in block_ids if int(block_id) > 0],
            summary_text,
            estimated_tokens,
            source_message_count,
            start_message_id,
            end_message_id,
            level,
            kind,
            lifecycle,
            source_kind,
            list(parent_block_ids or []),
            list(topic_labels or []),
            list(actor_labels or []),
            time_start,
            time_end,
            retained_raw_excerpt_count,
            validator_status,
            validator_score,
            structured_data or {},
            list(source_message_ids or []),
        )

    def _replace_memory_blocks_sync(
        self,
        session_id: str,
        block_ids: list[int],
        summary_text: str,
        estimated_tokens: int,
        source_message_count: int,
        start_message_id: int | None,
        end_message_id: int | None,
        level: int,
        kind: str,
        lifecycle: str,
        source_kind: str,
        parent_block_ids: list[int],
        topic_labels: list[str],
        actor_labels: list[str],
        time_start: str | None,
        time_end: str | None,
        retained_raw_excerpt_count: int,
        validator_status: str | None,
        validator_score: float | None,
        structured_data: dict,
        source_message_ids: list[int],
    ) -> MemoryBlock:
        if not block_ids:
            raise ValueError('block_ids cannot be empty when replacing memory blocks')
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT COALESCE(MIN(sequence_no), 0) AS base_seq FROM compaction_blocks WHERE session_id = ? AND hidden = 0 AND id IN ({','.join('?' for _ in block_ids)})",
                (session_id, *block_ids),
            ).fetchone()
            base_seq = int(row['base_seq'] or 0)
            if base_seq <= 0:
                raise ValueError('visible replacement block sequence not found')
            cursor = conn.execute(
                """
                INSERT INTO compaction_blocks (
                    session_id, sequence_no, summary_text, estimated_tokens, source_message_count, start_message_id, end_message_id,
                    level, kind, lifecycle, source_kind, parent_block_ids_json, topic_labels_json, actor_labels_json, time_start, time_end,
                    retained_raw_excerpt_count, validator_status, validator_score, structured_data_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id, base_seq, summary_text, estimated_tokens, int(source_message_count), start_message_id, end_message_id,
                    int(level), kind, lifecycle, source_kind, json.dumps(parent_block_ids, ensure_ascii=False),
                    json.dumps(topic_labels, ensure_ascii=False), json.dumps(actor_labels, ensure_ascii=False),
                    time_start, time_end, int(retained_raw_excerpt_count), validator_status, validator_score,
                    json.dumps(structured_data, ensure_ascii=False),
                ),
            )
            new_block_id = int(cursor.lastrowid)
            if source_message_ids:
                conn.executemany(
                    "UPDATE messages SET compacted = 1, compacted_level = ?, compacted_by_block_id = ? WHERE id = ? AND session_id = ?",
                    [(int(level), new_block_id, message_id, session_id) for message_id in source_message_ids],
                )
            conn.execute(
                f"UPDATE compaction_blocks SET hidden = 1, superseded_at = CURRENT_TIMESTAMP, superseded_by_block_id = ? WHERE session_id = ? AND hidden = 0 AND id IN ({','.join('?' for _ in block_ids)})",
                (new_block_id, session_id, *block_ids),
            )
            conn.execute(
                "UPDATE compaction_blocks SET sequence_no = sequence_no + 1 WHERE session_id = ? AND hidden = 0 AND id != ? AND sequence_no >= ?",
                (session_id, new_block_id, base_seq),
            )
            row = conn.execute(
                """
                SELECT id, sequence_no, summary_text, estimated_tokens, source_message_count, start_message_id, end_message_id,
                       level, kind, lifecycle, source_kind, parent_block_ids_json, topic_labels_json, actor_labels_json,
                       time_start, time_end, retained_raw_excerpt_count, validator_status, validator_score, structured_data_json
                FROM compaction_blocks WHERE id = ?
                """,
                (new_block_id,),
            ).fetchone()
        return self._hydrate_memory_block(row)

    async def clear_messages(self, session_id: str) -> None:
        await asyncio.to_thread(self._clear_messages_sync, session_id)

    def _clear_messages_sync(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE messages SET hidden = 1 WHERE session_id = ?", (session_id,))
            conn.execute("UPDATE compaction_blocks SET hidden = 1 WHERE session_id = ?", (session_id,))

