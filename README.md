# tgchatbot v2

A Telegram-first chat bot with persistent session history, per-chat runtime controls, OpenAI and Gemini providers, and SSH-backed shell/Python tools.

## What changed in this cleanup

- Global default chat mode is now operator-configurable with `DEFAULT_CHAT_MODE`.
- Session defaults now flow from config instead of being re-created with hard-coded values in multiple places.
- Session persistence now normalizes missing or invalid values against config defaults instead of trusting old rows blindly.
- Telegram reply timing is split into three operator-facing knobs:
  - `DEFAULT_PRIVATE_REPLY_DELAY_S`
  - `DEFAULT_GROUP_REPLY_DELAY_S`
  - `DEFAULT_GROUP_SPONTANEOUS_REPLY_DELAY_S`
- Gemini is treated as a real provider path, not just a placeholder import. The app can start with Gemini only, OpenAI only, or both.
- Tool schemas now include parameter descriptions and use OpenAI strict-tool compatible nullable optional parameters.
- Remote input handling is now remote-authoritative: syncable uploads are pushed to the remote workspace immediately, recorded in history by remote path, and their local originals are discarded after sync. Visual previews stay inline in history and do not rely on local files.
- `file_send` is the dedicated remote-to-chat file return tool, while `return_output_files` and Python-side shell preambles remain removed from the active tool surface. Tool declarations stay provider-neutral with OpenAI/Gemini-specific adapters only at provider edges.

## Implemented features

- Telegram transport
- SQLite-backed session persistence
- Per-chat mode toggles: `chat`, `assist`, `agent`
- Per-chat process visibility: `off`, `minimal`, `status`, `verbose`, `full`
- Per-chat response delivery: `edit`, `final_new`
- Provider switching between `openai` and `gemini` when configured
- SSH-backed `shell_exec` and `python_exec`
- Local sticker retrieval and Telegram sticker sending
- Link prefetching modes: `off`, `title`, `snippet`
- Conversation compaction into durable memory blocks

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m tgchatbot.app
```

## Provider startup rules

The app starts when at least one provider key is set.

- `OPENAI_API_KEY` only: OpenAI available
- `GEMINI_API_KEY` only: Gemini available
- both keys set: both available
- neither key set: startup fails

`DEFAULT_PROVIDER` must point to a configured provider. Startup fails fast if that provider is not configured.

## Commands

- `/start`
- `/help`
- `/status [full]`
- `/mode chat|assist|agent`
- `/process off|minimal|status|verbose|full`
- `/delivery edit|final_new`
- `/stickers off|auto`
- `/provider <available provider>`
- `/model <name>|default`
- `/params`
- `/param <name> <value|default>`
- `/prompt`
- `/presets`
- `/preset <name> [augment|exact]`
- `/retry`
- `/rollback [count]`
- `/reset [history|session|all]`

## Session parameter names and allowed values

Use `/param <name> <value>`; `/status` is concise by default, and `/status full` includes the old detailed status plus the current parameter block. Telegram `/params` and `/param` help adapt to the active provider/model.

- `reasoning_effort`: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `default` (OpenAI only)
- `reasoning_summary`: `off`, `on`, `auto`, `detailed`, `concise`, `default` (OpenAI only; `on` normalizes to `auto`)
- `text_verbosity`: `low`, `medium`, `high`, `default` (OpenAI only)
- `include_thoughts`: `on`, `off`, `default` (Gemini only; maps to `generationConfig.thinkingConfig.includeThoughts`)
- `thinking_budget`: Gemini only; use the current model's advertised range. Gemini 2.5 uses `thinkingBudget` directly. Gemini 3 treats it as a legacy fallback and ignores it when `thinking_level` is set.
- `thinking_level`: Gemini 3 only; `minimal`, `low`, `medium`, `high`, or `default`. Gemini 3 Pro models do not support `minimal`.
- `native_web_search`: `on`, `off`, `default`
- `native_web_search_max`: integer `0..100`, or `default` (OpenAI only; `0` disables the explicit built-in web search cap)
- `temperature`: number `0..2`, or `default` (Gemini native)
- `top_p`: number `0..1`, or `default` (Gemini native)
- `top_k`: integer `1..1000`, or `default` (Gemini native)
- `link_prefetch`: `off`, `title`, `snippet`, `default`
- `max_output_tokens`: integer `64..65536`, or `default`
- `max_input_images`: integer `0..100000`, or `default` (`0` disables the image-count cap)
- `compact_target_images`: integer `0..100000`, or `default` (`0` disables the separate image compaction target; compaction then falls back to `max_input_images`)
- `compact_trigger_tokens`: integer `256..10000000`, or `default`
- `compact_target_tokens`: integer `256..10000000`, or `default`
- `compact_batch_tokens`: integer `256..10000000`, or `default`
- `compact_keep_recent_ratio`: number `0..0.95`, percentage like `50%`, or `default`
- `compact_tool_ratio_threshold`: number `1..100`, or `default`
- `compact_tool_min_tokens`: integer `256..10000000`, or `default`
- `compact_min_messages`: integer `2..1000`, or `default`
- `min_raw_messages_reserve`: integer `0..1000`, or `default`
- `max_interaction_rounds`: integer `1..64`, or `default`
- `spontaneous_reply_chance`: integer `0..100`, or `default`
- `group_spontaneous_reply_delay_s`: number `0..86400`, or `default`
- `private_reply_delay_s`: number `0..600`, or `default`
- `group_reply_delay_s`: number `0..600`, or `default`
- `provider_retry_count`: integer `0..5`, or `default`
- `metadata`: `on`, `off`, `default`
- `metadata_timezone`: any IANA timezone such as `UTC`, `Asia/Tokyo`, `Europe/Berlin`, or `default`
- `tool_history_mode`: `translated`, `native_same_provider`, `default`

Provider note: Gemini native thinking controls are `include_thoughts`, `thinking_budget`, and `thinking_level`. On Gemini 3, prefer `thinking_level` and do not send it together with `thinking_budget` in the same request.

Provider note: Gemini native Google Search is only combined with custom function tools on Gemini 3. On Gemini 2.5, the adapter only sends `googleSearch` when no custom function tools are attached in that request.

Legacy note: `/param reply_delay_s ...` still works as a compatibility alias and sets both private and explicit-group reply delays together.


## Remote input history model

- regular Telegram photos and preview frames are stored inline in history as image payloads
- document/file uploads are synced to the remote workspace immediately
- file records in history keep the remote path, not a local path, whenever the remote workspace is enabled
- successful remote sync adds an explicit history note, and any files rotated out of the remote input cache are listed immediately before that success note
- uncompressed image files keep both a remote file record and inline preview image parts
- once sync is attempted, the local original file is not used as runtime input anymore
- remote input rotation only affects the remote input cache; chat history remains unchanged
- when remote execution is disabled, syncable files degrade to historical descriptors plus any inline previews, with no local-path fallback

## Prompt reconstruction and relaunch behavior

- every reply is reconstructed from SQLite-backed session state; inline image previews are reloaded from persisted `data_b64`, not from local artifact files
- delayed replies use the newest available session context when they finally fire; reply scheduling does not lock an older prompt snapshot
- provider replay history is cached only for the latest visible state and invalidated on new messages, retries, rollbacks, compaction, or session reload
- after process relaunch, session settings, raw visible messages, and compacted memory blocks are loaded back into live runtime state before the next turn

## Failure behavior

- if remote input syncing fails or a synced file rotates out immediately, the original local ephemeral copy is still discarded and history keeps an explicit text note instead of a fake usable path
- provider calls are retried up to `DEFAULT_PROVIDER_RETRY_COUNT` or the per-session override; if they still fail, the failure is logged and no synthetic assistant reply is added to session history or sent as a fallback message
- tool failures return structured `{"ok": false, ...}` outputs back to the model instead of crashing the whole turn
- unknown tool calls are converted into explicit tool errors
- delivery failures do not append synthetic assistant replies to session history; history is updated only after successful final delivery

## Tool surface

### shell_exec

Purpose: run shell commands in the per-chat remote workspace.

Parameters:
- `command` (string, required)
- `timeout_s` (integer or `null`)
- `cwd_subdir` (string or `null`)

Behavior:
- syncable chat files are uploaded into the per-chat remote input directory during ingest, and tools read only the remote workspace afterward
- older remote inputs can rotate out when `SSH_EXEC_MAX_INPUT_FILES` is exceeded
- output is returned through `stdout` and `stderr`
- file-return behavior is intentionally not exposed through the tool schema anymore

### python_exec

Purpose: run Python 3 in the per-chat remote workspace.

Parameters:
- `code` (string, required)
- `timeout_s` (integer or `null`)

Behavior:
- syncable chat files are uploaded into the per-chat remote input directory during ingest, and tools read only the remote workspace afterward
- older remote inputs can rotate out when `SSH_EXEC_MAX_INPUT_FILES` is exceeded
- output is returned through `stdout` and `stderr`
- shell preambles and output-file return flags are intentionally removed

### sticker_send

Purpose: choose a Telegram sticker from the local index.

Required parameter:
- `reaction`

Optional selector parameters:
- `emotion`, `action`, `situation`, `style`, `style_family`, `persona`, `sticker_subject_type`, `eye_style`, `detail_bias`
- `medium_family`, `framing_style`, `interaction_type`, `caption_social_stance`, `caption_target_direction`, `intimacy_level`
- `meme_dependence`, `context_fit`, `conversation_role`, `tone`, `meme_template`
- `intensity` (`1..5`)
- `timing` (`before_final` or `after_final`)
- `prefer_static`, `allow_animated`, `diversify`

## Environment variables

Every variable in `.env.example` is supported. The most important groups are below.

### Core app

- `APP_DATA_DIR`: local data root
- `LOG_LEVEL`: Python logging level
- `DEFAULT_PROVIDER`: `openai` or `gemini` (defaults to `gemini`)
- `DEFAULT_CHAT_MODE`: `chat`, `assist`, or `agent`
- `DEFAULT_PROCESS_VISIBILITY`: `off`, `minimal`, `status`, `verbose`, or `full`
- `DEFAULT_RESPONSE_DELIVERY`: `edit` or `final_new`
- `DEFAULT_STICKER_MODE`: `off` or `auto`
- `DEFAULT_PROMPT_INJECTION_MODE`: `augment` or `exact`
- `DEFAULT_TOOL_HISTORY_MODE`: `translated` or `native_same_provider`
- `DEFAULT_LINK_PREFETCH_MODE`: `off`, `title`, or `snippet`
- `DEFAULT_METADATA_INJECTION_MODE`: `on` or `off`
- `DEFAULT_METADATA_TIMEZONE`: IANA timezone
- `DEFAULT_SYSTEM_PROMPT`: optional full prompt override

### Turn-shaping defaults

- `DEFAULT_CHAT_MAX_ROUNDS`: integer `1..64`
- `DEFAULT_ASSIST_MAX_ROUNDS`: integer `1..64`
- `DEFAULT_AGENT_MAX_ROUNDS`: integer `1..64`
- `DEFAULT_PROVIDER_RETRY_COUNT`: integer `0..5`

### Reply timing defaults

- `DEFAULT_PRIVATE_REPLY_DELAY_S`
- `DEFAULT_GROUP_REPLY_DELAY_S`
- `DEFAULT_GROUP_SPONTANEOUS_REPLY_CHANCE`
- `DEFAULT_GROUP_SPONTANEOUS_REPLY_DELAY_S`

Legacy alias: `DEFAULT_GROUP_SPONTANEOUS_REPLY_IDLE_S` is still accepted and maps to `DEFAULT_GROUP_SPONTANEOUS_REPLY_DELAY_S`.

Reply triggers only come from plain-text Telegram messages. Messages with links, captions, photos, stickers, documents, videos, or other media are still ingested into history, but they do not schedule a reply. Group spontaneous replies roll their probability after the idle delay for the newest eligible plain-text message.

### Telegram transport

- `TGBOT_TOKEN`
- `TGBOT_WHITELIST`
- `TGBOT_CONTROL_UID_WHITELIST`
- `TGBOT_KEYWORDS`
- `TGBOT_IGNORE_KEYWORDS`
- `TGBOT_REPLY_TO_USER_MESSAGE`: `true` to reply to the triggering user message, `false` to send a fresh chat message
- `TGBOT_MIN_EDIT_INTERVAL_S`
- `TGBOT_MAX_PHOTO_BYTES`
- `TGBOT_MAX_DOCUMENT_BYTES`
- `TGBOT_MAX_STICKER_BYTES`
- `TGBOT_MAX_STICKER_FRAMES`
- `TGBOT_MAX_VISUAL_FILE_FRAMES`
- `TGBOT_MAX_INLINE_TEXT_CHARS`
- `TGBOT_LINK_PREFETCH_TIMEOUT_S`
- `TGBOT_LINK_PREFETCH_MAX_URLS`
- `TGBOT_LINK_PREFETCH_MAX_CHARS`

### OpenAI provider

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL`
- `OPENAI_MODEL`
- `OPENAI_REASONING_EFFORT`
- `OPENAI_REASONING_SUMMARY`: `off`, `on`, `auto`, `detailed`, or `concise` (`on` normalizes to `auto`)
- `OPENAI_TEXT_VERBOSITY`
- `OPENAI_MAX_OUTPUT_TOKENS`
- `OPENAI_MAX_INPUT_IMAGES`: integer `0..100000`; `0` disables the image-count cap
- `OPENAI_COMPACT_TARGET_IMAGES`: integer `0..100000`; `0` disables the separate image compaction target
- `OPENAI_ENABLE_NATIVE_WEB_SEARCH`
- `OPENAI_NATIVE_WEB_SEARCH_MAX`: integer `0..100`; `0` disables the explicit built-in web search cap
- `OPENAI_REQUEST_TIMEOUT_S`
- `OPENAI_CONNECT_TIMEOUT_S`

### Gemini provider

- `GEMINI_API_KEY`
- `GEMINI_BASE_URL`
- `GEMINI_MODEL`
- `GEMINI_TEMPERATURE`
- `GEMINI_TOP_P`
- `GEMINI_TOP_K`
- `GEMINI_INCLUDE_THOUGHTS`: `true` or `false`; maps to `generationConfig.thinkingConfig.includeThoughts`
- `GEMINI_THINKING_BUDGET`: native `thinkingBudget`. Gemini 2.5 uses it directly. Gemini 3 treats it as a legacy fallback when `GEMINI_THINKING_LEVEL` is unset.
- `GEMINI_THINKING_LEVEL`: Gemini 3 native `thinkingLevel`; `minimal`, `low`, `medium`, or `high` with model-specific support (`minimal` is not accepted by Gemini 3 Pro)
- `GEMINI_ENABLE_NATIVE_WEB_SEARCH`
- `GEMINI_MAX_OUTPUT_TOKENS`
- `GEMINI_MAX_INPUT_IMAGES`
- `GEMINI_COMPACT_TARGET_IMAGES`
- `GEMINI_REQUEST_TIMEOUT_S`
- `GEMINI_CONNECT_TIMEOUT_S`

### Remote execution

- `SSH_EXEC_ENABLED`
- `SSH_EXEC_HOST`
- `SSH_EXEC_PORT`
- `SSH_EXEC_WORKDIR`
- `SSH_EXEC_IDENTITY_FILE`
- `SSH_EXEC_CONNECT_TIMEOUT_S`
- `SSH_EXEC_DEFAULT_TIMEOUT_S`
- `SSH_EXEC_MAX_TOOL_TIMEOUT_S`
- `SSH_EXEC_MAX_STDOUT_CHARS`
- `SSH_EXEC_MAX_STDERR_CHARS`
- `SSH_EXEC_MAX_INPUT_FILES`
- `SSH_EXEC_MAX_INPUT_FILE_BYTES`
- `SSH_EXEC_MAX_OUTPUT_FILES`
- `SSH_EXEC_MAX_OUTPUT_FILE_BYTES`
- `SSH_EXEC_SERVER_ALIVE_INTERVAL_S`
- `SSH_EXEC_SERVER_ALIVE_COUNT_MAX`
- `SSH_EXEC_CONTROL_PERSIST_S`

### Context compaction

- `CONTEXT_COMPACT_TRIGGER_TOKENS`
- `CONTEXT_COMPACT_TARGET_TOKENS`
- `CONTEXT_COMPACT_BATCH_TOKENS`
- `CONTEXT_COMPACT_MIN_MESSAGES`
- `CONTEXT_MIN_RAW_MESSAGES_RESERVE`

## Examples

### Gemini as the only provider

```env
DEFAULT_PROVIDER=gemini
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-2.5-flash
OPENAI_API_KEY=
```

### Fast private replies, slower explicit group replies

```env
DEFAULT_PRIVATE_REPLY_DELAY_S=0
DEFAULT_GROUP_REPLY_DELAY_S=4
DEFAULT_GROUP_SPONTANEOUS_REPLY_CHANCE=5
DEFAULT_GROUP_SPONTANEOUS_REPLY_DELAY_S=45
```

### More conservative OpenAI settings

```env
OPENAI_MODEL=gpt-5.4-nano
OPENAI_REASONING_EFFORT=none
OPENAI_TEXT_VERBOSITY=low
OPENAI_MAX_OUTPUT_TOKENS=4096
OPENAI_ENABLE_NATIVE_WEB_SEARCH=false
```

### Remote workspace with custom timeouts

```env
SSH_EXEC_ENABLED=true
SSH_EXEC_HOST=user@host
SSH_EXEC_DEFAULT_TIMEOUT_S=20
SSH_EXEC_MAX_TOOL_TIMEOUT_S=180
SSH_EXEC_SERVER_ALIVE_INTERVAL_S=20
SSH_EXEC_SERVER_ALIVE_COUNT_MAX=3
SSH_EXEC_CONTROL_PERSIST_S=900
```

## Project layout

```text
./tgchatbot/
  app.py
  config.py
  domain/
  core/
  providers/
  tools/
  transports/
  storage/
  media/
```

## Legacy migration

```bash
python scripts/migrate_legacy.py --legacy-root /path/to/old/tgchatbot --data-dir ./data --provider openai --model gpt-5.4-nano
```

Imported chat metadata goes into SQLite, while image and file payloads are written into `data/artifacts/`. Imported presets are copied into `data/presets/`.

## Sticker corpus and index

Standard paths:
- sticker files: `data/stickers/`
- sticker index database: `data/sticker_index.sqlite3`

Build or refresh the index:

```bash
python scripts/build_sticker_index.py
```

Append into an existing index:

```bash
python scripts/build_sticker_index.py --append
```

Local retrieval test:

```bash
python scripts/query_sticker_index.py --query "gentle proud laugh" --persona warm --style-family cute_character --prefer-static
```


## Sticker retrieval stack

The sticker system now targets:
- PaddleOCR-based caption extraction for CJK overlays
- Tantivy lexical retrieval sidecar for caption evidence
- local semantic vector matrix for caption/sticker semantic cards
- app-side fusion with session style memory

See `docs/sticker_retrieval.md` for the current architecture and commands.

# Sticker retrieval patch notes

This bundle has been unified around one current architecture for human-level sticker retrieval.

## What changed

- Added an authoritative shared sticker schema in `tgchatbot/stickers/schema.py`.
- Split semantics into four explicit cards:
  - `caption_card`
  - `subtle_cue_card`
  - `sticker_card`
  - `style_card`
- Moved build-time schema ownership out of `scripts/build_sticker_index.py` and into the shared module.
- Strengthened the build prompts and review prompts to preserve pragmatic force, including the `无语` correction.
- Kept real PaddleOCR in the build path and made it a hard requirement in `requirements.txt`.
- Added `semantic_signature` so multi-candidate selection can diversify away from near-duplicates.
- Updated the runtime ranking path to use:
  - semantic-first fusion
  - caption evidence as a strong supporting signal
  - subtle visual cue semantics for textless stickers
  - style/session prior
  - MMR-style diversification with semantic-signature penalties
- Replaced the sticker-query interface with one compact plan:
  - `send`
  - `intent_core`
  - `secondary_goals`
  - `text_priority`
  - `max_harshness`
  - `max_intimacy`
  - `forbid`
  - `allow_animation`
  - `style_policy`
  - `candidate_budget`
- Set practical defaults to `3` for `max_harshness`, `4` for `max_intimacy`, and `5` for `candidate_budget`.
- Updated the CLI query script to match the compact plan.
- Added targeted regression tests for:
  - `无语`-style dismissive rebuttal vs neutral “speechless” flattening
  - textless stickers where subtle visual cues dominate meaning

## Important rebuild note

The build script now intentionally resets the sticker index schema and rebuilds from source stickers. Old index databases should be treated as invalid and rebuilt.

## Validation run in this environment

- Python compile pass: OK
- Unit tests: OK (`python -m unittest discover -s tests -v`)
- Rust sidecar compile check: not run here because `cargo` is not installed in this environment

## Recommended rebuild flow

1. Install Python deps from `requirements.txt`.
2. Ensure `OPENAI_API_KEY` is set.
3. Rebuild the sticker index:
   - `python scripts/build_sticker_index.py --stickers-dir ./data/stickers --index-db ./data/sticker_index.sqlite3`
4. Rebuild the Tantivy sidecar index:
   - `python scripts/rebuild_tantivy_index.py --index-db ./data/sticker_index.sqlite3`
5. Start the Rust retriever sidecar and verify `/health` returns schema version `human_sticker_v1`.
6. Smoke-test retrieval with `scripts/query_sticker_index.py`.
