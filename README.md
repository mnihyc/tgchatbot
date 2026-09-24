# tgchatbot

A self-hosted Telegram chatbot with persistent conversations, participant-aware
memory, configurable personalities, optional remote tools, and contextual stickers.
Each chat owns its history and settings; group participants share that agent.

Use OpenAI, Gemini, DeepSeek, OpenRouter, or a compatible endpoint. Different
chats can use different providers without replacing their conversation state.

## Install

You need Linux, Docker with Compose, a Telegram bot token, and a configured model
provider. Keep deployment files and retained data separate from the source checkout.

Download and unpack the [latest deployment bundle](https://github.com/mnihyc/tgchatbot/releases/latest),
then run `./update.sh` from the extracted directory.

The first run creates `.env`. Set `TGBOT_TOKEN` and a provider key, then run
`./update.sh` again. Existing `compose.yml` and `.env` settings are retained. The updater configures
the bundled PostgreSQL database; an explicit `DATABASE_URL` selects your own
PostgreSQL database with pgvector.

See [deployment and backups](deploy/README.md). Available configuration is listed
in [.env.example](.env.example) and [.env.full.example](.env.full.example).

## Providers

| Provider | Configuration |
| --- | --- |
| OpenAI | `OPENAI_API_KEY`; optional `OPENAI_MODEL` |
| Gemini | `GEMINI_API_KEY`; optional `GEMINI_MODEL` |
| DeepSeek | `DEEPSEEK_API_KEY`; optional `DEEPSEEK_MODEL` |
| OpenRouter | `OPENROUTER_API_KEY` and `OPENROUTER_MODEL` |

Configure any combination and set `DEFAULT_PROVIDER` explicitly when a particular
route should be the default. `/provider <name>` switches the current chat and
selects that provider's configured model. `/model <id>` overrides it. The prompt
and history survive switching; unavailable routes produce an error rather than
silently sending the conversation to a different provider.

Custom Chat Completions profiles use `LLM_PROVIDERS_JSON`:

```dotenv
CUSTOM_LLM_KEY=replace-me
LLM_PROVIDERS_JSON=[{"name":"custom","api_key_env":"CUSTOM_LLM_KEY","base_url":"https://api.example.com/v1","model":"your-model-id","structured_output":"json_object"}]
DEFAULT_PROVIDER=custom
```

Model capabilities differ. Enable image input only on a compatible model; choose
`json_schema`, `json_object`, or `prompt` for its structured-output support.
Sampling, reasoning, context and timeout controls remain configurable in `.env`.

Embeddings are independent of generation. `EMBEDDING_*` configures conversation
search; the default uses the existing Gemini credentials and 1536 dimensions.
An OpenAI embedding route uses `EMBEDDING_PROVIDER=openai`; a compatible endpoint
also sets `EMBEDDING_BASE_URL` and `EMBEDDING_MODEL`. Without embeddings, lexical
history search remains available. Sticker embeddings have independent
`STICKER_EMBEDDING_*` overrides. See [maintenance](scripts/README.md).

## Conversation controls

Private text starts a reply. In groups, use a configured keyword or reply to the
bot. Media and captions add context; send a text question to request a response.
Optional spontaneous group replies can be configured separately. Set
`TGBOT_WHITELIST` to restrict access; an empty whitelist allows all chats.

| Command | Purpose |
| --- | --- |
| `/help [replies\|model\|context\|tools]` | Short command guide, with focused help by topic |
| `/status [context\|memory\|tools]`, `/status full` | Context usage and input shares, topic details, or full diagnostics |
| `/context`, `/context recent`, `/context summaries` | Inspect the saved context, recent message excerpts, or included summaries |
| `/context profiles`, `/context tools`, `/context full` | Browse saved profiles and fetched snapshots, tool exchanges, or the complete readable context |
| `/mode chat`, `/mode assist`, `/mode agent` | Conversation with read-only memory, occasional tools, or multistep tools |
| `/provider`, `/model` | Inspect or change the chat's generation route |
| `/presets`, `/preset <name>`, `/prompt` | Manage personality prompts |
| `/params` or `/settings`, `/params <group>` | Browse model, context, replies or tools settings; `full` shows all values |
| `/param <name>`, `/param <name> <value>` | Inspect one setting and its accepted inputs, or change it; `default` restores its configured default |
| `/process off`, `/delivery final_new` | Control progress visibility and final-answer delivery |
| `/stickers auto`, `/stickers off` | Enable or disable sticker tools in a tool-enabled mode |
| `/retry` | Retry the latest eligible user message without duplicating it |
| `/rollback <count>` | Abandon recent consecutive user/bot blocks in the current context |
| `/compact` | Run layered compaction now toward the configured target; retain searchable originals and profiles |
| `/reset` or `/reset history` | Clear working context; retain searchable history, profiles and settings |
| `/reset_full` or `/reset all` | Start a fresh agent with defaults; previous generations become operator-audit-only |
| `/reset session` | Restore session settings while retaining conversation and learned profiles |

Commands send ordinary, persistent chat messages. Settings apply to the entire
chat; `TGBOT_CONTROL_UIDS` restricts `/param` to trusted users when configured.
Agent replies support Markdown formatting, including bold text, links and code
blocks. Long replies retain formatting across messages.

Command results stay in one message when they fit; larger results become one text
file, including `full` views. `/help context` explains browsing and pagination.
Input shares estimate the saved context for the current provider/model, rather
than billing or cache hits. Memory includes summaries and recalled originals;
Profiles counts fetched snapshots still in context, not every saved profile.
Inspection does not generate replies, compact history or refresh profiles.
For comprehensive operator reads and exports, see [memory inspection](scripts/README.md#memory-jobs-and-operator-audit).

Retry and rollback affect the current context. They do not retract Telegram
messages or undo remote actions. Full reset leaves the shared sticker library and
optional SSH workspace intact. Participants are identified by stable source IDs,
not display names; forwarded content retains its separate attribution.

Memory search includes image references beside attributed conversation context.
The agent can open selected images with `memory_read`. Compressed image evidence
stays in PostgreSQL after prompt compaction and `/reset`; `/reset_full` makes the
old generation audit-only. Missing exported image files remain unavailable.

`user_profile_fetch` reads asynchronously updated profiles with fact references. The agent can
open their supporting originals with `memory_read(profile_fact_ids=[...])`;
neither read triggers profile learning. Profiles learn in background batches;
compaction can process a pending batch, and `/reset` queues catch-up without waiting.

Context compacts silently after an hour of inactivity above 600K estimated tokens,
or before a reply at the 800K ceiling, toward a 50K target. Incoming messages are
still stored during compaction; replies wait for it. The target preserves complete
recent messages and tool exchanges, so some contexts remain larger. Generation
stops if compaction cannot fit the ceiling. Configure these thresholds in `.env`
for your model's context window; `/status context` shows effective chat settings.
Set `CONTEXT_COMPACT_IDLE_TRIGGER_TOKENS=0` to disable idle compaction. Existing
per-chat overrides continue to take precedence.

Stickers require an explicitly built catalog. The agent retrieves candidates,
inspects available evidence, and chooses a sticker or text. Query previews are
not automatically sent to Telegram. Photos and documents depend on the selected
model's capabilities. SSH tools remain disabled until a remote host is configured.

## Development

The Python application separates runtime, provider adapters, Telegram transport,
tools and storage under `tgchatbot/`. Maintain those boundaries when adding a
provider or changing a workflow. Intake idempotency, participant attribution,
reset isolation, compaction continuity and delivery acknowledgments must remain
covered by business tests.

With Python 3.12 or 3.13 and uv:

```sh
uv sync --frozen
TEST_DATABASE_URL=postgresql://user:password@localhost:5432/test_database \
  uv run --frozen python -m unittest discover -s tests -v
```

Use a disposable PostgreSQL database with pgvector. Tests mock Telegram, model,
SSH and release-download boundaries; they require no production credentials.
[Maintenance commands](scripts/README.md) cover sticker catalogs, Desktop imports
and memory jobs. Keep deployment data and credentials outside the Git checkout.

[Apache License 2.0](LICENSE).
