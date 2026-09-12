# Maintenance

Run these commands from the deployment directory. They use the existing `.env`
and PostgreSQL configuration and do not send Telegram messages. Start the database
with `./update.sh` before maintenance if it is stopped. For source development,
replace `docker compose run --rm --no-deps bot python` with `uv run --frozen python`.
Use each command's `--help` for its arguments.

## Sticker catalog

Keep originals under `data/stickers/<pack>/`. Build new entries or update only
selected content:

```sh
docker compose run --rm --no-deps bot python -m scripts.build_sticker_index
docker compose run --rm --no-deps bot python -m scripts.build_sticker_index --file 'pack/sticker.webp'
docker compose run --rm --no-deps bot python -m scripts.build_sticker_index --pack 'pack'
docker compose run --rm --no-deps bot python -m scripts.build_sticker_index --asset-id 'sha256:FULL_CONTENT_HASH'
docker compose run --rm --no-deps bot python -m scripts.build_sticker_index --resume REVISION_ID
```

The default operation appends new files and aliases, reusing completed compatible
analysis. Unsupported files are listed in the result; supported files in the same
source are still processed. Selection flags may repeat. `scripts.reset_sticker` is an alias for the
same regeneration workflow. Neither command deletes original media. Failed work
stays in a staging revision; only a complete revision replaces the active catalog.
Resume with the reported revision ID, original source directory and semantic
settings. Output allowance, transport timeout, concurrency and service tier can change on resume.
Successful annotation and embedding channels are checkpointed independently.

Annotation uses `STICKER_BUILD_PROVIDER`, `STICKER_BUILD_MODEL` and
`STICKER_BUILD_SERVICE_TIER`. The configurable default is Gemini Flash with Flex;
blank or `off` selects the provider's normal tier. Capacity failures remain visible,
with no automatic Standard fallback. These settings do not change chat generation.
Annotation and embedding requests can incur provider charges.

`STICKER_EMBEDDING_*` overrides the shared embedding route. Sticker dimensions
default to 3072, independently of conversation memory's 1536 dimensions. Changing
the embedding space rebuilds vectors while reusing unchanged cards. Text-only
embedding routes use card descriptions for best-effort appearance retrieval;
candidate previews still come from original images. Sampling and
other settings are listed in [the full configuration example](../.env.full.example).

Apply reviewed corrections with `--corrections /app/data/corrections.json`:

```json
{
  "sha256:FULL_CONTENT_HASH": {
    "card": {"caption": "Correct visible caption"},
    "family_ids": ["a-supported-character-identity"]
  }
}
```

Corrections survive regeneration. Family labels need evidence; sharing a pack
does not prove that two stickers depict the same character. Keep all original
files needed for delivery. The bot verifies content identity before sending.

Inspect candidates without sending or changing preferences:

```sh
docker compose run --rm --no-deps bot python -m scripts.query_sticker_index \
  --intent-core 'Offer a warm greeting'
```

`--session-id` includes that chat's saved preferences and confirmed delivery
history. `--plan` accepts the structured `sticker_query` arguments. Candidate
previews are evidence for selection; they are not automatically sent to Telegram.

## Import Telegram Desktop history

Place an Export messages JSON file under `data/imports/`. Stop only the bot,
leaving PostgreSQL running:

```sh
docker compose stop bot
docker compose run --rm --no-deps bot python -m tgchatbot.tools.import_desktop \
  --file /app/data/imports/result.json --chat-id=-1001234567890
```

Replace the example with the destination Telegram chat ID. For a full export,
`--export-chat-id` selects its source conversation. The importer preserves source
identities and reply references, deduplicates unchanged messages on rerun, and
queues memory work. It does not execute historical commands, call models or
upload media. Parsable images included beside the export use the normal image
compression path and remain available to memory reads. Missing image files and
unsupported attachments remain references.

For a large import, keep the bot stopped and prepare its working context before
resuming it:

```sh
docker compose run --rm --no-deps bot python -m tgchatbot.tools.memory prepare-context --chat-id=-1001234567890
```

Preparation uses the chat's configured model and existing compaction layers in
bounded batches. It preserves searchable originals and refreshes participant
profiles after compaction. It can incur generation charges. Interrupted work
resumes from committed summaries when the command is run again. Search indexing
and background profile batches are processed separately by the memory worker.

## Memory jobs and operator audit

Run a dedicated worker while the live bot is stopped:

```sh
docker compose run --rm --no-deps bot python -m tgchatbot.tools.memory work --batch
```

`--batch` uses native Gemini asynchronous embedding Batch for bulk work. Other
embedding routes use `work` without that option. Profile extraction still uses
ordinary generation requests, inheriting the chat model unless `MEMORY_PROVIDER`
or `MEMORY_MODEL` overrides it. Processing can incur both embedding and generation
charges. `--once` performs one dispatch, not a complete queue drain.

Inspect work in another terminal:

```sh
docker compose run --rm --no-deps bot python -m tgchatbot.tools.memory status --chat-id=-1001234567890
docker compose run --rm --no-deps bot python -m tgchatbot.tools.memory status --chat-id=-1001234567890 --coverage
docker compose run --rm --no-deps bot python -m tgchatbot.tools.memory retry-jobs --chat-id=-1001234567890
docker compose run --rm --no-deps bot python -m tgchatbot.tools.memory rebuild --chat-id=-1001234567890
docker compose run --rm --no-deps bot python -m tgchatbot.tools.memory audit --chat-id=-1001234567890 --state
```

Ordinary status is inexpensive; `--coverage` scans active source spans. Rebuild
queues the selected embedding space without deleting originals. Retry preserves
accepted Batch identities; ambiguous submissions are reconciled before any new
submission. A changed or unknown space is reported rather than silently mixed.
Inspection, queueing and audit commands themselves make no model requests.

Audit is an operator-only view and may include hidden originals, prior generations
and settings snapshots. It is not an agent tool. `--message-id` refers to the
internal database ID, not a Telegram ID. Treat its output as retained chat data.

Stop the dedicated worker before restarting the bot with `./update.sh`. The bot
can continue accepted jobs. Do not run separate workers against the same live
queue merely to increase throughput.
