# Maintenance

Run these commands from the deployment directory. They use the existing `.env`
and PostgreSQL configuration and do not send Telegram messages. Start the database
with `./update.sh` before maintenance if it is stopped. For source development,
replace `docker compose run --rm --no-deps bot python` with `uv run --frozen python`.
Use each command's `--help` for its arguments.

## Workspace file reading

With remote tools enabled, the agent can use `read_doc(path, format)` with a path
relative to its session directory.
Text and PDF reads optionally select an inclusive, one-based `start`/`end` line
or page range. Ordinary attachments sync as files, with contents read on demand.
Images and PDF pages require a provider supporting visual tool results; audio/video
tool results are unsupported on the current API routes.
The tool does not send files to Telegram; use `file_send` for delivery.

The configured remote Python environment owns parsing dependencies; see
[remote setup](../deploy/README.md#optional-features-and-maintenance).
`READ_DOC_PDF_SCALE=2` renders two pixels per PDF point (144 dpi).
`READ_DOC_MAX_IMAGE_PIXELS=16777216` limits one rendered image to 16 megapixels
to bound remote rendering memory; set it to `0` to disable the pixel bound.
Both settings affect prepared images, leaving original files intact. Existing
SSH output-byte/time limits and the runtime's context/image allowance also apply.

## Sticker catalog

Keep originals under `data/stickers/<pack>/`. The first directory below the
sticker root defines the pack used by search, preferences, continuity and `--pack`.
Group related source sets beneath one directory to make an explicit virtual pack:

```text
data/stickers/
  Blue bird/
    Blue bird_birdPack/
      sticker.webp
    Blue bird 2_birdPackV2/
      webm/sticker.webm
  Sleepy fox_foxPack/
    sticker.webp
```

Nested source and media-format directories remain part of the file path; they do
not create separate packs. Virtual packs express your grouping, not a claim that
every sticker depicts the same character.

Optionally describe a pack's shared style or usage. The description appears beside
every matching candidate in every `sticker_query` response, helping the agent choose.
Keep it to a short, specific phrase since candidates repeat it.
Descriptions are manually supplied; unconfigured packs have none. Use the exact pack
ID returned by `list`:

```sh
docker compose run --rm --no-deps bot python -m scripts.sticker_pack list
docker compose run --rm --no-deps bot python -m scripts.sticker_pack get 'Blue bird'
docker compose run --rm --no-deps bot python -m scripts.sticker_pack set 'Blue bird' 'Quiet, understated reactions in a watercolor style.'
docker compose run --rm --no-deps bot python -m scripts.sticker_pack remove 'Blue bird'
```

These commands use the existing database configuration and make no model requests.
Descriptions are stored once per catalog revision and survive appending or
regenerating stickers in the same pack. They do not change ranking, filtering or
embeddings. `remove` clears only the description. After renaming a pack, set its
description under the new name. The old description remains listed and can be
removed with `remove 'Old pack name'`, or retained if that pack will be used again.

Build new entries or regenerate selected content:

```sh
docker compose run --rm --no-deps bot python -m scripts.build_sticker_index
docker compose run --rm --no-deps bot python -m scripts.build_sticker_index --file 'pack/sticker.webp'
docker compose run --rm --no-deps bot python -m scripts.build_sticker_index --pack 'pack'
docker compose run --rm --no-deps bot python -m scripts.build_sticker_index --asset-id 'sha256:FULL_CONTENT_HASH'
docker compose run --rm --no-deps bot python -m scripts.build_sticker_index --resume REVISION_ID
```

The default operation adds new content and refreshes current file paths and pack
membership, reusing completed compatible analysis. Run it after moving folders;
use selection flags only when you want to regenerate analysis. Explicitly saved
pack preferences use folder names, so update those preferences if you rename them.
Unsupported files are listed in the result; supported files in the same
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
queues memory work. It does not execute historical commands or call models.
Available ordinary files sync through the configured remote workspace, like live
attachments; import preserves the export originals. Rerunning retries unavailable
transfers without duplicating messages. Parsable images included beside the export use the normal image
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
