# Maintenance

Run these commands from the deployment directory. They use the existing `.env`
and PostgreSQL configuration and do not send Telegram messages. If the bundled
database is stopped, start only it with `docker compose up -d postgres`.
For source development, replace the Docker command prefix through `python`
with `uv run --frozen python`.
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

Discover IDs and inspect saved work without model requests or SQL:

```sh
docker compose run --rm --no-deps bot python -m scripts.sticker_catalog status
docker compose run --rm --no-deps bot python -m scripts.sticker_catalog revisions
docker compose run --rm --no-deps bot python -m scripts.sticker_catalog status --revision REVISION_ID
docker compose run --rm --no-deps bot python -m scripts.sticker_catalog assets --pack 'Blue bird'
docker compose run --rm --no-deps bot python -m scripts.sticker_catalog asset ASSET_ID
```

`assets` lists IDs, file paths, captions and processing states; use `--state failed`
and `--revision` to inspect interrupted work. `asset` shows the original generated
card, saved corrections, effective card and provenance, without binary vectors.
`revisions` finds staging IDs even if a build's terminal output was lost. Check that
revision's status for failures, saved channels, original source and model settings
before resuming. A staging revision whose parent is no longer current cannot resume;
start from the current catalog instead.

By default, `build_sticker_index` adds new content and refreshes current file paths and pack
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

Export saved corrections before editing, so unrelated overrides are preserved:

```sh
docker compose run --rm --no-deps -T bot python -m scripts.sticker_catalog corrections \
  --asset-id ASSET_ID > data/corrections.json
```

Use `--pack` to export a pack or omit selectors for all assets. Edit this JSON and
apply it:

```sh
docker compose run --rm --no-deps bot python -m scripts.build_sticker_index \
  --corrections /app/data/corrections.json
```

Example correction object:

```json
{
  "sha256:FULL_CONTENT_HASH": {
    "card": {"caption": "Correct visible caption"},
    "family_ids": ["a-supported-character-identity"]
  }
}
```

Each selected asset's correction object replaces its previous overrides; omitted
assets keep theirs. `{}` clears an asset's overrides, restoring its generated card.
The builder updates affected embeddings and still discovers new source files;
applying corrections can incur charges. Inspect the published asset to verify the
effective result. Corrections survive regeneration. Family labels need evidence;
sharing a pack does not prove that two stickers depict the same character. Keep all original
files needed for delivery. The bot verifies content identity before sending.

Inspect candidates without sending or changing preferences:

```sh
docker compose run --rm --no-deps bot python -m scripts.query_sticker_index \
  --intent-core 'Offer a warm greeting'
```

This semantic query can incur embedding charges. `--session-id` includes that
chat's saved preferences and confirmed delivery history. `--plan` accepts the
structured `sticker_query` arguments. Candidate
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
docker compose run --rm --no-deps bot python -m tgchatbot.tools.memory jobs --chat-id=-1001234567890 --status failed
docker compose run --rm --no-deps bot python -m tgchatbot.tools.memory profiles --chat-id=-1001234567890
docker compose run --rm --no-deps bot python -m tgchatbot.tools.memory context --chat-id=-1001234567890
docker compose run --rm --no-deps bot python -m tgchatbot.tools.memory retry-jobs --chat-id=-1001234567890
docker compose run --rm --no-deps bot python -m tgchatbot.tools.memory rebuild --chat-id=-1001234567890
docker compose run --rm --no-deps bot python -m tgchatbot.tools.memory audit --chat-id=-1001234567890 --state
```

Ordinary status is inexpensive; `--coverage` scans active source spans. Rebuild
queues the selected embedding space without deleting originals. Retry preserves
accepted Batch identities; ambiguous submissions are reconciled before any new
submission. A changed or unknown space is reported rather than silently mixed.
Inspection, queueing and audit commands themselves make no model requests.

`status` without a chat selector discovers existing sessions. `jobs` streams saved
errors, attempts, scheduling and provider checkpoint details; filter by `--job-id`,
`--kind`, `--status` or `--generation`. It defaults to the current generation.
`profiles` discovers participant IDs and shows their current bounded profiles,
evidence and pending learning material; `--actor-id` focuses on one participant.
It never triggers a refresh. `context` shows current raw-message counts and the
remaining compaction summaries with source ranges and layers, without dumping
original bodies. Traversal page sizes control fetches, not total output limits.

Check what the agent can retrieve:

```sh
docker compose run --rm --no-deps bot python -m tgchatbot.tools.memory search \
  --chat-id=-1001234567890 --query 'the travel plans' --lexical-only
docker compose run --rm --no-deps bot python -m tgchatbot.tools.memory read \
  --chat-id=-1001234567890 --message-id MESSAGE_ID
docker compose run --rm --no-deps bot python -m tgchatbot.tools.memory image \
  --chat-id=-1001234567890 --message-id MESSAGE_ID --image-id IMAGE_ID --output /app/data/recalled.png
```

Omit `--lexical-only` to use the configured semantic embedding route, which can
incur query charges. Search accepts participant/time filters; read supports
`--offset`, `--length` and `--neighbors`. Both reuse the agent's retrieval bounds
and return source identities and available image IDs. `image` saves the selected
compressed image to a new file; it does not overwrite an existing output. These
reads include history before `/reset` and exclude history before `/reset_full`.

Audit is an operator-only view and may include hidden originals, prior generations
and settings snapshots. It is not an agent tool. `--message-id` refers to the
internal database ID, not a Telegram ID. Treat its output as retained chat data.
Use `audit --chat-id=CHAT_ID --telegram-message-id TELEGRAM_ID` to find an original
directly from its Telegram ID, including delivery-chunk aliases. `--generation`
selects retained history; without it audit includes all generations.

Stop the dedicated worker before restarting the bot with `./update.sh`. The bot
can continue accepted jobs. Do not run separate workers against the same live
queue merely to increase throughput.
