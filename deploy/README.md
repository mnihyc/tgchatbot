# Deployment

Keep the deployment directory separate from the optional source checkout.
Docker runs the bot and PostgreSQL; `.env` and retained data belong to the
installation and survive application updates.

## Install and update

You need Linux, Docker with Compose v2, Bash, curl, tar, gzip and flock.
Download and unpack the release bundle. Alternatively, copy `deploy/update.sh`
from the checkout into a separate deployment directory. The updater keeps the
installation beside its own file.

```sh
./update.sh                 # install or update to the latest release
./update.sh vX.Y.Z          # select a published version
./update.sh rollback        # restore the previous compatible application

docker compose ps
docker compose logs -f
```

A new installation creates `.env`; fill in `TGBOT_TOKEN` and a model provider
key, then run the updater again. Existing settings are preserved. The updater
generates a bundled database password in that same file when needed. See the
[configuration examples](https://github.com/mnihyc/tgchatbot/blob/main/.env.full.example) for optional settings.

Docker installs the dependencies on the first run and caches them for updates.

The existing bot keeps running while Docker prepares the candidate image.
The updater checks database compatibility before switching, waits for application
health, and retains the previous image with its matching Compose definition.
Failed activation may restore that previous compatible application. Rollback
does not reverse database changes or external actions.
Use `./update.sh` to start stopped services or apply changes to `.env`.

An explicit `DATABASE_URL` selects an existing PostgreSQL database with pgvector.
Otherwise the updater starts the bundled database. No separate Compose profile
setting is required. The bundled database exposes no host port.

## Upgrading older installations

To refresh an older updater, download the latest deployment bundle and extract
only `update.sh` into the existing deployment directory:

```sh
tar -xzf /path/to/tgchatbot-deploy-vX.Y.Z.tar.gz update.sh
./update.sh
```

Keep the existing `compose.yml` until the updater runs; it is needed for service
cleanup and rollback. Keep `.env` and `data/` in place.

This version requires a fresh conversation schema. An older database is rejected
without changing its records. Automatic database migration is not provided;
keep the existing installation and its backups when preparing a new database.
Original Telegram Desktop exports can be imported into the new database, and
the sticker catalog can be rebuilt from original media.

## Retained data

| Location | Contents |
| --- | --- |
| `.env` | Credentials and application configuration |
| `data/postgres/` | Bundled PostgreSQL database, including conversations and sticker catalog |
| `data/presets/` | Personality prompt files |
| `data/stickers/` | Original sticker media |
| `data/home/` | Persistent SSH home and trusted-host state |
| `tmp/` | Temporary application files and updater downloads |
| `backups/` | Recovery copies you create |

PostgreSQL owns its cluster files. Do not copy a running cluster directory as a
backup; use a database dump or stop the database first. Keep original sticker
media available: the catalog records descriptions and vectors, not another copy
of the media library. Files uploaded through optional SSH tools reside on the
configured remote host.

For a bundled-database backup, leave PostgreSQL running while stopping the bot:

```sh
docker compose stop bot
mkdir -p backups
umask 077
backup_stamp=$(date -u +%Y%m%dT%H%M%SZ)
docker compose exec -T postgres pg_dump -U tgchatbot -d tgchatbot -Fc > "backups/chat-$backup_stamp.dump"
tar --exclude=data/postgres -czf "backups/files-$backup_stamp.tar.gz" .env data
./update.sh
```

Copy backups to your recovery storage. For an external database, use its backup
tooling. Restore into a compatible PostgreSQL/pgvector installation while the
bot is stopped, retaining the corresponding `.env` and file backup.

## Optional features and maintenance

Configure SSH in `.env` before enabling remote tools. Keep an SSH identity under
`data/`, for example `SSH_EXEC_IDENTITY_FILE=./data/your_key`. Known hosts are
retained under `data/home/.ssh/known_hosts`.

Each chat has one working directory, also available as `TGCHATBOT_SESSION_DIR`.
Uploads use `YYYY-MM-DD/name_<file-id>.ext`; tool paths are relative to that
directory. Files remain until explicitly removed.

`read_doc` reads a selected workspace file through the remote host's `python3`.
Install Pillow for images and pypdfium2 for PDFs in that Python environment
(for example, `uv pip install --python python3 pillow pypdfium2`). Parsing and
rendering run remotely; only selected text and compressed images return to the
conversation. Text reads need no additional Python packages. Audio/video tool
results report unsupported on the current API routes.

The bot starts with an empty sticker catalog. Add original media and build it
explicitly; normal updates do not regenerate it or make annotation requests.
See [maintenance commands](https://github.com/mnihyc/tgchatbot/blob/main/scripts/README.md) for catalog inspection, selected regeneration,
corrections, imports, profiles and memory work.

To stop everything, use `docker compose stop`; restart with `./update.sh` so the
bundled database is started as needed. For maintenance that uses PostgreSQL,
stop only `bot`. A stopped profiled database is not restarted by every plain
`docker compose up` command.
