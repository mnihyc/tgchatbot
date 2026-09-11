# Deployment

Keep the deployment directory separate from the optional source checkout.
Docker runs the bot and PostgreSQL; `.env` and retained data belong to the
installation and survive application updates.

## Install and update

You need Linux, Docker with Compose v2, Bash, curl, tar, gzip, sha256sum and flock.
Download and unpack the release bundle, or use `deploy/update.sh` from the checkout.
Run the updater in the directory that will hold your deployment.

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

For v0.2.0 or earlier, refresh the updater once before upgrading. Run this from
the existing deployment directory:

```sh
curl -fL https://github.com/mnihyc/tgchatbot/releases/latest/download/update.sh -o update.sh.new &&
chmod 755 update.sh.new &&
mv update.sh.new update.sh &&
./update.sh
```

Keep the existing `compose.yml` until the updater runs; it is needed for service
cleanup and rollback. Keep `.env` and `data/` in place.

- Upgrading from v0.2.0 retains PostgreSQL conversation data. The replacement
  sticker catalog must be built explicitly from original media. Legacy sticker
  indexes are neither imported nor deleted.
- Upgrading from v0.1.5 or earlier starts a fresh PostgreSQL conversation database;
  legacy SQLite history is not migrated. Keep the original files. Those older
  versions cannot validate the PostgreSQL schema for automatic rollback.

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

The bot starts with an empty sticker catalog. Add original media and build it
explicitly; normal updates do not regenerate it or make annotation requests.
See [maintenance commands](https://github.com/mnihyc/tgchatbot/blob/main/scripts/README.md) for append, selected regeneration,
corrections, imports and memory work.

To stop everything, use `docker compose stop`; restart with `./update.sh` so the
bundled database is started as needed. For maintenance that uses PostgreSQL,
stop only `bot`. A stopped profiled database is not restarted by every plain
`docker compose up` command.
