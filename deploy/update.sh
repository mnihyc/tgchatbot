#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

log() { printf 'tgchatbot-update: %s\n' "$*"; }
fail() { log "$*" >&2; exit 1; }
valid_tag() { [[ $1 =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; }
if [[ ${1:-} == -h || ${1:-} == --help ]]; then
  echo 'Usage: ./update.sh [latest|vX.Y.Z|rollback]'
  echo '       ./update.sh backup [path.sql]'
  echo '       ./update.sh restore path.sql'
  echo 'Default: build and install the latest code release using Docker and your existing .env.'
  echo 'Backup/restore use the running bundled PostgreSQL service; restore replaces application schemas.'
  exit 0
fi
target=${1:-latest}
case "$target" in
  backup) [[ $# -le 2 ]] || fail 'Usage: ./update.sh backup [path.sql]' ;;
  restore) [[ $# == 2 ]] || fail 'Usage: ./update.sh restore path.sql' ;;
  *)
    [[ $# -le 1 ]] || fail 'Expected at most one argument'
    case "$target" in latest|rollback) ;; *) valid_tag "$target" || fail 'Expected latest, vX.Y.Z, rollback, backup or restore';; esac
    ;;
esac
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$root"
repository=${TGCHATBOT_RELEASE_REPO:-mnihyc/tgchatbot}
timeout=${TGCHATBOT_HEALTH_TIMEOUT:-180}
[[ $repository =~ ^[0-9A-Za-z_.-]+/[0-9A-Za-z_.-]+$ ]] || fail 'Invalid release repository'
[[ $timeout =~ ^[1-9][0-9]*$ ]] || fail 'Health timeout must be a positive number of seconds'
required_tools=(docker flock)
if [[ $target != backup && $target != restore ]]; then required_tools+=(curl tar cmp); fi
for tool in "${required_tools[@]}"; do command -v "$tool" >/dev/null || fail "Missing required tool: $tool"; done
docker compose version >/dev/null || fail 'Docker Compose v2 with up --wait support is required'
mkdir -p tmp/update data
exec 9>tmp/update.lock
flock -n 9 || fail 'Another update is running'
work=$root/tmp/update
backup_pending=
# Only this updater's fixed scratch files are removed, including after failure.
cleanup() {
  rm -f -- "$work/deploy.tar.gz" \
    "$work/compose.next.yml" "$work/update.next.sh" "$work/update.install.sh"
  rm -rf -- "$work/build-context"
  if [[ -n $backup_pending ]]; then rm -f -- "$backup_pending"; fi
}
trap cleanup EXIT
cleanup

if [[ $target == backup || $target == restore ]]; then
  [[ -f compose.yml ]] || fail 'Run ./update.sh to prepare the deployment first'
  running=$(docker compose ps --status running --services)
  printf '%s\n' "$running" | awk '$0 == "postgres" { found=1 } END { exit !found }' \
    || fail 'The bundled PostgreSQL service must be running'
  if [[ $target == backup ]]; then
    destination=${2:-backups/database-$(date -u +%Y%m%dT%H%M%SZ)-$$.sql}
    mkdir -p -- "$(dirname -- "$destination")"
    backup_pending=$(mktemp -- "$destination.partial.XXXXXX")
    # PostgreSQL supplies a consistent snapshot while the application stays live.
    docker compose exec -T postgres sh -ec \
      'export PGPASSWORD="$POSTGRES_PASSWORD"; exec pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" --format=plain --clean --if-exists --no-owner --no-privileges' \
      > "$backup_pending"
    mv -- "$backup_pending" "$destination"
    backup_pending=
    log "Database backup saved: $destination"
    exit 0
  fi

  [[ -f $2 && -r $2 && -s $2 ]] || fail 'Restore requires a readable, nonempty SQL backup'
  running_apps=()
  while IFS= read -r service; do
    if [[ -n $service && $service != postgres ]]; then running_apps+=("$service"); fi
  done <<< "$running"
  resume_apps() {
    if [[ ${#running_apps[@]} != 0 ]]; then
      docker compose start "${running_apps[@]}"
    fi
  }
  trap 'log "Restore interrupted; its result is unconfirmed. Check PostgreSQL before restarting application services." >&2; exit 130' INT
  trap 'log "Restore interrupted; its result is unconfirmed. Check PostgreSQL before restarting application services." >&2; exit 143' TERM
  if [[ ${#running_apps[@]} != 0 ]] && ! docker compose stop "${running_apps[@]}"; then
    resume_apps || fail 'Application pause and restart failed; restore has not started'
    fail 'Could not pause application services; restore has not started'
  fi
  log 'Restoring application schemas from SQL'
  restored=0
  # An explicit final COMMIT also makes a broken input stream roll back instead
  # of letting psql's automatic single-transaction mode commit a premature EOF.
  if (
    cat <<'SQL'
BEGIN;
SET LOCAL client_min_messages = warning;
DO $restore$
DECLARE item record;
BEGIN
  FOR item IN SELECT nspname FROM pg_namespace
      WHERE nspname !~ '^pg_' AND nspname <> 'information_schema'
  LOOP
    EXECUTE format('DROP SCHEMA IF EXISTS %I CASCADE', item.nspname);
  END LOOP;
END $restore$;
CREATE SCHEMA public AUTHORIZATION pg_database_owner;
GRANT USAGE ON SCHEMA public TO PUBLIC;
SQL
    cat -- "$2" || exit 1
    printf '\nCOMMIT;\n'
  ) | docker compose exec -T postgres sh -ec \
      'export PGPASSWORD="$POSTGRES_PASSWORD"; exec psql -X -q -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB" -f -'; then
    restored=1
  fi
  if ! resume_apps; then
    if [[ $restored == 1 ]]; then
      fail 'Database restored, but application restart failed; inspect docker compose logs'
    fi
    fail 'Restore and application restart failed; inspect PostgreSQL output and docker compose logs'
  fi
  [[ $restored == 1 ]] || fail 'Restore failed or its result could not be confirmed; inspect PostgreSQL output'
  trap - INT TERM
  log "Database restored: $2"
  exit 0
fi

download() { curl --fail --location --retry 3 --connect-timeout 15 --output "$2" "$1"; }
# Use the operator's Compose configuration throughout, excluding the separately
# managed database when starting or stopping application services.
application_services() {
  docker compose config --services | awk '$0 != "postgres"'
}
start() {
  local services
  services=$(application_services)
  [[ -n $services ]] || fail 'Deployment has no application services'
  local -a names
  mapfile -t names <<< "$services"
  docker compose up -d --no-build --pull never --force-recreate --wait --wait-timeout "$timeout" "${names[@]}"
}
stop_application() {
  local services
  services=$(application_services)
  if [[ -n $services ]]; then
    local -a names
    mapfile -t names <<< "$services"
    docker compose stop "${names[@]}"
  fi
}
schema_compatible() {
  TGCHATBOT_IMAGE="$1" docker compose run --rm --no-deps bot python -m tgchatbot.storage.postgres_store --check-schema
}
install_updater() {
  cp "$work/update.next.sh" "$work/update.install.sh"
  chmod 755 "$work/update.install.sh"
  mv -f "$work/update.install.sh" update.sh
}
active=$(docker image inspect --format '{{.Id}}' tgchatbot:current 2>/dev/null || true)

if [[ $target == rollback ]]; then
  [[ -f .env && -f compose.yml ]] || fail 'Run ./update.sh before rollback'
  candidate=$(docker image inspect --format '{{.Id}}' tgchatbot:previous 2>/dev/null) || fail 'No previous image; install an exact release tag instead'
else
  if [[ $target == latest ]]; then
    url=$(curl --fail --silent --show-error --location --retry 3 --connect-timeout 15 --output /dev/null --write-out '%{url_effective}' "https://github.com/$repository/releases/latest")
    target=${url##*/}
    valid_tag "$target" || fail 'Cannot resolve latest to an exact release tag'
  fi
  base=https://github.com/$repository/releases/download/$target
  bundle=tgchatbot-deploy-$target.tar.gz
  log "Downloading release $target deployment files"
  download "$base/$bundle" "$work/deploy.tar.gz"
  [[ $(tar -xOzf "$work/deploy.tar.gz" RELEASE_TAG) == "$target" ]] || fail 'Bundle tag mismatch'
  commit=$(tar -xOzf "$work/deploy.tar.gz" RELEASE_COMMIT)
  [[ $commit =~ ^[0-9a-f]{40}$ ]] || fail 'Invalid release commit'
  # Extract only named contents; archive paths/links never choose host paths.
  tar -xOzf "$work/deploy.tar.gz" update.sh > "$work/update.next.sh"
  [[ -s "$work/update.next.sh" ]] || fail 'Incomplete deployment bundle'
  bash -n "$work/update.next.sh" || fail 'Invalid release updater'
  # Transfer control before interpreting the new deployment, not after trying
  # to activate it with old service/schema assumptions. Re-opening descriptor 9
  # in the replacement process releases/reacquires the same update lock; if a
  # competing updater wins, it exits without changing application services.
  if ! cmp -s "$root/update.sh" "$work/update.next.sh"; then
    install_updater
    log 'Continuing with the release updater'
    exec bash "$root/update.sh" "$target"
  fi
  if [[ ! -f compose.yml ]]; then
    tar -xOzf "$work/deploy.tar.gz" compose.yml > "$work/compose.next.yml"
    [[ -s "$work/compose.next.yml" ]] || fail 'Missing Compose template'
    cp "$work/compose.next.yml" compose.yml
  fi
  if [[ ! -f .env ]]; then
    tar -xOzf "$work/deploy.tar.gz" .env.example > .env
    chmod 600 .env
    install_updater
    log 'Created .env. Fill in the Telegram token and one provider key/model, then run ./update.sh again.'
    exit 0
  fi
  docker compose config --quiet
  mapfile -t wheels < <(tar -tzf "$work/deploy.tar.gz" | awk '/^build\/tgchatbot-[0-9A-Za-z_.+-]+-py3-none-any\.whl$/')
  [[ ${#wheels[@]} == 1 ]] || fail 'Release must contain one CPU-neutral tgchatbot wheel'
  mkdir -p "$work/build-context/build" "$work/build-context/deploy"
  for member in Dockerfile .dockerignore build/runtime-requirements.txt "${wheels[0]}" deploy/entrypoint.sh deploy/configure_database.py; do
    tar -xOzf "$work/deploy.tar.gz" "$member" > "$work/build-context/$member"
  done
  image=tgchatbot:$target
  log 'Preparing the application with Docker'
  docker build --build-arg "RELEASE_TAG=$target" --build-arg "RELEASE_COMMIT=$commit" \
    --tag "$image" "$work/build-context"
  [[ $(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.version"}}' "$image") == "$target" ]] || fail 'Image release label mismatch'
  [[ $(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$image") == "$commit" ]] || fail 'Image commit label mismatch'
  candidate=$(docker image inspect --format '{{.Id}}' "$image")
fi

activation_started=0
on_failure() {
  trap - ERR INT TERM
  if [[ $activation_started == 0 ]]; then
    log 'Preparation failed; existing application services were not restarted.' >&2
    exit 1
  fi
  log 'Activation failed; stopping attempted services' >&2
  stop_application || true
  # Database state is never rolled back by swapping an image. The previous
  # application must prove it can read the current schema before it may restart.
  if [[ -n $active ]] && schema_compatible "$active"; then
    docker tag "$active" tgchatbot:current
    if start; then log 'Restored the prior image'; else log 'Restart failed; inspect docker compose logs' >&2; fi
  else
    log 'Services remain stopped: no compatible prior image. Retained data is unchanged by the updater; install a compatible release or restore a matching backup.' >&2
  fi
  exit 1
}
trap on_failure ERR INT TERM
# Compose owns dotenv parsing. The published helper sees the resolved bot
# environment and creates only a missing local database password, without
# printing credentials or requiring Python on the Docker host.
database_mode=$(docker compose config --format json | docker run --rm -i --network none \
  --user "$(id -u):$(id -g)" --entrypoint python \
  -v "$root:/deployment" -v "$root/data:/deployment/data:ro" \
  "$candidate" /usr/local/lib/tgchatbot-deploy-configure.py)
case "$database_mode" in
  local)
    docker compose pull --policy missing postgres
    docker compose up -d --no-build --pull never --wait --wait-timeout "$timeout" postgres
    ;;
  external)
    # Validate the external database before stopping any existing local service.
    ;;
  *) fail 'Could not determine database configuration' ;;
esac
schema_compatible "$candidate"
activation_started=1
if [[ -n $active ]]; then stop_application; fi
if [[ $database_mode == external ]]; then
  # The previous bot may still have used this database until it stopped above.
  docker compose stop postgres
fi
# Keep a named recovery image before current changes, even if power is lost
# before the health check can finish. Reinstalling one image keeps the prior tag.
if [[ -n $active && $active != "$candidate" ]]; then
  docker tag "$active" tgchatbot:previous
fi
docker tag "$candidate" tgchatbot:current
start
if [[ $target != rollback ]]; then
  install_updater
fi
trap - ERR INT TERM
version=$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.version"}}' tgchatbot:current)
log "Healthy release $version. Use ./update.sh to start/update, or docker compose logs/stop directly."
