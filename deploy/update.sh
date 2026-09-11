#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

log() { printf 'tgchatbot-update: %s\n' "$*"; }
fail() { log "$*" >&2; exit 1; }
valid_tag() { [[ $1 =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; }
if [[ ${1:-} == -h || ${1:-} == --help ]]; then
  echo 'Usage: ./update.sh [latest|vX.Y.Z|rollback]'
  echo 'Default: install the latest prebuilt release using your existing .env.'
  exit 0
fi
[[ $# -le 1 ]] || fail 'Expected at most one argument'
target=${1:-latest}
case "$target" in latest|rollback) ;; *) valid_tag "$target" || fail 'Expected latest, vX.Y.Z or rollback';; esac
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$root"
repository=${TGCHATBOT_RELEASE_REPO:-mnihyc/tgchatbot}
timeout=${TGCHATBOT_HEALTH_TIMEOUT:-180}
[[ $repository =~ ^[0-9A-Za-z_.-]+/[0-9A-Za-z_.-]+$ ]] || fail 'Invalid release repository'
[[ $timeout =~ ^[1-9][0-9]{1,3}$ ]] && ((timeout <= 3600)) || fail 'Health timeout must be 10..3600 seconds'
for tool in docker curl sha256sum tar flock cmp; do command -v "$tool" >/dev/null || fail "Missing required tool: $tool"; done
docker compose version >/dev/null || fail 'Docker Compose v2 with up --wait support is required'
mkdir -p tmp/update data
exec 9>tmp/update.lock
flock -n 9 || fail 'Another update is running'
work=$root/tmp/update
# Only this updater's fixed scratch files are removed, including after failure.
cleanup() {
  rm -f -- "$work/SHA256SUMS" "$work/deploy.tar.gz" "$work/image.tar.gz" \
    "$work/compose.next.yml" "$work/compose.before.yml" "$work/update.next.sh" "$work/update.install.sh"
}
trap cleanup EXIT
download() { curl --fail --location --retry 3 --connect-timeout 15 --output "$2" "$1"; }
verify() {
  local expected
  expected=$(awk -v name="$1" '$2 == name { print $1; count++ } END { if (count != 1) exit 1 }' "$work/SHA256SUMS") || return 1
  [[ $expected =~ ^[0-9a-fA-F]{64}$ ]] || return 1
  printf '%s  %s\n' "$expected" "$2" | sha256sum --check --strict -
}
# Service changes belong to the selected release's Compose definition. Exclude
# the separately managed database, including when restoring an older release.
application_services() {
  docker compose --project-directory "$root" -f "$1" config --services | awk '$0 != "postgres"'
}
start() {
  local services
  services=$(application_services compose.yml)
  [[ -n $services ]] || fail 'Deployment has no application services'
  local -a names
  mapfile -t names <<< "$services"
  docker compose up -d --no-build --pull never --force-recreate --wait --wait-timeout "$timeout" "${names[@]}"
}
stop_application() {
  local services
  services=$(application_services "$1")
  if [[ -n $services ]]; then
    local -a names
    mapfile -t names <<< "$services"
    docker compose --project-directory "$root" -f "$1" stop "${names[@]}"
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
  [[ -f compose.previous.yml ]] || fail 'No matching previous Compose definition; install an exact release tag instead'
  cp compose.previous.yml "$work/compose.next.yml"
else
  if [[ $target == latest ]]; then
    url=$(curl --fail --silent --show-error --location --retry 3 --connect-timeout 15 --output /dev/null --write-out '%{url_effective}' "https://github.com/$repository/releases/latest")
    target=${url##*/}
    valid_tag "$target" || fail 'Cannot resolve latest to an exact release tag'
  fi
  [[ $(uname -s) == Linux ]] || fail 'Release images support Linux Docker hosts'
  case "$(uname -m)" in x86_64|amd64) arch=amd64;; aarch64|arm64) arch=arm64;; *) fail 'Releases support amd64 and arm64';; esac
  base=https://github.com/$repository/releases/download/$target
  bundle=tgchatbot-deploy-$target.tar.gz
  log "Downloading release $target deployment files"
  download "$base/SHA256SUMS" "$work/SHA256SUMS"
  download "$base/$bundle" "$work/deploy.tar.gz"
  verify "$bundle" "$work/deploy.tar.gz" || fail 'Deployment checksum mismatch'
  [[ $(tar -xOzf "$work/deploy.tar.gz" RELEASE_TAG) == "$target" ]] || fail 'Bundle tag mismatch'
  commit=$(tar -xOzf "$work/deploy.tar.gz" RELEASE_COMMIT)
  [[ $commit =~ ^[0-9a-f]{40}$ ]] || fail 'Invalid release commit'
  # Extract only named contents; archive paths/links never choose host paths.
  tar -xOzf "$work/deploy.tar.gz" compose.yml > "$work/compose.next.yml"
  tar -xOzf "$work/deploy.tar.gz" update.sh > "$work/update.next.sh"
  [[ -s "$work/compose.next.yml" && -s "$work/update.next.sh" ]] || fail 'Incomplete deployment bundle'
  bash -n "$work/update.next.sh" || fail 'Invalid release updater'
  # Transfer control before interpreting the new deployment, not after trying
  # to activate it with old service/schema assumptions. Re-opening descriptor 9
  # in the replacement process releases/reacquires the same update lock; if a
  # competing updater wins, it exits without changing application services.
  if ! cmp -s "$root/update.sh" "$work/update.next.sh"; then
    install_updater
    log 'Continuing with the verified release updater'
    exec bash "$root/update.sh" "$target"
  fi
  [[ -f compose.yml ]] || cp "$work/compose.next.yml" compose.yml
  if [[ ! -f .env ]]; then
    tar -xOzf "$work/deploy.tar.gz" .env.example > .env
    chmod 600 .env
    install_updater
    log 'Created .env and compose.yml. Fill in the Telegram token and one provider key/model, then run ./update.sh again.'
    exit 0
  fi
  docker compose --project-directory "$root" -f "$work/compose.next.yml" config --quiet
  image_asset=tgchatbot-linux-$arch.tar.gz
  log "Downloading prebuilt $arch image"
  download "$base/$image_asset" "$work/image.tar.gz"
  verify "$image_asset" "$work/image.tar.gz" || fail 'Image checksum mismatch'
  docker load --input "$work/image.tar.gz"
  image=tgchatbot:$target-$arch
  [[ $(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.version"}}' "$image") == "$target" ]] || fail 'Image release label mismatch'
  [[ $(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$image") == "$commit" ]] || fail 'Image commit label mismatch'
  candidate=$(docker image inspect --format '{{.Id}}' "$image")
fi

cp compose.yml "$work/compose.before.yml"
activation_started=0
on_failure() {
  trap - ERR INT TERM
  if [[ $activation_started == 0 ]]; then
    cp "$work/compose.before.yml" compose.yml
    log 'Preparation failed; existing application services were not restarted.' >&2
    exit 1
  fi
  log 'Activation failed; stopping attempted services' >&2
  stop_application compose.yml || true
  # Database state is never rolled back by swapping an image. The previous
  # application must prove it can read the current schema before it may restart.
  if [[ -n $active ]] && schema_compatible "$active"; then
    cp "$work/compose.before.yml" compose.yml
    docker tag "$active" tgchatbot:current
    if start; then log 'Restored the prior image'; else log 'Restart failed; inspect docker compose logs' >&2; fi
  else
    log 'Services remain stopped: no compatible prior image. Retained data is unchanged by the updater; install a compatible release or restore a matching backup.' >&2
  fi
  exit 1
}
trap on_failure ERR INT TERM
cp "$work/compose.next.yml" compose.yml
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
if [[ -n $active ]]; then stop_application "$work/compose.before.yml"; fi
if [[ $database_mode == external ]]; then
  # The previous bot may still have used this database until it stopped above.
  docker compose stop postgres
fi
# Keep a named recovery image before current changes, even if power is lost
# before the health check can finish. Reinstalling one image keeps the prior tag.
if [[ -n $active && $active != "$candidate" ]]; then
  cp "$work/compose.before.yml" compose.previous.yml
  docker tag "$active" tgchatbot:previous
fi
docker tag "$candidate" tgchatbot:current
start
if [[ $target != rollback ]]; then
  # Remove only retired, already-stopped application containers after success.
  # Their mounted data and images remain intact; no orphan/volume sweep is used.
  old_services=$(application_services "$work/compose.before.yml")
  new_services=$(application_services compose.yml)
  while IFS= read -r service; do
    [[ -n $service ]] || continue
    if ! printf '%s\n' "$new_services" | awk -v wanted="$service" '$0 == wanted { found=1 } END { exit !found }'; then
      if ! docker compose --project-directory "$root" -f "$work/compose.before.yml" rm -f "$service"; then
        log "Retired service $service is stopped but its container could not be removed" >&2
      fi
    fi
  done <<< "$old_services"
  install_updater
fi
trap - ERR INT TERM
version=$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.version"}}' tgchatbot:current)
log "Healthy release $version. Use ./update.sh to start/update, or docker compose logs/stop directly."
