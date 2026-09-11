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
for tool in docker curl sha256sum tar flock; do command -v "$tool" >/dev/null || fail "Missing required tool: $tool"; done
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
start() { docker compose up -d --no-build --pull never --force-recreate --wait --wait-timeout "$timeout"; }
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
on_failure() {
  trap - ERR INT TERM
  log 'Activation failed; stopping attempted services' >&2
  docker compose stop bot retriever || true
  cp "$work/compose.before.yml" compose.yml
  if [[ -n $active ]]; then
    docker tag "$active" tgchatbot:current
    if start; then log 'Restored the prior image'; else log 'Restart failed; inspect docker compose logs' >&2; fi
  fi
  exit 1
}
trap on_failure ERR INT TERM
[[ $target == rollback ]] || cp "$work/compose.next.yml" compose.yml
if [[ -n $active ]]; then docker compose stop bot retriever; fi
# Keep a named recovery image before current changes, even if power is lost
# before the health check can finish. Reinstalling one image keeps the prior tag.
if [[ -n $active && $active != "$candidate" ]]; then docker tag "$active" tgchatbot:previous; fi
docker tag "$candidate" tgchatbot:current
start
if [[ $target != rollback ]]; then
  install_updater
fi
trap - ERR INT TERM
version=$(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.version"}}' tgchatbot:current)
log "Healthy release $version. Use docker compose up -d, logs, or stop directly."
