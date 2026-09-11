#!/usr/bin/env bash
set -Eeuo pipefail
umask 077

log() { printf 'tgchatbot-update: %s\n' "$*"; }
fail() { log "$*" >&2; exit 1; }
usage() {
  cat <<'EOF'
Usage: ./update.sh [latest|vX.Y.Z|rollback|recover|status]
Downloads a checksummed release Docker image; never builds locally.
Local .env and data/ are preserved. Failed activation restores previous code.
TGCHATBOT_RELEASE_REPO defaults to the upstream repository.
TGCHATBOT_HEALTH_TIMEOUT defaults to 180 seconds (10..3600).
TGCHATBOT_UID/GID default to your current nonroot account on first install.
EOF
}
[[ ${1:-} != -h && ${1:-} != --help ]] || { usage; exit 0; }
[[ $# -le 1 ]] || fail 'Expected at most one argument'
target=${1:-latest}
valid_tag() { [[ $1 =~ ^v[0-9]+\.[0-9]+\.[0-9]+(-[0-9A-Za-z.-]+)?$ ]]; }
case "$target" in latest|rollback|recover|status) ;; *) valid_tag "$target" || fail 'Expected latest, vX.Y.Z, rollback, recover or status';; esac

root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime=$root/runtime
repository=${TGCHATBOT_RELEASE_REPO:-mnihyc/tgchatbot}
timeout=${TGCHATBOT_HEALTH_TIMEOUT:-180}
[[ $repository =~ ^[0-9A-Za-z_.-]+/[0-9A-Za-z_.-]+$ ]] || fail 'Invalid release repository'
[[ $timeout =~ ^[1-9][0-9]{1,3}$ ]] && ((timeout >= 10 && timeout <= 3600)) || fail 'Health timeout must be 10..3600 seconds'
for tool in docker curl sha256sum tar gzip flock; do command -v "$tool" >/dev/null || fail "Missing required tool: $tool"; done
docker compose version >/dev/null || fail 'Docker Compose v2 with up --wait support is required'
mkdir -p "$runtime/releases" "$runtime/downloads" "$runtime/transactions"
exec 9>"$runtime/update.lock"
flock -n 9 || fail 'Another update is running'

compose() {
  local release=$1; shift
  # Shell exports must not override the selected, verified image. The bot's
  # .env is bind-mounted and is never sourced or interpreted by this script.
  env -u TGCHATBOT_IMAGE -u TGCHATBOT_UID -u TGCHATBOT_GID docker compose \
    --project-directory "$root" --env-file "$release/release.env" \
    -f "$release/compose.yml" "$@"
}
current() { [[ -L "$runtime/current" ]] && readlink -f "$runtime/current"; }
link_release() {
  ln -sfn "$1" "$runtime/$2.next"
  mv -Tf "$runtime/$2.next" "$runtime/$2"
}
archive_pending() {
  [[ ! -f "$runtime/pending" ]] || mv "$runtime/pending" "$runtime/transactions/$(date -u +%Y%m%dT%H%M%S)-$$-$RANDOM"
}
start() { compose "$1" up -d --no-build --pull never --force-recreate --wait --wait-timeout "$timeout"; }
install_entrypoints() {
  local release=$1
  cp "$release/compose.yml" "$root/compose.yml"
  cp "$release/release.env" "$runtime/release.env"
  cp "$release/RELEASE_TAG" "$root/RELEASE_TAG"
  cp "$release/update.sh" "$root/update.sh.next"
  chmod 755 "$root/update.sh.next"
  mv -f "$root/update.sh.next" "$root/update.sh"
}
recover() {
  local pending prior
  [[ -f "$runtime/pending" ]] || { log 'No interrupted activation'; return 0; }
  pending=$(cat "$runtime/pending")
  [[ $pending == "$runtime/releases/"* && -f "$pending/release.env" ]] || fail 'Invalid pending release; inspect ./runtime/pending'
  prior=$(current || true)
  compose "$pending" stop bot retriever || return 1
  if [[ -n $prior ]]; then
    start "$prior" || return 1
    install_entrypoints "$prior" || return 1
    log "Restored $(cat "$prior/RELEASE_TAG")"
  else
    log 'First activation stopped; local data retained'
  fi
  archive_pending
}
if [[ $target == status ]]; then
  active=$(current || true)
  [[ -n $active ]] || { log 'No active release'; exit 0; }
  log "Active $(cat "$active/RELEASE_TAG")"
  compose "$active" ps
  [[ ! -f "$runtime/pending" ]] || log 'Interrupted activation: run ./update.sh recover'
  exit 0
fi
if [[ $target == recover ]]; then recover; exit; fi
[[ ! -f "$runtime/pending" ]] || fail 'Interrupted activation: run ./update.sh recover first'
[[ -f "$root/.env" ]] || fail 'Create ./.env from .env.example before starting'
mkdir -p "$root/data"
active=$(current || true)

download() { curl --fail --location --retry 3 --connect-timeout 15 --output "$2" "$1"; }
verify() {
  local name=$1 directory=$2
  (cd "$directory" && awk -v name="$name" '$2 == name { print; count++ } END { if (count != 1) exit 1 }' SHA256SUMS | sha256sum --check --strict -)
}

if [[ $target == rollback ]]; then
  [[ -L "$runtime/previous" ]] || fail 'No previous release to roll back to'
  release=$(readlink -f "$runtime/previous")
  [[ -f "$release/release.env" ]] || fail 'Previous release is incomplete'
  image=$(sed -n 's/^TGCHATBOT_IMAGE=//p' "$release/release.env")
  docker image inspect "$image" >/dev/null || fail 'Previous image is not loaded; install its exact release tag again'
else
  if [[ $target == latest ]]; then
    url=$(curl --fail --silent --show-error --location --retry 3 --connect-timeout 15 --output /dev/null --write-out '%{url_effective}' "https://github.com/$repository/releases/latest")
    target=${url##*/}
    valid_tag "$target" || fail 'Cannot resolve latest to an exact release tag'
  fi
  case "$(uname -m)" in x86_64|amd64) arch=amd64;; aarch64|arm64) arch=arm64;; *) fail 'Releases support Linux amd64 and arm64';; esac
  [[ $(uname -s) == Linux ]] || fail 'This deployment script supports Linux Docker hosts'
  image_asset=tgchatbot-linux-$arch.tar.gz
  bundle_asset=tgchatbot-deploy-$target.tar.gz
  download_dir=$(mktemp -d "$runtime/downloads/$target-$arch.XXXXXX")
  base=https://github.com/$repository/releases/download/$target
  log "Downloading exact release $target ($arch)"
  download "$base/SHA256SUMS" "$download_dir/SHA256SUMS"
  download "$base/$bundle_asset" "$download_dir/$bundle_asset"
  verify "$bundle_asset" "$download_dir" || fail 'Deployment bundle checksum mismatch'
  download "$base/$image_asset" "$download_dir/$image_asset"
  verify "$image_asset" "$download_dir" || fail 'Image checksum mismatch'
  # Extract named regular-file contents into paths chosen here. Never unpack
  # release archive paths, links, permissions or ownership onto the host.
  release=$(mktemp -d "$runtime/releases/$target-$arch.XXXXXX")
  for name in compose.yml update.sh README.md RELEASE_TAG RELEASE_COMMIT; do
    tar -xOzf "$download_dir/$bundle_asset" "$name" > "$release/$name"
    [[ -s "$release/$name" ]] || fail "Missing deployment bundle member: $name"
  done
  [[ $(cat "$release/RELEASE_TAG") == "$target" ]] || fail 'Bundle tag mismatch'
  commit=$(cat "$release/RELEASE_COMMIT")
  [[ $commit =~ ^[0-9a-f]{40}$ ]] || fail 'Invalid release commit'
  image=tgchatbot:$target-$arch
  docker load --input "$download_dir/$image_asset"
  [[ $(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.version"}}' "$image") == "$target" ]] || fail 'Image release label mismatch'
  [[ $(docker image inspect --format '{{index .Config.Labels "org.opencontainers.image.revision"}}' "$image") == "$commit" ]] || fail 'Image commit label mismatch'
  uid=${TGCHATBOT_UID:-$(id -u)}
  gid=${TGCHATBOT_GID:-$(id -g)}
  if [[ -n $active ]]; then
    uid=${TGCHATBOT_UID:-$(sed -n 's/^TGCHATBOT_UID=//p' "$active/release.env")}
    gid=${TGCHATBOT_GID:-$(sed -n 's/^TGCHATBOT_GID=//p' "$active/release.env")}
  fi
  [[ $uid =~ ^[1-9][0-9]*$ && $gid =~ ^[0-9]+$ ]] || fail 'Set TGCHATBOT_UID/GID to the nonroot data owner'
  printf 'TGCHATBOT_IMAGE=%s\nTGCHATBOT_UID=%s\nTGCHATBOT_GID=%s\n' "$image" "$uid" "$gid" > "$release/release.env"
fi

# Check the Compose contract before stopping the old bot. A persistent pending
# marker makes SIGKILL/power loss recoverable; current changes only on success.
compose "$release" config --quiet
printf '%s\n' "$release" > "$runtime/pending"
on_failure() {
  trap - ERR INT TERM
  log 'Activation failed; attempting to restart the active release' >&2
  recover || log 'Recovery incomplete: inspect docker compose logs, then run ./update.sh recover' >&2
  exit 1
}
trap on_failure ERR INT TERM
if [[ -n $active ]]; then compose "$active" stop bot retriever; fi
start "$release"
# Stage convenience files before promoting current. If a copy fails or the
# process is interrupted, recover still knows which healthy release to restore.
install_entrypoints "$release"
if [[ -n $active && $active != "$release" ]]; then link_release "$active" previous; fi
link_release "$release" current
archive_pending
trap - ERR INT TERM
log "Healthy release $(cat "$release/RELEASE_TAG"). Data and .env preserved."
