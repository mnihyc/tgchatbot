#!/bin/sh
set -eu

# The account must exist in passwd for OpenSSH, including on hosts whose data
# owner is not UID 1000. Only the dedicated home is created; data is not chowned.
if [ "$(id -u)" = 0 ]; then
    uid=${TGCHATBOT_UID:-1000}
    gid=${TGCHATBOT_GID:-1000}
    case "$uid" in ''|0|0[0-9]*|*[!0-9]*) echo 'TGCHATBOT_UID must be a canonical nonzero integer' >&2; exit 1;; esac
    case "$gid" in ''|*[!0-9]*) echo 'TGCHATBOT_GID must be an integer' >&2; exit 1;; esac
    groupmod --non-unique --gid "$gid" tgchatbot
    usermod --non-unique --uid "$uid" --gid "$gid" tgchatbot
    install -d -m 700 -o "$uid" -g "$gid" /app/data/home
    exec setpriv --reuid "$uid" --regid "$gid" --init-groups "$0" "$@"
fi

if [ "${1:-}" = retriever ]; then
    index_dir=/app/data/tantivy_index
    if [ ! -f "$index_dir/meta.json" ]; then
        # Never silently replace a partial/corrupt index. Recovery is explicit.
        if [ -d "$index_dir" ] && [ -n "$(ls -A "$index_dir")" ]; then
            echo 'Existing Tantivy index lacks meta.json; inspect ./data/tantivy_index before rebuilding' >&2
            exit 1
        fi
        docs=/app/data/tantivy_docs.jsonl
        if [ ! -f "$docs" ]; then
            docs=/tmp/tgchatbot-empty-stickers.jsonl
            : > "$docs"
        fi
        sticker-retriever build --docs-jsonl "$docs" --index-dir "$index_dir"
    fi
    exec sticker-retriever serve --index-dir "$index_dir" --bind 0.0.0.0 --port 4107
fi
exec "$@"
