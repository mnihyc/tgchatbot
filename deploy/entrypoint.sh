#!/bin/sh
set -eu

# The account must exist in passwd for OpenSSH, including on hosts whose data
# owner is not UID 1000. Only the dedicated home is created; data is not chowned.
if [ "$(id -u)" = 0 ]; then
    uid=$(stat -c %u /app/data)
    gid=$(stat -c %g /app/data)
    install -d -m 700 -o "$uid" -g "$gid" /app/data/home
    # Docker may create a new scratch bind mount as root. Change only its top
    # directory; retained data and existing descendants keep their ownership.
    install -d -m 700 -o "$uid" -g "$gid" /tmp
    if [ "$uid" != 0 ]; then
        groupmod --non-unique --gid "$gid" tgchatbot
        usermod --non-unique --uid "$uid" --gid "$gid" tgchatbot
        exec setpriv --reuid "$uid" --regid "$gid" --init-groups "$0" "$@"
    fi
    # Root-owned bind mounts are valid too; continue once as their owner.
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
