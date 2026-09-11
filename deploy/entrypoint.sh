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

exec "$@"
