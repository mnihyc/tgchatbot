"""Initialize only the deployment's local database credential.

Compose supplies its already parsed configuration on stdin. The updater runs
this inside the published image, so a host Python installation is unnecessary.
"""
from __future__ import annotations

import json
import io
import os
from pathlib import Path
import re
import secrets
import stat
import sys
import tempfile

from dotenv.parser import parse_stream


def write_environment(env_path: Path, contents: bytes) -> None:
    """Commit one complete credential update without truncating existing keys."""
    env_path = env_path.resolve(strict=True)
    original = env_path.stat()
    descriptor, name = tempfile.mkstemp(prefix='.env-update-', dir=env_path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            created = os.fstat(stream.fileno())
            if (created.st_uid, created.st_gid) != (original.st_uid, original.st_gid):
                os.fchown(stream.fileno(), original.st_uid, original.st_gid)
            os.fchmod(stream.fileno(), stat.S_IMODE(original.st_mode))
            stream.write(contents)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, env_path)
        directory = os.open(env_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def configure(configuration: dict, env_path: Path, data_path: Path) -> str:
    environment = configuration['services']['bot'].get('environment') or {}
    if environment.get('DATABASE_URL'):
        return 'external'
    if environment.get('POSTGRES_PASSWORD'):
        return 'local'

    cluster = data_path / 'postgres'
    try:
        initialized = cluster.is_dir() and any(cluster.iterdir())
    except PermissionError:
        initialized = True
    if initialized:
        raise RuntimeError('Existing PostgreSQL data needs its original POSTGRES_PASSWORD; restore the matching .env backup.')

    # Preserve all existing bot configuration verbatim. An empty placeholder is
    # filled in place; an absent key is appended once. Never source a dotenv file.
    original = env_path.read_bytes()
    declaration = re.compile(rb'^[ \t]*(?:export[ \t]+)?POSTGRES_PASSWORD(?:[ \t]*=.*)?[ \t]*\r?$', re.MULTILINE)
    line = ('POSTGRES_PASSWORD=' + secrets.token_hex(32)).encode('ascii')
    pieces, replaced = [], False
    # Locate real declarations with the dotenv parser. A multiline system
    # prompt can itself contain "POSTGRES_PASSWORD=" as ordinary quoted text.
    for binding in parse_stream(io.StringIO(original.decode('utf-8', errors='surrogateescape'))):
        piece = binding.original.string.encode('utf-8', errors='surrogateescape')
        if binding.key == 'POSTGRES_PASSWORD':
            piece, count = declaration.subn(lambda _match: line, piece, count=1)
            replaced = replaced or bool(count)
        pieces.append(piece)
    if replaced:
        updated = b''.join(pieces)
    else:
        updated = original + (b'' if original.endswith(b'\n') or not original else b'\n') + line + b'\n'
    write_environment(env_path, updated)
    return 'local'


if __name__ == '__main__':
    try:
        print(configure(json.load(sys.stdin), Path('/deployment/.env'), Path('/deployment/data')))
    except (KeyError, ValueError, OSError, RuntimeError) as exc:
        print(f'Database configuration failed: {exc}', file=sys.stderr)
        raise SystemExit(1) from exc
