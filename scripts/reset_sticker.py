"""Offline reset of one generated sticker row so the builder will analyze it again."""
from __future__ import annotations

import argparse
from contextlib import closing
from pathlib import Path
import sqlite3
import sys


def reset_sticker(relative_path: str, *, index_db: Path) -> bool:
    """Remove the exact catalog key and its optional legacy FTS row atomically."""
    if not relative_path:
        raise ValueError('The sticker relative path must not be empty')
    # mode=rw prevents a misspelled database path from creating an empty catalog.
    uri = index_db.expanduser().resolve().as_uri() + '?mode=rw'
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        with connection:
            connection.execute('BEGIN IMMEDIATE')
            row = connection.execute(
                'SELECT rowid FROM stickers WHERE relative_path = ? COLLATE BINARY',
                (relative_path,),
            ).fetchone()
            if row is None:
                return False
            legacy_fts = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'sticker_fts'",
            ).fetchone()
            if legacy_fts is not None:
                connection.execute('DELETE FROM sticker_fts WHERE rowid = ?', (row[0],))
            connection.execute('DELETE FROM stickers WHERE rowid = ?', (row[0],))
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Reset one generated sticker catalog row. Stop services before using this offline maintenance tool.',
    )
    parser.add_argument('relative_path', metavar='RELATIVE_PATH', help='Exact relative_path stored in the sticker catalog; source files are preserved.')
    parser.add_argument('--index-db', type=Path, default=Path('./data/sticker_index.sqlite3'), help='Existing sticker catalog (default: ./data/sticker_index.sqlite3).')
    args = parser.parse_args(argv)
    try:
        found = reset_sticker(args.relative_path, index_db=args.index_db)
    except (OSError, sqlite3.Error, ValueError) as exc:
        print(f'Sticker reset failed: {exc}', file=sys.stderr)
        return 1
    if not found:
        print(f'No matching sticker for {args.relative_path!r}; catalog unchanged.', file=sys.stderr)
        return 1
    print(f'Reset generated catalog row for {args.relative_path!r}. Source files were preserved.')
    print('Run the sticker builder again, then refresh derived indexes before restarting services.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
