"""Inspect sticker catalogs and export corrections without calling model APIs."""
from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
from pathlib import Path

from dotenv import load_dotenv
from psycopg import Error as DatabaseError

from tgchatbot.config import load_config
from tgchatbot.domain.timestamps import format_timestamp
from tgchatbot.storage.postgres_store import PostgresStore
from tgchatbot.storage.sticker_catalog import StickerCatalogStore


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('revisions', help='List all revisions, including interrupted staging work')
    for name, help_text in (
        ('status', 'Show counts, saved build channels and failures'),
        ('assets', 'List asset IDs, file aliases, captions and states'),
        ('asset', 'Inspect one original description, effective card and corrections'),
        ('corrections', 'Export complete overrides accepted by build_sticker_index --corrections'),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument('--revision', help='Catalog revision ID; defaults to the active catalog')
        if name == 'asset':
            command.add_argument('asset_id', help='Content ID from assets or sid reference from sticker_query')
        if name in {'assets', 'corrections'}:
            command.add_argument('--pack', help='Filter by exact current pack membership')
        if name == 'assets':
            command.add_argument('--state', choices=['ready', 'pending', 'failed'])
        if name == 'corrections':
            command.add_argument('--asset-id', action='append', help='Select an asset; may repeat')
    return parser.parse_args(argv)


def emit(record):
    def present(value):
        if isinstance(value, datetime):
            return format_timestamp(value)
        raise TypeError(f'Cannot render {type(value).__name__}')
    print(json.dumps(record, ensure_ascii=False, default=present), flush=True)


async def run(args):
    load_dotenv(Path.cwd() / '.env')
    config = load_config(require_telegram=False)
    store = PostgresStore(config.database_url)
    try:
        await store.pool.open(wait=True)
        catalog = StickerCatalogStore(store)
        await catalog.initialize()
        if args.command == 'revisions':
            async for revision in catalog.iter_revisions():
                emit(revision)
        elif args.command == 'status':
            emit(await catalog.inspect_revision(args.revision))
        elif args.command == 'assets':
            async for asset in catalog.iter_assets(args.revision, pack=args.pack, state=args.state):
                emit(asset)
        elif args.command == 'asset':
            emit(await catalog.inspect_asset(args.asset_id, args.revision))
        elif args.command == 'corrections':
            emit(await catalog.export_corrections(args.revision, asset_ids=args.asset_id, pack=args.pack))
        return 0
    finally:
        await store.close()


def main(argv=None):
    try:
        return asyncio.run(run(parse_args(argv)))
    except (KeyError, ValueError, RuntimeError, OSError, DatabaseError) as exc:
        emit({'error': str(exc.args[0]) if isinstance(exc, KeyError) else str(exc)})
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
