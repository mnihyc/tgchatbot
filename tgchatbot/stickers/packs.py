"""List, inspect or edit optional descriptions for existing sticker packs."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv
from psycopg import Error as DatabaseError

from tgchatbot.config import load_config
from tgchatbot.storage.postgres_store import PostgresStore
from tgchatbot.storage.sticker_catalog import StickerCatalogStore


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('list', help='List exact pack IDs and their descriptions')
    get = commands.add_parser('get', help='Show one pack description')
    get.add_argument('pack', help='Exact pack ID shown by list')
    set_description = commands.add_parser('set', help='Set or replace a pack description')
    set_description.add_argument('pack', help='Exact pack ID shown by list')
    set_description.add_argument('description', help='Description to show beside each matching candidate')
    remove = commands.add_parser('remove', help='Remove a description, preserving the pack and its stickers')
    remove.add_argument('pack', help='Exact pack ID shown by list')
    return parser.parse_args(argv)


async def run(args):
    load_dotenv(Path.cwd() / '.env')
    config = load_config(require_telegram=False)
    store = PostgresStore(config.database_url)
    try:
        await store.pool.open(wait=True)
        catalog = StickerCatalogStore(store)
        await catalog.initialize()
        if args.command in {'list', 'get'}:
            descriptions = await catalog.list_pack_descriptions()
            if args.command == 'list':
                result = descriptions
            else:
                if args.pack not in descriptions:
                    raise KeyError(f'Unknown sticker pack: {args.pack}')
                result = {'pack': args.pack, 'description': descriptions[args.pack]}
        else:
            description = args.description if args.command == 'set' else None
            revision_id = await catalog.update_pack_descriptions({args.pack: description})
            result = {'pack': args.pack, 'description': description, 'revision_id': revision_id}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    finally:
        await store.close()


def main(argv=None):
    try:
        return asyncio.run(run(parse_args(argv)))
    except (KeyError, ValueError, RuntimeError, OSError, DatabaseError) as exc:
        print(json.dumps({'error': str(exc.args[0]) if isinstance(exc, KeyError) else str(exc)}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
