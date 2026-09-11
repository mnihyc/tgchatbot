#!/usr/bin/env python3
"""Inspect a catalog shortlist without changing preferences or sending Telegram media."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv
from tgchatbot.config import load_config
from tgchatbot.embeddings import EmbeddingClient, sticker_embedding_config
from tgchatbot.stickers.catalog import StickerCatalog
from tgchatbot.stickers.plan import StickerRetrievalPlan
from tgchatbot.storage.postgres_store import PostgresStore
from tgchatbot.storage.sticker_catalog import StickerCatalogStore
from tgchatbot.storage.sticker_delivery import StickerDeliveryStore
from tgchatbot.tools.sticker_send import _candidate_payload


async def run(args):
    load_dotenv()
    config = load_config(require_telegram=False)
    store = PostgresStore(config.database_url)
    embeddings = EmbeddingClient(sticker_embedding_config())
    try:
        await store.initialize()
        catalog_store = StickerCatalogStore(store)
        await catalog_store.initialize()
        delivery_store = StickerDeliveryStore(store)
        await delivery_store.initialize()
        catalog = StickerCatalog(catalog_store, args.source or config.sticker_dir,
            persona_store=store, embedding_client=embeddings, delivery_store=delivery_store)
        payload = json.loads(args.plan) if args.plan else {'intent_core': args.intent_core}
        if args.candidate_budget is not None:
            payload['candidate_budget'] = args.candidate_budget
        if args.allow_animation:
            payload['allow_animation'] = True
        plan = StickerRetrievalPlan.from_payload(payload, config=catalog.config)
        state, persona = await catalog.aprepare_query_context(plan=plan, session_id=args.session_id, persist_persona=False)
        matches = await catalog.achoose(plan=plan, session_id=args.session_id, session_state=state, persona_context=persona)
        print(json.dumps({'catalog': catalog.stats(), 'candidates': [_candidate_payload(m) for m in matches]}, ensure_ascii=False, indent=2))
    finally:
        await embeddings.aclose()
        await store.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    query = parser.add_mutually_exclusive_group(required=True)
    query.add_argument('--intent-core', help='What the intended reply should convey')
    query.add_argument('--plan', help='JSON object using the sticker_query tool arguments')
    parser.add_argument('--source', type=Path, help='Original media root (default APP_DATA_DIR/stickers)')
    parser.add_argument('--session-id', default='cli', help='Read this chat persona and confirmed delivery history')
    parser.add_argument('--candidate-budget', type=int)
    parser.add_argument('--allow-animation', action='store_true')
    asyncio.run(run(parser.parse_args(argv)))


if __name__ == '__main__':
    main()
