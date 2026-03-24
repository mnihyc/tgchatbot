#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tgchatbot.stickers.catalog import StickerCatalog
from tgchatbot.stickers.plan import StickerRetrievalPlan


def main() -> None:
    parser = argparse.ArgumentParser(description='Query the current sticker system locally with one transparent sticker plan.')
    parser.add_argument('--stickers-dir', default='./data/stickers')
    parser.add_argument('--index-db', default='./data/sticker_index.sqlite3')
    parser.add_argument('--intent-core', required=True)
    parser.add_argument('--secondary-goal', action='append', default=[])
    parser.add_argument('--text-priority', default='prefer', choices=['require', 'prefer', 'ignore'])
    parser.add_argument('--style-goal', default='keep_current', choices=['keep_current', 'allow_switch', 'prefer_switch', 'ignore_style'])
    parser.add_argument('--style-policy', choices=['continue', 'neutral', 'prefer_switch', 'hard_switch'], help='Deprecated alias for --style-goal.')
    parser.add_argument('--style-hint', action='append', default=[])
    parser.add_argument('--prefer-pack', default='', help='Preferred source_pack_id or closely related pack family string.')
    parser.add_argument('--prefer-cluster', default='', help='Preferred style_cluster id or nearby cluster family string.')
    parser.add_argument('--max-harshness', type=int, default=3)
    parser.add_argument('--max-intimacy', type=int, default=4)
    parser.add_argument('--max-meme-dependence', type=int, default=4)
    parser.add_argument('--forbid', action='append', default=[])
    parser.add_argument('--allow-animation', action='store_true')
    parser.add_argument('--session-id', default='cli')
    parser.add_argument('--candidate-budget', type=int, default=5)
    parser.add_argument('--skip-send', action='store_true')
    args = parser.parse_args()

    retriever_url = os.getenv('STICKER_RETRIEVER_URL', 'http://127.0.0.1:4107')
    if not os.getenv('OPENAI_API_KEY'):
        raise RuntimeError('OPENAI_API_KEY is required for live query embeddings.')

    os.environ.setdefault('STICKER_RETRIEVER_URL', retriever_url)
    catalog = StickerCatalog(Path(args.index_db).expanduser().resolve(), Path(args.stickers_dir).expanduser().resolve())
    catalog.load()
    plan_payload = {
        'send': not args.skip_send,
        'intent_core': args.intent_core,
        'secondary_goals': args.secondary_goal,
        'text_constraints': {'text_priority': args.text_priority},
        'style_focus': {
            'style_goal': args.style_goal,
            'style_hints': args.style_hint,
            'prefer_pack': args.prefer_pack,
            'prefer_cluster': args.prefer_cluster,
        },
        'intensity_limits': {
            'max_harshness': args.max_harshness,
            'max_intimacy': args.max_intimacy,
            'max_meme_dependence': args.max_meme_dependence,
            'allow_animation': args.allow_animation,
        },
        'forbid': args.forbid,
        'candidate_budget': args.candidate_budget,
    }
    if args.style_policy:
        plan_payload['style_policy'] = args.style_policy
    plan = StickerRetrievalPlan.from_payload(plan_payload)
    matches = catalog.choose(plan=plan, session_id=args.session_id)
    print('query_interpretation:', plan.query_interpretation())
    print('style_context:', catalog.describe_style_context(args.session_id))
    for match in matches:
        print(f'{match.score:7.3f}  {match.entry.relative_path} :: {match.entry.preview_text}')
        print(f'         style={match.entry.style_cluster} pack={match.entry.source_pack_id} semsig={match.entry.semantic_signature}')
        print('         summary:', match.selection_summary)
        if match.entry.source_overlay_text:
            print(f'         overlay={match.entry.source_overlay_text}')
        if match.matched_terms:
            print('         matched:', ', '.join(match.matched_terms[:12]))
        if match.score_breakdown:
            print('         score_breakdown:', ', '.join(f'{k}={v:.2f}' for k, v in sorted(match.score_breakdown.items(), key=lambda kv: -kv[1])))
        if match.match_profile.get('hard_mismatches'):
            print('         hard_mismatches:', '; '.join(match.match_profile['hard_mismatches']))


if __name__ == '__main__':
    main()
