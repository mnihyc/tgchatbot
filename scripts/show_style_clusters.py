#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9_:+-]+", re.IGNORECASE)
_STYLE_FIELD_LABELS: list[tuple[str, str]] = [
    ('style_rendering_type', 'render'),
    ('style_palette_family', 'palette'),
    ('style_character_family', 'character'),
    ('line_weight', 'line'),
    ('meme_intensity', 'meme'),
    ('style_text_prominence', 'text'),
]


@dataclass(slots=True)
class ClusterMember:
    relative_path: str
    source_pack_id: str | None
    style_text: str
    style_card: dict[str, Any]


@dataclass(slots=True)
class ClusterSummary:
    style_cluster: str
    count: int
    field_hints: dict[str, str]
    pack_hint: str
    top_terms: list[str]
    members: list[ClusterMember]


def _norm_text(value: Any) -> str:
    return ' '.join(str(value or '').replace('\n', ' ').replace('\t', ' ').split()).strip()


def _tokenize_style_value(value: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(value)]


def _load_clusters(index_db: Path) -> list[ClusterSummary]:
    con = sqlite3.connect(index_db)
    con.row_factory = sqlite3.Row
    try:
        grouped: dict[str, list[ClusterMember]] = defaultdict(list)
        rows = con.execute(
            '''
            select relative_path, source_pack_id, style_text, style_card_json, style_cluster
            from stickers
            where coalesce(style_cluster, '') != ''
            order by style_cluster, relative_path
            '''
        ).fetchall()
    finally:
        con.close()

    for row in rows:
        try:
            style_card = json.loads(row['style_card_json'] or '{}')
            if not isinstance(style_card, dict):
                style_card = {}
        except Exception:
            style_card = {}
        grouped[str(row['style_cluster'])].append(
            ClusterMember(
                relative_path=str(row['relative_path']),
                source_pack_id=(str(row['source_pack_id']).strip() or None) if row['source_pack_id'] is not None else None,
                style_text=str(row['style_text'] or ''),
                style_card=style_card,
            )
        )

    summaries: list[ClusterSummary] = []
    for style_cluster, members in grouped.items():
        field_hints: dict[str, str] = {}
        for field_name, label in _STYLE_FIELD_LABELS:
            counter = Counter(
                _norm_text(member.style_card.get(field_name, ''))
                for member in members
                if _norm_text(member.style_card.get(field_name, ''))
            )
            field_hints[label] = counter.most_common(1)[0][0] if counter else '-'

        pack_counter = Counter(member.source_pack_id for member in members if member.source_pack_id)
        pack_hint = pack_counter.most_common(1)[0][0] if pack_counter else '-'

        token_counter: Counter[str] = Counter()
        for member in members:
            style_values = [
                member.style_text,
                member.style_card.get('style_rendering_type', ''),
                member.style_card.get('line_weight', ''),
                member.style_card.get('style_palette_family', ''),
                member.style_card.get('meme_intensity', ''),
                member.style_card.get('style_text_prominence', ''),
                member.style_card.get('style_character_family', ''),
            ]
            for style_value in style_values:
                for token in _tokenize_style_value(_norm_text(style_value)):
                    token_counter[token] += 1

        top_terms = [token for token, _ in token_counter.most_common(12)]
        summaries.append(
            ClusterSummary(
                style_cluster=style_cluster,
                count=len(members),
                field_hints=field_hints,
                pack_hint=pack_hint,
                top_terms=top_terms,
                members=members,
            )
        )

    summaries.sort(key=lambda item: (-item.count, item.style_cluster))
    return summaries


def _filter_clusters(
    clusters: list[ClusterSummary],
    *,
    selected: set[str],
    max_clusters: int,
    min_cluster_size: int,
) -> list[ClusterSummary]:
    filtered = [cluster for cluster in clusters if cluster.count >= min_cluster_size]
    if selected:
        filtered = [cluster for cluster in filtered if cluster.style_cluster in selected]
    if max_clusters > 0:
        filtered = filtered[:max_clusters]
    return filtered


def _cluster_hint_text(cluster: ClusterSummary) -> str:
    field_text = ' | '.join(
        f'{label}={cluster.field_hints.get(label, "-")}'
        for _, label in _STYLE_FIELD_LABELS
    )
    terms = ', '.join(cluster.top_terms[:10]) if cluster.top_terms else '-'
    return f'{field_text}\npack={cluster.pack_hint}\nterms={terms}'


def _print_clusters(clusters: list[ClusterSummary], *, examples_per_cluster: int) -> None:
    for cluster in clusters:
        print(f'{cluster.style_cluster}  count={cluster.count}')
        print(f'  {_cluster_hint_text(cluster)}')
        for member in cluster.members[:examples_per_cluster]:
            print(f'  - {member.relative_path}')
        print()


def _render_clusters(
    clusters: list[ClusterSummary],
    *,
    stickers_dir: Path,
    examples_per_cluster: int,
    thumb_size: int,
) -> None:
    try:
        import tkinter as tk
        from tkinter import ttk
    except Exception as exc:
        raise RuntimeError('Tkinter is required for GUI display. Use --print-only in headless environments.') from exc

    try:
        from PIL import Image, ImageOps, ImageTk, UnidentifiedImageError
    except Exception as exc:
        raise RuntimeError('Pillow is required for image display.') from exc

    def fit_thumbnail(image: Image.Image) -> Image.Image:
        contained = ImageOps.contain(image.convert('RGBA'), (thumb_size, thumb_size))
        canvas = Image.new('RGBA', (thumb_size, thumb_size), (246, 246, 246, 255))
        offset = ((thumb_size - contained.width) // 2, (thumb_size - contained.height) // 2)
        canvas.alpha_composite(contained, offset)
        return canvas.convert('RGB')

    def load_thumbnail(path: Path) -> Image.Image | None:
        if not path.exists():
            return None
        suffix = path.suffix.lower()
        if suffix in {'.webm', '.mp4'}:
            try:
                import av
            except Exception:
                return None
            try:
                with av.open(str(path)) as container:
                    frames = container.decode(video=0)
                    chosen_frame = None
                    for index, frame in enumerate(frames, start=1):
                        if random.randrange(index) == 0:
                            chosen_frame = frame
                if chosen_frame is None:
                    return None
                return fit_thumbnail(chosen_frame.to_image())
            except Exception:
                return None
        try:
            with Image.open(path) as image:
                image.seek(0)
                converted = image.convert('RGBA')
        except (OSError, UnidentifiedImageError):
            return None
        return fit_thumbnail(converted)

    root = tk.Tk()
    root.title('Sticker Style Clusters')

    outer = ttk.Frame(root, padding=8)
    outer.pack(fill='both', expand=True)

    canvas = tk.Canvas(outer, highlightthickness=0)
    scrollbar = ttk.Scrollbar(outer, orient='vertical', command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side='right', fill='y')
    canvas.pack(side='left', fill='both', expand=True)

    content = ttk.Frame(canvas)
    window_id = canvas.create_window((0, 0), window=content, anchor='nw')

    photo_refs: list[Any] = []

    def on_frame_configure(_event: Any) -> None:
        canvas.configure(scrollregion=canvas.bbox('all'))

    def on_canvas_configure(event: Any) -> None:
        canvas.itemconfigure(window_id, width=event.width)

    content.bind('<Configure>', on_frame_configure)
    canvas.bind('<Configure>', on_canvas_configure)

    for cluster_index, cluster in enumerate(clusters):
        group = ttk.LabelFrame(content, text=f'{cluster.style_cluster}  ({cluster.count} stickers)', padding=8)
        group.grid(row=cluster_index, column=0, sticky='ew', padx=4, pady=6)
        group.columnconfigure(0, weight=1)

        ttk.Label(
            group,
            text=_cluster_hint_text(cluster),
            justify='left',
            wraplength=max(640, thumb_size * 7),
        ).grid(row=0, column=0, sticky='w', pady=(0, 8))

        examples = ttk.Frame(group)
        examples.grid(row=1, column=0, sticky='w')

        for index, member in enumerate(cluster.members[:examples_per_cluster]):
            card = ttk.Frame(examples, padding=4)
            card.grid(row=index // 5, column=index % 5, sticky='n', padx=4, pady=4)

            image_path = stickers_dir / member.relative_path
            thumb = load_thumbnail(image_path)
            if thumb is not None:
                photo = ImageTk.PhotoImage(thumb)
                photo_refs.append(photo)
                image_label = ttk.Label(card, image=photo)
            else:
                image_label = tk.Label(card, text='preview\nunavailable', width=14, height=7, bg='#f2f2f2', relief='solid')
            image_label.pack()

            ttk.Label(
                card,
                text=Path(member.relative_path).name,
                justify='center',
                wraplength=thumb_size + 24,
            ).pack(pady=(4, 0))

    root.bind('<Escape>', lambda _event: root.destroy())
    root._photo_refs = photo_refs
    root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Show sticker style clusters with cluster-level style hints and example stickers, without generating temp output files.'
    )
    parser.add_argument('--index-db', default='./data/sticker_index.sqlite3')
    parser.add_argument('--stickers-dir', default='./data/stickers')
    parser.add_argument('--cluster', action='append', default=[], help='Specific style_cluster name to show. Repeatable.')
    parser.add_argument('--max-clusters', type=int, default=0, help='Maximum number of clusters to show. 0 means all.')
    parser.add_argument('--min-cluster-size', type=int, default=1, help='Hide clusters smaller than this size.')
    parser.add_argument('--examples-per-cluster', type=int, default=20)
    parser.add_argument('--thumb-size', type=int, default=120)
    parser.add_argument('--print-only', action='store_true', help='Print cluster summaries instead of opening a GUI window.')
    args = parser.parse_args()

    index_db = Path(args.index_db).expanduser().resolve()
    stickers_dir = Path(args.stickers_dir).expanduser().resolve()
    if not index_db.exists():
        raise RuntimeError(f'Sticker index database not found: {index_db}')
    if not stickers_dir.exists():
        raise RuntimeError(f'Sticker directory not found: {stickers_dir}')

    clusters = _load_clusters(index_db)
    clusters = _filter_clusters(
        clusters,
        selected={value.strip() for value in args.cluster if value.strip()},
        max_clusters=max(0, args.max_clusters),
        min_cluster_size=max(1, args.min_cluster_size),
    )
    if not clusters:
        raise RuntimeError('No clusters matched the requested filters.')

    if args.print_only:
        _print_clusters(clusters, examples_per_cluster=max(1, args.examples_per_cluster))
        return

    _render_clusters(
        clusters,
        stickers_dir=stickers_dir,
        examples_per_cluster=max(1, args.examples_per_cluster),
        thumb_size=max(48, args.thumb_size),
    )


if __name__ == '__main__':
    main()
