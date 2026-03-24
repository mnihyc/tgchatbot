#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Build or rebuild the Tantivy lexical index from tantivy_docs.jsonl.')
    parser.add_argument('--repo-root', default='.')
    parser.add_argument('--docs-jsonl', default='./data/tantivy_docs.jsonl')
    parser.add_argument('--index-dir', default='./data/tantivy_index')
    args = parser.parse_args()

    repo_root = Path(args.repo_root).expanduser().resolve()
    docs_jsonl = Path(args.docs_jsonl).expanduser().resolve()
    index_dir = Path(args.index_dir).expanduser().resolve()
    if shutil.which('cargo') is None:
        raise RuntimeError('cargo is required to build the Tantivy retriever sidecar.')
    if not docs_jsonl.exists():
        raise RuntimeError(f'Tantivy docs JSONL not found: {docs_jsonl}')
    subprocess.run(
        [
            'cargo', 'run', '--release', '--manifest-path', str(repo_root / 'retriever' / 'Cargo.toml'), '--',
            'build', '--docs-jsonl', str(docs_jsonl), '--index-dir', str(index_dir),
        ],
        check=True,
    )


if __name__ == '__main__':
    main()
