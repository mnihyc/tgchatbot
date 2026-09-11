#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Build or rebuild the Tantivy lexical index from tantivy_docs.jsonl.')
    parser.add_argument('--repo-root', default='.', help='Source checkout used only with --build-from-source.')
    parser.add_argument('--build-from-source', action='store_true', help='Developer option: compile the retriever with locked Cargo dependencies.')
    parser.add_argument('--docs-jsonl', default='./data/tantivy_docs.jsonl')
    parser.add_argument('--index-dir', default='./data/tantivy_index')
    args = parser.parse_args()

    docs_jsonl = Path(args.docs_jsonl).expanduser().resolve()
    index_dir = Path(args.index_dir).expanduser().resolve()
    if not docs_jsonl.is_file():
        raise RuntimeError(f'Tantivy docs JSONL not found: {docs_jsonl}')
    if args.build_from_source:
        cargo = shutil.which('cargo')
        if cargo is None:
            raise RuntimeError('The developer --build-from-source option requires Cargo.')
        repo_root = Path(args.repo_root).expanduser().resolve()
        command = [cargo, 'run', '--locked', '--release', '--manifest-path', str(repo_root / 'retriever' / 'Cargo.toml'), '--']
    else:
        binary = shutil.which('sticker-retriever')
        if binary is None:
            raise RuntimeError('sticker-retriever is not on PATH. Run this tool in the release image, or use --build-from-source in a source checkout.')
        command = [binary]
    subprocess.run(command + ['build', '--docs-jsonl', str(docs_jsonl), '--index-dir', str(index_dir)], check=True)


if __name__ == '__main__':
    main()
