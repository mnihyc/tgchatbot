#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Rebuild the Tantivy lexical index offline. Stop the retriever before replacing its index.')
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
    index_dir.parent.mkdir(parents=True, exist_ok=True)
    # A sibling stays on the same filesystem so directory promotion uses rename.
    # Build errors never touch the previous index, even after partial indexing.
    staging = Path(tempfile.mkdtemp(prefix=f'.{index_dir.name}.build-', dir=index_dir.parent))
    previous = None
    try:
        subprocess.run(command + ['build', '--docs-jsonl', str(docs_jsonl), '--index-dir', str(staging)], check=True)
        metadata = json.loads((staging / 'meta.json').read_text())
        if not isinstance(metadata, dict) or not isinstance(metadata.get('schema'), list) or not isinstance(metadata.get('segments'), list):
            raise RuntimeError('Retriever did not produce valid index metadata; the previous index is unchanged.')
        if index_dir.exists():
            previous = Path(tempfile.mkdtemp(prefix=f'.{index_dir.name}.previous-', dir=index_dir.parent))
            # rename replaces this empty reserved directory, without a name race.
            try:
                index_dir.rename(previous)
            except BaseException:
                previous.rmdir()
                raise
        try:
            staging.rename(index_dir)
        except BaseException:
            if previous is not None:
                previous.rename(index_dir)
            raise
        if previous is not None:
            shutil.rmtree(previous)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == '__main__':
    main()
