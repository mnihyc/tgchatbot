from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from scripts.rebuild_tantivy_index import main


class RebuildTantivyToolTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.docs = self.root / 'documents.jsonl'
        self.docs.write_text('{"sticker_id":"fixture"}\n')
        self.index = self.root / 'index'
        self.arguments = ['rebuild_tantivy_index.py', '--docs-jsonl', str(self.docs), '--index-dir', str(self.index)]

    def test_release_binary_runs_directly_without_cargo(self):
        with patch('sys.argv', self.arguments), \
             patch('scripts.rebuild_tantivy_index.shutil.which', return_value='/usr/local/bin/sticker-retriever') as which, \
             patch('scripts.rebuild_tantivy_index.subprocess.run') as run:
            main()
        which.assert_called_once_with('sticker-retriever')
        run.assert_called_once_with(['/usr/local/bin/sticker-retriever', 'build', '--docs-jsonl', str(self.docs), '--index-dir', str(self.index)], check=True)
        self.assertFalse(self.index.exists())

    def test_missing_documents_never_launch_or_build_anything(self):
        for missing in (self.root / 'missing.jsonl', self.root):
            arguments = ['rebuild_tantivy_index.py', '--docs-jsonl', str(missing), '--build-from-source']
            with self.subTest(missing=missing.name), patch('sys.argv', arguments), \
                 patch('scripts.rebuild_tantivy_index.shutil.which') as which, \
                 patch('scripts.rebuild_tantivy_index.subprocess.run') as run:
                with self.assertRaisesRegex(RuntimeError, 'docs JSONL not found'):
                    main()
                which.assert_not_called()
                run.assert_not_called()

    def test_missing_release_binary_does_not_silently_compile(self):
        with patch('sys.argv', self.arguments), \
             patch('scripts.rebuild_tantivy_index.shutil.which', return_value=None) as which, \
             patch('scripts.rebuild_tantivy_index.subprocess.run') as run:
            with self.assertRaisesRegex(RuntimeError, 'release image'):
                main()
        which.assert_called_once_with('sticker-retriever')
        run.assert_not_called()

    def test_explicit_developer_build_uses_locked_source_checkout(self):
        source = self.root / 'source'
        arguments = self.arguments + ['--build-from-source', '--repo-root', str(source)]
        with patch('sys.argv', arguments), \
             patch('scripts.rebuild_tantivy_index.shutil.which', return_value='/toolchain/cargo') as which, \
             patch('scripts.rebuild_tantivy_index.subprocess.run') as run:
            main()
        which.assert_called_once_with('cargo')
        run.assert_called_once_with(['/toolchain/cargo', 'run', '--locked', '--release', '--manifest-path', str(source / 'retriever' / 'Cargo.toml'), '--', 'build', '--docs-jsonl', str(self.docs), '--index-dir', str(self.index)], check=True)
