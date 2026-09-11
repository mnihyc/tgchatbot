from __future__ import annotations

import contextlib
import io
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest

from scripts.reset_sticker import main, reset_sticker


REPO = Path(__file__).resolve().parents[1]


class StickerMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory(prefix='fixture-sticker-reset-', dir=REPO / 'tests')
        self.addCleanup(self.fixture.cleanup)
        self.root = Path(self.fixture.name)
        self.data = self.root / 'data'
        self.data.mkdir()
        self.db = self.data / 'sticker_index.sqlite3'
        self.target = 'pack/target.webp'
        self.other = 'pack/other.webp'
        with sqlite3.connect(self.db) as connection:
            connection.executescript('''
                CREATE TABLE stickers (rowid INTEGER PRIMARY KEY, relative_path TEXT NOT NULL UNIQUE, summary TEXT NOT NULL);
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                INSERT INTO meta VALUES ('schema_version', 'fixture');
            ''')
            connection.executemany('INSERT INTO stickers VALUES (?, ?, ?)', [(1, self.target, 'target summary'), (2, self.other, 'other summary')])

    def rows(self):
        with sqlite3.connect(self.db) as connection:
            return connection.execute('SELECT rowid, relative_path, summary FROM stickers ORDER BY rowid').fetchall()

    def test_current_catalog_resets_only_target_preserving_source_and_derived_files(self):
        source = self.data / 'stickers' / self.target
        source.parent.mkdir(parents=True)
        source.write_bytes(b'original source sticker')
        derived = self.data / 'caption_embeddings.npy'
        derived.write_bytes(b'unchanged derived index fixture')
        before_source, before_derived = source.read_bytes(), derived.read_bytes()
        self.assertTrue(reset_sticker(self.target, index_db=self.db))
        self.assertEqual(self.rows(), [(2, self.other, 'other summary')])
        self.assertEqual(source.read_bytes(), before_source)
        self.assertEqual(derived.read_bytes(), before_derived)
        with sqlite3.connect(self.db) as connection:
            self.assertEqual(connection.execute('SELECT * FROM meta').fetchall(), [('schema_version', 'fixture')])

    def test_legacy_fts_removes_matching_rowid_without_affecting_other_search_rows(self):
        with sqlite3.connect(self.db) as connection:
            connection.execute('CREATE VIRTUAL TABLE sticker_fts USING fts5(summary)')
            connection.executemany('INSERT INTO sticker_fts(rowid, summary) VALUES (?, ?)', [(1, 'target summary'), (2, 'other summary')])
        self.assertTrue(reset_sticker(self.target, index_db=self.db))
        self.assertEqual(self.rows(), [(2, self.other, 'other summary')])
        with sqlite3.connect(self.db) as connection:
            self.assertEqual(connection.execute('SELECT rowid, summary FROM sticker_fts').fetchall(), [(2, 'other summary')])

    def test_quoted_path_is_an_exact_parameter_not_sql(self):
        unusual = "pack/O'Brien'; DELETE FROM stickers; --.webp"
        with sqlite3.connect(self.db) as connection:
            connection.execute('UPDATE stickers SET relative_path = ? WHERE rowid = 1', (unusual,))
        self.assertTrue(reset_sticker(unusual, index_db=self.db))
        self.assertEqual(self.rows(), [(2, self.other, 'other summary')])

    def test_missing_target_leaves_database_bytes_unchanged(self):
        before = self.db.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            result = main(['pack/missing.webp', '--index-db', str(self.db)])
        self.assertEqual(result, 1)
        self.assertIn('catalog unchanged', errors.getvalue())
        self.assertEqual(self.db.read_bytes(), before)
        self.assertEqual(len(self.rows()), 2)

    def test_missing_database_is_never_created(self):
        missing = self.data / 'misspelled.sqlite3'
        before = self.db.read_bytes()
        with contextlib.redirect_stderr(io.StringIO()) as errors:
            result = main([self.target, '--index-db', str(missing)])
        self.assertEqual(result, 1)
        self.assertIn('reset failed', errors.getvalue())
        self.assertFalse(missing.exists())
        self.assertEqual(self.db.read_bytes(), before)

    def test_fts_failure_rolls_back_trigger_side_effect_and_preserves_catalog(self):
        # A failing legacy FTS table models a broken catalog. RAISE(FAIL) keeps
        # the trigger's earlier UPDATE until the surrounding transaction rolls
        # back, making this check stronger than asserting an undeleted row alone.
        with sqlite3.connect(self.db) as connection:
            connection.executescript('''
                CREATE TABLE sticker_fts (rowid INTEGER PRIMARY KEY, summary TEXT);
                INSERT INTO sticker_fts VALUES (1, 'target summary'), (2, 'other summary');
                CREATE TRIGGER fail_fts_delete BEFORE DELETE ON sticker_fts BEGIN
                    UPDATE meta SET value = 'uncommitted change';
                    SELECT RAISE(FAIL, 'forced FTS failure');
                END;
            ''')
        before = self.rows()
        with self.assertRaisesRegex(sqlite3.IntegrityError, 'forced FTS failure'):
            reset_sticker(self.target, index_db=self.db)
        self.assertEqual(self.rows(), before)
        with sqlite3.connect(self.db) as connection:
            self.assertEqual(connection.execute('SELECT value FROM meta').fetchone()[0], 'fixture')
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM sticker_fts').fetchone()[0], 2)

    def test_catalog_failure_restores_already_deleted_fts_row(self):
        with sqlite3.connect(self.db) as connection:
            connection.executescript('''
                CREATE VIRTUAL TABLE sticker_fts USING fts5(summary);
                INSERT INTO sticker_fts(rowid, summary) VALUES (1, 'target summary'), (2, 'other summary');
                CREATE TRIGGER fail_catalog_delete BEFORE DELETE ON stickers BEGIN
                    SELECT RAISE(FAIL, 'forced catalog failure');
                END;
            ''')
        with self.assertRaisesRegex(sqlite3.IntegrityError, 'forced catalog failure'):
            reset_sticker(self.target, index_db=self.db)
        self.assertEqual(len(self.rows()), 2)
        with sqlite3.connect(self.db) as connection:
            self.assertEqual(connection.execute('SELECT rowid FROM sticker_fts ORDER BY rowid').fetchall(), [(1,), (2,)])

    def test_module_cli_uses_default_catalog_and_explains_required_rebuild(self):
        result = subprocess.run(
            [sys.executable, '-m', 'scripts.reset_sticker', self.target],
            cwd=self.root, env={'PYTHONPATH': str(REPO), 'PATH': os.environ.get('PATH', '')},
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Source files were preserved', result.stdout)
        self.assertIn('builder', result.stdout)
        self.assertIn('refresh derived indexes before restarting services', result.stdout)
        self.assertEqual(self.rows(), [(2, self.other, 'other summary')])
