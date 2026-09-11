"""Sticker maintenance outcomes with real image/SQLite work and fake OCR/model services."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

from scripts import build_sticker_index as builder
from scripts import query_sticker_index as query
from tgchatbot.stickers.retrieval_client import LexicalHit


class StickerBuildWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture = tempfile.TemporaryDirectory(prefix='fixture-sticker-build-', dir=Path(__file__).parent)
        self.addCleanup(self.fixture.cleanup)
        self.root = Path(self.fixture.name)
        self.data = self.root / 'data'
        self.stickers = self.data / 'stickers'
        self.stickers.mkdir(parents=True)
        self.db = self.data / 'sticker_index.sqlite3'
        # Two small real images are enough to exercise partial completion and
        # resume without installing OCR weights or making model requests.
        for name, color in [('01-complete.png', 'green'), ('02-retry.png', 'blue')]:
            with Image.new('RGB', (16, 16), color) as image:
                image.save(self.stickers / name)
        self.fail_path = '02-retry.png'
        self.analyzed = []
        self.semantics = {
            'summary': 'A friendly greeting', 'preview_text': 'Hello!', 'emoji': '👋',
            'selection_notes': 'Use to greet someone',
            'caption_card': {'caption_mode': 'visual_dominant', 'caption_meaning_en': 'Hello', 'harshness_level': 0, 'intimacy_level': 0, 'meme_dependence_level': 0},
            'subtle_cue_card': {},
            'sticker_card': {'fused_pragmatic_meaning': 'A friendly greeting'},
            'style_card': {'style_rendering_type': 'simple illustration'},
        }

    def run_build(self):
        test = self

        class FakeOCR:
            available = True

            def __init__(self, **kwargs):
                pass

            def extract(self, image):
                return {'lines': [], 'joined_text': '', 'confidence': 0.0, 'coverage_ratio': 0.0}

        class FakeAnalyzer:
            def __init__(self, **kwargs):
                pass

            def analyze(self, *, relative_path, **kwargs):
                test.analyzed.append(relative_path)
                if relative_path == test.fail_path:
                    raise RuntimeError('simulated sticker analysis outage')
                return copy.deepcopy(test.semantics)

            def close(self):
                pass

        with ExitStack() as stack:
            stack.enter_context(patch.dict(os.environ, {'APP_DATA_DIR': str(self.data), 'GEMINI_API_KEY': 'synthetic-key'}, clear=True))
            stack.enter_context(patch('sys.argv', ['build_sticker_index', '--stickers-dir', str(self.stickers), '--index-db', str(self.db), '--workers', '1', '--no-embeddings']))
            # Only external model/OCR initialization and local .env discovery are
            # substituted; source scans, row commits, resume and exports are real.
            stack.enter_context(patch.object(builder, 'load_dotenv', return_value=False))
            stack.enter_context(patch.object(builder, 'PaddleOCR', object()))
            stack.enter_context(patch.object(builder, 'PaddleOCRExtractor', FakeOCR))
            stack.enter_context(patch.object(builder, 'StickerAnalysisClient', FakeAnalyzer))
            for name in ('_WORKER_CONFIG', '_WORKER_OCR', '_WORKER_LLM'):
                stack.enter_context(patch.object(builder, name, None))
            output = stack.enter_context(redirect_stdout(io.StringIO()))
            builder.main()
            return output.getvalue()

    def catalog_rows(self):
        with sqlite3.connect(self.db) as connection:
            return connection.execute('SELECT relative_path, summary FROM stickers ORDER BY relative_path').fetchall()

    def test_failed_analysis_stops_job_but_completed_work_resumes_without_reanalysis(self):
        docs = self.data / 'tantivy_docs.jsonl'
        docs.write_text('previous published index\n')
        with self.assertRaisesRegex(RuntimeError, 'failed for 1 sticker'):
            self.run_build()
        self.assertEqual(self.catalog_rows(), [('01-complete.png', 'A friendly greeting')])
        self.assertEqual(docs.read_text(), 'previous published index\n')
        failures = [json.loads(line) for line in (self.data / 'build_failures.jsonl').read_text().splitlines()]
        self.assertEqual([failure['relative_path'] for failure in failures], ['02-retry.png'])

        self.fail_path = None
        self.analyzed.clear()
        self.run_build()
        self.assertEqual(self.analyzed, ['02-retry.png'])
        self.assertEqual(self.catalog_rows(), [('01-complete.png', 'A friendly greeting'), ('02-retry.png', 'A friendly greeting')])
        published = [json.loads(line)['relative_path'] for line in docs.read_text().splitlines()]
        self.assertEqual(published, ['01-complete.png', '02-retry.png'])

    def test_existing_lexical_catalog_can_be_queried_without_any_model_key(self):
        self.fail_path = None
        self.run_build()
        with sqlite3.connect(self.db) as connection:
            sticker_id = connection.execute('SELECT sticker_id FROM stickers WHERE relative_path = ?', ('01-complete.png',)).fetchone()[0]

        class FakeRetriever:
            def __init__(self, base_url):
                pass

            def ensure_healthy(self, **kwargs):
                pass

            def search(self, payload):
                return [LexicalHit(sticker_id, 1.0, ['caption'], {'caption_lexical': 1.0, 'sticker_lexical': 1.0})]

            def close(self):
                pass

        with patch.dict(os.environ, {'STICKER_SEMANTIC_MODE': 'off'}, clear=True), \
             patch('tgchatbot.stickers.catalog.TantivyRetrieverClient', FakeRetriever), \
             patch('sys.argv', ['query_sticker_index', '--stickers-dir', str(self.stickers), '--index-db', str(self.db), '--intent-core', 'friendly greeting']), \
             redirect_stdout(io.StringIO()) as output:
            query.main()
        self.assertIn('01-complete.png :: Hello!', output.getvalue())
        self.assertEqual(len(self.catalog_rows()), 2)
