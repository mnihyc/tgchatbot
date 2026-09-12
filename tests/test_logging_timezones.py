"""Operator log presentation follows configuration while exchange audits stay UTC."""
from __future__ import annotations

from datetime import datetime, timezone
import io
import json
import logging
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from tgchatbot.logging_config import configure_logging, dump_llm_exchange


class LoggingTimezoneTests(unittest.TestCase):
    def setUp(self):
        self.root = logging.getLogger()
        self.previous_handlers, self.previous_level = list(self.root.handlers), self.root.level
        self.noisy_levels = {name: logging.getLogger(name).level for name in ('httpx', 'httpcore')}
        self.addCleanup(self.restore_logging)

    def restore_logging(self):
        for handler in self.root.handlers:
            if handler not in self.previous_handlers:
                handler.close()
        self.root.handlers = self.previous_handlers
        self.root.setLevel(self.previous_level)
        for name, level in self.noisy_levels.items():
            logging.getLogger(name).setLevel(level)

    def test_operator_log_uses_single_configured_timezone_across_calendar_and_dst_boundaries(self):
        cases = [('Asia/Singapore', '2026-09-06T16:24:08+00:00', '2026-09-07T00:24:08+08:00'),
                 ('America/New_York', '2026-01-01T02:24:08+00:00', '2025-12-31T21:24:08-05:00'),
                 ('America/New_York', '2026-07-01T02:24:08+00:00', '2026-06-30T22:24:08-04:00')]
        for zone, instant, expected in cases:
            with self.subTest(zone=zone, instant=instant):
                output = io.StringIO()
                with patch.dict(os.environ, {'DEFAULT_METADATA_TIMEZONE': zone, 'TZ': 'UTC'}), \
                        patch('sys.stderr', output):
                    configure_logging('INFO')
                    record = logging.LogRecord('test.operator', logging.INFO, __file__, 1,
                        'Processing ordinary input', (), None)
                    record.created = datetime.fromisoformat(instant).timestamp()
                    logging.getLogger('test.operator').handle(record)
                    self.assertEqual(os.environ['TZ'], 'UTC', 'Formatter must not change the process timezone')
                self.assertIn(f'[{expected}] INFO test.operator: Processing ordinary input', output.getvalue())
                self.assertEqual(record.created, datetime.fromisoformat(instant).timestamp())

    def test_debug_exchange_keeps_utc_audit_time_and_exact_provider_payload(self):
        with tempfile.TemporaryDirectory(prefix='log-audit-', dir=Path(__file__).parent) as temporary:
            output = io.StringIO()
            instant = datetime(2026, 9, 6, 16, 24, 8, tzinfo=timezone.utc)
            payload = {'contents': [{'text': 'Keep literal 2026-09-06T16:24:08Z unchanged.'}],
                       'provider_time': '2026-09-06T16:24:08Z'}
            response_body = {'timestamp': '2026-09-06T16:24:08Z', 'raw': 'Native evidence'}
            response = Mock(status_code=200, headers={'Date': 'Sun, 06 Sep 2026 16:24:08 GMT'})
            response.json.return_value = response_body
            with patch.dict(os.environ, {'DEFAULT_METADATA_TIMEZONE': 'Asia/Singapore', 'APP_DATA_DIR': temporary}), \
                    patch('sys.stderr', output):
                configure_logging('DEBUG')
                with patch('tgchatbot.logging_config.datetime', wraps=datetime) as clock:
                    clock.now.return_value = instant
                    saved = dump_llm_exchange(provider='fixture', model='fixture-model',
                        url='https://example.invalid/generate', payload=payload, response=response)
            record = json.loads(saved.read_text())
            self.assertEqual(record['timestamp'], '2026-09-06T16:24:08+00:00')
            self.assertIn('20260906T162408_000000Z', saved.name)
            self.assertEqual(record['request']['payload'], payload)
            self.assertEqual(record['response']['body'], response_body)
            self.assertEqual(record['response']['headers'], response.headers)
