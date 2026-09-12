import asyncio
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tgchatbot.healthcheck import healthy, start_heartbeat, stop_heartbeat


class HealthTests(unittest.IsolatedAsyncioTestCase):
    async def test_only_running_poller_reports_health_and_shutdown_invalidates(self):
        with TemporaryDirectory() as tmp, patch.dict(os.environ, {"APP_HEALTH_FILE": tmp + "/health.json"}):
            app = SimpleNamespace(running=False, updater=SimpleNamespace(running=False), bot_data={})
            with patch("tgchatbot.healthcheck.HEARTBEAT_INTERVAL_S", 0.001):
                await start_heartbeat(app)
                await asyncio.sleep(0.005)
                self.assertFalse(healthy())
                app.running = app.updater.running = True
                await asyncio.sleep(0.01)
                self.assertTrue(healthy())
                await stop_heartbeat(app)
                self.assertFalse(healthy())

    def test_missing_corrupt_stale_future_or_dead_process_is_unhealthy(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "health"
            self.assertFalse(healthy(path))
            for value in ["broken", "{}", json.dumps({"pid": os.getpid(), "time": 1}),
                          json.dumps({"pid": os.getpid(), "time": 1000})]:
                path.write_text(value)
                self.assertFalse(healthy(path, now=100))
            path.write_text(json.dumps({"pid": os.getpid(), "time": 99}))
            with patch("tgchatbot.healthcheck.os.kill", side_effect=ProcessLookupError):
                self.assertFalse(healthy(path, now=100))

    def test_configured_age_controls_readiness(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'health'
            path.write_text(json.dumps({'pid': os.getpid(), 'time': 100}))
            with patch.dict(os.environ, {'APP_HEALTH_MAX_AGE_S': '60'}):
                self.assertTrue(healthy(path, now=145))
                self.assertFalse(healthy(path, now=161))
