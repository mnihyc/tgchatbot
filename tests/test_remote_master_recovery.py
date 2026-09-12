"""An interrupted transport must not prevent the next import from using SSH."""
from __future__ import annotations

import asyncio
import errno
import os
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from tgchatbot.config import load_config
from tgchatbot.tools.remote_workspace import RemoteWorkspaceClient


class RemoteMasterRecovery(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        with patch.dict(os.environ, {
            'APP_DATA_DIR': str(self.root / 'd'),
            'APP_TEMP_DIR': str(self.root / 't'),
            'SSH_EXEC_HOST': 'remote.invalid',
        }, clear=True):
            self.remote = RemoteWorkspaceClient(load_config(require_telegram=False))
        # Keep the OS socket address short even in deeply nested checkouts.
        self.control_address = os.path.relpath(self.remote._control_path)
        self.listeners = []
        self.addCleanup(lambda: [listener.close() for listener in self.listeners])
        self.starts = 0

    def bind_control_socket(self, *, listening):
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listeners.append(listener)
        listener.bind(self.control_address)
        if listening:
            listener.listen()
        else:
            # Killing the owning container leaves this filesystem entry behind.
            listener.close()
        return listener

    async def ssh_process(self, *args, **kwargs):
        if '-O' in args:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as probe:
                try:
                    probe.connect(self.control_address)
                    code = 0
                except OSError:
                    code = 255
        else:
            self.starts += 1
            try:
                self.bind_control_socket(listening=True)
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise
                # A connection may start successfully without acquiring its
                # multiplexing socket. Readiness must still be established.
            code = 0
        return SimpleNamespace(returncode=code, wait=AsyncMock(return_value=code))

    async def test_restart_recovers_after_previous_container_left_a_dead_socket(self):
        self.bind_control_socket(listening=False)
        unrelated = self.remote._control_dir / 'unrelated.txt'
        unrelated.write_text('keep this file', encoding='utf-8')
        with patch('asyncio.create_subprocess_exec', side_effect=self.ssh_process):
            await self.remote.ensure_master()
            await self.remote.ensure_master()
        self.assertTrue(self.remote._master_started)
        self.assertEqual(self.starts, 1)
        self.assertEqual(unrelated.read_text(encoding='utf-8'), 'keep this file')

    async def test_live_master_is_reused_without_replacing_its_socket(self):
        self.bind_control_socket(listening=True)
        before = self.remote._control_path.stat().st_ino
        with patch('asyncio.create_subprocess_exec', side_effect=self.ssh_process):
            await self.remote.ensure_master()
        self.assertTrue(self.remote._master_started)
        self.assertEqual(self.starts, 0)
        self.assertEqual(self.remote._control_path.stat().st_ino, before)

    async def test_unexpected_regular_file_is_preserved_when_master_cannot_start(self):
        self.remote._control_path.write_text('not a transport socket', encoding='utf-8')
        self.remote._wait_for_master_ready = AsyncMock(return_value=False)
        with patch('asyncio.create_subprocess_exec', side_effect=self.ssh_process):
            with self.assertRaisesRegex(RuntimeError, 'without becoming ready'):
                await self.remote.ensure_master()
        self.assertFalse(self.remote._master_started)
        self.assertEqual(self.remote._control_path.read_text(encoding='utf-8'),
                         'not a transport socket')

    async def test_cancelled_probe_does_not_replace_an_unverified_control_socket(self):
        self.bind_control_socket(listening=True)
        before = self.remote._control_path.stat().st_ino
        self.remote._check_master_alive = AsyncMock(side_effect=asyncio.CancelledError())
        with patch('asyncio.create_subprocess_exec', new_callable=AsyncMock) as start:
            with self.assertRaises(asyncio.CancelledError):
                await self.remote.ensure_master()
        self.assertFalse(self.remote._master_started)
        self.assertEqual(self.remote._control_path.stat().st_ino, before)
        start.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
