"""Remote originals, disposable transfers and Telegram receipts have separate owners."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from telegram.error import BadRequest, TimedOut

from tgchatbot.config import load_config
from tgchatbot.domain.models import OutboundArtifact
from tgchatbot.tools.base import ToolContext
from tgchatbot.tools.file_send import FileSendTool
from tgchatbot.tools.remote_workspace import RemoteSessionPaths, RemoteWorkspaceClient
from tgchatbot.transports.artifact_delivery import deliver_artifact


class RemoteTransferWorkflows(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='fixture-transfer-', dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        with patch.dict(os.environ, {'APP_DATA_DIR': str(self.root / 'data'),
                'APP_TEMP_DIR': str(self.root / 'tmp'), 'SSH_EXEC_HOST': 'remote.invalid'}, clear=True):
            self.config = load_config(require_telegram=False)
        self.remote = RemoteWorkspaceClient(self.config)
        self.paths = self.remote.session_paths('telegram:1')
        self.remote.ensure_session_dirs = AsyncMock(return_value=self.paths)

    async def test_same_name_files_and_repeated_fetches_keep_distinct_original_bytes(self):
        originals = {self.paths.outputs + '/a/report.txt': b'first report',
                     self.paths.outputs + '/b/report.txt': b'second report'}
        self.remote._resolve_remote_paths = AsyncMock(side_effect=lambda paths, selected: selected)

        async def scp(*arguments, **kwargs):
            remote_path = arguments[-2].split(':', 1)[1]
            Path(arguments[-1]).write_bytes(originals[remote_path])
            return SimpleNamespace(returncode=0, communicate=AsyncMock(return_value=(b'', b'')))

        with patch('asyncio.create_subprocess_exec', side_effect=scp):
            first = await self.remote.fetch_files(session_id='telegram:1', remote_paths=list(originals))
            second = await self.remote.fetch_files(session_id='telegram:1', remote_paths=[next(iter(originals))])
        self.assertEqual([item.path.read_bytes() for item in first + second],
                         [b'first report', b'second report', b'first report'])
        self.assertEqual(len({item.path for item in first + second}), 3)
        self.assertTrue(all(item.filename == 'report.txt' for item in first + second))
        for item in first + second:
            item.discard()
        self.assertFalse(list(self.config.artifact_dir.rglob('fetch-*')))

    async def test_interrupted_fetch_releases_partial_copies_and_subprocess(self):
        self.remote._resolve_remote_paths = AsyncMock(side_effect=lambda paths, selected: selected)
        process = SimpleNamespace(returncode=None, kill=lambda: None,
            communicate=AsyncMock(side_effect=[asyncio.CancelledError(), (b'', b'')]))
        process.kill = unittest.mock.Mock()
        with patch('asyncio.create_subprocess_exec', AsyncMock(return_value=process)):
            with self.assertRaises(asyncio.CancelledError):
                await self.remote.fetch_files(session_id='telegram:1', remote_paths=['report.txt'])
        process.kill.assert_called_once()
        self.assertFalse(list(self.config.artifact_dir.rglob('fetch-*')))

    def process_workspace(self):
        root = self.root / 'workspace'
        paths = RemoteSessionPaths(str(root), str(root / 'inputs'), str(root / 'outputs'))
        Path(paths.inputs).mkdir(parents=True)
        Path(paths.outputs).mkdir()
        self.remote.ensure_session_dirs.return_value = paths
        self.remote.ensure_master = AsyncMock()
        self.remote._ssh_base_args = lambda: ['sh', '-c']
        self.remote._scp_base_args = lambda: [sys.executable, '-c',
            'import shutil,sys; shutil.copyfile(sys.argv[1].split(":",1)[1],sys.argv[2])']
        return paths

    async def test_file_send_rejects_symlink_to_a_different_workspace_before_fetching_bytes(self):
        paths = self.process_workspace()
        outside = self.root / 'other-session.txt'
        outside.write_bytes(b'Unrelated session document')
        Path(paths.outputs, 'selected.txt').symlink_to(outside)
        transfer = unittest.mock.Mock(wraps=self.remote._scp_base_args)
        self.remote._scp_base_args = transfer
        result = await FileSendTool(self.config, self.remote).run(
            {'scope': 'outputs', 'paths': ['selected.txt']}, ToolContext('telegram:1', 'Participant'))
        try:
            self.assertFalse(result.output['ok'], 'File selection must retain its session-workspace ownership')
            self.assertEqual(result.artifacts, [])
            transfer.assert_not_called()
            self.assertEqual(outside.read_bytes(), b'Unrelated session document')
            self.assertFalse(list(self.config.artifact_dir.rglob('fetch-*')))
        finally:
            for artifact in result.artifacts:
                artifact.discard()

    async def test_file_send_follows_an_in_workspace_alias_and_retains_the_requested_name(self):
        paths = self.process_workspace()
        # Shell display truncation does not own the internal path-selection protocol.
        self.remote.ssh = replace(self.remote.ssh, max_stdout_chars=1)
        original = Path(paths.inputs, 'source.txt')
        original.write_bytes(b'Selected document')
        Path(paths.outputs, 'chosen.txt').symlink_to(original)
        result = await FileSendTool(self.config, self.remote).run(
            {'scope': 'outputs', 'paths': ['chosen.txt']}, ToolContext('telegram:1', 'Participant'))
        try:
            self.assertTrue(result.output['ok'], result.output)
            self.assertEqual(result.output['prepared_files'], ['chosen.txt'])
            self.assertEqual([artifact.path.read_bytes() for artifact in result.artifacts], [b'Selected document'])
            self.assertEqual(original.read_bytes(), b'Selected document')
        finally:
            for artifact in result.artifacts:
                artifact.discard()

    async def test_large_input_inventory_preserves_upload_and_rotation_receipts(self):
        paths = self.process_workspace()
        self.remote.ssh = replace(self.remote.ssh, max_stdout_chars=16000, max_input_files=180)
        originals = []
        for number in range(180):
            original = Path(paths.inputs, f'{number:03d}-' + '历史附件' * 12 + '.txt')
            original.write_bytes(b'previous attachment')
            os.utime(original, ns=(number + 1, number + 1))
            originals.append(str(original))
        incoming = self.root / '新的附件.txt'
        incoming.write_bytes(b'new attachment')
        expected = str(Path(paths.inputs, incoming.name))
        # The retained files fit the operator's inventory policy, while their
        # JSON receipt exceeds the independent shell-display allowance.
        self.assertGreater(len(json.dumps({'kept': originals[1:] + [expected],
            'rotated': originals[:1]}, ensure_ascii=False)), self.remote.ssh.max_stdout_chars)
        self.remote._scp_base_args = unittest.mock.Mock(return_value=[sys.executable, '-c',
            'import shutil,sys; shutil.copy(sys.argv[1],sys.argv[2].split(":",1)[1])'])
        result = await self.remote.sync_inputs('telegram:1', [incoming])
        self.assertEqual(result.kept_paths, [expected])
        self.assertEqual(result.rotated_paths, originals[:1])
        self.assertEqual(Path(expected).read_bytes(), incoming.read_bytes())
        self.assertFalse(Path(originals[0]).exists())
        self.assertTrue(all(Path(path).is_file() for path in originals[1:]))
        repeated = await self.remote.sync_inputs('telegram:1', [incoming])
        self.assertEqual(repeated.kept_paths, [expected])
        self.assertEqual(repeated.rotated_paths, [])
        self.remote._scp_base_args.assert_called_once()
        displayed = await self.remote.run_shell(session_id='telegram:1',
            command='python3 -c "print(chr(22909) * 20000)"', timeout_s=10)
        self.assertEqual(displayed['stdout'], '好' * self.remote.ssh.max_stdout_chars)

    async def test_file_send_skips_a_missing_file_without_losing_available_files(self):
        paths = self.process_workspace()
        original = Path(paths.outputs, 'available.txt')
        original.write_bytes(b'Available requested output')
        result = await FileSendTool(self.config, self.remote).run(
            {'scope': 'outputs', 'paths': ['missing.txt', 'available.txt']},
            ToolContext('telegram:1', 'Participant'))
        try:
            self.assertTrue(result.output['ok'], result.output)
            self.assertEqual(result.output['requested_paths'], 2)
            self.assertEqual(result.output['prepared_files'], ['available.txt'])
            self.assertEqual([artifact.path.read_bytes() for artifact in result.artifacts], [original.read_bytes()])
            self.assertEqual(list(self.config.artifact_dir.rglob('fetch-*')), [result.artifacts[0].path])
        finally:
            for artifact in result.artifacts:
                artifact.discard()
        self.assertFalse(list(self.config.artifact_dir.rglob('fetch-*')))

    async def test_file_inventory_respects_file_selection_independently_of_shell_display(self):
        paths = self.process_workspace()
        self.remote.ssh = replace(self.remote.ssh, max_stdout_chars=1)
        for name in ('一份报告.txt', '另一份报告.txt'):
            Path(paths.outputs, name).write_bytes(b'report')
        selected = await self.remote.list_files(session_id='telegram:1', max_files=1)
        self.assertEqual(selected, [{'name': '一份报告.txt', 'size_bytes': 6,
                                    'path': str(Path(paths.outputs, '一份报告.txt'))}])
        self.assertEqual(len(list(Path(paths.outputs).iterdir())), 2)

    async def test_python_source_with_shell_delimiter_executes_unchanged(self):
        paths = RemoteSessionPaths(str(self.root), str(self.root / 'inputs'), str(self.root / 'outputs'))
        self.remote.ensure_session_dirs.return_value = paths

        async def local_process(command, *, timeout_s):
            # Execute the real transport's generated shell locally, in this
            # isolated fixture. No SSH process or arbitrary input is involved.
            process = await asyncio.create_subprocess_exec('sh', '-c', command,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            stdout, stderr = await process.communicate()
            return {'ok': process.returncode == 0, 'returncode': process.returncode,
                    'stdout': stdout.decode(), 'stderr': stderr.decode()}

        self.remote._run_ssh_command = local_process
        code = "text = '''first\nPYCODE\nlast'''\nprint(text)"
        result = await self.remote.run_python(session_id='telegram:1', code=code, timeout_s=10)
        self.assertTrue(result['ok'], result['stderr'])
        self.assertEqual(result['stdout'], 'first\nPYCODE\nlast\n')
        self.assertEqual((self.root / 'run.py').read_text(), code)

    async def test_delivery_receipt_follows_ack_and_only_transfer_copy_is_deleted(self):
        original = self.root / 'original.txt'
        transfer = self.root / 'transfer.txt'
        original.write_bytes(b'report')
        transfer.write_bytes(original.read_bytes())
        bot = SimpleNamespace(send_document=AsyncMock(return_value=SimpleNamespace(message_id=42)))
        receipt = await deliver_artifact(bot, chat_id=1,
            artifact=OutboundArtifact(transfer, 'report.txt', temporary=True))
        self.assertTrue(receipt['sent'])
        self.assertEqual(receipt['telegram_message_id'], 42)
        self.assertFalse(transfer.exists())
        bot.send_document.side_effect = BadRequest('delivery rejected')
        with self.assertLogs('tgchatbot.transports.artifact_delivery', level='ERROR'):
            receipt = await deliver_artifact(bot, chat_id=1, artifact=OutboundArtifact(original, 'report.txt'))
        self.assertFalse(receipt['sent'])
        self.assertEqual(receipt['delivery_state'], 'failed')
        self.assertNotIn('telegram_message_id', receipt)
        self.assertEqual(original.read_bytes(), b'report')
        bot.send_document.side_effect = TimedOut()
        with self.assertLogs('tgchatbot.transports.artifact_delivery', level='ERROR'):
            uncertain = await deliver_artifact(bot, chat_id=1, artifact=OutboundArtifact(original, 'report.txt'))
        self.assertEqual(uncertain['delivery_state'], 'unknown')
        self.assertEqual(original.read_bytes(), b'report')

    async def test_verbose_execution_drains_both_pipes_and_keeps_configured_unicode_prefixes(self):
        self.remote.ssh = replace(self.remote.ssh, max_stdout_chars=7, max_stderr_chars=5)
        self.remote.ensure_master = AsyncMock()
        spawn = asyncio.create_subprocess_exec
        async def local_process(*args, **kwargs):
            return await spawn(sys.executable, '-c',
                "import sys; sys.stdout.write('好' * 1000000); sys.stderr.write('界' * 1000000)", **kwargs)
        with patch('asyncio.create_subprocess_exec', side_effect=local_process):
            result = await self.remote._run_ssh_command('fixture', timeout_s=10)
        self.assertEqual(result, {'ok': True, 'returncode': 0, 'stdout': '好' * 7, 'stderr': '界' * 5})

    async def test_cancelled_execution_reaps_local_process(self):
        self.remote.ensure_master = AsyncMock()
        spawn = asyncio.create_subprocess_exec
        ready = asyncio.Event()
        processes = []
        async def local_process(*args, **kwargs):
            process = await spawn(sys.executable, '-c', 'import time; time.sleep(60)', **kwargs)
            processes.append(process)
            ready.set()
            return process
        with patch('asyncio.create_subprocess_exec', side_effect=local_process):
            task = asyncio.create_task(self.remote._run_ssh_command('fixture', timeout_s=120))
            await ready.wait()
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertIsNotNone(processes[0].returncode)

    async def assert_cancelled_child_is_reaped(self, operation):
        spawn = asyncio.create_subprocess_exec
        ready = asyncio.Event()
        processes = []
        async def local_process(*args, **kwargs):
            process = await spawn(sys.executable, '-c', 'import time; time.sleep(60)', **kwargs)
            processes.append(process)
            ready.set()
            return process
        with patch('asyncio.create_subprocess_exec', side_effect=local_process):
            task = asyncio.create_task(operation())
            try:
                await asyncio.wait_for(ready.wait(), timeout=5)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                self.assertIsNotNone(processes[0].returncode,
                    'Cancellation must reap the child before its owner releases the transfer copy')
            finally:
                if not task.done():
                    task.cancel()
                for process in processes:
                    if process.returncode is None:
                        process.kill()
                        await process.wait()

    async def test_cancelled_input_upload_stops_child_before_reporting_or_caching_success(self):
        path = self.root / 'input.txt'
        path.write_bytes(b'original attachment')
        self.remote.ensure_master = AsyncMock()
        self.remote._prune_and_list_input_paths = AsyncMock()
        await self.assert_cancelled_child_is_reaped(
            lambda: self.remote.sync_inputs('telegram:1', [path]))
        self.remote._prune_and_list_input_paths.assert_not_awaited()
        self.assertEqual(self.remote._synced_stats.get('telegram:1'), {})
        self.assertEqual(path.read_bytes(), b'original attachment')

    async def test_cancelled_remote_path_resolution_leaves_no_transfer_copies(self):
        self.remote.ensure_master = AsyncMock()
        await self.assert_cancelled_child_is_reaped(
            lambda: self.remote.fetch_files(session_id='telegram:1', remote_paths=['report.txt']))
        self.assertFalse(list(self.config.artifact_dir.rglob('fetch-*')))

    async def test_cancelled_master_start_reaps_local_starter(self):
        self.remote._check_master_alive = AsyncMock(return_value=False)
        await self.assert_cancelled_child_is_reaped(self.remote.ensure_master)
        self.assertFalse(self.remote._master_started)

    async def test_cancelled_master_check_reaps_local_probe(self):
        await self.assert_cancelled_child_is_reaped(self.remote._check_master_alive)

    async def test_cancelled_master_exit_reaps_local_control_request(self):
        self.remote._master_started = True
        await self.assert_cancelled_child_is_reaped(self.remote.aclose)
