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
from tgchatbot.tools.read_doc import ReadDocTool
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
        originals = {self.paths.root + '/a/report.txt': b'first report',
                     self.paths.root + '/b/report.txt': b'second report'}
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
        paths = RemoteSessionPaths(str(root))
        Path(paths.root).mkdir(parents=True)
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
        Path(paths.root, 'selected.txt').symlink_to(outside)
        transfer = unittest.mock.Mock(wraps=self.remote._scp_base_args)
        self.remote._scp_base_args = transfer
        result = await FileSendTool(self.config, self.remote).run(
            {'paths': ['selected.txt']}, ToolContext('telegram:1', 'Participant'))
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
        original = Path(paths.root, 'source.txt')
        original.write_bytes(b'Selected document')
        Path(paths.root, 'chosen.txt').symlink_to(original)
        result = await FileSendTool(self.config, self.remote).run(
            {'paths': ['chosen.txt']}, ToolContext('telegram:1', 'Participant'))
        try:
            self.assertTrue(result.output['ok'], result.output)
            self.assertEqual(result.output['prepared_files'], [{'filename': 'chosen.txt', 'workspace_path': 'chosen.txt'}])
            self.assertEqual([artifact.path.read_bytes() for artifact in result.artifacts], [b'Selected document'])
            self.assertEqual(original.read_bytes(), b'Selected document')
        finally:
            for artifact in result.artifacts:
                artifact.discard()

    async def test_sync_preserves_existing_and_generated_files_and_shell_display_bounds(self):
        paths = self.process_workspace()
        self.remote.ssh = replace(self.remote.ssh, max_stdout_chars=7)
        originals = {}
        for relative in ('legacy.txt', '2026-04-29/old.txt', 'custom/generated.txt'):
            original = Path(paths.root, relative)
            original.parent.mkdir(parents=True, exist_ok=True)
            original.write_text(relative)
            originals[original] = original.read_bytes()
        empty = Path(paths.root, 'chosen-empty-directory')
        empty.mkdir()
        incoming = self.root / '新的附件.txt'
        incoming.write_bytes(b'new attachment')
        self.remote._scp_base_args = lambda: [sys.executable, '-c',
            'import shutil,sys; shutil.copyfile(sys.argv[1],sys.argv[2].split(":",1)[1])']
        result = await self.remote.sync_inputs('telegram:1', [incoming], sent_at='2026-04-30T00:00:00+00:00')
        expected = Path(result.paths_by_source[str(incoming.resolve())])
        self.assertEqual(expected.parent, Path(paths.root, '2026-04-30'))
        self.assertRegex(expected.name, r'新的附件_[0-9a-f]+\.txt')
        self.assertEqual(expected.read_bytes(), incoming.read_bytes())
        self.assertEqual({path: path.read_bytes() for path in originals}, originals)
        self.assertTrue(empty.is_dir())
        displayed = await self.remote.run_shell(session_id='telegram:1',
            command='python3 -c "print(chr(22909) * 20000)"', timeout_s=10)
        self.assertEqual(displayed['stdout'], '好' * self.remote.ssh.max_stdout_chars)

    async def test_input_dates_follow_original_messages_in_configured_timezone_and_paths_remain_readable(self):
        paths = self.process_workspace()
        self.remote._scp_base_args = unittest.mock.Mock(side_effect=lambda: [sys.executable, '-c',
            'import shutil,sys; shutil.copyfile(sys.argv[1],sys.argv[2].split(":",1)[1])'])
        for zone, before, after in (
            ('Asia/Shanghai', '2026-04-29T15:59:59+00:00', '2026-04-29T16:00:00+00:00'),
            ('America/Los_Angeles', '2026-04-30T06:59:59+00:00', '2026-04-30T07:00:00+00:00'),
        ):
            self.remote.config = replace(self.config, default_metadata_timezone=zone)
            for number, sent_at in enumerate((before, after), start=29):
                with self.subTest(zone=zone, sent_at=sent_at):
                    incoming = self.root / f'{zone.rsplit("/", 1)[-1]}-{number}-一份\'报告.txt'
                    incoming.write_text(f'Document from {sent_at}', encoding='utf-8')
                    # An export copied today has an unrelated filesystem date.
                    os.utime(incoming, ns=(1, 1))
                    result = await self.remote.sync_inputs('telegram:1', [incoming], sent_at=sent_at)
                    expected = Path(result.paths_by_source[str(incoming.resolve())])
                    self.assertEqual(expected.parent, Path(paths.root, f'2026-04-{number}'))
                    self.assertTrue(expected.stem.startswith(incoming.stem + '_'))
                    self.assertEqual(expected.suffix, incoming.suffix)
                    self.assertEqual(expected.read_bytes(), incoming.read_bytes())
                    self.remote._scp_base_args.reset_mock()
                    repeated = await self.remote.sync_inputs('telegram:1', [incoming], sent_at=sent_at)
                    self.assertEqual(repeated.paths_by_source, result.paths_by_source)
                    self.remote._scp_base_args.assert_called_once()
                    relative = expected.relative_to(paths.root).as_posix()
                    read = await ReadDocTool(self.config, self.remote).run(
                        {'path': relative, 'format': 'text'},
                        ToolContext('telegram:1', 'Participant'))
                    self.assertTrue(read.output['ok'], read.output)
                    self.assertEqual(read.evidence_parts[0].text, incoming.read_text())
        retained = list(Path(paths.root).rglob('*.txt'))
        self.assertEqual(len(retained), 4)
        self.assertTrue(all(path.relative_to(paths.root).parts[0] in {'2026-04-29', '2026-04-30'} for path in retained))

    async def test_same_basename_bytes_get_distinct_paths_and_replays_keep_the_original_path(self):
        paths = self.process_workspace()
        first = self.root / 'staged-first.bin'
        second = self.root / 'staged-second.bin'
        first.write_bytes(b'First original')
        second.write_bytes(b'Second original')
        self.remote._scp_base_args = lambda: [sys.executable, '-c',
            'import shutil,sys; shutil.copyfile(sys.argv[1],sys.argv[2].split(":",1)[1])']
        selected_paths = []
        for selected in (first, second, first):
            result = await self.remote.sync_inputs('telegram:1', [selected],
                sent_at='2026-04-30T00:00:00+00:00',
                filenames={str(selected.resolve()): '报告.txt'})
            remote_path = Path(result.paths_by_source[str(selected.resolve())])
            self.assertEqual(remote_path.parent, Path(paths.root, '2026-04-30'))
            self.assertRegex(remote_path.name, r'报告_[0-9a-f]+\.txt')
            self.assertEqual(remote_path.read_bytes(), selected.read_bytes())
            selected_paths.append(remote_path)
        self.assertNotEqual(selected_paths[0], selected_paths[1])
        self.assertEqual(selected_paths[0], selected_paths[2])
        self.assertEqual({path: path.read_bytes() for path in Path(paths.root).rglob('*.txt')},
            {selected_paths[0]: first.read_bytes(), selected_paths[1]: second.read_bytes()})
        # A new local staging name must not produce another remote identity.
        restaged = self.root / 'restaged.bin'
        restaged.write_bytes(first.read_bytes())
        selected_paths[0].write_bytes(b'Changed remotely before replay')
        replay = await self.remote.sync_inputs('telegram:1', [restaged],
            sent_at='2026-04-30T00:00:00+00:00', filenames={str(restaged.resolve()): '报告.txt'})
        self.assertEqual(Path(replay.paths_by_source[str(restaged.resolve())]), selected_paths[0])
        self.assertEqual(selected_paths[0].read_bytes(), first.read_bytes())
        # A different date remains a distinct user-addressable original.
        later = await self.remote.sync_inputs('telegram:1', [second],
            sent_at='2026-05-01T00:00:00+00:00', filenames={str(second.resolve()): '报告.txt'})
        later_path = Path(later.paths_by_source[str(second.resolve())])
        self.assertNotEqual(later_path, selected_paths[1])
        self.assertEqual(later_path.read_bytes(), second.read_bytes())
        self.assertEqual(selected_paths[0].read_bytes(), first.read_bytes())

    async def test_workspace_environment_stays_stable_and_nested_paths_can_be_read_and_sent(self):
        paths = self.process_workspace()
        script = ('import json,os; from pathlib import Path; '
            'root=Path(os.environ["TGCHATBOT_SESSION_DIR"]); '
            'target=root/"custom"/"report.txt"; target.parent.mkdir(parents=True,exist_ok=True); '
            'target.write_text("A persistent report"); '
            'print(json.dumps({"root":str(root),"cwd":os.getcwd()}))')
        result = await self.remote.run_python(session_id='telegram:1', code=script, timeout_s=10)
        expected_environment = {'root': paths.root, 'cwd': paths.root}
        self.assertTrue(result['ok'], result)
        self.assertEqual(json.loads(result['stdout']), expected_environment)
        later = await self.remote.run_shell(session_id='telegram:1',
            command='python3 -c \'import json,os; print(json.dumps({"root":os.environ["TGCHATBOT_SESSION_DIR"],"cwd":os.getcwd()}))\'', timeout_s=10)
        self.assertEqual(json.loads(later['stdout']), expected_environment)
        read = await ReadDocTool(self.config, self.remote).run(
            {'path': 'custom/report.txt', 'format': 'text'},
            ToolContext('telegram:1', 'Participant'))
        self.assertTrue(read.output['ok'], read.output)
        self.assertEqual(read.evidence_parts[0].text, 'A persistent report')
        sent = await FileSendTool(self.config, self.remote).run(
            {'paths': ['custom/report.txt']}, ToolContext('telegram:1', 'Participant'))
        try:
            self.assertTrue(sent.output['ok'], sent.output)
            self.assertEqual([(item.filename, item.workspace_path, item.path.read_bytes()) for item in sent.artifacts],
                [('report.txt', 'custom/report.txt', b'A persistent report')])
            self.assertEqual(Path(paths.root, 'custom/report.txt').read_text(), 'A persistent report')
        finally:
            for artifact in sent.artifacts:
                artifact.discard()

    async def test_file_send_skips_a_missing_file_without_losing_available_files(self):
        paths = self.process_workspace()
        original = Path(paths.root, 'available.txt')
        original.write_bytes(b'Available requested output')
        result = await FileSendTool(self.config, self.remote).run(
            {'paths': ['missing.txt', 'available.txt']},
            ToolContext('telegram:1', 'Participant'))
        try:
            self.assertTrue(result.output['ok'], result.output)
            self.assertEqual(result.output['requested_paths'], 2)
            self.assertEqual(result.output['prepared_files'], [{'filename': 'available.txt', 'workspace_path': 'available.txt'}])
            self.assertEqual([artifact.path.read_bytes() for artifact in result.artifacts], [original.read_bytes()])
            self.assertEqual(list(self.config.artifact_dir.rglob('fetch-*')), [result.artifacts[0].path])
        finally:
            for artifact in result.artifacts:
                artifact.discard()
        self.assertFalse(list(self.config.artifact_dir.rglob('fetch-*')))

    async def test_partial_send_of_same_basename_dates_reports_each_workspace_path(self):
        paths = self.process_workspace()
        originals = {'2026-04-29/report.txt': b'Earlier report', '2026-04-30/report.txt': b'Later report'}
        for relative, content in originals.items():
            path = Path(paths.root, relative)
            path.parent.mkdir(parents=True)
            path.write_bytes(content)
        result = await FileSendTool(self.config, self.remote).run(
            {'paths': ['missing/report.txt', *originals]}, ToolContext('telegram:1', 'Participant'))
        try:
            self.assertTrue(result.output['ok'], result.output)
            self.assertEqual(result.output['requested_paths'], 3)
            self.assertEqual(result.output['prepared_files'], [
                {'filename': 'report.txt', 'workspace_path': relative} for relative in originals])
            self.assertEqual([artifact.path.read_bytes() for artifact in result.artifacts], list(originals.values()))
            bot = SimpleNamespace(send_document=AsyncMock(side_effect=[SimpleNamespace(message_id=42), BadRequest('rejected')]))
            first = await deliver_artifact(bot, chat_id=1, artifact=result.artifacts[0])
            with self.assertLogs('tgchatbot.transports.artifact_delivery', level='ERROR'):
                second = await deliver_artifact(bot, chat_id=1, artifact=result.artifacts[1])
            self.assertTrue(first['sent'])
            self.assertFalse(second['sent'])
            self.assertEqual([first['workspace_path'], second['workspace_path']], list(originals))
            self.assertTrue(all(Path(paths.root, relative).read_bytes() == content for relative, content in originals.items()))
        finally:
            for artifact in result.artifacts:
                artifact.discard()

    async def test_python_source_with_shell_delimiter_executes_unchanged(self):
        paths = RemoteSessionPaths(str(self.root))
        self.remote.ensure_session_dirs.return_value = paths
        existing_script = self.root / 'run.py'
        existing_script.write_text('print("My saved script")\n')

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
        self.assertEqual(existing_script.read_text(), 'print("My saved script")\n')

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

    async def test_cancelled_input_upload_stops_child_before_reporting_success(self):
        path = self.root / 'input.txt'
        path.write_bytes(b'original attachment')
        self.remote.ensure_master = AsyncMock()
        await self.assert_cancelled_child_is_reaped(
            lambda: self.remote.sync_inputs('telegram:1', [path]))
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
