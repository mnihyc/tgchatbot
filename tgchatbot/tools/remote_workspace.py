from __future__ import annotations

import asyncio
import contextlib
import codecs
import hashlib
import json
import logging
import os
import posixpath
import tempfile
import re
from importlib.resources import files
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tgchatbot.config import AppConfig
from tgchatbot.domain.models import OutboundArtifact
from tgchatbot.domain.timestamps import format_timestamp
from tgchatbot.logging_config import clip_for_log

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemoteSessionPaths:
    root: str


@dataclass(frozen=True)
class RemoteSyncResult:
    paths_by_source: dict[str, str]


@dataclass(frozen=True)
class RemoteFileResult:
    workspace_path: str
    artifact: OutboundArtifact | None = None
    error: str | None = None


class RemoteWorkspaceClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.ssh = config.ssh_exec
        self._master_lock = asyncio.Lock()
        self._master_started = False
        digest = hashlib.sha1(f"{self.ssh.host}:{self.ssh.port}".encode('utf-8')).hexdigest()[:12] if self.ssh.host else 'disabled'
        self._control_dir = self.config.temp_dir / 'ssh_mux'
        self._control_dir.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(PermissionError, FileNotFoundError):
            self._control_dir.chmod(0o700)
        self._control_path = self._control_dir / f'mux-{digest}'

    @property
    def enabled(self) -> bool:
        return self.ssh.enabled and bool(self.ssh.host)

    async def warmup(self) -> None:
        if not self.enabled:
            return
        try:
            await self.ensure_master()
        except Exception:
            logger.exception('Failed to warm up persistent SSH master connection')

    def _safe_session_dir(self, session_id: str) -> str:
        base = re.sub(r'[^A-Za-z0-9._-]+', '_', session_id).strip('._-') or 'session'
        digest = hashlib.sha1(session_id.encode('utf-8')).hexdigest()[:4]
        return f'{base}-{digest}'

    def session_paths(self, session_id: str) -> RemoteSessionPaths:
        session_dir = self._safe_session_dir(session_id)
        root = f"{self.ssh.workdir.rstrip('/')}/{session_dir}"
        return RemoteSessionPaths(root=root)

    async def ensure_master(self) -> None:
        if not self.enabled:
            raise RuntimeError('SSH remote workspace is not configured')
        async with self._master_lock:
            if self._master_started and await self._check_master_alive():
                return
            if await self._check_master_alive():
                self._master_started = True
                logger.info('remote.master.reuse host=%s port=%s', self.ssh.host, self.ssh.port)
                return
            # An interrupted container can leave its dead multiplexing socket.
            if self._control_path.is_socket():
                self._control_path.unlink(missing_ok=True)
            cmd = [
                'ssh',
                '-MNf',
                '-p',
                str(self.ssh.port),
                '-o', f'ConnectTimeout={self.ssh.connect_timeout_s}',
                '-o', 'BatchMode=yes',
                '-o', f'ServerAliveInterval={self.ssh.server_alive_interval_s}',
                '-o', f'ServerAliveCountMax={self.ssh.server_alive_count_max}',
                '-o', 'ControlMaster=yes',
                '-o', f'ControlPersist={self.ssh.control_persist_s}',
                '-S',
                str(self._control_path),
            ]
            if self.ssh.identity_file:
                cmd.extend(['-i', self.ssh.identity_file])
            cmd.append(self.ssh.host)
            with tempfile.TemporaryFile() as stderr_file:
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=stderr_file,
                )
                try:
                    await asyncio.wait_for(proc.wait(), timeout=self.ssh.connect_timeout_s + 5)
                except asyncio.CancelledError:
                    await self._terminate_process(proc)
                    raise
                except asyncio.TimeoutError:
                    await self._terminate_process(proc)
                    raise RuntimeError('Timed out starting persistent SSH master')
                stderr_file.flush()
                stderr_file.seek(0)
                stderr = stderr_file.read().decode('utf-8', errors='replace')
            if proc.returncode != 0:
                raise RuntimeError(f'Failed to start persistent SSH master: {stderr[:400] or f"exit {proc.returncode}"}')
            if not await self._wait_for_master_ready():
                raise RuntimeError('Persistent SSH master exited without becoming ready')
            self._master_started = True
            logger.info('remote.master.start host=%s port=%s', self.ssh.host, self.ssh.port)

    async def _check_master_alive(self) -> bool:
        if not self.enabled:
            return False
        cmd = [
            'ssh',
            '-O',
            'check',
            '-p',
            str(self.ssh.port),
            '-o', 'BatchMode=yes',
            '-o', f'ConnectTimeout={self.ssh.connect_timeout_s}',
            '-S',
            str(self._control_path),
        ]
        if self.ssh.identity_file:
            cmd.extend(['-i', self.ssh.identity_file])
        cmd.append(self.ssh.host)
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        try:
            await asyncio.wait_for(proc.wait(), timeout=self.ssh.connect_timeout_s + 2)
        except asyncio.CancelledError:
            await self._terminate_process(proc)
            raise
        except asyncio.TimeoutError:
            await self._terminate_process(proc)
            return False
        return proc.returncode == 0

    async def aclose(self) -> None:
        if not self.enabled:
            return
        async with self._master_lock:
            if not self._master_started and not await self._check_master_alive():
                return
            cmd = [
                'ssh', '-O', 'exit',
                '-p', str(self.ssh.port),
                '-o', 'BatchMode=yes',
                '-o', f'ConnectTimeout={self.ssh.connect_timeout_s}',
                '-S', str(self._control_path),
            ]
            if self.ssh.identity_file:
                cmd.extend(['-i', self.ssh.identity_file])
            cmd.append(self.ssh.host)
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            try:
                await asyncio.wait_for(proc.wait(), timeout=self.ssh.connect_timeout_s + 5)
            except asyncio.CancelledError:
                await self._terminate_process(proc)
                raise
            except asyncio.TimeoutError:
                await self._terminate_process(proc)

            still_alive = await self._check_master_alive()
            self._master_started = still_alive
            if still_alive:
                logger.warning('remote.master.stop_failed host=%s port=%s', self.ssh.host, self.ssh.port)
            else:
                logger.info('remote.master.stop host=%s port=%s', self.ssh.host, self.ssh.port)

    async def ensure_session_dirs(self, session_id: str) -> RemoteSessionPaths:
        await self.ensure_master()
        paths = self.session_paths(session_id)
        mkdir_cmd = f"mkdir -p {shq(paths.root)}"
        result = await self._run_ssh_command(mkdir_cmd, timeout_s=self.ssh.connect_timeout_s + 10)
        if result['returncode'] != 0:
            raise RuntimeError(result['stderr'] or 'Failed to create remote session directories')
        logger.debug('remote.session.ready sid=%s root=%s', clip_for_log(session_id, limit=48), paths.root)
        return paths

    async def sync_inputs(self, session_id: str, local_paths: tuple[Path, ...] | list[Path], *,
                          sent_at: datetime | str | None = None,
                          filenames: dict[str, str] | None = None) -> RemoteSyncResult:
        paths = await self.ensure_session_dirs(session_id)
        # Intake owns the original UTC timestamp and filename. The workspace
        # owns their remote location and reports each successful overwrite.
        day = format_timestamp(sent_at if sent_at is not None else datetime.now(timezone.utc),
            self.config.default_metadata_timezone).split('T', 1)[0]
        upload_dir = f"{paths.root.rstrip('/')}/{day}"
        to_upload: list[tuple[Path, str]] = []
        skipped_oversize = 0
        for path in local_paths:
            if not path.exists() or not path.is_file():
                continue
            stat = path.stat()
            if stat.st_size > self.ssh.max_input_file_bytes:
                skipped_oversize += 1
                logger.warning('remote.sync.skip_oversize sid=%s file=%s size=%s limit=%s', clip_for_log(session_id, limit=48), path.name, stat.st_size, self.ssh.max_input_file_bytes)
                continue
            key = str(path.resolve())
            filename = await asyncio.to_thread(self._upload_filename, path,
                (filenames or {}).get(key) or path.name)
            to_upload.append((path, f'{upload_dir}/{filename}'))
        if to_upload:
            result = await self._run_ssh_command(f"mkdir -p {shq(upload_dir)}",
                timeout_s=self.ssh.connect_timeout_s + 10)
            if result['returncode'] != 0:
                raise RuntimeError(result['stderr'] or 'Failed to create remote upload directory')
        paths_by_source: dict[str, str] = {}
        for path, remote_path in to_upload:
            scp_cmd = self._scp_base_args()
            scp_cmd.extend([str(path), f"{self.ssh.host}:{remote_path}"])
            proc = await asyncio.create_subprocess_exec(
                *scp_cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _stdout, stderr = await asyncio.wait_for(proc.communicate(),
                    timeout=self.ssh.default_timeout_s + self.ssh.connect_timeout_s)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                await self._terminate_process(proc)
                raise
            if proc.returncode != 0:
                raise RuntimeError(f'Failed to sync input files: {stderr.decode("utf-8", errors="replace")[:400]}')
            paths_by_source[str(path.resolve())] = remote_path
        logger.info('remote.sync.done sid=%s uploaded=%s skipped_oversize=%s',
            clip_for_log(session_id, limit=48), len(paths_by_source), skipped_oversize)
        return RemoteSyncResult(paths_by_source=paths_by_source)

    @staticmethod
    def _upload_filename(path: Path, original_name: str) -> str:
        filename = Path(original_name).name
        if filename in {'', '.', '..'}:
            filename = path.name
        with path.open('rb') as stream:
            file_id = hashlib.file_digest(stream, 'sha256').hexdigest()[:16]
        named = Path(filename)
        return f'{named.stem}_{file_id}{named.suffix}'

    async def run_shell(self, *, session_id: str, command: str, timeout_s: int) -> dict[str, Any]:
        paths = await self.ensure_session_dirs(session_id)
        env = f"TGCHATBOT_SESSION_DIR={shq(paths.root)} "
        wrapped = f"set -e; cd {shq(paths.root)}; {env} sh -lc {shq(command)}"
        logger.info('remote.shell sid=%s timeout_s=%s cmd=%s', clip_for_log(session_id, limit=48), timeout_s, clip_for_log(command, limit=140))
        return await self._run_ssh_command(wrapped, timeout_s=timeout_s)

    async def run_python(
        self,
        *,
        session_id: str,
        code: str,
        timeout_s: int,
    ) -> dict[str, Any]:
        paths = await self.ensure_session_dirs(session_id)
        wrapped = (
            f"set -e; cd {shq(paths.root)}; "
            f"TGCHATBOT_SESSION_DIR={shq(paths.root)} python3 -c {shq(code)}"
        )
        logger.info('remote.python sid=%s timeout_s=%s code=%s', clip_for_log(session_id, limit=48), timeout_s, clip_for_log(code, limit=140))
        return await self._run_ssh_command(wrapped, timeout_s=timeout_s)

    async def fetch_files(
        self,
        *,
        session_id: str,
        remote_paths: list[str] | None = None,
        max_files: int | None = None,
    ) -> list[RemoteFileResult]:
        paths = await self.ensure_session_dirs(session_id)
        max_files = self.ssh.max_output_files if max_files is None else max_files
        if remote_paths:
            remote_paths = [paths.root.rstrip('/') + '/' + path if not posixpath.isabs(path) else path for path in remote_paths]
            selected = [self._validate_remote_path(paths, value) for value in remote_paths]
        else:
            raise RuntimeError('At least one remote path must be specified for fetching')
        if not selected:
            return []
        resolved = await self._resolve_remote_paths(paths, selected[:max_files])
        local_dir = self.config.artifact_dir / session_id / 'remote_fetch'
        local_dir.mkdir(parents=True, exist_ok=True)
        results: list[RemoteFileResult] = []
        try:
            for index, requested_path in enumerate(selected):
                workspace_path = posixpath.relpath(requested_path, paths.root)
                if index >= max_files:
                    results.append(RemoteFileResult(workspace_path,
                        error=f'Request exceeds the configured file limit ({max_files}).'))
                    continue
                remote_path = resolved[index]
                filename = posixpath.basename(requested_path)
                local_path = None
                retained = False
                try:
                    descriptor, temporary = tempfile.mkstemp(prefix='fetch-', suffix='-' + filename, dir=local_dir)
                    os.close(descriptor)
                    local_path = Path(temporary)
                    scp_cmd = self._scp_base_args()
                    scp_cmd.extend([f"{self.ssh.host}:{remote_path}", str(local_path)])
                    proc = await asyncio.create_subprocess_exec(
                        *scp_cmd,
                        stdin=asyncio.subprocess.DEVNULL,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    try:
                        _stdout, stderr = await asyncio.wait_for(proc.communicate(),
                            timeout=self.ssh.default_timeout_s + self.ssh.connect_timeout_s)
                    except asyncio.CancelledError:
                        await self._terminate_process(proc)
                        raise
                    except asyncio.TimeoutError:
                        await self._terminate_process(proc)
                        results.append(RemoteFileResult(workspace_path, error='Timed out fetching the file.'))
                        continue
                    if proc.returncode != 0:
                        logger.warning('Failed to fetch remote file %s: %s', remote_path, stderr.decode('utf-8', errors='replace')[:300])
                        results.append(RemoteFileResult(workspace_path, error='Remote file transfer failed.'))
                        continue
                    if not local_path.is_file():
                        results.append(RemoteFileResult(workspace_path, error='Fetched path is not a regular file.'))
                        continue
                    if local_path.stat().st_size > self.ssh.max_output_file_bytes:
                        results.append(RemoteFileResult(workspace_path,
                            error=f'File exceeds the configured output size limit ({self.ssh.max_output_file_bytes} bytes).'))
                        continue
                    artifact = OutboundArtifact(path=local_path, filename=filename, temporary=True,
                        workspace_path=workspace_path)
                    results.append(RemoteFileResult(workspace_path, artifact=artifact))
                    retained = True
                except OSError as exc:
                    logger.exception('Local file transfer failed for %s', requested_path)
                    results.append(RemoteFileResult(workspace_path,
                        error=f'Local file transfer failed: {exc.__class__.__name__}'))
                finally:
                    if local_path is not None and not retained:
                        local_path.unlink(missing_ok=True)
        except BaseException:
            for result in results:
                if result.artifact is not None:
                    result.artifact.discard()
            raise
        return results

    async def _resolve_remote_paths(self, paths: RemoteSessionPaths, selected: list[str]) -> list[str]:
        # Lexical prefix checks cannot see remote symlinks. Resolve once before
        # any transfer, preserving the requested alias only as its upload name.
        program = (
            'import json\nfrom pathlib import Path\n'
            f'root = Path({paths.root!r}).resolve()\n'
            'resolved = []\n'
            f'for value in {selected!r}:\n'
            '    path = Path(value).resolve()\n'
            '    if not path.is_relative_to(root):\n'
            '        raise ValueError("Requested remote path is outside the session workspace")\n'
            '    resolved.append(str(path))\n'
            'print(json.dumps(resolved, ensure_ascii=False))\n'
        )
        result = await self._run_ssh_command(f'python3 -c {shq(program)}',
            timeout_s=self.ssh.default_timeout_s, full_stdout=True)
        if not result['ok']:
            raise RuntimeError(result['stderr'] or 'Could not resolve selected workspace files')
        return json.loads(result['stdout'])

    async def inspect_file(self, *, session_id: str, path: str,
                           format: str, start: int | None, end: int | None,
                           limits: dict[str, Any]) -> dict[str, Any]:
        paths = await self.ensure_session_dirs(session_id)
        selected = path if posixpath.isabs(path) else paths.root.rstrip('/') + '/' + path
        selected = self._validate_remote_path(paths, selected)
        request = {'root': paths.root, 'path': selected, 'format': format,
                   'start': start, 'end': end, 'limits': limits}
        program = (files('tgchatbot.media').joinpath('image_encoding.py').read_text()
                   + '\n' + files('tgchatbot.tools').joinpath('remote_reader.py').read_text())
        result = await self._run_ssh_command(
            f'python3 -c {shq(program)} {shq(json.dumps(request))}',
            timeout_s=self.ssh.max_tool_timeout_s,
            # Prepared binary evidence is base64 in this private transport, not
            # shell stdout for the agent. Leave room for encoding and labels.
            stdout_limit=2 * limits['bytes'] + self.ssh.max_stdout_chars)
        if not result['ok']:
            return {'ok': False, 'error': result['stderr'] or 'Remote file inspection failed'}
        try:
            return json.loads(result['stdout'])
        except json.JSONDecodeError as exc:
            raise RuntimeError('Remote reader did not return a complete result') from exc

    async def _run_ssh_command(self, command: str, *, timeout_s: int,
                               stdout_limit: int | None = None, full_stdout: bool = False) -> dict[str, Any]:
        await self.ensure_master()
        cmd = self._ssh_base_args() + [command]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_capture: dict[str, Any] = {}
        stderr_capture: dict[str, Any] = {}
        try:
            stdout, stderr, _ = await asyncio.wait_for(asyncio.gather(
                self._read_output(proc.stdout, None if full_stdout else
                    self.ssh.max_stdout_chars if stdout_limit is None else stdout_limit,
                    capture=stdout_capture),
                self._read_output(proc.stderr, self.ssh.max_stderr_chars, capture=stderr_capture),
                proc.wait()), timeout=timeout_s + self.ssh.connect_timeout_s)
        except asyncio.CancelledError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.communicate()
            raise
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.communicate()
            logger.warning('remote.exec.timeout timeout_s=%s', timeout_s)
            result = {'ok': False, 'returncode': None, 'outcome': 'unknown',
                'error': 'Timed out waiting for execution; remote completion is unconfirmed.',
                'stdout': stdout_capture.get('text', ''), 'stderr': stderr_capture.get('text', '')}
        else:
            result = {
                'ok': proc.returncode == 0,
                'returncode': proc.returncode,
                'stdout': stdout,
                'stderr': stderr,
            }
            # SSH reserves 255 for errors; a remote exit of 255 is indistinguishable.
            if proc.returncode == 255 or proc.returncode < 0:
                result.update(returncode=None, outcome='unknown',
                    error='Remote connection ended without a confirmed command result.')
        for channel, capture in (('stdout', stdout_capture), ('stderr', stderr_capture)):
            if capture.get('truncated'):
                result[f'{channel}_truncated'] = True
        level = logger.info if result['ok'] else logger.warning
        level('remote.exec.done rc=%s stdout=%s stderr=%s', result['returncode'], len(result['stdout']), len(result['stderr']))
        return result

    @staticmethod
    async def _read_output(stream: asyncio.StreamReader, limit: int | None, *,
                           capture: dict[str, Any] | None = None) -> str:
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        kept: list[str] = []
        remaining = None if limit is None else max(0, limit)
        truncated = False
        try:
            while chunk := await stream.read(64 * 1024):
                if remaining == 0:
                    truncated = True
                    continue
                decoded = decoder.decode(chunk)
                if remaining is None:
                    kept.append(decoded)
                else:
                    text = decoded[:remaining]
                    if text:
                        kept.append(text)
                    remaining -= len(text)
                    truncated = truncated or len(text) < len(decoded)
        finally:
            final = decoder.decode(b'', final=True)
            if remaining is None:
                kept.append(final)
            else:
                kept.append(final[:remaining])
                truncated = truncated or len(final) > remaining
            if capture is not None:
                capture.update(text=''.join(kept), truncated=truncated)
        return ''.join(kept)

    async def _terminate_process(self, proc: asyncio.subprocess.Process) -> None:
        with contextlib.suppress(ProcessLookupError):
            proc.kill()
        with contextlib.suppress(Exception):
            await proc.wait()

    async def _wait_for_master_ready(self, *, timeout_s: float = 2.0) -> bool:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + max(timeout_s, 0.1)
        while True:
            if await self._check_master_alive():
                return True
            if loop.time() >= deadline:
                return False
            await asyncio.sleep(0.1)

    def _ssh_base_args(self) -> list[str]:
        args = [
            'ssh',
            '-p',
            str(self.ssh.port),
            '-o', f'ConnectTimeout={self.ssh.connect_timeout_s}',
            '-o', f'ServerAliveInterval={self.ssh.server_alive_interval_s}',
            '-o', f'ServerAliveCountMax={self.ssh.server_alive_count_max}',
            '-o', f'ControlPath={self._control_path}',
            '-o', 'ControlMaster=auto',
            '-o', 'BatchMode=yes',
        ]
        if self.ssh.identity_file:
            args.extend(['-i', self.ssh.identity_file])
        args.append(self.ssh.host)
        return args

    def _scp_base_args(self) -> list[str]:
        args = [
            'scp',
            '-P',
            str(self.ssh.port),
            '-o', f'ConnectTimeout={self.ssh.connect_timeout_s}',
            '-o', f'ServerAliveInterval={self.ssh.server_alive_interval_s}',
            '-o', f'ServerAliveCountMax={self.ssh.server_alive_count_max}',
            '-o', f'ControlPath={self._control_path}',
            '-o', 'ControlMaster=auto',
            '-o', 'BatchMode=yes',
        ]
        if self.ssh.identity_file:
            args.extend(['-i', self.ssh.identity_file])
        return args

    @staticmethod
    def _validate_remote_path(paths: RemoteSessionPaths, remote_path: str) -> str:
        normalized = posixpath.normpath(remote_path.strip())
        root = posixpath.normpath(paths.root)
        if normalized == root:
            return normalized
        if not normalized.startswith(root.rstrip('/') + '/'):
            raise RuntimeError('Requested remote path is outside the session workspace')
        return normalized


def shq(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"
