from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import posixpath
import tempfile
import re
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tgchatbot.config import AppConfig
from tgchatbot.domain.models import OutboundArtifact
from tgchatbot.logging_config import clip_for_log

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RemoteSessionPaths:
    root: str
    inputs: str
    outputs: str


@dataclass(frozen=True)
class RemoteSyncResult:
    kept_paths: list[str]
    rotated_paths: list[str]


class RemoteWorkspaceClient:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.ssh = config.ssh_exec
        self._master_lock = asyncio.Lock()
        self._master_started = False
        digest = hashlib.sha1(f"{self.ssh.host}:{self.ssh.port}".encode('utf-8')).hexdigest()[:12] if self.ssh.host else 'disabled'
        self._control_dir = self.config.data_dir / 'ssh_mux'
        self._control_dir.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(PermissionError, FileNotFoundError):
            self._control_dir.chmod(0o700)
        self._control_path = self._control_dir / f'mux-{digest}'
        self._synced_stats: dict[str, dict[str, tuple[int, int, str]]] = {}

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
        return RemoteSessionPaths(root=root, inputs=f'{root}/inputs', outputs=f'{root}/outputs')

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
        mkdir_cmd = f"mkdir -p {shq(paths.root)} {shq(paths.inputs)} {shq(paths.outputs)}"
        result = await self._run_ssh_command(mkdir_cmd, timeout_s=self.ssh.connect_timeout_s + 10)
        if result['returncode'] != 0:
            raise RuntimeError(result['stderr'] or 'Failed to create remote session directories')
        logger.debug('remote.session.ready sid=%s root=%s', clip_for_log(session_id, limit=48), paths.root)
        return paths

    async def sync_inputs(self, session_id: str, local_paths: tuple[Path, ...] | list[Path]) -> RemoteSyncResult:
        await self.ensure_master()
        paths = await self.ensure_session_dirs(session_id)
        requested_remote_paths: list[str] = []
        per_session = self._synced_stats.setdefault(session_id, {})
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
            remote_path = f"{paths.inputs.rstrip('/')}/{Path(path).name}"
            requested_remote_paths.append(remote_path)
            key = str(path.resolve())
            stamp = (stat.st_size, int(stat.st_mtime_ns), remote_path)
            if per_session.get(key) == stamp:
                logger.debug('remote.sync.skip sid=%s file=%s', clip_for_log(session_id, limit=48), path.name)
                continue
            to_upload.append((path, remote_path))
        if to_upload:
            scp_cmd = self._scp_base_args()
            scp_cmd.extend([str(path) for path, _remote in to_upload])
            scp_cmd.append(f"{self.ssh.host}:{paths.inputs.rstrip('/')}/")
            proc = await asyncio.create_subprocess_exec(
                *scp_cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f'Failed to sync input files: {stderr.decode("utf-8", errors="replace")[:400]}')
            for path, remote_path in to_upload:
                stat = path.stat()
                per_session[str(path.resolve())] = (stat.st_size, int(stat.st_mtime_ns), remote_path)
        prune_result = await self._prune_and_list_input_paths(paths)
        surviving = prune_result['kept']
        rotated = prune_result['rotated']
        surviving_set = set(surviving)
        stale_keys = [key for key, stamp in per_session.items() if stamp[2] not in surviving_set]
        for key in stale_keys:
            per_session.pop(key, None)
        requested_surviving = [remote_path for remote_path in requested_remote_paths if remote_path in surviving_set]
        logger.info('remote.sync.done sid=%s uploaded=%s kept=%s rotated=%s skipped_oversize=%s', clip_for_log(session_id, limit=48), len(to_upload), len(requested_surviving), len(rotated), skipped_oversize)
        return RemoteSyncResult(kept_paths=requested_surviving, rotated_paths=rotated)

    async def _prune_and_list_input_paths(self, paths: RemoteSessionPaths) -> dict[str, list[str]]:
        py = (
            "import json, os\n"
            f"base={paths.inputs!r}\n"
            f"limit={int(self.ssh.max_input_files)}\n"
            "rows=[]\n"
            "rotated=[]\n"
            "if os.path.isdir(base):\n"
            "    for name in os.listdir(base):\n"
            "        p=os.path.join(base,name)\n"
            "        if os.path.isfile(p):\n"
            "            st=os.stat(p)\n"
            "            rows.append((st.st_mtime_ns, name, p))\n"
            "rows.sort()\n"
            "if limit <= 0:\n"
            "    for _mtime, _name, p in rows:\n"
            "        rotated.append(p)\n"
            "        try:\n"
            "            os.remove(p)\n"
            "        except FileNotFoundError:\n"
            "            pass\n"
            "    rows = []\n"
            "elif len(rows) > limit:\n"
            "    for _mtime, _name, p in rows[:-limit]:\n"
            "        rotated.append(p)\n"
            "        try:\n"
            "            os.remove(p)\n"
            "        except FileNotFoundError:\n"
            "            pass\n"
            "    rows = rows[-limit:]\n"
            "print(json.dumps({'kept': [p for _mtime, _name, p in rows], 'rotated': rotated}, ensure_ascii=False))\n"
        )
        result = await self._run_ssh_command(f"python3 - <<'PY'\n{py}PY", timeout_s=max(5, self.ssh.connect_timeout_s + 10))
        if result['returncode'] != 0:
            raise RuntimeError(result['stderr'] or 'Failed to prune remote input files')
        try:
            data = json.loads(result['stdout'].strip() or '{"kept": [], "rotated": []}')
        except json.JSONDecodeError as exc:
            raise RuntimeError(f'Invalid remote input listing: {exc}') from exc
        if not isinstance(data, dict):
            raise RuntimeError('Invalid remote input listing payload')
        kept = [str(item) for item in data.get('kept', []) if isinstance(item, str)]
        rotated = [str(item) for item in data.get('rotated', []) if isinstance(item, str)]
        return {'kept': kept, 'rotated': rotated}

    async def run_shell(self, *, session_id: str, command: str, timeout_s: int, cwd_subdir: str | None = None) -> dict[str, Any]:
        paths = await self.ensure_session_dirs(session_id)
        cwd = paths.root if not cwd_subdir else f"{paths.root.rstrip('/')}/{cwd_subdir.lstrip('/')}"
        env = (
            f"TGCHATBOT_SESSION_DIR={shq(paths.root)} "
            f"TGCHATBOT_INPUT_DIR={shq(paths.inputs)} "
            f"TGCHATBOT_OUTPUT_DIR={shq(paths.outputs)} "
        )
        wrapped = f"set -e; cd {shq(cwd)}; {env} sh -lc {shq(command)}"
        logger.info('remote.shell sid=%s cwd=%s timeout_s=%s cmd=%s', clip_for_log(session_id, limit=48), clip_for_log(cwd_subdir or '.', limit=32), timeout_s, clip_for_log(command, limit=140))
        return await self._run_ssh_command(wrapped, timeout_s=timeout_s)

    async def run_python(
        self,
        *,
        session_id: str,
        code: str,
        timeout_s: int,
    ) -> dict[str, Any]:
        paths = await self.ensure_session_dirs(session_id)
        remote_script = f'{paths.root}/run.py'
        wrapped = (
            f"set -e; cat > {shq(remote_script)} <<'PYCODE'\n"
            f"{code}\n"
            "PYCODE\n"
            f"cd {shq(paths.root)}; "
            f"TGCHATBOT_SESSION_DIR={shq(paths.root)} "
            f"TGCHATBOT_INPUT_DIR={shq(paths.inputs)} "
            f"TGCHATBOT_OUTPUT_DIR={shq(paths.outputs)} python3 {shq(remote_script)}"
        )
        logger.info('remote.python sid=%s timeout_s=%s code=%s', clip_for_log(session_id, limit=48), timeout_s, clip_for_log(code, limit=140))
        return await self._run_ssh_command(wrapped, timeout_s=timeout_s)

    async def list_files(self, *, session_id: str, scope: str = 'outputs', max_files: int = 20) -> list[dict[str, Any]]:
        paths = await self.ensure_session_dirs(session_id)
        base = self._scope_to_path(paths, scope)
        py = (
            "import json, os\n"
            f"base={base!r}\n"
            f"limit={int(max_files)}\n"
            "rows=[]\n"
            "if os.path.isdir(base):\n"
            "    for name in sorted(os.listdir(base))[:limit]:\n"
            "        p=os.path.join(base,name)\n"
            "        if os.path.isfile(p):\n"
            "            rows.append({'name': name, 'size_bytes': os.path.getsize(p), 'path': p})\n"
            "print(json.dumps(rows, ensure_ascii=False))\n"
        )
        result = await self._run_ssh_command(f"python3 - <<'PY'\n{py}PY", timeout_s=20)
        if result['returncode'] != 0:
            raise RuntimeError(result['stderr'] or 'Failed to list remote files')
        try:
            data = json.loads(result['stdout'].strip() or '[]')
        except json.JSONDecodeError as exc:
            raise RuntimeError(f'Invalid remote file listing: {exc}') from exc
        return data if isinstance(data, list) else []

    async def fetch_files(
        self,
        *,
        session_id: str,
        remote_paths: list[str] | None = None,
        scope: str = 'outputs',
        max_files: int | None = None,
    ) -> list[OutboundArtifact]:
        paths = await self.ensure_session_dirs(session_id)
        max_files = max_files or self.ssh.max_output_files
        if remote_paths:
            remote_paths = [self._scope_to_path(paths, scope.rstrip('/')) + '/' + path.lstrip('/') if not os.path.isabs(path) else path for path in remote_paths]
            selected = [self._validate_remote_path(paths, value) for value in remote_paths]
        else:
            raise RuntimeError('At least one remote path must be specified for fetching')
        selected = selected[:max_files]
        if not selected:
            return []
        local_dir = self.config.artifact_dir / session_id / 'remote_fetch'
        local_dir.mkdir(parents=True, exist_ok=True)
        artifacts: list[OutboundArtifact] = []
        for remote_path in selected:
            filename = os.path.basename(remote_path)
            local_path = local_dir / filename
            scp_cmd = self._scp_base_args()
            scp_cmd.extend([f"{self.ssh.host}:{remote_path}", str(local_path)])
            proc = await asyncio.create_subprocess_exec(
                *scp_cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await proc.communicate()
            if proc.returncode != 0:
                logger.warning('Failed to fetch remote file %s: %s', remote_path, stderr.decode('utf-8', errors='replace')[:300])
                continue
            if not local_path.exists() or not local_path.is_file():
                continue
            if local_path.stat().st_size > self.ssh.max_output_file_bytes:
                local_path.unlink(missing_ok=True)
                continue
            artifacts.append(OutboundArtifact(path=local_path, filename=filename))
        return artifacts

    async def _run_ssh_command(self, command: str, *, timeout_s: int) -> dict[str, Any]:
        await self.ensure_master()
        cmd = self._ssh_base_args() + [command]
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s + self.ssh.connect_timeout_s)
        except asyncio.TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            await proc.communicate()
            logger.warning('remote.exec.timeout timeout_s=%s', timeout_s)
            return {'ok': False, 'returncode': -9, 'stdout': '', 'stderr': f'Timed out after {timeout_s}s'}
        result = {
            'ok': proc.returncode == 0,
            'returncode': proc.returncode,
            'stdout': stdout_b.decode('utf-8', errors='replace')[: self.ssh.max_stdout_chars],
            'stderr': stderr_b.decode('utf-8', errors='replace')[: self.ssh.max_stderr_chars],
        }
        level = logger.info if result['ok'] else logger.warning
        level('remote.exec.done rc=%s stdout=%s stderr=%s', result['returncode'], len(result['stdout']), len(result['stderr']))
        return result

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
    def _scope_to_path(paths: RemoteSessionPaths, scope: str) -> str:
        if scope == 'inputs':
            return paths.inputs
        if scope == 'workspace':
            return paths.root
        return paths.outputs

    @staticmethod
    def _validate_remote_path(paths: RemoteSessionPaths, remote_path: str) -> str:
        normalized = posixpath.normpath(remote_path.strip())
        allowed_roots = (
            posixpath.normpath(paths.root),
            posixpath.normpath(paths.inputs),
            posixpath.normpath(paths.outputs),
        )
        if normalized in allowed_roots:
            return normalized
        allowed_prefixes = tuple(root.rstrip('/') + '/' for root in allowed_roots)
        if not normalized.startswith(allowed_prefixes):
            raise RuntimeError('Requested remote path is outside the session workspace')
        return normalized


def shq(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"
