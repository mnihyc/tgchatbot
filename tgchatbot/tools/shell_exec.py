from __future__ import annotations

import logging
from typing import Any

from tgchatbot.config import AppConfig
from tgchatbot.domain.models import ToolResult
from tgchatbot.tools.base import ToolContext, ToolSpec
from tgchatbot.tools.remote_workspace import RemoteWorkspaceClient

logger = logging.getLogger(__name__)


class ShellExecTool:
    def __init__(self, config: AppConfig, remote: RemoteWorkspaceClient) -> None:
        self.config = config
        self.remote = remote
        self.spec = ToolSpec(
            name='shell_exec',
            description=(
                'Run POSIX sh commands in this chat\'s persistent remote workspace for file inspection and command-line tasks. '
                'The working directory is the session root unless cwd_subdir is supplied. '
                'Successfully mirrored uploads are in TGCHATBOT_INPUT_DIR; TGCHATBOT_OUTPUT_DIR holds outputs, '
                'and TGCHATBOT_SESSION_DIR is the root. Files may change after edits. '
                'Inspect stdout, stderr and returncode; use file_send for files the user should receive.'
            ),
            parameters_schema={
                'type': 'object',
                'properties': {
                    'command': {
                        'type': 'string',
                        'description': 'Shell command to run inside the remote session workspace.',
                    },
                    'timeout_s': {
                        'type': 'integer',
                        'minimum': 1,
                        'maximum': self.config.ssh_exec.max_tool_timeout_s,
                        'description': 'Execution timeout in seconds. Use null to fall back to the operator default timeout.',
                    },
                    'cwd_subdir': {
                        'type': 'string',
                        'description': 'Optional subdirectory under the remote session root to use as the working directory. Use null for the session root.',
                    },
                },
                'required': ['command'],
                'additionalProperties': False,
            },
            runner=self,
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            return await self._run_impl(args, ctx)
        except Exception as exc:
            logger.exception('shell_exec failed')
            return ToolResult(call_id='', name=self.spec.name, output={'ok': False, 'error': f'{exc.__class__.__name__}: {exc}'})

    async def _run_impl(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        command = str(args.get('command', '')).strip()
        if not command:
            return ToolResult(call_id='', name=self.spec.name, output={'ok': False, 'error': 'Empty command'})
        timeout_raw = args.get('timeout_s')
        timeout_s = int(timeout_raw) if timeout_raw is not None else self.config.ssh_exec.default_timeout_s
        timeout_s = max(1, min(timeout_s, self.config.ssh_exec.max_tool_timeout_s))
        cwd_raw = args.get('cwd_subdir')
        cwd_subdir = str(cwd_raw).strip() if cwd_raw is not None else ''
        cwd_subdir = cwd_subdir or None
        result = await self.remote.run_shell(session_id=ctx.session_id, command=command, timeout_s=timeout_s, cwd_subdir=cwd_subdir)
        return ToolResult(
            call_id='',
            name=self.spec.name,
            output={
                'ok': result['ok'],
                'returncode': result['returncode'],
                'stdout': result['stdout'],
                'stderr': result['stderr'],
            },
        )
