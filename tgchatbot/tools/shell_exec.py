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
                'Run POSIX sh in a new process in this chat\'s remote workspace (TGCHATBOT_SESSION_DIR). '
                'Files persist; shell state does not. User-uploaded attachments are in attachments/YYYY-MM-DD/; create dated or custom folders. '
                'Each call executes again. Inspect stdout, stderr and returncode; file_send delivers files.'
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
                        'description': 'Time to wait for execution, in seconds; defaults to the configured timeout. '
                                       'A timeout may leave completion unconfirmed.',
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
        result = await self.remote.run_shell(session_id=ctx.session_id, command=command, timeout_s=timeout_s)
        return ToolResult(
            call_id='',
            name=self.spec.name,
            output={
                'ok': result['ok'],
                'returncode': result['returncode'],
                'stdout': result['stdout'],
                'stderr': result['stderr'],
                **{key: result[key] for key in ('stdout_truncated', 'stderr_truncated') if result.get(key)},
                **{key: result[key] for key in ('outcome', 'error') if key in result},
            },
        )
