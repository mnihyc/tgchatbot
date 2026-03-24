from __future__ import annotations

import logging
from typing import Any

from tgchatbot.config import AppConfig
from tgchatbot.domain.models import ToolResult
from tgchatbot.tools.base import ToolContext, ToolSpec
from tgchatbot.tools.remote_workspace import RemoteWorkspaceClient

logger = logging.getLogger(__name__)


class PythonExecTool:
    def __init__(self, config: AppConfig, remote: RemoteWorkspaceClient) -> None:
        self.config = config
        self.remote = remote
        self.spec = ToolSpec(
            name='python_exec',
            description=(
                'Run Python 3 code inside the persistent remote container for this chat session. '
                'Use it for analysis, transformations, scraping local files, and structured computation. '
                'Input files from the current chat are mirrored into the remote input directory automatically, and results should be summarized through stdout or stderr.'
            ),
            parameters_schema={
                'type': 'object',
                'properties': {
                    'code': {
                        'type': 'string',
                        'description': 'Python source code to execute in the remote session workspace.',
                    },
                    'timeout_s': {
                        'type': 'integer',
                        'minimum': 1,
                        'maximum': self.config.ssh_exec.max_tool_timeout_s,
                        'description': 'Execution timeout in seconds. Use null to fall back to the operator default timeout.',
                    },
                },
                'required': ['code'],
                'additionalProperties': False,
            },
            runner=self,
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            return await self._run_impl(args, ctx)
        except Exception as exc:
            logger.exception('python_exec failed')
            return ToolResult(call_id='', name=self.spec.name, output={'ok': False, 'error': f'{exc.__class__.__name__}: {exc}'})

    async def _run_impl(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        code = str(args.get('code', ''))
        timeout_raw = args.get('timeout_s')
        timeout_s = int(timeout_raw) if timeout_raw is not None else self.config.ssh_exec.default_timeout_s
        timeout_s = max(1, min(timeout_s, self.config.ssh_exec.max_tool_timeout_s))
        if not code.strip():
            return ToolResult(call_id='', name=self.spec.name, output={'ok': False, 'error': 'Empty code'})
        result = await self.remote.run_python(
            session_id=ctx.session_id,
            code=code,
            timeout_s=timeout_s,
        )
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
