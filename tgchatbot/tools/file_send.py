from __future__ import annotations

import logging
from typing import Any

from tgchatbot.config import AppConfig
from tgchatbot.domain.models import ToolResult
from tgchatbot.tools.base import ToolContext, ToolSpec
from tgchatbot.tools.remote_workspace import RemoteWorkspaceClient

logger = logging.getLogger(__name__)

_ALLOWED_SCOPES = {'outputs', 'inputs', 'workspace'}


class FileSendTool:
    def __init__(self, config: AppConfig, remote: RemoteWorkspaceClient) -> None:
        self.config = config
        self.remote = remote
        self.spec = ToolSpec(
            name='file_send',
            description=(
                'Prepare selected workspace files for delivery to this chat. '
                'Paths resolve within the requested inputs, outputs or workspace scope; known workspace absolute paths are accepted. '
                'This returns file and delivery status, not file contents for analysis. '
                'Wait for a confirmed delivery result before saying the files were sent.'
            ),
            parameters_schema={
                'type': 'object',
                'properties': {
                    'scope': {'type': 'string', 'enum': ['outputs', 'inputs', 'workspace']},
                    'paths': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1,
                              'maxItems': config.ssh_exec.max_output_files},
                },
                'required': ['scope', 'paths'],
                'additionalProperties': False,
            },
            runner=self,
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            scope = self._normalize_scope(args.get('scope'))
            paths = self._normalize_paths(args.get('paths'))
            artifacts = await self.remote.fetch_files(
                session_id=ctx.session_id,
                remote_paths=paths,
                scope=scope,
            )
            output = {
                'ok': bool(artifacts),
                'scope': scope,
                'requested_paths': len(paths),
                'prepared_files': [artifact.filename for artifact in artifacts],
                'delivery_state': 'pending' if artifacts else 'unavailable',
                'count': len(artifacts),
            }
            if not artifacts:
                output['ok'] = False
                output['error'] = 'No matching files were available to send'
            return ToolResult(call_id='', name=self.spec.name, output=output, artifacts=artifacts)
        except Exception as exc:
            logger.exception('file_send failed scope=%s', args.get('scope', 'outputs'))
            return ToolResult(call_id='', name=self.spec.name, output={'ok': False, 'error': f'{exc.__class__.__name__}: {exc}'})

    @staticmethod
    def _normalize_scope(value: Any) -> str:
        scope = str(value or 'outputs').strip().lower()
        if scope not in _ALLOWED_SCOPES:
            raise RuntimeError(f'Unsupported file scope: {scope}')
        return scope

    @staticmethod
    def _normalize_paths(value: Any) -> list[str]:
        if not isinstance(value, list):
            raise RuntimeError('paths must be an array of remote file paths')
        seen: set[str] = set()
        normalized: list[str] = []
        for raw in value:
            item = str(raw).strip()
            if not item or item in seen:
                continue
            seen.add(item)
            normalized.append(item)
        if not normalized:
            raise ValueError('Select at least one remote file path')
        return normalized
