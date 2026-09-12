from __future__ import annotations

import logging
from typing import Any

from tgchatbot.config import AppConfig
from tgchatbot.domain.models import ToolResult
from tgchatbot.tools.base import ToolContext, ToolSpec
from tgchatbot.tools.remote_workspace import RemoteWorkspaceClient

logger = logging.getLogger(__name__)

class FileSendTool:
    def __init__(self, config: AppConfig, remote: RemoteWorkspaceClient) -> None:
        self.config = config
        self.remote = remote
        self.spec = ToolSpec(
            name='file_send',
            description=(
                'Prepare selected workspace files for delivery to this chat. '
                'Paths are relative to the session directory; absolute paths within it are also accepted. '
                'Returned workspace_path values are relative to that directory. '
                'This returns file and delivery status, not file contents for analysis. '
                'Wait for a confirmed delivery result before saying the files were sent.'
            ),
            parameters_schema={
                'type': 'object',
                'properties': {
                    'paths': {'type': 'array', 'items': {'type': 'string'}, 'minItems': 1,
                              'maxItems': config.ssh_exec.max_output_files},
                },
                'required': ['paths'],
                'additionalProperties': False,
            },
            runner=self,
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            paths = self._normalize_paths(args.get('paths'))
            artifacts = await self.remote.fetch_files(
                session_id=ctx.session_id,
                remote_paths=paths,
            )
            output = {
                'ok': bool(artifacts),
                'requested_paths': len(paths),
                'prepared_files': [dict(filename=artifact.filename,
                    **({'workspace_path': artifact.workspace_path} if artifact.workspace_path else {}))
                    for artifact in artifacts],
                'delivery_state': 'pending' if artifacts else 'unavailable',
                'count': len(artifacts),
            }
            if not artifacts:
                output['ok'] = False
                output['error'] = 'No matching files were available to send'
            return ToolResult(call_id='', name=self.spec.name, output=output, artifacts=artifacts)
        except OSError as exc:
            logger.exception('file_send local transfer failed')
            return ToolResult(call_id='', name=self.spec.name,
                output={'ok': False, 'error': f'Local file transfer failed: {exc.__class__.__name__}'})
        except Exception as exc:
            logger.exception('file_send failed')
            return ToolResult(call_id='', name=self.spec.name, output={'ok': False, 'error': f'{exc.__class__.__name__}: {exc}'})

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
