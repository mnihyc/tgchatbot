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
                'Add workspace files to your reply for automatic delivery after your reply text. Each call creates new upload requests, '
                'even for previously requested paths. Results give each file\'s status: queued means accepted for automatic delivery. '
                'Use paths relative to the session directory, or absolute paths within it.'
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
            results = await self.remote.fetch_files(
                session_id=ctx.session_id,
                remote_paths=paths,
            )
            artifacts = [result.artifact for result in results if result.artifact is not None]
            output = {
                'ok': bool(artifacts),
                'status': ('queued' if len(artifacts) == len(results) else 'partial') if artifacts else 'failed',
                'delivery_timing': 'after_text',
                'files': [{'workspace_path': result.workspace_path,
                    'status': 'queued' if result.artifact is not None else 'failed',
                    **({'error': result.error} if result.error else {})} for result in results],
            }
            if not artifacts:
                output['error'] = 'No files were accepted for delivery'
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
