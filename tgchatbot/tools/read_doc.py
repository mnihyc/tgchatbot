"""On-demand inspection of a file in the configured remote workspace."""
from __future__ import annotations

from tgchatbot.core.token_estimator import TokenEstimator
from tgchatbot.domain.models import MessagePart, PartKind, ToolResult
from tgchatbot.tools.base import ToolContext, ToolSpec


class ReadDocTool:
    def __init__(self, config, remote):
        self.config, self.remote = config, remote
        self.spec = ToolSpec(name='read_doc',
            description='Read one file in this chat\'s remote workspace in the requested format. Optionally select text lines or PDF pages. Returns contents for inspection.',
            parameters_schema={'type': 'object', 'properties': {
                'path': {'type': 'string', 'description': 'Path relative to the session directory, or an absolute path within it.'},
                'format': {'type': 'string', 'enum': ['text', 'image', 'pdf']},
                'start': {'type': 'integer', 'description': 'First line (text) or page (PDF), one-based and inclusive.'},
                'end': {'type': 'integer', 'description': 'Last line or page, inclusive. Omit the range to read the complete file.'},
            }, 'required': ['path', 'format'], 'additionalProperties': False}, runner=self)

    async def run(self, args: dict, ctx: ToolContext) -> ToolResult:
        output, parts = {}, []
        try:
            if not self.remote.enabled:
                raise ValueError('Remote workspace is not configured')
            path, fmt = str(args.get('path') or '').strip(), args.get('format')
            if not path:
                raise ValueError('Specify a workspace file path')
            if fmt not in {'text', 'image', 'pdf', 'audio', 'video'}:
                raise ValueError('Unsupported file format')
            if fmt in {'audio', 'video'}:
                raise ValueError('The selected API cannot receive audio/video in tool results')
            if fmt in {'image', 'pdf'} and not ctx.tool_images:
                raise ValueError('The selected API cannot receive images in tool results')
            tokens = ctx.evidence_tokens if ctx.evidence_tokens is not None else self.config.context.compact_trigger_tokens
            limits = {'tokens': max(0, tokens - TokenEstimator.MESSAGE_OVERHEAD),
                'images': ctx.evidence_images, 'bytes': self.config.ssh_exec.max_output_file_bytes,
                'chars_per_token': TokenEstimator.TEXT_CHARS_PER_TOKEN,
                'image_tokens': TokenEstimator.IMAGE_TOKENS, 'part_overhead': TokenEstimator.PART_OVERHEAD,
                'pdf_scale': self.config.read_doc.pdf_scale, 'max_image_pixels': self.config.read_doc.max_image_pixels}
            result = await self.remote.inspect_file(session_id=ctx.session_id, path=path,
                format=fmt, start=args.get('start'), end=args.get('end'), limits=limits)
            output = {key: value for key, value in result.items() if key != 'parts'}
            output.update(path=path, format=fmt)
            if result.get('ok'):
                parts = [MessagePart(kind=PartKind(part['kind']), text=part.get('text'),
                    filename=part.get('filename'), mime_type=part.get('mime_type'),
                    data_b64=part.get('data_b64'), remote_sync=False, origin='file_read')
                    for part in result['parts']]
        except Exception as exc:
            output, parts = {'ok': False, 'error': str(exc)}, []
        return ToolResult(call_id='', name=self.spec.name, output=output, evidence_parts=parts)
