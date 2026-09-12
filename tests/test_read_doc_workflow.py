"""Read actual workspace files through the tool, persistence and provider wire."""
from __future__ import annotations

import asyncio
import base64
from dataclasses import replace
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
from PIL import Image

from tests.business_helpers import BusinessTestCase
from tgchatbot.core.memory import MemoryService
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import ChatMode, ConversationMessage, MessagePart, MessageRole, PartKind
from tgchatbot.providers.gemini import GeminiProvider
from tgchatbot.stickers.config import StickerConfig
from tgchatbot.storage.previews import PreviewCache
from tgchatbot.tools.base import ToolContext
from tgchatbot.tools.read_doc import ReadDocTool
from tgchatbot.tools.registry import ToolRegistry
from tgchatbot.tools.remote_workspace import RemoteWorkspaceClient


class ProcessWorkspace(RemoteWorkspaceClient):
    """Replace SSH authentication/transport only; execute the actual remote program."""
    def __init__(self, config):
        super().__init__(config)
        self.commands = []

    async def ensure_master(self):
        pass

    async def _run_ssh_command(self, command, *, timeout_s, stdout_limit=None):
        self.commands.append(command)
        env = dict(os.environ)
        env['PATH'] = str(Path(__file__).resolve().parents[1] / '.venv' / 'bin') + os.pathsep + env['PATH']
        proc = await asyncio.create_subprocess_exec('bash', '-c', command,
            cwd=self.config.temp_dir, env=env, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr, _ = await asyncio.wait_for(asyncio.gather(
                self._read_output(proc.stdout, self.ssh.max_stdout_chars if stdout_limit is None else stdout_limit),
                self._read_output(proc.stderr, self.ssh.max_stderr_chars), proc.wait()), timeout=timeout_s)
        except BaseException:
            if proc.returncode is None:
                proc.kill()
                await proc.wait()
            raise
        return {'ok': proc.returncode == 0, 'returncode': proc.returncode, 'stdout': stdout, 'stderr': stderr}


def write_pdf(path):
    """Two ordinary text pages; no extra PDF writer dependency is needed."""
    streams = [f'BT /F1 12 Tf 20 100 Td ({text}) Tj ET'.encode()
        for text in ('First page: departure at noon.', 'Second page: bring the blue ticket.')]
    objects = [b'<< /Type /Catalog /Pages 2 0 R >>',
        b'<< /Type /Pages /Kids [3 0 R 4 0 R] /Count 2 >>',
        *[f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 240 160] '
          f'/Resources << /Font << /F1 5 0 R >> >> /Contents {number} 0 R >>'.encode()
          for number in (6, 7)],
        b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>',
        *[f'<< /Length {len(stream)} >>\nstream\n'.encode() + stream + b'\nendstream' for stream in streams]]
    content, offsets = bytearray(b'%PDF-1.4\n'), [0]
    for number, obj in enumerate(objects, 1):
        offsets.append(len(content))
        content.extend(f'{number} 0 obj\n'.encode() + obj + b'\nendobj\n')
    start = len(content)
    content.extend(f'xref\n0 {len(offsets)}\n0000000000 65535 f \n'.encode())
    for offset in offsets[1:]:
        content.extend(f'{offset:010} 00000 n \n'.encode())
    content.extend(f'trailer\n<< /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n'.encode())
    path.write_bytes(content)


class ReadDocWorkflows(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.config = replace(self.config, ssh_exec=replace(self.config.ssh_exec,
            enabled=True, host='test-workspace', workdir=str(self.path / 'remote')))
        self.remote = ProcessWorkspace(self.config)
        self.remote.fetch_files = AsyncMock(side_effect=AssertionError('Do not download original files to inspect them'))
        self.paths = await self.remote.ensure_session_dirs(self.session)
        self.inputs = Path(self.paths.inputs)
        self.outputs = Path(self.paths.outputs)
        self.inputs.joinpath('notes.txt').write_text('\ufeffFirst line.\n第二行：带上蓝色车票。\nLast line.\n', encoding='utf-8')
        self.outputs.joinpath('notes.txt').write_text('Different output file.\n', encoding='utf-8')
        self.inputs.joinpath('odd \' $ name.txt').write_text('Literal filename selected.\n', encoding='utf-8')
        with Image.new('RGB', (32, 16), 'red') as image:
            image.save(self.inputs / 'picture.png')
        with Image.new('RGB', (8, 8), 'red') as first, Image.new('RGB', (8, 8), 'blue') as second:
            first.save(self.inputs / 'animated.gif', save_all=True, append_images=[second], duration=100, loop=0)
        self.inputs.joinpath('empty.txt').write_text('', encoding='utf-8')
        write_pdf(self.inputs / 'schedule.pdf')
        self.tool = ReadDocTool(self.config, self.remote)
        self.catalog = SimpleNamespace(stats=lambda: {'stickers': 0}, aensure_loaded=AsyncMock(), config=StickerConfig())
        self.registry = ToolRegistry(self.config, self.remote, self.catalog)
        self.memory = MemoryService(self.store, SimpleNamespace(enabled=False))
        self.wire = []

    async def read(self, path, format, **selection):
        return await self.tool.run({'scope': 'inputs', 'path': path, 'format': format, **selection},
            ToolContext(self.session, 'Participant', evidence_tokens=20000, evidence_images=4))

    async def gemini_runtime(self, scripts):
        provider = GeminiProvider(replace(self.config.gemini, api_key='synthetic-key'))
        await provider.aclose()
        async def respond(request):
            self.wire.append(json.loads(request.content))
            return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP',
                'content': {'role': 'model', 'parts': scripts.pop(0)}}]})
        provider._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(provider.aclose)
        await self.settings(provider='gemini', model='gemini-3.8-flash', mode=ChatMode.ASSIST,
            compact_trigger_tokens=100000, max_input_images=4, compact_target_images=1,
            max_interaction_rounds=2)
        runtime = AgentRuntime(config=self.config, store=self.store, tool_registry=self.registry,
            providers={'gemini': provider}, memory=self.memory, preview_cache=self.preview_cache)
        return runtime, provider

    @staticmethod
    def call(path, format, call_id, **selection):
        return {'functionCall': {'name': 'read_doc', 'id': call_id,
            'args': {'scope': 'inputs', 'path': path, 'format': format, **selection}},
            'thoughtSignature': base64.b64encode(b'synthetic signature').decode()}

    def results(self, wire):
        return [part['functionResponse'] for content in wire['contents'] for part in content.get('parts', [])
            if part.get('functionResponse', {}).get('name') == 'read_doc']

    async def test_selected_text_image_and_pdf_are_parsed_in_workspace_without_fetching_originals(self):
        # These parent-process parsers must not run. Child processes have their
        # own imports and execute the actual reader over the remote fixture tree.
        with patch('PIL.Image.open', side_effect=AssertionError('Unexpected local image parsing')), \
             patch('pypdfium2.PdfDocument', side_effect=AssertionError('Unexpected local PDF parsing')):
            text = await self.read('notes.txt', 'text', start=2, end=2)
            picture = await self.read('picture.png', 'image')
            pdf = await self.read('schedule.pdf', 'pdf', start=2, end=2)
        self.assertEqual(text.output['selection'], {'start': 2, 'end': 2, 'unit': 'lines'})
        self.assertEqual(text.evidence_parts[0].text, '第二行：带上蓝色车票。\n')
        self.assertTrue(picture.output['ok'], picture.output)
        with Image.open(io.BytesIO(base64.b64decode(picture.evidence_parts[0].data_b64))) as image:
            self.assertEqual(image.size, (32, 16))
        self.assertTrue(pdf.output['ok'], pdf.output)
        self.assertEqual(pdf.output['selection'], {'start': 2, 'end': 2, 'unit': 'pages'})
        self.assertEqual([part.kind for part in pdf.evidence_parts], [PartKind.TEXT, PartKind.IMAGE])
        self.assertIn('Second page: bring the blue ticket.', pdf.evidence_parts[0].text)
        self.assertNotIn('departure', pdf.evidence_parts[0].text)
        full = await self.read('schedule.pdf', 'pdf')
        self.assertEqual(full.output['selection'], {'start': 1, 'end': 2, 'unit': 'pages'})
        self.assertEqual(sum(part.kind == PartKind.IMAGE for part in full.evidence_parts), 2)
        animated = await self.read('animated.gif', 'image')
        self.assertTrue(animated.output['ok'], animated.output)
        self.assertEqual(animated.output['selection'], {'frame': 1, 'total_frames': 2})
        self.assertEqual(len(animated.evidence_parts), 1)
        with Image.open(io.BytesIO(base64.b64decode(animated.evidence_parts[0].data_b64))) as image:
            red, _, blue = image.convert('RGB').getpixel((0, 0))
            self.assertGreater(red, blue, 'The result explicitly reports the first frame only.')
        empty = await self.read('empty.txt', 'text')
        self.assertTrue(empty.output['ok'], empty.output)
        self.assertEqual(empty.evidence_parts[0].text, '')
        self.assertFalse((await self.read('empty.txt', 'text', start=1, end=1)).output['ok'])
        self.assertIn('First line.', (await self.read('notes.txt', 'text')).evidence_parts[0].text)
        odd = await self.read('odd \' $ name.txt', 'text')
        self.assertEqual(odd.evidence_parts[0].text, 'Literal filename selected.\n')
        for scope, name, expected in [('outputs', 'notes.txt', 'Different output file.\n'),
                                     ('workspace', 'inputs/notes.txt', 'First line.')]:
            result = await self.tool.run({'scope': scope, 'path': name, 'format': 'text'},
                ToolContext(self.session, 'Participant'))
            self.assertTrue(result.output['ok'], result.output)
            self.assertIn(expected, result.evidence_parts[0].text)
        self.remote.fetch_files.assert_not_awaited()
        self.assertFalse(any(result.artifacts for result in (text, picture, pdf, full)))

    async def test_missing_corrupt_outside_and_oversized_selections_return_no_partial_evidence(self):
        outside = self.path / 'other-session.txt'
        outside.write_text('Outside session secret.', encoding='utf-8')
        self.inputs.joinpath('escape.txt').symlink_to(outside)
        self.inputs.joinpath('corrupt.pdf').write_bytes(b'not a PDF')
        self.inputs.joinpath('huge.txt').write_text('Unselected prefix\n' + 'x' * 100000 + '\n', encoding='utf-8')
        for path, fmt, selection in [
            ('missing.txt', 'text', {}), ('escape.txt', 'text', {}),
            (str(outside), 'text', {}), ('../../other-session.txt', 'text', {}),
            ('corrupt.pdf', 'pdf', {}), ('notes.txt', 'image', {}),
            ('notes.txt', 'text', {'start': 4}), ('notes.txt', 'text', {'end': 9}),
            ('schedule.pdf', 'pdf', {'start': 2, 'end': 1}),
            ('schedule.pdf', 'pdf', {'end': 3}), ('huge.txt', 'text', {})]:
            with self.subTest(path=path, format=fmt, selection=selection):
                result = await self.read(path, fmt, **selection)
                self.assertFalse(result.output['ok'], result.output)
                self.assertTrue(result.output['error'])
                self.assertEqual(result.evidence_parts, [])
                self.assertNotIn('Outside session secret.', json.dumps(result.output))
        selected = await self.read('huge.txt', 'text', start=1, end=1)
        self.assertEqual(selected.evidence_parts[0].text, 'Unselected prefix\n')

    async def test_unsupported_recordings_and_text_only_visual_routes_do_no_remote_work(self):
        before = len(self.remote.commands)
        for fmt, images in [('audio', True), ('video', True), ('image', False), ('pdf', False)]:
            with self.subTest(format=fmt):
                result = await self.tool.run({'scope': 'inputs', 'path': 'anything', 'format': fmt},
                    ToolContext(self.session, 'Participant', tool_images=images))
                self.assertFalse(result.output['ok'])
                self.assertEqual(result.evidence_parts, [])
        self.assertEqual(len(self.remote.commands), before)
        self.remote.fetch_files.assert_not_awaited()

    async def test_workspace_tool_requires_configuration_and_assist_permissions(self):
        names = lambda registry, allow: {spec.name for spec in registry.list_tools(
            allow_python_exec=allow, allow_stickers=False)}
        self.assertIn('read_doc', names(self.registry, True))
        self.assertNotIn('read_doc', names(self.registry, False))
        disabled = ToolRegistry(self.config, SimpleNamespace(enabled=False), self.catalog)
        self.assertNotIn('read_doc', names(disabled, True))
        runtime, _ = await self.gemini_runtime([[self.call('notes.txt', 'text', 'unavailable')],
            [{'text': 'Workspace inspection is unavailable in this mode.'}]])
        await self.settings(mode=ChatMode.CHAT)
        before = len(self.remote.commands)
        await runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Check my notes.'))
        self.assertEqual(len(self.remote.commands), before)
        self.assertFalse(self.results(self.wire[-1])[0]['response']['result']['ok'])

    async def test_multimodal_results_survive_original_changes_cold_reconstruction_compaction_and_resets(self):
        runtime, provider = await self.gemini_runtime([[
            self.call('notes.txt', 'text', 'text', start=2, end=2),
            self.call('schedule.pdf', 'pdf', 'pdf', start=2, end=2)],
            [{'text': 'The note and the second page both specify the blue ticket.'}]])
        result = await runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Compare the note with page two.'))
        self.assertEqual(result.artifacts, [])
        responses = self.results(self.wire[-1])
        self.assertEqual([response['id'] for response in responses], ['text', 'pdf'])
        self.assertTrue(all(response['response']['result']['ok'] for response in responses))
        pixels = [part['inlineData']['data'] for response in responses for part in response.get('parts', [])]
        self.assertEqual(len(pixels), 1)
        self.assertIn('Second page: bring the blue ticket.', json.dumps(responses))
        self.assertIn('第二行：带上蓝色车票。', json.dumps(responses, ensure_ascii=False))
        self.inputs.joinpath('notes.txt').write_text('Changed after inspection.', encoding='utf-8')
        self.inputs.joinpath('schedule.pdf').rename(self.path / 'no-longer-at-original-path.pdf')

        settings = await self.settings()
        live = await runtime._get_live_state(self.session)
        reader = await self.new_store()
        cache = PreviewCache(reader, max_bytes=0)
        self.addCleanup(cache.close)
        cold = AgentRuntime(config=self.config, store=reader, tool_registry=self.registry,
            providers={'gemini': provider}, preview_cache=cache)
        before = len(self.remote.commands)
        rebuilt = await cold._get_live_state(self.session)
        self.assertEqual(rebuilt.raw_messages, live.raw_messages)
        async def history(agent, state, previews):
            projected = agent._build_provider_history(state, settings=settings, provider_name='gemini')
            return await previews.materialize_many(self.session, projected, vision=True)
        self.assertEqual(await history(runtime, live, self.preview_cache), await history(cold, rebuilt, cache))
        self.assertEqual(len(self.remote.commands), before, 'Reconstruction must read the database, not the workspace.')
        tool_rows = [row for row in live.raw_messages if row.message.name == 'read_doc']
        self.assertEqual([row.message.metadata['tool_phase'] for row in tool_rows], ['call', 'call', 'result', 'result'])
        self.assertTrue(all(row.message.role == MessageRole.TOOL for row in tool_rows))
        pdf_row = next(row for row in tool_rows if row.message.metadata.get('tool_phase') == 'result'
            and row.message.metadata['tool_payload']['call_id'] == 'pdf')
        images = await self.store.describe_message_images(self.session, [pdf_row.db_id])
        image_id = images[pdf_row.db_id][0]['image_id']
        self.assertEqual(await runtime._compact_oldest_images(session_id=self.session, settings=settings,
            state=live, target_images=0), 1)
        runtime.invalidate_session(self.session)
        retired = await runtime._get_live_state(self.session)
        retired_history = await history(runtime, retired, self.preview_cache)
        self.assertFalse(any(part.data_b64 for row in retired_history for part in row.parts))
        self.assertIn('Image compacted', str(retired_history))
        retained = await self.store.resolve_message_images(self.session, [pdf_row.db_id], [image_id])
        self.assertEqual(retained['image_results'][0]['status'], 'selected')
        self.assertEqual(next(part.data_b64 for part in retained['evidence_parts'] if part.kind == PartKind.IMAGE), pixels[0])
        await self.store.reset_context(self.session)
        retained = await self.store.resolve_message_images(self.session, [pdf_row.db_id], [image_id])
        self.assertEqual(retained['image_results'][0]['status'], 'selected')
        await self.store.reset_full(self.session, self.config.default_session_settings())
        denied = await self.store.resolve_message_images(self.session, [pdf_row.db_id], [image_id])
        self.assertEqual(denied['image_results'][0]['status'], 'unavailable')
        self.assertEqual(denied['evidence_parts'], [])
        audit = await self.store.list_message_revisions(self.session, pdf_row.db_id)
        self.assertIn('Second page: bring the blue ticket.', audit[0]['body'])

    async def test_multiple_reads_share_image_and_text_allowances_without_silent_clipping(self):
        runtime, _ = await self.gemini_runtime([[
            self.call('picture.png', 'image', 'first'), self.call('schedule.pdf', 'pdf', 'second', start=2, end=2)],
            [{'text': 'The image was inspected; the PDF selection did not fit.'}]])
        await self.settings(max_input_images=1)
        await runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Inspect these files.'))
        responses = self.results(self.wire[-1])
        self.assertEqual([response['response']['result']['ok'] for response in responses], [True, False])
        self.assertEqual(sum(len(response.get('parts', [])) for response in responses), 1)

        await self.store.reset_context(self.session)
        self.wire.clear()
        self.inputs.joinpath('long.txt').write_text('a' * 20000, encoding='utf-8')
        runtime, _ = await self.gemini_runtime([[
            self.call('long.txt', 'text', 'first'), self.call('long.txt', 'text', 'second')],
            [{'text': 'One complete selection fit this request.'}]])
        # An artificial context size makes two individually fitting selections
        # compete in one real model turn, without a large-message experiment.
        await self.settings(compact_trigger_tokens=8000, compact_target_tokens=6000)
        await runtime.run_turn(session_id=self.session, user_display_name='Participant',
            incoming_message=ConversationMessage.user_text('Read both requested copies.'))
        responses = self.results(self.wire[-1])
        self.assertEqual([response['response']['result']['ok'] for response in responses], [True, False])
        self.assertIn('a' * 20000, json.dumps(responses[0]))
        self.assertNotIn('a' * 100, json.dumps(responses[1]))
        self.remote.fetch_files.assert_not_awaited()

    async def test_file_and_memory_reads_share_the_image_allowance_in_either_order(self):
        source = await self.store.append_message(self.session, ConversationMessage(MessageRole.USER,
            [MessagePart(PartKind.TEXT, text='The previously shared red ticket.'),
             MessagePart(PartKind.IMAGE, mime_type='image/png',
                data_b64=base64.b64encode(self.inputs.joinpath('picture.png').read_bytes()).decode())],
            metadata={'actor_id': 'telegram:user:11', 'actor_kind': 'user'}))
        descriptions = await self.store.describe_message_images(self.session, [source.db_id])
        image_id = descriptions[source.db_id][0]['image_id']
        file_call = self.call('picture.png', 'image', 'file')
        memory_call = {'functionCall': {'name': 'memory_read', 'id': 'memory',
            'args': {'message_ids': [source.db_id], 'image_ids': [image_id]}},
            'thoughtSignature': base64.b64encode(b'synthetic signature').decode()}
        for first in ('file', 'memory'):
            with self.subTest(first=first):
                await self.store.reset_context(self.session)
                self.wire.clear()
                calls = [file_call, memory_call] if first == 'file' else [memory_call, file_call]
                runtime, _ = await self.gemini_runtime([calls, [{'text': 'One visual selection fit.'}]])
                await self.settings(max_input_images=1)
                await runtime.run_turn(session_id=self.session, user_display_name='Participant',
                    incoming_message=ConversationMessage.user_text('Inspect the stored and workspace pictures.'))
                responses = [part['functionResponse'] for content in self.wire[-1]['contents']
                    for part in content.get('parts', []) if 'functionResponse' in part]
                by_name = {response['name']: response for response in responses}
                self.assertEqual(sum(len(response.get('parts', [])) for response in responses), 1)
                self.assertEqual(by_name['read_doc']['response']['result']['ok'], first == 'file')
                status = by_name['memory_read']['response']['result']['image_results'][0]['status']
                self.assertEqual(status, 'opened' if first == 'memory' else 'omitted')
