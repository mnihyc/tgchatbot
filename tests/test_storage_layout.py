from __future__ import annotations

import base64
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tests.business_helpers import BusinessTestCase, FixtureTools, ScriptedProvider
from tgchatbot.config import load_config
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind, ProviderResponse
from tgchatbot.storage.artifacts import ArtifactStore
from tgchatbot.storage.presets import PresetStore
from tgchatbot.storage.previews import PreviewCache
from tgchatbot.tools.remote_workspace import RemoteSyncResult, RemoteWorkspaceClient
from tgchatbot.transports.telegram_adapter import TelegramBotApp


class StorageLayoutTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.root = self.path

    def layout_config(self, **environment):
        with patch.dict(os.environ, {
            'APP_DATA_DIR': str(self.root / 'data'),
            'TGBOT_TOKEN': '123456:mock-token',
            'OPENAI_API_KEY': 'mock-key',
            'DATABASE_URL': self.test_dsn,
            **environment,
        }, clear=True):
            return load_config()

    async def test_lowered_native_cache_setting_discards_replay_but_retains_originals(self):
        root = self.root / 'provider-replay'
        old = ArtifactStore(root, max_bytes=100)
        replay = old.save_bytes(chat_id=self.session, filename='native.json', data=b'opaque replay')
        source = await self.store.append_message(self.session, ConversationMessage.user_text('Durable original'))
        config = self.layout_config(MEMORY_REPLAY_CACHE_BYTES='0')
        disabled = ArtifactStore(root, max_bytes=config.memory.replay_cache_bytes)
        self.assertFalse(replay.exists())
        with self.assertRaises(ValueError):
            disabled.save_bytes(chat_id=self.session, filename='next.json', data=b'next replay')
        retained = await self.store.read_messages(self.session, [source.db_id])
        self.assertEqual(retained[0].message.parts[0].text, 'Durable original')

    async def test_concurrent_intake_shares_replay_capacity_without_losing_originals(self):
        root = self.root / 'provider-replay'
        self.store.artifact_store = ArtifactStore(root, max_bytes=4000)
        sources = await asyncio.gather(*(self.store.append_message(self.session,
            ConversationMessage.user_text(f'Original {number}', metadata={
                'provider_native': {'provider': 'openai', 'items': [{'text': 'x' * 3000}]}}))
            for number in range(20)))
        self.assertLessEqual(sum(path.stat().st_size for path in root.rglob('*') if path.is_file()), 4000)
        retained = await self.store.read_messages(self.session, [source.db_id for source in sources], limit=len(sources))
        self.assertEqual({source.message.parts[0].text for source in retained},
                         {f'Original {number}' for number in range(20)})

    async def test_cleared_temp_can_be_recreated_without_losing_retained_state(self):
        # A separate context removes only disposable fixture state, simulating
        # clearing tmp while the bot is stopped. The outer fixture retains data.
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            config = self.layout_config(APP_TEMP_DIR=temporary)
            artifacts = ArtifactStore(config.artifact_dir)
            staged = artifacts.save_bytes(chat_id=self.session, filename='upload.bin', data=b'upload bytes')
            self.assertTrue(staged.is_relative_to(temporary))
            remote = RemoteWorkspaceClient(config)
            remote._control_path.write_bytes(b'fixture socket placeholder')
            store = await self.new_store()
            settings = config.default_session_settings()
            settings.system_prompt = 'Keep this conversation voice.'
            await store.save_session(self.session, settings)
            preview = base64.b64encode(b'inline preview bytes').decode('ascii')
            previews = PreviewCache(config.temp_dir / 'previews', max_bytes=32 * 1024 * 1024)
            self.addCleanup(previews.close)
            image_message = previews.externalize(ConversationMessage(MessageRole.USER, [
                MessagePart(PartKind.TEXT, text='Remember the image'),
                MessagePart(PartKind.IMAGE, mime_type='image/png', data_b64=preview, remote_sync=False),
            ]))
            await store.append_message(self.session, image_message)
            self.assertEqual(previews.materialize(image_message, vision=True).parts[1].data_b64, preview)
            preset = PresetStore(config.preset_dir).save_text('fixture', 'saved preset')
            retained = [preset, config.sticker_dir / 'fixture.webp', config.data_dir / 'ssh' / 'identity', config.data_dir / 'ssh' / 'known_hosts']
            for path in retained[1:]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture retained content')
            snapshots = {path: path.read_bytes() for path in retained}
        self.assertFalse(staged.exists())
        self.assertFalse(config.temp_dir.exists())
        await store.close()
        restarted = await self.new_store()
        self.assertEqual((await restarted.get_or_create_session(self.session, config.default_session_settings())).system_prompt, 'Keep this conversation voice.')
        history = await restarted.list_messages(self.session)
        self.assertIsNone(history[0].parts[1].data_b64)
        self.assertTrue(history[0].parts[1].preview_ref)
        for path, expected in snapshots.items():
            self.assertEqual(path.read_bytes(), expected)
        new_file = ArtifactStore(config.artifact_dir).save_bytes(chat_id=self.session, filename='next.bin', data=b'next upload')
        new_remote = RemoteWorkspaceClient(config)
        self.assertTrue(new_file.is_file())
        self.assertTrue(new_remote._control_dir.is_dir())
        provider = ScriptedProvider(responses=[ProviderResponse(final_text='continued')])
        runtime = AgentRuntime(config=config, store=restarted, tool_registry=FixtureTools(), providers={'openai': provider}, preview_cache=self.preview_cache)
        await runtime.run_turn(session_id=self.session, user_display_name='tester', incoming_message=ConversationMessage.user_text('Continue'))
        request_parts = [part for message in provider.requests[0]['messages'] for part in message.parts]
        self.assertTrue(any(part.kind == PartKind.TEXT and 'temporary preview expired or unavailable' in (part.text or '') for part in request_parts))
        self.assertFalse(any(part.kind == PartKind.IMAGE and part.data_b64 for part in request_parts))

    async def test_remote_upload_history_keeps_remote_locator_after_temp_clear(self):
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            config = self.layout_config(APP_TEMP_DIR=temporary, SSH_EXEC_HOST='remote.invalid')
            local = ArtifactStore(config.artifact_dir).save_bytes(chat_id=self.session, filename='report.txt', data=b'report')
            remote = RemoteWorkspaceClient(config)
            remote_path = remote.session_paths(self.session).inputs + '/' + local.name
            uploaded = {}

            async def sync_inputs(session_id, paths):
                for path in paths:
                    destination = remote.session_paths(session_id).inputs + '/' + path.name
                    uploaded[destination] = path.read_bytes()
                return RemoteSyncResult(list(uploaded), [])

            remote.sync_inputs = sync_inputs
            app = TelegramBotApp.__new__(TelegramBotApp)
            app.remote_workspace = remote
            parts = await app._sync_parts_to_remote(self.session, [MessagePart(PartKind.FILE, filename='report.txt', artifact_path=str(local), remote_sync=True)])
            self.assertEqual(uploaded, {remote_path: b'report'})
            self.assertFalse(local.exists())
            store = await self.new_store()
            await store.append_message(self.session, ConversationMessage(MessageRole.USER, parts))
        await store.close()
        stored = (await (await self.new_store()).list_messages(self.session))[0]
        attachment = next(part for part in stored.parts if part.kind == PartKind.FILE)
        self.assertEqual(attachment.artifact_path, remote_path)
        self.assertTrue(attachment.remote_sync)
        self.assertIn(remote_path, stored.parts[0].text)
        self.assertEqual(RemoteWorkspaceClient(config).session_paths(self.session), remote.session_paths(self.session))

    async def test_remote_fetch_stages_in_temp_artifacts(self):
        config = self.layout_config(SSH_EXEC_HOST='remote.invalid')
        remote = RemoteWorkspaceClient(config)
        paths = remote.session_paths(self.session)
        remote.ensure_session_dirs = AsyncMock(return_value=paths)

        async def scp(*arguments, **kwargs):
            # Only the process boundary is mocked: fetch path construction and
            # output validation use the real implementation.
            Path(arguments[-1]).write_bytes(b'fetched report')
            return SimpleNamespace(returncode=0, communicate=AsyncMock(return_value=(b'', b'')))

        with patch('tgchatbot.tools.remote_workspace.asyncio.create_subprocess_exec', side_effect=scp) as execute:
            artifacts = await remote.fetch_files(session_id=self.session, remote_paths=['report.txt'])
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].path, self.root / 'tmp' / 'bot' / 'artifacts' / self.session / 'remote_fetch' / 'report.txt')
        self.assertEqual(artifacts[0].path.read_bytes(), b'fetched report')
        self.assertIn('remote.invalid:' + paths.outputs + '/report.txt', execute.call_args.args)
        self.assertFalse((config.data_dir / 'artifacts').exists())

    async def test_unavailable_upload_retains_searchable_descriptor_without_original_bytes(self):
        for number, enabled in enumerate((True, False), 1):
            with self.subTest(ssh_enabled=enabled):
                config = self.layout_config(SSH_EXEC_HOST='remote.invalid', SSH_EXEC_ENABLED=str(enabled).lower())
                original = b'original PDF bytes that must not enter PostgreSQL'
                local = ArtifactStore(config.artifact_dir).save_bytes(chat_id=self.session,
                    filename=f'report{number}.pdf', data=original)
                remote = RemoteWorkspaceClient(config)
                remote.sync_inputs = AsyncMock(side_effect=OSError('synthetic upload failure'))
                app = TelegramBotApp.__new__(TelegramBotApp)
                app.remote_workspace = remote
                attachment = MessagePart(PartKind.FILE, filename=f'report{number}.pdf',
                    mime_type='application/pdf', size_bytes=len(original), artifact_path=str(local), remote_sync=True)
                if enabled:
                    with self.assertLogs('tgchatbot.transports.telegram_adapter', level='ERROR'):
                        parts = await app._sync_parts_to_remote(self.session, [attachment])
                    remote.sync_inputs.assert_awaited_once()
                else:
                    parts = await app._sync_parts_to_remote(self.session, [attachment])
                    remote.sync_inputs.assert_not_awaited()
                self.assertFalse(local.exists(), 'Upload staging bytes survived completed intake')
                source = await self.store.append_message(self.session, ConversationMessage(MessageRole.USER,
                    [MessagePart(PartKind.TEXT, text='Please remember this report.'), *parts], metadata={
                        'source': 'telegram', 'source_chat_id': '100', 'source_message_id': str(number),
                        'actor_id': 'telegram:user:11', 'actor_kind': 'user', 'actor_name': 'Alex'}))
                reopened = await self.new_store()
                saved = (await reopened.read_messages(self.session, [source.db_id]))[0].message
                descriptor = next(part for part in saved.parts if part.kind == PartKind.FILE)
                self.assertEqual(descriptor.filename, f'report{number}.pdf')
                self.assertEqual(descriptor.mime_type, 'application/pdf')
                self.assertEqual(descriptor.size_bytes, len(original))
                self.assertIn('unavailable', descriptor.detail)
                self.assertIsNone(descriptor.artifact_path)
                self.assertIsNone(descriptor.data_b64)
                self.assertFalse(descriptor.remote_sync)
                found = await reopened.search_messages(self.session, f'report{number}.pdf')
                self.assertIn(source.db_id, {row['id'] for row in found})
                revisions = json.dumps(await reopened.list_message_revisions(self.session, source.db_id), default=str)
                self.assertNotIn(original.decode(), revisions)
                self.assertNotIn(base64.b64encode(original).decode(), revisions)
                self.assertNotIn(str(local), revisions)

    async def test_ssh_socket_moves_but_identity_and_remote_session_stay_stable(self):
        identity = self.root / 'data' / 'ssh' / 'identity'
        config = self.layout_config(SSH_EXEC_HOST='remote.invalid', SSH_EXEC_IDENTITY_FILE=str(identity))
        remote = RemoteWorkspaceClient(config)
        self.assertEqual(remote._control_dir, config.temp_dir / 'ssh_mux')
        self.assertEqual(remote._control_dir.stat().st_mode & 0o777, 0o700)
        self.assertIn(str(identity), remote._ssh_base_args())
        self.assertIn(str(identity), remote._scp_base_args())
        self.assertFalse((config.data_dir / 'ssh_mux').exists())
        moved_temp = RemoteWorkspaceClient(replace(config, temp_dir=self.root / 'other-temp'))
        self.assertEqual(moved_temp.session_paths(self.session), remote.session_paths(self.session))
        self.assertNotEqual(moved_temp._control_path, remote._control_path)


class PreviewRestartTests(unittest.TestCase):
    def test_crash_restart_discards_only_old_previews_and_preserves_one_live_cache(self):
        with tempfile.TemporaryDirectory(prefix='preview-restart-', dir=Path(__file__).parent) as directory:
            root = Path(directory)
            retained = root / 'artifacts' / 'retained.txt'
            retained.parent.mkdir()
            retained.write_text('unrelated temporary content')
            (root / 'previews-link').symlink_to(retained.parent, target_is_directory=True)
            # A hard exit leaves real cache files behind and releases the OS lock;
            # no private implementation hooks simulate the restart boundary.
            child = subprocess.run([sys.executable, '-B', '-c', '''
import base64, json, os, sys
from pathlib import Path
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind
from tgchatbot.storage.previews import PreviewCache
cache = PreviewCache(Path(sys.argv[1]), max_bytes=8)
message = cache.externalize(ConversationMessage(MessageRole.USER, [MessagePart(PartKind.IMAGE,
    filename='old.jpg', data_b64=base64.b64encode(b'oldold').decode())]))
print(json.dumps({'directory': str(cache.root), 'reference': message.parts[0].preview_ref}), flush=True)
os._exit(0)
''', str(root)], check=True, capture_output=True, text=True)
            previous = json.loads(child.stdout)
            self.assertTrue(Path(previous['directory']).is_dir())
            with patch.dict(os.environ, {'MEMORY_PREVIEW_CACHE_BYTES': '8'}):
                cache = PreviewCache(root)
            try:
                self.assertFalse(Path(previous['directory']).exists())
                self.assertEqual(retained.read_text(), 'unrelated temporary content')
                expired = cache.materialize(ConversationMessage(MessageRole.USER,
                    [MessagePart(PartKind.TEXT, text='Original caption'),
                     MessagePart(PartKind.IMAGE, filename='old.jpg', preview_ref=previous['reference'])]), vision=True)
                self.assertEqual(expired.parts[0].text, 'Original caption')
                self.assertIn('expired or unavailable', expired.parts[1].text)
                cached = []
                for data in (b'aaaaaa', b'bbbbbb'):
                    cached.append(cache.externalize(ConversationMessage(MessageRole.USER,
                        [MessagePart(PartKind.IMAGE, data_b64=base64.b64encode(data).decode())])))
                self.assertLessEqual(sum(path.stat().st_size for path in cache.root.iterdir()), 8)
                self.assertIsNone(cache.materialize(cached[0], vision=True).parts[0].data_b64)
                self.assertEqual(cache.materialize(cached[1], vision=True).parts[0].data_b64, base64.b64encode(b'bbbbbb').decode())
                with self.assertRaisesRegex(RuntimeError, 'Another bot'):
                    PreviewCache(root, max_bytes=8)
                self.assertEqual(cache.materialize(cached[1], vision=True).parts[0].data_b64, base64.b64encode(b'bbbbbb').decode())
            finally:
                cache.close()
