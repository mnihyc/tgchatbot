from __future__ import annotations

import base64
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tests.business_helpers import FixtureTools, ScriptedProvider
from tgchatbot.config import load_config
from tgchatbot.core.runtime import AgentRuntime
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind, ProviderResponse
from tgchatbot.storage.artifacts import ArtifactStore
from tgchatbot.storage.presets import PresetStore
from tgchatbot.storage.sqlite_store import SQLiteStore
from tgchatbot.tools.remote_workspace import RemoteSyncResult, RemoteWorkspaceClient
from tgchatbot.transports.telegram_adapter import TelegramBotApp


class StorageLayoutTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.fixture = tempfile.TemporaryDirectory(prefix='fixture-layout-', dir=Path(__file__).parent)
        self.addCleanup(self.fixture.cleanup)
        self.root = Path(self.fixture.name)
        self.session = 'telegram:100'

    def config(self, **environment):
        with patch.dict(os.environ, {
            'APP_DATA_DIR': str(self.root / 'data'),
            'TGBOT_TOKEN': '123456:mock-token',
            'OPENAI_API_KEY': 'mock-key',
            **environment,
        }, clear=True):
            return load_config()

    async def test_cleared_temp_can_be_recreated_without_losing_retained_state(self):
        # A separate context removes only disposable fixture state, simulating
        # clearing tmp while the bot is stopped. The outer fixture retains data.
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            config = self.config(APP_TEMP_DIR=temporary)
            artifacts = ArtifactStore(config.artifact_dir)
            staged = artifacts.save_bytes(chat_id=self.session, filename='upload.bin', data=b'upload bytes')
            self.assertTrue(staged.is_relative_to(temporary))
            remote = RemoteWorkspaceClient(config)
            remote._control_path.write_bytes(b'fixture socket placeholder')
            store = SQLiteStore(config.db_path)
            settings = config.default_session_settings()
            settings.system_prompt = 'Keep this conversation voice.'
            await store.save_session(self.session, settings)
            preview = base64.b64encode(b'inline preview bytes').decode('ascii')
            await store.append_message(self.session, ConversationMessage(MessageRole.USER, [
                MessagePart(PartKind.TEXT, text='Remember the image'),
                MessagePart(PartKind.IMAGE, mime_type='image/png', data_b64=preview, remote_sync=False),
            ]))
            preset = PresetStore(config.preset_dir).save_text('fixture', 'saved preset')
            retained = [preset, config.sticker_dir / 'fixture.webp', config.data_dir / 'ssh' / 'identity', config.data_dir / 'ssh' / 'known_hosts']
            for path in retained[1:]:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b'fixture retained content')
            snapshots = {path: path.read_bytes() for path in retained}
        self.assertFalse(staged.exists())
        self.assertFalse(config.temp_dir.exists())
        restarted = SQLiteStore(config.db_path)
        self.assertEqual((await restarted.get_or_create_session(self.session, config.default_session_settings())).system_prompt, 'Keep this conversation voice.')
        history = await restarted.list_messages(self.session)
        self.assertEqual(history[0].parts[1].data_b64, preview)
        for path, expected in snapshots.items():
            self.assertEqual(path.read_bytes(), expected)
        new_file = ArtifactStore(config.artifact_dir).save_bytes(chat_id=self.session, filename='next.bin', data=b'next upload')
        new_remote = RemoteWorkspaceClient(config)
        self.assertTrue(new_file.is_file())
        self.assertTrue(new_remote._control_dir.is_dir())
        provider = ScriptedProvider(responses=[ProviderResponse(final_text='continued')])
        runtime = AgentRuntime(config=config, store=restarted, tool_registry=FixtureTools(), providers={'openai': provider})
        await runtime.run_turn(session_id=self.session, user_display_name='tester', incoming_message=ConversationMessage.user_text('Continue'))
        self.assertEqual(provider.requests[0]['messages'][0].parts[1].data_b64, preview)

    async def test_remote_upload_history_keeps_remote_locator_after_temp_clear(self):
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            config = self.config(APP_TEMP_DIR=temporary, SSH_EXEC_HOST='remote.invalid')
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
            store = SQLiteStore(config.db_path)
            await store.append_message(self.session, ConversationMessage(MessageRole.USER, parts))
        stored = (await SQLiteStore(config.db_path).list_messages(self.session))[0]
        attachment = next(part for part in stored.parts if part.kind == PartKind.FILE)
        self.assertEqual(attachment.artifact_path, remote_path)
        self.assertTrue(attachment.remote_sync)
        self.assertIn(remote_path, stored.parts[0].text)
        self.assertEqual(RemoteWorkspaceClient(config).session_paths(self.session), remote.session_paths(self.session))

    async def test_remote_fetch_stages_in_temp_artifacts(self):
        config = self.config(SSH_EXEC_HOST='remote.invalid')
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

    async def test_ssh_socket_moves_but_identity_and_remote_session_stay_stable(self):
        identity = self.root / 'data' / 'ssh' / 'identity'
        config = self.config(SSH_EXEC_HOST='remote.invalid', SSH_EXEC_IDENTITY_FILE=str(identity))
        remote = RemoteWorkspaceClient(config)
        self.assertEqual(remote._control_dir, config.temp_dir / 'ssh_mux')
        self.assertEqual(remote._control_dir.stat().st_mode & 0o777, 0o700)
        self.assertIn(str(identity), remote._ssh_base_args())
        self.assertIn(str(identity), remote._scp_base_args())
        self.assertFalse((config.data_dir / 'ssh_mux').exists())
        moved_temp = RemoteWorkspaceClient(replace(config, temp_dir=self.root / 'other-temp'))
        self.assertEqual(moved_temp.session_paths(self.session), remote.session_paths(self.session))
        self.assertNotEqual(moved_temp._control_path, remote._control_path)
