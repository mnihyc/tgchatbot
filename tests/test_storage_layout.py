from __future__ import annotations

import base64
import asyncio
import json
import os
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

    async def test_concurrent_native_history_admissions_preserve_originals_and_replay(self):
        sources = await asyncio.gather(*(self.store.append_message(self.session,
            ConversationMessage.user_text(f'Original {number}', metadata={
                'provider_native': {'provider': 'openai', 'model': 'fixture',
                    'items': [{'text': f'Native {number}'}]}})) for number in range(20)))
        restarted = await self.new_store()
        _, restored = await restarted.load_live_context(self.session)
        self.assertEqual(restored, sorted(sources, key=lambda row: row.db_id))
        originals = await restarted.read_messages(self.session, [source.db_id for source in sources], limit=len(sources))
        self.assertEqual({source.message.parts[0].text for source in originals},
                         {f'Original {number}' for number in range(20)})
        self.assertFalse(any('provider_native' in row.message.metadata for row in originals))

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
            previews = PreviewCache(store, max_bytes=8)
            self.addCleanup(previews.close)
            image_message = ConversationMessage(MessageRole.USER, [
                MessagePart(PartKind.TEXT, text='Remember the image'),
                MessagePart(PartKind.IMAGE, mime_type='image/png', data_b64=preview, remote_sync=False),
            ])
            saved = await store.append_message(self.session, image_message)
            materialized = await previews.materialize_many(self.session, [saved.message], vision=True)
            self.assertEqual(materialized[0].parts[1].data_b64, preview)
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
        previews.close()
        restored_previews = PreviewCache(restarted, max_bytes=0)
        self.addCleanup(restored_previews.close)
        runtime = AgentRuntime(config=config, store=restarted, tool_registry=FixtureTools(), providers={'openai': provider}, preview_cache=restored_previews)
        await runtime.run_turn(session_id=self.session, user_display_name='tester', incoming_message=ConversationMessage.user_text('Continue'))
        request_parts = [part for message in provider.requests[0]['messages'] for part in message.parts]
        self.assertEqual([part.data_b64 for part in request_parts if part.kind == PartKind.IMAGE], [preview])

    async def test_remote_upload_history_keeps_remote_locator_after_temp_clear(self):
        with tempfile.TemporaryDirectory(dir=self.root) as temporary:
            config = self.layout_config(APP_TEMP_DIR=temporary, SSH_EXEC_HOST='remote.invalid')
            local = ArtifactStore(config.artifact_dir).save_bytes(chat_id=self.session, filename='report.txt', data=b'report')
            remote = RemoteWorkspaceClient(config)
            remote_path = remote.session_paths(self.session).root + '/2026-04-30/report_0123456789abcdef.txt'
            uploaded = {}

            async def sync_inputs(session_id, paths, *, sent_at=None, filenames=None):
                paths_by_source = {}
                for path in paths:
                    self.assertEqual(filenames[str(path.resolve())], 'report.txt')
                    destination = remote_path
                    uploaded[destination] = path.read_bytes()
                    paths_by_source[str(path.resolve())] = destination
                return RemoteSyncResult(paths_by_source)

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
        self.assertIn('paths relative to workspace', stored.parts[0].text)
        self.assertIn('2026-04-30/report_0123456789abcdef.txt', stored.parts[0].text)
        self.assertEqual(RemoteWorkspaceClient(config).session_paths(self.session), remote.session_paths(self.session))

    async def test_remote_fetch_stages_in_temp_artifacts(self):
        config = self.layout_config(SSH_EXEC_HOST='remote.invalid')
        remote = RemoteWorkspaceClient(config)
        paths = remote.session_paths(self.session)
        remote.ensure_session_dirs = AsyncMock(return_value=paths)
        remote._resolve_remote_paths = AsyncMock(side_effect=lambda paths, selected: selected)

        async def scp(*arguments, **kwargs):
            # Only the process boundary is mocked: fetch path construction and
            # output validation use the real implementation.
            Path(arguments[-1]).write_bytes(b'fetched report')
            return SimpleNamespace(returncode=0, communicate=AsyncMock(return_value=(b'', b'')))

        with patch('tgchatbot.tools.remote_workspace.asyncio.create_subprocess_exec', side_effect=scp) as execute:
            artifacts = await remote.fetch_files(session_id=self.session, remote_paths=['report.txt'])
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0].path.parent, self.root / 'tmp' / 'bot' / 'artifacts' / self.session / 'remote_fetch')
        self.assertTrue(artifacts[0].path.name.endswith('-report.txt'))
        self.assertTrue(artifacts[0].temporary)
        self.assertEqual(artifacts[0].path.read_bytes(), b'fetched report')
        self.assertIn('remote.invalid:' + paths.root + '/report.txt', execute.call_args.args)
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
                    with self.assertLogs('tgchatbot.media.attachments', level='ERROR'):
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


class PreviewReconstructionTests(BusinessTestCase):
    async def test_cache_eviction_restart_and_zero_cache_keep_identical_pixels(self):
        await self.settings(max_input_images=3, compact_target_images=1)
        cache = PreviewCache(self.store, max_bytes=2)
        self.addCleanup(cache.close)
        originals = []
        for text, image in [('first', 'YWFh'), ('second', 'YmJi')]:
            saved = await self.store.append_message(self.session, ConversationMessage(MessageRole.USER,
                [MessagePart(PartKind.TEXT, text=text), MessagePart(PartKind.IMAGE,
                 data_b64=image, mime_type='image/png')]))
            originals.append(saved)
        expected = await cache.materialize_many(self.session, [item.message for item in originals], vision=True)
        cache.close()
        restarted = await self.new_store()
        empty_cache = PreviewCache(restarted, max_bytes=0)
        self.addCleanup(empty_cache.close)
        _, raw = await restarted.load_live_context(self.session)
        reconstructed = await empty_cache.materialize_many(self.session, [item.message for item in raw], vision=True)
        self.assertEqual(reconstructed, expected)
        self.assertEqual([part.data_b64 for item in reconstructed for part in item.parts if part.kind == PartKind.IMAGE],
                         ['YWFh', 'YmJi'])
        audit = await restarted.list_message_revisions(self.session, originals[0].db_id)
        self.assertNotIn('data_b64', json.dumps(audit, default=str))

    async def test_image_retirement_is_atomic_and_deduplication_is_session_scoped(self):
        await self.settings()
        image = ConversationMessage(MessageRole.USER, [MessagePart(PartKind.STICKER, text='🙂'),
            MessagePart(PartKind.IMAGE, data_b64='YWJj', mime_type='image/png')])
        first = await self.store.append_message(self.session, image)
        second = await self.store.append_message(self.session, image)
        other = 'telegram:other'
        await self.store.get_or_create_session(other, self.config.default_session_settings())
        elsewhere = await self.store.append_message(other, image)
        reference = first.message.parts[-1].preview_ref
        self.assertEqual(reference, second.message.parts[-1].preview_ref)
        self.assertEqual((await self.store.retire_context_images(self.session, target_images=1)).removed_images, 1)
        self.assertEqual(await self.store.load_preview_data(self.session, [reference]), {reference: b'abc'})
        self.assertEqual((await self.store.retire_context_images(self.session, target_images=0)).removed_images, 1)
        self.assertEqual(await self.store.load_preview_data(self.session, [reference]), {reference: b'abc'})
        self.assertEqual(await self.store.load_preview_data(other, [reference]), {reference: b'abc'})
        _, rows = await self.store.load_live_context(self.session)
        self.assertTrue(all(item.message.parts[0].text == '🙂' for item in rows))
        self.assertTrue(all(item.message.parts[-1].text == '[Image compacted]' for item in rows))
        canonical = await self.store.read_messages(self.session, [first.db_id])
        self.assertEqual(canonical[0].message.parts[-1].preview_ref, reference)

    async def test_native_continuation_reconstructs_without_a_sidecar_or_runtime(self):
        native = {'provider': 'gemini', 'items': [{'role': 'model', 'parts': [
            {'text': 'Visible answer', 'thoughtSignature': 'opaque-signature'}]}]}
        saved = await self.store.append_message(self.session, ConversationMessage.assistant_text(
            'Visible answer', metadata={'provider_native': native}))
        restarted = await self.new_store()
        _, reconstructed = await restarted.load_live_context(self.session)
        self.assertEqual(reconstructed, [saved])
        self.assertEqual(reconstructed[0].message.metadata['provider_native'], native)
        audit = await restarted.list_message_revisions(self.session, saved.db_id)
        self.assertNotIn('provider_native', audit[0]['metadata'])
