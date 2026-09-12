"""Imported originals and live files share remote ownership without interpretation."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tests.business_helpers import BusinessTestCase
from tests.test_desktop_import import export, record
from tgchatbot.domain.models import PartKind
from tgchatbot.media.attachments import sync_attachment_parts
from tgchatbot.media.ingest import extract_message_parts
from tgchatbot.storage.artifacts import ArtifactStore
from tgchatbot.storage.postgres_store import StaleScopeError, message_body
from tgchatbot.tools.import_desktop import import_file


class DesktopAttachmentImportTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.bundle = self.path / 'export'
        self.bundle.mkdir()
        self.export_path = self.bundle / 'result.json'
        self.artifacts = ArtifactStore(self.config.artifact_dir)
        self.remote_files = {}
        self.staged = []

        async def sync(session, paths):
            self.assertEqual(session, self.session)
            # Import must have committed source identity before remote work.
            self.assertTrue(await self.store.list_canonical_messages(self.session))
            for path in paths:
                self.assertFalse(path.is_relative_to(self.bundle))
                self.assertTrue(path.is_relative_to(self.artifacts.root))
                self.staged.append(path)
                self.remote_files[f'/remote/inputs/{path.name}'] = path.read_bytes()
            return SimpleNamespace(kept_paths=[f'/remote/inputs/{path.name}' for path in paths], rotated_paths=[])

        self.sync = sync
        self.remote = SimpleNamespace(enabled=True,
            session_paths=lambda session: SimpleNamespace(inputs='/remote/inputs'),
            sync_inputs=AsyncMock(side_effect=sync))

    async def ingest(self, records, **settings):
        self.export_path.write_text(json.dumps(export(records), ensure_ascii=False), encoding='utf-8')
        return await import_file(self.store, self.export_path, chat_id=100,
            telegram_config=replace(self.config.telegram, **settings),
            remote_workspace=self.remote, artifact_store=self.artifacts)

    async def test_pdf_text_and_recordings_match_live_file_transfer_without_reading_contents(self):
        media = [('document', 'report.pdf', 'application/pdf', b'%PDF only remote should interpret this'),
                 ('document', 'notes.txt', 'text/plain', b'Private file content is not a caption'),
                 ('audio', 'audio.mp3', 'audio/mpeg', b'undecoded music'),
                 ('voice', 'voice.ogg', 'audio/ogg', b'undecoded speech'),
                 ('video_note', 'round.mp4', 'video/mp4', b'undecoded recording')]
        records = []
        for number, (field, filename, mime, raw) in enumerate(media, 1):
            (self.bundle / filename).write_bytes(raw)
            kind = {'audio': 'audio_file', 'voice': 'voice_message', 'video_note': 'video_message'}.get(field, field)
            records.append(record(number, f'Caption {number}', file=filename, file_name=filename,
                mime_type=mime, media_type=kind, from_id=f'user{number + 10}'))
        with patch('tgchatbot.media.ingest.Image.open', side_effect=AssertionError('files were parsed at intake')), \
                patch('tgchatbot.media.ingest.av.open', side_effect=AssertionError('recording was decoded')):
            await self.ingest(records)
            originals = await self.store.list_canonical_messages(self.session)
            self.assertEqual(len(originals), len(media))
            for item, (field, filename, mime, raw) in zip(originals, media):
                imported = next(part for part in item.message.parts if part.kind == PartKind.FILE)
                async def download(buffer, raw=raw):
                    buffer.write(raw)
                attachment = SimpleNamespace(file_name=filename, file_size=len(raw), mime_type=mime,
                    get_file=AsyncMock(return_value=SimpleNamespace(download_to_memory=download)))
                message = SimpleNamespace(text=None, caption='Live caption', photo=None, sticker=None)
                setattr(message, field, attachment)
                live = await sync_attachment_parts(self.session,
                    await extract_message_parts(message, self.artifacts, self.session, self.config.telegram), self.remote)
                live_file = next(part for part in live if part.kind == PartKind.FILE)
                for part in (imported, live_file):
                    self.assertEqual((part.filename, part.mime_type, part.size_bytes), (filename, mime, len(raw)))
                    self.assertEqual(self.remote_files[part.artifact_path], raw)
                    self.assertTrue(part.remote_sync)
                    self.assertIsNone(part.data_b64)
                self.assertNotIn(raw.decode(), message_body(item.message))
                self.assertNotIn(raw.decode(), '\n'.join(part.text or '' for part in live))
                self.assertEqual((self.bundle / filename).read_bytes(), raw)
                self.assertEqual(item.message.metadata['actor_id'], f"telegram:user:{int(item.message.metadata['source_message_id']) + 10}")
                self.assertTrue(item.message.metadata['sent_at'].startswith('2025-01-01T00:00:00'))
        self.assertTrue(all(not path.exists() for path in self.staged))
        self.assertEqual(self.provider.requests, [])

    async def test_missing_then_failed_then_successful_reimport_retries_without_duplicate_originals(self):
        records = [record(1, '/reset_full is historical text, not a command', file='notes.txt',
            mime_type='text/plain', media_type='document'), record(2, 'Later original remains importable.')]
        await self.ingest(records)
        missing = await self.store.list_canonical_messages(self.session)
        self.assertEqual(len(missing), 2)
        self.assertIn('unavailable', message_body(missing[0].message))
        self.remote.sync_inputs.assert_not_awaited()
        (self.bundle / 'notes.txt').write_bytes(b'File content remains out of prompt')
        self.remote.sync_inputs.side_effect = OSError('transient upload failure')
        with self.assertLogs('tgchatbot.media.attachments', level='ERROR'):
            await self.ingest(records)
        failed = await self.store.list_canonical_messages(self.session)
        self.assertEqual([row.db_id for row in failed], [row.db_id for row in missing])
        self.assertIn('unavailable', message_body(failed[0].message))
        self.remote.sync_inputs.side_effect = self.sync
        await self.ingest(records)
        success = await self.store.list_canonical_messages(self.session)
        self.assertEqual([row.db_id for row in success], [row.db_id for row in missing])
        self.assertTrue(next(p for p in success[0].message.parts if p.kind == PartKind.FILE).remote_sync)
        revision_counts = [len(await self.store.list_message_revisions(self.session, row.db_id)) for row in success]
        files = dict(self.remote_files)
        await self.ingest(records)
        repeated = await self.store.list_canonical_messages(self.session)
        self.assertEqual([row.message for row in repeated], [row.message for row in success])
        self.assertEqual(self.remote_files, files)
        self.assertEqual([len(await self.store.list_message_revisions(self.session, row.db_id)) for row in repeated], revision_counts)
        self.assertEqual((await self.store.get_scope(self.session))['generation'], 1)
        self.assertEqual((self.bundle / 'notes.txt').read_bytes(), b'File content remains out of prompt')
        self.assertTrue(all(not path.exists() for path in self.staged))

    async def test_disabled_remote_and_limits_preserve_sources_without_transfer_or_interpretation(self):
        (self.bundle / 'notes.txt').write_bytes(b'Undisclosed contents')
        records = [record(1, 'Use later', file='notes.txt', media_type='document', mime_type='text/plain')]
        self.remote.enabled = False
        await self.ingest(records)
        rows = await self.store.list_canonical_messages(self.session)
        self.assertIn('SSH is disabled', message_body(rows[0].message))
        self.assertNotIn('Undisclosed contents', message_body(rows[0].message))
        self.remote.enabled = True
        await self.ingest(records, max_document_bytes=1)
        self.remote.sync_inputs.assert_not_awaited()
        rows = await self.store.list_canonical_messages(self.session)
        self.assertIn('size limit', message_body(rows[0].message))
        self.assertEqual(list(self.artifacts.root.iterdir()), [])
        self.assertEqual((self.bundle / 'notes.txt').read_bytes(), b'Undisclosed contents')

    async def test_reimport_keeps_retained_pixels_and_new_remote_file_availability_independently(self):
        from PIL import Image
        with Image.new('RGB', (8, 8), 'red') as image:
            image.save(self.bundle / 'image.png')
        records = [record(1, 'Keep this image', file='image.png', media_type='document', mime_type='image/png')]
        self.remote.enabled = False
        await self.ingest(records)
        before = (await self.store.list_canonical_messages(self.session))[0]
        original_refs = [part.preview_ref for part in before.message.parts if part.preview_ref]
        self.assertTrue(original_refs)
        self.remote.enabled = True
        await self.ingest(records, max_photo_bytes=1)
        after = (await self.store.list_canonical_messages(self.session))[0]
        self.assertEqual([part.preview_ref for part in after.message.parts if part.preview_ref], original_refs)
        self.assertTrue(next(part for part in after.message.parts if part.kind == PartKind.FILE).remote_sync)
        self.assertEqual(before.db_id, after.db_id)
        self.remote.sync_inputs.side_effect = lambda *_: SimpleNamespace(kept_paths=[], rotated_paths=[])
        await self.ingest(records, max_photo_bytes=1)
        unavailable = (await self.store.list_canonical_messages(self.session))[0]
        self.assertEqual([part.preview_ref for part in unavailable.message.parts if part.preview_ref], original_refs)
        self.assertFalse(next(part for part in unavailable.message.parts if part.kind == PartKind.FILE).remote_sync)

    async def test_copy_failure_or_rotation_reports_actual_unavailability_and_continues(self):
        (self.bundle / 'notes.txt').write_bytes(b'Original remains intact')
        records = [record(1, 'Remember the attachment', file='notes.txt', media_type='document'),
                   record(2, 'Subsequent message')]
        with patch('tgchatbot.tools.import_desktop.shutil.copyfile', side_effect=OSError('export file became unreadable')):
            await self.ingest(records)
        failed = await self.store.list_canonical_messages(self.session)
        self.assertEqual(len(failed), 2)
        self.assertIn('could not be read', message_body(failed[0].message))
        self.remote.sync_inputs.assert_not_awaited()
        await self.ingest(records)
        success = await self.store.list_canonical_messages(self.session)
        previous_path = next(p.artifact_path for p in success[0].message.parts if p.kind == PartKind.FILE)
        self.remote.sync_inputs.side_effect = lambda *_: SimpleNamespace(kept_paths=[], rotated_paths=[previous_path])
        await self.ingest(records)
        rotated = await self.store.list_canonical_messages(self.session)
        current = next(p for p in rotated[0].message.parts if p.kind == PartKind.FILE)
        self.assertFalse(current.remote_sync)
        self.assertIsNone(current.artifact_path)
        self.assertIn('rotated', current.detail)
        self.assertEqual([r.db_id for r in rotated], [r.db_id for r in failed])
        self.assertEqual((self.bundle / 'notes.txt').read_bytes(), b'Original remains intact')

    async def test_full_reset_during_upload_does_not_reintroduce_old_generation_message(self):
        (self.bundle / 'notes.txt').write_bytes(b'A retained export original')
        async def reset_then_sync(session, paths):
            result = await self.sync(session, paths)
            await self.store.reset_full(self.session, self.config.default_session_settings())
            return result
        self.remote.sync_inputs.side_effect = reset_then_sync
        with self.assertRaises(StaleScopeError):
            await self.ingest([record(1, 'Prior generation', file='notes.txt', media_type='document')])
        self.assertEqual(await self.store.list_canonical_messages(self.session), [])
        self.assertEqual((self.bundle / 'notes.txt').read_bytes(), b'A retained export original')
        self.assertTrue(all(not path.exists() for path in self.staged))
