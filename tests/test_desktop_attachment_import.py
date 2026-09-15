"""Imported originals and live files share remote ownership without interpretation."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from tests.business_helpers import BusinessTestCase
from tests.test_desktop_import import export, record
from tgchatbot.domain.models import ConversationMessage, PartKind, ProviderResponse
from tgchatbot.media.attachments import sync_attachment_parts
from tgchatbot.media.ingest import extract_message_parts
from tgchatbot.storage.artifacts import ArtifactStore
from tgchatbot.storage.postgres_store import StaleScopeError, message_body
from tgchatbot.tools.import_desktop import import_file
from tgchatbot.tools.remote_workspace import RemoteSyncResult


class DesktopAttachmentImportTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.bundle = self.path / 'export'
        self.bundle.mkdir()
        self.export_path = self.bundle / 'result.json'
        self.artifacts = ArtifactStore(self.config.artifact_dir)
        self.remote_files = {}
        self.staged = []

        async def sync(session, paths, *, sent_at=None, filenames=None):
            self.assertEqual(session, self.session)
            # Import must have committed source identity before remote work.
            self.assertTrue(await self.store.list_canonical_messages(self.session))
            receipts = {}
            for path in paths:
                self.assertFalse(path.is_relative_to(self.bundle))
                self.assertTrue(path.is_relative_to(self.artifacts.root))
                self.staged.append(path)
                original = Path(filenames[str(path.resolve())])
                destination = f'/remote/2025-01-01/{original.stem}_0123456789abcdef{original.suffix}'
                self.remote_files[destination] = path.read_bytes()
                receipts[str(path.resolve())] = destination
            return RemoteSyncResult(receipts)

        self.sync = sync
        self.remote = SimpleNamespace(enabled=True,
            session_paths=lambda session: SimpleNamespace(root='/remote'),
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
                revisions = await self.store.list_message_revisions(self.session, item.db_id)
                self.assertIn(f'export_reference="{filename}"', revisions[-1]['body'])
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

    async def test_user_only_attachment_rerun_preserves_committed_remote_receipt(self):
        raw = b'An original document for later reading'
        (self.bundle / 'notes.txt').write_bytes(raw)
        self.export_path.write_text(json.dumps({
            'type': 'bot_chat', 'id': 99, 'name': 'Previous bot', 'messages': [
                record(1, 'Keep my notes', from_id='user100', file='notes.txt',
                    mime_type='text/plain', media_type='document'),
                record(2, 'A bot document', from_id='user99', file='missing.pdf',
                    mime_type='application/pdf', media_type='document'),
            ]}), encoding='utf-8')

        async def import_again():
            return await import_file(self.store, self.export_path, chat_id=100, user_only=True,
                telegram_config=self.config.telegram, remote_workspace=self.remote,
                artifact_store=self.artifacts)

        result = await import_again()
        self.assertEqual((result.messages, result.skipped), (1, 1))
        original = (await self.store.list_canonical_messages(self.session))[0]
        revisions = await self.store.list_message_revisions(self.session, original.db_id)
        await import_again()
        restored = await self.store.list_canonical_messages(self.session)
        self.assertEqual(restored, [original])
        self.assertEqual(await self.store.list_message_revisions(self.session, original.db_id), revisions)
        attachment = next(part for part in original.message.parts if part.kind == PartKind.FILE)
        self.assertEqual(self.remote_files, {attachment.artifact_path: raw})
        self.assertTrue(all(not path.exists() for path in self.staged))
        self.assertEqual(self.provider.requests, [])

    async def test_import_uses_original_dates_and_retains_remote_receipts_across_reimport(self):
        await self.settings()
        # These UTC instants straddle midnight in the configured UTC+8 zone.
        # Import runs much later; neither that time nor edit time owns placement.
        dates = {'2026-04-30T15:59:00+00:00': '2026-04-30',
                 '2026-04-30T16:01:00+00:00': '2026-05-01'}
        expected_paths = {}
        seen = []
        async def sync(session, paths, *, sent_at=None, filenames=None):
            self.assertEqual(session, self.session)
            day = dates[sent_at]
            receipts = {}
            for path in paths:
                self.staged.append(path)
                original = Path(filenames[str(path.resolve())])
                destination = f'/remote/{day}/{original.stem}_0123456789abcdef{original.suffix}'
                self.remote_files[destination] = path.read_bytes()
                receipts[str(path.resolve())] = destination
                expected_paths[sent_at] = destination
            seen.append(sent_at)
            return RemoteSyncResult(receipts)
        self.remote.sync_inputs.side_effect = sync
        records = []
        for number, sent_at in enumerate(dates, 1):
            filename = f'Original report {number}.txt'
            (self.bundle / filename).write_bytes(f'Unparsed original {number}'.encode())
            records.append(record(number, f'Caption {number}', file=filename,
                file_name=filename, media_type='document', from_id=f'user{10 + number}',
                date_unixtime=str(int(datetime.fromisoformat(sent_at).timestamp())),
                edited_unixtime=str(int(datetime(2026, 5, 2, tzinfo=timezone.utc).timestamp()))))
        await self.ingest(records)
        first = await self.store.list_canonical_messages(self.session)
        await self.ingest(records)
        reopened = await self.new_store()
        restored = await reopened.list_canonical_messages(self.session)
        self.assertEqual(restored, first)
        self.assertEqual(seen, list(dates) * 2)
        self.assertEqual(len(self.remote_files), 2)
        for row, number in zip(restored, (1, 2)):
            message = row.message
            attachment = next(part for part in message.parts if part.kind == PartKind.FILE)
            self.assertEqual(attachment.artifact_path, expected_paths[message.metadata['sent_at']])
            self.assertEqual(attachment.filename, f'Original report {number}.txt')
            self.assertEqual(message.metadata['actor_id'], f'telegram:user:{10 + number}')
            self.assertEqual(message.metadata['source_message_id'], str(number))
            self.assertEqual(Path(attachment.artifact_path).name, f'Original report {number}_0123456789abcdef.txt')
            self.assertIn(f"{dates[message.metadata['sent_at']]}/{Path(attachment.artifact_path).name}",
                message_body(message))
            self.assertIn('paths relative to workspace', message_body(message))
            self.assertEqual(self.remote_files[attachment.artifact_path], f'Unparsed original {number}'.encode())
        self.assertTrue(all(not path.exists() for path in self.staged))
        self.assertEqual(self.provider.requests, [])

        self.provider.responses.append(ProviderResponse(final_text='I can inspect the selected report.'))
        await self.runtime.run_turn(session_id=self.session, user_display_name='Alex',
            incoming_message=ConversationMessage.user_text('Read the first report.'))
        presented = '\n'.join(part.text or '' for message in self.provider.requests[-1]['messages']
            for part in message.parts if part.origin == 'auto_note')
        for remote_path in expected_paths.values():
            self.assertIn(remote_path.removeprefix('/remote/'), presented)
        self.assertIn('paths relative to workspace', presented)

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
        self.remote.sync_inputs.side_effect = lambda *_, **__: RemoteSyncResult({})
        await self.ingest(records, max_photo_bytes=1)
        unavailable = (await self.store.list_canonical_messages(self.session))[0]
        self.assertEqual([part.preview_ref for part in unavailable.message.parts if part.preview_ref], original_refs)
        self.assertFalse(next(part for part in unavailable.message.parts if part.kind == PartKind.FILE).remote_sync)

    async def test_copy_failure_or_unconfirmed_upload_reports_unavailability_and_continues(self):
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
        self.assertTrue(next(p for p in success[0].message.parts if p.kind == PartKind.FILE).remote_sync)
        self.remote.sync_inputs.side_effect = lambda *_, **__: RemoteSyncResult({})
        await self.ingest(records)
        unconfirmed = await self.store.list_canonical_messages(self.session)
        current = next(p for p in unconfirmed[0].message.parts if p.kind == PartKind.FILE)
        self.assertFalse(current.remote_sync)
        self.assertIsNone(current.artifact_path)
        self.assertIn('upload failed', current.detail)
        self.assertEqual([r.db_id for r in unconfirmed], [r.db_id for r in failed])
        self.assertEqual((self.bundle / 'notes.txt').read_bytes(), b'Original remains intact')

    async def test_full_reset_during_upload_does_not_reintroduce_old_generation_message(self):
        (self.bundle / 'notes.txt').write_bytes(b'A retained export original')
        async def reset_then_sync(session, paths, *, sent_at=None, filenames=None):
            result = await self.sync(session, paths, sent_at=sent_at, filenames=filenames)
            await self.store.reset_full(self.session, self.config.default_session_settings())
            return result
        self.remote.sync_inputs.side_effect = reset_then_sync
        with self.assertRaises(StaleScopeError):
            await self.ingest([record(1, 'Prior generation', file='notes.txt', media_type='document')])
        self.assertEqual(await self.store.list_canonical_messages(self.session), [])
        self.assertEqual((self.bundle / 'notes.txt').read_bytes(), b'A retained export original')
        self.assertTrue(all(not path.exists() for path in self.staged))
