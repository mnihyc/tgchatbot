from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image
from telegram import Audio, Chat, Message, Update, User, VideoNote, Voice
from telegram.ext import MessageHandler

from tests.business_helpers import BusinessTestCase
from tgchatbot.domain.models import PartKind
from tgchatbot.storage.artifacts import ArtifactStore
from tgchatbot.tools.memory import audit_records
from tgchatbot.tools.remote_workspace import RemoteSyncResult
from tgchatbot.transports.telegram_adapter import TelegramBotApp


class TelegramIntakeTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.app = TelegramBotApp.__new__(TelegramBotApp)
        self.app.config, self.app.runtime, self.app.store = self.config, self.runtime, self.store
        self.artifact_store = ArtifactStore(self.config.artifact_dir)
        self.app.artifact_store = self.artifact_store
        self.app.remote_workspace = SimpleNamespace(enabled=False)
        self.app._chat_states = {}
        self.app._ensure_reply_worker = AsyncMock()
        self.app._promote_candidate_after_delay = AsyncMock()
        self.app._promote_spontaneous_candidate_after_delay = AsyncMock()
        self.app._notify_user_error = AsyncMock()
        self.chat = SimpleNamespace(id=100, type='private')
        await self.settings()

    def update(self, text='Hello', *, source_id=1, actor=7, photo=None, document=None, caption=None, edited=False):
        user = SimpleNamespace(id=actor, username=f'user{actor}', full_name='Alex', is_bot=False)
        message = SimpleNamespace(chat=self.chat, message_id=source_id, text=text, caption=caption,
            from_user=user, sender_chat=None, date=datetime(2026, 1, 1, tzinfo=timezone.utc),
            edit_date=datetime(2026, 1, 2, tzinfo=timezone.utc) if edited else None,
            entities=[], caption_entities=[], reply_to_message=None,
            photo=photo, document=document, sticker=None, animation=None, video=None,
            audio=None, voice=None, video_note=None, reply_text=AsyncMock())
        return SimpleNamespace(effective_chat=self.chat, effective_message=message, effective_user=user)

    async def ingest(self, update, *, reply=True, edit=False):
        await self.app._ingest_update(update, should_reply=reply, is_group=False, is_edit=edit)
        await asyncio.sleep(0)  # Let the mocked delayed-promotion coroutine finish.

    def photo(self, download):
        file = SimpleNamespace(download_to_memory=AsyncMock(side_effect=download))
        return SimpleNamespace(file_id='synthetic-photo-file', file_unique_id='unique-photo', width=2,
                               height=2, file_size=80, get_file=AsyncMock(return_value=file))

    async def test_registered_audio_voice_and_video_note_enter_file_history_without_reply_or_previews(self):
        handlers = []
        self.app.application = SimpleNamespace(add_handler=handlers.append)
        self.app._register_handlers()
        transferred = []

        async def sync(_session, paths, *, sent_at=None, filenames=None):
            transferred.extend((path.name, path.read_bytes()) for path in paths)
            receipts = {}
            for path in paths:
                original = Path(filenames[str(path.resolve())])
                receipts[str(path.resolve())] = f'/remote/2026-01-01/{original.stem}_0123456789abcdef{original.suffix}'
            return RemoteSyncResult(receipts)

        self.app.remote_workspace = SimpleNamespace(enabled=True,
            session_paths=lambda _session: SimpleNamespace(root='/remote'),
            sync_inputs=AsyncMock(side_effect=sync))
        raw = b'Original media bytes; no automatic interpretation.'
        async def download(buffer):
            buffer.write(raw)
        file = SimpleNamespace(download_to_memory=AsyncMock(side_effect=download))
        source_id = 900
        for chat_type in ('private', 'group'):
            for media_type, media_class, kwargs in (
                ('audio', Audio, {'duration': 1, 'file_name': 'recording.mp3', 'mime_type': 'audio/mpeg'}),
                ('voice', Voice, {'duration': 1, 'mime_type': 'audio/ogg'}),
                ('video_note', VideoNote, {'length': 64, 'duration': 1}),
            ):
                source_id += 1
                with self.subTest(chat=chat_type, media=media_type):
                    media = media_class(file_id=f'file-{source_id}', file_unique_id=f'unique-{source_id}',
                        file_size=len(raw), **kwargs)
                    message = Message(message_id=source_id, date=datetime(2026, 1, 1, tzinfo=timezone.utc),
                        chat=Chat(100, chat_type), from_user=User(7, 'Participant', False), **{media_type: media})
                    update = Update(source_id, message=message)
                    matching = [handler for handler in handlers
                        if isinstance(handler, MessageHandler) and handler.check_update(update)]
                    self.assertEqual(len(matching), 1, 'Live media must reach exactly one registered intake handler')
                    with patch.object(media_class, 'get_file', AsyncMock(return_value=file)) as get_file, \
                         patch('tgchatbot.media.ingest.Image.open') as image_decoder, \
                         patch('tgchatbot.media.ingest.av.open') as video_decoder:
                        await matching[0].callback(update, SimpleNamespace(bot=SimpleNamespace(id=999)))
                    get_file.assert_awaited_once()
                    image_decoder.assert_not_called()
                    video_decoder.assert_not_called()
        originals = await self.store.list_canonical_messages(self.session)
        self.assertEqual(len(originals), 6)
        for original in originals:
            files = [part for part in original.message.parts if part.kind == PartKind.FILE]
            self.assertEqual(len(files), 1)
            self.assertTrue(files[0].artifact_path.startswith('/remote/2026-01-01/'))
            filename = Path(files[0].filename)
            self.assertEqual(Path(files[0].artifact_path).name, f'{filename.stem}_0123456789abcdef{filename.suffix}')
            self.assertFalse(any(part.kind == PartKind.IMAGE for part in original.message.parts))
            self.assertEqual(original.message.metadata['actor_id'], 'telegram:user:7')
            self.assertEqual(original.message.metadata['telegram_intake_stage'], 'complete')
        self.assertEqual([data for _, data in transferred], [raw] * 6)
        self.assertFalse([path for path in self.artifact_store.root.rglob('*') if path.is_file()])
        self.app._promote_candidate_after_delay.assert_not_awaited()
        self.app._promote_spontaneous_candidate_after_delay.assert_not_awaited()
        self.app._notify_user_error.assert_not_awaited()
        self.assertFalse(self.provider.requests)

    async def _reset_during_download(self, *, full):
        entered, release = asyncio.Event(), asyncio.Event()
        async def download(buffer):
            entered.set()
            await release.wait()
            Image.new('RGB', (2, 2), 'red').save(buffer, format='PNG')
        photo = self.photo(download)
        update = self.update(None, caption='Please inspect this photo.', photo=[photo])
        task = asyncio.create_task(self.ingest(update))
        self.addAsyncCleanup(self._finish_task, task, release)
        await asyncio.wait_for(entered.wait(), timeout=3)
        before = [row async for row in audit_records(self.store, self.session)]
        self.assertEqual(len(before), 1, 'The original must exist before the media download finishes')
        self.assertIn('Please inspect this photo.', before[0]['body'])
        self.assertIn('Attachment reference', before[0]['body'])
        self.assertEqual(before[0]['metadata']['actor_id'], 'telegram:user:7')
        self.assertEqual(before[0]['metadata']['telegram_attachments'][0]['file_id'], 'synthetic-photo-file')
        if full:
            await self.store.reset_full(self.session, self.config.default_session_settings())
        else:
            await self.store.reset_context(self.session)
        self.runtime.invalidate_session(self.session)
        release.set()
        await asyncio.wait_for(task, timeout=3)
        after = [row async for row in audit_records(self.store, self.session)]
        self.assertEqual(after, before, 'Late enrichment must not modify or resurrect the original')
        self.assertEqual(await self.store.list_messages(self.session), [])
        recalled = await self.store.search_messages(self.session, 'inspect')
        self.assertEqual(bool(recalled), not full)
        self.app._promote_candidate_after_delay.assert_not_awaited()
        self.app._ensure_reply_worker.assert_not_awaited()
        self.app._notify_user_error.assert_not_awaited()
        self.assertEqual(self.app._flow_state(100).ingest_inflight, 0)

    @staticmethod
    async def _finish_task(task, release):
        release.set()
        if not task.done():
            task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_soft_reset_during_download_keeps_original_for_recall_without_late_reply(self):
        await self._reset_during_download(full=False)

    async def test_full_reset_during_download_keeps_only_old_generation_audit_without_late_reply(self):
        await self._reset_during_download(full=True)

    async def test_reset_during_document_download_discards_only_new_transfer_copy(self):
        async def download(buffer):
            await self.store.reset_context(self.session)
            self.runtime.invalidate_session(self.session)
            buffer.write(b'private document fixture')

        file = SimpleNamespace(download_to_memory=AsyncMock(side_effect=download))
        document = SimpleNamespace(file_id='synthetic-document', file_unique_id='unique-document',
            file_name='report.pdf', mime_type='application/pdf', file_size=24,
            get_file=AsyncMock(return_value=file))
        await self.ingest(self.update(None, caption='Remember this report.', document=document))
        self.assertFalse([path for path in self.artifact_store.root.rglob('*') if path.is_file()])
        self.assertEqual(await self.store.list_messages(self.session), [])
        self.assertTrue(await self.store.search_messages(self.session, 'report'))
        self.app._ensure_reply_worker.assert_not_awaited()

    async def test_plain_text_keeps_one_original_revision_and_one_reply_candidate(self):
        update = self.update('Plain text with stable metadata.')
        await self.ingest(update)
        rows = [row async for row in audit_records(self.store, self.session)]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['revision'], 1)
        self.assertEqual(rows[0]['body'].count('[Message metadata:'), 1)
        self.assertEqual(rows[0]['body'].count('Plain text with stable metadata.'), 1)
        original = (await self.store.list_canonical_messages(self.session))[0].message
        self.assertEqual([part.origin for part in original.parts], ['provenance', None])
        self.assertEqual(original.metadata['actor_id'], 'telegram:user:7')
        self.assertEqual(original.metadata['actor_name'], 'Alex')
        self.assertEqual(original.metadata['source_message_id'], '1')
        self.assertEqual(original.parts[1].text, 'Plain text with stable metadata.')
        self.app._promote_candidate_after_delay.assert_awaited_once()
        candidate = self.app._promote_candidate_after_delay.await_args.args[2]
        self.assertEqual(candidate.stored_message_id, rows[0]['message_id'])
        await self.ingest(update, reply=False)
        self.assertEqual(len([row async for row in audit_records(self.store, self.session)]), 1,
                         'Telegram redelivery must not create a second original revision')
        self.app._ensure_reply_worker.assert_awaited_once()

    async def test_metadata_option_off_preserves_literal_user_metadata_text_and_identity(self):
        await self.settings(metadata_injection_mode='off')
        text = '[Message metadata: this is what I typed, not an application header]'
        await self.ingest(self.update(text), reply=False)
        original = (await self.store.list_canonical_messages(self.session))[0].message
        self.assertEqual([(part.text, part.origin) for part in original.parts], [(text, None)])
        self.assertEqual(original.metadata['actor_id'], 'telegram:user:7')
        self.assertEqual(original.metadata['source_message_id'], '1')

    async def test_unrelated_actor_edit_between_scope_capture_and_initial_save_does_not_drop_new_input(self):
        await self.ingest(self.update('Alice original.', source_id=1, actor=7), reply=False)
        entered, release = asyncio.Event(), asyncio.Event()
        ingest = self.runtime.ingest_user_message
        first_bob = True
        async def interleaved(**kwargs):
            nonlocal first_bob
            if kwargs['incoming_message'].metadata['actor_id'] == 'telegram:user:8' and first_bob:
                first_bob = False
                entered.set()
                await release.wait()
            return await ingest(**kwargs)
        with patch.object(self.runtime, 'ingest_user_message', side_effect=interleaved):
            bob = asyncio.create_task(self.ingest(self.update('Bob independent message.', source_id=2, actor=8), reply=False))
            self.addAsyncCleanup(self._finish_task, bob, release)
            await asyncio.wait_for(entered.wait(), timeout=3)
            await self.ingest(self.update('Alice corrected.', source_id=1, actor=7, edited=True), reply=False, edit=True)
            release.set()
            await asyncio.wait_for(bob, timeout=3)
        messages = await self.store.list_canonical_messages(self.session)
        self.assertEqual(len(messages), 2)
        by_actor = {item.message.metadata['actor_id']: item for item in messages}
        self.assertIn('Alice corrected.', '\n'.join(part.text or '' for part in by_actor['telegram:user:7'].message.parts))
        self.assertIn('Bob independent message.', '\n'.join(part.text or '' for part in by_actor['telegram:user:8'].message.parts))
        self.assertEqual(by_actor['telegram:user:8'].message.metadata['source_revision'], 1)

    async def test_successful_document_enrichment_keeps_same_source_and_existing_remote_upload(self):
        async def download(buffer):
            rows = [row async for row in audit_records(self.store, self.session)]
            self.assertEqual(len(rows), 1)
            self.assertIn('notes.txt', rows[0]['body'])
            buffer.write(b'Synthetic document text.')
        file = SimpleNamespace(download_to_memory=AsyncMock(side_effect=download))
        document = SimpleNamespace(file_id='synthetic-document', file_unique_id='unique-document',
            file_name='notes.txt', mime_type='text/plain', file_size=24, get_file=AsyncMock(return_value=file))
        uploaded = []
        async def sync(session, paths, *, sent_at=None, filenames=None):
            uploaded.extend(paths)
            self.assertTrue(all(path.exists() for path in paths))
            self.assertEqual(list(filenames.values()), ['notes.txt'])
            return RemoteSyncResult({str(path.resolve()): '/remote/2026-01-01/notes_0123456789abcdef.txt' for path in paths})
        self.app.remote_workspace = SimpleNamespace(enabled=True,
            session_paths=lambda session: SimpleNamespace(root='/remote'), sync_inputs=AsyncMock(side_effect=sync))
        update = self.update(None, caption='Read these notes.', document=document)
        await self.ingest(update)
        await self.ingest(update)
        rows = [row async for row in audit_records(self.store, self.session)]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['message_id'], rows[1]['message_id'])
        self.assertIn('enrichment pending', rows[0]['body'])
        self.assertNotIn('Synthetic document text.', rows[1]['body'])
        self.assertIn('Attachment synced to remote for tool use', rows[1]['body'])
        current = (await self.store.list_canonical_messages(self.session))[0]
        remote_parts = [part for part in current.message.parts if part.kind == PartKind.FILE]
        self.assertEqual(len(remote_parts), 1)
        self.assertEqual(remote_parts[0].artifact_path, '/remote/2026-01-01/notes_0123456789abcdef.txt')
        self.assertEqual(rows[1]['metadata']['telegram_attachments'][0]['file_id'], 'synthetic-document')
        self.app.remote_workspace.sync_inputs.assert_awaited_once()
        self.assertTrue(uploaded and all(not path.exists() for path in uploaded), 'Temporary downloads keep their existing cleanup behavior')
        self.app._promote_candidate_after_delay.assert_awaited_once()

    async def test_identical_photo_redelivery_serializes_download_and_preserves_enriched_revision(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def download(buffer):
            entered.set()
            await release.wait()
            Image.new('RGB', (2, 2), 'blue').save(buffer, format='PNG')
        photo = self.photo(download)
        update = self.update(None, caption='A blue photo.', photo=[photo])
        first = asyncio.create_task(self.ingest(update))
        self.addAsyncCleanup(self._finish_task, first, release)
        await asyncio.wait_for(entered.wait(), timeout=3)
        duplicate = asyncio.create_task(self.ingest(update))
        self.addAsyncCleanup(self._finish_task, duplicate, release)
        await asyncio.sleep(0)
        release.set()
        await asyncio.wait_for(asyncio.gather(first, duplicate), timeout=3)
        # A later redelivery follows the persisted completion cache as well.
        await self.ingest(update)
        photo.get_file.assert_awaited_once()
        rows = [row async for row in audit_records(self.store, self.session)]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[-1]['metadata']['telegram_intake_stage'], 'complete')
        self.assertEqual(len(await self.store.list_canonical_messages(self.session)), 1)
        self.app._promote_candidate_after_delay.assert_awaited_once()
        self.assertEqual(self.app._intake_inflight, {})

    async def test_delayed_document_and_edit_keep_original_date_and_exact_remote_reference(self):
        dates = {'2026-04-30T15:59:00+00:00': '2026-04-30',
                 '2026-04-30T16:01:00+00:00': '2026-05-01'}
        uploaded, observed = {}, []
        async def sync(session, paths, *, sent_at=None, filenames=None):
            self.assertEqual(session, self.session)
            observed.append(sent_at)
            receipts = {}
            for path in paths:
                self.assertEqual(filenames[str(path.resolve())], 'My report.txt')
                identifier = '1111111111111111' if path.read_bytes() == b'Original file content' else '2222222222222222'
                destination = f'/remote/{dates[sent_at]}/My report_{identifier}.txt'
                uploaded[destination] = path.read_bytes()
                receipts[str(path.resolve())] = destination
            return RemoteSyncResult(receipts)
        self.app.remote_workspace = SimpleNamespace(enabled=True, sync_inputs=AsyncMock(side_effect=sync),
            session_paths=lambda session: SimpleNamespace(root='/remote'))
        raw = b'Original file content'
        async def download(buffer):
            buffer.write(raw)
        document = SimpleNamespace(file_id='dated-document', file_unique_id='dated-file',
            file_name='My report.txt', mime_type='text/plain', file_size=21,
            get_file=AsyncMock(return_value=SimpleNamespace(download_to_memory=AsyncMock(side_effect=download))))
        for number, sent_at in enumerate(dates, 1):
            update = self.update(None, source_id=number, actor=number + 10,
                caption=f'Read report {number}.', document=document)
            update.effective_message.date = datetime.fromisoformat(sent_at)
            await self.ingest(update, reply=False)
        edited = self.update(None, source_id=1, actor=11,
            caption='Corrected report caption.', document=document, edited=True)
        edited.effective_message.date = datetime.fromisoformat(next(iter(dates)))
        edited.effective_message.edit_date = datetime(2026, 5, 2, tzinfo=timezone.utc)
        raw = b'Updated file content'
        await self.ingest(edited, reply=False, edit=True)
        await self.ingest(edited, reply=False, edit=True)
        reopened = await self.new_store()
        restored = await reopened.list_canonical_messages(self.session)
        self.assertEqual(len(restored), 2)
        self.assertEqual(observed, [*dates, next(iter(dates))])
        self.assertEqual(len(uploaded), 3)
        self.assertEqual(uploaded['/remote/2026-04-30/My report_1111111111111111.txt'], b'Original file content')
        for row in restored:
            message = row.message
            number = int(message.metadata['source_message_id'])
            attachment = next(part for part in message.parts if part.kind == PartKind.FILE)
            identifier = '2222222222222222' if number == 1 else '1111111111111111'
            expected = f"/remote/{dates[message.metadata['sent_at']]}/My report_{identifier}.txt"
            self.assertEqual(attachment.artifact_path, expected)
            self.assertEqual(attachment.filename, 'My report.txt')
            self.assertEqual(message.metadata['actor_id'], f'telegram:user:{number + 10}')
            note = next(part.text for part in message.parts if part.origin == 'auto_note')
            self.assertIn('paths relative to workspace', note)
            self.assertIn(expected.removeprefix('/remote/'), note)
            self.assertEqual(uploaded[expected], b'Updated file content' if number == 1 else b'Original file content')
        self.assertFalse([path for path in self.artifact_store.root.rglob('*') if path.is_file()])
        self.assertFalse(self.provider.requests)

    async def test_changed_attachment_caption_is_a_new_revision_not_a_cached_redelivery(self):
        async def download(buffer):
            Image.new('RGB', (2, 2), 'blue').save(buffer, format='PNG')
        photo = self.photo(download)
        await self.ingest(self.update(None, caption='Original caption.', photo=[photo]), reply=False)
        await self.ingest(self.update(None, caption='Corrected caption.', photo=[photo], edited=True), reply=False, edit=True)
        # Delayed redelivery of the original must not upload stale media again.
        await self.ingest(self.update(None, caption='Original caption.', photo=[photo]), reply=False)
        rows = [row async for row in audit_records(self.store, self.session)]
        self.assertEqual(len(rows), 4)
        self.assertIn('Original caption.', rows[0]['body'])
        self.assertIn('Corrected caption.', rows[-1]['body'])
        self.assertEqual(photo.get_file.await_count, 2)
        self.assertEqual(len(await self.store.list_canonical_messages(self.session)), 1)

    async def test_download_failure_retains_caption_searchable_file_reference_and_reply_trigger(self):
        async def failed_download(buffer):
            raise TimeoutError('synthetic download timeout')
        photo = self.photo(failed_download)
        await self.ingest(self.update(None, caption='Keep this caption.', photo=[photo]))
        rows = [row async for row in audit_records(self.store, self.session)]
        self.assertEqual(len(rows), 2)
        self.assertIn('Keep this caption.', rows[-1]['body'])
        self.assertIn('Attachment download failed: TimeoutError', rows[-1]['body'])
        self.assertIn('Attachment reference', rows[-1]['body'])
        self.assertIn('media content unavailable', rows[-1]['body'])
        self.assertEqual(rows[-1]['metadata']['telegram_attachments'][0]['file_id'], 'synthetic-photo-file')
        original = (await self.store.list_canonical_messages(self.session))[0].message
        self.assertEqual([part.origin for part in original.parts
                          if part.text and part.text.startswith('[Attachment download failed:')], ['auto_note'])
        self.assertEqual([part.origin for part in original.parts
                          if part.text and part.text.startswith('[Message metadata:')], ['provenance'])
        self.app._promote_candidate_after_delay.assert_awaited_once()
