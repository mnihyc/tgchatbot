"""Available Desktop media enters durable visual memory without Telegram work."""
from __future__ import annotations

from dataclasses import replace
import io
import json
from unittest.mock import patch

import av
from PIL import Image

from tests.business_helpers import BusinessTestCase
from tests.test_desktop_import import export, record
from tgchatbot.domain.models import PartKind
from tgchatbot.storage.artifacts import ArtifactStore
from tgchatbot.storage.postgres_store import StaleScopeError, message_body
from tgchatbot.tools.import_desktop import import_file


class DesktopImageImportTests(BusinessTestCase):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.bundle = self.path / 'export'
        self.bundle.mkdir()
        self.export_path = self.bundle / 'result.json'

    def image(self, filename, *, format='PNG', color='red'):
        path = self.bundle / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        with Image.new('RGB', (16, 16), color) as image:
            image.save(path, format=format)
        return path

    def animation(self, filename):
        path = self.bundle / filename
        with Image.new('RGB', (16, 16), 'red') as first, Image.new('RGB', (16, 16), 'blue') as second:
            first.save(path, 'GIF', save_all=True, append_images=[second], duration=[100, 100], loop=0)
        return path

    def video(self, filename):
        path = self.bundle / filename
        with av.open(str(path), 'w') as container:
            stream = container.add_stream('mpeg4', rate=2)
            stream.width = stream.height = 16
            stream.pix_fmt = 'yuv420p'
            stream.codec_context.gop_size = 1
            for color in ('red', 'blue'):
                with Image.new('RGB', (16, 16), color) as image:
                    frame = av.VideoFrame.from_image(image)
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode():
                container.mux(packet)
        return path

    async def ingest(self, records, **settings):
        self.export_path.write_text(json.dumps(export(records), ensure_ascii=False), encoding='utf-8')
        return await import_file(self.store, self.export_path, chat_id=100,
            telegram_config=replace(self.config.telegram, **settings))

    async def images(self):
        originals = await self.store.list_canonical_messages(self.session)
        references = [part.preview_ref for row in originals for part in row.message.parts if part.preview_ref]
        payloads = await self.store.load_preview_data(self.session, references)
        return originals, payloads

    async def test_local_images_and_video_share_live_decoding_and_survive_database_reopen(self):
        self.image('photos/plain.jpg', format='JPEG')
        self.image('arbitrary.attachment', format='PNG', color='green')
        self.image('sample.tiff', format='TIFF', color='blue')
        self.image('sample.webp', format='WEBP', color='yellow')
        self.video('clip.mp4')
        records = [record(1, 'The original caption.', photo='photos/plain.jpg')]
        for number, name, mime in [(2, 'arbitrary.attachment', 'image/png'), (3, 'sample.tiff', 'image/tiff'),
                                   (4, 'sample.webp', 'image/webp'), (5, 'clip.mp4', 'video/mp4')]:
            records.append(record(number, '', file=name, file_name=name, mime_type=mime, media_type='document'))
        # Import can run beside existing transfer artifacts owned by intake.
        ArtifactStore(self.config.artifact_dir).save_bytes(chat_id=self.session,
            filename='existing-transfer.bin', data=b'Unrelated transfer remains intact.')

        def artifacts():
            return {path.relative_to(self.config.artifact_dir): path.read_bytes() if path.is_file() else None
                    for path in self.config.artifact_dir.rglob('*')}

        artifacts_before = artifacts()
        # Actual decoding and database writes are allowed; historical import has
        # no Telegram, external lookup or remote-workspace side effects.
        with patch('socket.getaddrinfo', side_effect=AssertionError('unexpected network lookup')), \
                patch('asyncio.create_subprocess_exec', side_effect=AssertionError('unexpected process')):
            await self.ingest(records, max_visual_file_frames=2)
        before, payloads = await self.images()
        self.assertEqual(len(before), 5)
        self.assertEqual(before[0].message.parts[0].text, 'The original caption.')
        for row in before:
            visuals = [part for part in row.message.parts if part.kind == PartKind.IMAGE]
            self.assertTrue(visuals)
            self.assertEqual(row.message.metadata['media_availability'], 'imported')
            self.assertTrue(all(part.preview_ref and not part.remote_sync for part in visuals))
            for visual in visuals:
                self.assertIn(visual.preview_ref, payloads)
                with Image.open(io.BytesIO(payloads[visual.preview_ref])) as decoded:
                    self.assertGreater(decoded.width, 0)
                    self.assertIn(decoded.format, {'PNG', 'JPEG'})
        self.assertEqual(len([p for p in before[-1].message.parts if p.preview_ref]), 2)
        await self.store.close()
        self.store = await self.new_store()
        after, reopened = await self.images()
        self.assertEqual([row.message for row in after], [row.message for row in before])
        self.assertEqual(reopened, payloads)
        self.assertEqual(artifacts(), artifacts_before,
                         'Visual import must leave existing transfer artifacts unchanged without adding copies')

    async def test_image_only_originals_keep_separate_photo_sticker_and_file_frame_settings(self):
        self.animation('animated.gif')
        records = [record(1, '', photo='animated.gif'),
                   record(2, '', file='animated.gif', media_type='sticker', mime_type='image/gif', sticker_emoji='🙂'),
                   record(3, '', file='animated.gif', media_type='document', mime_type='image/gif')]
        await self.ingest(records, max_sticker_frames=1, max_visual_file_frames=2)
        originals, payloads = await self.images()
        frames = [[part for part in row.message.parts if part.preview_ref] for row in originals]
        self.assertEqual([len(parts) for parts in frames], [1, 1, 2])
        self.assertEqual(originals[1].message.metadata['desktop']['sticker_emoji'], '🙂')
        self.assertIn('🙂', message_body(originals[1].message))
        self.assertEqual([row.message.parts[0].text for row in originals], ['', '', ''])
        self.assertTrue(all(part.kind in {PartKind.TEXT, PartKind.FILE} or part.preview_ref for row in originals for part in row.message.parts),
                        'Successful import must not leave an extra unavailable visual placeholder')
        self.assertEqual(frames[0][0].preview_ref, frames[1][0].preview_ref)
        self.assertEqual(frames[1][0].preview_ref, frames[2][0].preview_ref)
        with Image.open(io.BytesIO(payloads[frames[1][0].preview_ref])) as decoded:
            red, _, blue = decoded.convert('RGB').getpixel((0, 0))
            self.assertGreater(red, blue)
        self.assertEqual(frames[1][0].detail, 'low')
        self.assertEqual(frames[0][0].detail, 'auto')
        await self.store.retire_context_images(self.session, target_images=0)
        working = await self.store.list_messages(self.session)
        canonical, retained = await self.images()
        self.assertIn('🙂', message_body(working[1]))
        self.assertIn('🙂', message_body(canonical[1].message))
        self.assertEqual([row.message for row in canonical], [row.message for row in originals])
        self.assertEqual(retained, payloads)

    async def test_missing_unreadable_and_external_images_keep_structured_references(self):
        (self.bundle / 'corrupt.png').write_bytes(b'not decodable visual media')
        (self.bundle / 'notes.pdf').write_bytes(b'%PDF synthetic unsupported import document')
        with Image.new('RGB', (16, 16), 'red') as image:
            image.save(self.path / 'outside.png')
        omitted = '(File not included. Change data exporting settings to download.)'
        records = [record(1, '', photo=omitted),
                   record(2, '', file=omitted, media_type='sticker', file_name='sticker.webp', sticker_emoji='🙂'),
                   record(3, '', file='corrupt.png', mime_type='image/png', media_type='document'),
                   record(4, '', file='missing.png', mime_type='image/png', media_type='document'),
                   record(5, '', photo='../outside.png'),
                   record(6, 'Keep the document reference.', file='notes.pdf', media_type='document', mime_type='application/pdf')]
        await self.ingest(records)
        originals, payloads = await self.images()
        self.assertEqual(len(originals), len(records))
        self.assertEqual(payloads, {})
        for row in originals[:5]:
            placeholders = [part for part in row.message.parts if part.kind in {PartKind.IMAGE, PartKind.STICKER}]
            self.assertEqual(len(placeholders), 1)
            self.assertEqual(placeholders[0].origin, 'attachment_reference')
            self.assertFalse(placeholders[0].data_b64)
            self.assertFalse(placeholders[0].remote_sync)
            self.assertIn('unavailable', placeholders[0].text)
        self.assertEqual(originals[1].message.parts[1].kind, PartKind.STICKER)
        self.assertIn('🙂', message_body(originals[1].message))
        self.assertEqual([part.kind for part in originals[-1].message.parts], [PartKind.TEXT, PartKind.FILE])
        self.assertEqual(originals[-1].message.metadata['desktop']['file'], 'notes.pdf')
        self.assertEqual((self.bundle / 'notes.pdf').read_bytes(), b'%PDF synthetic unsupported import document')

    async def test_reimport_reuses_originals_and_compressed_payloads(self):
        self.image('shared.png')
        records = [record(1, 'First use.', photo='shared.png'), record(2, 'Another person uses it.', from_id='user22', photo='shared.png')]
        await self.ingest(records)
        before, payloads = await self.images()
        await self.ingest(records)
        after, reloaded = await self.images()
        self.assertEqual([row.db_id for row in after], [row.db_id for row in before])
        self.assertEqual([row.message for row in after], [row.message for row in before])
        self.assertEqual(payloads, reloaded)
        self.assertEqual(len(payloads), 1)
        self.assertEqual({row.message.metadata['actor_id'] for row in after}, {'telegram:user:11', 'telegram:user:22'})
        for row in after:
            self.assertEqual(len(await self.store.list_message_revisions(self.session, row.db_id)), 1)

    async def test_sticker_thumbnail_is_retained_when_original_cannot_be_shown(self):
        (self.bundle / 'animated.tgs').write_bytes(b'Unsupported sticker animation')
        self.image('thumbnails/sticker.jpg', format='JPEG')
        records = [record(1, 'Original caption.', file='animated.tgs', media_type='sticker',
                          thumbnail='thumbnails/sticker.jpg', sticker_emoji='🙂'),
                   record(2, '', file='missing.tgs', media_type='sticker',
                          thumbnail='thumbnails/sticker.jpg', sticker_emoji='🙂')]
        await self.ingest(records)
        before, payloads = await self.images()
        self.assertEqual(len(payloads), 1)
        for row, original in zip(before, records):
            self.assertEqual(row.message.metadata['desktop']['file'], original['file'])
            self.assertEqual(row.message.parts[0].text, original['text'])
            self.assertIn('🙂', message_body(row.message))
            self.assertIn('export thumbnail only', message_body(row.message))
            self.assertIn('animation unavailable', message_body(row.message))
            self.assertEqual(len([part for part in row.message.parts if part.preview_ref]), 1)
            self.assertFalse(any(part.remote_sync for part in row.message.parts))
        await self.ingest(records)
        after, reloaded = await self.images()
        self.assertEqual([row.db_id for row in after], [row.db_id for row in before])
        self.assertEqual([row.message for row in after], [row.message for row in before])
        self.assertEqual(payloads, reloaded)
        for row in after:
            self.assertEqual(len(await self.store.list_message_revisions(self.session, row.db_id)), 1)
        await self.store.retire_context_images(self.session, target_images=0)
        await self.store.close()
        self.store = await self.new_store()
        canonical, retained = await self.images()
        self.assertEqual([row.message for row in canonical], [row.message for row in before])
        self.assertEqual(retained, payloads)

    async def test_sticker_thumbnails_do_not_override_originals_limits_or_export_boundary(self):
        self.image('original.webp', format='WEBP', color='red')
        self.image('thumbnail.png', color='blue')
        (self.bundle / 'corrupt.png').write_bytes(b'Corrupt thumbnail')
        with Image.new('RGB', (16, 16), 'green') as image:
            image.save(self.path / 'outside.png')
        (self.bundle / 'outside-link.png').symlink_to(self.path / 'outside.png')
        records = [record(1, '', file='original.webp', media_type='sticker', thumbnail='thumbnail.png')]
        for number, thumbnail in enumerate(('missing.png', '../outside.png', 'outside-link.png', 'corrupt.png'), 2):
            records.append(record(number, '', file='missing.tgs', media_type='sticker', thumbnail=thumbnail))
        await self.ingest(records)
        originals, payloads = await self.images()
        self.assertEqual(len(payloads), 1)
        self.assertNotIn('export thumbnail only', message_body(originals[0].message))
        with Image.open(io.BytesIO(next(iter(payloads.values())))) as decoded:
            red, _, blue = decoded.convert('RGB').getpixel((0, 0))
            self.assertGreater(red, blue, 'The original must remain the preferred visual evidence')
        for row in originals[1:]:
            self.assertFalse(any(part.preview_ref for part in row.message.parts))
            self.assertIn('unavailable', message_body(row.message))
        limited = [record(6, '', file='missing.tgs', media_type='sticker', thumbnail='thumbnail.png')]
        await self.ingest(limited, max_sticker_frames=0)
        limited += [record(7, '', file='missing.tgs', media_type='sticker', thumbnail='thumbnail.png')]
        await self.ingest(limited, max_sticker_bytes=1)
        originals, limited_payloads = await self.images()
        self.assertEqual(limited_payloads, payloads)
        self.assertTrue(all(not any(part.preview_ref for part in row.message.parts) for row in originals[-2:]))

    async def test_configured_visual_limits_keep_unavailable_occurrence_without_discarding_caption(self):
        self.image('photo.png')
        self.animation('sticker.gif')
        records = [record(1, 'Caption remains.', photo='photo.png'),
                   record(2, 'Sticker caption remains.', file='sticker.gif', media_type='sticker', mime_type='image/gif'),
                   record(3, 'File caption remains.', file='photo.png', media_type='document', mime_type='image/png')]
        await self.ingest(records, max_photo_bytes=1, max_sticker_frames=0, max_document_bytes=1)
        originals, payloads = await self.images()
        self.assertEqual(payloads, {})
        self.assertEqual([row.message.parts[0].text for row in originals], [row['text'] for row in records])
        self.assertTrue(all(any(part.kind in {PartKind.IMAGE, PartKind.STICKER} for part in row.message.parts) for row in originals))

    async def test_reimport_without_local_pixels_does_not_downgrade_retained_original_image(self):
        image = self.image('retained.png')
        records = [record(1, 'Keep this original image.', photo='retained.png')]
        await self.ingest(records)
        before, payloads = await self.images()
        image.rename(self.bundle / 'not-in-this-export.png')
        await self.ingest(records)
        after, reloaded = await self.images()
        self.assertEqual(after[0].message, before[0].message)
        self.assertEqual(reloaded, payloads)

    async def test_reimport_adds_formerly_missing_pixels_and_does_not_ignore_changed_originals(self):
        records = [record(1, 'Original caption.', photo='later.png')]
        await self.ingest(records)
        missing, _ = await self.images()
        self.image('later.png')
        await self.ingest(records)
        available, payloads = await self.images()
        self.assertEqual(available[0].db_id, missing[0].db_id)
        self.assertTrue(payloads)
        self.assertTrue(any(part.preview_ref for part in available[0].message.parts))
        records[0] = record(1, 'Corrected caption and another image.', photo='different.png', edited_unixtime='1735776000')
        await self.ingest(records)
        changed, _ = await self.images()
        self.assertEqual(changed[0].message.parts[0].text, records[0]['text'])
        self.assertEqual(changed[0].message.metadata['desktop']['photo'], 'different.png')
        self.assertFalse(any(part.preview_ref for part in changed[0].message.parts))
        self.assertEqual(len(await self.store.list_message_revisions(self.session, changed[0].db_id)), 3)

    async def test_full_reset_during_retained_image_noop_stops_the_import(self):
        image = self.image('retained.png')
        records = [record(1, 'Keep this original image.', photo='retained.png')]
        await self.ingest(records)
        image.rename(self.bundle / 'not-in-this-export.png')
        describe = self.store.describe_message_images

        async def describe_and_reset(*args, **kwargs):
            result = await describe(*args, **kwargs)
            await self.store.reset_full(self.session, self.config.default_session_settings())
            return result

        with patch.object(self.store, 'describe_message_images', side_effect=describe_and_reset):
            with self.assertRaises(StaleScopeError):
                await self.ingest(records)
        self.assertEqual(await self.store.list_canonical_messages(self.session), [])

    async def test_soft_reset_during_retained_image_noop_keeps_searchable_original(self):
        image = self.image('retained.png')
        records = [record(1, 'Keep this original image.', photo='retained.png')]
        await self.ingest(records)
        before, payloads = await self.images()
        image.rename(self.bundle / 'not-in-this-export.png')
        describe = self.store.describe_message_images

        async def describe_and_reset(*args, **kwargs):
            result = await describe(*args, **kwargs)
            await self.store.reset_context(self.session)
            return result

        with patch.object(self.store, 'describe_message_images', side_effect=describe_and_reset):
            result = await self.ingest(records)
        after, reloaded = await self.images()
        self.assertEqual(result.messages, 1)
        self.assertEqual(after[0].message, before[0].message)
        self.assertEqual(reloaded, payloads)
