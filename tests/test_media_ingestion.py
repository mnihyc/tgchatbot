"""Live attachment previews preserve originals and stay within configured work."""
from __future__ import annotations

import base64
from dataclasses import replace
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

from PIL import Image

from tgchatbot.config import load_config
from tgchatbot.domain.models import PartKind
from tgchatbot.media.ingest import extract_message_parts
from tgchatbot.storage.artifacts import ArtifactStore


class MediaIngestionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='media-input-', dir=Path(__file__).parent)
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name)
        self.artifacts = ArtifactStore(self.path / 'transfers')
        with patch.dict(os.environ, {'APP_DATA_DIR': str(self.path), 'TGBOT_TOKEN': 'synthetic-token'}, clear=True):
            self.config = load_config().telegram

    @staticmethod
    def animation():
        with Image.new('RGB', (8, 8), 'red') as first, Image.new('RGB', (8, 8), 'blue') as second:
            output = io.BytesIO()
            first.save(output, 'GIF', save_all=True, append_images=[second], duration=[100, 100], loop=0)
            return output.getvalue()

    @staticmethod
    def attachment(raw, filename='clip.mp4', mime='video/mp4'):
        async def download(buffer):
            buffer.write(raw)
        file = SimpleNamespace(download_to_memory=AsyncMock(side_effect=download))
        return SimpleNamespace(file_name=filename, mime_type=mime, file_size=len(raw),
            get_file=AsyncMock(return_value=file), emoji='🙂', is_video=False, is_animated=True)

    async def ingest(self, field, attachment, **settings):
        message = SimpleNamespace(text=None, caption='Please look at this.', photo=None,
            sticker=None, animation=None, video=None, document=None)
        setattr(message, field, [attachment] if field == 'photo' else attachment)
        return await extract_message_parts(message, self.artifacts, 'telegram:fixture', replace(self.config, **settings))

    @staticmethod
    def pixel(part):
        with Image.open(io.BytesIO(base64.b64decode(part.data_b64))) as image:
            return image.convert('RGB').getpixel((0, 0))

    def assert_original(self, parts, raw):
        originals = [part for part in parts if part.kind == PartKind.FILE]
        self.assertEqual(len(originals), 1)
        self.assertTrue(originals[0].remote_sync)
        self.assertEqual(Path(originals[0].artifact_path).read_bytes(), raw)

    async def test_one_frame_animated_sticker_supplies_first_image_and_keeps_hint(self):
        raw = self.animation()
        with patch('tgchatbot.media.ingest.av.open') as decoder:
            parts = await self.ingest('sticker', self.attachment(raw, 'sticker.gif', 'image/gif'), max_sticker_frames=1)
        decoder.assert_not_called()
        images = [part for part in parts if part.kind == PartKind.IMAGE]
        self.assertEqual(len(images), 1)
        red, _, blue = self.pixel(images[0])
        self.assertGreater(red, blue)
        self.assertFalse(images[0].remote_sync)
        self.assertEqual([part.text for part in parts if part.kind == PartKind.STICKER], ['[User sent animated sticker 🙂]'])
        self.assertEqual(list(self.artifacts.root.rglob('*')), [], 'Inline previews must not create transfer artifacts')

    async def test_one_frame_animated_document_keeps_original_and_first_preview(self):
        raw = self.animation()
        parts = await self.ingest('document', self.attachment(raw, 'animation.gif', 'image/gif'), max_visual_file_frames=1)
        self.assert_original(parts, raw)
        images = [part for part in parts if part.kind == PartKind.IMAGE]
        self.assertEqual(len(images), 1)
        red, _, blue = self.pixel(images[0])
        self.assertGreater(red, blue)

    async def test_disabled_visual_previews_keep_original_video_animation_and_documents(self):
        raw = self.animation()
        for field, filename, mime in [('video', 'clip.mp4', 'video/mp4'),
                ('animation', 'animation.gif', 'image/gif'), ('document', 'photo.png', 'image/png'),
                ('document', 'clip.mp4', 'video/mp4')]:
            with self.subTest(field=field, mime=mime), patch('tgchatbot.media.ingest.Image.open') as decoder:
                parts = await self.ingest(field, self.attachment(raw, filename, mime), max_visual_file_frames=0)
                self.assert_original(parts, raw)
                self.assertFalse(any(part.kind == PartKind.IMAGE for part in parts))
                decoder.assert_not_called()

    async def test_audio_and_voice_preserve_originals_without_visual_or_transcription_claims(self):
        raw = b'synthetic undecoded audio bytes, not spoken text'
        for field, filename, mime in [('audio', 'recording.mp3', 'audio/mpeg'), ('voice', 'voice.ogg', 'audio/ogg')]:
            with self.subTest(field=field), patch('tgchatbot.media.ingest.Image.open') as decoder:
                parts = await self.ingest(field, self.attachment(raw, filename, mime))
                self.assert_original(parts, raw)
                self.assertEqual([part.kind for part in parts], [PartKind.TEXT, PartKind.FILE])
                self.assertEqual(parts[0].text, 'Please look at this.')
                decoder.assert_not_called()

    async def test_video_note_is_file_only_even_when_visual_previews_are_enabled(self):
        raw = self.animation()
        media = self.attachment(raw)
        del media.file_name
        del media.mime_type
        with patch('tgchatbot.media.ingest.Image.open') as image_decoder, \
             patch('tgchatbot.media.ingest.av.open') as video_decoder:
            parts = await self.ingest('video_note', media, max_visual_file_frames=3)
        image_decoder.assert_not_called()
        video_decoder.assert_not_called()
        self.assert_original(parts, raw)
        self.assertEqual([part.kind for part in parts], [PartKind.TEXT, PartKind.FILE])
        self.assertEqual(parts[-1].mime_type, 'video/mp4')

    async def test_original_size_allowance_stops_oversized_audio_before_download(self):
        media = self.attachment(b'oversized synthetic audio', 'voice.ogg', 'audio/ogg')
        parts = await self.ingest('voice', media, max_document_bytes=1)
        media.get_file.assert_not_awaited()
        self.assertFalse(any(part.kind == PartKind.FILE for part in parts))
        self.assertTrue(any('exceeds size limit' in (part.text or '') for part in parts))

    async def test_no_video_stream_keeps_original_without_leaking_container(self):
        container = Mock()
        container.streams.video = []
        context = Mock(__enter__=Mock(return_value=container), __exit__=Mock(return_value=False))
        raw = b'synthetic audio-only container'
        with patch('tgchatbot.media.ingest.av.open', return_value=context):
            parts = await self.ingest('video', self.attachment(raw))
        self.assert_original(parts, raw)
        self.assertFalse(any(part.kind == PartKind.IMAGE for part in parts))
        context.__exit__.assert_called_once()
        container.decode.assert_not_called()

    async def test_video_candidate_budget_keeps_chronological_sample_without_retaining_all_images(self):
        timestamps = [40, 0, 30, 10, 20, 50]
        colors = ['white', 'red', 'white', 'white', 'blue', 'white']
        frames = [SimpleNamespace(pts=at, to_image=Mock(side_effect=lambda color=color: Image.new('RGB', (8, 8), color)))
                  for at, color in zip(timestamps, colors)]
        contexts = []
        def open_video(_source):
            container = SimpleNamespace(streams=SimpleNamespace(video=[SimpleNamespace(skip_frame=None)]),
                decode=Mock(return_value=iter(frames)))
            context = Mock(__enter__=Mock(return_value=container), __exit__=Mock(return_value=False))
            contexts.append(context)
            return context
        raw = b'synthetic keyframe video'
        with patch('tgchatbot.media.ingest.av.open', side_effect=open_video):
            parts = await self.ingest('video', self.attachment(raw), max_visual_file_frames=2, max_video_keyframe_candidates=5)
        self.assert_original(parts, raw)
        images = [part for part in parts if part.kind == PartKind.IMAGE]
        self.assertEqual(len(images), 2)
        first, second = map(self.pixel, images)
        self.assertGreater(first[0], first[2])
        self.assertGreater(second[2], second[0])
        self.assertEqual([frame.to_image.call_count for frame in frames], [0, 1, 0, 0, 1, 0])
        self.assertEqual(len(contexts), 2)
        for context in contexts:
            context.__exit__.assert_called_once()

    async def test_preview_encoder_failure_keeps_sticker_hint_and_closes_decoded_images(self):
        raw = self.animation()
        closed = []
        original_close = Image.Image.close
        def failed_save(_image, *_args, **_kwargs):
            raise OSError('synthetic encoder failure')
        def close(image):
            closed.append(image)
            return original_close(image)
        with patch.object(Image.Image, 'save', failed_save), patch.object(Image.Image, 'close', close):
            parts = await self.ingest('sticker', self.attachment(raw, 'sticker.gif', 'image/gif'), max_sticker_frames=1)
        self.assertTrue(any(part.kind == PartKind.STICKER for part in parts))
        self.assertFalse(any(part.kind == PartKind.IMAGE for part in parts))
        self.assertTrue(any('preview omitted' in (part.text or '').lower() for part in parts))
        self.assertGreaterEqual(len(closed), 2, 'Decoded frame and encoder copy both release their pixel buffers')
