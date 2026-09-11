"""Original media evidence and explicit maintenance selections."""
from __future__ import annotations
import io
from pathlib import Path
import tempfile
import unittest
from PIL import Image

from tgchatbot.stickers.build import BuildConfig, parse_args
from tgchatbot.stickers.media import MediaConfig, prepare_media


class StickerMaintenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='fixture-media-', dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_static_caption_art_has_no_claimed_animation(self):
        path = self.root/'static.png'
        Image.new('RGBA', (80, 40), (255, 0, 0, 100)).save(path)
        original = path.read_bytes()
        media = prepare_media(path, MediaConfig(max_dimension=20))
        self.assertFalse(media.facts['animated'])
        self.assertEqual(media.facts['supplied_frames'], 1)
        self.assertEqual((media.frames[0].width, media.frames[0].height), (20, 10))
        self.assertEqual(path.read_bytes(), original)

    def test_variable_duration_animation_samples_time_and_final_short_expression(self):
        path = self.root/'animation.gif'
        frames = [Image.new('RGB', (30, 30), color) for color in ['red', 'green', 'blue', 'yellow']]
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=[800, 100, 100, 20], loop=0)
        media = prepare_media(path, MediaConfig(max_frames=3))
        self.assertTrue(media.facts['animated'])
        # Midpoint is still the first frame, so image dedup leaves first and final.
        self.assertEqual(len(media.frames), 2)
        self.assertEqual(media.frames[0].timestamp_s, 0)
        self.assertAlmostEqual(media.frames[-1].timestamp_s, 1.0)
        last = Image.open(io.BytesIO(media.frames[-1].data)).getpixel((0, 0))
        self.assertGreater(last[0], 200)
        self.assertGreater(last[1], 200)
        self.assertLess(last[2], 30)
        self.assertIn('may be omitted', media.facts['sampling'])

    def test_exact_alias_pack_and_content_id_are_explicit_selective_requests(self):
        args = parse_args(['--file', 'pack/a.webp', '--pack', 'other', '--asset-id', 'sha256:test'])
        self.assertEqual(args.file, ['pack/a.webp'])
        self.assertEqual(args.pack, ['other'])
        self.assertEqual(args.asset_id, ['sha256:test'])

    def test_operator_can_disable_flex_without_changing_model_or_extra_env(self):
        self.assertEqual(BuildConfig.from_env({}).service_tier, 'flex')
        self.assertEqual(BuildConfig.from_env({'STICKER_BUILD_SERVICE_TIER': ''}).service_tier, '')
        self.assertEqual(BuildConfig.from_env({'STICKER_BUILD_SERVICE_TIER': 'off'}).service_tier, '')
