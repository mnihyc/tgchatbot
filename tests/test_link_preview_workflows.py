from __future__ import annotations

import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from tgchatbot.config import load_config
from tgchatbot.media.link_prefetch import _fetch_one, fetch_link_previews, previews_to_parts


class LinkPreviewWorkflows(unittest.IsolatedAsyncioTestCase):
    async def test_preview_keeps_visible_code_but_excludes_script_and_style(self):
        page = ('<title>Repair guide</title><style>.secret{color:red}</style>'
            '<script>window.internalBootstrap = "not article content";</script>'
            '<p>Check the return code.</p><pre><code>if rc != 0: retry()</code></pre>')
        transport = httpx.MockTransport(lambda request: httpx.Response(200,
            headers={'content-type': 'text/html'}, text=page))
        with patch('tgchatbot.media.link_prefetch._url_is_safe', AsyncMock(return_value=True)):
            async with httpx.AsyncClient(transport=transport) as client:
                preview = await _fetch_one(client, 'https://example.invalid/guide', mode='snippet',
                    max_chars=500, max_bytes=4000, max_redirects=5)
        shown = previews_to_parts([preview], mode='snippet')[0]
        self.assertIn('if rc != 0: retry()', shown.text)
        self.assertIn('Check the return code.', shown.text)
        self.assertNotIn('internalBootstrap', shown.text)
        self.assertNotIn('.secret', shown.text)
        self.assertEqual(shown.origin, 'auto_note')

    async def test_exact_preview_repetition_is_removed_but_additional_context_survives(self):
        for content_type, body, expected_count in (
            ('text/plain', 'The service resumes Monday.', 1),
            ('text/html', '<meta name="description" content="The service resumes Monday.">'
             '<p>The service resumes Monday.</p>', 1),
            ('text/html', '<meta name="description" content="The service resumes Monday.">'
             '<p>The service resumes Monday. Bring your ticket.</p>', 2),
        ):
            transport = httpx.MockTransport(lambda request: httpx.Response(200,
                headers={'content-type': content_type}, text=body))
            with self.subTest(content_type=content_type, expected_count=expected_count), \
                 patch('tgchatbot.media.link_prefetch._url_is_safe', AsyncMock(return_value=True)):
                async with httpx.AsyncClient(transport=transport) as client:
                    preview = await _fetch_one(client, 'https://example.invalid/service', mode='snippet',
                        max_chars=500, max_bytes=4000, max_redirects=5)
                shown = previews_to_parts([preview], mode='snippet')[0].text
                self.assertEqual(shown.count('The service resumes Monday.'), expected_count)
                self.assertIn('URL: https://example.invalid/service', shown)
                if expected_count == 2:
                    self.assertIn('Bring your ticket.', shown)

    async def test_large_title_stays_within_the_context_character_allowance(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200,
            headers={'content-type': 'text/html'}, text='<title>' + '好' * 10000 + '</title>'))
        with patch('tgchatbot.media.link_prefetch._url_is_safe', AsyncMock(return_value=True)):
            async with httpx.AsyncClient(transport=transport) as client:
                preview = await _fetch_one(client, 'https://example.invalid/page', mode='title',
                    max_chars=100, max_bytes=40000, max_redirects=5)
        self.assertEqual(preview.title, '好' * 100)

    async def test_zero_url_allowance_does_not_fetch_a_link(self):
        with patch.dict(os.environ, {'TGBOT_LINK_PREFETCH_MAX_URLS': '0'}, clear=True):
            config = load_config(require_telegram=False)
        with patch('tgchatbot.media.link_prefetch._url_is_safe',
                   AsyncMock(side_effect=AssertionError('prefetch was disabled'))):
            self.assertEqual(await fetch_link_previews('https://example.invalid/page', mode='snippet',
                telegram=config.telegram), [])

    async def test_preview_closes_body_after_configured_allowance(self):
        class LongBody(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                yield b'Useful short preview.'
                raise AssertionError('Unbounded remainder was downloaded')

            async def aclose(self):
                self.closed = True

        body = LongBody()
        transport = httpx.MockTransport(lambda request:
            httpx.Response(200, headers={'content-type': 'text/plain'}, stream=body))
        with patch('tgchatbot.media.link_prefetch._url_is_safe', AsyncMock(return_value=True)):
            async with httpx.AsyncClient(transport=transport) as client:
                preview = await _fetch_one(client, 'https://example.invalid/page', mode='snippet',
                    max_chars=100, max_bytes=12, max_redirects=5)
        self.assertTrue(body.closed)
        self.assertEqual(preview.snippet, 'Useful short')

    async def test_redirect_destination_is_checked_before_request(self):
        requested = []

        def reply(request):
            requested.append(str(request.url))
            return httpx.Response(302, headers={'location': 'http://127.0.0.1/private'})

        with patch('tgchatbot.media.link_prefetch._url_is_safe', AsyncMock(side_effect=[True, False])):
            async with httpx.AsyncClient(transport=httpx.MockTransport(reply)) as client:
                self.assertIsNone(await _fetch_one(client, 'https://example.invalid/page', mode='title',
                    max_chars=100, max_bytes=100, max_redirects=5))
        self.assertEqual(requested, ['https://example.invalid/page'])
