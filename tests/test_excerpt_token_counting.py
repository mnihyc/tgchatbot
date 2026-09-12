"""Exact hosted-count decisions and source coverage during excerpt packing."""
from __future__ import annotations

from datetime import datetime
import json
from types import SimpleNamespace
import unittest

import httpx

from tgchatbot.core.memory_worker import ExcerptBuilder, source_body
from tgchatbot.domain.models import ConversationMessage, MessagePart, PartKind
from tgchatbot.embeddings import EmbeddingClient, EmbeddingConfig
from tgchatbot.storage.postgres_store import PostgresStore


class ExcerptTokenCountingTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.counted = []
        # Synthetic tokenizer behavior, not an approximation used by production.
        self.oracle = lambda text: len(text.encode('utf8'))

        def respond(request):
            self.assertTrue(request.url.path.endswith(':countTokens'))
            text = json.loads(request.content)['contents'][0]['parts'][0]['text']
            self.counted.append(text)
            return httpx.Response(200, json={'totalTokens': self.oracle(text)})

        self.http = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        self.addAsyncCleanup(self.http.aclose)
        self.embeddings = EmbeddingClient(EmbeddingConfig(api_key='local-fixture',
            base_url='https://fixture.invalid/v1beta', requests_per_minute=10_000_000),
            http_client=self.http)
        self.builder = ExcerptBuilder(self.embeddings)

    @staticmethod
    def original(number, text, actor=101):
        return SimpleNamespace(db_id=number, message=ConversationMessage.user_text(text, metadata={
            'actor_id': f'telegram:user:{actor}', 'actor_name': 'Participant',
            'sent_at': f'2026-09-12T00:{number // 60:02d}:{number % 60:02d}+00:00'}))

    async def assert_evidence_and_window(self, rows, chunks, expected_ranges):
        by_id = {row.db_id: row for row in rows}
        actual = [(span['message_id'], offset) for chunk in chunks for span in chunk['spans']
                  for offset in range(span['start'], span['end'])]
        expected = [(message_id, offset) for message_id, start, end in expected_ranges
                    for offset in range(start, end)]
        self.assertEqual(actual, expected, 'Every requested original character must appear once and in order')
        for chunk in chunks:
            sources = []
            for mid in chunk['source_ids']:
                row = by_id[mid]
                sources.append({'id': mid, 'body': source_body(row.message), 'role': 'user',
                    'generation': 1, 'actor_kind': 'user', 'actor_id': row.message.metadata['actor_id'],
                    'actor_name': 'Participant', 'source': 'fixture', 'source_chat_id': 'chat',
                    'source_message_id': str(mid), 'topic_id': None, 'source_revision': 1,
                    'sent_at': datetime.fromisoformat(row.message.metadata['sent_at']), 'metadata': {}})
            # Validate what storage would actually emit, not only the builder's preview.
            rendered = PostgresStore._render_excerpt(chunk, sources)
            self.assertLessEqual(await self.embeddings.count_tokens(rendered['text']), self.builder.token_limit)
            for mid in chunk['source_ids']:
                self.assertIn(by_id[mid].message.metadata['actor_id'], rendered['text'])

    async def test_short_multilingual_conversation_avoids_two_native_counts_per_original(self):
        rows = [self.original(i, ('Thanks, sounds good.', '今晚想吃火锅。', '🍵 nice 😊')[i % 3],
            actor=101 + i % 3) for i in range(96)]
        chunks = await self.builder.build(rows)
        await self.assert_evidence_and_window(rows, chunks,
            [(row.db_id, 0, len(source_body(row.message))) for row in rows])
        # A loose performance bound for short-message indexing, including final
        # exact validation. It does not require particular excerpt boundaries.
        self.assertLess(len(self.counted), len(rows) * 1.5,
            'Short-message indexing should avoid two remote tokenizer requests per original')

    async def test_overflow_long_multipart_and_retained_suffix_preserve_evidence(self):
        first = self.original(1, 'A short introduction.')
        long = self.original(2, '连续中文👩🏽‍💻' * 300 + '\nCorrection at the end.', actor=102)
        multipart = self.original(3, 'First genuine statement.')
        multipart.message.parts += [MessagePart(PartKind.TEXT, text='Internal context control.', origin='auto_note'),
            MessagePart(PartKind.TEXT, text='后来更正了，应该是明天。')]
        retained = self.original(4, 'Old part. Keep this exact suffix.')
        last_start = len(multipart.message.parts[0].text) + len(multipart.message.parts[1].text) + 2
        rows = [first, long, multipart, retained]
        retained_spans = {4: [{'message_id': 4, 'start': 10, 'end': len(source_body(retained.message))}]}
        chunks = await self.builder.build(rows, retained_spans=retained_spans)
        await self.assert_evidence_and_window(rows, chunks, [
            (1, 0, len(source_body(first.message))), (2, 0, len(source_body(long.message))),
            (3, 0, len(multipart.message.parts[0].text)),
            (3, last_start, len(source_body(multipart.message))),
            (4, 10, len(source_body(retained.message)))])
        self.assertFalse(any('Internal context control.' in chunk['text'] for chunk in chunks))

    async def test_fitting_assembly_does_not_require_monotonic_standalone_token_counts(self):
        # An opaque tokenizer can merge across context. Do not infer acceptance
        # from a character ratio, or require this constructed case of Gemini.
        self.oracle = lambda text: (8 if 'MERGE_LEFT' in text and 'MERGE_RIGHT' in text
            else 600 if 'MERGE_RIGHT' in text else len(text.encode('utf8')))
        rows = [self.original(1, 'MERGE_LEFT'), self.original(2, 'MERGE_RIGHT', actor=102)]
        chunks = await self.builder.build(rows)
        await self.assert_evidence_and_window(rows, chunks,
            [(row.db_id, 0, len(source_body(row.message))) for row in rows])
        self.assertTrue(any('MERGE_LEFT' in chunk['text'] and 'MERGE_RIGHT' in chunk['text'] for chunk in chunks),
            'The exact-fitting assembly can retain the complete exchange')
