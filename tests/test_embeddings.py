import asyncio
from dataclasses import replace
import json
import os
import unittest
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np

from tgchatbot.embeddings import BatchJob, EmbeddingClient, EmbeddingConfig, EmbeddingDocument, EmbeddingMedia, SyncEmbeddingClient


class EmbeddingConfigTests(unittest.TestCase):
    def test_hosted_default_reuses_gemini_route_without_selecting_generation(self):
        env = {"GEMINI_API_KEY": "embedding-secret", "GEMINI_BASE_URL": "https://proxy.invalid/v1beta/",
               "DEFAULT_PROVIDER": "deepseek", "OPENAI_API_KEY": "unrelated-secret"}
        config = EmbeddingConfig.from_env(env)
        self.assertEqual((config.provider, config.model, config.dimensions), ("gemini", "gemini-embedding-2", 1536))
        self.assertEqual(config.api_key, "embedding-secret")
        self.assertEqual(config.base_url, "https://proxy.invalid/v1beta")
        self.assertEqual(env["DEFAULT_PROVIDER"], "deepseek")
        self.assertNotIn("embedding-secret", repr(config))

    def test_constructor_never_reads_an_unrelated_secret(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "unrelated", "GEMINI_API_KEY": "also-unrelated"}, clear=True):
            self.assertFalse(EmbeddingConfig().enabled)
            self.assertEqual(EmbeddingConfig().api_key, "")
        self.assertFalse(EmbeddingConfig.from_env({"OPENAI_API_KEY": "unrelated"}).enabled)

    def test_separate_explicit_key_and_unauthenticated_compatible_route(self):
        config = EmbeddingConfig.from_env({"EMBEDDING_PROVIDER": "openai-compatible",
            "EMBEDDING_BASE_URL": "http://model.invalid/v1", "EMBEDDING_API_KEY": "",
            "OPENAI_API_KEY": "generation-only"})
        self.assertEqual(config.provider, "openai")
        self.assertTrue(config.enabled)
        self.assertEqual(config.api_key, "")
        config = EmbeddingConfig.from_env({"EMBEDDING_API_KEY_ENV": "VECTOR_KEY", "VECTOR_KEY": "selected", "GEMINI_API_KEY": "not-selected"})
        self.assertEqual(config.api_key, "selected")

    def test_space_identity_ignores_route_and_key_but_tracks_format_model_and_dimensions(self):
        config = EmbeddingConfig()
        self.assertEqual(config.space_id, replace(config, api_key="another", base_url="https://another.invalid/v1beta").space_id)
        for changed in [replace(config, dimensions=768), replace(config, model="gemini-embedding-001"), replace(config, model_revision="revision-2")]:
            self.assertNotEqual(config.space_id, changed.space_id)

    def test_sticker_profile_overrides_dimensions_without_changing_memory_or_key_route(self):
        env = {'GEMINI_API_KEY': 'shared-key', 'GEMINI_BASE_URL': 'https://proxy.invalid/v1beta',
               'EMBEDDING_DIMENSIONS': '1536', 'EMBEDDING_CACHE_ENTRIES': '20',
               'STICKER_EMBEDDING_DIMENSIONS': '3072', 'STICKER_EMBEDDING_CACHE_ENTRIES': '0'}
        memory = EmbeddingConfig.from_env(env)
        sticker = EmbeddingConfig.from_env(env, prefix='STICKER_EMBEDDING')
        self.assertEqual((memory.dimensions, sticker.dimensions), (1536, 3072))
        self.assertEqual((memory.cache_entries, sticker.cache_entries), (20, 0))
        self.assertEqual((memory.api_key, memory.base_url), (sticker.api_key, sticker.base_url))
        self.assertNotEqual(memory.space_id, sticker.space_id)

    def test_sticker_provider_change_does_not_inherit_a_foreign_key_model_or_url(self):
        env = {'EMBEDDING_MODEL': 'gemini-embedding-2', 'EMBEDDING_API_KEY': 'gemini-only',
               'EMBEDDING_BASE_URL': 'https://gemini.invalid/v1beta', 'OPENAI_API_KEY': 'selected-openai',
               'STICKER_EMBEDDING_PROVIDER': 'openai'}
        sticker = EmbeddingConfig.from_env(env, prefix='STICKER_EMBEDDING')
        self.assertEqual(sticker.model, 'text-embedding-3-large')
        self.assertEqual(sticker.api_key, 'selected-openai')
        self.assertEqual(sticker.base_url, 'https://api.openai.com/v1')
        pointer = EmbeddingConfig.from_env(env | {'STICKER_EMBEDDING_API_KEY_ENV': 'STICKER_KEY',
            'STICKER_KEY': 'explicit-sticker'}, prefix='STICKER_EMBEDDING')
        self.assertEqual(pointer.api_key, 'explicit-sticker')

    def test_invalid_limits_fail_before_network(self):
        for options in [{"dimensions": 0}, {"dimensions": 4096}, {"requests_per_minute": 0},
                        {"timeout_s": float("inf")}, {"cache_entries": -1}, {"max_retries": -1},
                        {"base_url": "https://secret:password@example.invalid/v1"}]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                EmbeddingConfig(**options)

    def test_transport_overrides_have_no_hidden_previous_ceilings_and_blank_keeps_defaults(self):
        config = EmbeddingConfig.from_env({'EMBEDDING_PROVIDER': 'openai',
            'EMBEDDING_MODEL': ' ', 'EMBEDDING_TIMEOUT_S': ' ', 'EMBEDDING_MAX_RETRIES': '9',
            'EMBEDDING_CONNECT_TIMEOUT_S': '12', 'EMBEDDING_SYNC_BATCH_ITEMS': '256',
            'EMBEDDING_BATCH_MAX_ITEMS': '1500', 'EMBEDDING_BATCH_LIST_PAGE_SIZE': '250',
            'EMBEDDING_BATCH_RECONCILE_MAX_PAGES': '30', 'EMBEDDING_RETRY_INITIAL_DELAY_S': '1.25',
            'EMBEDDING_RETRY_MAX_DELAY_S': '25', 'EMBEDDING_RETRY_AFTER_MAX_S': '180'})
        self.assertEqual(config.model, 'text-embedding-3-large')
        self.assertEqual(config.timeout_s, EmbeddingConfig().timeout_s)
        self.assertEqual((config.max_retries, config.connect_timeout_s, config.sync_batch_items), (9, 12, 256))
        self.assertEqual((config.batch_max_items, config.batch_list_page_size, config.batch_reconcile_max_pages), (1500, 250, 30))
        self.assertEqual((config.retry_initial_delay_s, config.retry_max_delay_s, config.retry_after_max_s), (1.25, 25, 180))


class EmbeddingClientTests(unittest.IsolatedAsyncioTestCase):
    def make_client(self, handler, **options):
        options = {"api_key": "fake", "dimensions": 2, "base_url": "https://test.invalid/v1beta",
                   "requests_per_minute": 1e12, "max_retries": 0, **options}
        http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        self.addAsyncCleanup(http.aclose)
        return EmbeddingClient(EmbeddingConfig(**options), http_client=http)

    async def test_multimodal_documents_preserve_frame_order_and_leave_text_memory_format_unchanged(self):
        seen = []
        def handle(request):
            payload = json.loads(request.content)
            seen.append(payload)
            return httpx.Response(200, json={'embeddings': [{'values': [3, 4]} for _ in payload['requests']]})
        client = self.make_client(handle)
        self.assertTrue(client.supports_media)
        docs = [EmbeddingDocument('image', media=(EmbeddingMedia('image/png', 'AAAA'), EmbeddingMedia('image/jpeg', 'BBBB'))),
                EmbeddingDocument('fused', 'offering comfort', media=(EmbeddingMedia('image/png', 'CCCC'),)),
                EmbeddingDocument('memory', 'Prefers jasmine tea.', 'Known preference')]
        results = await client.embed_documents(docs, purpose='sticker')
        self.assertEqual(len(results), 3)
        payloads = seen[0]['requests']
        self.assertEqual(payloads[0]['content']['parts'], [
            {'inlineData': {'mimeType': 'image/png', 'data': 'AAAA'}},
            {'inlineData': {'mimeType': 'image/jpeg', 'data': 'BBBB'}}])
        self.assertEqual(payloads[1]['content']['parts'][0], {'text': 'title: none | text: offering comfort'})
        self.assertEqual(payloads[2]['content']['parts'], [{'text': 'title: Known preference | text: Prefers jasmine tea.'}])
        self.assertTrue(all(p['embedContentConfig']['autoTruncate'] is False for p in payloads))

    async def test_multimodal_count_and_native_batch_use_the_same_parts_and_stable_ids(self):
        seen = []
        def handle(request):
            seen.append((request.url.path, json.loads(request.content)))
            if request.url.path.endswith(':countTokens'):
                return httpx.Response(200, json={'totalTokens': 258})
            return httpx.Response(200, json={'name': 'batches/media-job', 'metadata': {'state': 'JOB_STATE_PENDING'}})
        client = self.make_client(handle)
        document = EmbeddingDocument('asset-hash:frame-v1', media=(EmbeddingMedia('image/png', 'AAAA'),))
        self.assertEqual(await client.count_document_tokens(document), 258)
        self.assertEqual(await client.count_document_tokens(document), 258)
        job = await client.submit_batch([document], purpose='sticker', display_name='sticker-build-id')
        self.assertEqual(job.item_ids, ('asset-hash:frame-v1',))
        self.assertEqual(len(seen), 2)
        row = seen[1][1]['batch']['inputConfig']['requests']['requests'][0]
        self.assertEqual(row['request']['content'], seen[0][1]['contents'][0])
        self.assertEqual(row['metadata']['key'], document.item_id)
        self.assertEqual(row['metadata']['space_id'], client.space_id)
        await client.count_document_tokens(replace(document, media=(EmbeddingMedia('image/png', 'BBBB'),)))
        self.assertEqual(len(seen), 3)

    async def test_unsupported_media_and_excess_frames_fail_without_dropping_inputs(self):
        def no_network(request):
            self.fail('invalid media must fail before HTTP')
        for provider, model in [('gemini', 'gemini-embedding-001'), ('openai', 'text-model')]:
            client = self.make_client(no_network, provider=provider, model=model)
            self.assertFalse(client.supports_media)
            with self.assertRaisesRegex(NotImplementedError, 'text only'):
                await client.embed_documents([EmbeddingDocument('image', media=(EmbeddingMedia('image/png', 'AAAA'),))])
        client = self.make_client(no_network)
        for media in [(EmbeddingMedia('image/webp', 'AAAA'),), (EmbeddingMedia('image/png', 'AAAA'),) * 7]:
            with self.assertRaises(ValueError):
                await client.embed_documents([EmbeddingDocument('invalid', media=media)])

    async def test_gemini2_exact_query_and_document_contracts(self):
        seen = []
        def handle(request):
            payload = json.loads(request.content)
            seen.append((request.url.path, payload))
            self.assertEqual(request.headers["x-goog-api-key"], "fake")
            self.assertNotIn("authorization", request.headers)
            if request.url.path.endswith(":batchEmbedContents"):
                return httpx.Response(200, json={"embeddings": [{"values": [3, 4]} for _ in payload["requests"]]})
            return httpx.Response(200, json={"embedding": {"values": [3, 4]}})
        client = self.make_client(handle)
        np.testing.assert_allclose(await client.embed_query("  安慰朋友  ", purpose="sticker"), [0.6, 0.8])
        vectors = await client.embed_documents([EmbeddingDocument("a", "warm hug", "抱抱"), EmbeddingDocument("b", "sincere praise")], purpose="sticker")
        self.assertEqual(len(vectors), 2)
        self.assertEqual(seen[0][1], {"model": "models/gemini-embedding-2",
            "content": {"parts": [{"text": "task: search result | query: 安慰朋友"}]},
            "embedContentConfig": {"outputDimensionality": 2, "autoTruncate": False}})
        docs = seen[1][1]["requests"]
        self.assertEqual(docs[0]["content"]["parts"], [{"text": "title: 抱抱 | text: warm hug"}])
        self.assertEqual(docs[1]["content"]["parts"], [{"text": "title: none | text: sincere praise"}])
        self.assertTrue(all("taskType" not in doc and "outputDimensionality" not in doc for doc in docs))

    async def test_gemini001_uses_flat_task_title_and_dimensions_not_ignored_nested_fields(self):
        seen = []
        def handle(request):
            payload = json.loads(request.content)
            seen.append((request.url.path, payload))
            if request.url.path.endswith(":countTokens"):
                return httpx.Response(200, json={"totalTokens": 20})
            if request.url.path.endswith(":batchEmbedContents"):
                return httpx.Response(200, json={"embeddings": [{"values": [3, 4]}]})
            return httpx.Response(200, json={"embedding": {"values": [3, 4]}})
        client = self.make_client(handle, model="gemini-embedding-001")
        await client.embed_query("query")
        await client.embed_documents([EmbeddingDocument("d", "document", "title")])
        query = next(body for path, body in seen if path.endswith(":embedContent"))
        doc = next(body["requests"][0] for path, body in seen if path.endswith(":batchEmbedContents"))
        self.assertEqual(query["content"]["parts"][0]["text"], "query")
        self.assertEqual(query["taskType"], "RETRIEVAL_QUERY")
        self.assertEqual(doc["taskType"], "RETRIEVAL_DOCUMENT")
        self.assertEqual(doc["title"], "title")
        self.assertEqual(doc["outputDimensionality"], 2)
        self.assertNotIn("embedContentConfig", query)
        self.assertNotIn("embedContentConfig", doc)

    async def test_future_gemini_model_uses_default_dialect_without_name_whitelist(self):
        seen = []
        def handle(request):
            seen.append((str(request.url), json.loads(request.content)))
            return httpx.Response(200, json={'embedding': {'values': [3, 4]}})
        client = self.make_client(handle, model='gemini-next-compatible')
        await client.embed_query('query')
        self.assertIn('models/gemini-next-compatible:embedContent', seen[0][0])
        self.assertEqual(seen[0][1]['content']['parts'][0]['text'], 'task: search result | query: query')
        self.assertEqual(seen[0][1]['embedContentConfig'], {'outputDimensionality': 2, 'autoTruncate': False})

    async def test_legacy_oversized_input_is_rejected_before_silent_truncation(self):
        calls = []
        def handle(request):
            calls.append(request.url.path)
            return httpx.Response(200, json={"totalTokens": 2049})
        client = self.make_client(handle, model="gemini-embedding-001")
        with self.assertRaisesRegex(ValueError, "2048"):
            await client.embed_query("oversized")
        self.assertEqual(calls, ["/v1beta/models/gemini-embedding-001:countTokens"])

    async def test_count_tokens_includes_exact_retrieval_format_and_validates_count(self):
        requests = []
        def handle(request):
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"totalTokens": 17})
        client = self.make_client(handle)
        self.assertEqual(await client.count_tokens("hello", kind="query"), 17)
        self.assertEqual(await client.count_tokens("hello", kind="query"), 17)
        self.assertEqual(requests, [{"contents": [{"parts": [{"text": "task: search result | query: hello"}]}]}])
        bad = self.make_client(lambda request: httpx.Response(200, json={"totalTokens": True}))
        with self.assertRaises(ValueError):
            await bad.count_tokens("text")

    async def test_invalid_vectors_are_never_cached(self):
        for values in [[1], [0, 0], [float("nan"), 1], [float("inf"), 1], [[1, 2]], [True, False], ["1", "2"]]:
            with self.subTest(values=values):
                client = self.make_client(lambda request: httpx.Response(200, text=json.dumps({"embedding": {"values": values}})))
                with self.assertRaises(ValueError):
                    await client.embed_query("query")
                self.assertEqual(len(client._cache), 0)

    async def test_malformed_response_objects_fail_before_cache(self):
        for payload in [[], None, {"embedding": None}, {"error": {"message": "bad response"}}]:
            client = self.make_client(lambda request: httpx.Response(200, text=json.dumps(payload)))
            with self.assertRaises(ValueError):
                await client.embed_query("query")
            self.assertEqual(len(client._cache), 0)

    async def test_query_cache_is_bounded_purpose_specific_and_safe_from_caller_mutation(self):
        calls = []
        def handle(request):
            calls.append(request)
            return httpx.Response(200, json={"embedding": {"values": [3, 4]}})
        client = self.make_client(handle, cache_entries=2)
        vector = await client.embed_query("same", purpose="memory")
        vector[:] = 0
        np.testing.assert_allclose(await client.embed_query("same", purpose="memory"), [0.6, 0.8])
        await client.embed_query("same", purpose="sticker")
        await client.embed_query("third", purpose="memory")
        await client.embed_query("same", purpose="memory")
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(client._cache), 2)

    async def test_openai_compatible_batches_preserve_indices_and_need_no_dummy_key(self):
        counts = []
        def handle(request):
            self.assertNotIn("authorization", request.headers)
            payload = json.loads(request.content)
            count = len(payload["input"])
            counts.append(count)
            return httpx.Response(200, json={"data": [{"index": i, "embedding": [3, 4]} for i in reversed(range(count))]})
        client = self.make_client(handle, provider="openai", model="explicit-model", api_key="", base_url="http://compatible.invalid/v1")
        vectors = await client.embed_documents(["text"] * 129)
        self.assertEqual(counts, [128, 1])
        self.assertEqual(len(vectors), 129)
        np.testing.assert_allclose(vectors[0], [0.6, 0.8])
        with self.assertRaises(NotImplementedError):
            await client.count_tokens("text")
        with self.assertRaises(NotImplementedError):
            await client.submit_batch([EmbeddingDocument("d", "text")])

    async def test_response_count_or_duplicate_indices_cannot_shift_document_ownership(self):
        bad_rows = [[{"index": 0, "embedding": [1, 0]}],
                    [{"index": 0, "embedding": [1, 0]}, {"index": 0, "embedding": [0, 1]}]]
        for rows in bad_rows:
            client = self.make_client(lambda request: httpx.Response(200, json={"data": rows}), provider="openai", model="explicit-model")
            with self.assertRaises(ValueError):
                await client.embed_documents([EmbeddingDocument("alice", "A"), EmbeddingDocument("bob", "B")])
        client = self.make_client(lambda request: httpx.Response(200, json={"embeddings": [{"values": [1, 0]}]}))
        with self.assertRaises(ValueError):
            await client.embed_documents(["A", "B"])

    async def test_retry_is_bounded_and_does_not_fallback_models(self):
        seen = []
        def handle(request):
            seen.append(request)
            return httpx.Response(503)
        client = self.make_client(handle, max_retries=2)
        with patch("tgchatbot.embeddings.client.asyncio.sleep", new=AsyncMock()), self.assertRaises(httpx.HTTPStatusError):
            await client.embed_query("text")
        self.assertEqual(len(seen), 3)
        self.assertEqual(len({str(request.url) for request in seen}), 1)
        self.assertEqual(len(client._cache), 0)

    async def test_configured_retry_count_connect_timeout_and_backoff_are_applied(self):
        seen = []
        def handle(request):
            seen.append(request)
            if len(seen) < 8:
                return httpx.Response(503)
            return httpx.Response(200, json={'embedding': {'values': [3, 4]}})
        client = self.make_client(handle, max_retries=8, connect_timeout_s=12,
                                  retry_initial_delay_s=1.25, retry_max_delay_s=3)
        with patch('tgchatbot.embeddings.client.asyncio.sleep', new_callable=AsyncMock) as sleep:
            await client.embed_query('Keep retrying beyond the old five-attempt cap')
        self.assertEqual(len(seen), 8)
        self.assertTrue(all(request.extensions['timeout']['connect'] == 12 for request in seen))
        self.assertEqual([call.args[0] for call in sleep.await_args_list], [1.25, 2.5, 3, 3, 3, 3, 3])

    async def test_configured_server_cooldown_is_respected_without_early_retry(self):
        seen = []
        def handle(request):
            seen.append(request)
            return (httpx.Response(429, headers={'Retry-After': '120'}) if len(seen) == 1 else
                    httpx.Response(200, json={'embedding': {'values': [3, 4]}}))
        client = self.make_client(handle, max_retries=1, retry_after_max_s=180)
        with patch('tgchatbot.embeddings.client.asyncio.sleep', new_callable=AsyncMock) as sleep:
            await client.embed_query('query')
        sleep.assert_awaited_once_with(120)
        self.assertEqual(len(seen), 2)

    async def test_configured_sync_batch_sizes_preserve_all_items_and_gemini_protocol_limit(self):
        for provider, expected in [('openai', [256, 1]), ('gemini', [100, 100, 57])]:
            with self.subTest(provider=provider):
                counts = []
                def handle(request):
                    payload = json.loads(request.content)
                    size = len(payload['input'] if provider == 'openai' else payload['requests'])
                    counts.append(size)
                    return httpx.Response(200, json=({'data': [{'index': i, 'embedding': [3, 4]} for i in range(size)]}
                        if provider == 'openai' else {'embeddings': [{'values': [3, 4]} for _ in range(size)]}))
                client = self.make_client(handle, provider=provider, sync_batch_items=256)
                result = await client.embed_documents([f'document {i}' for i in range(257)])
                self.assertEqual(len(result), 257)
                self.assertEqual(counts, expected)

    async def test_configured_reconciliation_walks_past_old_twenty_page_ceiling(self):
        seen = []
        def handle(request):
            seen.append(request.url)
            page = int(request.url.params.get('pageToken', '0'))
            self.assertEqual(request.url.params['pageSize'], '250')
            if page < 21:
                return httpx.Response(200, json={'operations': [], 'nextPageToken': str(page + 1)})
            return httpx.Response(200, json={'operations': [{'name': 'batches/found', 'metadata': {'displayName': 'known'}}]})
        client = self.make_client(handle, batch_list_page_size=250, batch_reconcile_max_pages=30)
        self.assertEqual((await client.find_batch('known')).name, 'batches/found')
        self.assertEqual(len(seen), 22)

    async def test_auth_errors_and_long_server_cooldowns_are_not_retried(self):
        for status, headers in [(401, {}), (429, {"Retry-After": "120"})]:
            seen = []
            def handle(request):
                seen.append(request)
                return httpx.Response(status, headers=headers)
            client = self.make_client(handle, max_retries=2)
            with self.assertRaises(httpx.HTTPStatusError):
                await client.embed_query("text")
            self.assertEqual(len(seen), 1)

    async def test_cancellation_propagates_without_retry_and_event_loop_remains_live(self):
        entered, release = asyncio.Event(), asyncio.Event()
        calls = []
        async def handle(request):
            calls.append(request)
            entered.set()
            await release.wait()
            return httpx.Response(200, json={"embedding": {"values": [1, 0]}})
        client = self.make_client(handle, max_retries=2)
        task = asyncio.create_task(client.embed_query("query"))
        await entered.wait()
        await asyncio.sleep(0)
        self.assertFalse(task.done())
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertEqual(len(calls), 1)

    async def test_async_batch_uses_document_config_and_stable_id_metadata(self):
        seen = []
        def handle(request):
            seen.append((request.url.path, json.loads(request.content)))
            return httpx.Response(200, json={"name": "batches/b1", "metadata": {"state": "JOB_STATE_PENDING"}})
        client = self.make_client(handle)
        job = await client.submit_batch([EmbeddingDocument("source-a-v2", "A"), EmbeddingDocument("source-b-v1", "B")], display_name="persisted-submission-key")
        self.assertFalse(job.done)
        self.assertEqual(job.item_ids, ("source-a-v2", "source-b-v1"))
        self.assertEqual(seen[0][0], "/v1beta/models/gemini-embedding-2:asyncBatchEmbedContent")
        batch = seen[0][1]["batch"]
        self.assertEqual(batch["displayName"], "persisted-submission-key")
        requests = batch["inputConfig"]["requests"]["requests"]
        self.assertEqual([row["metadata"]["key"] for row in requests], list(job.item_ids))
        self.assertTrue(all(row["metadata"]["space_id"] == client.space_id for row in requests))
        self.assertEqual(requests[0]["request"]["content"]["parts"], [{"text": "title: none | text: A"}])

    async def test_ambiguous_batch_submit_is_not_automatically_repeated(self):
        seen = []
        def handle(request):
            seen.append(request)
            raise httpx.ReadTimeout("ambiguous", request=request)
        client = self.make_client(handle, max_retries=2)
        with self.assertRaises(httpx.ReadTimeout):
            await client.submit_batch([EmbeddingDocument("a", "A")])
        self.assertEqual(len(seen), 1)

    async def test_batch_input_limits_and_duplicate_ids_are_checked_before_submission(self):
        calls = []
        client = self.make_client(lambda request: calls.append(request), batch_max_items=2, batch_max_bytes=500)
        for items in [[], [EmbeddingDocument("a", "A")] * 2,
                      [EmbeddingDocument(str(i), "A") for i in range(3)], [EmbeddingDocument("a", "x" * 501)]]:
            with self.assertRaises(ValueError):
                await client.submit_batch(items)
        self.assertEqual(calls, [])

    async def test_partial_batch_output_maps_by_id_not_response_order(self):
        client = self.make_client(lambda request: None)
        job = BatchJob("batches/one", "JOB_STATE_SUCCEEDED", True, client.space_id,
            output={"inlinedResponses": {"inlinedResponses": [
                {"metadata": {"key": "bob"}, "response": {"embedding": {"values": [0, 2]}}},
                {"metadata": {"key": "alice"}, "error": {"code": 400, "message": "too long"}},
                {"metadata": {"key": "invalid"}, "response": {"embedding": {"values": [0, 0]}}},
            ]}})
        results = await client.read_batch_results(job, ["alice", "missing", "bob", "invalid"])
        self.assertEqual([row.item_id for row in results], ["alice", "missing", "bob", "invalid"])
        self.assertEqual(results[0].error["code"], 400)
        self.assertEqual(results[1].error["code"], "MISSING_RESULT")
        np.testing.assert_allclose(results[2].vector, [0, 1])
        self.assertEqual(results[3].error["code"], "INVALID_EMBEDDING")

    async def test_batch_output_rejects_unowned_duplicate_and_foreign_space_results(self):
        client = self.make_client(lambda request: None)
        for rows in [
            [{"metadata": {"key": "unknown"}}],
            [{"metadata": {"key": "a"}}, {"metadata": {"key": "a"}}],
            [{"metadata": {"key": "a", "space_id": "other"}}],
            [{"response": {"embedding": {"values": [1, 0]}}}],
        ]:
            job = BatchJob("batches/one", "JOB_STATE_SUCCEEDED", True, client.space_id,
                output={"inlinedResponses": {"inlinedResponses": rows}})
            with self.assertRaises(ValueError):
                await client.read_batch_results(job, ["a"])
        job.space_id = "different-model"
        with self.assertRaises(ValueError):
            await client.read_batch_results(job, ["a"])

    async def test_poll_and_keyed_result_file_use_native_endpoints(self):
        seen = []
        def handle(request):
            seen.append(str(request.url))
            if request.url.path.endswith(":download"):
                return httpx.Response(200, text=json.dumps({"key": "a", "response": {"embedding": {"values": [1, 0]}}}) + "\n")
            return httpx.Response(200, json={"name": "batches/one", "done": True,
                "metadata": {"state": "JOB_STATE_SUCCEEDED"}, "response": {"responsesFile": "files/result1"}})
        client = self.make_client(handle)
        job = await client.poll_batch("batches/one")
        rows = await client.read_batch_results(job, ["a"])
        self.assertTrue(rows[0].ok)
        self.assertEqual(seen, ["https://test.invalid/v1beta/batches/one", "https://test.invalid/download/v1beta/files/result1:download?alt=media"])

    async def test_reconciliation_follows_pages_and_rejects_ambiguous_matches(self):
        def handle(request):
            if "pageToken" not in request.url.params:
                return httpx.Response(200, json={"operations": [{"name": "batches/other", "metadata": {"displayName": "other"}}], "nextPageToken": "page&2"})
            self.assertEqual(request.url.params["pageToken"], "page&2")
            return httpx.Response(200, json={"operations": [{"name": "batches/found", "metadata": {"displayName": "submission-uuid"}}]})
        client = self.make_client(handle)
        self.assertEqual((await client.find_batch("submission-uuid")).name, "batches/found")
        bad = self.make_client(lambda request: httpx.Response(200, json={"operations": [
            {"name": f"batches/{key}", "metadata": {"displayName": "same"}} for key in ["a", "b"]]}))
        with self.assertRaisesRegex(RuntimeError, "Ambiguous"):
            await bad.find_batch("same")
        with self.assertRaisesRegex(RuntimeError, "page limit"):
            await client.find_batch("submission-uuid", max_pages=1)

    async def test_sync_facade_fails_visibly_if_called_on_bot_event_loop(self):
        facade = SyncEmbeddingClient(EmbeddingConfig())
        with self.assertRaisesRegex(RuntimeError, "event loop"):
            facade.embed_query("hello")
        await facade.client.aclose()
