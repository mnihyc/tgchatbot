"""Sticker/provider workflows with synthetic media and mocked HTTP only."""
from dataclasses import replace
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from PIL import Image

from tgchatbot.config import load_config
from tgchatbot.embeddings import EmbeddingClient, EmbeddingConfig, sticker_embedding_config
from tgchatbot.providers.factory import build_provider
from tgchatbot.stickers.build import BuildConfig, CatalogBuilder
from tgchatbot.stickers.catalog import StickerCatalog
from tgchatbot.stickers.media import MediaConfig, content_hash
from tgchatbot.stickers.plan import StickerRetrievalPlan
from tgchatbot.storage.sticker_catalog import CatalogAlias, CatalogAsset

CARD = {
    'caption': '抱抱', 'appearance': 'Two simple fictional shapes.', 'action': 'An offered embrace.',
    'readings': [{'meaning': 'Offering comfort.', 'context': 'A friend welcomes affection.'},
                 {'meaning': 'Warm appreciation.', 'context': 'Thanking a close friend.'}],
    'uncertainty': '',
    'compatibility': {'harshness_level': 0, 'intimacy_level': 1, 'meme_dependence_level': 0},
}

class StickerProviderSettingsTests(unittest.TestCase):
    def test_build_and_query_share_sticker_space_without_changing_chat_memory(self):
        env = {'GEMINI_API_KEY': 'shared', 'GEMINI_BASE_URL': 'https://shared.invalid/v1beta',
               'EMBEDDING_DIMENSIONS': '1536', 'DEFAULT_PROVIDER': 'deepseek', 'DEEPSEEK_API_KEY': 'chat-only'}
        memory, sticker = EmbeddingConfig.from_env(env), sticker_embedding_config(env)
        self.assertEqual((memory.dimensions, sticker.dimensions), (1536, 3072))
        self.assertEqual((sticker.api_key, sticker.base_url), ('shared', 'https://shared.invalid/v1beta'))
        self.assertEqual(sticker_embedding_config(env | {'STICKER_EMBEDDING_DIMENSIONS': '768'}).dimensions, 768)
        self.assertEqual(sticker.space_id, replace(sticker, base_url='https://moved.invalid/v1beta').space_id)
        self.assertNotEqual(sticker.space_id, replace(sticker, model='gemini-embedding-001').space_id)

class StickerProviderWorkflows(unittest.IsolatedAsyncioTestCase):
    async def test_empty_catalog_needs_no_embedding_key_or_local_index_service(self):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory, patch.dict(os.environ, {}, clear=True):
            catalog = StickerCatalog(None, Path(directory))
            matches = await catalog.achoose(plan=StickerRetrievalPlan.from_payload({'intent_core': 'hello'}))
            self.assertEqual(matches, [])
            self.assertEqual(catalog.stats()['stickers'], 0)

    async def run_build_asset(self, embedding_backend, generation_backend='gemini'):
        directory = TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        source = root / 'synthetic.png'
        Image.new('RGB', (12, 10), (20, 80, 180)).save(source)
        digest = content_hash(source)
        asset = CatalogAsset('sha256:' + digest, digest, (CatalogAlias(source.name, 'synthetic'),),
                             {}, None, {}, None, {}, None, None, 'pending')
        with patch.dict(os.environ, {'APP_DATA_DIR': directory.name, 'GEMINI_API_KEY': 'annotation-key',
                                    'GEMINI_MODEL': 'gemini-3.8-flash', 'OPENAI_API_KEY': 'annotation-key'}, clear=True):
            app_config = load_config(require_telegram=False)
        generation = build_provider(app_config, generation_backend)
        await generation.aclose()
        requests, vector_requests = [], []
        async def generation_http(request):
            requests.append(json.loads(request.content))
            if generation_backend == 'openai':
                self.assertEqual(request.headers['Authorization'], 'Bearer annotation-key')
                return httpx.Response(200, json={'output': [{'type': 'message', 'content': [{'type': 'output_text', 'text': json.dumps(CARD)}]}],
                    'usage': {'input_tokens': 70, 'output_tokens': 30, 'total_tokens': 100}, 'service_tier': 'flex'})
            self.assertEqual(request.headers['x-goog-api-key'], 'annotation-key')
            return httpx.Response(200, json={'candidates': [{'content': {'role': 'model', 'parts': [{'text': json.dumps(CARD)}]}}],
                'usageMetadata': {'promptTokenCount': 70, 'candidatesTokenCount': 30, 'totalTokenCount': 100, 'serviceTier': 'flex'}})
        generation._client = httpx.AsyncClient(base_url='https://annotation.invalid/v1/',
                                               headers={'x-goog-api-key': generation.config.api_key} if generation_backend == 'gemini' else {'Authorization': 'Bearer ' + generation.config.api_key},
                                               transport=httpx.MockTransport(generation_http))
        self.addAsyncCleanup(generation.aclose)
        async def embedding_http(request):
            payload = json.loads(request.content)
            vector_requests.append(payload)
            if embedding_backend == 'gemini':
                self.assertEqual(request.headers['x-goog-api-key'], 'embedding-key')
                return httpx.Response(200, json={'embeddings': [{'values': [3, 4]} for _ in payload['requests']]})
            self.assertEqual(request.headers['Authorization'], 'Bearer embedding-key')
            return httpx.Response(200, json={'data': [{'index': index, 'embedding': [3, 4]} for index in reversed(range(len(payload['input'])))]})
        http = httpx.AsyncClient(transport=httpx.MockTransport(embedding_http))
        self.addAsyncCleanup(http.aclose)
        embeddings = EmbeddingClient(EmbeddingConfig(provider=embedding_backend,
            model='gemini-embedding-2' if embedding_backend == 'gemini' else 'configured-text-model',
            dimensions=2, api_key='embedding-key', base_url='https://embedding.invalid/v1beta',
            requests_per_minute=1e12, max_retries=0), http_client=http)
        store = Mock(stage_asset=AsyncMock())
        builder = CatalogBuilder(store, generation, embeddings, config=BuildConfig(provider=generation_backend,
            model=app_config.default_model_for_provider(generation_backend), image_embeddings=embedding_backend == 'gemini'), media_config=MediaConfig(max_frames=2))
        await builder._process('new-staging-revision', root, asset)
        return requests, vector_requests, store.stage_asset.await_args_list, builder.recipe

    async def test_annotation_and_image_embedding_share_prepared_pixels_without_a_second_annotation_call(self):
        requests, vectors, stages, recipe = await self.run_build_asset('gemini')
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0]['service_tier'], 'flex')
        self.assertNotIn('tools', requests[0])
        generated_parts = requests[0]['contents'][0]['parts']
        self.assertIn('Observed media facts', generated_parts[0]['text'])
        self.assertEqual(len(vectors), 2)
        text_rows, image_rows = vectors[0]['requests'], vectors[1]['requests']
        self.assertEqual(len(text_rows), 2)
        self.assertIn('Context: A friend welcomes affection.', text_rows[0]['content']['parts'][0]['text'])
        self.assertEqual(image_rows[0]['content']['parts'], generated_parts[1:])
        self.assertTrue(recipe['image_embeddings'])
        self.assertIsNone(stages[0].kwargs['reading_vectors'])
        self.assertEqual(stages[0].kwargs['provenance']['annotation']['usage']['service_tier'], 'flex')
        final = stages[-1].kwargs
        self.assertEqual(final['state'], 'ready')
        self.assertEqual(final['reading_vectors'].shape, (2, 2))
        self.assertEqual(final['image_vector'].shape, (2,))

    async def test_text_embedding_provider_does_not_change_the_generation_provider(self):
        requests, vectors, stages, recipe = await self.run_build_asset('openai')
        self.assertEqual(len(requests), 1)
        self.assertIn('generationConfig', requests[0])
        self.assertEqual(len(vectors), 1)
        self.assertEqual(vectors[0]['model'], 'configured-text-model')
        self.assertEqual(vectors[0]['input'], ['Offering comfort.\nContext: A friend welcomes affection.',
                                             'Warm appreciation.\nContext: Thanking a close friend.'])
        self.assertFalse(recipe['image_embeddings'])
        self.assertIsNone(stages[-1].kwargs['image_vector'])
        self.assertEqual(stages[-1].kwargs['reading_vectors'].shape, (2, 2))

    async def test_openai_annotation_uses_native_images_and_strict_card_schema_with_gemini_vectors(self):
        requests, vectors, stages, recipe = await self.run_build_asset('gemini', 'openai')
        self.assertEqual(len(requests), 1)
        request = requests[0]
        self.assertEqual(request['tools'], [])
        self.assertEqual(request['service_tier'], 'flex')
        content = request['input'][0]['content']
        self.assertIn('Observed media facts', content[0]['text'])
        self.assertTrue(any(part['type'] == 'input_image' for part in content))
        output_format = request['text']['format']
        self.assertTrue(output_format['strict'])
        schema = output_format['schema']
        self.assertFalse(schema['additionalProperties'])
        self.assertFalse(schema['properties']['readings']['items']['additionalProperties'])
        self.assertFalse(schema['properties']['compatibility']['additionalProperties'])
        self.assertEqual(recipe['annotation']['provider'], 'openai')
        self.assertEqual(recipe['embedding_space']['provider'], 'gemini')
        self.assertEqual(len(vectors), 2)
        self.assertEqual(stages[-1].kwargs['state'], 'ready')
