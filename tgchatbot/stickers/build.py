"""Incremental catalog builder: persist work, then publish one complete revision."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, dataclass, replace
import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from tgchatbot.config import load_config
from tgchatbot.domain.models import ConversationMessage, MessagePart, MessageRole, PartKind, SessionSettings
from tgchatbot.embeddings import EmbeddingClient, EmbeddingDocument, EmbeddingMedia, sticker_embedding_config
from tgchatbot.operational import from_env
from tgchatbot.providers.factory import build_provider
from tgchatbot.storage.postgres_store import PostgresStore
from tgchatbot.storage.sticker_catalog import CatalogAlias, CatalogAsset, CatalogConflict, StickerCatalogStore
from tgchatbot.stickers.cards import CARD_PROMPT, CARD_RECIPE, CARD_SCHEMA, appearance_text, card_hash, effective_card, reading_texts, validate_card, validate_corrections
from tgchatbot.stickers.media import MediaConfig, SUPPORTED_EXTENSIONS, content_hash, prepare_media


@dataclass(frozen=True)
class BuildConfig:
    provider: str = 'gemini'
    model: str = 'gemini-3.8-flash'
    service_tier: str = 'flex'
    request_timeout_s: float = 600.0
    max_output_tokens: int = 8192
    concurrency: int = 1
    image_embeddings: bool = True

    def __post_init__(self):
        if self.concurrency < 1 or self.max_output_tokens < 1 or self.request_timeout_s <= 0:
            raise ValueError('Build concurrency, output allowance and timeout must be positive')

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        config = from_env(cls, 'STICKER_BUILD', env)
        # Unlike numeric allowances, an explicit empty tier means provider default.
        if 'STICKER_BUILD_SERVICE_TIER' in env and env['STICKER_BUILD_SERVICE_TIER'].strip().lower() in {'', 'off'}:
            config = replace(config, service_tier='')
        return config


@dataclass(frozen=True)
class BuildResult:
    revision_id: str | None
    active: bool
    completed: int
    failed: tuple[dict, ...]
    unsupported_files: tuple[str, ...] = ()


class CatalogBuilder:
    def __init__(self, store: StickerCatalogStore, provider, embeddings, *, settings: SessionSettings | None = None,
                 config: BuildConfig | None = None, media_config: MediaConfig | None = None, on_revision=None):
        self.store, self.provider, self.embeddings = store, provider, embeddings
        self.on_revision = on_revision
        self.config, self.media_config = config or BuildConfig.from_env(), media_config or MediaConfig.from_env()
        self.settings = replace(settings or SessionSettings(), provider=self.config.provider, model=self.config.model,
            service_tier=self.config.service_tier or None, max_output_tokens=self.config.max_output_tokens,
            native_web_search_mode='off', include_thoughts=False)

    @property
    def visual_embedding_source(self):
        if not self.config.image_embeddings:
            return None
        return 'image' if self.embeddings.supports_media else 'description'

    def _needs_visual(self, card):
        return self.visual_embedding_source == 'image' or (
            self.visual_embedding_source == 'description' and bool(appearance_text(card)))

    @property
    def recipe(self):
        return {'card_recipe': CARD_RECIPE, 'annotation': {'provider': self.config.provider, 'model': self.config.model}, 'media': asdict(self.media_config),
                'embedding_space_id': self.embeddings.space_id, 'embedding_space': self.embeddings.config.space_spec,
                'reading_input': 'meaning-newline-context-v1',
                'image_input': 'card-appearance-action-caption-v1' if self.visual_embedding_source == 'description' else 'sampled-frames-only-v1',
                'visual_embedding_source': self.visual_embedding_source,
                'image_embeddings': self.config.image_embeddings and self.embeddings.supports_media}

    async def _save(self, revision_id, asset, **changes):
        values = {'asset_id': asset.asset_id, 'content_hash': asset.content_hash,
                  'aliases': [asdict(alias) for alias in asset.aliases], 'media': asset.media,
                  'generated_card': asset.generated_card, 'corrections': asset.corrections, 'card': asset.card,
                  'provenance': asset.provenance, 'dimensions': self.embeddings.config.dimensions,
                  'image_vector': asset.image_vector, 'reading_vectors': asset.reading_vectors,
                  'state': asset.state, 'error': asset.error}
        values.update(changes)
        await self.store.stage_asset(revision_id, **values)

    async def build(self, source_root: Path | str, *, regenerate_ids=(), regenerate_packs=(), regenerate_files=(), corrections=None,
                    resume: str | None = None, prune_missing: bool = False) -> BuildResult:
        root = Path(source_root).resolve()
        corrections = corrections or {}
        source_files = [path for path in sorted(root.rglob('*')) if path.is_file()]
        unsupported = tuple(path.relative_to(root).as_posix() for path in source_files
                            if path.suffix.lower() not in SUPPORTED_EXTENSIONS)
        if resume:
            snapshot = await self.store.load_snapshot(resume)
            if snapshot.recipe != self.recipe or Path(snapshot.source_root) != root:
                raise ValueError('Resume uses the original source directory and build/embedding settings')
            if regenerate_ids or regenerate_packs or regenerate_files or corrections or prune_missing:
                raise ValueError('Resume continues existing work; selections/corrections belong to a new build')
            revision_id = resume
        else:
            if not root.is_dir():
                raise ValueError('Sticker source directory does not exist')
            snapshot = await self.store.load_snapshot()
            if any(identity.startswith('sid:') for identity in regenerate_ids):
                regenerate_ids = await self.store.resolve_asset_ids(list(regenerate_ids))
            previous = {asset.asset_id: asset for asset in snapshot.assets}
            inventory: dict[str, list[CatalogAlias]] = {}
            paths = {}
            for path in source_files:
                if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                # Catalog paths never escape the declared source directory.
                if not path.resolve().is_relative_to(root):
                    raise ValueError(f'Sticker alias escapes source directory: {path.relative_to(root)}')
                digest = await asyncio.to_thread(content_hash, path)
                asset_id = 'sha256:' + digest
                relative = path.relative_to(root)
                alias = CatalogAlias(relative.as_posix(), relative.parts[0] if len(relative.parts) > 1 else '')
                inventory.setdefault(asset_id, []).append(alias)
                paths[alias.path] = asset_id
            # Explicit source cleanup changes the active inventory only. Prior
            # revisions retain removed originals' cards, vectors and provenance.
            known = set(inventory) if prune_missing else set(previous) | set(inventory)
            unknown = (set(regenerate_ids) | set(corrections)) - known
            if unknown:
                raise ValueError(f'Unknown asset IDs: {sorted(unknown)}')
            packs = {alias.pack for asset in snapshot.assets for alias in asset.aliases} | {alias.pack for aliases in inventory.values() for alias in aliases}
            packs.update(Path(path).parts[0] if len(Path(path).parts) > 1 else '' for path in unsupported)
            if set(regenerate_packs) - packs:
                raise ValueError(f'Unknown packs: {sorted(set(regenerate_packs) - packs)}')
            unknown_files = set(regenerate_files) - (set(paths) | set(unsupported) |
                {a.path for old in previous.values() for a in old.aliases})
            if unknown_files:
                raise ValueError(f'Unknown relative files: {sorted(unknown_files)}')
            planned = []
            selected = False
            media_preparation = asdict(self.media_config)
            for asset_id in sorted(known):
                old = previous.get(asset_id)
                # Present originals take their current locations and packs from the
                # source tree. Prior revisions retain old locations after a move.
                # Completely missing originals remain evidence, except when their
                # filename now belongs to different bytes; retrieval checks availability.
                aliases = {alias.path: alias for alias in inventory.get(asset_id, old.aliases if old else ())
                           if paths.get(alias.path, asset_id) == asset_id}
                regenerate = asset_id in regenerate_ids or any(alias.pack in regenerate_packs or alias.path in regenerate_files for alias in aliases.values())
                selected = selected or regenerate
                correction = corrections.get(asset_id, old.corrections if old else {})
                if not isinstance(correction, dict):
                    raise ValueError('Each correction must be an object; use {} to explicitly clear it')
                correction = validate_corrections(correction)
                generated = None if regenerate else (old.generated_card if old else None)
                card = effective_card(generated, correction) if generated else None
                provenance = dict(old.provenance) if old else {}
                if regenerate:
                    provenance.pop('annotation', None)
                same_space = old is not None and old.provenance.get('embedding_space_id') == self.embeddings.space_id
                if card is not None:
                    provenance['effective_card_hash'] = card_hash(card)
                same_readings = old is not None and old.card is not None and card is not None and reading_texts(old.card) == reading_texts(card) and old.provenance.get('reading_input') == self.recipe['reading_input']
                previous_preparation = old.media.get('preparation') if old else None
                if old is not None and old.media.get('animated') is False and previous_preparation:
                    # A known static original always supplies one image; changing
                    # the animation frame allowance cannot change its pixels.
                    previous_preparation = {**previous_preparation, 'max_frames': media_preparation['max_frames']}
                # Explicit regeneration also rebuilds sampled visual evidence:
                # decoder fixes can change pixels without changing source bytes.
                same_visual = not regenerate and old is not None and old.provenance.get('image_input') == self.recipe['image_input'] and (
                    previous_preparation == media_preparation if self.visual_embedding_source == 'image'
                    else appearance_text(old.card) == appearance_text(card))
                readings = old.reading_vectors if same_space and same_readings else None
                image = old.image_vector if same_space and same_visual and self._needs_visual(card) else None
                ready = card is not None and readings is not None and (image is not None or not self._needs_visual(card))
                asset = CatalogAsset(asset_id, asset_id.removeprefix('sha256:'), tuple(aliases.values()),
                    old.media if old else {}, generated, correction, card, provenance, image, readings,
                    'ready' if ready else 'pending')
                planned.append(asset)
            if (regenerate_ids or regenerate_packs or regenerate_files) and not selected:
                raise ValueError('No processable assets match the selected files, packs or IDs; '
                                 'unsupported files: ' + ', '.join(unsupported))
            if not planned and not (prune_missing and snapshot.assets):
                return BuildResult(None, False, 0, (), unsupported)
            # Inventory and copied records commit together before any paid work.
            # A crash cannot leave a revision containing only half its intended input.
            revision_id = await self.store.begin_revision(source_root=str(root), recipe=self.recipe,
                assets=planned, dimensions=self.embeddings.config.dimensions, expected_parent=snapshot.revision_id)
        if self.on_revision:
            self.on_revision(revision_id)
        failures, completed = [], 0
        async with self.store.build_lease(revision_id):
            snapshot = await self.store.load_snapshot(revision_id)
            semaphore = asyncio.Semaphore(self.config.concurrency)
            async def process(asset):
                nonlocal completed
                if asset.state == 'ready':
                    return
                async with semaphore:
                    try:
                        await self._process(revision_id, root, asset)
                        completed += 1
                    except Exception as exc:
                        latest = await self.store.get_asset(asset.asset_id, revision_id)
                        await self._save(revision_id, latest, state='failed', error=f'{type(exc).__name__}: {exc}')
                        failures.append({'asset_id': asset.asset_id, 'error': f'{type(exc).__name__}: {exc}'})
            await asyncio.gather(*(process(asset) for asset in snapshot.assets))
            if not failures:
                await self.store.activate(revision_id)
        return BuildResult(revision_id, not failures, completed, tuple(failures), unsupported)

    async def _process(self, revision_id, root, asset):
        if asset.generated_card is None and not self.provider.capabilities.multimodal_input:
            raise ValueError('Sticker annotation requires a provider with image input')
        prepared = None
        if asset.generated_card is None or (self.recipe['image_embeddings'] and asset.image_vector is None):
            for alias in asset.aliases:
                candidate = (root / alias.path).resolve()
                if candidate.is_relative_to(root) and candidate.is_file() and await asyncio.to_thread(content_hash, candidate) == asset.content_hash:
                    prepared = await asyncio.to_thread(prepare_media, candidate, self.media_config)
                    if prepared.content_hash != asset.content_hash:
                        raise ValueError('Source changed since inventory')
                    break
            if prepared is None:
                raise ValueError('No original alias with matching content remains available')
            if not prepared.frames:
                raise ValueError('Sticker media preparation produced no usable image frames')
        provenance = dict(asset.provenance)
        if asset.generated_card is None:
            parts = [MessagePart(PartKind.TEXT, text='Observed media facts: ' + json.dumps(prepared.facts, ensure_ascii=False))]
            parts.extend(MessagePart(PartKind.IMAGE, mime_type=frame.mime_type, data_b64=frame.data_b64) for frame in prepared.frames)
            response = await self.provider.generate(settings=self.settings,
                messages=[ConversationMessage(MessageRole.USER, parts)], instructions=CARD_PROMPT,
                tools=[], response_schema=CARD_SCHEMA, response_schema_name='sticker_card')
            generated = validate_card(json.loads(response.final_text))
            provenance['annotation'] = {'provider': self.config.provider, 'model': self.config.model,
                'service_tier_requested': self.config.service_tier, 'recipe': CARD_RECIPE,
                'max_output_tokens': self.settings.max_output_tokens,
                'media_preparation': asdict(self.media_config), 'usage': asdict(response.usage)}
            reported_model = response.raw.get('modelVersion') or response.raw.get('model')
            if reported_model:
                provenance['annotation']['model_reported'] = reported_model
            asset = replace(asset, generated_card=generated, card=effective_card(generated, asset.corrections),
                            media=prepared.facts, provenance=provenance, state='pending', error=None)
            # Durable annotation before a second remote service: retrying a failed
            # embedding does not buy the same successful annotation again.
            await self._save(revision_id, asset)
        readings, image = asset.reading_vectors, asset.image_vector
        dimensions = self.embeddings.config.dimensions
        if readings is None:
            texts = reading_texts(asset.card)
            documents = [EmbeddingDocument(f'{asset.asset_id}:reading:{i}', text=text) for i, text in enumerate(texts)]
            readings = np.asarray(await self.embeddings.embed_documents(documents, purpose='sticker'), dtype=np.float32).reshape((-1, dimensions)) if documents else np.empty((0, dimensions), dtype=np.float32)
            await self._save(revision_id, asset, reading_vectors=readings)
        if self._needs_visual(asset.card) and image is None:
            document = (EmbeddingDocument(asset.asset_id + ':image',
                media=tuple(EmbeddingMedia(frame.mime_type, frame.data_b64) for frame in prepared.frames))
                if self.visual_embedding_source == 'image' else
                EmbeddingDocument(asset.asset_id + ':appearance', text=appearance_text(asset.card)))
            vectors = await self.embeddings.embed_documents([document], purpose='sticker')
            image = np.asarray(vectors[0], dtype=np.float32)
        provenance.update({'embedding_space_id': self.embeddings.space_id,
                           'embedding_space': self.embeddings.config.space_spec,
                           'effective_card_hash': card_hash(asset.card),
                           'visual_embedding_source': self.visual_embedding_source,
                           'reading_input': self.recipe['reading_input'], 'image_input': self.recipe['image_input']})
        await self._save(revision_id, asset, media=prepared.facts if prepared else asset.media,
                         image_vector=image, reading_vectors=readings, provenance=provenance, state='ready', error=None)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, help='Original sticker directory (default: APP_DATA_DIR/stickers)')
    parser.add_argument('--asset-id', action='append', default=[], help='Regenerate this content ID or sid reference; may repeat')
    parser.add_argument('--file', action='append', default=[], help='Regenerate this exact relative file alias; may repeat')
    parser.add_argument('--pack', action='append', default=[], help='Regenerate this exact pack; may repeat')
    parser.add_argument('--corrections', type=Path, help='JSON object keyed by asset ID; replaces explicit correction records')
    parser.add_argument('--prune-missing', action='store_true',
                        help='Remove entries with no remaining original from the active catalog; keep historical revisions')
    parser.add_argument('--resume', help='Resume the reported unfinished revision with the same settings')
    return parser.parse_args(argv)


async def run(args):
    load_dotenv(Path.cwd() / '.env')
    config, build_config = load_config(require_telegram=False), BuildConfig.from_env()
    # Build transport waits are independent of interactive chat latency settings.
    if build_config.provider in {'gemini', 'openai'}:
        route = replace(getattr(config, build_config.provider), request_timeout_s=build_config.request_timeout_s)
        config = replace(config, **{build_config.provider: route})
    else:
        config = replace(config, chat_completions=tuple(replace(route, request_timeout_s=build_config.request_timeout_s)
            if route.name == build_config.provider else route for route in config.chat_completions))
    provider = build_provider(config, build_config.provider)
    embeddings = EmbeddingClient(sticker_embedding_config())
    store = PostgresStore(config.database_url)
    try:
        await store.initialize()
        catalog = StickerCatalogStore(store)
        await catalog.initialize()
        corrections = json.loads(args.corrections.read_text()) if args.corrections else None
        builder = CatalogBuilder(catalog, provider, embeddings, settings=config.default_session_settings(), config=build_config,
            on_revision=lambda identity: print(json.dumps({'staging_revision': identity}), flush=True))
        result = await builder.build(args.source or config.sticker_dir, regenerate_ids=args.asset_id,
            regenerate_packs=args.pack, regenerate_files=args.file, corrections=corrections, resume=args.resume,
            prune_missing=args.prune_missing)
        print(json.dumps(asdict(result), ensure_ascii=False))
        return 0 if result.active else 1
    finally:
        await store.close()
        await embeddings.aclose()
        await provider.aclose()


def main(argv=None):
    try:
        return asyncio.run(run(parse_args(argv)))
    except (ValueError, RuntimeError, OSError) as exc:
        print(json.dumps({'error': str(exc)}, ensure_ascii=False))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
