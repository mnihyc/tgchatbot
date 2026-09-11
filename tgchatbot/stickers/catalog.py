"""Contextual sticker retrieval over one immutable PostgreSQL catalog revision.

Reading and image scores stay in their own channels. Rank interleaving supplies
possibilities; the conversation agent judges the actual words and visual evidence.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from itertools import zip_longest
from typing import Any

import numpy as np

from tgchatbot.embeddings import EmbeddingService
from tgchatbot.stickers.config import StickerConfig
from tgchatbot.stickers.media import MediaConfig, content_hash, prepare_media
from tgchatbot.stickers.persona import compact_persona_dict, merge_persona_dicts
from tgchatbot.stickers.plan import StickerRetrievalPlan
from tgchatbot.stickers.session_style import SessionStyleMemory, SessionStyleState


@dataclass(frozen=True)
class StickerIndexEntry:
    asset: Any
    absolute_path: Path
    revision_id: str

    @property
    def sticker_id(self):
        return self.asset.asset_id

    @property
    def summary(self):
        card = self.asset.card or {}
        return card.get('action') or card.get('caption') or 'Sticker'

    @property
    def emoji(self):
        return self.asset.media.get('emoji')

    @property
    def animated(self):
        return bool(self.asset.media.get('animated'))

    @property
    def source_pack_id(self):
        return self.asset.aliases[0].pack if self.asset.aliases else ''


@dataclass(frozen=True)
class StickerMatch:
    entry: StickerIndexEntry
    channels: tuple[str, ...] = ()
    reading: dict[str, str] | None = None
    recently_delivered: bool = False
    visually_similar_deliveries: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Index:
    revision_id: str | None = None
    recipe: dict = field(default_factory=dict)
    assets: tuple = ()
    reading_matrix: np.ndarray | None = None
    reading_assets: tuple[int, ...] = ()
    reading_positions: tuple[int, ...] = ()
    image_matrix: np.ndarray | None = None
    image_assets: tuple[int, ...] = ()


def _interleave(*rankings):
    """Round-robin ranked channels, skipping duplicate assets without extra votes."""
    seen = set()
    for positions in zip_longest(*rankings):
        for value in positions:
            if value is not None and value not in seen:
                seen.add(value)
                yield value



class StickerCatalog:
    def __init__(self, catalog_store, sticker_root: Path, persona_store=None, *,
                 embedding_client: EmbeddingService | None = None, delivery_store=None,
                 config: StickerConfig | None = None, media_config: MediaConfig | None = None):
        self.store = catalog_store
        self.sticker_root = Path(sticker_root)
        self.persona_store = persona_store
        self.embeddings = embedding_client
        self.delivery_store = delivery_store
        self.config = config or StickerConfig.from_env()
        self.media_config = media_config or MediaConfig.from_env()
        self.style_memory = SessionStyleMemory(max_sessions=self.config.cached_sessions)
        self._index = _Index()
        self._loaded = False
        self._load_lock = asyncio.Lock()
        self.entries_by_id: dict[str, StickerIndexEntry] = {}

    @property
    def loaded(self):
        return self._loaded

    def stats(self):
        return {'stickers': len(self._index.assets), 'revision': self._index.revision_id,
                'readings': len(self._index.reading_assets),
                'image_vectors': len(self._index.image_assets)}

    async def aensure_loaded(self):
        if self.store is None:
            self._loaded = True
            return
        async with self._load_lock:
            current = await self.store.active_revision_id()
            if self._loaded and current == self._index.revision_id:
                return
            snapshot = await self.store.load_snapshot(current)
            assets = snapshot.assets
            reading_rows, reading_assets, reading_positions = [], [], []
            image_rows, image_assets = [], []
            for i, asset in enumerate(assets):
                if asset.reading_vectors is not None:
                    for position, vector in enumerate(asset.reading_vectors):
                        reading_rows.append(vector)
                        reading_assets.append(i)
                        reading_positions.append(position)
                if asset.image_vector is not None:
                    image_rows.append(asset.image_vector)
                    image_assets.append(i)
            replacement = _Index(snapshot.revision_id, snapshot.recipe, assets,
                np.asarray(reading_rows, dtype=np.float32) if reading_rows else None,
                tuple(reading_assets), tuple(reading_positions),
                np.asarray(image_rows, dtype=np.float32) if image_rows else None, tuple(image_assets))
            # All derived arrays and row IDs belong to this snapshot. A query retains
            # its own reference even if another query observes a newly activated head.
            self._index = replacement
            self.entries_by_id = {a.asset_id: StickerIndexEntry(a, self.sticker_root / a.aliases[0].path,
                snapshot.revision_id) for a in assets if a.aliases}
            self._loaded = True

    def get_by_sticker_id(self, sticker_id):
        return self.entries_by_id.get(sticker_id)

    def reset_session(self, session_id):
        self.style_memory.clear(session_id)

    async def _aensure_session_state(self, session_id):
        state = self.style_memory.get(session_id)
        if not state.session_persona_loaded:
            saved = await self.persona_store.get_sticker_persona(session_id) if self.persona_store else None
            state.set_session_persona(saved)
        return state

    async def adescribe_style_context(self, session_id='default'):
        state = await self._aensure_session_state(session_id)
        if self.delivery_store:
            rows = await self.delivery_store.recent(session_id, limit=self.config.recent_deliveries)
            state.recent_sticker_ids = [row['sticker_id'] for row in rows]
            state.recent_source_pack_ids = list(dict.fromkeys(
                alias.pack for row in rows for entry in [self.entries_by_id.get(row['sticker_id'])]
                if entry for alias in entry.asset.aliases))
            state.source_pack_id = state.recent_source_pack_ids[0] if state.recent_source_pack_ids else None
        return state.to_context_dict()

    async def adescribe_persona_context(self, session_id='default', plan=None):
        state = await self._aensure_session_state(session_id)
        return self._persona_context(state, plan)

    @staticmethod
    def _persona_context(state, plan):
        saved = compact_persona_dict(state.session_persona)
        requested = compact_persona_dict(plan.persona.as_dict()) if plan else {}
        mode = plan.persona_mode if plan else 'inherit'
        if mode == 'clear_session_persona':
            saved = {}
        return {'persona_mode': mode, 'session_persona': saved,
                'effective_persona': merge_persona_dicts(saved, requested)}

    async def aprepare_query_context(self, *, plan, session_id, persist_persona, expected_scope=None):
        state = await self._aensure_session_state(session_id)
        context = self._persona_context(state, plan)
        change = (plan.persona_mode == 'clear_session_persona' or
                  plan.persona_mode == 'merge_and_remember' and bool(plan.persona.as_dict()))
        if persist_persona and plan.send and change:
            value = context['effective_persona'] if plan.persona_mode != 'clear_session_persona' else {}
            if self.persona_store:
                kwargs = {'expected_scope': expected_scope} if expected_scope is not None else {}
                if plan.persona_mode == 'clear_session_persona':
                    await self.persona_store.clear_sticker_persona(session_id, **kwargs)
                else:
                    await self.persona_store.save_sticker_persona(session_id, value, **kwargs)
            # Write-through after commit: failed/reset-racing writes cannot leak into cache.
            state.set_session_persona(value)
        return state, context

    def _available_path(self, asset, *, verify=True):
        for alias in asset.aliases:
            path = self.sticker_root / alias.path
            try:
                if path.is_file() and (not verify or content_hash(path) == asset.content_hash):
                    return path
            except OSError:
                continue
        return None

    async def aget_available(self, sticker_id):
        await self.aensure_loaded()
        entry = self.entries_by_id.get(sticker_id)
        if entry is None:
            return None
        path = await asyncio.to_thread(self._available_path, entry.asset)
        return StickerIndexEntry(entry.asset, path, entry.revision_id) if path else None

    async def evidence(self, matches):
        from tgchatbot.domain.models import MessagePart, PartKind
        parts = []
        for match in matches:
            entry = match.entry
            origin = 'sticker_candidate:' + entry.sticker_id
            try:
                prepared = await asyncio.to_thread(prepare_media, entry.absolute_path, self.media_config)
                if prepared.content_hash != entry.asset.content_hash:
                    raise ValueError('asset bytes changed')
                parts.append(MessagePart(kind=PartKind.TEXT, text=f'Candidate {entry.sticker_id}; '
                    f'animated={entry.animated}; sampled frame times: {prepared.facts["frame_times_s"]} seconds; '
                    'intermediate animation events may be omitted.', origin=origin, remote_sync=False))
                for frame in prepared.frames:
                    parts.append(MessagePart(kind=PartKind.IMAGE, data_b64=frame.data_b64,
                        mime_type=frame.mime_type, text=f'{entry.sticker_id} at {frame.timestamp_s:g}s',
                        origin=origin, remote_sync=False))
            except (OSError, ValueError) as exc:
                parts.append(MessagePart(kind=PartKind.TEXT, text=f'Candidate {entry.sticker_id}: '
                    f'visual evidence unavailable ({type(exc).__name__}); description only.', origin=origin, remote_sync=False))
        return parts

    async def achoose(self, *, plan: StickerRetrievalPlan, session_id='default',
                      session_state: SessionStyleState | None = None, persona_context=None):
        if not plan.send:
            return []
        await self.aensure_loaded()
        index = self._index
        if not index.assets:
            return []
        state = session_state or await self._aensure_session_state(session_id)
        persona_context = persona_context or self._persona_context(state, plan)
        await self.adescribe_style_context(session_id)
        recent = set(state.recent_sticker_ids)
        # Eligibility is evaluated before limiting a channel, so unavailable or
        # explicitly excluded formats cannot occupy the displayed slots.
        def eligible(asset):
            compatibility = (asset.card or {}).get('compatibility', {})
            return bool(asset.card) and (plan.allow_animation or not asset.media.get('animated')) and all(
                compatibility.get(key, 0) <= maximum for key, maximum in (
                ('harshness_level', plan.max_harshness), ('intimacy_level', plan.max_intimacy),
                ('meme_dependence_level', plan.max_meme_dependence))) and (
                not plan.required_pack or any(a.pack == plan.required_pack for a in asset.aliases)) and (
                not plan.required_character_family or plan.required_character_family in
                asset.family_ids) and (
                plan.text_priority != 'require' or bool(asset.card.get('caption')))
        allowed = {i for i, asset in enumerate(index.assets) if eligible(asset)}
        paths = await asyncio.to_thread(lambda: {i: path for i in allowed
            if (path := self._available_path(index.assets[i], verify=False)) is not None})
        allowed = set(paths)
        depth = max(self.config.retrieval_depth, plan.candidate_budget)
        rankings, best_reading, channel_members = [], {}, {}
        # Caption/ID identity is a separate route, not an AND across a prose query.
        literal = ' '.join((plan.text_hint or plan.intent_core).casefold().split())
        exact = [i for i in allowed if index.assets[i].asset_id == plan.intent_core or
                 (literal and ' '.join((index.assets[i].card or {}).get('caption', '').casefold().split()) == literal)]
        if exact:
            rankings.append(sorted(exact))
            channel_members['literal'] = set(exact)
        available_vectors = index.reading_matrix is not None or index.image_matrix is not None
        if available_vectors and not (self.embeddings and self.embeddings.enabled) and not exact:
            raise ValueError('Sticker semantic search requires the configured embedding credentials; known IDs and exact captions remain available')
        if not exact and available_vectors and self.embeddings and self.embeddings.enabled:
            if index.recipe.get('embedding_space_id') != self.embeddings.config.space_id:
                raise ValueError('Sticker embedding space changed; rebuild the catalog vectors with the configured sticker embedding route')
            intended = '; '.join(filter(None, [plan.intent_core, *plan.secondary_goals,
                plan.emotion_tone, plan.social_goal, plan.text_hint,
                plan.selection_lens.social_read, plan.selection_lens.subtext,
                *plan.semantic_focus.request_texts(), *plan.text_constraints.must_include]))
            visual = '; '.join(filter(None, [intended, plan.visual_hint,
                plan.selection_lens.face_and_pose, *plan.visual_focus.request_texts()]))
            queries = list(dict.fromkeys([intended, visual]))
            query_vectors = dict(zip(queries, await asyncio.gather(*(
                self.embeddings.embed_query(text, purpose='sticker') for text in queries))))
            def reading_rank(query):
                scores = index.reading_matrix @ query if index.reading_matrix is not None else []
                by_asset = {}
                for row, score in enumerate(scores):
                    i = index.reading_assets[row]
                    if i in allowed and (i not in by_asset or score > by_asset[i]):
                        by_asset[i] = float(score)
                        best_reading[i] = index.reading_positions[row]
                return sorted(by_asset, key=lambda i: (-by_asset[i], index.assets[i].asset_id))
            def image_rank(query):
                scores = index.image_matrix @ query if index.image_matrix is not None else []
                by_asset = {i: float(scores[row]) for row, i in enumerate(index.image_assets) if i in allowed}
                return sorted(by_asset, key=lambda i: (-by_asset[i], index.assets[i].asset_id))
            reading, visual_rank = reading_rank(query_vectors[intended]), image_rank(query_vectors[visual])
            global_lanes = []
            for name, values in [('reading', reading), ('image', visual_rank)]:
                lane = values[:depth]
                rankings.append(lane)
                global_lanes.append(lane)
                channel_members[name] = set(lane)
            # Preferences get a companion retrieval lane, never a global exclusion.
            persona = persona_context.get('effective_persona', {})
            identity = persona.get('visual_identity', {})
            pack = plan.prefer_pack or identity.get('prefer_pack')
            family = plan.preferred_character_family
            if plan.style_goal == 'preserve' and not pack and not family:
                pack = state.source_pack_id
            preferred = {i for i in allowed if (pack and any(a.pack == pack for a in index.assets[i].aliases)) or
                (family and family in index.assets[i].family_ids)}
            family_lane = None
            if preferred and plan.style_goal != 'ignore_style':
                family_lane = list(_interleave([i for i in reading if i in preferred],
                                                [i for i in visual_rank if i in preferred]))[:depth]
                rankings.append(family_lane)
                channel_members['preferred_family_or_pack'] = set(family_lane)
            style_words = [identity.get(k, '') for k in ('character_archetype', 'rendering_style', 'palette_mood')]
            style_words += list(identity.get('style_hints') or []) + plan.style_hints
            style_words += list((persona.get('affect_profile') or {}).values())
            if any(style_words) and plan.style_goal != 'ignore_style':
                preferred_text = visual + '; preferred expression/style: ' + '; '.join(filter(None, style_words))
                preferred_vector = await self.embeddings.embed_query(preferred_text, purpose='sticker')
                values = image_rank(preferred_vector)[:depth]
                rankings.append(values)
                channel_members['preferred_appearance'] = set(values)
            # Fresh alternatives are exposed alongside the strongest matches. A
            # repeated best fit stays eligible; visual proximity is not semantic proof.
            if recent:
                fresh = list(_interleave([i for i in reading if index.assets[i].asset_id not in recent],
                                         [i for i in visual_rank if index.assets[i].asset_id not in recent]))[:depth]
                if plan.diversity_preference == 'prefer_fresh_variant':
                    rankings.insert(0, fresh)
                else:
                    rankings.append(fresh)
                channel_members['fresh_alternative'] = set(fresh)
            if plan.style_goal == 'prefer_switch' and state.recent_source_pack_ids:
                different = [i for i in _interleave(reading, visual_rank) if not any(
                    alias.pack in state.recent_source_pack_ids for alias in index.assets[i].aliases)]
                rankings.append(different[:depth])
                channel_members['different_pack'] = set(different[:depth])
            if family_lane and preferred != allowed:
                # With a small display budget, two global channels must not use
                # every slot before the requested familiar alternative is seen.
                # Keep their existing balanced ordering inside the global lane;
                # novelty/style remain companion lanes rather than exclusions.
                global_lane = list(_interleave(reading[:depth], visual_rank[:depth]))
                companions = [lane for lane in rankings if lane is not family_lane
                    and not any(lane is global_lane for global_lane in global_lanes)]
                rankings = [global_lane, family_lane, *companions]
        selected = _interleave(*rankings)
        result = []
        delivered_vectors = [(a.asset_id, a.image_vector) for a in index.assets if a.asset_id in recent and a.image_vector is not None]
        for i in selected:
            asset = index.assets[i]
            verified_path = await asyncio.to_thread(self._available_path, asset)
            if verified_path is None:
                continue
            readings = asset.card.get('readings') or []
            similar = tuple(asset_id for asset_id, vector in delivered_vectors if asset_id != asset.asset_id
                and asset.image_vector is not None and float(asset.image_vector @ vector) >= self.config.near_duplicate_similarity)
            result.append(StickerMatch(StickerIndexEntry(asset, verified_path, index.revision_id),
                tuple(name for name, members in channel_members.items() if i in members),
                readings[best_reading[i]] if i in best_reading and best_reading[i] < len(readings) else None,
                asset.asset_id in recent, similar))
            if len(result) >= min(plan.candidate_budget, self.config.max_candidates):
                break
        return result
