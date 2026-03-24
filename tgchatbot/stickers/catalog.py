from __future__ import annotations

import json
import os
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from tgchatbot.stickers.plan import StickerRetrievalPlan
from tgchatbot.stickers.retrieval_client import TantivyRetrieverClient
from tgchatbot.stickers.schema import STICKER_SCHEMA_VERSION
from tgchatbot.stickers.semantic_index import EmbeddingProvider, SemanticIndex
from tgchatbot.stickers.session_style import SessionStyleMemory, SessionStyleState, normalize_style_goal

_WORD_RE = re.compile(r"[\w+\-']+", re.UNICODE)
_CJK_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uac00-\ud7af]")
_ANIMATED_FORMATS = {"webm", "gif", "mp4"}
_PACK_TOKEN_STOPWORDS = {'pack', 'sticker', 'stickers', 'set', 'by', 'bot', 'telegram'}
_STYLE_TOKEN_STOPWORDS = {'style', 'sticker', 'stickers', 'rendering', 'character', 'family'}
_OPAQUE_CLUSTER_RE = re.compile(r'^style_cluster_\d+$')


@dataclass(slots=True)
class StickerIndexEntry:
    sticker_id: str
    relative_path: str
    absolute_path: Path
    source_format: str
    source_pack_id: str | None
    source_pack_hash: int
    summary: str
    preview_text: str
    selection_notes: str
    emoji: str | None
    caption_mode: str
    source_overlay_text: str
    source_overlay_text_normalized: str
    caption_meaning_en: str
    caption_meaning_zh: str
    source_overlay_languages: list[str]
    source_ocr_confidence: float
    source_ocr_confidence_bucket: int
    caption_dominance_score: int
    caption_semantic_text: str
    sticker_semantic_text: str
    style_text: str
    style_cluster: str | None
    style_cluster_id: int
    semantic_signature: str
    caption_card: dict[str, Any]
    subtle_cue_card: dict[str, Any]
    sticker_card: dict[str, Any]
    style_card: dict[str, Any]
    harshness_level: int = 1
    intimacy_level: int = 1
    meme_dependence_level: int = 1
    animated: bool = False
    metadata_json: str = '{}'
    parsed_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_row(cls, root: Path, row: sqlite3.Row) -> 'StickerIndexEntry':
        def _value(name: str, default: Any = '') -> Any:
            return row[name] if name in row.keys() else default

        def _json(name: str) -> dict[str, Any]:
            raw = str(_value(name, '{}') or '{}')
            try:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}

        metadata_raw = str(_value('metadata_json', '{}') or '{}')
        try:
            parsed = json.loads(metadata_raw)
            if not isinstance(parsed, dict):
                parsed = {}
        except Exception:
            parsed = {}
        source_overlay_languages_raw = _value('source_overlay_languages', '[]')
        try:
            source_overlay_languages = json.loads(source_overlay_languages_raw) if source_overlay_languages_raw else []
            if not isinstance(source_overlay_languages, list):
                source_overlay_languages = []
        except Exception:
            source_overlay_languages = []
        summary = str(_value('summary', '') or '')
        preview_text = str(_value('preview_text', summary) or summary)
        return cls(
            sticker_id=str(_value('sticker_id')),
            relative_path=str(_value('relative_path')),
            absolute_path=(root / str(_value('relative_path'))).resolve(),
            source_format=str(_value('source_format', 'unknown') or 'unknown'),
            source_pack_id=(str(_value('source_pack_id', '')).strip() or None),
            source_pack_hash=int(_value('source_pack_hash', 0) or 0),
            summary=summary,
            preview_text=preview_text,
            selection_notes=str(_value('selection_notes', '') or ''),
            emoji=(str(_value('emoji', '')).strip() or None),
            caption_mode=str(_value('caption_mode', 'mixed') or 'mixed'),
            source_overlay_text=str(_value('source_overlay_text', '')),
            source_overlay_text_normalized=str(_value('source_overlay_text_normalized', '')),
            caption_meaning_en=str(_value('caption_meaning_en', '')),
            caption_meaning_zh=str(_value('caption_meaning_zh', '')),
            source_overlay_languages=[str(x) for x in source_overlay_languages],
            source_ocr_confidence=float(_value('source_ocr_confidence', 0.0) or 0.0),
            source_ocr_confidence_bucket=int(_value('source_ocr_confidence_bucket', 0) or 0),
            caption_dominance_score=int(_value('caption_dominance_score', 0) or 0),
            caption_semantic_text=str(_value('caption_semantic_text', '')),
            sticker_semantic_text=str(_value('sticker_semantic_text', '')),
            style_text=str(_value('style_text', '')),
            style_cluster=(str(_value('style_cluster', '')).strip() or None),
            style_cluster_id=int(_value('style_cluster_id', -1) or -1),
            semantic_signature=str(_value('semantic_signature', '') or ''),
            caption_card=_json('caption_card_json'),
            subtle_cue_card=_json('subtle_cue_card_json'),
            sticker_card=_json('sticker_card_json'),
            style_card=_json('style_card_json'),
            harshness_level=int(_value('harshness_level', 1) or 1),
            intimacy_level=int(_value('intimacy_level', 1) or 1),
            meme_dependence_level=int(_value('meme_dependence_level', 1) or 1),
            animated=bool(int(_value('animated', 0) or 0) or str(_value('source_format', '')).lower() in _ANIMATED_FORMATS),
            metadata_json=metadata_raw,
            parsed_metadata=parsed,
        )


@dataclass(slots=True)
class StickerMatch:
    entry: StickerIndexEntry
    score: float
    matched_terms: list[str]
    reasons: dict[str, float]
    base_score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    match_profile: dict[str, Any] = field(default_factory=dict)
    selection_summary: str = ''


class StickerCatalog:
    def __init__(self, index_db_path: Path, sticker_root: Path) -> None:
        self.index_db_path = Path(index_db_path)
        self.sticker_root = Path(sticker_root)
        self._loaded = False
        self._stats_cache: dict[str, int | bool] = {'loaded': False, 'stickers': 0, 'packs': 0, 'animated': 0, 'static': 0}
        self.entries_by_id: dict[str, StickerIndexEntry] = {}
        self.cluster_style_profiles: dict[str, dict[str, Any]] = {}
        self.style_memory = SessionStyleMemory()
        retriever_url = os.getenv('STICKER_RETRIEVER_URL', 'http://127.0.0.1:4107')
        self.retriever = TantivyRetrieverClient(retriever_url)
        self.semantic_index = SemanticIndex(
            self.index_db_path.parent,
            require_ready=True,
            embedding_provider=EmbeddingProvider(
                api_key=os.getenv('OPENAI_API_KEY'),
                model=os.getenv('STICKER_EMBEDDING_MODEL', 'text-embedding-3-large'),
                dimensions=int(os.getenv('STICKER_EMBEDDING_DIMENSIONS', '1024')),
                base_url=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                cache_db_path=self.index_db_path.parent / 'query_embedding_cache.sqlite3',
            ),
        )

    @property
    def loaded(self) -> bool:
        return self._loaded

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.index_db_path, timeout=20.0)
        con.row_factory = sqlite3.Row
        con.execute('PRAGMA busy_timeout=20000')
        return con

    def load(self) -> None:
        self.entries_by_id = {}
        self.cluster_style_profiles = {}
        if not self.index_db_path.exists():
            raise RuntimeError(f'Sticker index database not found: {self.index_db_path}')
        with self._connect() as con:
            rows = con.execute('SELECT * FROM stickers').fetchall()
            meta_rows = {str(row['key']): str(row['value']) for row in con.execute('SELECT key, value FROM meta').fetchall()}
        if meta_rows.get('needs_rebuild') == '1':
            raise RuntimeError('Sticker index requires a full rebuild with PaddleOCR, semantic analysis, embeddings, and Tantivy docs before use.')
        if meta_rows.get('schema_version') != STICKER_SCHEMA_VERSION:
            raise RuntimeError(
                f'Sticker index schema mismatch. Expected {STICKER_SCHEMA_VERSION}, got {meta_rows.get("schema_version") or "missing"}. Rebuild the sticker index.'
            )
        for row in rows:
            entry = StickerIndexEntry.from_row(self.sticker_root.resolve(), row)
            self.entries_by_id[entry.sticker_id] = entry
        self.cluster_style_profiles = _build_style_cluster_profiles(self.entries_by_id.values())
        animated = sum(1 for entry in self.entries_by_id.values() if entry.animated)
        packs = len({entry.source_pack_id or '' for entry in self.entries_by_id.values()})
        self._stats_cache = {'loaded': True, 'stickers': len(self.entries_by_id), 'packs': packs, 'animated': animated, 'static': max(0, len(self.entries_by_id) - animated)}
        self.semantic_index.ensure_ready()
        self.retriever.ensure_healthy(expected_schema_version=STICKER_SCHEMA_VERSION, expected_service='tantivy_retriever')
        self._loaded = True

    def stats(self) -> dict[str, int | bool]:
        if not self._loaded:
            self.load()
        return dict(self._stats_cache)

    def get_by_sticker_id(self, sticker_id: str) -> StickerIndexEntry | None:
        if not self._loaded:
            self.load()
        return self.entries_by_id.get(str(sticker_id))

    def describe_style_context(self, session_id: str = 'default') -> dict[str, Any]:
        if not self._loaded:
            self.load()
        return self.style_memory.get(session_id).to_context_dict()

    def choose(self, *, plan: StickerRetrievalPlan, session_id: str = 'default') -> list[StickerMatch]:
        if not self._loaded:
            self.load()
        if not plan.send or not plan.intent_core:
            return []
        caption_query_text = plan.caption_query_text()
        sticker_query_text = plan.sticker_query_text()
        lexical_hits = self._search_lexical(
            caption_query_text=caption_query_text,
            sticker_query_text=sticker_query_text,
            caption_importance=plan.text_priority,
            allow_animation=plan.allow_animation,
            limit=max(40, plan.candidate_budget * 12),
        )
        semantic_hits = self.semantic_index.search(
            caption_query_text=caption_query_text,
            sticker_query_text=sticker_query_text,
            top_k=max(50, plan.candidate_budget * 12),
        )
        lexical_map = {hit.sticker_id: hit for hit in lexical_hits}
        semantic_by_id: dict[str, dict[str, float]] = defaultdict(dict)
        for hit in semantic_hits:
            semantic_by_id[hit.sticker_id][hit.channel] = hit.score
        candidate_ids = list(dict.fromkeys([*lexical_map.keys(), *semantic_by_id.keys()]))
        session_style = self.style_memory.get(session_id)
        request_terms = plan.request_terms()
        style_terms = plan.style_request_terms()
        scored: list[StickerMatch] = []
        for sticker_id in candidate_ids:
            entry = self.entries_by_id.get(sticker_id)
            if entry is None:
                continue
            if entry.animated and not plan.allow_animation:
                continue
            if entry.harshness_level > plan.max_harshness or entry.intimacy_level > plan.max_intimacy or entry.meme_dependence_level > plan.max_meme_dependence:
                continue
            lexical_debug = lexical_map.get(sticker_id)
            caption_lexical = float(lexical_debug.debug.get('caption_lexical', 0.0)) if lexical_debug else 0.0
            sticker_lexical = float(lexical_debug.debug.get('sticker_lexical', 0.0)) if lexical_debug else 0.0
            caption_semantic = float(semantic_by_id.get(sticker_id, {}).get('caption_semantic', 0.0))
            sticker_semantic = float(semantic_by_id.get(sticker_id, {}).get('sticker_semantic', 0.0))
            base_score = _fuse_scores(
                caption_lexical=caption_lexical,
                sticker_lexical=sticker_lexical,
                caption_semantic=caption_semantic,
                sticker_semantic=sticker_semantic,
                text_priority=plan.text_priority,
                caption_mode=entry.caption_mode,
            )
            score_breakdown: dict[str, float] = {'base_fused': base_score}
            matched_terms: list[str] = []

            text_bonus = _caption_evidence_bonus(entry=entry, query_text=caption_query_text, text_priority=plan.text_priority)
            if text_bonus:
                score_breakdown['caption_evidence'] = text_bonus

            keyword_bonus, keyword_terms = _keyword_bonus(entry, request_terms)
            matched_terms.extend(keyword_terms)
            if keyword_bonus:
                score_breakdown['request_overlap'] = keyword_bonus

            compatibility = _compatibility_adjustment(entry=entry, plan=plan)
            if compatibility:
                score_breakdown['compatibility'] = compatibility

            semantic_score, semantic_profile = _semantic_axis_adjustment(entry=entry, plan=plan)
            if semantic_score:
                score_breakdown['semantic_axis_fit'] = semantic_score

            visual_score, visual_profile = _visual_axis_adjustment(entry=entry, plan=plan)
            if visual_score:
                score_breakdown['visual_axis_fit'] = visual_score

            text_constraint_score, text_profile = _text_constraint_adjustment(entry=entry, plan=plan)
            if text_constraint_score:
                score_breakdown['text_constraint_fit'] = text_constraint_score

            style_hint_score, style_hint_profile = _style_hint_adjustment(entry=entry, style_terms=style_terms, style_hints=plan.style_hints)
            if style_hint_score:
                score_breakdown['style_hint_fit'] = style_hint_score

            preferred_pack_score, preferred_pack_profile = _preferred_pack_adjustment(entry=entry, prefer_pack=plan.prefer_pack)
            if preferred_pack_score:
                score_breakdown['preferred_pack_fit'] = preferred_pack_score

            preferred_cluster_score, preferred_cluster_profile = _preferred_cluster_adjustment(
                entry=entry,
                prefer_cluster=plan.prefer_cluster,
                cluster_profiles=getattr(self, 'cluster_style_profiles', {}),
            )
            if preferred_cluster_score:
                score_breakdown['preferred_cluster_fit'] = preferred_cluster_score

            style_fit = session_style.style_bonus(
                candidate_style_cluster=entry.style_cluster,
                candidate_source_pack_id=entry.source_pack_id,
                style_goal=plan.style_goal,
            )
            if style_fit:
                score_breakdown['style_memory_fit'] = style_fit

            safety_fit = _safety_fit_bonus(entry=entry, plan=plan)
            if safety_fit:
                score_breakdown['safety_fit'] = safety_fit

            repeat_penalty = session_style.repeat_penalty(
                sticker_id=entry.sticker_id,
                source_pack_id=entry.source_pack_id,
                style_cluster=entry.style_cluster,
            )
            if repeat_penalty:
                score_breakdown['repeat_penalty'] = -repeat_penalty

            total_score = sum(score_breakdown.values())
            style_relation = session_style.relation(
                candidate_style_cluster=entry.style_cluster,
                candidate_source_pack_id=entry.source_pack_id,
                style_goal=plan.style_goal,
            )
            hard_mismatches = _hard_mismatches(plan=plan, style_relation=style_relation, semantic_profile=semantic_profile, visual_profile=visual_profile, text_profile=text_profile)
            reasons = {name: value for name, value in score_breakdown.items() if value}
            match_profile = {
                'semantic_axes': semantic_profile,
                'visual_cues': visual_profile,
                'text_fit': text_profile,
                'style_relation': {
                    **style_relation,
                    'matched_style_hints': style_hint_profile['matched'],
                    'missing_style_hints': style_hint_profile['missing'],
                    'preferred_pack_requested': preferred_pack_profile['requested'],
                    'preferred_pack_match': preferred_pack_profile['match_label'],
                    'preferred_pack_similarity': preferred_pack_profile['similarity'],
                    'preferred_pack_overlap': preferred_pack_profile['matched_terms'],
                    'preferred_cluster_requested': preferred_cluster_profile['requested'],
                    'preferred_cluster_match': preferred_cluster_profile['match_label'],
                    'preferred_cluster_similarity': preferred_cluster_profile['similarity'],
                    'preferred_cluster_overlap': preferred_cluster_profile['matched_terms'],
                },
                'safety': {
                    'harshness_level': entry.harshness_level,
                    'intimacy_level': entry.intimacy_level,
                    'meme_dependence_level': entry.meme_dependence_level,
                    'limits': plan.intensity_limits.as_dict(),
                },
                'hard_mismatches': hard_mismatches,
            }
            scored.append(
                StickerMatch(
                    entry=entry,
                    score=total_score,
                    matched_terms=sorted(dict.fromkeys(matched_terms))[:24],
                    reasons=reasons,
                    base_score=base_score,
                    score_breakdown=score_breakdown,
                    match_profile=match_profile,
                    selection_summary=_build_selection_summary(entry=entry, plan=plan, match_profile=match_profile),
                )
            )
        scored.sort(key=lambda item: (-item.score, item.entry.relative_path))
        return _select_with_style_goal(scored, top_k=plan.candidate_budget, session_style=session_style, style_goal=plan.style_goal)

    def record_selection(self, *, session_id: str, sticker_id: str) -> None:
        if not self._loaded:
            self.load()
        entry = self.entries_by_id.get(sticker_id)
        if entry is None:
            return
        self.style_memory.get(session_id).record_selection(
            sticker_id=entry.sticker_id,
            source_pack_id=entry.source_pack_id,
            style_cluster=entry.style_cluster,
        )

    def _search_lexical(self, *, caption_query_text: str, sticker_query_text: str, caption_importance: str, allow_animation: bool, limit: int) -> list[Any]:
        payload = {
            'caption_query_text': caption_query_text,
            'sticker_query_text': sticker_query_text,
            'caption_importance': caption_importance,
            'allow_animation': allow_animation,
            'top_k': limit,
        }
        return self.retriever.search(payload)


def _fuse_scores(*, caption_lexical: float, sticker_lexical: float, caption_semantic: float, sticker_semantic: float, text_priority: str, caption_mode: str) -> float:
    if text_priority == 'require' or caption_mode == 'caption_dominant':
        return (0.24 * caption_lexical) + (0.08 * sticker_lexical) + (0.42 * caption_semantic) + (0.26 * sticker_semantic)
    if text_priority == 'ignore' or caption_mode == 'visual_dominant':
        return (0.03 * caption_lexical) + (0.18 * sticker_lexical) + (0.16 * caption_semantic) + (0.63 * sticker_semantic)
    return (0.14 * caption_lexical) + (0.14 * sticker_lexical) + (0.34 * caption_semantic) + (0.38 * sticker_semantic)


def _caption_evidence_bonus(*, entry: StickerIndexEntry, query_text: str, text_priority: str) -> float:
    if text_priority == 'ignore':
        return 0.0
    if not entry.source_overlay_text_normalized and not entry.caption_meaning_en and not entry.caption_meaning_zh:
        return 0.0
    query_terms = set(_tokenize(query_text))
    caption_terms = set(_tokenize([entry.source_overlay_text_normalized, entry.caption_meaning_en, entry.caption_meaning_zh]))
    overlap = query_terms & caption_terms
    if not overlap:
        return 0.0
    base = 0.10 * len(overlap) * max(1, entry.caption_dominance_score)
    if entry.caption_mode == 'caption_dominant':
        base *= 1.35
    if text_priority == 'require':
        base *= 1.35
    return base


def _entry_semantic_text(entry: StickerIndexEntry) -> str:
    return ' '.join([
        entry.summary,
        entry.selection_notes,
        entry.caption_semantic_text,
        entry.sticker_semantic_text,
        entry.style_text,
        entry.caption_card.get('caption_pragmatic_meaning', ''),
        entry.caption_card.get('caption_discourse_role', ''),
        entry.caption_card.get('caption_social_stance', ''),
        entry.caption_card.get('caption_relationship_fit', ''),
        entry.caption_card.get('caption_conversation_role', ''),
        entry.caption_card.get('caption_use_when', ''),
        entry.caption_card.get('caption_avoid_when', ''),
        entry.subtle_cue_card.get('dominant_signal', ''),
        entry.subtle_cue_card.get('micro_expression', ''),
        entry.subtle_cue_card.get('eye_signal', ''),
        entry.subtle_cue_card.get('mouth_signal', ''),
        entry.subtle_cue_card.get('motion_signal', ''),
        entry.subtle_cue_card.get('visual_social_stance', ''),
        entry.subtle_cue_card.get('visual_reply_force', ''),
        entry.subtle_cue_card.get('visual_emotional_valence', ''),
        entry.subtle_cue_card.get('visual_irony_strength', ''),
        entry.subtle_cue_card.get('visual_relationship_fit', ''),
        entry.subtle_cue_card.get('visual_conversation_role', ''),
        entry.subtle_cue_card.get('visual_use_when', ''),
        entry.subtle_cue_card.get('visual_avoid_when', ''),
        entry.sticker_card.get('fused_pragmatic_meaning', ''),
        entry.sticker_card.get('sticker_reaction_type', ''),
        entry.sticker_card.get('sticker_reply_force', ''),
        entry.sticker_card.get('sticker_emotional_valence', ''),
        entry.sticker_card.get('sticker_irony_strength', ''),
        entry.sticker_card.get('sticker_relationship_fit', ''),
        entry.sticker_card.get('sticker_conversation_role', ''),
        entry.sticker_card.get('sticker_use_when', ''),
        entry.sticker_card.get('sticker_avoid_when', ''),
    ])


def _keyword_bonus(entry: StickerIndexEntry, request_terms: list[str]) -> tuple[float, list[str]]:
    terms = set(_tokenize(_entry_semantic_text(entry)))
    overlap = sorted(set(request_terms) & terms)
    if not overlap:
        return 0.0, []
    return 0.10 * len(overlap), overlap[:20]


def _compatibility_adjustment(*, entry: StickerIndexEntry, plan: StickerRetrievalPlan) -> float:
    score = 0.0
    use_terms = set(_tokenize([
        entry.caption_card.get('caption_use_when', ''),
        entry.subtle_cue_card.get('visual_use_when', ''),
        entry.sticker_card.get('sticker_use_when', ''),
    ]))
    avoid_terms = set(_tokenize([
        entry.caption_card.get('caption_avoid_when', ''),
        entry.subtle_cue_card.get('visual_avoid_when', ''),
        entry.sticker_card.get('sticker_avoid_when', ''),
    ]))
    desired_terms = set(plan.request_terms())
    if desired_terms & use_terms:
        score += min(0.30, 0.08 * len(desired_terms & use_terms))
    if desired_terms & avoid_terms:
        score -= min(0.45, 0.10 * len(desired_terms & avoid_terms))
    forbidden_terms = set(plan.avoid_terms())
    if forbidden_terms:
        entry_terms = set(_tokenize(_entry_semantic_text(entry)))
        overlap = forbidden_terms & entry_terms
        if overlap:
            score -= min(0.65, 0.16 * len(overlap))
    if plan.text_priority == 'require' and entry.caption_mode != 'caption_dominant':
        score -= 0.30 if entry.caption_mode == 'visual_dominant' else 0.18
    if plan.text_priority == 'ignore' and entry.caption_mode == 'caption_dominant':
        score -= 0.18
    return score


def _select_with_style_goal(scored: list[StickerMatch], *, top_k: int, session_style: SessionStyleState, style_goal: str) -> list[StickerMatch]:
    if not scored:
        return []
    selected: list[StickerMatch] = []
    pool = list(scored)
    goal = normalize_style_goal(style_goal)
    lambda_diversity = 0.70 if top_k > 1 else 1.0

    def similarity(a: StickerMatch, b: StickerMatch) -> float:
        sim = 0.0
        if a.entry.semantic_signature and b.entry.semantic_signature and a.entry.semantic_signature == b.entry.semantic_signature:
            sim += 0.80
        if a.entry.style_cluster and b.entry.style_cluster and a.entry.style_cluster == b.entry.style_cluster:
            sim += 0.45
        if a.entry.source_pack_id and b.entry.source_pack_id and a.entry.source_pack_id == b.entry.source_pack_id:
            sim += 0.28
        overlap = set(a.matched_terms) & set(b.matched_terms)
        if overlap:
            sim += min(0.22, 0.07 * len(overlap))
        if a.entry.caption_mode == b.entry.caption_mode:
            sim += 0.05
        return sim

    while pool and len(selected) < top_k:
        best_idx = 0
        best_value = float('-inf')
        best_diversity_penalty = 0.0
        best_style_guard = 0.0
        for idx, candidate in enumerate(pool):
            relevance = candidate.score
            diversity_penalty = 0.0
            style_guard_penalty = 0.0
            if not selected:
                if goal in {'keep_current', 'allow_switch'} and session_style.style_cluster and candidate.entry.style_cluster != session_style.style_cluster:
                    anchored = [m for m in pool if m.entry.style_cluster == session_style.style_cluster]
                    if anchored:
                        anchored_best = anchored[0].score
                        if not session_style.should_allow_switch(current_score=anchored_best, candidate_score=candidate.score, style_goal=goal):
                            style_guard_penalty = 1.2
                value = relevance - style_guard_penalty
            else:
                max_similarity = max(similarity(candidate, chosen) for chosen in selected)
                diversity_penalty = (1.0 - lambda_diversity) * max_similarity
                value = (lambda_diversity * relevance) - diversity_penalty
                if goal in {'keep_current', 'allow_switch'} and session_style.style_cluster and candidate.entry.style_cluster != session_style.style_cluster:
                    if not session_style.should_allow_switch(current_score=selected[0].base_score or selected[0].score, candidate_score=candidate.base_score or candidate.score, style_goal=goal):
                        style_guard_penalty = 0.9
                        value -= style_guard_penalty
            if value > best_value:
                best_value = value
                best_idx = idx
                best_diversity_penalty = diversity_penalty
                best_style_guard = style_guard_penalty
        chosen = pool.pop(best_idx)
        if best_diversity_penalty:
            chosen.score_breakdown['diversity_penalty'] = -best_diversity_penalty
            chosen.reasons['diversity_penalty'] = -best_diversity_penalty
        if best_style_guard:
            chosen.score_breakdown['style_switch_guard'] = -best_style_guard
            chosen.reasons['style_switch_guard'] = -best_style_guard
        chosen.score_breakdown['shortlist_value'] = best_value
        chosen.reasons['shortlist_value'] = best_value
        chosen.score = best_value
        selected.append(chosen)
    return selected[:top_k]


def _select_with_style_policy(scored: list[StickerMatch], *, top_k: int, session_style: SessionStyleState, style_policy: str) -> list[StickerMatch]:
    return _select_with_style_goal(scored, top_k=top_k, session_style=session_style, style_goal=style_policy)


def _semantic_axis_adjustment(*, entry: StickerIndexEntry, plan: StickerRetrievalPlan) -> tuple[float, dict[str, Any]]:
    active = plan.semantic_focus.active_fields()
    if not active:
        return 0.0, {'requested': {}, 'matched': [], 'missing': [], 'details': {}}
    details: dict[str, Any] = {}
    matched: list[str] = []
    missing: list[str] = []
    total = 0.0
    sources = _semantic_axis_sources(entry)
    for field_name, request_text in active.items():
        score, detail = _score_requested_field(
            request_text,
            sources.get(field_name, []),
            full_match_score=0.34,
            partial_base=0.16,
            partial_scale=0.18,
            miss_penalty=-0.08,
        )
        detail['score'] = round(score, 4)
        details[field_name] = detail
        total += score
        if score > 0:
            matched.append(field_name)
        else:
            missing.append(field_name)
    return total, {'requested': active, 'matched': matched, 'missing': missing, 'details': details}


def _visual_axis_adjustment(*, entry: StickerIndexEntry, plan: StickerRetrievalPlan) -> tuple[float, dict[str, Any]]:
    active = plan.visual_focus.active_fields()
    if not active:
        return 0.0, {'requested': {}, 'matched': [], 'missing': [], 'details': {}}
    details: dict[str, Any] = {}
    matched: list[str] = []
    missing: list[str] = []
    total = 0.0
    sources = _visual_axis_sources(entry)
    for field_name, request_text in active.items():
        score, detail = _score_requested_field(
            request_text,
            sources.get(field_name, []),
            full_match_score=0.30,
            partial_base=0.14,
            partial_scale=0.16,
            miss_penalty=-0.07,
        )
        detail['score'] = round(score, 4)
        details[field_name] = detail
        total += score
        if score > 0:
            matched.append(field_name)
        else:
            missing.append(field_name)
    return total, {'requested': active, 'matched': matched, 'missing': missing, 'details': details}


def _text_constraint_adjustment(*, entry: StickerIndexEntry, plan: StickerRetrievalPlan) -> tuple[float, dict[str, Any]]:
    caption_text = _entry_caption_text(entry)
    caption_terms = set(_tokenize(caption_text))
    has_visible_text = bool(entry.source_overlay_text_normalized or entry.caption_meaning_en or entry.caption_meaning_zh)
    matched_include: list[str] = []
    missing_include: list[str] = []
    blocked_avoid_terms: list[str] = []
    total = 0.0
    for item in plan.text_constraints.must_include:
        item_text = str(item or '').strip()
        if not item_text:
            continue
        item_terms = set(_tokenize(item_text))
        phrase_hit = item_text.lower() in caption_text.lower() if caption_text else False
        if phrase_hit or (item_terms and item_terms & caption_terms):
            matched_include.append(item_text)
            total += 0.16 if phrase_hit else 0.10
        else:
            missing_include.append(item_text)
            total -= 0.30 if plan.text_priority == 'require' else 0.12
    for item in plan.text_constraints.avoid_text_meanings:
        item_text = str(item or '').strip()
        if not item_text:
            continue
        item_terms = set(_tokenize(item_text))
        phrase_hit = item_text.lower() in caption_text.lower() if caption_text else False
        if phrase_hit or (item_terms and item_terms & caption_terms):
            blocked_avoid_terms.append(item_text)
            total -= 0.18
    if plan.text_priority == 'require':
        total += 0.10 if has_visible_text else -0.28
    elif plan.text_priority == 'prefer' and has_visible_text:
        total += 0.05
    return total, {
        'text_priority': plan.text_priority,
        'caption_mode': entry.caption_mode,
        'has_visible_text': has_visible_text,
        'matched_must_include': matched_include,
        'missing_must_include': missing_include,
        'blocked_avoid_terms': blocked_avoid_terms,
    }


def _style_hint_adjustment(*, entry: StickerIndexEntry, style_terms: list[str], style_hints: list[str]) -> tuple[float, dict[str, list[str]]]:
    if not style_hints:
        return 0.0, {'matched': [], 'missing': []}
    style_text = _entry_style_text(entry)
    style_token_set = set(_tokenize(style_text))
    matched: list[str] = []
    missing: list[str] = []
    total = 0.0
    for hint in style_hints:
        hint_text = str(hint or '').strip()
        if not hint_text:
            continue
        hint_terms = set(_tokenize(hint_text))
        phrase_hit = hint_text.lower() in style_text.lower() if style_text else False
        token_hit = bool(hint_terms and hint_terms & style_token_set)
        if phrase_hit or token_hit:
            matched.append(hint_text)
            total += 0.14 if phrase_hit else 0.09
        else:
            missing.append(hint_text)
            total -= 0.04
    if style_terms and not matched:
        total -= 0.03
    return total, {'matched': matched, 'missing': missing}


def _preferred_pack_adjustment(*, entry: StickerIndexEntry, prefer_pack: str) -> tuple[float, dict[str, Any]]:
    requested = str(prefer_pack or '').strip()
    candidate = str(entry.source_pack_id or '').strip()
    if not requested:
        return 0.0, {'requested': '', 'match_label': 'not_requested', 'similarity': 0.0, 'matched_terms': []}
    if not candidate:
        return -0.05, {'requested': requested, 'match_label': 'candidate_missing_pack_id', 'similarity': 0.0, 'matched_terms': []}

    requested_normalized = _normalize_pack_text(requested)
    candidate_normalized = _normalize_pack_text(candidate)
    requested_terms = set(_pack_terms(requested))
    candidate_terms = set(_pack_terms(candidate))
    overlap = sorted(requested_terms & candidate_terms)
    similarity = SequenceMatcher(None, requested_normalized, candidate_normalized).ratio() if requested_normalized and candidate_normalized else 0.0

    if requested_normalized == candidate_normalized:
        score = 0.42
        match_label = 'exact_preferred_pack'
    elif requested_normalized and candidate_normalized and (requested_normalized in candidate_normalized or candidate_normalized in requested_normalized):
        score = 0.28
        match_label = 'close_preferred_pack'
    elif overlap:
        coverage = len(overlap) / max(1, min(len(requested_terms), 4))
        score = 0.12 + (0.14 * coverage)
        match_label = 'related_preferred_pack'
    elif similarity >= 0.82:
        score = 0.16
        match_label = 'close_preferred_pack'
    elif similarity >= 0.68:
        score = 0.08
        match_label = 'loosely_related_preferred_pack'
    else:
        score = -0.05
        match_label = 'different_pack'
    return score, {
        'requested': requested,
        'match_label': match_label,
        'similarity': round(similarity, 4),
        'matched_terms': overlap[:8],
    }


def _preferred_cluster_adjustment(
    *,
    entry: StickerIndexEntry,
    prefer_cluster: str,
    cluster_profiles: dict[str, dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    requested = str(prefer_cluster or '').strip()
    candidate = str(entry.style_cluster or '').strip()
    if not requested:
        return 0.0, {'requested': '', 'match_label': 'not_requested', 'similarity': 0.0, 'matched_terms': []}
    if not candidate:
        return -0.05, {'requested': requested, 'match_label': 'candidate_missing_cluster', 'similarity': 0.0, 'matched_terms': []}

    requested_normalized = requested.lower()
    candidate_normalized = candidate.lower()
    if requested_normalized == candidate_normalized:
        return 0.46, {'requested': requested, 'match_label': 'exact_preferred_cluster', 'similarity': 1.0, 'matched_terms': []}

    requested_profile = cluster_profiles.get(requested)
    candidate_profile = cluster_profiles.get(candidate)
    if requested_profile and candidate_profile:
        similarity, overlap = _cluster_profile_similarity(requested_profile=requested_profile, candidate_profile=candidate_profile)
        if similarity >= 0.78:
            score = 0.26
            match_label = 'related_preferred_cluster'
        elif similarity >= 0.56:
            score = 0.12
            match_label = 'loosely_related_preferred_cluster'
        else:
            score = -0.05
            match_label = 'different_cluster'
        return score, {
            'requested': requested,
            'match_label': match_label,
            'similarity': round(similarity, 4),
            'matched_terms': overlap[:8],
        }

    if _OPAQUE_CLUSTER_RE.match(requested_normalized) and _OPAQUE_CLUSTER_RE.match(candidate_normalized):
        return -0.05, {'requested': requested, 'match_label': 'different_cluster', 'similarity': 0.0, 'matched_terms': []}

    requested_terms = set(_cluster_name_terms(requested))
    candidate_terms = set(_cluster_name_terms(candidate))
    overlap = sorted(requested_terms & candidate_terms)
    similarity = SequenceMatcher(None, requested_normalized, candidate_normalized).ratio() if requested_normalized and candidate_normalized else 0.0
    if overlap:
        score = 0.14
        match_label = 'related_preferred_cluster'
    elif similarity >= 0.72:
        score = 0.08
        match_label = 'loosely_related_preferred_cluster'
    else:
        score = -0.05
        match_label = 'different_cluster'
    return score, {
        'requested': requested,
        'match_label': match_label,
        'similarity': round(similarity, 4),
        'matched_terms': overlap[:8],
    }


def _safety_fit_bonus(*, entry: StickerIndexEntry, plan: StickerRetrievalPlan) -> float:
    headroom = max(plan.max_harshness - entry.harshness_level, 0)
    headroom += max(plan.max_intimacy - entry.intimacy_level, 0)
    headroom += max(plan.max_meme_dependence - entry.meme_dependence_level, 0)
    return min(0.18, 0.02 * headroom)


def _hard_mismatches(*, plan: StickerRetrievalPlan, style_relation: dict[str, Any], semantic_profile: dict[str, Any], visual_profile: dict[str, Any], text_profile: dict[str, Any]) -> list[str]:
    mismatches: list[str] = []
    if text_profile['missing_must_include'] and plan.text_priority == 'require':
        mismatches.append('missing required caption meaning: ' + ', '.join(text_profile['missing_must_include']))
    if text_profile['blocked_avoid_terms']:
        mismatches.append('caption meaning conflicts with avoid list: ' + ', '.join(text_profile['blocked_avoid_terms']))
    if plan.style_goal == 'keep_current' and style_relation.get('relation_label') == 'style_switch' and style_relation.get('current_style_cluster'):
        mismatches.append('switches away from the current style cluster')
    if semantic_profile['missing'] and len(semantic_profile['requested']) <= 2:
        mismatches.extend(f'missing semantic axis: {field}' for field in semantic_profile['missing'])
    if visual_profile['missing'] and len(visual_profile['requested']) <= 2:
        mismatches.extend(f'missing visual cue: {field}' for field in visual_profile['missing'])
    return mismatches[:6]


def _build_selection_summary(*, entry: StickerIndexEntry, plan: StickerRetrievalPlan, match_profile: dict[str, Any]) -> str:
    parts: list[str] = []
    semantic_profile = match_profile['semantic_axes']
    visual_profile = match_profile['visual_cues']
    text_profile = match_profile['text_fit']
    style_relation = match_profile['style_relation']
    matched_semantic = semantic_profile.get('matched', [])
    matched_visual = visual_profile.get('matched', [])
    if matched_semantic:
        requested = semantic_profile.get('requested', {})
        phrases = [requested[field] for field in matched_semantic[:2] if requested.get(field)]
        if phrases:
            parts.append('semantic fit: ' + ', '.join(phrases))
    elif plan.intent_core:
        parts.append('matches the requested vibe: ' + plan.intent_core)
    if matched_visual:
        requested = visual_profile.get('requested', {})
        phrases = [requested[field] for field in matched_visual[:2] if requested.get(field)]
        if phrases:
            parts.append('visual cues: ' + ', '.join(phrases))
    if text_profile.get('matched_must_include'):
        parts.append('caption supports: ' + ', '.join(text_profile['matched_must_include'][:2]))
    elif text_profile.get('has_visible_text') and plan.text_priority != 'ignore':
        parts.append('caption meaning is explicit')
    preferred_pack_requested = str(style_relation.get('preferred_pack_requested') or '').strip()
    preferred_pack_match = str(style_relation.get('preferred_pack_match') or '').strip()
    if preferred_pack_requested and preferred_pack_match == 'exact_preferred_pack':
        parts.append(f'matches preferred pack {preferred_pack_requested}')
    elif preferred_pack_requested and preferred_pack_match in {'close_preferred_pack', 'related_preferred_pack', 'loosely_related_preferred_pack'}:
        parts.append(f'leans toward preferred pack family {preferred_pack_requested}')
    preferred_cluster_requested = str(style_relation.get('preferred_cluster_requested') or '').strip()
    preferred_cluster_match = str(style_relation.get('preferred_cluster_match') or '').strip()
    if preferred_cluster_requested and preferred_cluster_match == 'exact_preferred_cluster':
        parts.append(f'matches preferred cluster {preferred_cluster_requested}')
    elif preferred_cluster_requested and preferred_cluster_match in {'related_preferred_cluster', 'loosely_related_preferred_cluster'}:
        parts.append(f'leans toward preferred cluster family {preferred_cluster_requested}')
    relation_label = style_relation.get('relation_label')
    if relation_label == 'same_cluster_same_pack':
        parts.append('stays in the current style family and pack')
    elif relation_label == 'same_cluster':
        parts.append('stays in the current style family')
    elif relation_label == 'style_switch' and plan.style_goal == 'prefer_switch':
        parts.append('intentionally switches style')
    elif relation_label == 'style_switch' and plan.style_goal == 'keep_current':
        parts.append('switches style despite the continuity preference')
    tone = str(entry.sticker_card.get('fused_pragmatic_meaning') or entry.summary or entry.preview_text or 'Sticker choice').strip().rstrip('.')
    detail = '; '.join(parts[:3]).strip()
    if detail:
        return f'{tone}. {detail}.'
    return tone + '.'


def _semantic_axis_sources(entry: StickerIndexEntry) -> dict[str, list[str]]:
    return {
        'reaction_type': [
            entry.sticker_card.get('sticker_reaction_type', ''),
            entry.sticker_card.get('fused_pragmatic_meaning', ''),
            entry.caption_card.get('caption_pragmatic_meaning', ''),
            entry.summary,
        ],
        'reply_force': [
            entry.subtle_cue_card.get('visual_reply_force', ''),
            entry.sticker_card.get('sticker_reply_force', ''),
            entry.caption_card.get('caption_reply_force', ''),
        ],
        'emotional_valence': [
            entry.subtle_cue_card.get('visual_emotional_valence', ''),
            entry.sticker_card.get('sticker_emotional_valence', ''),
            entry.caption_card.get('caption_emotional_valence', ''),
            entry.sticker_card.get('sticker_visual_emotion', ''),
        ],
        'irony_strength': [
            entry.subtle_cue_card.get('visual_irony_strength', ''),
            entry.sticker_card.get('sticker_irony_strength', ''),
            entry.caption_card.get('caption_irony_strength', ''),
            entry.sticker_card.get('sticker_humor_style', ''),
        ],
        'social_stance': [
            entry.subtle_cue_card.get('visual_social_stance', ''),
            entry.caption_card.get('caption_social_stance', ''),
            entry.sticker_card.get('fused_pragmatic_meaning', ''),
        ],
        'conversation_role': [
            entry.subtle_cue_card.get('visual_conversation_role', ''),
            entry.sticker_card.get('sticker_conversation_role', ''),
            entry.caption_card.get('caption_conversation_role', ''),
        ],
        'relationship_fit': [
            entry.subtle_cue_card.get('visual_relationship_fit', ''),
            entry.sticker_card.get('sticker_relationship_fit', ''),
            entry.caption_card.get('caption_relationship_fit', ''),
        ],
    }


def _visual_axis_sources(entry: StickerIndexEntry) -> dict[str, list[str]]:
    return {
        'eye_signal': [
            entry.subtle_cue_card.get('eye_signal', ''),
            entry.subtle_cue_card.get('dominant_signal', ''),
            entry.sticker_card.get('sticker_visual_emotion', ''),
        ],
        'mouth_signal': [
            entry.subtle_cue_card.get('mouth_signal', ''),
            entry.subtle_cue_card.get('micro_expression', ''),
            entry.subtle_cue_card.get('dominant_signal', ''),
        ],
        'motion_signal': [
            entry.subtle_cue_card.get('motion_signal', ''),
            entry.subtle_cue_card.get('pose_signal', ''),
            entry.subtle_cue_card.get('head_signal', ''),
            entry.subtle_cue_card.get('hand_signal', ''),
        ],
        'delivery_style': [
            entry.sticker_card.get('sticker_delivery_style', ''),
            entry.sticker_card.get('sticker_general_usefulness', ''),
            entry.subtle_cue_card.get('dominant_signal', ''),
        ],
        'humor_style': [
            entry.sticker_card.get('sticker_humor_style', ''),
            entry.sticker_card.get('fused_pragmatic_meaning', ''),
            entry.style_card.get('meme_intensity', ''),
        ],
    }


def _score_requested_field(
    request_text: str,
    source_values: list[str],
    *,
    full_match_score: float,
    partial_base: float,
    partial_scale: float,
    miss_penalty: float,
) -> tuple[float, dict[str, Any]]:
    source_text = _join_nonempty(source_values)
    requested = str(request_text or '').strip()
    if not requested:
        return 0.0, {'requested': '', 'matched_terms': [], 'phrase_hit': False}
    request_terms = set(_tokenize(requested))
    source_terms = set(_tokenize(source_text))
    phrase_hit = requested.lower() in source_text.lower() if source_text else False
    overlap = sorted(request_terms & source_terms)
    if phrase_hit:
        score = full_match_score
    elif overlap:
        coverage = len(overlap) / max(1, min(len(request_terms), 4))
        score = partial_base + (partial_scale * coverage)
    else:
        score = miss_penalty
    return score, {'requested': requested, 'matched_terms': overlap[:12], 'phrase_hit': phrase_hit}


def _entry_caption_text(entry: StickerIndexEntry) -> str:
    return _join_nonempty([
        entry.source_overlay_text_normalized,
        entry.caption_meaning_en,
        entry.caption_meaning_zh,
        entry.caption_semantic_text,
        entry.caption_card.get('caption_literal_meaning', ''),
        entry.caption_card.get('caption_pragmatic_meaning', ''),
        entry.caption_card.get('caption_close_aliases', ''),
    ])


def _entry_style_text(entry: StickerIndexEntry) -> str:
    return _join_nonempty([
        entry.style_text,
        entry.style_card.get('style_rendering_type', ''),
        entry.style_card.get('style_palette_family', ''),
        entry.style_card.get('style_character_family', ''),
        entry.style_card.get('style_text_prominence', ''),
        entry.style_card.get('line_weight', ''),
        entry.style_card.get('meme_intensity', ''),
    ])


def _build_style_cluster_profiles(entries: Any) -> dict[str, dict[str, Any]]:
    grouped_tokens: dict[str, Counter[str]] = defaultdict(Counter)
    grouped_counts: Counter[str] = Counter()
    for entry in entries:
        cluster = str(getattr(entry, 'style_cluster', '') or '').strip()
        if not cluster:
            continue
        grouped_counts[cluster] += 1
        grouped_tokens[cluster].update(_style_profile_terms(entry))
    profiles: dict[str, dict[str, Any]] = {}
    for cluster, token_counts in grouped_tokens.items():
        top_terms = [term for term, _ in token_counts.most_common(12)]
        profiles[cluster] = {
            'top_terms': top_terms,
            'style_signature': ' '.join(top_terms),
            'entry_count': grouped_counts[cluster],
        }
    return profiles


def _cluster_profile_similarity(
    *,
    requested_profile: dict[str, Any],
    candidate_profile: dict[str, Any],
) -> tuple[float, list[str]]:
    requested_terms = set(requested_profile.get('top_terms', []))
    candidate_terms = set(candidate_profile.get('top_terms', []))
    overlap = sorted(requested_terms & candidate_terms)
    coverage = len(overlap) / max(1, min(len(requested_terms), len(candidate_terms), 6))
    requested_signature = str(requested_profile.get('style_signature', '') or '')
    candidate_signature = str(candidate_profile.get('style_signature', '') or '')
    signature_similarity = (
        SequenceMatcher(None, requested_signature, candidate_signature).ratio()
        if requested_signature and candidate_signature
        else 0.0
    )
    return max(coverage, signature_similarity * 0.9), overlap


def _normalize_pack_text(value: str) -> str:
    return ' '.join(_pack_terms(value))


def _pack_terms(value: str) -> list[str]:
    return [token for token in _tokenize(str(value or '').replace('_', ' ').replace('-', ' ')) if len(token) > 2 and token not in _PACK_TOKEN_STOPWORDS]


def _cluster_name_terms(value: str) -> list[str]:
    return [token for token in _tokenize(str(value or '').replace('_', ' ').replace('-', ' ')) if len(token) > 2]


def _style_profile_terms(entry: StickerIndexEntry) -> list[str]:
    return [token for token in _tokenize(_entry_style_text(entry).replace('_', ' ').replace('-', ' ')) if len(token) > 2 and token not in _STYLE_TOKEN_STOPWORDS]


def _join_nonempty(values: list[str] | tuple[str, ...]) -> str:
    return ' '.join(str(value or '').strip() for value in values if str(value or '').strip())


def _tokenize(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        chunks = [str(v) for v in value if str(v).strip()]
        text = ' '.join(chunks)
    else:
        text = str(value or '')
    tokens: list[str] = []
    for match in _WORD_RE.finditer(text.lower()):
        tokens.append(match.group(0))
    cjk_chars = [char for char in text if _CJK_RE.match(char)]
    if cjk_chars:
        tokens.extend(cjk_chars)
        if len(cjk_chars) >= 2:
            tokens.extend(''.join(cjk_chars[i:i + 2]) for i in range(len(cjk_chars) - 1))
    return [token for token in tokens if token]
