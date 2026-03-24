from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Iterable

_STYLE_GOAL_ALIASES = {
    'continue': 'keep_current',
    'neutral': 'allow_switch',
    'prefer_switch': 'prefer_switch',
    'hard_switch': 'ignore_style',
    'keep_current': 'keep_current',
    'allow_switch': 'allow_switch',
    'ignore_style': 'ignore_style',
}


def normalize_style_goal(value: str | None) -> str:
    normalized = str(value or '').strip().lower()
    return _STYLE_GOAL_ALIASES.get(normalized, 'keep_current')


@dataclass(slots=True)
class SessionStyleState:
    session_id: str
    style_cluster: str | None = None
    source_pack_id: str | None = None
    style_confidence: float = 0.0
    recent_sticker_ids: Deque[str] = field(default_factory=lambda: deque(maxlen=12))
    recent_source_pack_ids: Deque[str] = field(default_factory=lambda: deque(maxlen=8))
    recent_style_clusters: Deque[str] = field(default_factory=lambda: deque(maxlen=8))
    last_style_switch_turn: int = 0
    turns: int = 0

    def style_bonus(
        self,
        *,
        candidate_style_cluster: str | None,
        candidate_source_pack_id: str | None,
        style_goal: str = 'keep_current',
        style_policy: str | None = None,
    ) -> float:
        goal = normalize_style_goal(style_policy or style_goal)
        if goal == 'ignore_style':
            return 0.0
        bonus = 0.0
        same_cluster = bool(candidate_style_cluster and self.style_cluster and candidate_style_cluster == self.style_cluster)
        same_pack = bool(candidate_source_pack_id and self.source_pack_id and candidate_source_pack_id == self.source_pack_id)
        seen_recent_cluster = bool(candidate_style_cluster and candidate_style_cluster in self.recent_style_clusters)
        if goal == 'prefer_switch':
            if same_cluster:
                bonus -= 0.55
            elif self.style_cluster and candidate_style_cluster:
                bonus += 0.28
            if same_pack:
                bonus -= 0.18
            elif self.source_pack_id and candidate_source_pack_id:
                bonus += 0.08
            return bonus
        if same_cluster:
            bonus += 1.35 if goal == 'keep_current' else 0.70
        if same_pack:
            bonus += 0.40 if goal == 'keep_current' else 0.18
        if seen_recent_cluster:
            bonus += 0.15 if goal == 'keep_current' else 0.10
        return bonus

    def repeat_penalty(self, *, sticker_id: str, source_pack_id: str | None, style_cluster: str | None) -> float:
        penalty = 0.0
        if sticker_id in self.recent_sticker_ids:
            penalty += 4.0 if (self.recent_sticker_ids and self.recent_sticker_ids[-1] == sticker_id) else 2.0
        if source_pack_id and source_pack_id in self.recent_source_pack_ids:
            penalty += 0.45 * sum(1 for value in self.recent_source_pack_ids if value == source_pack_id)
        if style_cluster and style_cluster in self.recent_style_clusters:
            penalty += 0.12 * sum(1 for value in self.recent_style_clusters if value == style_cluster)
        return penalty

    def should_allow_switch(
        self,
        *,
        current_score: float,
        candidate_score: float,
        style_goal: str = 'keep_current',
        style_policy: str | None = None,
    ) -> bool:
        goal = normalize_style_goal(style_policy or style_goal)
        if goal in {'prefer_switch', 'ignore_style'}:
            return True
        if self.style_cluster is None:
            return True
        threshold = 0.8 if goal == 'keep_current' else 0.45
        return (candidate_score - current_score) >= threshold

    def relation(
        self,
        *,
        candidate_style_cluster: str | None,
        candidate_source_pack_id: str | None,
        style_goal: str = 'keep_current',
        style_policy: str | None = None,
    ) -> dict[str, Any]:
        goal = normalize_style_goal(style_policy or style_goal)
        same_cluster = bool(candidate_style_cluster and self.style_cluster and candidate_style_cluster == self.style_cluster)
        same_pack = bool(candidate_source_pack_id and self.source_pack_id and candidate_source_pack_id == self.source_pack_id)
        recent_cluster = bool(candidate_style_cluster and candidate_style_cluster in self.recent_style_clusters)
        if same_cluster and same_pack:
            relation_label = 'same_cluster_same_pack'
        elif same_cluster:
            relation_label = 'same_cluster'
        elif same_pack:
            relation_label = 'same_pack'
        elif self.style_cluster and candidate_style_cluster:
            relation_label = 'style_switch'
        else:
            relation_label = 'no_anchor'
        return {
            'style_goal': goal,
            'current_style_cluster': self.style_cluster,
            'current_source_pack_id': self.source_pack_id,
            'candidate_style_cluster': candidate_style_cluster,
            'candidate_source_pack_id': candidate_source_pack_id,
            'same_cluster': same_cluster,
            'same_pack': same_pack,
            'recent_cluster': recent_cluster,
            'relation_label': relation_label,
        }

    def history_summary(self) -> dict[str, Any]:
        recent_stickers = list(self.recent_sticker_ids)[-5:]
        recent_packs = list(self.recent_source_pack_ids)[-4:]
        recent_clusters = list(self.recent_style_clusters)[-4:]
        cluster_counts = Counter(self.recent_style_clusters)
        summary_bits: list[str] = []
        if self.style_cluster:
            summary_bits.append(f'current cluster={self.style_cluster}')
        if self.source_pack_id:
            summary_bits.append(f'current pack={self.source_pack_id}')
        if recent_clusters:
            summary_bits.append('recent clusters=' + ', '.join(recent_clusters))
        if recent_stickers:
            summary_bits.append('recent stickers=' + ', '.join(recent_stickers))
        return {
            'recent_sticker_ids': recent_stickers,
            'recent_source_pack_ids': recent_packs,
            'recent_style_clusters': recent_clusters,
            'style_cluster_frequency': dict(cluster_counts),
            'last_style_switch_turn': self.last_style_switch_turn,
            'turns': self.turns,
            'summary': '; '.join(summary_bits) if summary_bits else 'No prior sticker style history for this session.',
        }

    def to_context_dict(self) -> dict[str, Any]:
        return {
            'session_id': self.session_id,
            'current_style_cluster': self.style_cluster,
            'current_source_pack_id': self.source_pack_id,
            'style_confidence': round(self.style_confidence, 4),
            'history': self.history_summary(),
        }

    def record_selection(self, *, sticker_id: str, source_pack_id: str | None, style_cluster: str | None) -> None:
        self.turns += 1
        self.recent_sticker_ids.append(sticker_id)
        if source_pack_id:
            self.recent_source_pack_ids.append(source_pack_id)
        if style_cluster:
            self.recent_style_clusters.append(style_cluster)
        if style_cluster and style_cluster != self.style_cluster:
            self.last_style_switch_turn = self.turns
        if style_cluster:
            self.style_cluster = style_cluster
            self.style_confidence = min(1.0, self.style_confidence + 0.2)
        if source_pack_id:
            self.source_pack_id = source_pack_id


class SessionStyleMemory:
    def __init__(self) -> None:
        self._states: dict[str, SessionStyleState] = {}

    def get(self, session_id: str) -> SessionStyleState:
        session_id = session_id or 'default'
        state = self._states.get(session_id)
        if state is None:
            state = SessionStyleState(session_id=session_id)
            self._states[session_id] = state
        return state

    def preload(
        self,
        session_id: str,
        *,
        recent_sticker_ids: Iterable[str] = (),
        recent_source_pack_ids: Iterable[str] = (),
        recent_style_clusters: Iterable[str] = (),
    ) -> SessionStyleState:
        state = self.get(session_id)
        for sticker_id in recent_sticker_ids:
            state.recent_sticker_ids.append(str(sticker_id))
        for source_pack_id in recent_source_pack_ids:
            if source_pack_id:
                state.recent_source_pack_ids.append(str(source_pack_id))
                state.source_pack_id = str(source_pack_id)
        for cluster in recent_style_clusters:
            if cluster:
                normalized = str(cluster)
                state.recent_style_clusters.append(normalized)
                state.style_cluster = normalized
        if state.recent_style_clusters:
            state.style_confidence = min(1.0, 0.2 * len(state.recent_style_clusters))
        state.turns = max(state.turns, len(state.recent_sticker_ids))
        return state
