from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Iterable

from tgchatbot.stickers.persona import build_persona_dict, compact_persona_dict, merge_persona_dicts, persona_has_values

_STYLE_GOAL_ALIASES = {
    'continue': 'preserve',
    'neutral': 'allow_switch',
    'prefer_switch': 'prefer_switch',
    'hard_switch': 'ignore_style',
    'preserve': 'preserve',
    'keep_current': 'preserve',
    'allow_switch': 'allow_switch',
    'ignore_style': 'ignore_style',
}


def normalize_style_goal(value: str | None) -> str:
    normalized = str(value or '').strip().lower()
    return _STYLE_GOAL_ALIASES.get(normalized, 'preserve')


@dataclass(slots=True)
class SessionStyleState:
    session_id: str
    style_cluster: str | None = None
    source_pack_id: str | None = None
    style_confidence: float = 0.0
    session_persona: dict[str, Any] | None = None
    session_persona_loaded: bool = False
    recent_sticker_ids: Deque[str] = field(default_factory=lambda: deque(maxlen=12))
    recent_source_pack_ids: Deque[str] = field(default_factory=lambda: deque(maxlen=8))
    recent_style_clusters: Deque[str] = field(default_factory=lambda: deque(maxlen=8))
    recent_shortlist_sticker_ids: Deque[str] = field(default_factory=lambda: deque(maxlen=18))
    recent_query_shortlists: Deque[tuple[str, ...]] = field(default_factory=lambda: deque(maxlen=3))
    recent_query_bundles: Deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=4))
    recent_selected_persona_snapshots: Deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=6))
    recent_shortlist_persona_snapshots: Deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=12))
    last_style_switch_turn: int = 0
    turns: int = 0

    def style_bonus(
        self,
        *,
        candidate_style_cluster: str | None,
        candidate_source_pack_id: str | None,
        style_goal: str = 'preserve',
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
            bonus += 1.35 if goal == 'preserve' else 0.70
        if same_pack:
            bonus += 0.40 if goal == 'preserve' else 0.18
        if seen_recent_cluster:
            bonus += 0.15 if goal == 'preserve' else 0.10
        return bonus

    def repeat_penalty(
        self,
        *,
        sticker_id: str,
        source_pack_id: str | None,
        style_cluster: str | None,
        diversity_preference: str = 'default',
    ) -> dict[str, Any]:
        sent_count = sum(1 for value in self.recent_sticker_ids if value == sticker_id)
        shortlist_count = sum(1 for value in self.recent_shortlist_sticker_ids if value == sticker_id)
        same_pack_count = sum(1 for value in self.recent_source_pack_ids if source_pack_id and value == source_pack_id)
        same_cluster_count = sum(1 for value in self.recent_style_clusters if style_cluster and value == style_cluster)

        penalty = 0.0
        labels: list[str] = []
        if diversity_preference == 'prefer_fresh_variant':
            if sent_count:
                penalty += 0.48 if (self.recent_sticker_ids and self.recent_sticker_ids[-1] == sticker_id) else 0.26
                labels.append('recently_sent_exact_sticker')
            elif shortlist_count:
                penalty += 0.18
                labels.append('recently_surfaced_exact_sticker')
            if same_pack_count:
                penalty += min(0.18, 0.06 * same_pack_count)
                labels.append('recent_pack_variant')
            if same_cluster_count:
                penalty += min(0.12, 0.04 * same_cluster_count)
                labels.append('recent_cluster_variant')
        else:
            if self.recent_sticker_ids and self.recent_sticker_ids[-1] == sticker_id:
                penalty += 0.22
                labels.append('immediate_repeat')

        return {
            'penalty': round(penalty, 4),
            'mode': diversity_preference,
            'labels': labels,
            'recent_sent_count': sent_count,
            'recent_shortlist_count': shortlist_count,
            'same_pack_recent_count': same_pack_count,
            'same_cluster_recent_count': same_cluster_count,
        }

    def should_allow_switch(
        self,
        *,
        current_score: float,
        candidate_score: float,
        style_goal: str = 'preserve',
        style_policy: str | None = None,
    ) -> bool:
        goal = normalize_style_goal(style_policy or style_goal)
        if goal in {'prefer_switch', 'ignore_style'}:
            return True
        if self.style_cluster is None:
            return True
        threshold = 0.8 if goal == 'preserve' else 0.45
        return (candidate_score - current_score) >= threshold

    def relation(
        self,
        *,
        candidate_style_cluster: str | None,
        candidate_source_pack_id: str | None,
        style_goal: str = 'preserve',
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
        recent_shortlist = list(self.recent_shortlist_sticker_ids)[-8:]
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
        if recent_shortlist:
            summary_bits.append('recent shortlist=' + ', '.join(recent_shortlist))
        return {
            'recent_sticker_ids': recent_stickers,
            'recent_source_pack_ids': recent_packs,
            'recent_style_clusters': recent_clusters,
            'recent_shortlist_sticker_ids': recent_shortlist,
            'style_cluster_frequency': dict(cluster_counts),
            'last_style_switch_turn': self.last_style_switch_turn,
            'turns': self.turns,
            'summary': '; '.join(summary_bits) if summary_bits else 'No prior sticker style history for this session.',
        }

    def set_session_persona(self, persona: dict[str, Any] | None) -> None:
        compact = compact_persona_dict(persona)
        self.session_persona = compact or None
        self.session_persona_loaded = True

    def clear_session_persona(self) -> None:
        self.session_persona = None
        self.session_persona_loaded = True

    def recent_implicit_persona(self) -> dict[str, Any]:
        visual_scores: dict[str, Counter[str]] = {
            'character_archetype': Counter(),
            'rendering_style': Counter(),
            'palette_mood': Counter(),
            'prefer_pack': Counter(),
            'prefer_cluster': Counter(),
            'style_hints': Counter(),
        }
        affect_scores: dict[str, Counter[str]] = {
            'default_tone': Counter(),
            'expression_bias': Counter(),
            'pose_bias': Counter(),
            'delivery_bias': Counter(),
            'humor_bias': Counter(),
        }

        def apply_snapshot(snapshot: dict[str, Any], weight: float) -> None:
            persona = compact_persona_dict(snapshot)
            visual = dict(persona.get('visual_identity') or {})
            affect = dict(persona.get('affect_profile') or {})
            for field_name, counter in visual_scores.items():
                if field_name == 'style_hints':
                    for hint in list(visual.get('style_hints') or []):
                        if hint:
                            counter[hint] += weight
                    continue
                value = str(visual.get(field_name) or '').strip()
                if value:
                    counter[value] += weight
            for field_name, counter in affect_scores.items():
                value = str(affect.get(field_name) or '').strip()
                if value:
                    counter[value] += weight

        for snapshot in self.recent_selected_persona_snapshots:
            apply_snapshot(snapshot, 1.0)
        for snapshot in self.recent_shortlist_persona_snapshots:
            apply_snapshot(snapshot, 0.35)

        if self.source_pack_id:
            visual_scores['prefer_pack'][self.source_pack_id] += 0.25
        if self.style_cluster:
            visual_scores['prefer_cluster'][self.style_cluster] += 0.25

        def top_value(counter: Counter[str]) -> str:
            if not counter:
                return ''
            return counter.most_common(1)[0][0]

        style_hints = [value for value, _ in visual_scores['style_hints'].most_common(4)]
        persona = build_persona_dict(
            visual_identity={
                'character_archetype': top_value(visual_scores['character_archetype']),
                'rendering_style': top_value(visual_scores['rendering_style']),
                'palette_mood': top_value(visual_scores['palette_mood']),
                'prefer_pack': top_value(visual_scores['prefer_pack']),
                'prefer_cluster': top_value(visual_scores['prefer_cluster']),
                'style_hints': style_hints,
            },
            affect_profile={
                'default_tone': top_value(affect_scores['default_tone']),
                'expression_bias': top_value(affect_scores['expression_bias']),
                'pose_bias': top_value(affect_scores['pose_bias']),
                'delivery_bias': top_value(affect_scores['delivery_bias']),
                'humor_bias': top_value(affect_scores['humor_bias']),
            },
        )
        confidence = min(
            0.72,
            (0.14 * len(self.recent_selected_persona_snapshots))
            + (0.04 * len(self.recent_shortlist_persona_snapshots))
            + (0.06 if self.source_pack_id or self.style_cluster else 0.0),
        )
        summary_bits: list[str] = []
        visual = dict(persona.get('visual_identity') or {})
        affect = dict(persona.get('affect_profile') or {})
        if visual.get('character_archetype'):
            summary_bits.append(f"recent character={visual['character_archetype']}")
        if visual.get('rendering_style'):
            summary_bits.append(f"recent render={visual['rendering_style']}")
        if affect.get('default_tone'):
            summary_bits.append(f"recent tone={affect['default_tone']}")
        if affect.get('expression_bias'):
            summary_bits.append(f"recent expression={affect['expression_bias']}")
        return {
            **persona,
            'confidence': round(confidence, 4),
            'summary': '; '.join(summary_bits) if summary_bits else 'No recent implicit sticker persona.',
        }

    def persona_context(self, *, effective_persona: dict[str, Any] | None = None, feedback_summary: str = '') -> dict[str, Any]:
        session_persona = compact_persona_dict(self.session_persona)
        recent_implicit = self.recent_implicit_persona()
        effective = compact_persona_dict(effective_persona)
        confidence = 0.0
        if effective:
            confidence = 0.9 if session_persona else float(recent_implicit.get('confidence', 0.0) or 0.0)
        return {
            'session_persona': session_persona or None,
            'recent_implicit_persona': recent_implicit if persona_has_values(recent_implicit) else None,
            'effective_persona': effective or None,
            'confidence': round(confidence, 4),
            'feedback_summary': feedback_summary or (
                'Using stored session sticker persona.'
                if session_persona
                else 'No stored persona; slight recent sticker continuity is available.'
                if persona_has_values(recent_implicit)
                else 'No stored or recent sticker persona.'
            ),
        }

    def to_context_dict(self) -> dict[str, Any]:
        return {
            'session_id': self.session_id,
            'current_style_cluster': self.style_cluster,
            'current_source_pack_id': self.source_pack_id,
            'style_confidence': round(self.style_confidence, 4),
            'history': self.history_summary(),
        }

    def record_selection(self, *, sticker_id: str, source_pack_id: str | None, style_cluster: str | None, persona_snapshot: dict[str, Any] | None = None) -> None:
        self.turns += 1
        self.recent_sticker_ids.append(sticker_id)
        if source_pack_id:
            self.recent_source_pack_ids.append(source_pack_id)
        if style_cluster:
            self.recent_style_clusters.append(style_cluster)
        if persona_has_values(persona_snapshot):
            self.recent_selected_persona_snapshots.append(compact_persona_dict(persona_snapshot))
        if style_cluster and style_cluster != self.style_cluster:
            self.last_style_switch_turn = self.turns
        if style_cluster:
            self.style_cluster = style_cluster
            self.style_confidence = min(1.0, self.style_confidence + 0.2)
        if source_pack_id:
            self.source_pack_id = source_pack_id

    def record_query(self, *, query_bundle: dict[str, Any], shortlist_ids: Iterable[str], shortlist_persona_snapshots: Iterable[dict[str, Any]] = ()) -> None:
        shortlist = tuple(str(sticker_id) for sticker_id in shortlist_ids if str(sticker_id).strip())
        self.recent_query_bundles.append(dict(query_bundle))
        if shortlist:
            self.recent_query_shortlists.append(shortlist)
            for sticker_id in shortlist:
                self.recent_shortlist_sticker_ids.append(sticker_id)
        for snapshot in shortlist_persona_snapshots:
            if persona_has_values(snapshot):
                self.recent_shortlist_persona_snapshots.append(compact_persona_dict(snapshot))

    def previous_query_bundle(self) -> dict[str, Any] | None:
        return dict(self.recent_query_bundles[-1]) if self.recent_query_bundles else None

    def query_delta(self, current_query_bundle: dict[str, Any]) -> dict[str, Any]:
        previous = self.previous_query_bundle()
        current = dict(current_query_bundle)
        if not previous:
            return {'previous': None, 'current': current, 'changed_fields': [], 'changed_values': {}}
        changed_fields: list[str] = []
        changed_values: dict[str, str] = {}
        for field_name, current_value in current.items():
            previous_value = previous.get(field_name)
            current_text = str(current_value or '').strip()
            previous_text = str(previous_value or '').strip()
            if current_text and current_text != previous_text:
                changed_fields.append(field_name)
                changed_values[field_name] = current_text
        return {
            'previous': previous,
            'current': current,
            'changed_fields': changed_fields,
            'changed_values': changed_values,
        }


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
        session_persona: dict[str, Any] | None = None,
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
        if persona_has_values(session_persona):
            state.set_session_persona(session_persona)
        state.turns = max(state.turns, len(state.recent_sticker_ids))
        return state
