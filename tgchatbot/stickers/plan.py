from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_TEXT_PRIORITIES = {'require', 'prefer', 'ignore'}
_STYLE_GOALS = {'keep_current', 'allow_switch', 'prefer_switch', 'ignore_style'}
_LEGACY_STYLE_POLICY_TO_GOAL = {
    'continue': 'keep_current',
    'neutral': 'allow_switch',
    'prefer_switch': 'prefer_switch',
    'hard_switch': 'ignore_style',
}
_STYLE_GOAL_TO_LEGACY_POLICY = {value: key for key, value in _LEGACY_STYLE_POLICY_TO_GOAL.items()}


def _norm_text(value: Any) -> str:
    return ' '.join(str(value or '').replace('\n', ' ').replace('\t', ' ').split()).strip()


def _norm_text_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    return [normalized for value in values if (normalized := _norm_text(value))]


def _norm_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _norm_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    text = _norm_text(value).lower()
    if text in {'1', 'true', 'yes', 'on'}:
        return True
    if text in {'0', 'false', 'no', 'off'}:
        return False
    return default


def _request_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for ch in text.lower():
        if ch.isalnum() or ch in "_+-'":
            current.append(ch)
            continue
        if current:
            tokens.append(''.join(current))
            current = []
        if '\u3040' <= ch <= '\u30ff' or '\u3400' <= ch <= '\u9fff' or '\uf900' <= ch <= '\ufaff' or '\uac00' <= ch <= '\ud7af':
            tokens.append(ch)
    if current:
        tokens.append(''.join(current))
    return [token for token in tokens if token]


@dataclass(slots=True)
class SemanticFocus:
    reaction_type: str = ''
    reply_force: str = ''
    emotional_valence: str = ''
    irony_strength: str = ''
    social_stance: str = ''
    conversation_role: str = ''
    relationship_fit: str = ''

    @classmethod
    def from_payload(cls, payload: Any) -> 'SemanticFocus':
        data = _norm_mapping(payload)
        return cls(
            reaction_type=_norm_text(data.get('reaction_type', '')),
            reply_force=_norm_text(data.get('reply_force', '')),
            emotional_valence=_norm_text(data.get('emotional_valence', '')),
            irony_strength=_norm_text(data.get('irony_strength', '')),
            social_stance=_norm_text(data.get('social_stance', '')),
            conversation_role=_norm_text(data.get('conversation_role', '')),
            relationship_fit=_norm_text(data.get('relationship_fit', '')),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            'reaction_type': self.reaction_type,
            'reply_force': self.reply_force,
            'emotional_valence': self.emotional_valence,
            'irony_strength': self.irony_strength,
            'social_stance': self.social_stance,
            'conversation_role': self.conversation_role,
            'relationship_fit': self.relationship_fit,
        }

    def active_fields(self) -> dict[str, str]:
        return {key: value for key, value in self.as_dict().items() if value}

    def request_texts(self) -> list[str]:
        return list(self.active_fields().values())


@dataclass(slots=True)
class VisualFocus:
    eye_signal: str = ''
    mouth_signal: str = ''
    motion_signal: str = ''
    delivery_style: str = ''
    humor_style: str = ''

    @classmethod
    def from_payload(cls, payload: Any) -> 'VisualFocus':
        data = _norm_mapping(payload)
        return cls(
            eye_signal=_norm_text(data.get('eye_signal', '')),
            mouth_signal=_norm_text(data.get('mouth_signal', '')),
            motion_signal=_norm_text(data.get('motion_signal', '')),
            delivery_style=_norm_text(data.get('delivery_style', '')),
            humor_style=_norm_text(data.get('humor_style', '')),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            'eye_signal': self.eye_signal,
            'mouth_signal': self.mouth_signal,
            'motion_signal': self.motion_signal,
            'delivery_style': self.delivery_style,
            'humor_style': self.humor_style,
        }

    def active_fields(self) -> dict[str, str]:
        return {key: value for key, value in self.as_dict().items() if value}

    def request_texts(self) -> list[str]:
        return list(self.active_fields().values())


@dataclass(slots=True)
class StyleFocus:
    style_goal: str = 'keep_current'
    style_hints: list[str] = field(default_factory=list)
    prefer_pack: str = ''
    prefer_cluster: str = ''

    @classmethod
    def from_payload(cls, payload: Any, *, legacy_style_policy: str = '') -> 'StyleFocus':
        data = _norm_mapping(payload)
        raw_style_goal = _norm_text(data.get('style_goal', '')).lower()
        if raw_style_goal not in _STYLE_GOALS:
            raw_style_goal = _LEGACY_STYLE_POLICY_TO_GOAL.get(legacy_style_policy, 'keep_current')
        return cls(
            style_goal=raw_style_goal,
            style_hints=_norm_text_list(data.get('style_hints')),
            prefer_pack=_norm_text(data.get('prefer_pack', '')),
            prefer_cluster=_norm_text(data.get('prefer_cluster', '')),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            'style_goal': self.style_goal,
            'style_hints': list(self.style_hints),
            'prefer_pack': self.prefer_pack,
            'prefer_cluster': self.prefer_cluster,
        }


@dataclass(slots=True)
class TextConstraints:
    text_priority: str = 'prefer'
    must_include: list[str] = field(default_factory=list)
    avoid_text_meanings: list[str] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: Any, *, legacy_text_priority: str = '') -> 'TextConstraints':
        data = _norm_mapping(payload)
        text_priority = _norm_text(data.get('text_priority', legacy_text_priority or 'prefer')).lower() or 'prefer'
        if text_priority not in _TEXT_PRIORITIES:
            text_priority = 'prefer'
        return cls(
            text_priority=text_priority,
            must_include=_norm_text_list(data.get('must_include')),
            avoid_text_meanings=_norm_text_list(data.get('avoid_text_meanings')),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            'text_priority': self.text_priority,
            'must_include': list(self.must_include),
            'avoid_text_meanings': list(self.avoid_text_meanings),
        }


@dataclass(slots=True)
class IntensityLimits:
    max_harshness: int = 3
    max_intimacy: int = 4
    max_meme_dependence: int = 4
    allow_animation: bool = False

    @classmethod
    def from_payload(cls, payload: Any, *, legacy_payload: dict[str, Any]) -> 'IntensityLimits':
        data = _norm_mapping(payload)
        return cls(
            max_harshness=_bounded_int(data.get('max_harshness', legacy_payload.get('max_harshness', 3)), default=3, minimum=0, maximum=4),
            max_intimacy=_bounded_int(data.get('max_intimacy', legacy_payload.get('max_intimacy', 4)), default=4, minimum=0, maximum=4),
            max_meme_dependence=_bounded_int(data.get('max_meme_dependence', legacy_payload.get('max_meme_dependence', 4)), default=4, minimum=0, maximum=4),
            allow_animation=_norm_bool(data.get('allow_animation', legacy_payload.get('allow_animation', False)), default=False),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            'max_harshness': self.max_harshness,
            'max_intimacy': self.max_intimacy,
            'max_meme_dependence': self.max_meme_dependence,
            'allow_animation': self.allow_animation,
        }


@dataclass(slots=True)
class StickerRetrievalPlan:
    intent_core: str
    secondary_goals: list[str] = field(default_factory=list)
    semantic_focus: SemanticFocus = field(default_factory=SemanticFocus)
    visual_focus: VisualFocus = field(default_factory=VisualFocus)
    style_focus: StyleFocus = field(default_factory=StyleFocus)
    text_constraints: TextConstraints = field(default_factory=TextConstraints)
    intensity_limits: IntensityLimits = field(default_factory=IntensityLimits)
    forbid: list[str] = field(default_factory=list)
    candidate_budget: int = 5
    send: bool = True
    deprecated_aliases_used: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> 'StickerRetrievalPlan':
        data = _norm_mapping(payload)
        intent_core = _norm_text(data.get('intent_core', ''))
        if not intent_core:
            raise ValueError('intent_core is required')
        secondary_goals = _norm_text_list(data.get('secondary_goals'))
        forbid = _norm_text_list(data.get('forbid'))
        legacy_text_priority = _norm_text(data.get('text_priority', '')).lower()
        legacy_style_policy = _norm_text(data.get('style_policy', '')).lower()
        deprecated_aliases_used: dict[str, Any] = {}
        for key in ('text_priority', 'max_harshness', 'max_intimacy', 'allow_animation', 'forbid', 'style_policy', 'safety_limits'):
            if key in data:
                deprecated_aliases_used[key] = data.get(key)
        intensity_payload = data.get('intensity_limits') if 'intensity_limits' in data else data.get('safety_limits')
        return cls(
            intent_core=intent_core,
            secondary_goals=secondary_goals,
            semantic_focus=SemanticFocus.from_payload(data.get('semantic_focus')),
            visual_focus=VisualFocus.from_payload(data.get('visual_focus')),
            style_focus=StyleFocus.from_payload(data.get('style_focus'), legacy_style_policy=legacy_style_policy),
            text_constraints=TextConstraints.from_payload(data.get('text_constraints'), legacy_text_priority=legacy_text_priority),
            intensity_limits=IntensityLimits.from_payload(intensity_payload, legacy_payload=data),
            forbid=forbid,
            candidate_budget=_bounded_int(data.get('candidate_budget', 5), default=5, minimum=1, maximum=8),
            send=_norm_bool(data.get('send', True), default=True),
            deprecated_aliases_used=deprecated_aliases_used,
        )

    @property
    def text_priority(self) -> str:
        return self.text_constraints.text_priority

    @property
    def max_harshness(self) -> int:
        return self.intensity_limits.max_harshness

    @property
    def max_intimacy(self) -> int:
        return self.intensity_limits.max_intimacy

    @property
    def max_meme_dependence(self) -> int:
        return self.intensity_limits.max_meme_dependence

    @property
    def allow_animation(self) -> bool:
        return self.intensity_limits.allow_animation

    @property
    def style_goal(self) -> str:
        return self.style_focus.style_goal

    @property
    def style_hints(self) -> list[str]:
        return list(self.style_focus.style_hints)

    @property
    def prefer_pack(self) -> str:
        return self.style_focus.prefer_pack

    @property
    def prefer_cluster(self) -> str:
        return self.style_focus.prefer_cluster

    @property
    def style_policy(self) -> str:
        return _STYLE_GOAL_TO_LEGACY_POLICY.get(self.style_goal, 'continue')

    def caption_query_text(self) -> str:
        parts = [self.intent_core, *self.secondary_goals, *self.semantic_focus.request_texts()]
        if self.text_priority == 'require':
            parts.append('visible caption text should dominate meaning')
        elif self.text_priority == 'prefer':
            parts.append('caption meaning should matter strongly')
        if self.text_constraints.must_include:
            parts.append('caption should include or imply: ' + '; '.join(self.text_constraints.must_include))
        if self.text_constraints.avoid_text_meanings:
            parts.append('avoid caption meaning: ' + '; '.join(self.text_constraints.avoid_text_meanings))
        if self.forbid:
            parts.append('avoid: ' + '; '.join(self.forbid))
        return '; '.join(part for part in parts if part)

    def sticker_query_text(self) -> str:
        parts = [
            self.intent_core,
            *self.secondary_goals,
            *self.semantic_focus.request_texts(),
            *self.visual_focus.request_texts(),
        ]
        if self.style_hints:
            parts.append('style hints: ' + '; '.join(self.style_hints))
        if self.forbid:
            parts.append('avoid: ' + '; '.join(self.forbid))
        return '; '.join(part for part in parts if part)

    def request_terms(self) -> list[str]:
        text = ' ; '.join([
            self.intent_core,
            *self.secondary_goals,
            *self.semantic_focus.request_texts(),
            *self.visual_focus.request_texts(),
        ])
        return _request_tokens(text)

    def style_request_terms(self) -> list[str]:
        return _request_tokens(' ; '.join(self.style_hints))

    def must_include_terms(self) -> list[str]:
        return _request_tokens(' ; '.join(self.text_constraints.must_include))

    def avoid_terms(self) -> list[str]:
        return _request_tokens(' ; '.join([*self.forbid, *self.text_constraints.avoid_text_meanings]))

    def query_interpretation(self) -> dict[str, Any]:
        return {
            'send': self.send,
            'intent_core': self.intent_core,
            'secondary_goals': list(self.secondary_goals),
            'semantic_focus': self.semantic_focus.as_dict(),
            'visual_focus': self.visual_focus.as_dict(),
            'style_focus': self.style_focus.as_dict(),
            'text_constraints': self.text_constraints.as_dict(),
            'intensity_limits': self.intensity_limits.as_dict(),
            'forbid': list(self.forbid),
            'candidate_budget': self.candidate_budget,
            'deprecated_aliases_used': dict(self.deprecated_aliases_used),
        }
