from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tgchatbot.stickers.config import StickerConfig
from tgchatbot.stickers.persona import PERSONA_MODES, build_persona_dict, persona_has_values

_TEXT_PRIORITIES = {'require', 'prefer', 'ignore'}
_STYLE_GOALS = {'preserve', 'allow_switch', 'prefer_switch', 'ignore_style'}
_DIVERSITY_PREFERENCES = {'default', 'prefer_fresh_variant'}
_LEGACY_STYLE_POLICY_TO_GOAL = {
    'continue': 'preserve',
    'neutral': 'allow_switch',
    'prefer_switch': 'prefer_switch',
    'hard_switch': 'ignore_style',
}


def _norm_text(value: Any) -> str:
    return ' '.join(str(value or '').replace('\n', ' ').replace('\t', ' ').split()).strip()


def _norm_text_list(value: Any) -> list[str]:
    return [text for item in value if (text := _norm_text(item))] if isinstance(value, list) else []


def _first_present(*values: Any) -> Any:
    # Strict provider schemas represent omitted optional fields as null.
    # Null must not erase an explicitly supplied compatibility alias/control.
    return next((value for value in values if value is not None and value != ''), None)


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


@dataclass(slots=True)
class SemanticFocus:
    reaction_type: str = ''
    reply_force: str = ''
    emotional_valence: str = ''
    irony_strength: str = ''
    social_stance: str = ''
    conversation_role: str = ''
    relationship_fit: str = ''

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
    style_goal: str = 'preserve'
    style_hints: list[str] = field(default_factory=list)
    prefer_pack: str = ''

    def as_dict(self) -> dict[str, Any]:
        return {
            'style_goal': self.style_goal,
            'style_hints': list(self.style_hints),
            'prefer_pack': self.prefer_pack,
        }

    def display_dict(self) -> dict[str, Any]:
        return {
            'style_goal': self.style_goal,
            'style_hints': list(self.style_hints),
            'preferred_pack': self.prefer_pack,
        }


@dataclass(slots=True)
class TextConstraints:
    text_priority: str = 'prefer'
    must_include: list[str] = field(default_factory=list)
    avoid_text_meanings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            'text_priority': self.text_priority,
            'must_include': list(self.must_include),
            'avoid_text_meanings': list(self.avoid_text_meanings),
        }


@dataclass(slots=True)
class IntensityLimits:
    max_harshness: int = 4
    max_intimacy: int = 4
    max_meme_dependence: int = 4
    allow_animation: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            'max_harshness': self.max_harshness,
            'max_intimacy': self.max_intimacy,
            'max_meme_dependence': self.max_meme_dependence,
            'allow_animation': self.allow_animation,
        }


@dataclass(slots=True)
class SimpleHints:
    emotion_tone: str = ''
    social_goal: str = ''
    visual_hint: str = ''
    text_hint: str = ''
    diversity_preference: str = 'default'

    def as_dict(self) -> dict[str, Any]:
        return {
            'emotion_tone': self.emotion_tone,
            'social_goal': self.social_goal,
            'visual_hint': self.visual_hint,
            'text_hint': self.text_hint,
            'diversity_preference': self.diversity_preference,
        }

    def request_texts(self) -> list[str]:
        return [value for value in [self.emotion_tone, self.social_goal, self.visual_hint, self.text_hint] if value]

    def display_dict(self) -> dict[str, Any]:
        return {
            'reaction_tone': self.emotion_tone,
            'social_intent': self.social_goal,
            'expression_cue': self.visual_hint,
            'caption_meaning': self.text_hint,
            'diversity_preference': self.diversity_preference,
        }


@dataclass(slots=True)
class PersonaVisualIdentity:
    character_archetype: str = ''
    rendering_style: str = ''
    palette_mood: str = ''
    style_hints: list[str] = field(default_factory=list)
    prefer_pack: str = ''

    def as_dict(self) -> dict[str, Any]:
        return build_persona_dict(visual_identity={
            'character_archetype': self.character_archetype,
            'rendering_style': self.rendering_style,
            'palette_mood': self.palette_mood,
            'style_hints': list(self.style_hints),
            'prefer_pack': self.prefer_pack,
        }).get('visual_identity', {})

    def active_fields(self) -> dict[str, str]:
        return {key: value for key, value in self.as_dict().items() if key != 'style_hints' and value}

    def request_texts(self) -> list[str]:
        fields = list(self.active_fields().values())
        return [*fields, *list(self.style_hints)]

    def display_dict(self) -> dict[str, Any]:
        return {
            'character_archetype': self.character_archetype,
            'rendering_style': self.rendering_style,
            'palette_mood': self.palette_mood,
            'style_hints': list(self.style_hints),
            'preferred_pack': self.prefer_pack,
        }


@dataclass(slots=True)
class PersonaAffectProfile:
    default_tone: str = ''
    expression_bias: str = ''
    pose_bias: str = ''
    delivery_bias: str = ''
    humor_bias: str = ''

    def as_dict(self) -> dict[str, Any]:
        return build_persona_dict(affect_profile={
            'default_tone': self.default_tone,
            'expression_bias': self.expression_bias,
            'pose_bias': self.pose_bias,
            'delivery_bias': self.delivery_bias,
            'humor_bias': self.humor_bias,
        }).get('affect_profile', {})

    def active_fields(self) -> dict[str, str]:
        return {key: value for key, value in self.as_dict().items() if value}

    def request_texts(self) -> list[str]:
        return list(self.active_fields().values())


@dataclass(slots=True)
class StickerPersona:
    visual_identity: PersonaVisualIdentity = field(default_factory=PersonaVisualIdentity)
    affect_profile: PersonaAffectProfile = field(default_factory=PersonaAffectProfile)

    def as_dict(self) -> dict[str, Any]:
        return build_persona_dict(
            visual_identity=self.visual_identity.as_dict(),
            affect_profile=self.affect_profile.as_dict(),
        )

    def has_values(self) -> bool:
        return persona_has_values(self.as_dict())

    def request_texts(self) -> list[str]:
        return [*self.visual_identity.request_texts(), *self.affect_profile.request_texts()]

    def display_dict(self) -> dict[str, Any]:
        visual_identity = self.visual_identity.display_dict()
        affect_profile = self.affect_profile.as_dict()
        payload: dict[str, Any] = {}
        has_visual_identity = any(bool(value) for key, value in visual_identity.items() if key == 'style_hints') or any(
            str(value or '').strip()
            for key, value in visual_identity.items()
            if key != 'style_hints'
        )
        if has_visual_identity:
            payload['visual_identity'] = visual_identity
        if affect_profile:
            payload['affect_profile'] = affect_profile
        return payload


@dataclass(slots=True)
class SelectionLens:
    social_read: str = ''
    subtext: str = ''
    face_and_pose: str = ''
    avoid_misread_as: str = ''

    def as_dict(self) -> dict[str, Any]:
        return {
            'social_read': self.social_read,
            'subtext': self.subtext,
            'face_and_pose': self.face_and_pose,
            'avoid_misread_as': self.avoid_misread_as,
        }

    def active_fields(self) -> dict[str, str]:
        return {key: value for key, value in self.as_dict().items() if value}

    def request_texts(self) -> list[str]:
        return [
            value
            for key, value in self.as_dict().items()
            if key != 'avoid_misread_as' and value
        ]


@dataclass(slots=True)
class StickerRetrievalPlan:
    intent_core: str
    secondary_goals: list[str] = field(default_factory=list)
    simple_hints: SimpleHints = field(default_factory=SimpleHints)
    persona: StickerPersona = field(default_factory=StickerPersona)
    persona_mode: str = 'inherit'
    selection_lens: SelectionLens = field(default_factory=SelectionLens)
    semantic_focus: SemanticFocus = field(default_factory=SemanticFocus)
    visual_focus: VisualFocus = field(default_factory=VisualFocus)
    style_focus: StyleFocus = field(default_factory=StyleFocus)
    text_constraints: TextConstraints = field(default_factory=TextConstraints)
    intensity_limits: IntensityLimits = field(default_factory=IntensityLimits)
    forbid: list[str] = field(default_factory=list)
    candidate_budget: int = 5
    preferred_character_family: str = ''
    required_pack: str = ''
    required_character_family: str = ''
    style_goal_explicit: bool = False

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, config: StickerConfig | None = None) -> 'StickerRetrievalPlan':
        config = config or StickerConfig.from_env()
        data = _norm_mapping(payload)
        advanced = _norm_mapping(data.get('advanced'))
        intent_core = _norm_text(data.get('intent_core', ''))
        if not intent_core:
            raise ValueError('intent_core is required and must contain semantic content')

        secondary_goals = _norm_text_list(data.get('secondary_goals'))
        forbid = _norm_text_list(_first_present(advanced.get('forbid'), data.get('forbid')))

        simple_hints = SimpleHints(
            emotion_tone=_norm_text(_first_present(data.get('reaction_tone'), data.get('emotion_tone'))),
            social_goal=_norm_text(_first_present(data.get('social_intent'), data.get('social_goal'))),
            visual_hint=_norm_text(_first_present(data.get('expression_cue'), data.get('visual_hint'))),
            text_hint=_norm_text(_first_present(data.get('caption_meaning'), data.get('text_hint'))),
            diversity_preference=_normalize_diversity_preference(data.get('diversity_preference')),
        )
        persona_source = _norm_mapping(data.get('persona'))
        persona_visual_source = _norm_mapping(persona_source.get('visual_identity'))
        persona_affect_source = _norm_mapping(persona_source.get('affect_profile'))
        selection_lens_source = _norm_mapping(data.get('selection_lens'))

        semantic_source = _norm_mapping(_first_present(advanced.get('semantic_focus'), data.get('semantic_focus')))
        visual_source = _norm_mapping(_first_present(advanced.get('visual_focus'), data.get('visual_focus')))
        style_source = _norm_mapping(_first_present(advanced.get('style_focus'), data.get('style_focus')))
        text_source = _norm_mapping(_first_present(advanced.get('text_constraints'), data.get('text_constraints')))
        intensity_sources = [_norm_mapping(source) for source in (
            advanced.get('intensity_limits'), advanced.get('safety_limits'),
            data.get('intensity_limits'), data.get('safety_limits'))]

        def intensity_value(name, default):
            return _first_present(*(source.get(name) for source in intensity_sources), data.get(name), default)

        legacy_text_priority = _norm_text(data.get('text_priority', '')).lower()
        legacy_style_policy = _norm_text(data.get('style_policy', '')).lower()

        semantic_focus = SemanticFocus(
            reaction_type=_norm_text(semantic_source.get('reaction_type')),
            reply_force=_norm_text(semantic_source.get('reply_force')),
            emotional_valence=_norm_text(semantic_source.get('emotional_valence')),
            irony_strength=_norm_text(semantic_source.get('irony_strength')),
            social_stance=_norm_text(semantic_source.get('social_stance')),
            conversation_role=_norm_text(semantic_source.get('conversation_role')),
            relationship_fit=_norm_text(semantic_source.get('relationship_fit')),
        )
        visual_focus = VisualFocus(
            eye_signal=_norm_text(visual_source.get('eye_signal')),
            mouth_signal=_norm_text(visual_source.get('mouth_signal')),
            motion_signal=_norm_text(visual_source.get('motion_signal')),
            delivery_style=_norm_text(visual_source.get('delivery_style')),
            humor_style=_norm_text(visual_source.get('humor_style')),
        )
        style_hints = _norm_text_list(style_source.get('style_hints'))
        persona_style_hints = _norm_text_list(persona_visual_source.get('style_hints'))
        raw_style_goal = _norm_text(style_source.get('style_goal', '')).lower()
        style_goal_explicit = raw_style_goal in _STYLE_GOALS or legacy_style_policy in _LEGACY_STYLE_POLICY_TO_GOAL
        if raw_style_goal not in _STYLE_GOALS:
            raw_style_goal = _LEGACY_STYLE_POLICY_TO_GOAL.get(legacy_style_policy, 'preserve')
        style_focus = StyleFocus(
            style_goal=raw_style_goal,
            style_hints=style_hints,
            prefer_pack=_norm_text(_first_present(style_source.get('preferred_pack'), style_source.get('prefer_pack'), data.get('preferred_pack'), data.get('prefer_pack'))),
        )
        must_include = _norm_text_list(text_source.get('must_include'))
        avoid_text_meanings = _norm_text_list(text_source.get('avoid_text_meanings'))
        text_priority = _norm_text(_first_present(text_source.get('text_priority'), legacy_text_priority, 'prefer')).lower() or 'prefer'
        if text_priority not in _TEXT_PRIORITIES:
            text_priority = 'prefer'
        text_constraints = TextConstraints(
            text_priority=text_priority,
            must_include=must_include,
            avoid_text_meanings=avoid_text_meanings,
        )
        intensity_limits = IntensityLimits(
            max_harshness=_bounded_int(intensity_value('max_harshness', 4), default=4, minimum=0, maximum=4),
            max_intimacy=_bounded_int(intensity_value('max_intimacy', 4), default=4, minimum=0, maximum=4),
            max_meme_dependence=_bounded_int(intensity_value('max_meme_dependence', 4), default=4, minimum=0, maximum=4),
            allow_animation=_norm_bool(intensity_value('allow_animation', True), default=True),
        )
        persona = StickerPersona(
            visual_identity=PersonaVisualIdentity(
                character_archetype=_norm_text(persona_visual_source.get('character_archetype')),
                rendering_style=_norm_text(persona_visual_source.get('rendering_style')),
                palette_mood=_norm_text(persona_visual_source.get('palette_mood')),
                style_hints=persona_style_hints,
                prefer_pack=_norm_text(_first_present(persona_visual_source.get('preferred_pack'), persona_visual_source.get('prefer_pack'))),
            ),
            affect_profile=PersonaAffectProfile(
                default_tone=_norm_text(persona_affect_source.get('default_tone')),
                expression_bias=_norm_text(persona_affect_source.get('expression_bias')),
                pose_bias=_norm_text(persona_affect_source.get('pose_bias')),
                delivery_bias=_norm_text(persona_affect_source.get('delivery_bias')),
                humor_bias=_norm_text(persona_affect_source.get('humor_bias')),
            ),
        )
        selection_lens = SelectionLens(
            social_read=_norm_text(selection_lens_source.get('social_read')),
            subtext=_norm_text(selection_lens_source.get('subtext')),
            face_and_pose=_norm_text(selection_lens_source.get('face_and_pose')),
            avoid_misread_as=_norm_text(selection_lens_source.get('avoid_misread_as')),
        )

        return cls(
            intent_core=intent_core,
            secondary_goals=secondary_goals,
            simple_hints=simple_hints,
            persona=persona,
            persona_mode=_normalize_persona_mode(data.get('persona_mode'), has_persona=persona.has_values()),
            selection_lens=selection_lens,
            semantic_focus=semantic_focus,
            visual_focus=visual_focus,
            style_focus=style_focus,
            text_constraints=text_constraints,
            intensity_limits=intensity_limits,
            forbid=forbid,
            candidate_budget=_bounded_int(data.get('candidate_budget'), default=config.candidate_count, minimum=1, maximum=config.max_candidates),
            preferred_character_family=_norm_text(data.get('preferred_character_family')),
            required_pack=_norm_text(data.get('required_pack')),
            required_character_family=_norm_text(data.get('required_character_family')),
            style_goal_explicit=style_goal_explicit,
        )

    @property
    def emotion_tone(self) -> str:
        return self.simple_hints.emotion_tone

    @property
    def social_goal(self) -> str:
        return self.simple_hints.social_goal

    @property
    def visual_hint(self) -> str:
        return self.simple_hints.visual_hint

    @property
    def text_hint(self) -> str:
        return self.simple_hints.text_hint

    @property
    def diversity_preference(self) -> str:
        return self.simple_hints.diversity_preference

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


def _normalize_diversity_preference(value: Any) -> str:
    normalized = _norm_text(value).lower() or 'default'
    return normalized if normalized in _DIVERSITY_PREFERENCES else 'default'


def _normalize_persona_mode(value: Any, *, has_persona: bool) -> str:
    normalized = _norm_text(value).lower()
    if normalized in PERSONA_MODES:
        return normalized
    return 'merge_and_remember' if has_persona else 'inherit'
