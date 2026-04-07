from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

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
_STYLE_GOAL_TO_LEGACY_POLICY = {value: key for key, value in _LEGACY_STYLE_POLICY_TO_GOAL.items()}
_NOISE_TOKEN_RE = re.compile(r"[@#]?[A-Za-z0-9_./:\\-]+")
_URL_RE = re.compile(r'^(?:https?://|www\.)', re.IGNORECASE)
_PATH_RE = re.compile(r'^(?:[A-Za-z]:[\\/]|[./]{1,2}[\\/]|[/\\])')
_NOISE_LITERAL_TOKENS = {
    'assistant',
    'username',
    'nickname',
    'telegram',
    'discord',
    'shell_exec',
    'python_exec',
    'file_send',
    'sticker_query',
    'sticker_send_selected',
    'intent_action',
}


def _norm_text(value: Any) -> str:
    return ' '.join(str(value or '').replace('\n', ' ').replace('\t', ' ').split()).strip()


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


def _sanitize_text_field(field_name: str, value: Any) -> tuple[str, list[str], str | None]:
    text = _norm_text(value)
    if not text:
        return '', [], None
    dropped: list[str] = []
    kept_tokens: list[str] = []
    for raw_token in text.split():
        cleaned = raw_token.strip(".,;:!?()[]{}<>\"'")
        lowered = cleaned.lower()
        if not cleaned:
            continue
        if cleaned.startswith('@') or _URL_RE.match(cleaned) or _PATH_RE.match(cleaned):
            dropped.append(cleaned)
            continue
        if lowered in _NOISE_LITERAL_TOKENS:
            dropped.append(cleaned)
            continue
        if re.fullmatch(r'(?:user(?:name)?|assistant|bot)[:=_-]?[A-Za-z0-9_]+', lowered):
            dropped.append(cleaned)
            continue
        if re.fullmatch(r'[A-Za-z0-9_]+bot', lowered) and len(cleaned) <= 32:
            dropped.append(cleaned)
            continue
        if re.fullmatch(r'[A-Za-z_]+-\d{3,}', lowered) or re.fullmatch(r'[A-Za-z_]*\d{4,}[A-Za-z_]*', lowered):
            dropped.append(cleaned)
            continue
        kept_tokens.append(cleaned)
    sanitized = ' '.join(kept_tokens).strip()
    warning = None
    if dropped and sanitized:
        warning = f'{field_name} dropped likely non-semantic tokens: {", ".join(dropped[:6])}'
    elif dropped and not sanitized:
        warning = f'{field_name} only contained likely non-semantic noise and became empty'
    return sanitized, dropped[:12], warning


def _sanitize_text_list(field_name: str, values: Any) -> tuple[list[str], list[str], list[str]]:
    if not isinstance(values, list):
        return [], [], []
    cleaned_values: list[str] = []
    dropped: list[str] = []
    warnings: list[str] = []
    for index, value in enumerate(values):
        cleaned, removed, warning = _sanitize_text_field(f'{field_name}[{index}]', value)
        if cleaned:
            cleaned_values.append(cleaned)
        dropped.extend(removed)
        if warning:
            warnings.append(warning)
    return cleaned_values, dropped[:24], warnings[:12]


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
    prefer_cluster: str = ''

    def as_dict(self) -> dict[str, Any]:
        return {
            'style_goal': self.style_goal,
            'style_hints': list(self.style_hints),
            'prefer_pack': self.prefer_pack,
            'prefer_cluster': self.prefer_cluster,
        }

    def display_dict(self) -> dict[str, Any]:
        return {
            'style_goal': self.style_goal,
            'style_hints': list(self.style_hints),
            'preferred_pack': self.prefer_pack,
            'preferred_style_cluster': self.prefer_cluster,
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
    max_harshness: int = 3
    max_intimacy: int = 4
    max_meme_dependence: int = 4
    allow_animation: bool = False

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
    prefer_cluster: str = ''

    def as_dict(self) -> dict[str, Any]:
        return build_persona_dict(visual_identity={
            'character_archetype': self.character_archetype,
            'rendering_style': self.rendering_style,
            'palette_mood': self.palette_mood,
            'style_hints': list(self.style_hints),
            'prefer_pack': self.prefer_pack,
            'prefer_cluster': self.prefer_cluster,
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
            'preferred_style_cluster': self.prefer_cluster,
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
    continuity_note: str = ''
    avoid_misread_as: str = ''

    def as_dict(self) -> dict[str, Any]:
        return {
            'social_read': self.social_read,
            'subtext': self.subtext,
            'face_and_pose': self.face_and_pose,
            'continuity_note': self.continuity_note,
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
    send: bool = True
    deprecated_aliases_used: dict[str, Any] = field(default_factory=dict)
    field_warnings: list[str] = field(default_factory=list)
    dropped_noise_terms: list[str] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> 'StickerRetrievalPlan':
        data = _norm_mapping(payload)
        advanced = _norm_mapping(data.get('advanced'))
        deprecated_aliases_used: dict[str, Any] = {}
        for key in (
            'emotion_tone',
            'social_goal',
            'visual_hint',
            'text_hint',
            'prefer_pack',
            'prefer_cluster',
            'semantic_focus',
            'visual_focus',
            'style_focus',
            'text_constraints',
            'intensity_limits',
            'safety_limits',
            'text_priority',
            'max_harshness',
            'max_intimacy',
            'max_meme_dependence',
            'allow_animation',
            'forbid',
            'style_policy',
        ):
            if key in data:
                deprecated_aliases_used[key] = data.get(key)

        field_warnings: list[str] = []
        dropped_noise_terms: list[str] = []

        intent_core, dropped, warning = _sanitize_text_field('intent_core', data.get('intent_core', ''))
        dropped_noise_terms.extend(dropped)
        if warning:
            field_warnings.append(warning)
        if not intent_core:
            raise ValueError('intent_core is required and must contain semantic content')

        secondary_goals, dropped, warnings = _sanitize_text_list('secondary_goals', data.get('secondary_goals'))
        dropped_noise_terms.extend(dropped)
        field_warnings.extend(warnings)
        forbid, dropped, warnings = _sanitize_text_list('forbid', data.get('forbid'))
        dropped_noise_terms.extend(dropped)
        field_warnings.extend(warnings)

        def sanitize_from(*, field_name: str, primary: Any, fallback: Any = '') -> str:
            cleaned, removed, local_warning = _sanitize_text_field(field_name, primary if primary not in (None, '') else fallback)
            dropped_noise_terms.extend(removed)
            if local_warning:
                field_warnings.append(local_warning)
            return cleaned

        simple_hints = SimpleHints(
            emotion_tone=sanitize_from(field_name='reaction_tone', primary=data.get('reaction_tone'), fallback=data.get('emotion_tone')),
            social_goal=sanitize_from(field_name='social_intent', primary=data.get('social_intent'), fallback=data.get('social_goal')),
            visual_hint=sanitize_from(field_name='expression_cue', primary=data.get('expression_cue'), fallback=data.get('visual_hint')),
            text_hint=sanitize_from(field_name='caption_meaning', primary=data.get('caption_meaning'), fallback=data.get('text_hint')),
            diversity_preference=_normalize_diversity_preference(data.get('diversity_preference')),
        )
        persona_source = _norm_mapping(data.get('persona'))
        persona_visual_source = _norm_mapping(persona_source.get('visual_identity'))
        persona_affect_source = _norm_mapping(persona_source.get('affect_profile'))
        selection_lens_source = _norm_mapping(data.get('selection_lens'))

        semantic_source = _norm_mapping(advanced.get('semantic_focus') if 'semantic_focus' in advanced else data.get('semantic_focus'))
        visual_source = _norm_mapping(advanced.get('visual_focus') if 'visual_focus' in advanced else data.get('visual_focus'))
        style_source = _norm_mapping(advanced.get('style_focus') if 'style_focus' in advanced else data.get('style_focus'))
        text_source = _norm_mapping(advanced.get('text_constraints') if 'text_constraints' in advanced else data.get('text_constraints'))
        intensity_source = _norm_mapping(
            advanced.get('intensity_limits')
            if 'intensity_limits' in advanced
            else advanced.get('safety_limits')
            if 'safety_limits' in advanced
            else data.get('intensity_limits')
            if 'intensity_limits' in data
            else data.get('safety_limits')
        )

        legacy_text_priority = _norm_text(data.get('text_priority', '')).lower()
        legacy_style_policy = _norm_text(data.get('style_policy', '')).lower()
        if 'prefer_pack' in style_source:
            deprecated_aliases_used['advanced.style_focus.prefer_pack'] = style_source.get('prefer_pack')
        if 'prefer_cluster' in style_source:
            deprecated_aliases_used['advanced.style_focus.prefer_cluster'] = style_source.get('prefer_cluster')
        if 'prefer_pack' in persona_visual_source:
            deprecated_aliases_used['persona.visual_identity.prefer_pack'] = persona_visual_source.get('prefer_pack')
        if 'prefer_cluster' in persona_visual_source:
            deprecated_aliases_used['persona.visual_identity.prefer_cluster'] = persona_visual_source.get('prefer_cluster')

        semantic_focus = SemanticFocus(
            reaction_type=sanitize_from(field_name='advanced.semantic_focus.reaction_type', primary=semantic_source.get('reaction_type')),
            reply_force=sanitize_from(field_name='advanced.semantic_focus.reply_force', primary=semantic_source.get('reply_force')),
            emotional_valence=sanitize_from(field_name='advanced.semantic_focus.emotional_valence', primary=semantic_source.get('emotional_valence')),
            irony_strength=sanitize_from(field_name='advanced.semantic_focus.irony_strength', primary=semantic_source.get('irony_strength')),
            social_stance=sanitize_from(field_name='advanced.semantic_focus.social_stance', primary=semantic_source.get('social_stance')),
            conversation_role=sanitize_from(field_name='advanced.semantic_focus.conversation_role', primary=semantic_source.get('conversation_role')),
            relationship_fit=sanitize_from(field_name='advanced.semantic_focus.relationship_fit', primary=semantic_source.get('relationship_fit')),
        )
        visual_focus = VisualFocus(
            eye_signal=sanitize_from(field_name='advanced.visual_focus.eye_signal', primary=visual_source.get('eye_signal')),
            mouth_signal=sanitize_from(field_name='advanced.visual_focus.mouth_signal', primary=visual_source.get('mouth_signal')),
            motion_signal=sanitize_from(field_name='advanced.visual_focus.motion_signal', primary=visual_source.get('motion_signal')),
            delivery_style=sanitize_from(field_name='advanced.visual_focus.delivery_style', primary=visual_source.get('delivery_style')),
            humor_style=sanitize_from(field_name='advanced.visual_focus.humor_style', primary=visual_source.get('humor_style')),
        )
        style_hints, dropped, warnings = _sanitize_text_list('advanced.style_focus.style_hints', style_source.get('style_hints'))
        dropped_noise_terms.extend(dropped)
        field_warnings.extend(warnings)
        persona_style_hints, dropped, warnings = _sanitize_text_list('persona.visual_identity.style_hints', persona_visual_source.get('style_hints'))
        dropped_noise_terms.extend(dropped)
        field_warnings.extend(warnings)
        raw_style_goal = _norm_text(style_source.get('style_goal', '')).lower()
        if raw_style_goal not in _STYLE_GOALS:
            raw_style_goal = _LEGACY_STYLE_POLICY_TO_GOAL.get(legacy_style_policy, 'preserve')
        style_focus = StyleFocus(
            style_goal=raw_style_goal,
            style_hints=style_hints,
            prefer_pack=_norm_text(style_source.get('preferred_pack', style_source.get('prefer_pack', data.get('preferred_pack', data.get('prefer_pack', ''))))),
            prefer_cluster=_norm_text(style_source.get('preferred_style_cluster', style_source.get('prefer_cluster', data.get('preferred_style_cluster', data.get('prefer_cluster', ''))))),
        )
        must_include, dropped, warnings = _sanitize_text_list('advanced.text_constraints.must_include', text_source.get('must_include'))
        dropped_noise_terms.extend(dropped)
        field_warnings.extend(warnings)
        avoid_text_meanings, dropped, warnings = _sanitize_text_list('advanced.text_constraints.avoid_text_meanings', text_source.get('avoid_text_meanings'))
        dropped_noise_terms.extend(dropped)
        field_warnings.extend(warnings)
        text_priority = _norm_text(text_source.get('text_priority', legacy_text_priority or 'prefer')).lower() or 'prefer'
        if text_priority not in _TEXT_PRIORITIES:
            text_priority = 'prefer'
        text_constraints = TextConstraints(
            text_priority=text_priority,
            must_include=must_include,
            avoid_text_meanings=avoid_text_meanings,
        )
        intensity_limits = IntensityLimits(
            max_harshness=_bounded_int(intensity_source.get('max_harshness', data.get('max_harshness', 3)), default=3, minimum=0, maximum=4),
            max_intimacy=_bounded_int(intensity_source.get('max_intimacy', data.get('max_intimacy', 4)), default=4, minimum=0, maximum=4),
            max_meme_dependence=_bounded_int(intensity_source.get('max_meme_dependence', data.get('max_meme_dependence', 4)), default=4, minimum=0, maximum=4),
            allow_animation=_norm_bool(intensity_source.get('allow_animation', data.get('allow_animation', False)), default=False),
        )
        persona = StickerPersona(
            visual_identity=PersonaVisualIdentity(
                character_archetype=sanitize_from(field_name='persona.visual_identity.character_archetype', primary=persona_visual_source.get('character_archetype')),
                rendering_style=sanitize_from(field_name='persona.visual_identity.rendering_style', primary=persona_visual_source.get('rendering_style')),
                palette_mood=sanitize_from(field_name='persona.visual_identity.palette_mood', primary=persona_visual_source.get('palette_mood')),
                style_hints=persona_style_hints,
                prefer_pack=_norm_text(persona_visual_source.get('preferred_pack', persona_visual_source.get('prefer_pack', ''))),
                prefer_cluster=_norm_text(persona_visual_source.get('preferred_style_cluster', persona_visual_source.get('prefer_cluster', ''))),
            ),
            affect_profile=PersonaAffectProfile(
                default_tone=sanitize_from(field_name='persona.affect_profile.default_tone', primary=persona_affect_source.get('default_tone')),
                expression_bias=sanitize_from(field_name='persona.affect_profile.expression_bias', primary=persona_affect_source.get('expression_bias')),
                pose_bias=sanitize_from(field_name='persona.affect_profile.pose_bias', primary=persona_affect_source.get('pose_bias')),
                delivery_bias=sanitize_from(field_name='persona.affect_profile.delivery_bias', primary=persona_affect_source.get('delivery_bias')),
                humor_bias=sanitize_from(field_name='persona.affect_profile.humor_bias', primary=persona_affect_source.get('humor_bias')),
            ),
        )
        selection_lens = SelectionLens(
            social_read=sanitize_from(field_name='selection_lens.social_read', primary=selection_lens_source.get('social_read')),
            subtext=sanitize_from(field_name='selection_lens.subtext', primary=selection_lens_source.get('subtext')),
            face_and_pose=sanitize_from(field_name='selection_lens.face_and_pose', primary=selection_lens_source.get('face_and_pose')),
            continuity_note=sanitize_from(field_name='selection_lens.continuity_note', primary=selection_lens_source.get('continuity_note')),
            avoid_misread_as=sanitize_from(field_name='selection_lens.avoid_misread_as', primary=selection_lens_source.get('avoid_misread_as')),
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
            candidate_budget=_bounded_int(data.get('candidate_budget', 5), default=5, minimum=1, maximum=8),
            send=_norm_bool(data.get('send', True), default=True),
            deprecated_aliases_used=deprecated_aliases_used,
            field_warnings=field_warnings[:16],
            dropped_noise_terms=sorted(dict.fromkeys(dropped_noise_terms))[:32],
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

    @property
    def prefer_cluster(self) -> str:
        return self.style_focus.prefer_cluster

    @property
    def style_policy(self) -> str:
        return _STYLE_GOAL_TO_LEGACY_POLICY.get(self.style_goal, 'continue')

    @property
    def has_persona_request(self) -> bool:
        return self.persona.has_values()

    def helper_hint_terms(self) -> list[str]:
        return _request_tokens(' ; '.join(self.simple_hints.request_texts()))

    def caption_query_text(self) -> str:
        parts = [
            self.intent_core,
            *self.secondary_goals,
            self.emotion_tone,
            self.social_goal,
            self.text_hint,
            self.selection_lens.social_read,
            self.selection_lens.subtext,
            *self.semantic_focus.request_texts(),
        ]
        if self.text_priority == 'require':
            parts.append('visible caption text should dominate meaning')
        elif self.text_priority == 'prefer':
            parts.append('caption meaning should matter strongly')
        if self.selection_lens.avoid_misread_as:
            parts.append('should not read as: ' + self.selection_lens.avoid_misread_as)
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
            self.emotion_tone,
            self.social_goal,
            self.visual_hint,
            *self.persona.request_texts(),
            *self.selection_lens.request_texts(),
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
            *self.simple_hints.request_texts(),
            *self.persona.request_texts(),
            *self.selection_lens.request_texts(),
            *self.semantic_focus.request_texts(),
            *self.visual_focus.request_texts(),
        ])
        return _request_tokens(text)

    def style_request_terms(self) -> list[str]:
        return _request_tokens(' ; '.join(self.style_hints))

    def avoid_terms(self) -> list[str]:
        return _request_tokens(' ; '.join([*self.forbid, *self.text_constraints.avoid_text_meanings, self.selection_lens.avoid_misread_as]))

    def memory_query_bundle(self) -> dict[str, Any]:
        return {
            'intent_core': self.intent_core,
            'emotion_tone': self.emotion_tone,
            'social_goal': self.social_goal,
            'visual_hint': self.visual_hint,
            'text_hint': self.text_hint,
            'persona_mode': self.persona_mode,
            'persona_character_archetype': self.persona.visual_identity.character_archetype,
            'persona_rendering_style': self.persona.visual_identity.rendering_style,
            'persona_default_tone': self.persona.affect_profile.default_tone,
            'selection_social_read': self.selection_lens.social_read,
            'selection_subtext': self.selection_lens.subtext,
            'selection_face_and_pose': self.selection_lens.face_and_pose,
            'prefer_pack': self.prefer_pack,
            'prefer_cluster': self.prefer_cluster,
            'style_goal': self.style_goal,
            'diversity_preference': self.diversity_preference,
        }

    def likely_culprit_fields(self) -> list[str]:
        fields: list[str] = []
        if self.text_constraints.must_include or self.text_constraints.avoid_text_meanings or self.text_priority == 'require':
            fields.append('advanced.text_constraints')
        if self.semantic_focus.active_fields():
            fields.append('advanced.semantic_focus')
        if self.visual_focus.active_fields():
            fields.append('advanced.visual_focus')
        if self.prefer_pack:
            fields.append('preferred_pack')
        if self.prefer_cluster:
            fields.append('preferred_style_cluster')
        if self.has_persona_request:
            fields.append('persona')
        if self.selection_lens.active_fields():
            fields.append('selection_lens')
        if self.max_harshness < 3 or self.max_intimacy < 4 or self.max_meme_dependence < 4:
            fields.append('advanced.intensity_limits')
        if self.field_warnings or self.dropped_noise_terms:
            fields.append('sanitation')
        return list(dict.fromkeys(fields))

    def query_interpretation(self) -> dict[str, Any]:
        return {
            'send': self.send,
            'intent_core': self.intent_core,
            'secondary_goals': list(self.secondary_goals),
            'simple_hints': self.simple_hints.display_dict(),
            'persona_mode': self.persona_mode,
            'persona': self.persona.display_dict(),
            'selection_lens': self.selection_lens.as_dict(),
            'advanced': {
                'semantic_focus': self.semantic_focus.as_dict(),
                'visual_focus': self.visual_focus.as_dict(),
                'style_focus': self.style_focus.display_dict(),
                'text_constraints': self.text_constraints.as_dict(),
                'intensity_limits': self.intensity_limits.as_dict(),
                'forbid': list(self.forbid),
            },
            'candidate_budget': self.candidate_budget,
            'field_warnings': list(self.field_warnings),
            'dropped_noise_terms': list(self.dropped_noise_terms),
            'deprecated_aliases_used': dict(self.deprecated_aliases_used),
        }


def _normalize_diversity_preference(value: Any) -> str:
    normalized = _norm_text(value).lower() or 'default'
    return normalized if normalized in _DIVERSITY_PREFERENCES else 'default'


def _normalize_persona_mode(value: Any, *, has_persona: bool) -> str:
    normalized = _norm_text(value).lower()
    if normalized in PERSONA_MODES:
        return normalized
    return 'merge_and_remember' if has_persona else 'inherit'
