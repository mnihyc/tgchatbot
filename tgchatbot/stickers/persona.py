from __future__ import annotations

from typing import Any

PERSONA_MODES = {'inherit', 'merge_and_remember', 'use_once', 'clear_session_persona'}
VISUAL_IDENTITY_FIELDS = (
    'character_archetype',
    'rendering_style',
    'palette_mood',
    'prefer_pack',
    'prefer_cluster',
)
AFFECT_PROFILE_FIELDS = (
    'default_tone',
    'expression_bias',
    'pose_bias',
    'delivery_bias',
    'humor_bias',
)


def _norm_text(value: Any) -> str:
    return ' '.join(str(value or '').replace('\n', ' ').replace('\t', ' ').split()).strip()


def _norm_text_list(values: Any) -> list[str]:
    if not isinstance(values, list):
        return []
    cleaned: list[str] = []
    for value in values:
        text = _norm_text(value)
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def build_persona_dict(*, visual_identity: dict[str, Any] | None = None, affect_profile: dict[str, Any] | None = None) -> dict[str, Any]:
    visual_source = dict(visual_identity or {})
    affect_source = dict(affect_profile or {})
    visual: dict[str, Any] = {}
    affect: dict[str, Any] = {}
    for field_name in VISUAL_IDENTITY_FIELDS:
        value = _norm_text(visual_source.get(field_name))
        if value:
            visual[field_name] = value
    style_hints = _norm_text_list(visual_source.get('style_hints'))
    if style_hints:
        visual['style_hints'] = style_hints
    for field_name in AFFECT_PROFILE_FIELDS:
        value = _norm_text(affect_source.get(field_name))
        if value:
            affect[field_name] = value
    persona: dict[str, Any] = {}
    if visual:
        persona['visual_identity'] = visual
    if affect:
        persona['affect_profile'] = affect
    return persona


def compact_persona_dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return build_persona_dict(
        visual_identity=value.get('visual_identity'),
        affect_profile=value.get('affect_profile'),
    )


def persona_has_values(value: Any) -> bool:
    return bool(compact_persona_dict(value))


def merge_persona_dicts(base: Any, override: Any) -> dict[str, Any]:
    base_persona = compact_persona_dict(base)
    override_persona = compact_persona_dict(override)
    merged = compact_persona_dict(base_persona)
    if not override_persona:
        return merged
    visual = dict(merged.get('visual_identity') or {})
    override_visual = dict(override_persona.get('visual_identity') or {})
    if override_visual:
        for field_name in VISUAL_IDENTITY_FIELDS:
            value = _norm_text(override_visual.get(field_name))
            if value:
                visual[field_name] = value
        style_hints = list(visual.get('style_hints') or [])
        for hint in _norm_text_list(override_visual.get('style_hints')):
            if hint not in style_hints:
                style_hints.append(hint)
        if style_hints:
            visual['style_hints'] = style_hints
    affect = dict(merged.get('affect_profile') or {})
    override_affect = dict(override_persona.get('affect_profile') or {})
    if override_affect:
        for field_name in AFFECT_PROFILE_FIELDS:
            value = _norm_text(override_affect.get(field_name))
            if value:
                affect[field_name] = value
    return build_persona_dict(visual_identity=visual, affect_profile=affect)


def persona_visual_identity(value: Any) -> dict[str, Any]:
    return dict(compact_persona_dict(value).get('visual_identity') or {})


def persona_affect_profile(value: Any) -> dict[str, Any]:
    return dict(compact_persona_dict(value).get('affect_profile') or {})
