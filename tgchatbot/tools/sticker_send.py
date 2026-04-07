from __future__ import annotations

import logging
from typing import Any

from tgchatbot.domain.models import OutboundSticker, StickerTiming, ToolResult
from tgchatbot.stickers.catalog import StickerCatalog, StickerMatch
from tgchatbot.stickers.plan import StickerRetrievalPlan
from tgchatbot.tools.base import ToolContext, ToolSpec

logger = logging.getLogger(__name__)


def _param(type_: str | list[str], description: str, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {'type': type_, 'description': description}
    payload.update(extra)
    return payload


def _round_number(value: Any) -> Any:
    return round(value, 4) if isinstance(value, float) else value


def _round_mapping(values: dict[str, Any]) -> dict[str, Any]:
    return {key: _round_number(value) for key, value in values.items()}


def _compact_text(value: Any, *, limit: int = 160) -> str:
    text = ' '.join(str(value or '').split()).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + '...'


def _semantic_summary(entry: Any) -> str:
    return _compact_text(
        entry.sticker_card.get('fused_pragmatic_meaning')
        or entry.sticker_semantic_text
        or entry.caption_semantic_text
        or entry.summary
    )


def _style_summary(entry: Any) -> str:
    return _compact_text(entry.style_text or '')


def _caption_summary(entry: Any) -> str:
    return _compact_text(entry.caption_meaning_en or entry.caption_meaning_zh or entry.source_overlay_text_normalized or '')


def _compact_entry_payload(entry: Any) -> dict[str, Any]:
    return {
        'sticker_id': entry.sticker_id,
        'summary': entry.summary,
        'source_pack_id': entry.source_pack_id,
        'style_cluster': entry.style_cluster,
        'style_summary': _style_summary(entry),
        'semantic_summary': _semantic_summary(entry),
        'caption_summary': _caption_summary(entry),
        'animated': entry.animated,
        'emoji': entry.emoji,
    }


def _first_requested_phrase(profile: dict[str, Any], fields: tuple[str, ...]) -> str:
    requested = dict(profile.get('requested') or {})
    for field_name in fields:
        if requested.get(field_name):
            return str(requested[field_name])
    details = dict(profile.get('details') or {})
    for field_name in fields:
        detail = dict(details.get(field_name) or {})
        requested_value = str(detail.get('requested') or '').strip()
        if requested_value:
            return requested_value
    return ''


def _candidate_fit_signals(match: StickerMatch) -> list[str]:
    signals: list[str] = []
    message_intent = match.match_profile.get('message_intent', {})
    intent_lens = dict(message_intent.get('selection_lens') or {})
    if intent_lens.get('social_read', {}).get('requested'):
        signals.append('social read: ' + str(intent_lens['social_read']['requested']))
    if intent_lens.get('subtext', {}).get('requested'):
        signals.append('subtext: ' + str(intent_lens['subtext']['requested']))
    simple_profile = match.match_profile.get('simple_hints', {})
    simple_requested = simple_profile.get('requested', {})
    for field_name in simple_profile.get('matched', [])[:2]:
        if simple_requested.get(field_name):
            signals.append(str(simple_requested[field_name]))
    semantic_profile = match.match_profile.get('semantic_axes', {})
    semantic_requested = semantic_profile.get('requested', {})
    for field_name in semantic_profile.get('matched', [])[:2]:
        if semantic_requested.get(field_name):
            signals.append(str(semantic_requested[field_name]))
    visual_profile = match.match_profile.get('visual_cues', {})
    visual_requested = visual_profile.get('requested', {})
    for field_name in visual_profile.get('matched', [])[:2]:
        if visual_requested.get(field_name):
            signals.append(str(visual_requested[field_name]))
    text_fit = match.match_profile.get('text_fit', {})
    if text_fit.get('matched_must_include'):
        signals.append('caption supports: ' + ', '.join(text_fit['matched_must_include'][:2]))
    elif text_fit.get('has_visible_text'):
        signals.append('visible caption meaning available')
    style_relation = match.match_profile.get('style_relation', {})
    if style_relation.get('preferred_pack_match') == 'exact_preferred_pack':
        signals.append(f"preferred pack: {style_relation.get('preferred_pack_requested')}")
    elif style_relation.get('preferred_pack_match') in {'close_preferred_pack', 'related_preferred_pack', 'loosely_related_preferred_pack'}:
        signals.append(f"near preferred pack: {style_relation.get('preferred_pack_requested')}")
    if style_relation.get('preferred_cluster_match') == 'exact_preferred_cluster':
        signals.append(f"preferred cluster: {style_relation.get('preferred_cluster_requested')}")
    elif style_relation.get('preferred_cluster_match') in {'related_preferred_cluster', 'loosely_related_preferred_cluster'}:
        signals.append(f"near preferred cluster: {style_relation.get('preferred_cluster_requested')}")
    continuity = match.match_profile.get('continuity', {})
    if continuity.get('session_persona_used'):
        signals.append('aligned with stored sticker persona')
    elif continuity.get('recent_implicit_used'):
        signals.append('aligned with recent sticker continuity')
    return signals[:6]


def _candidate_warnings(match: StickerMatch) -> list[str]:
    warnings = list(match.match_profile.get('hard_mismatches', []))
    diversity = match.match_profile.get('diversity_relation', {})
    labels = diversity.get('labels') or []
    if 'recently_sent_exact_sticker' in labels:
        warnings.append('same sticker was sent recently')
    elif 'recently_surfaced_exact_sticker' in labels:
        warnings.append('same sticker already appeared in a recent shortlist')
    return warnings[:6]


def _candidate_payload(match: StickerMatch) -> dict[str, Any]:
    diversity_relation = match.match_profile.get('diversity_relation', {})
    continuity = match.match_profile.get('continuity', {})
    message_intent = match.match_profile.get('message_intent', {})
    affect = match.match_profile.get('affect', {})
    visual_identity = match.match_profile.get('visual_identity', {})
    return {
        **_compact_entry_payload(match.entry),
        'selection_summary': match.selection_summary,
        'social_read': _first_requested_phrase(message_intent.get('selection_lens', {}), ('social_read', 'subtext')) or str(match.entry.sticker_card.get('fused_pragmatic_meaning') or match.entry.summary),
        'persona_fit': _first_requested_phrase(visual_identity.get('persona_visual_identity', {}), ('character_archetype', 'rendering_style', 'palette_mood')) or continuity.get('note', ''),
        'expression_fit': _first_requested_phrase(affect.get('persona_affect', {}), ('default_tone', 'expression_bias', 'pose_bias', 'delivery_bias', 'humor_bias')) or _first_requested_phrase(affect.get('visual_axes', {}), ('eye_signal', 'mouth_signal', 'motion_signal', 'delivery_style', 'humor_style')),
        'continuity_note': str(continuity.get('note') or ''),
        'fit_signals': _candidate_fit_signals(match),
        'warnings': _candidate_warnings(match),
        'score': round(match.score, 4),
        'debug': {
            'score_breakdown': _round_mapping(match.score_breakdown),
            'match_profile': match.match_profile,
            'diversity_relation': diversity_relation,
        },
    }


def _send_selection_summary(entry: Any, style_context_after_send: dict[str, Any]) -> str:
    tone = str(entry.sticker_card.get('fused_pragmatic_meaning') or entry.summary or 'Sticker selected').strip().rstrip('.')
    current_cluster = style_context_after_send.get('current_style_cluster')
    if current_cluster:
        return f'{tone}. Recorded into style memory under cluster {current_cluster}.'
    return tone + '.'


def _advanced_schema() -> dict[str, Any]:
    return {
        'type': 'object',
        'description': 'Optional advanced overrides when the bot knows exactly which axes or constraints matter.',
        'properties': {
            'semantic_focus': {
                'type': 'object',
                'description': 'Advanced semantic axes such as silent disbelief, firm rejection, teasing, or reassurance.',
                'properties': {
                    'reaction_type': _param('string', 'Advanced semantic cue, for example side-eye disbelief or bashful approval.'),
                    'reply_force': _param('string', 'Advanced force cue, for example mild nudge or firm rejection.'),
                    'emotional_valence': _param('string', 'Advanced emotional tone such as negative, warm, or mixed.'),
                    'irony_strength': _param('string', 'Advanced irony cue such as none, light irony, or deadpan irony.'),
                    'social_stance': _param('string', 'Advanced stance cue such as teasing, supportive, or dismissive.'),
                    'conversation_role': _param('string', 'Advanced role cue such as reaction shot, comeback, or acknowledgement.'),
                    'relationship_fit': _param('string', 'Advanced relationship cue such as close-friends, flirty, or formal.'),
                },
                'additionalProperties': False,
            },
            'visual_focus': {
                'type': 'object',
                'description': 'Advanced subtle visual cues when eye/mouth/motion details materially matter.',
                'properties': {
                    'eye_signal': _param('string', 'Advanced eye cue, for example side-eye, blank stare, or sparkling eyes.'),
                    'mouth_signal': _param('string', 'Advanced mouth cue, for example tight smile, flat mouth, or pout.'),
                    'motion_signal': _param('string', 'Advanced motion cue, for example small shrug, head tilt, or bouncing.'),
                    'delivery_style': _param('string', 'Advanced delivery cue, for example deadpan, theatrical, or understated.'),
                    'humor_style': _param('string', 'Advanced humor cue, for example meme-y, deadpan irony, or playful teasing.'),
                },
                'additionalProperties': False,
            },
            'style_focus': {
                'type': 'object',
                'description': 'Advanced style continuity controls.',
                'properties': {
                    'style_goal': _param('string', 'How strongly to preserve or switch style families.', enum=['preserve', 'allow_switch', 'prefer_switch', 'ignore_style'], default='preserve'),
                    'style_hints': _param('array', 'Optional visual-family hints like rough manga line, pastel, or deadpan meme.', items={'type': 'string'}),
                    'preferred_pack': _param('string', 'Optional pack family or source pack id to preserve or stay near.'),
                    'preferred_style_cluster': _param('string', 'Optional style cluster id to preserve or stay near.'),
                },
                'additionalProperties': False,
            },
            'text_constraints': {
                'type': 'object',
                'description': 'Advanced caption meaning constraints.',
                'properties': {
                    'text_priority': _param('string', 'How strongly visible overlay text should control meaning.', enum=['require', 'prefer', 'ignore'], default='prefer'),
                    'must_include': _param('array', 'Caption meanings that should be present or strongly implied.', items={'type': 'string'}),
                    'avoid_text_meanings': _param('array', 'Caption meanings to avoid.', items={'type': 'string'}),
                },
                'additionalProperties': False,
            },
            'intensity_limits': {
                'type': 'object',
                'description': 'Advanced intensity caps.',
                'properties': {
                    'max_harshness': _param('integer', 'Maximum tolerated harshness on a 0-4 scale.', minimum=0, maximum=4, default=3),
                    'max_intimacy': _param('integer', 'Maximum tolerated intimacy on a 0-4 scale.', minimum=0, maximum=4, default=4),
                    'max_meme_dependence': _param('integer', 'Maximum tolerated meme dependence on a 0-4 scale.', minimum=0, maximum=4, default=4),
                    'allow_animation': _param('boolean', 'Allow animated stickers if they fit better.', default=False),
                },
                'additionalProperties': False,
            },
            'forbid': _param('array', 'Advanced meanings or usages to avoid.', items={'type': 'string'}),
        },
        'additionalProperties': False,
    }


def _persona_schema() -> dict[str, Any]:
    return {
        'type': 'object',
        'description': 'Optional persistent sticker persona for this session when the bot wants a recurring visual family or expressive bias.',
        'properties': {
            'visual_identity': {
                'type': 'object',
                'description': 'Persistent art-family identity such as anime catgirl, rough manga render, or soft pastel creature family.',
                'properties': {
                    'character_archetype': _param('string', 'Persistent character or archetype family, for example anime 2d catgirl or sleepy fox mascot.'),
                    'rendering_style': _param('string', 'Persistent rendering family, for example flat anime sticker, rough manga, or glossy chibi.'),
                    'palette_mood': _param('string', 'Persistent palette or visual mood, for example soft pastel, monochrome, or bright candy.'),
                    'style_hints': _param('array', 'Persistent style-family hints.', items={'type': 'string'}),
                    'preferred_pack': _param('string', 'Persistent pack family or source pack id to preserve or stay near.'),
                    'preferred_style_cluster': _param('string', 'Persistent style cluster id to preserve or stay near.'),
                },
                'additionalProperties': False,
            },
            'affect_profile': {
                'type': 'object',
                'description': 'Persistent expressive bias such as default tone, face read, pose tendency, delivery, or humor style.',
                'properties': {
                    'default_tone': _param('string', 'Persistent emotional baseline, for example dry amused, warm, smug, or slightly sad.'),
                    'expression_bias': _param('string', 'Persistent face-expression bias, for example side-eye, deadpan blink, or pouty smile.'),
                    'pose_bias': _param('string', 'Persistent pose or motion bias, for example tiny shrug, leaning in, or frozen stare.'),
                    'delivery_bias': _param('string', 'Persistent delivery bias, for example understated, theatrical, or matter-of-fact.'),
                    'humor_bias': _param('string', 'Persistent humor bias, for example deadpan irony or playful teasing.'),
                },
                'additionalProperties': False,
            },
        },
        'additionalProperties': False,
    }


def _selection_lens_schema() -> dict[str, Any]:
    return {
        'type': 'object',
        'description': 'Optional soft guidance for how to think about subtle human sticker choice. These fields guide ranking softly rather than acting as hard constraints.',
        'properties': {
            'social_read': _param('string', 'How the sticker should read socially, for example gentle acknowledgement, teasing disbelief, or cool detachment.'),
            'subtext': _param('string', 'Hidden meaning or implication that should come through, for example ironic support or not-actually-angry refusal.'),
            'face_and_pose': _param('string', 'Precise face, mouth, eyes, and pose read that should carry the reaction.'),
            'continuity_note': _param('string', 'Soft note about how close or different this should feel relative to recent stickers or the current persona.'),
            'avoid_misread_as': _param('string', 'Meaning or mood that this should not accidentally read as.'),
        },
        'additionalProperties': False,
    }


class StickerQueryTool:
    def __init__(self, catalog: StickerCatalog) -> None:
        self.catalog = catalog
        self.spec = ToolSpec(
            name='sticker_query',
            description=(
                'Query the sticker system with a simple-first plan. Always provide intent_core. '
                'Only add one to three helper hints when they clearly matter: reaction_tone, social_intent, expression_cue, caption_meaning, preferred_pack, or preferred_style_cluster. '
                'Use diversity_preference=prefer_fresh_variant only when you want a slightly fresher variant than recent similar queries. '
                'Use persona when the sticker should keep a recurring visual or expressive identity across the session. '
                'Use selection_lens when subtle subtext, face, pose, or social read matters. '
                'Leave advanced empty unless you intentionally want axis-level control. Do not put usernames, bot names, or transport metadata into semantic fields.'
            ),
            parameters_schema={
                'type': 'object',
                'properties': {
                    'send': _param('boolean', 'Whether the bot actually wants to send a sticker after this query.', default=True),
                    'intent_core': _param('string', 'Required core reaction meaning in plain language, for example dry amused refusal or warm supportive acknowledgement.'),
                    'secondary_goals': _param('array', 'Optional extra nuances that materially refine the reaction.', items={'type': 'string'}),
                    'reaction_tone': _param('string', 'Optional simple reaction tone, for example dry amused, warm, irritated, bashful, or smug.'),
                    'social_intent': _param('string', 'Optional simple social intent, for example reassure, lightly tease, acknowledge, celebrate, or dismiss.'),
                    'expression_cue': _param('string', 'Optional simple face or pose cue, for example side-eye, blank stare, pout, or tiny shrug.'),
                    'caption_meaning': _param('string', 'Optional caption or overlay meaning hint when visible text matters.'),
                    'preferred_pack': _param('string', 'Optional pack family or source pack id to preserve or stay near.'),
                    'preferred_style_cluster': _param('string', 'Optional style cluster id to preserve or stay near.'),
                    'diversity_preference': _param('string', 'Whether to keep normal ranking or slightly prefer fresher variants than very recent ones.', enum=['default', 'prefer_fresh_variant'], default='default'),
                    'allow_animation': _param('boolean', 'Allow animated stickers if they fit better.', default=False),
                    'candidate_budget': _param('integer', 'How many ranked candidates to return.', minimum=1, maximum=8, default=5),
                    'persona': _persona_schema(),
                    'persona_mode': _param('string', 'How to use the optional persona for this query.', enum=['inherit', 'merge_and_remember', 'use_once', 'clear_session_persona']),
                    'selection_lens': _selection_lens_schema(),
                    'advanced': _advanced_schema(),
                },
                'required': ['intent_core'],
                'additionalProperties': False,
            },
            runner=self,
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            if not self.catalog.loaded:
                self.catalog.load()
            plan = StickerRetrievalPlan.from_payload(args)
            query_understanding = plan.query_interpretation()
            style_context = self.catalog.describe_style_context(ctx.session_id)
            if not plan.send:
                _, persona_context = self.catalog.prepare_query_context(plan=plan, session_id=ctx.session_id, persist_persona=False)
                return ToolResult(
                    call_id='',
                    name=self.spec.name,
                    output={
                        'ok': True,
                        'skipped': True,
                        'reason': 'send=false',
                        'query_understanding': query_understanding,
                        'style_context': style_context,
                        'persona_context': persona_context,
                        'field_warnings': list(plan.field_warnings),
                        'dropped_noise_terms': list(plan.dropped_noise_terms),
                    },
                )
            session_state, persona_context = self.catalog.prepare_query_context(plan=plan, session_id=ctx.session_id, persist_persona=True)
            matches = self.catalog.choose(plan=plan, session_id=ctx.session_id, session_state=session_state, persona_context=persona_context)
            if not matches:
                return ToolResult(
                    call_id='',
                    name=self.spec.name,
                    output={
                        'ok': False,
                        'error': 'No sticker matched the requested intent',
                        'query_understanding': query_understanding,
                        'style_context': style_context,
                        'persona_context': persona_context,
                        'field_warnings': list(plan.field_warnings),
                        'dropped_noise_terms': list(plan.dropped_noise_terms),
                        'likely_culprit_fields': plan.likely_culprit_fields(),
                        'retry_suggestion': 'Retry with intent_core plus at most one or two helper hints, or remove the highlighted advanced constraints.',
                    },
                )
            return ToolResult(
                call_id='',
                name=self.spec.name,
                output={
                    'ok': True,
                    'query_understanding': query_understanding,
                    'style_context': style_context,
                    'persona_context': persona_context,
                    'field_warnings': list(plan.field_warnings),
                    'dropped_noise_terms': list(plan.dropped_noise_terms),
                    'candidate_count': len(matches),
                    'candidates': [_candidate_payload(match) for match in matches],
                    'selection_guidance': 'Pick one candidate sticker_id using selection_summary, social_read, expression_fit, persona_fit, continuity_note, fit_signals, and warnings first; then pass it as selected_sticker_id to sticker_send_selected. Inspect candidate.debug.score_breakdown only when you need to compare close variants.',
                },
            )
        except Exception as exc:
            logger.exception('sticker_query failed')
            return ToolResult(call_id='', name=self.spec.name, output={'ok': False, 'error': f'{exc.__class__.__name__}: {exc}'})


class StickerSendSelectedTool:
    def __init__(self, catalog: StickerCatalog) -> None:
        self.catalog = catalog
        self.spec = ToolSpec(
            name='sticker_send_selected',
            description='Send one previously chosen sticker from the local sticker library and echo back the compact sticker metadata.',
            parameters_schema={
                'type': 'object',
                'properties': {
                    'selected_sticker_id': _param('string', 'Exact sticker_id returned by sticker_query.'),
                    'delivery_timing': _param('string', 'Whether to send the sticker immediately now or after the final text.', enum=['send_now', 'after_final', 'before_final']),
                },
                'required': ['selected_sticker_id'],
                'additionalProperties': False,
            },
            runner=self,
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            if not self.catalog.loaded:
                self.catalog.load()
            sticker_id = str(args.get('selected_sticker_id', args.get('sticker_id', '')) or '').strip()
            if not sticker_id:
                return ToolResult(call_id='', name=self.spec.name, output={'ok': False, 'error': 'Empty selected_sticker_id'})
            timing_raw = str(args.get('delivery_timing', args.get('timing', 'after_final')) or 'after_final').strip().lower()
            timing = StickerTiming.parse(timing_raw)
            entry = self.catalog.get_by_sticker_id(sticker_id)
            if entry is None:
                return ToolResult(call_id='', name=self.spec.name, output={'ok': False, 'error': 'Unknown sticker_id', 'sticker_id': sticker_id})
            self.catalog.record_selection(session_id=ctx.session_id, sticker_id=entry.sticker_id)
            style_context_after_send = self.catalog.describe_style_context(ctx.session_id)
            persona_context_after_send = self.catalog.describe_persona_context(ctx.session_id)
            sticker = OutboundSticker(
                path=entry.absolute_path,
                emoji=entry.emoji,
                timing=timing,
                label=entry.summary or entry.sticker_id,
                source_id=entry.sticker_id,
            )
            return ToolResult(
                call_id='',
                name=self.spec.name,
                output={
                    'ok': True,
                    'status': 'success',
                    'delivery_timing': timing.value,
                    'timing': timing.value,
                    'selection_summary': _send_selection_summary(entry, style_context_after_send),
                    'style_context_after_send': style_context_after_send,
                    'persona_context_after_send': persona_context_after_send,
                    'social_read': str(entry.sticker_card.get('fused_pragmatic_meaning') or entry.summary),
                    **_compact_entry_payload(entry),
                },
                stickers=[sticker],
            )
        except Exception as exc:
            logger.exception('sticker_send_selected failed')
            return ToolResult(call_id='', name=self.spec.name, output={'ok': False, 'error': f'{exc.__class__.__name__}: {exc}'})
