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


def _candidate_payload(match: StickerMatch) -> dict[str, Any]:
    asset = match.entry.asset
    readings = [dict(reading) for reading in asset.card.get('readings', [])]
    if match.reading in readings:
        readings[readings.index(match.reading)]['retrieval_match'] = True
    optional = {
        **{key: asset.card.get(key) for key in ('caption', 'appearance', 'action', 'uncertainty')},
        'packs': list(dict.fromkeys(alias.pack for alias in asset.aliases if alias.pack)),
        'pack_descriptions': dict(match.pack_descriptions),
        'character_families': list(asset.family_ids), 'style_tags': list(asset.style_tags),
        'recently_delivered': match.recently_delivered,
        'visually_similar_deliveries': list(match.visually_similar_deliveries),
    }
    # A legacy/external match may carry an interpretation absent from the card.
    # Keep it rather than silently discarding meaningful text.
    if match.reading and match.reading not in asset.card.get('readings', []):
        optional['matched_reading'] = match.reading
    return {
        'sticker_id': asset.agent_id,
        **{key: value for key, value in optional.items() if value},
        'readings': readings,
        'animated': match.entry.animated,
    }


def _advanced_schema() -> dict[str, Any]:
    return {
        'type': 'object',
        'description': 'Specific retrieval axes and constraints when needed.',
        'properties': {
            'semantic_focus': {
                'type': 'object',
                'description': 'Meaning and conversational relationship.',
                'properties': {
                    'reaction_type': _param('string', 'Semantic cue, e.g. side-eye disbelief or bashful approval.'),
                    'reply_force': _param('string', 'Force cue, e.g. mild nudge or firm rejection.'),
                    'emotional_valence': _param('string', 'Emotional tone such as negative, warm, or mixed.'),
                    'irony_strength': _param('string', 'Irony cue such as none, light irony, or deadpan irony.'),
                    'social_stance': _param('string', 'Stance cue such as teasing, supportive, or dismissive.'),
                    'conversation_role': _param('string', 'Role cue such as reaction shot, comeback, or acknowledgement.'),
                    'relationship_fit': _param('string', 'Relationship cue such as close-friends, flirty, or formal.'),
                },
                'additionalProperties': False,
            },
            'visual_focus': {
                'type': 'object',
                'description': 'Expression and motion cues.',
                'properties': {
                    'eye_signal': _param('string', 'Eye cue, e.g. side-eye, blank stare, or sparkling eyes.'),
                    'mouth_signal': _param('string', 'Mouth cue, e.g. tight smile, flat mouth, or pout.'),
                    'motion_signal': _param('string', 'Motion cue, e.g. small shrug, head tilt, or bouncing.'),
                    'delivery_style': _param('string', 'Delivery cue, e.g. deadpan, theatrical, or understated.'),
                    'humor_style': _param('string', 'Humor cue, e.g. meme-y, deadpan irony, or playful teasing.'),
                },
                'additionalProperties': False,
            },
            'style_focus': {
                'type': 'object',
                'description': 'Style continuity.',
                'properties': {
                    'style_goal': _param('string', 'Preserve or switch style.', enum=['preserve', 'allow_switch', 'prefer_switch', 'ignore_style'], default='preserve'),
                    'style_hints': _param('array', 'Visual-family hints like rough manga line, pastel, or deadpan meme.', items={'type': 'string'}),
                    'preferred_pack': _param('string', 'Preferred pack; keeps global alternatives eligible.'),
                },
                'additionalProperties': False,
            },
            'text_constraints': {
                'type': 'object',
                'description': 'Caption meaning constraints.',
                'properties': {
                    'text_priority': _param('string', 'Caption matching priority during retrieval.', enum=['require', 'prefer', 'ignore'], default='prefer'),
                    'must_include': _param('array', 'Caption meanings to include or imply.', items={'type': 'string'}),
                    'avoid_text_meanings': _param('array', 'Caption meanings to avoid.', items={'type': 'string'}),
                },
                'additionalProperties': False,
            },
            'intensity_limits': {
                'type': 'object',
                'description': 'Limits; all intensity levels and animations are eligible by default.',
                'properties': {
                    'max_harshness': _param('integer', 'Maximum tolerated harshness on a 0-4 scale.', minimum=0, maximum=4, default=4),
                    'max_intimacy': _param('integer', 'Maximum tolerated intimacy on a 0-4 scale.', minimum=0, maximum=4, default=4),
                    'max_meme_dependence': _param('integer', 'Maximum tolerated meme dependence on a 0-4 scale.', minimum=0, maximum=4, default=4),
                    'allow_animation': _param('boolean', 'Include animations; false restricts results to static stickers.', default=True),
                },
                'additionalProperties': False,
            },
            'forbid': _param('array', 'Meanings or usages to avoid.', items={'type': 'string'}),
        },
        'additionalProperties': False,
    }


def _persona_schema() -> dict[str, Any]:
    return {
        'type': 'object',
        'description': 'Recurring visual identity and expressive preferences for this session.',
        'properties': {
            'visual_identity': {
                'type': 'object',
                'description': 'Visual identity, such as an anime catgirl or pastel animal mascot.',
                'properties': {
                    'character_archetype': _param('string', 'Character or archetype family, e.g. anime 2d catgirl or sleepy fox mascot.'),
                    'rendering_style': _param('string', 'Rendering family, e.g. flat anime sticker, rough manga, or glossy chibi.'),
                    'palette_mood': _param('string', 'Palette or visual mood, for example soft pastel, monochrome, or bright candy.'),
                    'style_hints': _param('array', 'Style-family hints.', items={'type': 'string'}),
                    'preferred_pack': _param('string', 'Preferred pack.'),
                },
                'additionalProperties': False,
            },
            'affect_profile': {
                'type': 'object',
                'description': 'Expressive preferences.',
                'properties': {
                    'default_tone': _param('string', 'Emotional baseline, for example dry amused, warm, smug, or slightly sad.'),
                    'expression_bias': _param('string', 'Face-expression bias, e.g. side-eye, deadpan blink, or pouty smile.'),
                    'pose_bias': _param('string', 'Pose or motion bias, e.g. tiny shrug, leaning in, or frozen stare.'),
                    'delivery_bias': _param('string', 'Delivery bias, e.g. understated, theatrical, or matter-of-fact.'),
                    'humor_bias': _param('string', 'Humor bias, e.g. deadpan irony or playful teasing.'),
                },
                'additionalProperties': False,
            },
        },
        'additionalProperties': False,
    }


def _selection_lens_schema() -> dict[str, Any]:
    return {
        'type': 'object',
        'description': 'Soft ranking guidance; not hard constraints.',
        'properties': {
            'social_read': _param('string', 'Social reading, e.g. gentle acknowledgement or teasing disbelief.'),
            'subtext': _param('string', 'Subtext, e.g. ironic support or playful refusal.'),
            'face_and_pose': _param('string', 'Face and pose that carry the reaction.'),
            'avoid_misread_as': _param('string', 'Misinterpretation to avoid.'),
        },
        'additionalProperties': False,
    }


class StickerQueryTool:
    def __init__(self, catalog: StickerCatalog) -> None:
        self.catalog = catalog
        self.spec = ToolSpec(
            name='sticker_query',
            description=(
                'Find stickers for the reply you intend to convey. Inspect supplied images, captions and who acts or is addressed; '
                'select a fitting sticker_id with sticker_send_selected, refine the query or use text. '
                'Readings are conditional; retrieval_match marks search affinity, not the only valid meaning. '
                'Use clear evidence despite uncertain details, without inventing relationship history or consent to justify a choice. '
                'Claim visual inspection only when images are supplied. Missing optional fields are unset; an absent caption means none was captured. '
                'The shortlist is not the entire catalog and nothing has been sent. Recent/visually similar deliveries guide variety, not bans; '
                'unsent candidates do not establish preference. Keep words coherent with the asset. '
                'Remember or clear persona only for an intended continuing change; a recipient\'s mood or taste is not your identity. '
                'Use use_once for temporary persona preferences. '
                'Leave advanced empty unless its controls help.'
            ),
            parameters_schema={
                'type': 'object',
                'properties': {
                    'intent_core': _param('string', 'Intended message to the recipient, including direction: offer comfort, request a hug, accept blame or hand it back. Use the exchange, not copied identities or profiles.'),
                    'secondary_goals': _param('array', 'Extra nuances that materially refine the reaction.', items={'type': 'string'}),
                    'reaction_tone': _param('string', 'Reaction tone, for example dry amused, warm, irritated, bashful, or smug.'),
                    'social_intent': _param('string', 'Social intent, for example reassure, lightly tease, acknowledge, celebrate, or dismiss.'),
                    'expression_cue': _param('string', 'Face or pose cue, e.g. side-eye, blank stare, pout, or tiny shrug.'),
                    'caption_meaning': _param('string', 'Caption or overlay meaning hint when visible text matters.'),
                    'preferred_pack': _param('string', 'Preferred pack; keeps global alternatives eligible.'),
                    'diversity_preference': _param('string', 'Prefer fresh variants or use normal ranking.', enum=['default', 'prefer_fresh_variant'], default='default'),
                    'allow_animation': _param('boolean', 'Include animations; false restricts results to static stickers.', default=True),
                    'candidate_budget': _param('integer', 'How many candidates to inspect.', minimum=1, maximum=catalog.config.max_candidates, default=catalog.config.candidate_count),
                    'required_pack': _param('string', 'Require this exact returned pack.'),
                    'required_character_family': _param('string', 'Require this cataloged character family; a pack alone does not prove identity.'),
                    'preferred_character_family': _param('string', 'Preferred cataloged character family, alongside global candidates.'),
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
            plan = StickerRetrievalPlan.from_payload(args, config=self.catalog.config)
            state, persona = await self.catalog.aprepare_query_context(plan=plan, session_id=ctx.session_id,
                persist_persona=True, expected_scope=ctx.scope)
            matches = await self.catalog.achoose(plan=plan, session_id=ctx.session_id,
                session_state=state, persona_context=persona)
            channels = {channel for match in matches for channel in match.channels}
            retrieval = ('semantic' if channels - {'literal', 'asset_id'} else
                'asset_id' if 'asset_id' in channels else 'literal' if 'literal' in channels else 'none')
            constraints = {key: value for key, value in {
                'must_include': plan.text_constraints.must_include,
                'avoid_text_meanings': plan.text_constraints.avoid_text_meanings,
                'avoid_misread_as': plan.selection_lens.avoid_misread_as,
                'forbid': plan.forbid,
                'text_priority': plan.text_priority,
                'style_goal': plan.style_goal,
                'diversity_preference': plan.diversity_preference,
            }.items() if value}
            scope = {key: value for key, value in {
                'retrieval': retrieval, 'required_pack': plan.required_pack,
                'required_character_family': plan.required_character_family,
                'intensity_limits': plan.intensity_limits.as_dict(),
            }.items() if value}
            return ToolResult(call_id='', name=self.spec.name, output={
                'ok': True, 'status': 'candidates' if matches else 'no_candidates',
                'intent': next((match.entry.agent_id for match in matches
                                if match.entry.sticker_id == plan.intent_core), plan.intent_core),
                'constraints': constraints,
                'search_scope': scope,
                **({'persona': persona['effective_persona']} if persona['effective_persona'] else {}),
                'candidates': [_candidate_payload(match) for match in matches],
            }, evidence_parts=await self.catalog.evidence(matches))
        except Exception as exc:
            logger.exception('sticker_query failed')
            return ToolResult(call_id='', name=self.spec.name, output={'ok': False, 'error': f'{exc.__class__.__name__}: {exc}'})


class StickerSendSelectedTool:
    def __init__(self, catalog: StickerCatalog) -> None:
        self.catalog = catalog
        self.spec = ToolSpec(
            name='sticker_send_selected',
            description='Select an exact known sticker for delivery. after_final dispatches when the turn ends, even with no final text; no second selection is needed. Queued is not confirmed delivery; inspect the receipt. Do not blindly repeat an unknown delivery or substitute for an unavailable original.',
            parameters_schema={
                'type': 'object',
                'properties': {
                    'selected_sticker_id': _param('string', 'Exact sticker_id returned by sticker_query.'),
                    'delivery_timing': _param('string', 'send_now and before_final send immediately; after_final dispatches at turn completion, with or without text.', enum=['send_now', 'after_final', 'before_final']),
                },
                'required': ['selected_sticker_id'],
                'additionalProperties': False,
            },
            runner=self,
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            sticker_id = str(args.get('selected_sticker_id', args.get('sticker_id', '')) or '').strip()
            if not sticker_id:
                return ToolResult(call_id='', name=self.spec.name, output={'ok': False, 'error': 'Empty selected_sticker_id'})
            timing_raw = str(args.get('delivery_timing', args.get('timing', 'after_final')) or 'after_final').strip().lower()
            timing = StickerTiming.parse(timing_raw)
            entry = await self.catalog.aget_available(sticker_id)
            if entry is None:
                reference = self.catalog.agent_sticker_id(sticker_id)
                return ToolResult(call_id='', name=self.spec.name, output={'ok': False,
                    'error': 'Selected sticker is unknown or its original bytes are unavailable; no substitute sent',
                    **({'sticker_id': reference} if reference else {})})
            sticker = OutboundSticker(path=entry.absolute_path, emoji=entry.emoji, timing=timing,
                label=entry.summary, source_id=entry.sticker_id, content_sha256=entry.asset.content_hash)
            return ToolResult(call_id='', name=self.spec.name, output={
                'ok': True, 'status': 'queued', 'sticker_id': entry.agent_id,
                'delivery_timing': timing.value,
                'caption': entry.asset.card.get('caption', ''), 'action': entry.asset.card.get('action', ''),
            }, stickers=[sticker])
        except Exception as exc:
            logger.exception('sticker_send_selected failed')
            return ToolResult(call_id='', name=self.spec.name, output={'ok': False, 'error': f'{exc.__class__.__name__}: {exc}'})
