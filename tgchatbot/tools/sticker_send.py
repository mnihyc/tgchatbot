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
    return {
        'sticker_id': asset.asset_id,
        'caption': asset.card.get('caption', ''),
        'appearance': asset.card.get('appearance', ''),
        'action': asset.card.get('action', ''),
        'readings': asset.card.get('readings', []),
        'uncertainty': asset.card.get('uncertainty', ''),
        'matched_reading': match.reading,
        'packs': list(dict.fromkeys(alias.pack for alias in asset.aliases)),
        'character_families': list(asset.family_ids),
        'style_tags': list(asset.style_tags),
        'animated': match.entry.animated,
        'appearance_embedding_source': (asset.provenance.get('visual_embedding_source') or (
            'image' if asset.provenance.get('image_input') == 'sampled-frames-only-v1' else None))
            if asset.image_vector is not None else None,
        'recently_delivered': match.recently_delivered,
        'visually_similar_deliveries': list(match.visually_similar_deliveries),
    }


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
                'Find stickers for the reply you intend to express. Put the social move in intent_core and add only useful tone, caption, expression or style constraints. '
                'Candidates have not been sent. Inspect available images and captions, then select, refine the query or reply without a sticker. '
                'Use diversity_preference=prefer_fresh_variant to prefer alternatives to recent deliveries. '
                'Change persistent persona only for an intended continuing preference. Leave advanced empty unless you need its specific controls.'
            ),
            parameters_schema={
                'type': 'object',
                'properties': {
                    'send': _param('boolean', 'false skips the query and preference changes entirely; true inspects candidates without sending them.', default=True),
                    'intent_core': _param('string', 'What this reply should convey to its recipient. Include direction when it matters: offering comfort, asking for a hug, accepting blame or playfully handing it back. Use the actual exchange, without copying identities or whole profiles.'),
                    'secondary_goals': _param('array', 'Optional extra nuances that materially refine the reaction.', items={'type': 'string'}),
                    'reaction_tone': _param('string', 'Optional simple reaction tone, for example dry amused, warm, irritated, bashful, or smug.'),
                    'social_intent': _param('string', 'Optional simple social intent, for example reassure, lightly tease, acknowledge, celebrate, or dismiss.'),
                    'expression_cue': _param('string', 'Optional simple face or pose cue, for example side-eye, blank stare, pout, or tiny shrug.'),
                    'caption_meaning': _param('string', 'Optional caption or overlay meaning hint when visible text matters.'),
                    'preferred_pack': _param('string', 'Optional pack family or source pack id to preserve or stay near.'),
                    'diversity_preference': _param('string', 'Whether to keep normal ranking or slightly prefer fresher variants than very recent ones.', enum=['default', 'prefer_fresh_variant'], default='default'),
                    'allow_animation': _param('boolean', 'Allow animated stickers if they fit better.', default=False),
                    'candidate_budget': _param('integer', 'How many candidates to inspect.', minimum=1, maximum=catalog.config.max_candidates, default=catalog.config.candidate_count),
                    'required_pack': _param('string', 'Restrict to this exact returned pack identifier only when the request requires it.'),
                    'required_character_family': _param('string', 'Restrict to an explicitly cataloged character family when required; pack membership alone does not prove identity.'),
                    'preferred_character_family': _param('string', 'Soft preference for a cataloged character family, alongside global candidates.'),
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
            # A skip does no retrieval, media preparation or persona writes.
            if args.get('send') is False or str(args.get('send', '')).lower() in {'false', '0', 'no', 'off'}:
                return ToolResult(call_id='', name=self.spec.name,
                                  output={'ok': True, 'skipped': True, 'reason': 'send=false'})
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
            return ToolResult(call_id='', name=self.spec.name, output={
                'ok': True, 'status': 'candidates' if matches else 'no_candidates',
                'catalog_revision': matches[0].entry.revision_id if matches else self.catalog.stats()['revision'],
                'intent': plan.intent_core, 'constraints': constraints,
                'search_scope': {'retrieval': retrieval, 'required_pack': plan.required_pack or None,
                    'required_character_family': plan.required_character_family or None,
                    'allow_animation': plan.allow_animation,
                    'intensity_limits': plan.intensity_limits.as_dict()},
                'persona': persona['effective_persona'],
                'recent_deliveries': list(state.recent_sticker_ids),
                'candidate_count': len(matches),
                'candidates': [_candidate_payload(match) for match in matches],
                'guidance': 'Inspect supplied images and captions as part of the complete reply: who acts, who is addressed, and what the words and image convey together. '
                    'Literal captions constrain meaning; descriptions are conditional interpretations, not proof of fit. '
                    'Do not claim visual inspection without images or invent relationship consent or history to rescue a choice. '
                    'An uncertain detail need not invalidate what is clear. This shortlist is not the entire catalog, and no sticker has been sent. '
                    'Select a fitting sticker_id with sticker_send_selected, refine the query, or use text. '
                    'Recent and visually similar deliveries guide variety, not repeat bans; unsent candidates do not establish preference.',
            }, evidence_parts=await self.catalog.evidence(matches))
        except Exception as exc:
            logger.exception('sticker_query failed')
            return ToolResult(call_id='', name=self.spec.name, output={'ok': False, 'error': f'{exc.__class__.__name__}: {exc}'})


class StickerSendSelectedTool:
    def __init__(self, catalog: StickerCatalog) -> None:
        self.catalog = catalog
        self.spec = ToolSpec(
            name='sticker_send_selected',
            description='Select a known sticker for delivery at the requested timing. A queued selection is not confirmed delivery; inspect the actual receipt.',
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
            sticker_id = str(args.get('selected_sticker_id', args.get('sticker_id', '')) or '').strip()
            if not sticker_id:
                return ToolResult(call_id='', name=self.spec.name, output={'ok': False, 'error': 'Empty selected_sticker_id'})
            timing_raw = str(args.get('delivery_timing', args.get('timing', 'after_final')) or 'after_final').strip().lower()
            timing = StickerTiming.parse(timing_raw)
            entry = await self.catalog.aget_available(sticker_id)
            if entry is None:
                return ToolResult(call_id='', name=self.spec.name, output={'ok': False,
                    'error': 'Selected sticker is unknown or its original bytes are unavailable; no substitute sent',
                    'sticker_id': sticker_id})
            sticker = OutboundSticker(path=entry.absolute_path, emoji=entry.emoji, timing=timing,
                label=entry.summary, source_id=entry.sticker_id, content_sha256=entry.asset.content_hash)
            return ToolResult(call_id='', name=self.spec.name, output={
                'ok': True, 'status': 'queued', 'sticker_id': entry.sticker_id,
                'catalog_revision': entry.revision_id, 'delivery_timing': timing.value,
                'caption': entry.asset.card.get('caption', ''), 'action': entry.asset.card.get('action', ''),
            }, stickers=[sticker])
        except Exception as exc:
            logger.exception('sticker_send_selected failed')
            return ToolResult(call_id='', name=self.spec.name, output={'ok': False, 'error': f'{exc.__class__.__name__}: {exc}'})
