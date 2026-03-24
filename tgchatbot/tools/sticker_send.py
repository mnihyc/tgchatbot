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


def _entry_payload(entry: Any) -> dict[str, Any]:
    return {
        'sticker_id': entry.sticker_id,
        'relative_path': entry.relative_path,
        'source_pack_id': entry.source_pack_id,
        'source_format': entry.source_format,
        'animated': entry.animated,
        'emoji': entry.emoji,
        'summary': entry.summary,
        'preview_text': entry.preview_text,
        'selection_notes': entry.selection_notes,
        'caption_mode': entry.caption_mode,
        'source_overlay_text': entry.source_overlay_text,
        'caption_meaning_en': entry.caption_meaning_en,
        'caption_meaning_zh': entry.caption_meaning_zh,
        'caption_semantic_text': entry.caption_semantic_text,
        'sticker_semantic_text': entry.sticker_semantic_text,
        'style_text': entry.style_text,
        'style_cluster': entry.style_cluster,
        'semantic_signature': entry.semantic_signature,
        'source_ocr_confidence': round(entry.source_ocr_confidence, 4),
        'caption_dominance_score': entry.caption_dominance_score,
        'harshness_level': entry.harshness_level,
        'intimacy_level': entry.intimacy_level,
        'meme_dependence_level': entry.meme_dependence_level,
        'caption_card': entry.caption_card,
        'subtle_cue_card': entry.subtle_cue_card,
        'sticker_card': entry.sticker_card,
        'style_card': entry.style_card,
    }


def _candidate_payload(match: StickerMatch) -> dict[str, Any]:
    payload = _entry_payload(match.entry)
    payload.update(
        {
            'matched_terms': match.matched_terms,
            'reasons': _round_mapping(match.reasons),
            'base_score': round(match.base_score, 4),
            'score_breakdown': _round_mapping(match.score_breakdown),
            'match_profile': match.match_profile,
            'selection_summary': match.selection_summary,
            'score': round(match.score, 4),
        }
    )
    return payload


def _send_selection_summary(entry: Any, style_context_after_send: dict[str, Any]) -> str:
    tone = str(entry.sticker_card.get('fused_pragmatic_meaning') or entry.summary or entry.preview_text or 'Sticker selected').strip().rstrip('.')
    current_cluster = style_context_after_send.get('current_style_cluster')
    if current_cluster:
        return f'{tone}. Recorded into style memory under cluster {current_cluster}.'
    return tone + '.'


class StickerQueryTool:
    def __init__(self, catalog: StickerCatalog) -> None:
        self.catalog = catalog
        self.spec = ToolSpec(
            name='sticker_query',
            description=(
                'Query the sticker system with an explicit hybrid plan. Describe the intended social act in intent_core, '
                'add semantic_focus and visual_focus when you care about subtle cues, use style_focus.style_goal to control continuity, '
                'style_focus.prefer_pack to lean toward a specific pack family, and style_focus.prefer_cluster to lean toward one style cluster or a closely related cluster family, '
                'text_constraints for caption meaning, intensity_limits for harshness/intimacy/meme tolerance, then inspect score_breakdown and match_profile before sending.'
            ),
            parameters_schema={
                'type': 'object',
                'properties': {
                    'send': _param('boolean', 'Whether the bot actually wants to send a sticker after this query.', default=True),
                    'intent_core': _param('string', 'Primary sticker intent in plain language. Focus on social meaning, not brittle tags.'),
                    'secondary_goals': _param('array', 'Optional supporting goals or nuances.', items={'type': 'string'}),
                    'semantic_focus': {
                        'type': 'object',
                        'description': 'Optional semantic axes to steer subtle pragmatic meaning.',
                        'properties': {
                            'reaction_type': _param('string', 'Requested reaction type.'),
                            'reply_force': _param('string', 'Requested reply force.'),
                            'emotional_valence': _param('string', 'Requested emotional valence.'),
                            'irony_strength': _param('string', 'Requested irony strength.'),
                            'social_stance': _param('string', 'Requested social stance.'),
                            'conversation_role': _param('string', 'Requested conversation role.'),
                            'relationship_fit': _param('string', 'Requested relationship fit.'),
                        },
                        'additionalProperties': False,
                    },
                    'visual_focus': {
                        'type': 'object',
                        'description': 'Optional visual cue constraints for subtle expression and delivery.',
                        'properties': {
                            'eye_signal': _param('string', 'Requested eye cue.'),
                            'mouth_signal': _param('string', 'Requested mouth cue.'),
                            'motion_signal': _param('string', 'Requested motion cue.'),
                            'delivery_style': _param('string', 'Requested delivery style.'),
                            'humor_style': _param('string', 'Requested humor style.'),
                        },
                        'additionalProperties': False,
                    },
                    'style_focus': {
                        'type': 'object',
                        'description': 'Optional style continuity and switching controls.',
                        'properties': {
                            'style_goal': _param('string', 'How strongly to keep or switch style families.', enum=['keep_current', 'allow_switch', 'prefer_switch', 'ignore_style'], default='keep_current'),
                            'style_hints': _param('array', 'Optional style hints like pastel, rough manga line, or deadpan meme.', items={'type': 'string'}),
                            'prefer_pack': _param('string', 'Optional preferred source_pack_id or nearby pack family name. Exact or closely related pack ids get a ranking boost.'),
                            'prefer_cluster': _param('string', 'Optional preferred style_cluster id. Exact matches or style-nearby clusters get a ranking boost.'),
                        },
                        'additionalProperties': False,
                    },
                    'text_constraints': {
                        'type': 'object',
                        'description': 'Optional caption and overlay text constraints.',
                        'properties': {
                            'text_priority': _param('string', 'How strongly visible caption text should control meaning.', enum=['require', 'prefer', 'ignore'], default='prefer'),
                            'must_include': _param('array', 'Caption meanings that should be present or strongly implied.', items={'type': 'string'}),
                            'avoid_text_meanings': _param('array', 'Caption meanings to avoid.', items={'type': 'string'}),
                        },
                        'additionalProperties': False,
                    },
                    'intensity_limits': {
                        'type': 'object',
                        'description': 'Intensity caps for how strong the sticker may be.',
                        'properties': {
                            'max_harshness': _param('integer', 'Maximum tolerated harshness on a 0-4 scale.', minimum=0, maximum=4, default=3),
                            'max_intimacy': _param('integer', 'Maximum tolerated intimacy on a 0-4 scale.', minimum=0, maximum=4, default=4),
                            'max_meme_dependence': _param('integer', 'Maximum tolerated meme dependence on a 0-4 scale.', minimum=0, maximum=4, default=4),
                            'allow_animation': _param('boolean', 'Allow animated stickers if they fit better.', default=False),
                        },
                        'additionalProperties': False,
                    },
                    'candidate_budget': _param('integer', 'How many ranked candidates to return.', minimum=1, maximum=8, default=5),
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
            query_interpretation = plan.query_interpretation()
            style_context = self.catalog.describe_style_context(ctx.session_id)
            if not plan.send:
                return ToolResult(
                    call_id='',
                    name=self.spec.name,
                    output={
                        'ok': True,
                        'skipped': True,
                        'reason': 'send=false',
                        'plan': query_interpretation,
                        'query_interpretation': query_interpretation,
                        'style_context': style_context,
                    },
                )
            matches = self.catalog.choose(plan=plan, session_id=ctx.session_id)
            if not matches:
                return ToolResult(
                    call_id='',
                    name=self.spec.name,
                    output={
                        'ok': False,
                        'error': 'No sticker matched the requested intent',
                        'plan': query_interpretation,
                        'query_interpretation': query_interpretation,
                        'style_context': style_context,
                    },
                )
            return ToolResult(
                call_id='',
                name=self.spec.name,
                output={
                    'ok': True,
                    'plan': query_interpretation,
                    'query_interpretation': query_interpretation,
                    'style_context': style_context,
                    'candidate_count': len(matches),
                    'candidates': [_candidate_payload(match) for match in matches],
                    'selection_guidance': 'Inspect score_breakdown, match_profile, and selection_summary; then choose one sticker_id and call sticker_send_selected.',
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
            description='Send one previously chosen sticker from the local sticker library and echo back what was sent.',
            parameters_schema={
                'type': 'object',
                'properties': {
                    'sticker_id': _param('string', 'Exact sticker_id returned by sticker_query.'),
                    'timing': _param('string', 'Whether to send the sticker immediately now or after the final text.', enum=['send_now', 'after_final', 'before_final']),
                },
                'required': ['sticker_id'],
                'additionalProperties': False,
            },
            runner=self,
        )

    async def run(self, args: dict[str, Any], ctx: ToolContext) -> ToolResult:
        try:
            if not self.catalog.loaded:
                self.catalog.load()
            sticker_id = str(args.get('sticker_id', '') or '').strip()
            if not sticker_id:
                return ToolResult(call_id='', name=self.spec.name, output={'ok': False, 'error': 'Empty sticker_id'})
            timing_raw = str(args.get('timing', 'after_final') or 'after_final').strip().lower()
            timing = StickerTiming.parse(timing_raw)
            entry = self.catalog.get_by_sticker_id(sticker_id)
            if entry is None:
                return ToolResult(call_id='', name=self.spec.name, output={'ok': False, 'error': 'Unknown sticker_id', 'sticker_id': sticker_id})
            self.catalog.record_selection(session_id=ctx.session_id, sticker_id=entry.sticker_id)
            style_context_after_send = self.catalog.describe_style_context(ctx.session_id)
            sticker = OutboundSticker(
                path=entry.absolute_path,
                emoji=entry.emoji,
                timing=timing,
                label=entry.summary or entry.relative_path,
                source_id=entry.sticker_id,
            )
            return ToolResult(
                call_id='',
                name=self.spec.name,
                output={
                    'ok': True,
                    'status': 'success',
                    'timing': timing.value,
                    'selection_summary': _send_selection_summary(entry, style_context_after_send),
                    'style_context_after_send': style_context_after_send,
                    **_entry_payload(entry),
                },
                stickers=[sticker],
            )
        except Exception as exc:
            logger.exception('sticker_send_selected failed')
            return ToolResult(call_id='', name=self.spec.name, output={'ok': False, 'error': f'{exc.__class__.__name__}: {exc}'})
