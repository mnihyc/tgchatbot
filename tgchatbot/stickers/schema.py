from __future__ import annotations

from copy import deepcopy
from typing import Any
import hashlib

STICKER_SCHEMA_VERSION = 'human_sticker_v1'

CAPTION_CARD_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'description': 'Caption-level meaning only. Use this even when the sticker is partly visual, but keep purely visual cues out of this card.',
    'additionalProperties': False,
    'properties': {
        'caption_mode': {
            'type': 'string',
            'enum': ['caption_dominant', 'mixed', 'visual_dominant'],
            'description': 'Whether visible caption text dominates the sticker meaning, shares meaning with the art, or is secondary to the art.',
            'examples': ['caption_dominant'],
        },
        'caption_literal_meaning': {
            'type': 'string',
            'description': 'Plain-language rendering of what the visible text literally says.',
            'examples': ["I can't keep talking to you"],
        },
        'caption_pragmatic_meaning': {
            'type': 'string',
            'description': 'What social act the caption performs in conversation. Preserve force and slang; do not sanitize.',
            'examples': ['Dismissive rebuttal that shuts down the other person\'s take.'],
        },
        'caption_discourse_role': {
            'type': 'string',
            'description': 'Conversation function performed by the caption.',
            'examples': ['rebuttal'],
        },
        'caption_target_direction': {
            'type': 'string',
            'description': 'Who or what the sticker is directed toward.',
            'examples': ['other_person'],
        },
        'caption_social_stance': {
            'type': 'string',
            'description': 'Interpersonal stance expressed by the caption.',
            'examples': ['dismissive, annoyed, lightly aggressive'],
        },
        'caption_reply_force': {
            'type': 'string',
            'description': 'How strongly the caption pushes, rejects, comforts, teases, or closes the exchange.',
            'examples': ['strong shutdown'],
        },
        'caption_emotional_valence': {
            'type': 'string',
            'description': 'Overall emotional valence of the caption meaning.',
            'examples': ['negative'],
        },
        'caption_irony_strength': {
            'type': 'string',
            'description': 'How much irony, sarcasm, or deadpan distance is present in the caption.',
            'examples': ['medium'],
        },
        'caption_relationship_fit': {
            'type': 'string',
            'description': 'Relationship settings where the caption is a good fit.',
            'examples': ['close friends, chaotic group chat, meme-native peers'],
        },
        'caption_conversation_role': {
            'type': 'string',
            'description': 'What role the caption plays in the local exchange.',
            'examples': ['turn-ending clapback'],
        },
        'caption_use_when': {
            'type': 'string',
            'description': 'Situations where the caption is a good fit.',
            'examples': ['When replying to a baffling take in casual chat.'],
        },
        'caption_avoid_when': {
            'type': 'string',
            'description': 'Situations where the caption should not be used.',
            'examples': ['When the other person needs comfort or when de-escalation matters.'],
        },
        'caption_close_aliases': {
            'type': 'string',
            'description': 'Nearby paraphrases that preserve the same social act.',
            'examples': ["I'm done arguing; this is too absurd to continue."],
        },
        'caption_meaning_en': {
            'type': 'string',
            'description': 'Short English meaning paraphrase used for multilingual retrieval.',
            'examples': ["I can't even; I'm done talking because this take is ridiculous."],
        },
        'caption_meaning_zh': {
            'type': 'string',
            'description': 'Chinese meaning paraphrase or compact restatement when helpful for multilingual retrieval.',
            'examples': ['受不了了，不想再和你争了'],
        },
        'harshness_level': {'type': 'integer', 'minimum': 0, 'maximum': 4, 'description': 'How sharp or cutting the caption is on a 0-4 scale.'},
        'intimacy_level': {'type': 'integer', 'minimum': 0, 'maximum': 4, 'description': 'How much relationship closeness is assumed on a 0-4 scale.'},
        'meme_dependence_level': {'type': 'integer', 'minimum': 0, 'maximum': 4, 'description': 'How much the caption relies on meme/slang literacy on a 0-4 scale.'},
    },
    'required': [
        'caption_mode', 'caption_literal_meaning', 'caption_pragmatic_meaning', 'caption_discourse_role', 'caption_target_direction',
        'caption_social_stance', 'caption_reply_force', 'caption_emotional_valence', 'caption_irony_strength',
        'caption_relationship_fit', 'caption_conversation_role', 'caption_use_when', 'caption_avoid_when', 'caption_close_aliases',
        'caption_meaning_en', 'caption_meaning_zh', 'harshness_level', 'intimacy_level', 'meme_dependence_level',
    ],
}

SUBTLE_CUE_CARD_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'description': 'Visual-expression-level pragmatics only. This card carries hidden cues for textless stickers and the art-side signal for captioned stickers.',
    'additionalProperties': False,
    'properties': {
        'dominant_signal': {'type': 'string', 'description': 'Main hidden cue a human would notice first.', 'examples': ['flat stare plus tired head tilt']},
        'micro_expression': {'type': 'string', 'description': 'Small facial-expression read, not the whole sticker meaning.', 'examples': ['micro-scowl with deadpan eyes']},
        'eye_signal': {'type': 'string', 'description': 'Meaning carried by eye shape or gaze.', 'examples': ['side-eye disbelief']},
        'mouth_signal': {'type': 'string', 'description': 'Meaning carried by mouth shape.', 'examples': ['tiny grimace that signals fed-up restraint']},
        'pose_signal': {'type': 'string', 'description': 'Meaning carried by posture or body pose.', 'examples': ['slumped recoil that reads as over-it']},
        'head_signal': {'type': 'string', 'description': 'Meaning carried by head angle or tilt.', 'examples': ['head tilt that adds contemptuous disbelief']},
        'hand_signal': {'type': 'string', 'description': 'Meaning carried by hands or gesture.', 'examples': ['dismissive hand flick']},
        'color_signal': {'type': 'string', 'description': 'Meaning carried by palette, saturation, or color temperature.', 'examples': ['washed-out palette that adds drained irritation']},
        'composition_signal': {'type': 'string', 'description': 'Meaning carried by framing, crop, or emphasis.', 'examples': ['tight crop that intensifies the glare']},
        'motion_signal': {'type': 'string', 'description': 'Meaning carried by movement or implied motion.', 'examples': ['small shake that reads as no thanks']},
        'visual_social_stance': {'type': 'string', 'description': 'Interpersonal stance conveyed visually.', 'examples': ['dryly dismissive, unimpressed']},
        'visual_reply_force': {'type': 'string', 'description': 'How strongly the visual expression pushes, rejects, comforts, or teases.', 'examples': ['firm rejection without open rage']},
        'visual_emotional_valence': {'type': 'string', 'description': 'Overall emotional valence carried by the visual expression.', 'examples': ['negative']},
        'visual_irony_strength': {'type': 'string', 'description': 'How much irony or deadpan distance is carried by the visual expression.', 'examples': ['medium']},
        'visual_relationship_fit': {'type': 'string', 'description': 'Relationship settings where the visual expression fits.', 'examples': ['friends, meme chat, sarcastic coworkers']},
        'visual_conversation_role': {'type': 'string', 'description': 'What role the visual expression plays in the exchange.', 'examples': ['silent side-eye reaction']},
        'visual_use_when': {'type': 'string', 'description': 'Situations where the visual expression is a good fit.', 'examples': ['When words would be too much and a look says enough.']},
        'visual_avoid_when': {'type': 'string', 'description': 'Situations where the visual expression should not be used.', 'examples': ['When warmth, clarity, or sincere reassurance is needed.']},
    },
    'required': [
        'dominant_signal', 'micro_expression', 'eye_signal', 'mouth_signal', 'pose_signal', 'head_signal', 'hand_signal',
        'color_signal', 'composition_signal', 'motion_signal', 'visual_social_stance', 'visual_reply_force',
        'visual_emotional_valence', 'visual_irony_strength', 'visual_relationship_fit', 'visual_conversation_role',
        'visual_use_when', 'visual_avoid_when',
    ],
}

STICKER_CARD_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'description': 'Fused overall sticker interpretation. This is the final social meaning after combining caption truth and visual-expression cues.',
    'additionalProperties': False,
    'properties': {
        'sticker_reaction_type': {'type': 'string', 'description': 'Overall reaction category for the sticker as a whole.', 'examples': ['dismissive disagreement']},
        'fused_pragmatic_meaning': {'type': 'string', 'description': 'Best overall social meaning of the whole sticker after fusing text and art.', 'examples': ['A cute-looking but actually fed-up rebuttal that closes the exchange.']},
        'sticker_visual_emotion': {'type': 'string', 'description': 'Emotion conveyed visually by the art.', 'examples': ['annoyed deadpan']},
        'sticker_humor_style': {'type': 'string', 'description': 'Humor mode or comedic framing.', 'examples': ['meme sarcasm']},
        'sticker_delivery_style': {'type': 'string', 'description': 'How the sticker delivers the reaction overall.', 'examples': ['cute shell around a sharper shutdown']},
        'sticker_subject_type': {'type': 'string', 'description': 'Type of depicted subject.', 'examples': ['chibi animal character']},
        'sticker_art_softening_strength': {'type': 'string', 'description': 'How much the art softens or intensifies the meaning.', 'examples': ['softens slightly but does not neutralize the rebuttal']},
        'sticker_general_usefulness': {'type': 'string', 'description': 'How reusable the sticker is across contexts.', 'examples': ['useful for dismissive disagreement and baffled shutdown moments']},
        'sticker_reply_force': {'type': 'string', 'description': 'Overall reply force of the whole sticker.', 'examples': ['strong rebuttal']},
        'sticker_emotional_valence': {'type': 'string', 'description': 'Overall emotional valence of the whole sticker.', 'examples': ['negative']},
        'sticker_irony_strength': {'type': 'string', 'description': 'Overall irony or sarcasm strength of the whole sticker.', 'examples': ['medium_high']},
        'sticker_relationship_fit': {'type': 'string', 'description': 'Relationship settings where the whole sticker fits.', 'examples': ['friends, meme-native peers, bantering group chat']},
        'sticker_conversation_role': {'type': 'string', 'description': 'Conversation role of the whole sticker.', 'examples': ['turn-ending reaction shot']},
        'sticker_use_when': {'type': 'string', 'description': 'Situations where the whole sticker is a good fit.', 'examples': ['When you want to shut down a goofy argument without writing a paragraph.']},
        'sticker_avoid_when': {'type': 'string', 'description': 'Situations where the whole sticker is a bad fit.', 'examples': ['When reassurance, diplomacy, or genuine neutrality is needed.']},
    },
    'required': [
        'sticker_reaction_type', 'fused_pragmatic_meaning', 'sticker_visual_emotion', 'sticker_humor_style', 'sticker_delivery_style',
        'sticker_subject_type', 'sticker_art_softening_strength', 'sticker_general_usefulness', 'sticker_reply_force',
        'sticker_emotional_valence', 'sticker_irony_strength', 'sticker_relationship_fit', 'sticker_conversation_role',
        'sticker_use_when', 'sticker_avoid_when',
    ],
}

STYLE_CARD_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'description': 'Rendering and pack/style descriptors only. Never place semantic meaning in this card.',
    'additionalProperties': False,
    'properties': {
        'style_rendering_type': {'type': 'string', 'description': 'Visual rendering family only, with no semantic meaning mixed in.', 'examples': ['flat anime meme sticker']},
        'line_weight': {'type': 'string', 'description': 'Observed stroke weight or outline heaviness.', 'examples': ['medium']},
        'style_palette_family': {'type': 'string', 'description': 'Overall color family or palette impression.', 'examples': ['soft pastel']},
        'meme_intensity': {'type': 'string', 'description': 'How memey or internet-coded the rendering feels.', 'examples': ['high']},
        'style_text_prominence': {'type': 'string', 'description': 'How visually prominent the text is in the composition.', 'examples': ['high']},
        'style_character_family': {'type': 'string', 'description': 'Visual character family or mascot family.', 'examples': ['round chibi cat']},
    },
    'required': ['style_rendering_type', 'line_weight', 'style_palette_family', 'meme_intensity', 'style_text_prominence', 'style_character_family'],
}

STICKER_RESPONSE_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'additionalProperties': False,
    'properties': {
        'summary': {'type': 'string', 'description': 'One-sentence human-readable summary of the sticker.', 'examples': ['Dismissive rebuttal with cute-character shell.']},
        'preview_text': {'type': 'string', 'description': 'Short label shown in candidate lists.', 'examples': ['dismissive rebuttal / annoyed shutdown']},
        'selection_notes': {'type': 'string', 'description': 'Compact usage guidance for ranking and operator review.', 'examples': ['Use for fed-up rebuttal or dry disbelief; avoid for comfort or de-escalation.']},
        'emoji': {'type': 'string', 'description': 'Optional emoji label if one cleanly fits.', 'examples': ['🙄']},
        'caption_card': CAPTION_CARD_SCHEMA,
        'subtle_cue_card': SUBTLE_CUE_CARD_SCHEMA,
        'sticker_card': STICKER_CARD_SCHEMA,
        'style_card': STYLE_CARD_SCHEMA,
    },
    'required': ['summary', 'preview_text', 'selection_notes', 'emoji', 'caption_card', 'subtle_cue_card', 'sticker_card', 'style_card'],
}

STICKER_REVIEW_SCHEMA: dict[str, Any] = {
    'type': 'object',
    'additionalProperties': False,
    'properties': {
        'result': STICKER_RESPONSE_SCHEMA,
        'alignment_score': {'type': 'number'},
        'caption_authority': {'type': 'number'},
        'notes': {'type': 'string'},
    },
    'required': ['result', 'alignment_score', 'caption_authority', 'notes'],
}

PRIMARY_ANALYSIS_PROMPT = '''
You are preparing a sticker retrieval index for a chat bot.
Return strict JSON that follows the schema exactly.

Core rules:
1. If readable overlay text is clear and dominant, the caption defines the sticker's semantic identity.
2. In caption_dominant stickers, the character art is secondary and only softens or intensifies delivery.
3. Preserve pragmatic meaning precisely. Represent the social act, reply force, irony, valence, and stance explicitly instead of flattening everything into broad emotions.
4. Distinguish three layers cleanly:
   - caption_card = caption-level meaning only
   - subtle_cue_card = visual-expression-level meaning only
   - sticker_card = fused whole-sticker meaning
5. Visual-only stickers still need force, valence, irony, social stance, conversation role, use_when, and avoid_when. Do not leave those dimensions empty just because there is no text.
6. Use English for semantic retrieval text, but keep Chinese/Japanese/Korean wording intact inside meaning fields where useful.
7. Do not sanitize meme or rude meaning into generic safe words if the visible caption is sharper than that.
8. Important correction: Chinese “无语” is not neutral “speechless.” In many sticker contexts it is closer to exasperated, annoyed, fed up, dismissive disbelief, or “I can’t even.” Treat it as stronger rebuttal/disengagement when the local evidence supports that reading.

What to produce:
- summary and preview_text for humans
- caption_card = structured caption meaning with explicit field responsibilities and no jargon shortcuts
- subtle_cue_card = eye/head/color/pose/micro-expression signals and their pragmatic force
- sticker_card = the broader fused sticker reaction and how text plus art combine
- style_card = rendering/style descriptors only, never semantic meaning
- selection_notes = concise usage and avoidance guidance
'''

REVIEW_PROMPT = '''
You are the semantic quality gate for a sticker index.
Given the OCR evidence, image frames, and a candidate structured analysis, return a revised JSON object that is maximally faithful and globally consistent.

You must reason from first principles, not from canned labels.

Global invariants:
1. Preserve caption_literal_meaning and caption_pragmatic_meaning as separate things.
2. If overlay text is clear, legible, and occupies meaningful visual attention, the caption may dominate the semantic identity.
3. Do not flatten dismissive, rejecting, oppositional, teasing, comforting, flirtatious, or boundary-setting speech acts into generic emotion labels.
4. Ensure caption_card, subtle_cue_card, sticker_card, and style_card agree with each other while staying in their own lane.
5. Ensure use_when, avoid_when, relationship_fit, and conversation_role follow directly from the inferred force and register.
6. Keep style descriptors purely stylistic; do not move semantics into style_card.
7. Prefer precise pragmatic meaning over sanitized or vague summaries.
8. For visual-only stickers, do not leave irony, force, valence, or social stance trapped in caption fields.
9. Important correction: treat Chinese “无语” as potentially exasperated, annoyed, fed up, dismissive disbelief, or “I can’t even,” not merely neutral speechlessness.
10. Produce a revised result even if the candidate is mostly correct.
'''


def strict_response_schema() -> dict[str, Any]:
    return deepcopy(STICKER_RESPONSE_SCHEMA)


def strict_review_schema() -> dict[str, Any]:
    return deepcopy(STICKER_REVIEW_SCHEMA)


def _normalize_text(value: Any) -> str:
    return ' '.join(str(value or '').replace('\n', ' ').replace('\t', ' ').split()).strip()


def compose_caption_semantic_text(semantics: dict[str, Any]) -> str:
    caption = (semantics or {}).get('caption_card', {}) or {}
    fields = [
        caption.get('caption_pragmatic_meaning', ''),
        caption.get('caption_discourse_role', ''),
        caption.get('caption_target_direction', ''),
        caption.get('caption_social_stance', ''),
        caption.get('caption_reply_force', ''),
        caption.get('caption_emotional_valence', ''),
        caption.get('caption_irony_strength', ''),
        caption.get('caption_relationship_fit', ''),
        caption.get('caption_conversation_role', ''),
        caption.get('caption_meaning_en', ''),
        caption.get('caption_meaning_zh', ''),
        caption.get('caption_use_when', ''),
        caption.get('caption_avoid_when', ''),
        caption.get('caption_close_aliases', ''),
    ]
    return ' ; '.join(filter(None, (_normalize_text(value) for value in fields)))


def compose_visual_semantic_text(semantics: dict[str, Any]) -> str:
    subtle = (semantics or {}).get('subtle_cue_card', {}) or {}
    fields = [
        subtle.get('dominant_signal', ''),
        subtle.get('micro_expression', ''),
        subtle.get('eye_signal', ''),
        subtle.get('mouth_signal', ''),
        subtle.get('pose_signal', ''),
        subtle.get('head_signal', ''),
        subtle.get('hand_signal', ''),
        subtle.get('color_signal', ''),
        subtle.get('composition_signal', ''),
        subtle.get('motion_signal', ''),
        subtle.get('visual_social_stance', ''),
        subtle.get('visual_reply_force', ''),
        subtle.get('visual_emotional_valence', ''),
        subtle.get('visual_irony_strength', ''),
        subtle.get('visual_relationship_fit', ''),
        subtle.get('visual_conversation_role', ''),
        subtle.get('visual_use_when', ''),
        subtle.get('visual_avoid_when', ''),
    ]
    return ' ; '.join(filter(None, (_normalize_text(value) for value in fields)))


def compose_sticker_semantic_text(semantics: dict[str, Any]) -> str:
    sticker = (semantics or {}).get('sticker_card', {}) or {}
    fields = [
        sticker.get('sticker_reaction_type', ''),
        sticker.get('fused_pragmatic_meaning', ''),
        sticker.get('sticker_visual_emotion', ''),
        sticker.get('sticker_humor_style', ''),
        sticker.get('sticker_delivery_style', ''),
        sticker.get('sticker_subject_type', ''),
        sticker.get('sticker_art_softening_strength', ''),
        sticker.get('sticker_general_usefulness', ''),
        sticker.get('sticker_reply_force', ''),
        sticker.get('sticker_emotional_valence', ''),
        sticker.get('sticker_irony_strength', ''),
        sticker.get('sticker_relationship_fit', ''),
        sticker.get('sticker_conversation_role', ''),
        sticker.get('sticker_use_when', ''),
        sticker.get('sticker_avoid_when', ''),
        compose_visual_semantic_text(semantics),
    ]
    return ' ; '.join(filter(None, (_normalize_text(value) for value in fields)))


def compose_style_text(semantics: dict[str, Any]) -> str:
    style = (semantics or {}).get('style_card', {}) or {}
    fields = [
        style.get('style_rendering_type', ''),
        style.get('line_weight', ''),
        style.get('style_palette_family', ''),
        style.get('meme_intensity', ''),
        style.get('style_text_prominence', ''),
        style.get('style_character_family', ''),
    ]
    return ' ; '.join(filter(None, (_normalize_text(value) for value in fields)))


def semantic_signature(semantics: dict[str, Any]) -> str:
    caption = (semantics or {}).get('caption_card', {}) or {}
    subtle = (semantics or {}).get('subtle_cue_card', {}) or {}
    sticker = (semantics or {}).get('sticker_card', {}) or {}
    base = ' | '.join(
        _normalize_text(value).lower()
        for value in [
            caption.get('caption_pragmatic_meaning', ''),
            caption.get('caption_discourse_role', ''),
            caption.get('caption_reply_force', ''),
            sticker.get('sticker_reaction_type', ''),
            sticker.get('fused_pragmatic_meaning', ''),
            sticker.get('sticker_reply_force', ''),
            subtle.get('dominant_signal', ''),
            subtle.get('visual_social_stance', ''),
            subtle.get('visual_reply_force', ''),
        ]
        if _normalize_text(value)
    )
    if not base:
        return ''
    return hashlib.sha1(base.encode('utf-8')).hexdigest()[:16]
