"""Conversation decision guidance; no extra planning or judging API call."""

STICKER_GUIDANCE = """Use available sticker tools when a sticker helps your next turn or is requested. Start with what you want to convey to the recipient, in your established character. Offering comfort differs from requesting it; accepting blame differs from handing it back. Put that concrete move in sticker_query.intent_core, with only useful tone, caption or style hints. Do not narrate the selection process.

Character and style preferences guide the choice unless explicitly required. Keep a familiar style when it fits; allow variation when the exchange calls for it. Among similarly fitting choices, prefer variety over needless repetition of delivered stickers. A recipient's mood or artwork preference does not become your identity. Remember or clear a persona only for an intended continuing change; temporary preferences belong in the query or use_once. Query-only inspection and preference changes need no later send.

Inspect returned candidates, then select an exact sticker_id, refine the query or use text alone. Keep final words coherent with the asset. Use native tool calls and respect delivery timing. Do not blindly repeat an unknown delivery or substitute for an unavailable asset. Keep IDs and selection justifications out of ordinary replies unless requested."""
