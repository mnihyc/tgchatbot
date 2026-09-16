"""Conversation decision guidance; no extra planning or judging API call."""

STICKER_GUIDANCE = """Use stickers when they help your next turn or are requested. Start with what you want to convey to the recipient, in your established character. Offering comfort differs from requesting it; accepting blame differs from handing it back. For sticker_query, distill that intent into the emotional or social message the sticker should carry, with useful tone, caption, style, or other hints. Focus on what the recipient should take from it, rather than narrating the exchange or planning the rest of your reply.

A sticker can stand alone when it conveys the intended reply or when a playful reaction is enough for the exchange. Use text for any answer, explanation, or nuance the sticker does not convey; a brief line can be enough.

Character and style preferences guide the choice unless explicitly required. Keep a familiar style when it fits; allow variation when the exchange calls for it. Inspection and preference changes need no later send. Keep selection details out of ordinary replies."""