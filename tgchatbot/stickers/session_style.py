"""A disposable cache of explicit session persona and confirmed delivery references.

Search exposure never establishes a preference. PostgreSQL owns durable history.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Iterable
from tgchatbot.stickers.persona import compact_persona_dict


@dataclass(slots=True)
class SessionStyleState:
    session_id: str
    session_persona: dict[str, Any] | None = None
    session_persona_loaded: bool = False
    recent_sticker_ids: list[str] = field(default_factory=list)
    recent_source_pack_ids: list[str] = field(default_factory=list)
    source_pack_id: str | None = None

    def set_session_persona(self, persona):
        self.session_persona = compact_persona_dict(persona) or None
        self.session_persona_loaded = True

    def clear_session_persona(self):
        self.set_session_persona(None)

    def to_context_dict(self):
        return {'recent_sticker_ids': list(self.recent_sticker_ids),
                'recent_source_pack_ids': list(self.recent_source_pack_ids)}


class SessionStyleMemory:
    def __init__(self, *, max_sessions: int = 32):
        self._states: OrderedDict[str, SessionStyleState] = OrderedDict()
        self.max_sessions = max_sessions

    def get(self, session_id: str) -> SessionStyleState:
        if session_id not in self._states:
            while len(self._states) >= self.max_sessions:
                self._states.popitem(last=False)
            self._states[session_id] = SessionStyleState(session_id)
        self._states.move_to_end(session_id)
        return self._states[session_id]

    def clear(self, session_id: str):
        self._states.pop(session_id, None)

    def preload(self, session_id: str, *, recent_sticker_ids: Iterable[str] = (), recent_source_pack_ids: Iterable[str] = ()):
        state = self.get(session_id)
        state.recent_sticker_ids = list(recent_sticker_ids)
        state.recent_source_pack_ids = list(recent_source_pack_ids)
        state.source_pack_id = state.recent_source_pack_ids[0] if state.recent_source_pack_ids else None
        return state
