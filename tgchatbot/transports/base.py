from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransportCapabilities:
    can_edit_messages: bool
    can_show_typing: bool
    can_upload_files: bool
    can_render_buttons: bool
    max_message_length: int
