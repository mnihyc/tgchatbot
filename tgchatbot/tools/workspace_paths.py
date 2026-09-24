"""Remote file selection, shared by inspection and delivery (stdlib only)."""
from datetime import date
from pathlib import Path
import os
import re


def resolve_workspace_file(root, value):
    root = Path(os.path.abspath(root))
    selected = Path(value)
    if not selected.is_absolute():
        selected = root / selected
    selected = Path(os.path.normpath(selected))
    if not selected.is_relative_to(root) or not selected.resolve().is_relative_to(root.resolve()):
        raise ValueError('Requested remote path is outside the session workspace')
    relative = selected.relative_to(root)
    # Existing files (including symlink aliases) always win. Only the former
    # dated upload layout is eligible; no basename search or arbitrary prefixing.
    if not os.path.lexists(selected) and len(relative.parts) >= 2:
        day = relative.parts[0]
        if re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}', day):
            try:
                date.fromisoformat(day)
            except ValueError:
                pass
            else:
                candidate = root / 'attachments' / relative
                if not candidate.resolve().is_relative_to(root.resolve()):
                    raise ValueError('Requested remote path is outside the session workspace')
                if candidate.is_file():
                    selected = candidate
    return {'path': str(selected.resolve()),
            'workspace_path': selected.relative_to(root).as_posix()}
