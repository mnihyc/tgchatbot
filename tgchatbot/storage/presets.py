from __future__ import annotations

from pathlib import Path


class PresetStore:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def list_names(self) -> list[str]:
        return sorted(path.stem for path in self.root.glob('*.txt'))

    def get_text(self, name: str) -> str | None:
        path = self.root / f'{name}.txt'
        if not path.exists() or not path.is_file():
            return None
        return path.read_text(encoding='utf-8')

    def save_text(self, name: str, text: str) -> Path:
        safe_name = Path(name).stem.strip().replace('/', '_') or 'default'
        path = self.root / f'{safe_name}.txt'
        path.write_text(text, encoding='utf-8')
        return path
