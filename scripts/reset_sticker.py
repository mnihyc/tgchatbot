"""Regenerate selected sticker assets/packs without deleting active catalog records.

Use --asset-id or --pack; source files and explicit corrections are preserved.
"""
from tgchatbot.stickers.build import main

if __name__ == '__main__':
    raise SystemExit(main())
