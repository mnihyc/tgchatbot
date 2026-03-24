from __future__ import annotations

import contextlib
import html
import logging
import re
import time
from dataclasses import dataclass, field

from telegram import Bot, InputFile, Message
from telegram.error import BadRequest

from tgchatbot.core.events import RuntimeEvent
from tgchatbot.domain.models import OutboundArtifact, OutboundSticker, ProcessVisibility, ResponseDelivery

from telegram.helpers import escape_markdown
from telegramify_markdown import markdownify
from telegramify_markdown.config import get_runtime_config

cfg = get_runtime_config()
cfg.markdown_symbol.heading_level_1 = "-"
cfg.markdown_symbol.heading_level_2 = "--"
cfg.markdown_symbol.heading_level_3 = "---"
cfg.markdown_symbol.heading_level_4 = "----"

logger = logging.getLogger(__name__)
MAX_TELEGRAM_TEXT_CHARS = 3900


def _is_parse_error(exc: BadRequest) -> bool:
    return "can't parse entities" in str(exc).lower()

def _to_safe_markdown_v2(text: str) -> str:
    raw = text or "..."
    if markdownify is not None:
        return markdownify(raw)
    return escape_markdown(raw, version=2)


def _rendered_len(text: str) -> int:
    return len(_to_safe_markdown_v2(text))


async def bot_message_safe(client: Any, method: str, /, **kwargs):
    fn = getattr(client, method)

    text = kwargs.get("text")
    parse_mode = kwargs.get("parse_mode")

    if text is None or parse_mode != "MarkdownV2":
        return await fn(**kwargs)

    raw_text = text or "..."

    try:
        return await fn(**{**kwargs, "text": _to_safe_markdown_v2(raw_text)})
    except BadRequest as exc:
        if not _is_parse_error(exc):
            raise

    plain_kwargs = dict(kwargs)
    plain_kwargs["text"] = raw_text
    plain_kwargs.pop("parse_mode", None)
    return await fn(**plain_kwargs)


def _chunk_text(text: str, *, limit: int = MAX_TELEGRAM_TEXT_CHARS) -> list[str]:
    text = text or ''
    if len(text) <= limit:
        return [text]
    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        window = remaining[:limit]
        split_at = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(" "))
        if split_at < max(32, limit // 2):
            split_at = limit
        elif window[split_at:split_at + 2] == "\n\n":
            split_at += 2
        else:
            split_at += 1
        chunk = remaining[:split_at].rstrip()
        if not chunk:
            chunk = remaining[:limit]
            split_at = limit
        chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()
    return chunks


def _truncate_text_for_telegram(
    text: str,
    *,
    limit: int = MAX_TELEGRAM_TEXT_CHARS,
    suffix: str = "\n[continued]",
) -> str:
    text = text or ''
    if _rendered_len(text) <= limit:
        return text
    if _rendered_len(suffix) >= limit:
        return suffix[: max(1, limit)]
    lo, hi = 0, len(text)
    best = ''
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + suffix
        if _rendered_len(candidate) <= limit:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best or suffix


def _fit_text_with_suffix_for_telegram(
    text: str,
    *,
    limit: int = MAX_TELEGRAM_TEXT_CHARS,
    suffix: str = "\n[continued]",
) -> str:
    text = text or ''
    candidate = text.rstrip() + suffix
    if _rendered_len(candidate) <= limit:
        return candidate
    return _truncate_text_for_telegram(text, limit=limit, suffix=suffix)


def _chunk_text_for_telegram(text: str, *, limit: int = MAX_TELEGRAM_TEXT_CHARS) -> list[str]:
    text = text or ''
    if _rendered_len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if _rendered_len(remaining) <= limit:
            chunks.append(remaining)
            break

        lo, hi = 1, len(remaining)
        fit = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = remaining[:mid]
            if _rendered_len(candidate) <= limit:
                fit = mid
                lo = mid + 1
            else:
                hi = mid - 1

        window = remaining[:fit]
        split_at = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(" "))
        if split_at < max(32, fit // 2):
            split_at = fit
        elif window[split_at:split_at + 2] == "\n\n":
            split_at += 2
        else:
            split_at += 1

        chunk = remaining[:split_at].rstrip()
        if not chunk:
            chunk = remaining[:fit]
            split_at = fit
        chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()
    return chunks


def _chunk_text_for_telegram_with_continuation(
    text: str,
    *,
    limit: int = MAX_TELEGRAM_TEXT_CHARS,
    suffix: str = "\n[continued]",
) -> list[str]:
    text = text or ''
    if _rendered_len(text) <= limit:
        return [text]
    if _rendered_len(suffix) >= limit:
        return _chunk_text_for_telegram(text, limit=limit)

    chunks: list[str] = []
    remaining = text
    while remaining:
        if _rendered_len(remaining) <= limit:
            chunks.append(remaining)
            break

        lo, hi = 1, len(remaining)
        fit = 1
        while lo <= hi:
            mid = (lo + hi) // 2
            candidate = remaining[:mid].rstrip() + suffix
            if _rendered_len(candidate) <= limit:
                fit = mid
                lo = mid + 1
            else:
                hi = mid - 1

        window = remaining[:fit]
        split_at = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(" "))
        if split_at < max(32, fit // 2):
            split_at = fit
        elif window[split_at:split_at + 2] == "\n\n":
            split_at += 2
        else:
            split_at += 1

        chunk = remaining[:split_at].rstrip()
        if not chunk:
            chunk = remaining[:fit].rstrip()
            split_at = fit
        chunks.append(chunk + suffix)
        remaining = remaining[split_at:].lstrip()
    return chunks


def _visibility_value(value: ProcessVisibility | str | None) -> str:
    if value is None:
        return ProcessVisibility.STATUS.value
    return getattr(value, 'value', str(value))


@dataclass
class TelegramRenderState:
    lines: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)
    answer: str = ''
    live_text: str = ''
    last_render_text: str = ''
    last_render_at: float = 0.0


class TelegramMessageRenderer:
    def __init__(
        self,
        message: Message | None,
        *,
        response_delivery: ResponseDelivery,
        min_edit_interval_s: float,
        source_message: Message | None = None,
        reply_to_source_message: bool = False,
        process_visibility: ProcessVisibility | str | None = None,
    ) -> None:
        self.message = message
        self.response_delivery = response_delivery
        self.min_edit_interval_s = min_edit_interval_s
        self.source_message = source_message
        self.reply_to_source_message = reply_to_source_message
        self.process_visibility = _visibility_value(process_visibility)
        self.state = TelegramRenderState()

    def _is_full(self) -> bool:
        return self.process_visibility == ProcessVisibility.FULL.value

    def _is_verbose(self) -> bool:
        return self.process_visibility == ProcessVisibility.VERBOSE.value

    def _is_status(self) -> bool:
        return self.process_visibility == ProcessVisibility.STATUS.value

    def _is_minimal(self) -> bool:
        return self.process_visibility == ProcessVisibility.MINIMAL.value

    def _is_none(self) -> bool:
        return self.process_visibility in {ProcessVisibility.OFF.value, 'none'}

    async def begin(self) -> None:
        if self.message is None or self._is_none():
            return
        if self._is_minimal():
            return
        await self._edit_text('Status: receiving request', force=True)

    async def emit(self, event: RuntimeEvent) -> None:
        if self._is_none() or self._is_minimal():
            return
        lines = [f'Status: {event.title}']
        if event.detail.strip():
            lines.append(event.detail)
        if self._is_full():
            block = '\n'.join(line for line in lines if line.strip()).strip()
            if not block:
                return
            if self.state.blocks and self.state.blocks[-1] == block:
                return
            self.state.blocks.append(block)
            await self._append_full_block(block)
            return
        else:
            self.state.lines = lines
        await self._flush()

    async def abort(self) -> None:
        if self.message is None:
            return
        with contextlib.suppress(Exception):
            if self._is_minimal():
                await self._edit_text('Status: failed', force=True)
            else:
                await self._edit_text('Status: failed', force=True)

    async def finalize(self, text: str) -> None:
        self.state.answer = text or ''
        if self.message is None or self._is_none():
            await self._send_text_chunks(_chunk_text_for_telegram(self.state.answer))
            return

        if self._is_minimal():
            await self._replace_with_chunked_text(self.state.answer)
            return

        if self._is_full():
            header = self.state.live_text.strip()
        else:
            header = '\n'.join(line for line in self.state.lines if line.strip()).strip()

        if self.response_delivery == ResponseDelivery.FINAL_NEW:
            await self._edit_text(header or 'Done', force=True)
            await self._send_text_chunks(_chunk_text_for_telegram(self.state.answer))
            return
        if not self.state.answer:
            await self._edit_text(header or 'Done', force=True)
            return
        if self._is_full() and header:
            await self._replace_with_chunked_text(f'{header}\n\n{self.state.answer}')
            return

        await self._replace_with_chunked_text(self.state.answer)

    async def _flush(self, final: bool = False, force: bool = False) -> None:
        if self._is_none() or self._is_minimal():
            return
        if self._is_full():
            header = self.state.live_text.strip()
        else:
            header = '\n'.join(line for line in self.state.lines if line.strip())
        body = self.state.answer if final else ''
        text = '\n\n'.join(part for part in [header, body] if part).strip() or '...'
        if not self._is_full() and _rendered_len(text) > MAX_TELEGRAM_TEXT_CHARS:
            return
        await self._edit_text(text, force=force, on_too_long='ignore')

    async def _append_full_block(self, block: str) -> None:
        current = self.state.live_text.strip()
        candidate = f'{current}\n\n{block}' if current else block
        if _rendered_len(candidate) <= MAX_TELEGRAM_TEXT_CHARS:
            self.state.live_text = candidate
            await self._edit_text(candidate, on_too_long='ignore')
            return

        if current:
            continued = _fit_text_with_suffix_for_telegram(current, limit=MAX_TELEGRAM_TEXT_CHARS, suffix='\n[continued]')
            await self._edit_text(continued or '...', force=True, on_too_long='ignore')
            await self._send_new_live_text(block)
            return

        continued_chunks = _chunk_text_for_telegram_with_continuation(block)
        await self._edit_text(continued_chunks[0] or '...', force=True, on_too_long='ignore')
        self.state.live_text = continued_chunks[0] or '...'
        if len(continued_chunks) > 1:
            await self._send_exact_chunks(continued_chunks[1:], update_current=True)

    async def _send_new_live_text(self, text: str) -> None:
        await self._send_exact_chunks(_chunk_text_for_telegram_with_continuation(text), update_current=True)

    async def _send_exact_chunks(self, chunks: list[str], *, update_current: bool = False) -> Message | None:
        target = self._delivery_target()
        bot = target.get_bot()
        last_message: Message | None = None
        last_text = ''
        for chunk in chunks:
            safe_chunk = chunk or '...'
            last_message = await self._send_text_via_bot(bot, target.chat.id, safe_chunk)
            last_text = safe_chunk
        if update_current and last_message is not None:
            self.message = last_message
            self.state.live_text = last_text
            self.state.last_render_text = last_text
            self.state.last_render_at = time.monotonic()
        return last_message

    async def _edit_text(self, text: str, force: bool = False, on_too_long: str = 'replace') -> None:
        if self.message is None:
            return
        now = time.monotonic()
        if not force:
            if text == self.state.last_render_text:
                return
            if now - self.state.last_render_at < self.min_edit_interval_s:
                return
        try:
            await bot_message_safe(self.message, 'edit_text', text=text, parse_mode='MarkdownV2', disable_web_page_preview=True)
        except BadRequest as exc:
            if self._is_message_not_modified(exc):
                self.state.last_render_text = text
                self.state.last_render_at = now
                return
            if self._is_message_too_long(exc):
                if on_too_long == 'ignore':
                    logger.debug('Telegram edit exceeded size limit; ignoring oversize live update')
                    return
                logger.info('Telegram edit exceeded size limit; switching to chunked replacement')
                await self._replace_message_with_chunked_text(text)
                self.state.last_render_text = text
                self.state.last_render_at = now
                return
            logger.warning('Telegram placeholder edit failed; falling back to fresh reply: %s', exc)
            await self._fallback_send_text(text)
            self.state.last_render_text = text
            self.state.last_render_at = now
            return
        self.state.last_render_text = text
        self.state.last_render_at = now

    @staticmethod
    def _is_message_not_modified(exc: BadRequest) -> bool:
        return 'message is not modified' in str(exc).lower()

    @staticmethod
    def _is_message_too_long(exc: BadRequest) -> bool:
        text = str(exc).lower().replace(' ', '_')
        return 'message_too_long' in text or 'message is too long' in text

    def _delivery_target(self) -> Message:
        target = self.source_message or self.message
        if target is None:
            raise RuntimeError('No Telegram message available for delivery')
        return target

    async def _delete_message_if_possible(self, message: Message | None) -> None:
        if message is None:
            return
        with contextlib.suppress(Exception):
            await message.delete()

    def _reply_to_message_id(self) -> int | None:
        if not self.reply_to_source_message or self.source_message is None:
            return None
        return self.source_message.message_id

    async def _send_text_via_bot(self, bot: Bot, chat_id: int, text: str) -> Message:
        return await bot_message_safe(bot, 'send_message', chat_id=chat_id, text=text, parse_mode='MarkdownV2', disable_web_page_preview=True, reply_to_message_id=self._reply_to_message_id())

    async def _send_text_chunks(self, chunks: list[str]) -> Message | None:
        target = self._delivery_target()
        bot = target.get_bot()
        last_message: Message | None = None
        for chunk in chunks:
            for safe_chunk in _chunk_text_for_telegram(chunk):
                last_message = await self._send_text_via_bot(bot, target.chat.id, safe_chunk or '...')
        return last_message

    async def _replace_message_with_chunked_text(self, text: str) -> None:
        original_message = self.message
        chunks = _chunk_text_for_telegram(text)
        if original_message is None:
            await self._send_text_chunks(chunks)
            return
        try:
            await bot_message_safe(original_message, 'edit_text', text=(chunks[0] or '...'), parse_mode='MarkdownV2', disable_web_page_preview=True)
            self.message = original_message
            if len(chunks) > 1:
                await self._send_text_chunks(chunks[1:])
        except BadRequest as exc:
            logger.warning('Telegram chunked replacement edit failed; falling back to fresh reply: %s', exc)
            replacement = await self._send_text_chunks(chunks)
            if replacement is not None:
                self.message = replacement
            if replacement is not None and original_message is not None and replacement.message_id != original_message.message_id:
                await self._delete_message_if_possible(original_message)

    async def _replace_with_chunked_text(self, text: str) -> None:
        chunks = _chunk_text_for_telegram(text)
        if self.message is None:
            await self._send_text_chunks(chunks)
            return
        await self._edit_text(chunks[0] or '...', force=True)
        if len(chunks) > 1:
            await self._send_text_chunks(chunks[1:])

    async def send_text(self, text: str) -> None:
        await self._send_text_chunks(_chunk_text_for_telegram(text))

    async def _fallback_send_text(self, text: str) -> None:
        original_message = self.message
        message = await self._send_text_chunks(_chunk_text_for_telegram(text))
        if message is not None:
            self.message = message
        if message is not None and original_message is not None and message.message_id != original_message.message_id:
            await self._delete_message_if_possible(original_message)

    async def _send_new_text(self, text: str) -> None:
        await self.send_text(text)

    async def send_artifacts(self, artifacts: list[OutboundArtifact]) -> None:
        target = self._delivery_target()
        bot = target.get_bot()
        chat_id = target.chat.id
        reply_to_message_id = self._reply_to_message_id()
        for artifact in artifacts:
            try:
                suffix = artifact.path.suffix.lower()
                if suffix in {'.png', '.jpg', '.jpeg', '.webp'}:
                    with artifact.path.open('rb') as fh:
                        await bot.send_photo(chat_id=chat_id, photo=fh, caption=artifact.caption, reply_to_message_id=reply_to_message_id)
                else:
                    with artifact.path.open('rb') as fh:
                        await bot.send_document(chat_id=chat_id, document=InputFile(fh, filename=artifact.filename), caption=artifact.caption, reply_to_message_id=reply_to_message_id)
            except Exception:
                logger.exception('Failed to send artifact %s', artifact.path)

    async def send_stickers(self, stickers: list[OutboundSticker]) -> list[dict[str, object]]:
        target = self._delivery_target()
        bot = target.get_bot()
        reply_to_message_id = self._reply_to_message_id()
        receipts: list[dict[str, object]] = []
        for sticker in stickers:
            receipt: dict[str, object] = {
                'sticker_id': sticker.source_id,
                'relative_path': str(sticker.path),
                'label': sticker.label,
                'timing': sticker.timing.value,
                'emoji': sticker.emoji,
                'delivery_state': 'failed',
                'sent': False,
            }
            if not sticker.path.exists():
                receipt['error'] = 'missing_file'
                logger.warning('Sticker file missing: %s', sticker.path)
                receipts.append(receipt)
                continue
            try:
                with sticker.path.open('rb') as fh:
                    sent_message = await bot.send_sticker(chat_id=target.chat.id, sticker=fh, emoji=sticker.emoji, reply_to_message_id=reply_to_message_id)
                receipt['delivery_state'] = 'sent'
                receipt['sent'] = True
                receipt['telegram_message_id'] = getattr(sent_message, 'message_id', None)
            except Exception as exc:
                logger.exception('Failed to send sticker %s', sticker.path)
                receipt['error'] = exc.__class__.__name__
            receipts.append(receipt)
        return receipts
