from __future__ import annotations

import contextlib
import logging
import time
from dataclasses import dataclass, field
from typing import Any

from telegram import Bot, Message, MessageEntity
from telegram.error import BadRequest

from tgchatbot.transports.sticker_delivery import send_sticker as deliver_sticker
from tgchatbot.transports.telegram_routing import topic_arguments
from tgchatbot.core.events import RuntimeEvent
from tgchatbot.domain.models import OutboundArtifact, OutboundSticker, ProcessVisibility, ResponseDelivery

from telegramify_markdown import convert, split_entities, utf16_len
from telegramify_markdown import MessageEntity as FormattingEntity
from telegramify_markdown.config import get_runtime_config

cfg = get_runtime_config()
cfg.markdown_symbol.heading_level_1 = "-"
cfg.markdown_symbol.heading_level_2 = "--"
cfg.markdown_symbol.heading_level_3 = "---"
cfg.markdown_symbol.heading_level_4 = "----"

logger = logging.getLogger(__name__)
# Retain the existing headroom below Telegram's 4,096 UTF-16-unit text limit.
MAX_TELEGRAM_TEXT_CHARS = 3900


@dataclass(frozen=True)
class TelegramText:
    """Display text and its formatting; never feed a rendered chunk back into Markdown."""

    text: str
    entities: tuple[FormattingEntity, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.text)


TelegramContent = str | TelegramText


def _formatted(text: TelegramContent) -> TelegramText:
    if isinstance(text, TelegramText):
        return text
    plain, entities = convert(text or '...')
    return TelegramText(plain, tuple(entities))


def _join_text(parts: list[TelegramContent], separator: str = '') -> TelegramText:
    """Join rendered segments, retaining each segment's existing entity offsets."""
    texts: list[str] = []
    entities: list[FormattingEntity] = []
    offset = 0
    for part in parts:
        if not part:
            continue
        if texts:
            texts.append(separator)
            offset += utf16_len(separator)
        rendered = _formatted(part)
        texts.append(rendered.text)
        entities.extend(entity.copy_with(offset=offset + entity.offset) for entity in rendered.entities)
        offset += utf16_len(rendered.text)
    return TelegramText(''.join(texts), tuple(entities))


def _rendered_len(text: TelegramContent) -> int:
    return utf16_len(_formatted(text).text)


async def bot_message_safe(client: Any, method: str, /, **kwargs):
    fn = getattr(client, method)
    text = kwargs.get('text')
    if text is None or (kwargs.get('parse_mode') != 'MarkdownV2' and not isinstance(text, TelegramText)):
        return await fn(**kwargs)

    rendered = _formatted(text)
    # Native entities avoid Telegram parsing Markdown a second time. Malformed
    # source Markdown stays literal without discarding neighbouring valid styles.
    return await fn(**{
        **kwargs,
        'text': rendered.text or '...',
        'parse_mode': None,
        'entities': [MessageEntity.de_json(entity.to_dict(), bot=None) for entity in rendered.entities],
    })


def _chunk_text_for_telegram(
    text: TelegramContent, *, limit: int = MAX_TELEGRAM_TEXT_CHARS,
) -> list[TelegramText]:
    rendered = _formatted(text)
    chunks = split_entities(rendered.text, list(rendered.entities), max_utf16_len=limit)
    return [TelegramText(plain, tuple(entities)) for plain, entities in chunks] or [TelegramText('...')]


def _fit_text_with_suffix_for_telegram(
    text: TelegramContent,
    *,
    limit: int = MAX_TELEGRAM_TEXT_CHARS,
    suffix: str = "\n[continued]",
) -> TelegramText:
    ending = _formatted(suffix)
    budget = limit - _rendered_len(ending)
    if budget <= 0:
        return _chunk_text_for_telegram(ending, limit=limit)[0]
    first = _chunk_text_for_telegram(text, limit=budget)[0]
    return _join_text([first, ending])


def _chunk_text_for_telegram_with_continuation(
    text: TelegramContent,
    *,
    limit: int = MAX_TELEGRAM_TEXT_CHARS,
    suffix: str = "\n[continued]",
) -> list[TelegramText]:
    if _rendered_len(text) <= limit:
        return [_formatted(text)]
    ending = _formatted(suffix)
    budget = limit - _rendered_len(ending)
    if budget <= 0:
        return _chunk_text_for_telegram(text, limit=limit)
    chunks = _chunk_text_for_telegram(text, limit=budget)
    return [_join_text([chunk, ending]) for chunk in chunks[:-1]] + chunks[-1:]


def _visibility_value(value: ProcessVisibility | str | None) -> str:
    if value is None:
        return ProcessVisibility.STATUS.value
    return getattr(value, 'value', str(value))


@dataclass
class TelegramRenderState:
    lines: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)
    answer: str = ''
    live_text: TelegramContent = ''
    last_render_text: TelegramContent = ''
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
        sticker_delivery=None,
    ) -> None:
        self.sticker_delivery = sticker_delivery
        self.message = message
        self.response_delivery = response_delivery
        self.min_edit_interval_s = min_edit_interval_s
        self.source_message = source_message
        self.reply_to_source_message = reply_to_source_message
        self.process_visibility = _visibility_value(process_visibility)
        self.state = TelegramRenderState()
        self._final_messages: list[Message] | None = None

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
        try:
            await self._edit_text('Status: receiving request', force=True)
        except Exception:
            logger.debug('tg.progress.begin.failed', exc_info=True)

    async def emit(self, event: RuntimeEvent) -> None:
        if self._is_none() or self._is_minimal():
            return
        lines = [f'Status: {event.title}']
        if event.detail.strip():
            lines.append(event.detail)
        try:
            if self._is_full():
                block = '\n'.join(line for line in lines if line.strip()).strip()
                if not block:
                    return
                if self.state.blocks and self.state.blocks[-1] == block:
                    return
                self.state.blocks.append(block)
                await self._append_full_block(block)
            else:
                self.state.lines = lines
                await self._flush()
        except Exception:
            logger.debug('tg.progress.emit.failed', exc_info=True)

    async def abort(self) -> None:
        if self.message is None:
            return
        with contextlib.suppress(Exception):
            if self._is_minimal():
                await self._edit_text('Status: failed', force=True)
            else:
                await self._edit_text('Status: failed', force=True)

    def _remember_final_message(self, message: Message) -> None:
        if self._final_messages is not None and not any(
                item.message_id == message.message_id for item in self._final_messages):
            self._final_messages.append(message)

    async def finalize(self, text: str) -> list[Message]:
        self._final_messages = []
        try:
            await self._finalize_text(text)
            return list(self._final_messages)
        finally:
            self._final_messages = None

    async def _finalize_text(self, text: str) -> None:
        self.state.answer = text or ''
        if self.message is None or self._is_none():
            await self._send_exact_chunks(_chunk_text_for_telegram(self.state.answer))
            return

        if self._is_minimal():
            await self._replace_with_chunked_text(self.state.answer)
            return

        if self._is_full():
            header = self.state.live_text
        else:
            header = '\n'.join(line for line in self.state.lines if line.strip()).strip()

        if self.response_delivery == ResponseDelivery.FINAL_NEW:
            receipts, self._final_messages = self._final_messages, None
            try:
                await self._edit_text(header or 'Done', force=True)
            except Exception:
                logger.debug('tg.progress.finalize.failed', exc_info=True)
            finally:
                self._final_messages = receipts
            await self._send_exact_chunks(_chunk_text_for_telegram(self.state.answer))
            return
        if not self.state.answer:
            await self._edit_text(header or 'Done', force=True)
            return
        if self._is_full() and header:
            await self._replace_with_chunked_text(_join_text([header, self.state.answer], '\n\n'))
            return

        await self._replace_with_chunked_text(self.state.answer)

    async def complete_without_answer(self) -> None:
        if self.message is None or self._is_none():
            return
        if self._is_minimal():
            await self._delete_message_if_possible(self.message)
            self.message = None
            return
        if self._is_full():
            header = self.state.live_text
        else:
            header = '\n'.join(line for line in self.state.lines if line.strip()).strip()
        try:
            await self._edit_text(header or 'Done', force=True)
        except Exception:
            logger.debug('tg.progress.complete.failed', exc_info=True)

    async def _flush(self, final: bool = False, force: bool = False) -> None:
        if self._is_none() or self._is_minimal():
            return
        if self._is_full():
            header = self.state.live_text
        else:
            header = '\n'.join(line for line in self.state.lines if line.strip())
        body = self.state.answer if final else ''
        text = _join_text([header, body], '\n\n') or '...'
        if not self._is_full() and _rendered_len(text) > MAX_TELEGRAM_TEXT_CHARS:
            return
        await self._edit_text(text, force=force, on_too_long='ignore')

    async def _append_full_block(self, block: str) -> None:
        current = self.state.live_text
        candidate = _join_text([current, block], '\n\n')
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

    async def _send_exact_chunks(self, chunks: list[TelegramText], *, update_current: bool = False,
                                 delivered_messages: list[Message] | None = None) -> Message | None:
        target = self._delivery_target()
        bot = target.get_bot()
        last_message: Message | None = None
        last_text = ''
        for chunk in chunks:
            safe_chunk = chunk or '...'
            last_message = await self._send_text_via_bot(bot, target.chat.id, safe_chunk)
            if delivered_messages is not None:
                delivered_messages.append(last_message)
            last_text = safe_chunk
        if update_current and last_message is not None:
            self.message = last_message
            self.state.live_text = last_text
            self.state.last_render_text = last_text
            self.state.last_render_at = time.monotonic()
        return last_message

    async def _edit_text(self, text: TelegramContent, force: bool = False, on_too_long: str = 'replace') -> None:
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
                self._remember_final_message(self.message)
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
        self._remember_final_message(self.message)
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
            if self._final_messages is not None:
                self._final_messages[:] = [item for item in self._final_messages if item.message_id != message.message_id]

    def _reply_to_message_id(self) -> int | None:
        if not self.reply_to_source_message or self.source_message is None:
            return None
        return self.source_message.message_id

    async def _send_text_via_bot(self, bot: Bot, chat_id: int, text: TelegramContent) -> Message:
        message = await bot_message_safe(bot, 'send_message', chat_id=chat_id, text=text, parse_mode='MarkdownV2', disable_web_page_preview=True, reply_to_message_id=self._reply_to_message_id(), **topic_arguments(self._delivery_target()))
        self._remember_final_message(message)
        return message

    async def _replace_message_with_chunked_text(self, text: TelegramContent) -> None:
        original_message = self.message
        chunks = _chunk_text_for_telegram(text)
        if original_message is None:
            await self._send_exact_chunks(chunks)
            return
        try:
            await bot_message_safe(original_message, 'edit_text', text=(chunks[0] or '...'), parse_mode='MarkdownV2', disable_web_page_preview=True)
        except BadRequest as exc:
            logger.warning('Telegram chunked replacement edit failed; falling back to fresh reply: %s', exc)
            replacement = await self._send_exact_chunks(chunks)
            if replacement is not None:
                self.message = replacement
            if replacement is not None and replacement.message_id != original_message.message_id:
                await self._delete_message_if_possible(original_message)
            return
        self.message = original_message
        self._remember_final_message(original_message)
        if len(chunks) > 1:
            await self._send_exact_chunks(chunks[1:])

    async def _replace_with_chunked_text(self, text: TelegramContent) -> None:
        chunks = _chunk_text_for_telegram(text)
        if self.message is None:
            await self._send_exact_chunks(chunks)
            return
        await self._edit_text(chunks[0] or '...', force=True)
        if len(chunks) > 1:
            await self._send_exact_chunks(chunks[1:])

    async def send_text(self, text: str) -> list[Message]:
        delivered_messages = []
        await self._send_exact_chunks(_chunk_text_for_telegram(text), delivered_messages=delivered_messages)
        return delivered_messages

    async def _fallback_send_text(self, text: TelegramContent) -> None:
        original_message = self.message
        message = await self._send_exact_chunks(_chunk_text_for_telegram(text))
        if message is not None:
            self.message = message
        if message is not None and original_message is not None and message.message_id != original_message.message_id:
            await self._delete_message_if_possible(original_message)

    async def _send_new_text(self, text: str) -> None:
        await self.send_text(text)

    async def send_artifacts(self, artifacts: list[OutboundArtifact]) -> list[dict]:
        from tgchatbot.transports.artifact_delivery import deliver_artifact
        target = self._delivery_target()
        bot = target.get_bot()
        chat_id = target.chat.id
        reply_to_message_id = self._reply_to_message_id()
        receipts = []
        for artifact in artifacts:
            receipts.append(await deliver_artifact(bot, chat_id=chat_id, artifact=artifact,
                reply_to_message_id=reply_to_message_id, **topic_arguments(target)))
        return receipts

    async def send_stickers(self, stickers: list[OutboundSticker]) -> list[dict[str, object]]:
        target = self._delivery_target()
        receipts = []
        for sticker in stickers:
            receipts.append(await deliver_sticker(target.get_bot(), chat_id=target.chat.id,
                sticker=sticker, reply_to_message_id=self._reply_to_message_id(),
                deliveries=self.sticker_delivery, **topic_arguments(target)))
        return receipts
