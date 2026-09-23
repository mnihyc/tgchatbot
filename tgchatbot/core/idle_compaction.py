"""Disposable idle timers; the runtime owns compaction and its session lock."""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from collections.abc import Callable

from tgchatbot.core.events import RuntimeEvent
from tgchatbot.core.runtime import AgentRuntime, EventCallback

logger = logging.getLogger(__name__)


@dataclass
class _Session:
    timer: asyncio.Task | None = None
    operation: asyncio.Task | None = None
    emit: EventCallback | None = None
    latest_event: RuntimeEvent | None = None


class IdleCompaction:
    def __init__(self, runtime: AgentRuntime, *, busy: Callable[[str], bool]) -> None:
        self.runtime = runtime
        self.busy = busy
        self.sessions: dict[str, _Session] = {}
        self.closed = False

    def touch(self, session_id: str) -> None:
        """Real chat activity restarts the timer, never an in-flight compaction."""
        if self.closed:
            return
        state = self.sessions.setdefault(session_id, _Session())
        if state.timer is not None:
            state.timer.cancel()
        state.timer = asyncio.create_task(self._wait(session_id, state))

    async def _wait(self, session_id: str, state: _Session) -> None:
        try:
            settings = await self.runtime.store.get_or_create_session(
                session_id, self.runtime.config.default_session_settings())
            if not self.runtime._effective_compact_idle_trigger_tokens(settings):
                return
            await asyncio.sleep(self.runtime._effective_compact_idle_seconds(settings))
            if self.busy(session_id) or state.operation is not None:
                return
            state.latest_event = None
            state.operation = asyncio.create_task(self._run(session_id, state))
            state.operation.add_done_callback(lambda task: self._finished(session_id, state, task))
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception('compact.idle.schedule_failed sid=%s', session_id)
        finally:
            if state.timer is asyncio.current_task():
                state.timer = None

    async def _run(self, session_id: str, state: _Session) -> dict:
        async def emit(event: RuntimeEvent) -> None:
            state.latest_event = event
            if state.emit is not None:
                await self._forward_progress(session_id, state.emit, event)
        return await self.runtime.compact_idle_context(session_id=session_id, emit=emit)

    @staticmethod
    async def _forward_progress(session_id: str, emit: EventCallback, event: RuntimeEvent) -> None:
        try:
            await emit(event)
        except Exception:
            # Telegram status delivery does not own paid summary work.
            logger.warning('compact.idle.progress_failed sid=%s', session_id, exc_info=True)

    def _finished(self, session_id: str, state: _Session, task: asyncio.Task) -> None:
        if state.operation is task:
            state.operation = None
        if not task.cancelled():
            error = task.exception()
            if error is not None:
                logger.warning('compact.idle.failed sid=%s error=%s', session_id, type(error).__name__)
            else:
                logger.info('compact.idle.done sid=%s result=%s', session_id, task.result())

    async def join(self, session_id: str, emit: EventCallback | None = None) -> bool:
        state = self.sessions.get(session_id)
        if state is None or state.operation is None:
            return False
        operation = state.operation
        state.emit = emit
        try:
            if emit is not None and state.latest_event is not None:
                await self._forward_progress(session_id, emit, state.latest_event)
            # Debouncing/cancelling a waiting reply must not cancel the paid
            # operation. Reset and shutdown use cancel() explicitly.
            await asyncio.shield(operation)
            return True
        finally:
            if state.emit is emit:
                state.emit = None

    async def cancel(self, session_id: str) -> None:
        state = self.sessions.pop(session_id, None)
        if state is None:
            return
        tasks = [task for task in (state.timer, state.operation) if task is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    async def close(self) -> None:
        self.closed = True
        await asyncio.gather(*(self.cancel(session_id) for session_id in list(self.sessions)))
