"""Local polling readiness: no external API requests or credentials in output."""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
from pathlib import Path
import time

# Six missed five-second event-loop ticks marks an unhealthy process. This
# tolerates brief scheduling delays while detecting a blocked polling loop.
HEARTBEAT_INTERVAL_S = 5
MAX_AGE_S = 30


def health_path() -> Path:
    return Path(os.getenv('APP_HEALTH_FILE', '/tmp/tgchatbot-health.json'))


async def _heartbeat(application) -> None:
    while True:
        if application.running and application.updater and application.updater.running:
            path = health_path()
            partial = path.with_suffix('.tmp')
            partial.write_text(json.dumps({'pid': os.getpid(), 'time': time.time()}))
            partial.replace(path)
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)


async def start_heartbeat(application) -> None:
    # post_init precedes Application.start; the task writes only after both the
    # application and updater report that they are running.
    application.bot_data['_health_task'] = asyncio.create_task(_heartbeat(application))


async def stop_heartbeat(application) -> None:
    task = application.bot_data.pop('_health_task', None)
    if task is not None:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
    # Truncation deliberately invalidates stale state without relying on file
    # deletion or assuming /tmp persists across container recreation.
    health_path().write_text('{}')


def healthy(path: Path | None = None, *, now: float | None = None) -> bool:
    try:
        payload = json.loads((path or health_path()).read_text())
        age = (time.time() if now is None else now) - float(payload['time'])
        pid = int(payload['pid'])
        if pid <= 0 or not 0 <= age <= MAX_AGE_S:
            return False
        os.kill(pid, 0)
        return True
    except (OSError, ValueError, TypeError, KeyError):
        return False


if __name__ == '__main__':
    raise SystemExit(0 if healthy() else 1)
