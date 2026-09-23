"""Transport retry timing; each workflow owns its retry count and outcomes."""
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import math

import httpx


def transient_retry_delay(exc: Exception, delay: float) -> float | None:
    if isinstance(exc, httpx.TransportError):
        return delay
    if not isinstance(exc, httpx.HTTPStatusError):
        return None
    response = exc.response
    if response.status_code != 429 and not 500 <= response.status_code <= 599:
        return None
    retry_after = response.headers.get('retry-after', '').strip()
    if retry_after:
        try:
            seconds = float(retry_after)
        except ValueError:
            try:
                when = parsedate_to_datetime(retry_after)
                seconds = (when - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                seconds = 0
        if math.isfinite(seconds):
            delay = max(delay, seconds)
    return delay
