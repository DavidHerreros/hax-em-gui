"""Server-push event bus: job status transitions, fanned out over /ws/events.

One socket per browser tab replaces waiting out the 3–5 s polling interval for
status changes: the views keep their pollers (the fallback when the socket is
down), but an event kicks an immediate refresh, and the notifications module
watches the same stream. Publishers are synchronous and never block: a slow
subscriber's queue simply drops events — correctness comes from the poll, the
socket only adds immediacy.
"""

from __future__ import annotations

import asyncio
from typing import Any

_subs: set[asyncio.Queue] = set()


def subscribe() -> asyncio.Queue:
    q: asyncio.Queue = asyncio.Queue(maxsize=500)
    _subs.add(q)
    return q


def unsubscribe(q: asyncio.Queue) -> None:
    _subs.discard(q)


def publish(event: dict[str, Any]) -> None:
    for q in list(_subs):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass   # slow client: drop — its poller still converges
