"""
Per-user rate limits for the web requests that reach another server: the
new-water wizard's Open-Meteo lookups and the feed setup's fetches.

A sliding window held in this process, which is enough for the single web
process the app deploys as. Each bucket has a limit of hits per window, and
counts each key (a user id) apart. A refused hit is not counted, so a user
who waits out the window always gets back in.

The app holds one `RateLimiter` in `app.state.rate_limiter`; tests swap it
for one with small limits and a fake clock, or call `reset()`.
"""

import threading
import time
from collections import deque

from fastapi import Depends, HTTPException, Request

from db.models import User
from web.auth import require_user

LIMITS: dict[str, tuple[int, float]] = {   # bucket: (hits, per seconds)
    "geocode": (30, 60),
    "point": (30, 60),
    "feed_fetch": (10, 60),
}
MESSAGES = {
    "geocode": "Too many place searches in a minute. Wait a minute and try again.",
    "point": "Too many weather checks in a minute. Wait a minute and try again.",
    "feed_fetch": "Too many feed tests in a minute. Wait a minute and try again.",
}


class RateLimiter:
    def __init__(self, limits: dict[str, tuple[int, float]] = LIMITS, clock=time.monotonic):
        self.limits, self.clock = dict(limits), clock
        self._hits: dict[tuple[str, object], deque] = {}
        self._lock = threading.Lock()

    def allow(self, bucket: str, key) -> bool:
        """Count a hit for `key` in `bucket` and return True, or return False if it is over the limit."""
        limit, window = self.limits[bucket]
        now = self.clock()
        with self._lock:
            hits = self._hits.setdefault((bucket, key), deque())
            while hits and hits[0] <= now - window:
                hits.popleft()
            if len(hits) >= limit:
                return False
            hits.append(now)
            return True

    def reset(self) -> None:
        with self._lock:
            self._hits.clear()


def limited(bucket: str):
    """A route dependency that answers 429 when the signed-in user is over `bucket`'s limit."""
    def check(request: Request, user: User = Depends(require_user)) -> None:
        if not request.app.state.rate_limiter.allow(bucket, user.id):
            raise HTTPException(429, MESSAGES[bucket])
    return check
