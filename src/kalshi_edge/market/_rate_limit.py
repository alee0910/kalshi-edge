"""Trivial token-bucket limiter.

Kalshi's public tier is documented in req/s. We don't need a distributed
limiter; a single-process leaky bucket is sufficient for this system's
scheduler-driven access pattern.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    def __init__(self, rate_per_second: float, burst: int | None = None) -> None:
        if rate_per_second <= 0:
            raise ValueError("rate_per_second must be > 0")
        self._rate = rate_per_second
        self._capacity = float(burst if burst is not None else max(1, int(rate_per_second)))
        self._tokens = self._capacity
        self._last = time.monotonic()
        self._lock = threading.Lock()

    def acquire(self, tokens: float = 1.0) -> None:
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._tokens = min(self._capacity, self._tokens + elapsed * self._rate)
            self._last = now
            if self._tokens >= tokens:
                self._tokens -= tokens
                return
            deficit = tokens - self._tokens
            sleep = deficit / self._rate
        time.sleep(sleep)
        with self._lock:
            self._tokens = 0.0
            self._last = time.monotonic()
