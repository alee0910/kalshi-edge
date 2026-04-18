"""Rate limiter correctness properties."""

from __future__ import annotations

import time

from kalshi_edge.market._rate_limit import RateLimiter


def test_burst_then_sleep() -> None:
    # 10 rps, burst 10. First 10 acquires are immediate; the 11th must wait.
    rl = RateLimiter(rate_per_second=10.0, burst=10)
    t0 = time.monotonic()
    for _ in range(10):
        rl.acquire()
    t1 = time.monotonic()
    assert t1 - t0 < 0.05, "burst should be near-instant"
    rl.acquire()  # should sleep ~100ms
    t2 = time.monotonic()
    # Give a wide tolerance because of CI jitter.
    assert t2 - t1 >= 0.08


def test_rejects_nonpositive_rate() -> None:
    import pytest
    with pytest.raises(ValueError):
        RateLimiter(rate_per_second=0)
