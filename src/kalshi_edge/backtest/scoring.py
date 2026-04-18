"""Proper scoring rules for binary probabilistic forecasts.

Brier and log-loss are the two scores the backtest pipeline reports per
forecaster, both ``lower = better``. We clip P(YES) to ``[eps, 1-eps]``
before taking logs to avoid a single 0/1 forecast blowing log-loss to
infinity; ``eps = 1e-9`` is small enough that any honest well-calibrated
forecaster doesn't feel the clip, but large enough to keep the score
finite when a model incorrectly asserts certainty.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


_EPS = 1e-9


def brier(p: float, outcome: int) -> float:
    """Brier score for a single binary forecast: ``(p - y)^2``."""
    y = 1 if outcome else 0
    d = float(p) - y
    return d * d


def log_loss(p: float, outcome: int) -> float:
    """Negative log-likelihood of ``outcome`` under Bernoulli(p), clipped.

    ``-log p`` when outcome is 1, ``-log(1-p)`` otherwise.
    """
    p = min(max(float(p), _EPS), 1.0 - _EPS)
    return -math.log(p) if outcome else -math.log(1.0 - p)


def mean_brier(ps: Iterable[float], ys: Iterable[int]) -> float | None:
    scores = [brier(p, y) for p, y in zip(ps, ys)]
    return sum(scores) / len(scores) if scores else None


def mean_log_loss(ps: Iterable[float], ys: Iterable[int]) -> float | None:
    scores = [log_loss(p, y) for p, y in zip(ps, ys)]
    return sum(scores) / len(scores) if scores else None


def reliability_bins(
    ps: Iterable[float],
    ys: Iterable[int],
    *,
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Return per-decile reliability data: {mean_p, empirical_rate, n}.

    The usual calibration diagnostic: bucket forecasts by predicted P(YES)
    into ``n_bins`` equal-width bins on [0, 1]. An honest forecaster has
    ``mean_p ≈ empirical_rate`` in every bin. Empty bins are dropped.
    """
    ps_l = list(ps)
    ys_l = list(ys)
    if not ps_l:
        return []

    width = 1.0 / n_bins
    buckets: list[tuple[list[float], list[int]]] = [
        ([], []) for _ in range(n_bins)
    ]
    for p, y in zip(ps_l, ys_l):
        p = min(max(float(p), 0.0), 1.0)
        idx = min(int(p / width), n_bins - 1)
        buckets[idx][0].append(p)
        buckets[idx][1].append(1 if y else 0)

    out: list[dict[str, float]] = []
    for b_ps, b_ys in buckets:
        n = len(b_ps)
        if n == 0:
            continue
        out.append({
            "mean_p": sum(b_ps) / n,
            "empirical_rate": sum(b_ys) / n,
            "n": n,
        })
    return out
