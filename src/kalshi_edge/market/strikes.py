"""Parse Kalshi's YES-resolution criterion from structured strike fields.

Kalshi markets expose the resolution direction in three fields on the
``/markets`` payload:

    strike_type    one of: less, less_or_equal, greater, greater_or_equal,
                   between, structured, yes_no, ...
    cap_strike     upper bound (used by "less" / "less_or_equal" / "between")
    floor_strike   lower bound (used by "greater" / "greater_or_equal" / "between")

This is the *authoritative* source of the YES comparator. We intentionally
do not infer direction from the ticker suffix or from ``rules_primary``
prose — both have surprised us before:

- Weather tickers use a bare ``-T{N}`` suffix with no sign convention.
  The underlying contract can be ``< N`` or ``> N`` depending solely on
  ``strike_type`` — the ticker tells you the threshold, not the side.
  Defaulting to "above" once flipped the sign on an entire series.
- Rules-text parsing is brittle across phrasing drift.

If ``strike_type`` is missing or a shape we don't recognize (e.g.
"structured" range buckets), we return None so the caller can abstain
rather than guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


Direction = Literal["below", "above", "between"]


@dataclass(frozen=True, slots=True)
class YesCriterion:
    """How to evaluate P(YES) against an underlying continuous forecast.

    For ``below``: YES iff value <= high (treating strict < and <= as equal
    for continuous distributions — the difference is measure-zero and the
    NWS/BLS-reported integer already absorbs any boundary ambiguity).

    For ``above``: YES iff value >= low.

    For ``between``: YES iff low <= value <= high.
    """

    direction: Direction
    low: float | None = None
    high: float | None = None

    def bound(self) -> float:
        """Single threshold for directional criteria. Error on ``between``."""
        if self.direction == "below":
            assert self.high is not None
            return self.high
        if self.direction == "above":
            assert self.low is not None
            return self.low
        raise ValueError("bound() is undefined for 'between' criteria")


def parse_yes_criterion(raw: dict[str, Any]) -> YesCriterion | None:
    stype = str(raw.get("strike_type") or "").lower().strip()
    cap = _num(raw.get("cap_strike"))
    floor = _num(raw.get("floor_strike"))
    if stype in ("less", "less_or_equal"):
        if cap is None:
            return None
        return YesCriterion(direction="below", high=cap)
    if stype in ("greater", "greater_or_equal"):
        if floor is None:
            return None
        return YesCriterion(direction="above", low=floor)
    if stype == "between":
        if floor is None or cap is None:
            return None
        return YesCriterion(direction="between", low=floor, high=cap)
    return None


def _num(v: Any) -> float | None:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None
