"""Join an NDJSON forecast corpus to resolutions and compute metrics.

This is the ``backtest`` CLI's analytical core. It's intentionally
framework-free (no pandas) so it runs in the CI runner without extra
deps and stays easy to reason about for small-to-medium corpora. If the
corpus ever grows past ~10M forecasts we'd reach for DuckDB on the
NDJSON files directly, but we're nowhere near that.

Metrics produced per (forecaster, category):

* ``n_forecasts`` and ``n_resolved`` — coverage
* ``brier`` and ``log_loss`` — proper scores
* ``reliability`` — calibration-curve bin data
* ``edge_pnl`` — hypothetical flat-stake PnL: we "bet" 1¢ on each
  forecast whose absolute edge vs the contemporaneous market mid
  exceeds a threshold, and settle at the eventual outcome. This is a
  crude sanity check, not a trading simulator — position sizing,
  fees, and liquidity are ignored.

The matching logic is: for each forecast, find the market_snapshot on
the same ticker with the closest ts ≤ forecast ts, and the resolution
for that ticker (if any). Forecasts without a resolution contribute to
coverage but are excluded from scoring.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kalshi_edge.backtest.persistence import load_table
from kalshi_edge.backtest.scoring import (
    brier,
    log_loss,
    mean_brier,
    mean_log_loss,
    reliability_bins,
)


# ----------------------------------------------------------------------
# Data wrangling
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Joined:
    """One forecast joined to its market snapshot + (maybe) a resolution."""
    ticker: str
    forecaster: str
    category: str
    ts: datetime
    p_yes: float
    market_mid_cents: float | None
    outcome: int | None           # 1/0 if resolved, None otherwise
    resolved_at: datetime | None


def join_corpus(data_dir: Path) -> list[_Joined]:
    """Load the NDJSON corpus and join forecasts to snapshots + resolutions."""
    forecasts = load_table(data_dir, "forecasts")
    snapshots = load_table(data_dir, "market_snapshots")
    resolutions = load_table(data_dir, "resolutions")
    contracts = load_table(data_dir, "contracts_lite")

    # ticker → category (so we can bucket Brier by category in analysis)
    category_by_ticker: dict[str, str] = {
        str(c.get("ticker", "")): str(c.get("category") or "uncategorized")
        for c in contracts
    }

    # ticker → sorted list of (ts, mid_cents). Built once so the per-forecast
    # lookup is O(log n) via bisect.
    snaps_by_ticker: dict[str, list[tuple[datetime, float | None]]] = {}
    for s in snapshots:
        ticker = str(s.get("ticker", ""))
        ts = _parse_iso(s.get("ts"))
        if not ticker or ts is None:
            continue
        mid = _snapshot_mid_cents(s)
        snaps_by_ticker.setdefault(ticker, []).append((ts, mid))
    for lst in snaps_by_ticker.values():
        lst.sort(key=lambda x: x[0])

    # ticker → (outcome_int_or_None, resolved_at)
    resolutions_by_ticker: dict[str, tuple[int | None, datetime | None]] = {}
    for r in resolutions:
        ticker = str(r.get("ticker", ""))
        if not ticker:
            continue
        outcome_s = str(r.get("outcome") or "").upper()
        if outcome_s == "YES":
            outcome: int | None = 1
        elif outcome_s == "NO":
            outcome = 0
        else:
            outcome = None  # VOID or unknown — excluded from scoring
        resolutions_by_ticker[ticker] = (outcome, _parse_iso(r.get("resolved_at")))

    joined: list[_Joined] = []
    for f in forecasts:
        p_yes = f.get("p_yes")
        if not isinstance(p_yes, (int, float)):
            continue
        ticker = str(f.get("ticker", ""))
        if not ticker:
            continue
        ts = _parse_iso(f.get("ts"))
        if ts is None:
            continue

        mid = _nearest_mid_at_or_before(snaps_by_ticker.get(ticker, []), ts)
        outcome, resolved_at = resolutions_by_ticker.get(ticker, (None, None))
        joined.append(_Joined(
            ticker=ticker,
            forecaster=str(f.get("forecaster") or "unknown"),
            category=category_by_ticker.get(ticker, "uncategorized"),
            ts=ts,
            p_yes=float(p_yes),
            market_mid_cents=mid,
            outcome=outcome,
            resolved_at=resolved_at,
        ))
    return joined


def _parse_iso(val: Any) -> datetime | None:
    if isinstance(val, datetime):
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)
    if not isinstance(val, str) or not val:
        return None
    try:
        dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _snapshot_mid_cents(s: dict[str, Any]) -> float | None:
    """Best-effort mid from a snapshot row: YES bid/ask if both present."""
    yb = s.get("yes_bid")
    ya = s.get("yes_ask")
    if isinstance(yb, (int, float)) and isinstance(ya, (int, float)) and ya > 0:
        return (float(yb) + float(ya)) / 2.0
    last = s.get("last_price")
    if isinstance(last, (int, float)) and last > 0:
        return float(last)
    return None


def _nearest_mid_at_or_before(
    sorted_snaps: list[tuple[datetime, float | None]],
    ts: datetime,
) -> float | None:
    """Walk the sorted snapshot list; return the latest mid at ts or before."""
    chosen: float | None = None
    for s_ts, mid in sorted_snaps:
        if s_ts > ts:
            break
        if mid is not None:
            chosen = mid
    return chosen


# ----------------------------------------------------------------------
# Metric aggregation
# ----------------------------------------------------------------------


@dataclass(slots=True)
class ForecasterMetrics:
    forecaster: str
    category: str
    n_forecasts: int = 0
    n_resolved: int = 0
    brier: float | None = None
    log_loss: float | None = None
    reliability: list[dict[str, float]] = field(default_factory=list)
    edge_pnl_cents: float = 0.0           # sum of realized cents at flat stake
    edge_bets: int = 0                    # how many forecasts triggered a bet


def compute_metrics(
    joined: Iterable[_Joined],
    *,
    edge_threshold_cents: float = 5.0,
    reliability_bins_n: int = 10,
) -> list[ForecasterMetrics]:
    """Aggregate per (forecaster, category). Unresolved forecasts count toward
    ``n_forecasts`` but not toward scoring. The edge-PnL simulation trades 1
    unit per eligible resolved forecast, long YES if the forecaster's P(YES)
    is ``edge_threshold_cents`` above mid, short YES if it is that far below.
    """
    buckets: dict[tuple[str, str], list[_Joined]] = {}
    for j in joined:
        buckets.setdefault((j.forecaster, j.category), []).append(j)

    out: list[ForecasterMetrics] = []
    for (fc, cat), rows in buckets.items():
        m = ForecasterMetrics(forecaster=fc, category=cat, n_forecasts=len(rows))
        resolved = [r for r in rows if r.outcome is not None]
        m.n_resolved = len(resolved)
        if resolved:
            ps = [r.p_yes for r in resolved]
            ys = [int(r.outcome) for r in resolved]  # type: ignore[arg-type]
            m.brier = mean_brier(ps, ys)
            m.log_loss = mean_log_loss(ps, ys)
            m.reliability = reliability_bins(ps, ys, n_bins=reliability_bins_n)

        # Hypothetical flat-stake PnL. Per resolved forecast with a live
        # market mid, bet 1 unit at |edge| >= threshold, pocket (payoff - cost).
        for r in resolved:
            if r.market_mid_cents is None:
                continue
            fair_cents = r.p_yes * 100.0
            edge = fair_cents - r.market_mid_cents
            if abs(edge) < edge_threshold_cents:
                continue
            if edge > 0:
                cost = r.market_mid_cents
                payoff = 100.0 if r.outcome == 1 else 0.0
            else:
                cost = 100.0 - r.market_mid_cents
                payoff = 100.0 if r.outcome == 0 else 0.0
            m.edge_pnl_cents += (payoff - cost)
            m.edge_bets += 1

        out.append(m)

    out.sort(key=lambda x: (x.forecaster, x.category))
    return out
