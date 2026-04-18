"""Pull settled Kalshi markets and record their outcomes.

Kalshi's public API exposes settled markets via ``list_markets(status="settled")``.
Each settled market carries a ``result`` string (``"yes" | "no" | ""``) and a
``settlement_value`` (0, 50, or 100 cents for binary markets; scalar values
for range markets). We map those to our ``Outcome`` enum and upsert to the
``resolutions`` table.

Two knobs worth flagging:

* ``--since`` filters on ``close_time`` so we don't re-ingest the entire
  settled history on every run. Our cron just asks for the last 48h; a
  retro-backfill pass can use a longer window.
* We iterate ``series_ticker=`` one at a time. Kalshi's settled endpoint
  is paginated and filtering by series keeps response sizes tractable.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone

from kalshi_edge.logging_ import get_logger
from kalshi_edge.market.kalshi import KalshiClient
from kalshi_edge.storage.db import Database
from kalshi_edge.types import Outcome, Resolution


log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class ResolutionPull:
    series: str
    n_seen: int
    n_resolved: int   # rows we actually wrote to the resolutions table


def pull_series_resolutions(
    client: KalshiClient,
    db: Database,
    *,
    series_tickers: Iterable[str],
    since: datetime | None = None,
    max_per_series: int | None = None,
) -> list[ResolutionPull]:
    """Walk each series' settled markets and upsert resolutions.

    ``since`` bounds the ``close_time`` so we only touch recently settled
    markets. Passing ``None`` pulls everything the API returns (with the
    usual pagination cap); useful for a one-shot backfill.
    """
    results: list[ResolutionPull] = []
    for series in series_tickers:
        n_seen = 0
        n_resolved = 0
        for mkt in client.iter_markets(
            series_ticker=series,
            status="settled",
            max_items=max_per_series,
        ):
            n_seen += 1
            resolution = _resolution_from_market(mkt)
            if resolution is None:
                continue
            if since is not None and resolution.resolved_at < since:
                # Markets come back newest-first on most endpoints, but we
                # don't rely on that — skip instead of break.
                continue
            db.upsert_resolution(resolution)
            n_resolved += 1
        log.info(
            "resolutions_pulled",
            series=series, n_seen=n_seen, n_resolved=n_resolved,
        )
        results.append(ResolutionPull(series=series, n_seen=n_seen, n_resolved=n_resolved))
    return results


def _resolution_from_market(mkt: dict) -> Resolution | None:
    """Extract a Resolution from a settled-market payload.

    Kalshi exposes several overlapping fields; we prefer ``result`` + one
    of ``expiration_time`` / ``settle_time`` / ``close_time`` as the
    resolution timestamp. Any market without both ``ticker`` and a
    parseable result is skipped (and logged).
    """
    ticker = mkt.get("ticker")
    if not isinstance(ticker, str) or not ticker:
        return None

    result = str(mkt.get("result") or "").lower()
    if result == "yes":
        outcome = Outcome.YES
    elif result == "no":
        outcome = Outcome.NO
    elif result in ("", "void", "voided"):
        # Settled with no winner (cancelled/refunded market).
        outcome = Outcome.VOID
    else:
        log.warning("resolution_unknown_result", ticker=ticker, result=result)
        return None

    resolved_at_raw = (
        mkt.get("expiration_time")
        or mkt.get("settle_time")
        or mkt.get("close_time")
    )
    resolved_at = _parse_iso(resolved_at_raw)
    if resolved_at is None:
        log.warning("resolution_missing_timestamp", ticker=ticker)
        return None

    settled_price = _coerce_price(mkt)
    return Resolution(
        ticker=ticker,
        resolved_at=resolved_at,
        outcome=outcome,
        settled_price=settled_price,
    )


def _parse_iso(val: object) -> datetime | None:
    if not isinstance(val, str) or not val:
        return None
    try:
        dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _coerce_price(mkt: dict) -> float:
    """Best-effort settled-price in cents. Binary markets are 0/50/100."""
    for key in ("settlement_value", "settle_value", "last_price"):
        v = mkt.get(key)
        if isinstance(v, (int, float)):
            return float(v)
    # Fall back to 100 / 0 based on result string; at least directional.
    result = str(mkt.get("result") or "").lower()
    if result == "yes":
        return 100.0
    if result == "no":
        return 0.0
    return 50.0
