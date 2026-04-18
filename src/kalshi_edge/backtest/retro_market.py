"""Retroactive calibration of the Kalshi market itself.

For domains where we can't run the forecaster on historical data (sports,
most notably — the-odds-api free tier has no archive), we can still score
*the market's own closing price* against the eventual outcome. That
gives us the baseline the forecaster must beat: if the Kalshi close price
has a Brier of 0.15 on NBA game lines, any forecaster claiming edge
should be scoring below that.

Pipeline:

1. Walk settled markets for the requested series + date range.
2. For each, pull candlesticks ending at ``close_time`` and grab the
   last traded / closing bid-ask midpoint as the market's closing
   probability.
3. Brier / log-loss against the settled result.

No forecasts are involved — this is a property of the market, not of us.
It runs on demand (not every cron) because it costs one candlesticks
API call per settled ticker.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from kalshi_edge.backtest.scoring import brier, log_loss
from kalshi_edge.logging_ import get_logger
from kalshi_edge.market.kalshi import KalshiClient


log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RetroRecord:
    ticker: str
    series: str
    resolved_at: datetime
    outcome: int                   # 1 YES, 0 NO
    market_close_p_yes: float      # on [0, 1]
    brier: float
    log_loss: float


@dataclass(frozen=True, slots=True)
class RetroSummary:
    series: str
    n_markets: int
    n_scored: int
    mean_brier: float | None
    mean_log_loss: float | None
    records: tuple[RetroRecord, ...]


def retro_calibrate_series(
    client: KalshiClient,
    series: str,
    *,
    since: datetime,
    until: datetime | None = None,
    max_markets: int | None = None,
    candle_minutes: int = 60,
) -> RetroSummary:
    """Score Kalshi's own closing prices for one series over a date window.

    Binary (YES/NO) markets only — scalar/range markets require a different
    settlement-value→probability mapping. Markets with no close candlestick
    in the window are skipped with a log (not a failure), so a partial
    outage doesn't kill the whole summary.
    """
    until = until or datetime.now(timezone.utc)
    n_markets = 0
    records: list[RetroRecord] = []

    for mkt in client.iter_markets(
        series_ticker=series,
        status="settled",
        max_items=max_markets,
    ):
        n_markets += 1
        ticker = mkt.get("ticker")
        if not isinstance(ticker, str) or not ticker:
            continue

        close_time = _parse_iso(mkt.get("close_time"))
        resolved_at = _parse_iso(
            mkt.get("expiration_time")
            or mkt.get("settle_time")
            or mkt.get("close_time")
        )
        if close_time is None or resolved_at is None:
            continue
        if close_time < since or close_time > until:
            continue

        outcome = _binary_outcome(mkt)
        if outcome is None:
            continue  # void / scalar — not in this backtest

        try:
            candles = client.get_candlesticks(
                series,
                ticker,
                start_ts=close_time - timedelta(hours=6),
                end_ts=close_time + timedelta(minutes=5),
                period_interval=candle_minutes,
            )
        except Exception as e:  # noqa: BLE001 — log + continue, don't fail the batch
            log.warning("retro_candle_error", ticker=ticker, err=str(e)[:200])
            continue

        p_close = _close_p_yes_from_candles(candles)
        if p_close is None:
            continue

        records.append(RetroRecord(
            ticker=ticker,
            series=series,
            resolved_at=resolved_at,
            outcome=outcome,
            market_close_p_yes=p_close,
            brier=brier(p_close, outcome),
            log_loss=log_loss(p_close, outcome),
        ))

    n_scored = len(records)
    mean_b = sum(r.brier for r in records) / n_scored if n_scored else None
    mean_ll = sum(r.log_loss for r in records) / n_scored if n_scored else None
    log.info(
        "retro_calibration_done",
        series=series, n_markets=n_markets, n_scored=n_scored,
        mean_brier=mean_b, mean_log_loss=mean_ll,
    )
    return RetroSummary(
        series=series,
        n_markets=n_markets,
        n_scored=n_scored,
        mean_brier=mean_b,
        mean_log_loss=mean_ll,
        records=tuple(records),
    )


def _binary_outcome(mkt: dict[str, Any]) -> int | None:
    result = str(mkt.get("result") or "").lower()
    if result == "yes":
        return 1
    if result == "no":
        return 0
    return None


def _close_p_yes_from_candles(candles: Iterable[dict[str, Any]]) -> float | None:
    """Return the last candle's implied P(YES) on [0, 1].

    Prefers the midpoint of the YES bid/ask if present (``yes_bid.close`` /
    ``yes_ask.close``); falls back to the ``price.close`` field in cents.
    Kalshi's candle schema nests OHLC under named keys — we read whichever
    key is populated.
    """
    last = None
    for c in candles:
        last = c
    if last is None:
        return None

    mid_cents = _mid_from_candle(last)
    if mid_cents is None:
        return None
    return max(0.0, min(1.0, mid_cents / 100.0))


def _mid_from_candle(candle: dict[str, Any]) -> float | None:
    yb = _candle_close(candle.get("yes_bid"))
    ya = _candle_close(candle.get("yes_ask"))
    if yb is not None and ya is not None and ya > 0:
        return (yb + ya) / 2.0
    price = _candle_close(candle.get("price"))
    if price is not None:
        return price
    return None


def _candle_close(block: Any) -> float | None:
    if isinstance(block, dict):
        v = block.get("close")
        return float(v) if isinstance(v, (int, float)) else None
    if isinstance(block, (int, float)):
        return float(block)
    return None


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


__all__ = [
    "RetroRecord",
    "RetroSummary",
    "retro_calibrate_series",
]
