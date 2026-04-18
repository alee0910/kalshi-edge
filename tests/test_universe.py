"""Universe-filter threshold logic."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from kalshi_edge.config import UniverseFilterConfig
from kalshi_edge.market.universe import UniverseFilter
from kalshi_edge.types import Category, Contract, MarketSnapshot, MarketStatus


NOW = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)


def _make(
    *, category: Category = Category.WEATHER,
    status: MarketStatus = MarketStatus.OPEN,
    dte: float = 10.0, volume: float = 1000.0,
    yes_bid: float | None = 40.0, yes_ask: float | None = 44.0,
) -> tuple[Contract, MarketSnapshot]:
    close_time = NOW + timedelta(days=dte)
    contract = Contract(
        ticker="T", event_ticker="E", series_ticker="S",
        title="", subtitle=None, category=category,
        open_time=None, close_time=close_time, expiration_time=close_time,
        status=status,
    )
    snap = MarketSnapshot(
        ticker="T", ts=NOW,
        yes_bid=yes_bid, yes_ask=yes_ask, no_bid=None, no_ask=None,
        last_price=None, last_trade_ts=None,
        volume=volume, volume_24h=volume, open_interest=0.0,
    )
    return contract, snap


def _cfg() -> UniverseFilterConfig:
    return UniverseFilterConfig(
        min_dte_days=7, min_volume=500, max_spread_cents=8,
        allowed_categories=["weather", "economics", "rates"],
        status="open",
    )


def test_keeps_passing_market() -> None:
    kept, stats = UniverseFilter(_cfg()).filter([_make()], now=NOW)
    assert stats.kept == 1 and len(kept) == 1
    assert kept[0].dte_days == 10.0
    assert kept[0].spread_cents == 4.0


def test_drops_short_dte() -> None:
    _, stats = UniverseFilter(_cfg()).filter([_make(dte=3)], now=NOW)
    assert stats.dropped_dte == 1 and stats.kept == 0


def test_drops_low_volume() -> None:
    _, stats = UniverseFilter(_cfg()).filter([_make(volume=100)], now=NOW)
    assert stats.dropped_volume == 1


def test_drops_wide_spread() -> None:
    _, stats = UniverseFilter(_cfg()).filter([_make(yes_bid=30, yes_ask=42)], now=NOW)
    assert stats.dropped_spread == 1


def test_drops_missing_prices() -> None:
    _, stats = UniverseFilter(_cfg()).filter([_make(yes_bid=None, yes_ask=None)], now=NOW)
    assert stats.dropped_missing_prices == 1


def test_drops_unsupported_category() -> None:
    _, stats = UniverseFilter(_cfg()).filter(
        [_make(category=Category.EARNINGS)], now=NOW
    )
    assert stats.dropped_category == 1


def test_drops_closed_status() -> None:
    _, stats = UniverseFilter(_cfg()).filter(
        [_make(status=MarketStatus.CLOSED)], now=NOW
    )
    assert stats.dropped_status == 1
