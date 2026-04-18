"""Category inference + market parsing."""

from __future__ import annotations

import pytest

from kalshi_edge.market.parser import (
    UnsupportedMarket,
    derive_series_ticker,
    infer_category,
    parse_market,
    parse_orderbook,
)
from kalshi_edge.types import Category, MarketStatus


CATS = {
    "weather": {"series_prefixes": ["KXHIGH", "KXLOW"], "title_keywords": ["temperature"]},
    "economics": {"series_prefixes": ["KXCPI"], "title_keywords": ["CPI"]},
    "rates": {"series_prefixes": ["KXFED", "KXFEDDECISION"], "title_keywords": []},
    "politics": {"series_prefixes": ["KXPRES"], "title_keywords": ["election"]},
    "sports": {"series_prefixes": ["KXMLB", "KXMLBGAME"], "title_keywords": []},
}


def test_derive_series_ticker() -> None:
    assert derive_series_ticker("KXMLBGAME-26APR181605CWSATH") == "KXMLBGAME"
    assert derive_series_ticker("KXHIGHNY-26MAR01") == "KXHIGHNY"
    assert derive_series_ticker("") == ""


def test_infer_category_longest_prefix_wins() -> None:
    # KXFEDDECISION (13) beats KXFED (5) because we sort by prefix length.
    assert infer_category("KXFEDDECISION", "Fed decision", CATS) == Category.RATES
    # KXMLBGAME (9) beats KXMLB (5).
    assert infer_category("KXMLBGAME", "", CATS) == Category.SPORTS


def test_infer_category_prefix_beats_title() -> None:
    assert infer_category("KXHIGHNY", "election-day high temp NYC", CATS) == Category.WEATHER


def test_infer_category_title_fallback() -> None:
    assert infer_category("UNKPREFIX", "CPI YoY in March", CATS) == Category.ECONOMICS


def test_infer_category_no_match_is_other() -> None:
    assert infer_category("NOTHING", "unrelated", CATS) == Category.OTHER


def test_parse_market_basic_new_schema() -> None:
    """Parse a realistic Kalshi /markets item (dollar-string prices, *_fp volumes)."""
    raw = {
        "ticker": "KXHIGHNY-26MAR01-T55",
        "event_ticker": "KXHIGHNY-26MAR01",
        # series_ticker NOT present — must be derived from event_ticker.
        "title": "Will NYC high on Mar 1 be above 55°F?",
        "status": "active",
        "open_time": "2026-02-22T14:00:00Z",
        "close_time": "2026-03-01T21:00:00Z",
        "expiration_time": "2026-03-01T21:00:00Z",
        "yes_bid_dollars": "0.4200", "yes_ask_dollars": "0.4600",
        "no_bid_dollars": "0.5400", "no_ask_dollars": "0.5800",
        "last_price_dollars": "0.4400",
        "volume_fp": "1500.00", "volume_24h_fp": "200.00", "open_interest_fp": "800.00",
        "liquidity_dollars": "42.50",
    }
    contract, snap = parse_market(raw, CATS)
    assert contract.ticker == raw["ticker"]
    assert contract.series_ticker == "KXHIGHNY"
    assert contract.category == Category.WEATHER
    assert contract.status == MarketStatus.OPEN
    # 0.42 dollars -> 42 cents.
    assert snap.yes_bid == pytest.approx(42.0)
    assert snap.yes_ask == pytest.approx(46.0)
    assert snap.yes_mid == pytest.approx(44.0)
    assert snap.yes_spread == pytest.approx(4.0)
    assert snap.volume_24h == pytest.approx(200.0)
    assert snap.liquidity == pytest.approx(42.50)


def test_parse_market_rejects_mv_parlay() -> None:
    raw = {
        "ticker": "KXMVECROSSCATEGORY-X",
        "event_ticker": "KXMVECROSSCATEGORY-S123",
        "mve_collection_ticker": "KXMVECROSSCATEGORY-R",
        "status": "active", "title": "parlay",
    }
    with pytest.raises(UnsupportedMarket):
        parse_market(raw, CATS)


def test_parse_market_legacy_int_cents_still_works() -> None:
    """Backwards-compat: pre-existing int-cents payloads still parse."""
    raw = {
        "ticker": "T", "event_ticker": "SER-E", "series_ticker": "SER",
        "title": "t", "status": "active",
        "yes_bid": 42, "yes_ask": 46,
        "volume": 100, "volume_24h": 20, "open_interest": 5,
    }
    _, snap = parse_market(raw, CATS)
    assert snap.yes_bid == 42.0
    assert snap.yes_ask == 46.0


def test_parse_orderbook_levels() -> None:
    ob = parse_orderbook({"yes": [[50, 100], [48, 200]], "no": [[49, 150]]}, "TKR")
    assert [(lv.price, lv.size) for lv in ob.yes] == [(50.0, 100), (48.0, 200)]
    assert [(lv.price, lv.size) for lv in ob.no] == [(49.0, 150)]
