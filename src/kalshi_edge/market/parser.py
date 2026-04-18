"""Map raw Kalshi JSON into our typed Contract / MarketSnapshot / OrderBook.

Kalshi wire format notes (verified against live `/markets` responses
on 2026-04-18):

* Prices come as decimal-string **dollars** under keys ending in `_dollars`
  (e.g. ``yes_bid_dollars: "0.6100"``) — not integer cents. We convert to
  cents internally to match the rest of the system.
* Sizes / volumes / liquidity come as decimal-string floats under keys ending
  in ``_fp`` or ``_dollars`` (``volume_fp``, ``volume_24h_fp``,
  ``open_interest_fp``, ``liquidity_dollars``).
* ``series_ticker`` is NOT in the ``/markets`` list response; only present
  in ``/events/{ticker}``. We derive it from the ``event_ticker`` prefix
  (the chunk before the first hyphen). Reliable for leaf markets because
  Kalshi structures event tickers as ``{SERIES}-{event-suffix}``.
* **Multi-venue parlay markets** (``KXMVE*`` prefix, ``mve_collection_ticker``
  field) aggregate legs across unrelated events. They have no forecastable
  DGP for us — skipped via the explicit ``UnsupportedMarket`` exception so
  the CLI can count them separately from parse failures.

The parser is intentionally permissive: we degrade rather than crash on
field drift, as long as the minimum (ticker, event_ticker) is present.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable

from kalshi_edge.types import (
    Category,
    Contract,
    MarketSnapshot,
    MarketStatus,
    OrderBook,
    OrderBookLevel,
)


class UnsupportedMarket(Exception):
    """Raised for markets we intentionally refuse (e.g. MV parlays)."""


_KALSHI_STATUS_MAP = {
    "active": MarketStatus.OPEN,
    "open": MarketStatus.OPEN,
    "closed": MarketStatus.CLOSED,
    "settled": MarketStatus.SETTLED,
    "finalized": MarketStatus.SETTLED,
    "determined": MarketStatus.SETTLED,
    "unopened": MarketStatus.UNOPENED,
}


def _parse_ts(v: Any) -> datetime | None:
    if v is None or v == "":
        return None
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    s = str(v).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return None


def _parse_number(v: Any) -> float | None:
    """Parse Kalshi's stringly-typed numerics. Empty / invalid -> None."""
    if v is None or v == "":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _dollars_to_cents(v: Any) -> float | None:
    d = _parse_number(v)
    return None if d is None else d * 100.0


def derive_series_ticker(event_ticker: str) -> str:
    """First chunk of the event_ticker, e.g. 'KXMLBGAME-26APR181605CWSATH' -> 'KXMLBGAME'.

    Kalshi's event-ticker convention encodes the series as the hyphen-delimited
    prefix. If that convention changes we'll need the /events lookup, but for
    now this is two orders of magnitude cheaper than one API call per market.
    """
    return event_ticker.split("-", 1)[0] if event_ticker else ""


def infer_category(
    series_ticker: str,
    title: str,
    categories_config: dict[str, dict[str, list[str]]],
) -> Category:
    """Longest-prefix wins; then title-keyword fallback."""
    s = series_ticker.upper()
    t = title.lower()

    prefix_hits: list[tuple[int, Category]] = []
    for name, cfg in categories_config.items():
        try:
            category = Category(name)
        except ValueError:
            continue
        for prefix in cfg.get("series_prefixes", []) or []:
            if s.startswith(str(prefix).upper()):
                prefix_hits.append((len(prefix), category))
    if prefix_hits:
        prefix_hits.sort(key=lambda x: x[0], reverse=True)
        return prefix_hits[0][1]

    for name, cfg in categories_config.items():
        try:
            category = Category(name)
        except ValueError:
            continue
        for kw in cfg.get("title_keywords", []) or []:
            if str(kw).lower() in t:
                return category

    return Category.OTHER


def parse_market(
    market: dict[str, Any],
    categories_config: dict[str, dict[str, list[str]]],
) -> tuple[Contract, MarketSnapshot]:
    """Split a Kalshi /markets item into Contract + MarketSnapshot.

    Raises:
        UnsupportedMarket: for multi-venue parlay aggregates (no DGP for us).
        ValueError: for markets missing minimum required fields.
    """
    ticker = str(market.get("ticker") or "")
    event_ticker = str(market.get("event_ticker") or "")
    if not ticker or not event_ticker:
        raise ValueError(f"Market missing ticker/event_ticker: ticker={ticker!r}")

    # Skip MV parlays. These combine legs from unrelated sources; we model leaves.
    if market.get("mve_collection_ticker") or event_ticker.startswith("KXMVE"):
        raise UnsupportedMarket(f"{ticker} is a multi-venue parlay market")

    series_ticker = str(market.get("series_ticker") or derive_series_ticker(event_ticker))

    status = _KALSHI_STATUS_MAP.get(str(market.get("status") or "").lower(), MarketStatus.UNKNOWN)
    category = infer_category(series_ticker, str(market.get("title") or ""), categories_config)

    contract = Contract(
        ticker=ticker,
        event_ticker=event_ticker,
        series_ticker=series_ticker,
        title=str(market.get("title") or ""),
        subtitle=market.get("yes_sub_title") or market.get("subtitle"),
        category=category,
        open_time=_parse_ts(market.get("open_time")),
        close_time=_parse_ts(market.get("close_time")),
        expiration_time=_parse_ts(
            market.get("expiration_time") or market.get("expected_expiration_time")
        ),
        status=status,
        rules_primary=market.get("rules_primary"),
        rules_secondary=market.get("rules_secondary"),
        resolution_criteria={},   # populated by category-specific rule parsers later
        raw=dict(market),
    )

    # Kalshi's wire format (as of 2026-04-18):
    #   *_dollars  -> decimal-string USD. We store cents, so multiply by 100.
    #   *_fp       -> decimal-string float (volume, OI).
    #   liquidity  -> dollars, stored as dollars (not cents) because it's
    #                 a notional depth, not a per-contract price.
    def _cents(*keys: str) -> float | None:
        for k in keys:
            if k in market:
                return _dollars_to_cents(market[k])
        return None

    def _num(*keys: str) -> float | None:
        for k in keys:
            if k in market:
                return _parse_number(market[k])
        return None

    yes_bid = _cents("yes_bid_dollars")
    yes_ask = _cents("yes_ask_dollars")
    no_bid = _cents("no_bid_dollars")
    no_ask = _cents("no_ask_dollars")
    last_price = _cents("last_price_dollars")
    # Legacy int-cents fields (used by some older tests/fixtures).
    if yes_bid is None and "yes_bid" in market:
        yes_bid = _parse_number(market["yes_bid"])
    if yes_ask is None and "yes_ask" in market:
        yes_ask = _parse_number(market["yes_ask"])
    if no_bid is None and "no_bid" in market:
        no_bid = _parse_number(market["no_bid"])
    if no_ask is None and "no_ask" in market:
        no_ask = _parse_number(market["no_ask"])
    if last_price is None and "last_price" in market:
        last_price = _parse_number(market["last_price"])

    volume = _num("volume_fp", "volume") or 0.0
    volume_24h = _num("volume_24h_fp", "volume_24h") or 0.0
    open_interest = _num("open_interest_fp", "open_interest") or 0.0
    liquidity = _num("liquidity_dollars", "liquidity")

    snap = MarketSnapshot(
        ticker=ticker,
        ts=datetime.now(timezone.utc),
        yes_bid=yes_bid,
        yes_ask=yes_ask,
        no_bid=no_bid,
        no_ask=no_ask,
        last_price=last_price,
        last_trade_ts=_parse_ts(market.get("last_trade_time") or market.get("latest_trade_time")),
        volume=volume,
        volume_24h=volume_24h,
        open_interest=open_interest,
        liquidity=liquidity,
    )
    return contract, snap


def parse_orderbook(ob: dict[str, Any], ticker: str) -> OrderBook:
    def _levels(raw: Iterable[Any] | None) -> list[OrderBookLevel]:
        if not raw:
            return []
        out: list[OrderBookLevel] = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                price, size = item[0], item[1]
            elif isinstance(item, dict):
                price, size = item.get("price"), item.get("size") or item.get("quantity")
            else:
                continue
            if price is None or size is None:
                continue
            out.append(OrderBookLevel(price=float(price), size=int(size)))
        return out

    return OrderBook(
        ticker=ticker,
        ts=datetime.now(timezone.utc),
        yes=_levels(ob.get("yes")),
        no=_levels(ob.get("no")),
    )
