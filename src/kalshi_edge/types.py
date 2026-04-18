"""Domain types for markets, contracts, orderbooks, and resolutions.

These are the structures that flow between modules. Forecast-side types
(distributions, ForecastResult) live in forecast.py to keep this file focused
on the market surface.

All monetary prices are represented in CENTS as floats (e.g. 62.5 for 62.5¢).
Kalshi itself uses integer cents for bid/ask; we widen to float so we can
carry mids and spreads cleanly through the ranker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class Category(str, Enum):
    WEATHER = "weather"
    ECONOMICS = "economics"
    RATES = "rates"
    SPORTS = "sports"
    POLITICS = "politics"
    EARNINGS = "earnings"
    OTHER = "other"


class MarketStatus(str, Enum):
    OPEN = "open"
    CLOSED = "closed"
    SETTLED = "settled"
    UNOPENED = "unopened"
    UNKNOWN = "unknown"


class Outcome(str, Enum):
    YES = "YES"
    NO = "NO"
    VOID = "VOID"


@dataclass(frozen=True, slots=True)
class Contract:
    """Parsed and categorized Kalshi contract.

    Combines metadata from Kalshi's market + event objects with our own
    category tag and structured resolution criteria. A Contract is the
    single unit a Forecaster consumes.
    """

    ticker: str
    event_ticker: str
    series_ticker: str
    title: str
    subtitle: str | None
    category: Category

    open_time: datetime | None
    close_time: datetime | None
    expiration_time: datetime | None
    status: MarketStatus

    # Raw contract rules — free text from Kalshi, used by the parser.
    rules_primary: str | None = None
    rules_secondary: str | None = None

    # Structured resolution criteria, populated by market/parser.py.
    # Shape is category-specific. Keep as dict rather than a union so
    # forecasters can add fields without churning this type.
    resolution_criteria: dict[str, Any] = field(default_factory=dict)

    # Free-form metadata from Kalshi we want to keep around.
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MarketSnapshot:
    """Point-in-time market state.

    ``ts`` is the timestamp we observed the market, not a Kalshi server time.
    ``last_trade_ts`` lets the ranker penalize stale markets where the
    mid may not reflect real market clearing.
    """

    ticker: str
    ts: datetime

    yes_bid: float | None
    yes_ask: float | None
    no_bid: float | None
    no_ask: float | None

    last_price: float | None
    last_trade_ts: datetime | None

    volume: float
    volume_24h: float
    open_interest: float
    # Depth at the top of book; dollar liquidity estimate.
    liquidity: float | None = None

    @property
    def yes_mid(self) -> float | None:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return 0.5 * (self.yes_bid + self.yes_ask)

    @property
    def yes_spread(self) -> float | None:
        if self.yes_bid is None or self.yes_ask is None:
            return None
        return self.yes_ask - self.yes_bid


@dataclass(frozen=True, slots=True)
class OrderBookLevel:
    price: float   # cents
    size: int      # contracts


@dataclass(frozen=True, slots=True)
class OrderBook:
    ticker: str
    ts: datetime
    yes: list[OrderBookLevel]   # bids, high-to-low
    no: list[OrderBookLevel]


@dataclass(frozen=True, slots=True)
class Resolution:
    ticker: str
    resolved_at: datetime
    outcome: Outcome
    settled_price: float  # 100 YES / 0 NO / partial for scalar markets
