"""Universe filter.

Pure function over (markets iterable, config). Keeps selection logic testable
without any network dependency. The scheduler plugs this between the Kalshi
client and the per-category forecaster dispatcher.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from kalshi_edge.config import UniverseFilterConfig
from kalshi_edge.logging_ import get_logger
from kalshi_edge.types import Category, Contract, MarketSnapshot

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class FilteredMarket:
    contract: Contract
    snapshot: MarketSnapshot
    dte_days: float
    spread_cents: float | None


@dataclass
class FilterStats:
    seen: int = 0
    kept: int = 0
    dropped_dte: int = 0
    dropped_volume: int = 0
    dropped_spread: int = 0
    dropped_category: int = 0
    dropped_status: int = 0
    dropped_missing_prices: int = 0


class UniverseFilter:
    """Configured filter. Instantiate once, call on (contract, snapshot) pairs."""

    def __init__(self, config: UniverseFilterConfig) -> None:
        self.config = config
        self._allowed = {Category(c) for c in config.allowed_categories}

    def filter(
        self,
        pairs: Iterable[tuple[Contract, MarketSnapshot]],
        now: datetime | None = None,
    ) -> tuple[list[FilteredMarket], FilterStats]:
        now = now or datetime.now(timezone.utc)
        stats = FilterStats()
        kept: list[FilteredMarket] = []

        for contract, snap in pairs:
            stats.seen += 1

            # Status: only consider open markets for trading signals.
            if contract.status.value != self.config.status:
                stats.dropped_status += 1
                continue

            # Category: must be in our forecaster-supported set.
            if contract.category not in self._allowed:
                stats.dropped_category += 1
                continue

            # DTE: reject near-expiry contracts (noise-dominated, less research value).
            if contract.close_time is None:
                stats.dropped_dte += 1
                continue
            dte_days = (contract.close_time - now).total_seconds() / 86400.0
            if dte_days < self.config.min_dte_days:
                stats.dropped_dte += 1
                continue

            # Volume threshold: below this, the market is unlikely to absorb
            # a meaningful position and prices are stale.
            total_volume = max(snap.volume_24h or 0.0, snap.volume or 0.0)
            if total_volume < self.config.min_volume:
                stats.dropped_volume += 1
                continue

            # Spread: wide book means transaction cost swamps plausible edge.
            if snap.yes_bid is None or snap.yes_ask is None:
                stats.dropped_missing_prices += 1
                continue
            spread = snap.yes_ask - snap.yes_bid
            if spread > self.config.max_spread_cents:
                stats.dropped_spread += 1
                continue

            kept.append(FilteredMarket(
                contract=contract, snapshot=snap,
                dte_days=dte_days, spread_cents=spread,
            ))
            stats.kept += 1

        log.info(
            "universe_filter",
            seen=stats.seen, kept=stats.kept,
            dropped_status=stats.dropped_status,
            dropped_category=stats.dropped_category,
            dropped_dte=stats.dropped_dte,
            dropped_volume=stats.dropped_volume,
            dropped_spread=stats.dropped_spread,
            dropped_missing_prices=stats.dropped_missing_prices,
        )
        return kept, stats
