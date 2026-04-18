"""Sports forecaster — intentional abstain.

Real sports models need either a rating-system history (Elo per team,
Massey for margin-based ratings) or a paid odds feed. The-odds-api has a
free tier (500 req/mo) but its coverage and cadence drop out quickly
against Kalshi's market volume, and a partially-populated sports
forecaster would hand out non-null p_yes on tickers we have no calibrated
view of.

Given the *no fake precision* principle, we abstain. The abstention is
explicit so the UI can show "no forecaster" instead of a silent gap.
"""

from __future__ import annotations

from kalshi_edge.forecast import ForecastResult
from kalshi_edge.forecasters.base import Forecaster
from kalshi_edge.types import Category, Contract, MarketSnapshot


class SportsAbstainForecaster(Forecaster):
    name = "sports_abstain"
    version = "1"

    @property
    def category(self) -> Category:
        return Category.SPORTS

    def _forecast_impl(
        self, contract: Contract, snapshot: MarketSnapshot | None,
    ) -> ForecastResult:
        return self._null(
            contract,
            "sports forecaster deferred: needs an Elo/Massey pipeline or "
            "odds-feed aggregator before shipping calibrated probabilities.",
        )
