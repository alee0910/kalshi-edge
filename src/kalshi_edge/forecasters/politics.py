"""Politics forecaster — intentional abstain.

Election markets want a polling-average backbone (538-style) plus a
fundamentals prior. The 538 archive is available but mid-cycle pipelines
(daily ingest of new polls, pollster-rating overrides, house-effect
deconvolution) are a project of their own. Until that exists, any
politics signal we ship risks being *miscalibrated in both directions*
— which a naive reader might trust.

We abstain with reason. When the polling ingest ships, this module is
where the polling-state-space + fundamentals mixture posterior lands.
"""

from __future__ import annotations

from kalshi_edge.forecast import ForecastResult
from kalshi_edge.forecasters.base import Forecaster
from kalshi_edge.types import Category, Contract, MarketSnapshot


class PoliticsAbstainForecaster(Forecaster):
    name = "politics_abstain"
    version = "1"

    @property
    def category(self) -> Category:
        return Category.POLITICS

    def _forecast_impl(
        self, contract: Contract, snapshot: MarketSnapshot | None,
    ) -> ForecastResult:
        return self._null(
            contract,
            "politics forecaster deferred: needs polling ingest + "
            "fundamentals prior + pollster ratings before shipping "
            "calibrated probabilities.",
        )
