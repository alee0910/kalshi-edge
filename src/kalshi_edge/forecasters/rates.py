"""Fed rates forecaster — intentional abstain.

Fed-decision markets are most accurately forecast by **fed funds futures**
(CME 30-Day Fed Fund — ticker ZQ). Absent a free futures source, any
model we ship here would be less calibrated than the Kalshi market price
itself, which violates the project principle *market price as Bayesian
prior — don't publish a signal that can't beat it*.

We abstain with a clear reason. When a ZQ-implied source is added this
module is the one-file place to slot it in.
"""

from __future__ import annotations

from kalshi_edge.forecast import ForecastResult
from kalshi_edge.forecasters.base import Forecaster
from kalshi_edge.types import Category, Contract, MarketSnapshot


class RatesAbstainForecaster(Forecaster):
    name = "rates_abstain"
    version = "1"

    @property
    def category(self) -> Category:
        return Category.RATES

    def _forecast_impl(
        self, contract: Contract, snapshot: MarketSnapshot | None,
    ) -> ForecastResult:
        return self._null(
            contract,
            "rates forecaster deferred: needs CME ZQ fed-funds-futures feed "
            "to produce a calibrated signal. Market price is a better prior "
            "than anything we'd ship without it.",
        )
