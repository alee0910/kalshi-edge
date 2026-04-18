"""Politics forecaster — Polymarket cross-market prior.

Methodology (principles 1, 3, 7, 8):

1. **Model the DGP.** Elections & political-event markets don't admit a
   first-principles model at the scope of this project. The calibrated
   alternatives are (a) a polling-average backbone (RealClearPolitics or
   538-style) combined with a fundamentals prior, and (b) other venues'
   prediction-market prices. Building a polling pipeline is a project of
   its own (daily poll ingest, pollster ratings, house-effect
   deconvolution, state-correlation model); in the meantime
   Polymarket's YES price on a matching contract is already a deeply-
   traded, continuously-updating probability estimator.

3. **Bayesian prior.** We convert the Polymarket YES price into a Beta
   posterior whose effective sample size scales with Polymarket
   liquidity (see ``polymarket_prior.py``). We apply a ``proxy_fidelity``
   haircut below 1.0 because Polymarket and Kalshi political-contract
   resolution criteria sometimes differ in subtle ways (e.g. exact
   "wins" definitions on Congressional majority contracts). The haircut
   widens the posterior so Kalshi prices get more room to express
   genuine venue-specific information.

7/8. **Calibration + decomposition** flow through base.finalize.

The forecaster abstains (with an explicit reason) when:
- no Polymarket market matches the Kalshi contract unambiguously
  (token-Jaccard below threshold or runner-up too close);
- the Polymarket YES price is degenerate (0 or 1 — resolved).

Roadmap (v2): replace the raw Polymarket prior with a mixture of
(Polymarket prior, RCP/538 polling-state-space posterior, fundamentals
prior) under Bayesian model averaging. Polymarket-only is a
scientifically defensible v1 prior because cross-venue market prices
are known to be well-calibrated on high-attention political events.
"""

from __future__ import annotations

from datetime import datetime, timezone

from kalshi_edge.data_sources.polymarket import PolymarketClient
from kalshi_edge.forecast import ForecastResult
from kalshi_edge.forecasters.base import Forecaster
from kalshi_edge.forecasters.polymarket_match import (
    default_window,
    find_best_match,
)
from kalshi_edge.forecasters.polymarket_prior import beta_posterior_from_polymarket
from kalshi_edge.types import Category, Contract, MarketSnapshot


# Coarse keyword set: Polymarket returns up to `limit` active markets
# filtered client-side. These keywords are an OR-filter, deliberately
# broad. The title-match step is what actually picks the counterpart.
_SEARCH_KEYWORDS = [
    "election", "primary", "presidential", "senate", "house",
    "governor", "congress", "republican", "democrat", "party",
    "nominee", "speaker", "president",
]


class PoliticsCrossMarketForecaster(Forecaster):
    name = "politics_polymarket_cross"
    version = "1"

    def __init__(self, client: PolymarketClient | None = None) -> None:
        self._client = client

    @property
    def category(self) -> Category:
        return Category.POLITICS

    def _ensure_client(self) -> PolymarketClient:
        if self._client is None:
            self._client = PolymarketClient()
        return self._client

    def _forecast_impl(
        self, contract: Contract, snapshot: MarketSnapshot | None,
    ) -> ForecastResult:
        # Anchor Polymarket's endDate window around Kalshi's close_time
        # when available — otherwise accept any active Polymarket market.
        window = None
        if contract.close_time is not None:
            window = default_window(contract.close_time, days=60)

        client = self._ensure_client()
        match = find_best_match(
            client=client,
            search_keywords=_SEARCH_KEYWORDS,
            title=contract.title,
            target_window=window,
        )
        if match is None:
            return self._null(
                contract,
                f"no unambiguous Polymarket political market for {contract.ticker}",
            )

        pm = match.market
        if pm.yes_price is None:
            return self._null(contract, "polymarket market has no YES price")
        if not (0.005 < pm.yes_price < 0.995):
            return self._null(
                contract,
                f"polymarket YES price degenerate ({pm.yes_price:.4f}) — likely resolved",
            )

        # Slightly wider posterior than rates: political resolution
        # criteria are more likely to diverge between venues.
        dist, confidence = beta_posterior_from_polymarket(
            yes_price=pm.yes_price, volume=pm.volume, liquidity=pm.liquidity,
            proxy_fidelity=0.7,
        )

        return ForecastResult(
            ticker=contract.ticker,
            ts=datetime.now(timezone.utc),
            forecaster=self.name,
            version=self.version,
            binary_posterior=dist,
            model_confidence=confidence,
            methodology={
                "model": "Polymarket cross-market Bayesian prior (Beta posterior)",
                "prior": "Beta(α, β) with α+β scaled by log(liquidity+3*vol)",
                "proxy_fidelity": 0.7,
                "match_scoring": "token Jaccard on question vs Kalshi title",
                "v2_roadmap": (
                    "mixture of (Polymarket, RCP polling posterior, fundamentals) "
                    "under Bayesian model averaging"
                ),
            },
            data_sources={"polymarket_gamma_api": pm.fetched_at},
            diagnostics={
                "polymarket_slug": pm.slug,
                "polymarket_question": pm.question,
                "polymarket_yes_price": pm.yes_price,
                "polymarket_volume": pm.volume,
                "polymarket_liquidity": pm.liquidity,
                "match_score": match.score,
                "runner_up_score": match.runner_up_score,
                "beta_a": dist.params["a"],
                "beta_b": dist.params["b"],
            },
        )
