"""Fed rates forecaster — Polymarket cross-market prior.

Methodology (principles 1, 3, 7, 8):

1. **Model the DGP.** Fed funds futures (CME ZQ) would give us a model-free
   implied probability for any FOMC decision, but there's no free feed.
   Polymarket runs deeply-traded parallel markets on each FOMC meeting
   (25bp cut / no change / 25bp hike / 50bp cut / …). Polymarket's YES
   price on a matching contract is a liquid cross-venue estimator of the
   true probability.

3. **Bayesian prior.** We convert the Polymarket YES price into a Beta
   posterior whose effective sample size scales with Polymarket liquidity
   (see ``polymarket_prior.py``). The Beta is intentionally wider than
   Polymarket's own market-maker quote so that even a deep Polymarket
   pool cannot swamp the Kalshi price by itself — this leaves room for
   a genuine Kalshi-vs-Polymarket mispricing to emerge as edge.

7/8. **Calibration + decomposition** flow through base.finalize.

The forecaster abstains (with an explicit reason) when:
- FOMC meeting date can't be parsed from the event_ticker;
- no Polymarket market matches the Kalshi contract unambiguously;
- the Polymarket YES price is degenerate (0 or 1 — signals a closed /
  just-resolved market we shouldn't port over).

Polymarket-resolution fidelity is assumed high (==1) for FOMC markets
because both venues resolve on the same press-release decision. When a
future series has looser resolution (e.g. "rate path by end-of-year"),
set ``proxy_fidelity`` < 1 to widen the posterior.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone

from kalshi_edge.data_sources.polymarket import PolymarketClient
from kalshi_edge.forecast import ForecastResult
from kalshi_edge.forecasters.base import Forecaster
from kalshi_edge.forecasters.polymarket_match import (
    default_window,
    find_best_match,
)
from kalshi_edge.forecasters.polymarket_prior import beta_posterior_from_polymarket
from kalshi_edge.types import Category, Contract, MarketSnapshot

_EVENT_MONTH_RE = re.compile(r"-(\d{2})([A-Z]{3})(?:-|$)")
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])}
_MONTH_WORD = {
    1: "january", 2: "february", 3: "march", 4: "april", 5: "may", 6: "june",
    7: "july", 8: "august", 9: "september", 10: "october", 11: "november",
    12: "december",
}


@dataclass(frozen=True, slots=True)
class FedContract:
    meeting_month: date  # first-of-month UTC


def parse_fed_contract(series_ticker: str, event_ticker: str) -> FedContract | None:
    # Polymarket's FOMC markets resolve on the press-release decision, which
    # is the same event Kalshi's KXFEDDECISION series resolves on. The
    # sibling KXFED series (end-of-month rate-level markets) asks a
    # different question that Polymarket doesn't offer a counterpart for,
    # so we deliberately don't support it here.
    if series_ticker.upper() != "KXFEDDECISION":
        return None
    m = _EVENT_MONTH_RE.search(event_ticker or "")
    if not m:
        return None
    yy = int(m.group(1))
    mon = _MONTHS.get(m.group(2))
    if mon is None:
        return None
    try:
        return FedContract(meeting_month=date(2000 + yy, mon, 1))
    except ValueError:
        return None


class RatesFOMCForecaster(Forecaster):
    name = "rates_polymarket_fomc"
    version = "1"

    _SEARCH_KEYWORDS = ["fed", "fomc", "basis points", "rate"]

    def __init__(self, client: PolymarketClient | None = None) -> None:
        self._client = client  # lazy-init so tests can inject a mock

    @property
    def category(self) -> Category:
        return Category.RATES

    def supports(self, contract: Contract) -> bool:
        if contract.category != Category.RATES:
            return False
        return parse_fed_contract(contract.series_ticker, contract.event_ticker) is not None

    def _ensure_client(self) -> PolymarketClient:
        if self._client is None:
            self._client = PolymarketClient()
        return self._client

    def _forecast_impl(
        self, contract: Contract, snapshot: MarketSnapshot | None,
    ) -> ForecastResult:
        fc = parse_fed_contract(contract.series_ticker, contract.event_ticker)
        if fc is None:
            return self._null(contract, "rates_contract_unparseable")

        anchor = datetime(
            fc.meeting_month.year, fc.meeting_month.month, 15, tzinfo=timezone.utc,
        )
        window = default_window(anchor, days=45)

        # AND-require the meeting month+year in the Polymarket question.
        # Keeps us from matching a FOMC market on the wrong meeting.
        required = {_MONTH_WORD[fc.meeting_month.month], str(fc.meeting_month.year)}

        client = self._ensure_client()
        match = find_best_match(
            client=client,
            search_keywords=self._SEARCH_KEYWORDS,
            title=contract.title,
            target_window=window,
            extra_required_tokens=required,
        )
        if match is None:
            return self._null(
                contract,
                f"no unambiguous Polymarket FOMC market for {contract.ticker} "
                f"(meeting={fc.meeting_month.isoformat()})",
            )

        pm = match.market
        if pm.yes_price is None:
            return self._null(contract, "polymarket market has no YES price")
        # Refuse degenerate prices — signals a market that already resolved.
        if not (0.005 < pm.yes_price < 0.995):
            return self._null(
                contract,
                f"polymarket YES price degenerate ({pm.yes_price:.4f}) — likely resolved",
            )

        dist, confidence = beta_posterior_from_polymarket(
            yes_price=pm.yes_price, volume=pm.volume, liquidity=pm.liquidity,
            proxy_fidelity=1.0,
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
                "match_scoring": "token Jaccard on question vs Kalshi title",
                "caveat": (
                    "Polymarket and Kalshi settle on the same FOMC press release, "
                    "so resolution fidelity is high. No free CME ZQ futures feed, "
                    "which would be the gold-standard alternative."
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
                "meeting_month": fc.meeting_month.isoformat(),
                "beta_a": dist.params["a"],
                "beta_b": dist.params["b"],
            },
        )
