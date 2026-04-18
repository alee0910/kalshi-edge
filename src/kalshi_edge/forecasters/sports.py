"""Sports forecaster — bookmaker-ensemble Bayesian prior.

Methodology (principles 1, 3, 4, 7, 8):

1. **Model the DGP.** The binary event "team X wins" has no first-principles
   model available at our scope. Professional bookmakers' prices, however,
   are the *market's* posterior after pricing in injuries, lineups, venue
   effects, travel, rest, and private information. We treat each
   bookmaker's devigged implied probability as a noisy estimator of the
   true win probability.

3. **Bayesian prior via devigging.** Raw bookmaker prices embed a
   vigorish. We apply multiplicative devigging (``p_i = (1/o_i) /
   Σ(1/o_j)``) — the MLE under symmetric vig — to recover each book's
   fair estimate. This is a soft prior: we never use a single book's
   estimate, we use the ensemble.

4. **Ensemble → distribution.** We bootstrap across bookmakers (with
   replacement) to build a posterior over p. This captures book-level
   disagreement as epistemic uncertainty. The result is a ``samples``
   distribution that base.finalize() hands to decompose_binary_uncertainty.

7/8. **Calibration + decomposition** flow through the standard base
   machinery.

The forecaster abstains (with an explicit reason) when:
- series is not in SERIES_TO_SPORT (e.g. KXATPMATCH — tennis not wired);
- event_ticker date doesn't parse;
- no odds-api game matches the contract (date + team);
- too few bookmakers (< 3) quoted the market, making the ensemble unstable.
"""

from __future__ import annotations

import os
from datetime import date, datetime, time, timedelta, timezone

import numpy as np

from kalshi_edge.data_sources.the_odds_api import (
    Game,
    SERIES_TO_SPORT,
    TheOddsAPIClient,
    devig_book,
)
from kalshi_edge.forecast import ForecastDistribution, ForecastResult
from kalshi_edge.forecasters.base import Forecaster
from kalshi_edge.forecasters.sports_rules import (
    match_game,
    parse_sports_contract,
)
from kalshi_edge.types import Category, Contract, MarketSnapshot

_BOOTSTRAP_DRAWS = 2000
_MIN_BOOKS = 3


class SportsOddsForecaster(Forecaster):
    name = "sports_odds_ensemble"
    version = "1"

    def __init__(
        self,
        client: TheOddsAPIClient | None = None,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.client = client  # may be None; we resolve lazily on first forecast
        # Cache is keyed by (sport_key, target_date.isoformat()) so we reuse
        # the events+odds fetch across all contracts for the same sport/day
        # within one forecast run, but refetch when we hit the next day.
        self._game_cache: dict[tuple[str, str], list[Game]] = {}
        self.rng = rng or np.random.default_rng(seed=0xD00D1E)

    @property
    def category(self) -> Category:
        return Category.SPORTS

    def supports(self, contract: Contract) -> bool:
        if contract.category != Category.SPORTS:
            return False
        return contract.series_ticker.upper() in SERIES_TO_SPORT

    def _forecast_impl(
        self, contract: Contract, snapshot: MarketSnapshot | None,
    ) -> ForecastResult:
        sc = parse_sports_contract(contract.series_ticker, contract.event_ticker)
        if sc is None:
            return self._null(contract, f"sports series not wired: {contract.series_ticker}")

        if self.client is None:
            key = os.environ.get("THE_ODDS_API_KEY")
            if not key:
                return self._null(
                    contract,
                    "THE_ODDS_API_KEY not set — sports forecaster needs odds-api access",
                )
            self.client = TheOddsAPIClient(key)

        games = self._games_for(sc.sport_key, sc.target_date)
        if not games:
            return self._null(contract, f"no odds-api games for {sc.sport_key}")

        hit = match_game(sc, contract.title, games)
        if hit is None:
            return self._null(
                contract,
                f"no odds-api game matched for {contract.ticker} on {sc.target_date}",
            )
        game, target_team = hit

        per_book_p = self._devig_per_book(game, target_team)
        if len(per_book_p) < _MIN_BOOKS:
            return self._null(
                contract,
                f"only {len(per_book_p)} book(s) quoted {target_team} — need >= {_MIN_BOOKS}",
            )

        samples = self._bootstrap_posterior(per_book_p)
        mean_p = float(np.mean(samples))
        std_p = float(np.std(samples))
        confidence = float(np.exp(-6.0 * std_p))   # 1 at collapse, ~0.22 at std=0.25

        return ForecastResult(
            ticker=contract.ticker,
            ts=datetime.now(timezone.utc),
            forecaster=self.name,
            version=self.version,
            binary_posterior=ForecastDistribution(kind="samples", samples=samples),
            model_confidence=confidence,
            methodology={
                "model": "bookmaker-ensemble devigged consensus",
                "devig": "multiplicative (MLE under symmetric vig)",
                "aggregation": "Bayesian bootstrap across bookmakers",
                "caveat": "ensemble of US books — no sharp-book weighting in v1",
            },
            data_sources={"the_odds_api": game.fetched_at},
            diagnostics={
                "sport_key": sc.sport_key,
                "target_team": target_team,
                "home_team": game.home_team,
                "away_team": game.away_team,
                "commence_time": game.commence_time.isoformat(),
                "n_books": len(per_book_p),
                "per_book_p": per_book_p,
                "mean_p": mean_p,
                "std_p": std_p,
            },
        )

    # ---- internals ------------------------------------------------------
    def _games_for(self, sport_key: str, target_date: date) -> list[Game]:
        key = (sport_key, target_date.isoformat())
        cached = self._game_cache.get(key)
        if cached is not None:
            return cached
        assert self.client is not None
        # Cap per-event quota spend: Kalshi's ticker date is the game's
        # ET-local calendar day, which in UTC can land anywhere from the
        # same morning (early East-coast games) to +1 day (late-night West
        # Coast games). Fetching through target_date + 2 days UTC covers
        # every same-day game with a safety margin, and keeps us from
        # paying for odds on games a week out that Kalshi isn't trading yet.
        commence_before = datetime.combine(
            target_date + timedelta(days=2), time(0, 0), tzinfo=timezone.utc,
        )
        games = self.client.h2h_odds(sport_key, commence_before=commence_before)
        self._game_cache[key] = games
        return games

    def _devig_per_book(self, game: Game, target_team: str) -> dict[str, float]:
        by_book: dict[str, dict[str, float]] = {}
        for o in game.outcomes:
            by_book.setdefault(o.bookmaker, {})[o.team] = o.decimal_price
        out: dict[str, float] = {}
        for book, prices in by_book.items():
            if target_team not in prices:
                continue
            devig = devig_book(prices)
            if target_team in devig:
                out[book] = float(devig[target_team])
        return out

    def _bootstrap_posterior(self, per_book_p: dict[str, float]) -> np.ndarray:
        ps = np.asarray(list(per_book_p.values()), dtype=float)
        n = ps.size
        idx = self.rng.integers(0, n, size=(_BOOTSTRAP_DRAWS, n))
        boot = ps[idx].mean(axis=1)
        return np.clip(boot, 1e-6, 1.0 - 1e-6)
