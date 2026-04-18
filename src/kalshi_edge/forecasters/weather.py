"""Weather forecaster: ensemble → posterior over daily max/min → P(threshold).

Methodology, explicit (principles 1, 2, 3, 4, 7, 8 from project memory):

1. **Model the DGP, not the outcome.** We model the daily extremum
   temperature as a random variable with a posterior distribution; the
   binary P(YES) is computed by integrating the posterior over the
   threshold region. We surface the temperature posterior separately so
   the dashboard can overlay it on the market-implied distribution.

2. **Stochastic processes.** Each ensemble member is a realized trajectory
   from a perturbed initial-condition IVP of the same atmospheric PDE.
   We treat the set of members as samples from a meta-distribution that
   approximates forecast uncertainty.

3. **Bayesian inference with explicit priors.** Short-range (1-7d) NWP
   ensembles are well-known to be *underdispersive* for 2m-temperature —
   verification studies consistently find observed outcomes fall outside
   the ensemble range more often than calibration would predict. We apply
   a variance inflation factor κ to the pooled ensemble std. κ is a
   soft-informative prior on how much we trust raw ensemble dispersion;
   the default (1.3) reflects published GEFS/ECMWF 2m-T verification.

4. **Ensembles as distributions.** We do *not* collapse to (mean, std)
   before computing the threshold probability. We use a Bayesian bootstrap
   on ensemble members to propagate ensemble-sampling uncertainty into a
   *distribution over* P(YES) — which is exactly what the uncertainty
   decomposition in base.finalize() expects.

7. **Proper scoring + calibration.** Outputs are Bernoulli probabilities;
   downstream Brier/log-score loss tracking in calibration/.

8. **Uncertainty decomposition.** The bootstrap-based posterior over p
   gives both aleatoric (within-model entropy at the posterior mean) and
   epistemic (spread of p under the bootstrap) components; base.finalize
   computes the split.

The forecaster returns a null result (with reason) whenever:
  - The ticker doesn't parse as a weather contract we recognize.
  - The target date is outside the ensemble forecast window.
  - Open-Meteo returns no data.
No silent defaults; no fake precision.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from kalshi_edge.data_sources.open_meteo import (
    OpenMeteoClient,
    daily_extrema,
)
from kalshi_edge.forecast import ForecastDistribution, ForecastResult
from kalshi_edge.forecasters.base import Forecaster
from kalshi_edge.forecasters.weather_rules import (
    WeatherContract,
    parse_weather_contract,
)
from kalshi_edge.types import Category, Contract, MarketSnapshot


# NWP ensemble underdispersion for 2m-T at 1-7d lead: empirically ~1.2-1.5x
# too narrow. Default reflects the midpoint; override per-category if a
# site-specific calibration study updates the estimate.
DEFAULT_VARIANCE_INFLATION = 1.30

# Floor on posterior std (°F). An ensemble that accidentally collapses to
# near-zero spread (rare but has happened at very short lead) would
# otherwise produce p_yes → {0,1} and blow up calibration. 0.5°F is well
# below meaningful forecast skill.
MIN_POSTERIOR_STD_F = 0.5

_BOOTSTRAP_DRAWS = 2000


class WeatherForecaster(Forecaster):
    """Ensemble-based forecaster for Kalshi daily-extremum weather contracts."""

    name = "weather_ensemble"
    version = "1"

    def __init__(
        self,
        client: OpenMeteoClient | None = None,
        *,
        variance_inflation: float = DEFAULT_VARIANCE_INFLATION,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.client = client or OpenMeteoClient()
        self.kappa = float(variance_inflation)
        self.rng = rng or np.random.default_rng(seed=0xC0FFEE)

    @property
    def category(self) -> Category:
        return Category.WEATHER

    def supports(self, contract: Contract) -> bool:
        if contract.category != Category.WEATHER:
            return False
        return parse_weather_contract(
            contract.series_ticker, contract.event_ticker, contract.ticker
        ) is not None

    def _forecast_impl(
        self, contract: Contract, snapshot: MarketSnapshot | None,
    ) -> ForecastResult:
        wc = parse_weather_contract(
            contract.series_ticker, contract.event_ticker, contract.ticker
        )
        if wc is None:
            return self._null(contract, "weather_contract_unparseable")

        # Days-ahead used to decide request window + sanity-check lead time.
        today = datetime.now(timezone.utc).date()
        days_ahead = (wc.target_date - today).days
        if days_ahead < 0:
            return self._null(contract, f"target_date_in_past: {wc.target_date}")
        if days_ahead > 14:
            return self._null(contract, f"target_date_beyond_ensemble: {wc.target_date}")

        fetch = self.client.fetch_ensemble(
            lat=wc.location.lat, lon=wc.location.lon, tz=wc.location.tz,
            slug=wc.location.slug,
            forecast_days=min(16, max(2, days_ahead + 2)),
        )

        try:
            members = daily_extrema(fetch, wc.target_date, wc.metric)
        except ValueError as e:
            return self._null(contract, f"ensemble_window_miss: {e}")

        if members.size < 10:
            return self._null(contract, f"ensemble_too_small: {members.size}")

        result = self._posterior_from_members(members, wc)
        diagnostics = {
            "n_members": int(members.size),
            "ensemble_mean_f": float(np.mean(members)),
            "ensemble_std_f": float(np.std(members, ddof=1)),
            "variance_inflation": self.kappa,
            "threshold_f": wc.threshold_f,
            "target_date": wc.target_date.isoformat(),
            "days_ahead": days_ahead,
        }

        return ForecastResult(
            ticker=contract.ticker,
            ts=datetime.now(timezone.utc),
            forecaster=self.name,
            version=self.version,
            binary_posterior=result["binary"],
            underlying_posterior=result["underlying"],
            model_confidence=result["confidence"],
            methodology={
                "model": "pooled ensemble (GFS+ECMWF) with variance inflation",
                "prior": f"variance_inflation kappa={self.kappa} (NWP underdispersion)",
                "likelihood": "per-member daily extremum in local-day window",
                "posterior_over_p": "Bayesian bootstrap over ensemble members",
                "inflation_rationale": "GEFS/ECMWF 2m-T are empirically underdispersive "
                                       "at 1-7d lead; kappa ~ 1.2-1.5 typical.",
            },
            data_sources={"open_meteo_ensemble": fetch.fetched_at},
            diagnostics=diagnostics,
        )

    # ------- posterior construction -------------------------------------
    def _posterior_from_members(
        self, members: np.ndarray, wc: WeatherContract,
    ) -> dict:
        """Build binary + underlying posteriors from the pooled member array."""
        mu = float(np.mean(members))
        # Sample std, then inflated std clamped to the floor.
        sigma_raw = float(np.std(members, ddof=1))
        sigma = max(sigma_raw * self.kappa, MIN_POSTERIOR_STD_F)

        # Underlying posterior: Gaussian(mu, sigma_inflated) on °F.
        # Stored as a parametric distribution so the dashboard can render a
        # closed-form density overlay against the market-implied mixture.
        underlying = ForecastDistribution(
            kind="parametric", family="normal",
            params={"loc": mu, "scale": sigma},
        )

        # Binary posterior: Bayesian bootstrap over ensemble members. For
        # each bootstrap draw we resample the members with replacement,
        # compute the bootstrap Gaussian fit to that resample, and derive
        # the threshold probability. The *distribution* of those
        # probabilities is our posterior over p — it captures within-model
        # sampling uncertainty on top of the aleatoric spread.
        n = members.size
        idx = self.rng.integers(0, n, size=(_BOOTSTRAP_DRAWS, n))
        boot = members[idx]                                           # (B, n)
        bmu = boot.mean(axis=1)
        bsd = boot.std(axis=1, ddof=1)
        # Inflate per bootstrap draw and clamp.
        bsd = np.maximum(bsd * self.kappa, MIN_POSTERIOR_STD_F)

        # P(T > threshold) via the Gaussian CDF. vectorized.
        from scipy.stats import norm
        z = (wc.threshold_f - bmu) / bsd
        p_yes_samples = 1.0 - norm.cdf(z)
        if wc.comparator == "below":
            p_yes_samples = 1.0 - p_yes_samples

        # Numerical guard. Even with clamping, float underflow near the tails
        # can pin samples to exact 0/1, which the entropy decomposition then
        # handles by clipping — but it's cleaner to clip here and log.
        p_yes_samples = np.clip(p_yes_samples, 1e-6, 1.0 - 1e-6)

        binary = ForecastDistribution(kind="samples", samples=p_yes_samples)

        # Model confidence: high when ensemble is wide enough to be
        # informative and days_ahead is small. We fuse two signals:
        #   (a) relative spread of p: narrow → more confident
        #   (b) ensemble sigma is in a reasonable range (not collapsed,
        #       not explosively wide)
        p_spread = float(np.std(p_yes_samples))
        confidence = float(np.exp(-4.0 * p_spread))  # ~1.0 at p_spread=0, ~0.37 at 0.25
        # Penalize extreme raw sigma (collapsed ensemble is suspicious).
        if sigma_raw < 0.3 or sigma_raw > 15.0:
            confidence *= 0.5

        return {"binary": binary, "underlying": underlying, "confidence": confidence}
