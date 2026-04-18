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
        return parse_weather_contract(contract) is not None

    def _forecast_impl(
        self, contract: Contract, snapshot: MarketSnapshot | None,
    ) -> ForecastResult:
        wc = parse_weather_contract(contract)
        if wc is None:
            return self._null(
                contract,
                "weather_contract_unparseable (unknown location, date, or strike_type)",
            )

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

        # Raw empirical P (no variance inflation, no bootstrap) — useful
        # as a sanity-check reference against the posterior mean.
        crit = wc.criterion
        if crit.direction == "above":
            raw_p = float((members > crit.low).mean())  # type: ignore[arg-type]
        elif crit.direction == "below":
            raw_p = float((members < crit.high).mean())  # type: ignore[arg-type]
        else:
            raw_p = float(
                ((members >= crit.low) & (members <= crit.high)).mean()
            )

        q = np.percentile(members, [5, 25, 50, 75, 95]).round(2).tolist()
        diagnostics = {
            "n_members": int(members.size),
            "ensemble_mean_f": round(float(np.mean(members)), 2),
            "ensemble_median_f": round(float(np.median(members)), 2),
            "ensemble_std_f": round(float(np.std(members, ddof=1)), 2),
            "ensemble_min_f": round(float(members.min()), 2),
            "ensemble_max_f": round(float(members.max()), 2),
            "ensemble_q05_f": q[0],
            "ensemble_q25_f": q[1],
            "ensemble_q75_f": q[3],
            "ensemble_q95_f": q[4],
            "raw_empirical_p_yes": round(raw_p, 4),
            "variance_inflation": self.kappa,
            "yes_direction": wc.criterion.direction,
            "yes_low_f": wc.criterion.low,
            "yes_high_f": wc.criterion.high,
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
                "model": "pooled 82-member ensemble (GFS-ENS 31 + ECMWF-IFS 51) "
                         "at 25 km grid",
                "threshold_test": (
                    "empirical count of members satisfying the YES criterion "
                    "(no Gaussian-CDF approximation)"
                ),
                "prior": (
                    f"variance_inflation kappa={self.kappa}: each member's "
                    f"deviation from the ensemble mean is scaled by kappa to "
                    f"correct NWP underdispersion at 1-7d lead"
                ),
                "posterior_over_p": (
                    "Bayesian bootstrap over inflated members (2000 draws); "
                    "P(YES) per draw is the empirical fraction above/below "
                    "the threshold — preserves any bimodality/skew in the "
                    "raw ensemble"
                ),
                "grid_resolution_caveat": (
                    "25 km cells don't resolve coastal marine-layer / urban "
                    "microclimate — ensemble can bias warm at LAX-type "
                    "shoreline stations; treat disagreements with liquid "
                    "markets as a flag for this known limitation"
                ),
            },
            data_sources={"open_meteo_ensemble": fetch.fetched_at},
            diagnostics=diagnostics,
        )

    # ------- posterior construction -------------------------------------
    def _posterior_from_members(
        self, members: np.ndarray, wc: WeatherContract,
    ) -> dict:
        """Build binary + underlying posteriors from the pooled member array.

        Empirical bootstrap — not Gaussian-on-moments. Principle 4 of the
        methodology says "we do *not* collapse to (mean, std) before
        computing the threshold probability." Prior versions of this code
        fit a Gaussian to each bootstrap resample and took 1-Φ(z). That
        inflates tail mass whenever the ensemble is bimodal or skewed —
        e.g. LAX on a Santa Ana / marine-layer boundary day where
        members split into "burns off" and "stays cool" clusters. We
        instead count the fraction of (inflation-scaled) members that
        satisfy the YES criterion directly.
        """
        mu = float(np.mean(members))
        sigma_raw = float(np.std(members, ddof=1))
        # Underlying posterior: Gaussian(mu, inflated sigma) — kept as a
        # *description* of the ensemble for the dashboard density overlay.
        # Not used to compute P(YES).
        underlying_sigma = max(sigma_raw * self.kappa, MIN_POSTERIOR_STD_F)
        underlying = ForecastDistribution(
            kind="parametric", family="normal",
            params={"loc": mu, "scale": underlying_sigma},
        )

        # Variance inflation on the members themselves. Scale each
        # member's deviation from the pooled mean by kappa — this widens
        # the ensemble without altering its shape (bimodality / skew
        # preserved). A sub-kappa-floor sigma is rescued to the floor by
        # pulling members toward the mean proportionally.
        if sigma_raw > 0:
            scale = max(self.kappa, MIN_POSTERIOR_STD_F / sigma_raw)
        else:
            scale = 1.0
        inflated = mu + (members - mu) * scale

        # Bayesian bootstrap: resample members (with replacement) to
        # propagate *ensemble-sampling uncertainty* into a posterior
        # over P(YES). Per draw we count the empirical fraction of
        # members satisfying the YES criterion. No Gaussian assumption.
        n = inflated.size
        idx = self.rng.integers(0, n, size=(_BOOTSTRAP_DRAWS, n))
        boot = inflated[idx]                                          # (B, n)

        crit = wc.criterion
        if crit.direction == "above":
            assert crit.low is not None
            hit = boot > crit.low
        elif crit.direction == "below":
            assert crit.high is not None
            hit = boot < crit.high
        else:
            assert crit.direction == "between"
            assert crit.low is not None and crit.high is not None
            hit = (boot >= crit.low) & (boot <= crit.high)
        p_yes_samples = hit.mean(axis=1)

        # Numerical guard. Even with an 82-member ensemble, a degenerate
        # bootstrap draw can be exactly 0 or 1; clip away from the tails
        # so the entropy decomposition in base.finalize doesn't NaN.
        p_yes_samples = np.clip(p_yes_samples, 1e-6, 1.0 - 1e-6)

        binary = ForecastDistribution(kind="samples", samples=p_yes_samples)

        # Model confidence: high when P is concentrated (ensemble members
        # agree on whether the threshold is cleared) and raw sigma is in
        # a sensible range.
        p_spread = float(np.std(p_yes_samples))
        confidence = float(np.exp(-4.0 * p_spread))
        if sigma_raw < 0.3 or sigma_raw > 15.0:
            confidence *= 0.5

        return {"binary": binary, "underlying": underlying, "confidence": confidence}
