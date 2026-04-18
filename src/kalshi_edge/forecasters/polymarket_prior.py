"""Shared helper: Polymarket YES price → Bayesian Beta posterior.

Used by rates_ and politics_ forecasters. The cross-market idea:

Given a Polymarket market whose resolution criterion matches (or nearly
matches) a Kalshi contract, Polymarket's YES price is an outside estimator
of the true probability. We do **not** treat it as ground truth — we
treat it as a noisy estimator and carry forward both its point and its
uncertainty.

Formally we use a Beta(α, β) posterior on p with

    α / (α + β) = yes_price            (point estimate)
    α + β       = ESS(liquidity, volume)  (effective sample size)

ESS is a monotonic function of Polymarket's volume and liquidity. We pin
ESS between ``ESS_FLOOR`` (thin markets — wide Beta) and ``ESS_CAP``
(deep markets — tight Beta). This is intentionally generous: even the
most liquid Polymarket FOMC market should not swamp the Kalshi market
price by itself. A 1k-dollar Polymarket market should not confidently
call a 50-50 contract.

We also reserve a confidence-haircut knob ``proxy_fidelity`` ∈ (0, 1]
for when the Polymarket resolution criterion is *similar but not
identical* to the Kalshi one — applied multiplicatively to ESS.

The returned ForecastDistribution is parametric Beta(α, β); base.finalize
will synthesize samples and run the uncertainty decomposition.
"""

from __future__ import annotations

import math

from kalshi_edge.forecast import ForecastDistribution


# Chosen so that a thin market (volume ~ $1k) gives a Beta with std ~ 0.14
# around its mean, and a deep market (volume ~ $1M+) collapses to std ~ 0.03.
ESS_FLOOR = 12.0
ESS_CAP = 240.0


def ess_from_liquidity(volume: float, liquidity: float) -> float:
    """Effective sample size for Beta posterior, as a function of market depth.

    Uses ``log1p(volume + 3 * liquidity)`` as the depth signal — liquidity
    (order-book depth) is weighted more heavily than volume because it
    reflects current dollars willing to trade, not historical flow.
    """
    signal = math.log1p(max(0.0, volume) + 3.0 * max(0.0, liquidity))
    # Rescale to [ESS_FLOOR, ESS_CAP]. log1p of $1k ≈ 7, $1M ≈ 14.
    # We interpolate over [5, 15] and clip.
    frac = (signal - 5.0) / (15.0 - 5.0)
    frac = max(0.0, min(1.0, frac))
    return ESS_FLOOR + frac * (ESS_CAP - ESS_FLOOR)


def beta_posterior_from_polymarket(
    yes_price: float, *, volume: float, liquidity: float,
    proxy_fidelity: float = 1.0,
) -> tuple[ForecastDistribution, float]:
    """Build the (Beta posterior, model-confidence) pair.

    ``yes_price`` must be in (0, 1). Callers should clip before calling.
    ``proxy_fidelity`` is a multiplicative ESS haircut in (0, 1] — lower
    values widen the posterior when the Polymarket market doesn't
    resolve under exactly the same criterion as the Kalshi contract.

    Returns a Beta ForecastDistribution. Confidence drops with std(p):
    same functional form as sports/weather, so posteriors are directly
    comparable across forecasters in the ranker.
    """
    if not (0.0 < yes_price < 1.0):
        raise ValueError(f"yes_price must be in (0, 1), got {yes_price}")
    if not (0.0 < proxy_fidelity <= 1.0):
        raise ValueError(f"proxy_fidelity must be in (0, 1], got {proxy_fidelity}")

    ess = ess_from_liquidity(volume, liquidity) * proxy_fidelity
    a = max(1e-3, ess * yes_price)
    b = max(1e-3, ess * (1.0 - yes_price))

    dist = ForecastDistribution(
        kind="parametric", family="beta", params={"a": a, "b": b},
    )
    # std of Beta(a, b) = sqrt(ab / ((a+b)^2 (a+b+1)))
    var = (a * b) / ((a + b) ** 2 * (a + b + 1.0))
    std = math.sqrt(var)
    confidence = math.exp(-6.0 * std)  # matches SportsOddsForecaster scale
    return dist, float(confidence)
