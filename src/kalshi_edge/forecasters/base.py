"""Abstract Forecaster + registry + Bayesian-model-averaging helper.

Concrete forecasters (weather, rates, polling, …) subclass ``Forecaster`` and
are registered under the Category they handle. The scheduler asks the registry
for a forecaster given a Contract; mismatches raise instead of silently
defaulting, because a silent default is how you end up reporting fake
precision on contracts you have no right to forecast.

Design choices worth calling out:

- ``forecast()`` returns a ``ForecastResult`` even on failure: we encode
  inability to forecast as ``null_reason != None`` rather than raising or
  returning ``None``. This preserves auditability (every attempted forecast
  is logged with its reason for abstaining).

- ``finalize()`` is a small post-processor every subclass should call at the
  end of ``forecast()`` before returning. It computes the uncertainty
  decomposition from the binary posterior samples (or synthesizes samples
  from a parametric binary posterior) and applies a staleness-aware
  adjustment to ``model_confidence``.

- ``bayesian_model_average`` implements BMA across a set of sub-model
  posteriors. We use weights ∝ exp(log-marginal-likelihood) normalized; when
  marginal likelihoods are unavailable, callers can supply stacking weights.
  This is the clean version of "ensemble averaging" the user rightly
  complained about.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Iterable

import numpy as np

from kalshi_edge.forecast import (
    ForecastDistribution,
    ForecastResult,
    decompose_binary_uncertainty,
)
from kalshi_edge.logging_ import get_logger
from kalshi_edge.types import Category, Contract, MarketSnapshot

log = get_logger(__name__)


_MAX_STALENESS_MINUTES = 6 * 60  # 6 hours; above this, confidence is halved.


class Forecaster(ABC):
    """Uniform forecasting interface.

    Subclasses must set ``name`` and ``version`` as class attributes and
    implement ``category``, ``supports``, and ``_forecast_impl``.
    """

    name: str = "abstract"
    version: str = "0"

    @property
    @abstractmethod
    def category(self) -> Category:
        """The Kalshi category this forecaster owns."""

    def supports(self, contract: Contract) -> bool:
        """Default: supports everything in its category. Override to narrow."""
        return contract.category == self.category

    @abstractmethod
    def _forecast_impl(
        self, contract: Contract, snapshot: MarketSnapshot | None
    ) -> ForecastResult:
        """Produce a ForecastResult without any finalization."""

    # Public entrypoint used by the scheduler.
    def forecast(
        self, contract: Contract, snapshot: MarketSnapshot | None = None
    ) -> ForecastResult:
        if not self.supports(contract):
            return self._null(contract, f"{self.name} does not support {contract.ticker}")
        try:
            result = self._forecast_impl(contract, snapshot)
        except Exception as e:  # noqa: BLE001 — we want to surface any failure as a null
            log.exception("forecast_failed", ticker=contract.ticker, forecaster=self.name)
            return self._null(contract, f"forecast_exception: {e.__class__.__name__}: {e}")
        return self.finalize(result)

    # ------- hooks used by subclasses -----------------------------------
    def _null(self, contract: Contract, reason: str) -> ForecastResult:
        """Build a ForecastResult that records an abstention."""
        return ForecastResult(
            ticker=contract.ticker,
            ts=datetime.now(timezone.utc),
            forecaster=self.name,
            version=self.version,
            # Placeholder uniform beta(1,1); downstream treats null_reason != None as no-op.
            binary_posterior=ForecastDistribution(
                kind="parametric", family="beta", params={"a": 1.0, "b": 1.0}
            ),
            null_reason=reason,
            model_confidence=0.0,
        )

    def finalize(self, r: ForecastResult) -> ForecastResult:
        """Compute the uncertainty decomposition + apply staleness haircut."""
        if r.is_null():
            return r

        # Decomposition. If the posterior is samples-backed we use them
        # directly; if parametric, synthesize enough samples for a stable estimate.
        if r.binary_posterior.kind == "samples":
            ps = r.binary_posterior.samples
            assert ps is not None
        else:
            ps = r.binary_posterior.draw(4000)
        r.uncertainty = decompose_binary_uncertainty(np.asarray(ps, dtype=float))

        # Staleness haircut: oldest data source older than _MAX_STALENESS_MINUTES
        # halves confidence. Linear between "fresh" and "max stale".
        if r.data_sources:
            now = datetime.now(timezone.utc)
            oldest = min(v for v in r.data_sources.values())
            if oldest.tzinfo is None:
                oldest = oldest.replace(tzinfo=timezone.utc)
            age_minutes = max((now - oldest).total_seconds() / 60.0, 0.0)
            if age_minutes > 0:
                haircut = max(0.5, 1.0 - 0.5 * (age_minutes / _MAX_STALENESS_MINUTES))
                r.model_confidence = max(0.0, min(1.0, r.model_confidence * haircut))
                r.diagnostics = {**r.diagnostics, "staleness_minutes": age_minutes,
                                 "staleness_haircut": haircut}
        return r


class ForecasterRegistry:
    """Dispatch from Contract to its Forecaster."""

    def __init__(self) -> None:
        self._by_category: dict[Category, list[Forecaster]] = {}

    def register(self, forecaster: Forecaster) -> None:
        self._by_category.setdefault(forecaster.category, []).append(forecaster)

    def get(self, contract: Contract) -> Forecaster | None:
        for f in self._by_category.get(contract.category, []):
            if f.supports(contract):
                return f
        return None

    def all(self) -> list[Forecaster]:
        return [f for fs in self._by_category.values() for f in fs]


_DEFAULT = ForecasterRegistry()


def default_registry() -> ForecasterRegistry:
    """Module-level singleton; weather/economics/rates modules register into it at import."""
    return _DEFAULT


# ---- Bayesian model averaging ------------------------------------------
def bayesian_model_average(
    posteriors: Iterable[ForecastDistribution],
    weights: Iterable[float],
    n_samples: int = 4000,
    rng: np.random.Generator | None = None,
) -> ForecastDistribution:
    """Weighted mixture of posterior distributions.

    For BMA in the proper sense, ``weights`` should be the posterior
    probabilities of each model (∝ marginal likelihood × prior). For stacking
    they should be the cross-validated stacking weights. We just consume
    nonnegative weights that sum to 1.

    The mixture is realized as a draw: pick a component by weight, then sample
    from it. This is exact for the mixture posterior without any Gaussian
    assumption and works whether components are samples or parametric.
    """
    posts = list(posteriors)
    w = np.asarray(list(weights), dtype=float)
    if len(posts) != w.size:
        raise ValueError("posteriors and weights must align")
    if np.any(w < 0):
        raise ValueError("weights must be nonnegative")
    if w.sum() <= 0:
        raise ValueError("weights must have positive sum")
    w = w / w.sum()

    r = rng or np.random.default_rng()
    component_idx = r.choice(len(posts), size=n_samples, p=w)
    out = np.empty(n_samples, dtype=float)
    for k, post in enumerate(posts):
        mask = component_idx == k
        if mask.any():
            out[mask] = post.draw(int(mask.sum()), rng=r)
    return ForecastDistribution(kind="samples", samples=out)
