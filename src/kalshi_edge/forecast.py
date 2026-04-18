"""Posterior-distribution types and ForecastResult.

The central discipline: every forecast is a probability distribution with
quantified uncertainty. ``ForecastDistribution`` represents either samples
(from MCMC / Monte Carlo) or a closed-form parametric family. ``ForecastResult``
wraps a binary P(YES) posterior with methodology provenance, uncertainty
decomposition, and diagnostics.

Uncertainty decomposition for binary outcomes (Kendall & Gal 2017 / Depeweg
et al 2018 adapted to Bernoulli):
    H(E[p])         = entropy of the posterior mean  ("total")
    E[H(p)]         = mean Bernoulli entropy under the posterior  ("aleatoric")
    H(E[p]) - E[H(p)] = epistemic uncertainty (reducible with more data)

Model uncertainty is a separate axis captured by BMA / stacking weights
across alternative models; see ``forecasters/base.py::average_forecasts``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import numpy as np
from scipy import stats


DistributionKind = Literal["samples", "parametric"]


@dataclass
class ForecastDistribution:
    """Posterior over a real-valued quantity.

    Either samples-backed (``samples`` set) or parametric (``family`` + ``params``).
    All summary methods (mean/std/quantile/cdf) work in either mode so callers
    never branch on representation.
    """

    kind: DistributionKind
    samples: np.ndarray | None = None
    family: str | None = None
    params: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind == "samples":
            if self.samples is None or self.samples.ndim != 1 or self.samples.size == 0:
                raise ValueError("samples kind requires 1D non-empty samples array")
        elif self.kind == "parametric":
            if not self.family:
                raise ValueError("parametric kind requires family")
        else:
            raise ValueError(f"unknown distribution kind: {self.kind}")

    # ------- parametric family dispatch ---------------------------------
    def _frozen(self) -> stats.rv_continuous | stats.rv_discrete:
        assert self.kind == "parametric" and self.family is not None
        f = self.family
        p = self.params
        if f == "normal":
            return stats.norm(loc=p["loc"], scale=p["scale"])
        if f == "beta":
            return stats.beta(a=p["a"], b=p["b"])
        if f == "lognormal":
            return stats.lognorm(s=p["s"], scale=p.get("scale", 1.0))
        if f == "t":
            return stats.t(df=p["df"], loc=p.get("loc", 0.0), scale=p.get("scale", 1.0))
        if f == "poisson":
            return stats.poisson(mu=p["mu"])
        if f == "negbin":
            return stats.nbinom(n=p["n"], p=p["p"])
        raise ValueError(f"unsupported parametric family: {f}")

    # ------- summary stats ----------------------------------------------
    def mean(self) -> float:
        if self.kind == "samples":
            assert self.samples is not None
            return float(np.mean(self.samples))
        return float(self._frozen().mean())

    def std(self) -> float:
        if self.kind == "samples":
            assert self.samples is not None
            return float(np.std(self.samples, ddof=1))
        return float(self._frozen().std())

    def quantile(self, q: float | np.ndarray) -> float | np.ndarray:
        if self.kind == "samples":
            assert self.samples is not None
            return np.quantile(self.samples, q)
        return self._frozen().ppf(q)

    def cdf(self, x: float | np.ndarray) -> float | np.ndarray:
        if self.kind == "samples":
            assert self.samples is not None
            # Empirical CDF. Vectorized via searchsorted on a sorted copy.
            s = np.sort(self.samples)
            return np.searchsorted(s, x, side="right") / s.size
        return self._frozen().cdf(x)

    def prob_above(self, threshold: float) -> float:
        """P(X > threshold). Convenience for threshold contracts."""
        return float(1.0 - self.cdf(threshold))

    def prob_between(self, low: float, high: float) -> float:
        return float(self.cdf(high) - self.cdf(low))

    def draw(self, n: int, rng: np.random.Generator | None = None) -> np.ndarray:
        if self.kind == "samples":
            assert self.samples is not None
            r = rng or np.random.default_rng()
            return r.choice(self.samples, size=n, replace=True)
        return np.asarray(self._frozen().rvs(size=n))

    # ------- serialization ----------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "samples": self.samples.tolist() if self.samples is not None else None,
            "family": self.family,
            "params": dict(self.params),
        }


def bernoulli_entropy(p: np.ndarray | float) -> np.ndarray | float:
    """Binary entropy in nats; vectorized. H(0)=H(1)=0 by convention."""
    p = np.asarray(p, dtype=float)
    # Clip to avoid log(0). 1e-12 is safely below any resolution we care about.
    pc = np.clip(p, 1e-12, 1.0 - 1e-12)
    out = -(pc * np.log(pc) + (1.0 - pc) * np.log(1.0 - pc))
    return float(out) if out.ndim == 0 else out


def decompose_binary_uncertainty(p_samples: np.ndarray) -> dict[str, float]:
    """Split binary-outcome uncertainty into aleatoric and epistemic components.

    Given S samples of the YES probability p (from a posterior that encodes
    parameter and data uncertainty), we return:
        total      = H(E[p])
        aleatoric  = E[H(p)]
        epistemic  = total - aleatoric  (>= 0)
    All entropies in nats.
    """
    if p_samples.ndim != 1:
        raise ValueError("p_samples must be 1D")
    mean_p = float(np.mean(p_samples))
    total = float(bernoulli_entropy(mean_p))
    aleatoric = float(np.mean(bernoulli_entropy(p_samples)))
    epistemic = max(total - aleatoric, 0.0)
    return {"total": total, "aleatoric": aleatoric, "epistemic": epistemic}


@dataclass
class ForecastResult:
    """Output of a Forecaster. Includes methodology for audit + UI."""

    ticker: str
    ts: datetime
    forecaster: str
    version: str

    # The posterior over P(YES). For a binary contract this is a distribution
    # on [0, 1]; mean is the reported p_yes. Spread quantifies epistemic
    # uncertainty in the probability itself.
    binary_posterior: ForecastDistribution

    # Optional posterior over the underlying continuous quantity (e.g. the
    # CPI print, the daily-max temperature). When present the dashboard overlays
    # it with the market-implied distribution, which is the whole visual
    # payoff of the "model the DGP, not the outcome" principle.
    underlying_posterior: ForecastDistribution | None = None

    # Uncertainty decomposition (nats, Bernoulli). Populated by Forecaster.finalize.
    uncertainty: dict[str, float] = field(default_factory=dict)

    # Subjective-but-calibrated 0-1 score: how much do we trust this forecast?
    # Ranker gates real-time alerts on this. Forecasters must document how
    # they compute it; low scores should flow through even if edge looks big.
    model_confidence: float = 0.0

    # Human- and audit-facing methodology record. Shape:
    #   { "model": "...", "prior": "...", "likelihood": "...",
    #     "notes": "...", "sensitivity": {...}, "assumptions": [...] }
    methodology: dict[str, Any] = field(default_factory=dict)

    # Data provenance: {source_name: last_update_timestamp}. Staleness of
    # any single source propagates to model_confidence in finalize().
    data_sources: dict[str, datetime] = field(default_factory=dict)

    # Convergence / goodness-of-fit diagnostics. MCMC r-hat, n_eff,
    # divergences; ensemble-vs-climate rank-histogram stat; etc.
    diagnostics: dict[str, Any] = field(default_factory=dict)

    # Populated instead of a posterior when a forecaster cannot responsibly
    # produce a forecast (missing data, unsupported resolution rule, etc).
    # Rationale: "no fake precision" per project principles.
    null_reason: str | None = None

    @property
    def p_yes(self) -> float:
        return self.binary_posterior.mean()

    @property
    def p_yes_std(self) -> float:
        return self.binary_posterior.std()

    def p_yes_quantile(self, q: float) -> float:
        return float(self.binary_posterior.quantile(q))

    def is_null(self) -> bool:
        return self.null_reason is not None
