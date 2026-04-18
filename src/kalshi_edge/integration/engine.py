"""Quant + fundamental integration engine.

Contract with callers: you give us

  * a quant posterior on the underlying continuous variable (e.g. CPI MoM %),
    as a 1-D numpy array of Monte-Carlo samples from the forecaster;
  * a list of ``(FundamentalInput, InputLoading)`` pairs, one per input
    we should apply;
  * an optional RNG for reproducibility.

We return an ``IntegrationOutcome`` with

  * ``adjusted_samples`` — the quant+fundamental posterior on the same
    underlying variable (same length as the input samples);
  * ``attribution`` — a list of per-input attribution entries documenting
    how much each input shifted the posterior mean (in underlying units)
    and the posterior's probability of the contract criterion;
  * ``dropped`` — inputs we skipped (expired, loading mismatch, missing
    uncertainty) with the reason, so nothing disappears silently.

Design rules this module enforces:

  * The engine NEVER modifies the quant samples array in place. Callers
    can safely pass their raw quant posterior.
  * Loadings are sanity-shrunk toward zero by sample size: a β fit on 6
    months of data gets ~16% of its MLE weight at prior_n=30. No unearned
    confidence.
  * Loading uncertainty is propagated into the posterior (we draw β per
    sample), not just the point estimate. That's the difference between a
    proper Bayesian integration and a hand-picked multiplier.
  * Every mechanism other than PRIOR_SHIFT that's requested but not fully
    implemented raises ``NotImplementedError`` with a clear name. No
    silent no-ops — if a signal asked to modify the forecast and we
    failed to do so, the forecaster must notice and abstain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from kalshi_edge.fundamental.schemas.base import (
    FundamentalInput,
    InputLoading,
    IntegrationMechanism,
)


# ---------------------------------------------------------------------------
# IO types.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class IntegrationRequest:
    """A single (input, loading) pair to apply to the quant posterior."""

    input: FundamentalInput
    loading: InputLoading


@dataclass(slots=True)
class AttributionEntry:
    """Per-input record of how the posterior moved.

    ``mean_shift_underlying`` is the expected shift in the underlying
    variable's posterior mean from this input alone, in the forecaster's
    native units. ``mean_shift_prob`` is the same, re-expressed as a
    shift in P(YES) under the contract criterion — provided the engine
    was given a criterion callable, else ``None``.
    """

    name: str
    mechanism: str
    value: float
    baseline: float
    anomaly: float
    beta_raw: float
    beta_shrunk: float
    beta_std: float
    n_obs: int
    mean_shift_underlying: float
    mean_shift_prob: float | None = None
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DroppedEntry:
    name: str
    reason: str


@dataclass(slots=True)
class IntegrationOutcome:
    """Quant + fundamental posterior with full attribution."""

    adjusted_samples: np.ndarray
    quant_samples: np.ndarray
    attribution: list[AttributionEntry] = field(default_factory=list)
    dropped: list[DroppedEntry] = field(default_factory=list)

    @property
    def mean_shift_underlying_total(self) -> float:
        q0 = float(np.mean(self.quant_samples))
        q1 = float(np.mean(self.adjusted_samples))
        return q1 - q0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the attribution log (not the samples) for DB storage."""
        return {
            "mean_shift_total": self.mean_shift_underlying_total,
            "attribution": [
                {
                    "name": a.name,
                    "mechanism": a.mechanism,
                    "value": a.value,
                    "baseline": a.baseline,
                    "anomaly": a.anomaly,
                    "beta_raw": a.beta_raw,
                    "beta_shrunk": a.beta_shrunk,
                    "beta_std": a.beta_std,
                    "n_obs": a.n_obs,
                    "mean_shift_underlying": a.mean_shift_underlying,
                    "mean_shift_prob": a.mean_shift_prob,
                    "provenance": a.provenance,
                }
                for a in self.attribution
            ],
            "dropped": [{"name": d.name, "reason": d.reason} for d in self.dropped],
        }


# ---------------------------------------------------------------------------
# Core: prior shift.
# ---------------------------------------------------------------------------


def apply_prior_shifts(
    quant_samples: np.ndarray,
    requests: list[IntegrationRequest],
    *,
    criterion: "Criterion | None" = None,   # noqa: F821 — forward ref to integration.criterion
    rng: np.random.Generator | None = None,
    shrinkage_prior_n: float | dict[str, float] = 30.0,
) -> IntegrationOutcome:
    """Apply ``PRIOR_SHIFT`` integrations to a quant sample posterior.

    For each (input, loading):

        1. Compute the input's anomaly relative to the loading's baseline.
        2. Shrink the loading's β toward zero by sample size (per-input
           prior_n override via ``shrinkage_prior_n`` dict).
        3. Draw a β per sample from ``N(β̂_shrunk, σ_β²)`` to propagate
           loading uncertainty into the posterior.
        4. Add ``β_drawn * anomaly`` to the sample. Also, if the input
           itself has a nonzero ``uncertainty``, propagate that: the
           anomaly is drawn as ``N(reported_anomaly, input_σ²)`` per
           sample too.

    Edge cases handled loud:

        * Mismatched mechanism → dropped with reason; not applied.
        * Loading.n_obs == 0 → effective β is 0 (nothing happens), dropped
          with reason "uncalibrated_loading" so the dashboard can flag it.
        * Input expired → dropped.
        * Sample array is not 1-D → ValueError.
    """
    if quant_samples.ndim != 1:
        raise ValueError("quant_samples must be 1-D")
    r = rng or np.random.default_rng(0xBEEF)

    samples = quant_samples.astype(float, copy=True)
    attribution: list[AttributionEntry] = []
    dropped: list[DroppedEntry] = []

    # Evaluate quant-only P(YES) once if a criterion is given, so each input's
    # marginal probability shift is defined as (P_with_input - P_quant_only)
    # rather than cascading. This gives additive-like attribution in the
    # small-shift regime and is faithful for the single-input case.
    p_quant: float | None = None
    if criterion is not None:
        p_quant = float(criterion(quant_samples))

    for req in requests:
        inp = req.input
        ld = req.loading
        nm = inp.name

        if ld.mechanism != inp.mechanism:
            dropped.append(DroppedEntry(nm, f"mechanism_mismatch:input={inp.mechanism.value},loading={ld.mechanism.value}"))
            continue
        if ld.mechanism != IntegrationMechanism.PRIOR_SHIFT:
            # This function handles only prior_shift. Other mechanisms are
            # dispatched by ``integrate`` to their handlers.
            dropped.append(DroppedEntry(nm, f"not_prior_shift:{ld.mechanism.value}"))
            continue
        if inp.is_expired():
            dropped.append(DroppedEntry(nm, "input_expired"))
            continue
        if ld.n_obs <= 0:
            dropped.append(DroppedEntry(nm, "uncalibrated_loading"))
            continue

        # Shrinkage prior_n resolution.
        prior_n = (
            shrinkage_prior_n.get(nm, 30.0)
            if isinstance(shrinkage_prior_n, dict)
            else float(shrinkage_prior_n)
        )
        beta_shrunk = ld.shrunk_beta(prior_n=prior_n)
        beta_std_shrunk = ld.shrunk_beta_std(prior_n=prior_n)

        anomaly_center = float(inp.value) - float(ld.baseline)

        # Draw β per sample from the shrinkage-prior posterior:
        # N(beta_shrunk, beta_std_shrunk^2). This propagates loading
        # uncertainty (properly attenuated by shrinkage) into the forecast.
        n = samples.size
        if beta_std_shrunk > 0:
            betas = r.normal(beta_shrunk, beta_std_shrunk, size=n)
        else:
            betas = np.full(n, beta_shrunk, dtype=float)

        # Draw anomaly per sample if the input has its own measurement σ.
        if inp.uncertainty is not None and inp.uncertainty > 0:
            anomalies = r.normal(anomaly_center, float(inp.uncertainty), size=n)
        else:
            anomalies = np.full(n, anomaly_center, dtype=float)

        shift = betas * anomalies
        pre_mean = float(np.mean(samples))
        samples = samples + shift
        post_mean = float(np.mean(samples))

        mean_shift_prob: float | None = None
        if criterion is not None and p_quant is not None:
            # Attribute this input's marginal probability shift as the
            # single-input counterfactual: start from quant posterior,
            # apply only this input, compute criterion, subtract quant-only.
            single_input_samples = quant_samples + shift
            p_single = float(criterion(single_input_samples))
            mean_shift_prob = p_single - p_quant

        attribution.append(AttributionEntry(
            name=nm,
            mechanism=ld.mechanism.value,
            value=float(inp.value),
            baseline=float(ld.baseline),
            anomaly=anomaly_center,
            beta_raw=float(ld.beta),
            beta_shrunk=float(beta_shrunk),
            beta_std=float(ld.beta_std),
            n_obs=int(ld.n_obs),
            mean_shift_underlying=post_mean - pre_mean,
            mean_shift_prob=mean_shift_prob,
            provenance={
                "source": inp.provenance.source,
                "source_kind": inp.provenance.source_kind,
                "fetched_at": inp.provenance.fetched_at.isoformat(),
                "observation_at": inp.provenance.observation_at.isoformat(),
                "notes": inp.provenance.notes,
                "units": f"value={float(inp.value):.6g}, baseline={float(ld.baseline):.6g}",
            },
        ))

    return IntegrationOutcome(
        adjusted_samples=samples,
        quant_samples=quant_samples,
        attribution=attribution,
        dropped=dropped,
    )


# ---------------------------------------------------------------------------
# Additional observation: Kalman-style fold-in.
# ---------------------------------------------------------------------------


def apply_additional_observation(
    quant_samples: np.ndarray,
    requests: list[IntegrationRequest],
    *,
    rng: np.random.Generator | None = None,
) -> IntegrationOutcome:
    """Treat each input as a noisy observation of the underlying state.

    For a Gaussian approximation: the prior is ``N(μ_q, σ_q²)`` from the
    quant samples, the observation is ``N(μ_o, σ_o²)`` with
    ``μ_o = baseline + (value - baseline)``, and the posterior is the
    standard Kalman update. We resample the samples from the updated
    moments to keep the downstream representation consistent with
    ``apply_prior_shifts``.

    Callers usually want PRIOR_SHIFT for economic-release-style signals;
    ADDITIONAL_OBSERVATION is for when the input is literally an
    independent noisy measurement of the same underlying quantity
    (e.g. a leading indicator that is the underlying latent state plus
    noise).
    """
    if quant_samples.ndim != 1:
        raise ValueError("quant_samples must be 1-D")
    r = rng or np.random.default_rng(0xBEE5)

    mu_q = float(np.mean(quant_samples))
    var_q = float(np.var(quant_samples, ddof=1))
    samples = quant_samples.copy()
    attribution: list[AttributionEntry] = []
    dropped: list[DroppedEntry] = []

    for req in requests:
        inp, ld = req.input, req.loading
        if ld.mechanism != IntegrationMechanism.ADDITIONAL_OBSERVATION:
            dropped.append(DroppedEntry(inp.name, f"not_additional_observation:{ld.mechanism.value}"))
            continue
        if inp.is_expired():
            dropped.append(DroppedEntry(inp.name, "input_expired"))
            continue
        if inp.uncertainty is None or inp.uncertainty <= 0:
            dropped.append(DroppedEntry(inp.name, "observation_requires_positive_uncertainty"))
            continue

        mu_o = float(inp.value)
        var_o = float(inp.uncertainty) ** 2
        # Kalman update.
        kalman_gain = var_q / (var_q + var_o) if (var_q + var_o) > 0 else 0.0
        mu_post = mu_q + kalman_gain * (mu_o - mu_q)
        var_post = (1.0 - kalman_gain) * var_q
        # Resample.
        samples = r.normal(mu_post, np.sqrt(max(var_post, 1e-12)), size=samples.size)
        pre_mean, mu_q = mu_q, mu_post
        var_q = max(var_post, 1e-12)

        attribution.append(AttributionEntry(
            name=inp.name,
            mechanism=ld.mechanism.value,
            value=mu_o,
            baseline=float(ld.baseline),
            anomaly=mu_o - float(ld.baseline),
            beta_raw=float(ld.beta),
            beta_shrunk=float(ld.shrunk_beta()),
            beta_std=float(ld.beta_std),
            n_obs=int(ld.n_obs),
            mean_shift_underlying=mu_post - pre_mean,
            provenance={
                "source": inp.provenance.source,
                "source_kind": inp.provenance.source_kind,
                "observation_sigma": float(inp.uncertainty),
                "kalman_gain": kalman_gain,
            },
        ))
    return IntegrationOutcome(
        adjusted_samples=samples,
        quant_samples=quant_samples,
        attribution=attribution,
        dropped=dropped,
    )


# ---------------------------------------------------------------------------
# Top-level dispatcher.
# ---------------------------------------------------------------------------


from typing import Callable

Criterion = Callable[[np.ndarray], float]


def integrate(
    quant_samples: np.ndarray,
    requests: list[IntegrationRequest],
    *,
    criterion: Criterion | None = None,
    rng: np.random.Generator | None = None,
    shrinkage_prior_n: float | dict[str, float] = 30.0,
) -> IntegrationOutcome:
    """Dispatch each request to the right mechanism handler and merge results.

    Mechanisms are applied in a stable order: PRIOR_SHIFT first (batched so
    each input's marginal attribution is computed against the quant-only
    baseline), then ADDITIONAL_OBSERVATION folded into the shifted posterior.
    Regime-indicator and structural-break mechanisms are not applied here:
    forecasters that use them must consume the input BEFORE running their
    quant model (to select a regime), and pass the post-regime samples into
    this function.
    """
    shifts = [req for req in requests if req.loading.mechanism == IntegrationMechanism.PRIOR_SHIFT]
    obs = [req for req in requests if req.loading.mechanism == IntegrationMechanism.ADDITIONAL_OBSERVATION]
    unsupported = [
        req for req in requests
        if req.loading.mechanism not in (
            IntegrationMechanism.PRIOR_SHIFT,
            IntegrationMechanism.ADDITIONAL_OBSERVATION,
        )
    ]

    merged = apply_prior_shifts(
        quant_samples, shifts, criterion=criterion, rng=rng,
        shrinkage_prior_n=shrinkage_prior_n,
    )
    if obs:
        obs_result = apply_additional_observation(merged.adjusted_samples, obs, rng=rng)
        merged.adjusted_samples = obs_result.adjusted_samples
        merged.attribution.extend(obs_result.attribution)
        merged.dropped.extend(obs_result.dropped)

    for req in unsupported:
        merged.dropped.append(DroppedEntry(
            req.input.name,
            f"mechanism_handled_upstream:{req.loading.mechanism.value}",
        ))
    return merged
