"""Core fundamental-input types.

A ``FundamentalInput`` is the atomic unit of the research layer. It bundles
a measured (or analyst-estimated) scalar together with everything downstream
code needs to use it responsibly:

    * ``value`` / ``uncertainty`` — the number and its one-σ bar. If uncertainty
      is unknown or meaningless (e.g., a regime indicator), ``uncertainty`` is
      ``None``; downstream code is required to notice and handle this.
    * ``provenance`` — where this number came from, when it was fetched, who
      (or what) entered it. The dashboard surfaces this verbatim: no
      "trust me" inputs.
    * ``expires_at`` — when the number must be refreshed. If expired at
      forecast time the integration engine drops the input and records the
      drop in the attribution log rather than using a stale value.
    * ``mechanism`` — the *named* way this input modifies the forecast.
      ``loading`` — the calibrated β (with uncertainty) that quantifies how
      much a unit anomaly in this input moves the forecast's underlying
      variable, in the units the forecaster uses.

The discipline: if you can't fill all of these fields for a proposed
research signal, the signal doesn't go into the system. The whole point
of this package is to prevent narrative → forecast-number leakage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class IntegrationMechanism(str, Enum):
    """How a fundamental input modifies the quant forecast.

    * ``prior_shift`` — shifts mean (optionally variance) of the forecaster's
      prior on its underlying continuous quantity. This is the workhorse for
      economic-release forecasters.
    * ``likelihood_weight`` — reweights historical observations the quant
      model is fitting to (e.g. upweight polls from similar-regime elections).
    * ``regime_indicator`` — selects which of several sub-models the
      forecaster runs (e.g., jump-diffusion during supply-chain shocks).
    * ``structural_break`` — tells the quant model to discount or truncate
      history before a flagged date.
    * ``additional_observation`` — the input is treated as a noisy observation
      of the latent underlying state and folded into the posterior via Bayes.

    These are the five mechanisms the integration engine knows how to apply.
    Adding a new mechanism requires extending ``integration.engine``.
    """

    PRIOR_SHIFT = "prior_shift"
    LIKELIHOOD_WEIGHT = "likelihood_weight"
    REGIME_INDICATOR = "regime_indicator"
    STRUCTURAL_BREAK = "structural_break"
    ADDITIONAL_OBSERVATION = "additional_observation"


class InputProvenance(BaseModel):
    """Where a fundamental-input value came from."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source: str                                # "FRED:GASREGW", "analyst:alee", ...
    source_kind: Literal["automated", "manual"]
    fetched_at: datetime                       # UTC
    observation_at: datetime                   # the event / observation timestamp
    notes: str | None = None                   # free-text, audit only, never used numerically

    @field_validator("fetched_at", "observation_at")
    @classmethod
    def _require_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            # Interpret naive as UTC. Fail loud if this was an accident upstream.
            return v.replace(tzinfo=timezone.utc)
        return v.astimezone(timezone.utc)


class InputLoading(BaseModel):
    """Calibrated β for a fundamental input.

    ``beta`` is the linear loading: a unit anomaly in the input moves the
    forecaster's underlying by ``beta`` units (in the forecaster's native
    units — e.g., CPI MoM percentage points). ``beta_std`` is the one-σ
    bootstrap uncertainty of that estimate. ``n_obs`` is the sample size
    the estimate was fit on — small-sample estimates get shrunk toward zero
    by the engine so we don't trust an overfit loading.

    ``mechanism`` must match the ``FundamentalInput.mechanism`` it applies
    to; the engine refuses to apply a mismatched loading.

    ``baseline`` is the zero-anomaly reference value the input's anomaly is
    measured against. For a first-difference input (Δ gasoline) this is
    zero; for a level input (current GSCPI) it's the rolling climatology.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    mechanism: IntegrationMechanism
    beta: float
    beta_std: float = 0.0
    baseline: float = 0.0
    n_obs: int = 0
    fit_method: str = "ols_bootstrap"
    fit_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("beta_std", "n_obs")
    @classmethod
    def _nonneg(cls, v: float) -> float:
        if v < 0:
            raise ValueError("beta_std and n_obs must be non-negative")
        return v

    @field_validator("fit_at")
    @classmethod
    def _require_utc(cls, v: datetime) -> datetime:
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)

    def shrinkage_weight(self, prior_n: float = 30.0) -> float:
        """Weight on the MLE under a normal-normal conjugate shrinkage prior.

        We interpret the MLE variance as ``~ 1/n_obs`` and the prior as
        ``β ~ N(0, 1/prior_n)``. The posterior mean is ``w · β̂`` and
        the posterior variance is ``w / (n_obs + prior_n)`` where
        ``w = n_obs / (n_obs + prior_n)``.
        """
        if self.n_obs <= 0:
            return 0.0
        return float(self.n_obs / (self.n_obs + float(prior_n)))

    def shrunk_beta(self, prior_n: float = 30.0) -> float:
        """Posterior mean of β under the shrinkage prior.

        30 is a reasonable default for monthly series where 2-3 years of
        history starts to earn trust. Forecasters override per-input via
        the schema's ``shrinkage_prior_n``.
        """
        return float(self.shrinkage_weight(prior_n) * self.beta)

    def shrunk_beta_std(self, prior_n: float = 30.0) -> float:
        """Posterior std of β. Under the normal-normal conjugate model the
        posterior variance is ``w · V_mle`` where ``V_mle = beta_std**2``,
        so the posterior std is ``sqrt(w) · beta_std``. This is the right
        thing to sample β from when propagating loading uncertainty into
        the forecast posterior.
        """
        w = self.shrinkage_weight(prior_n)
        return float(np.sqrt(w) * float(self.beta_std))


class FundamentalInput(BaseModel):
    """A single fundamental-research signal.

    Flows from ``/fundamental/automated`` (live data pulls) or ``/fundamental/
    manual`` (YAML committed to ``analyst_inputs/``) into the integration
    engine, which consumes the ``mechanism`` + ``loading`` together with a
    quant forecaster's prior to produce the quantamental posterior.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str                                  # stable identifier, e.g. "gasoline_weekly_delta"
    category: str                              # Kalshi category (cpi uses "economics")
    value: float
    uncertainty: float | None = None           # 1σ, same units as value
    mechanism: IntegrationMechanism

    provenance: InputProvenance
    expires_at: datetime                       # UTC

    # Optional free-form annotation for the engine / brief renderer.
    # E.g. {"transform": "mom"} to scope which CPI transform this applies to,
    # or {"regime": "supply_shock"} for regime indicators.
    scope: dict[str, Any] = Field(default_factory=dict)

    @field_validator("expires_at")
    @classmethod
    def _require_utc(cls, v: datetime) -> datetime:
        return v.replace(tzinfo=timezone.utc) if v.tzinfo is None else v.astimezone(timezone.utc)

    @model_validator(mode="after")
    def _uncertainty_sign(self) -> "FundamentalInput":
        if self.uncertainty is not None and self.uncertainty < 0:
            raise ValueError("uncertainty must be non-negative")
        return self

    def is_expired(self, now: datetime | None = None) -> bool:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return now >= self.expires_at

    def age_hours(self, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        if now.tzinfo is None:
            now = now.replace(tzinfo=timezone.utc)
        return max((now - self.provenance.fetched_at).total_seconds() / 3600.0, 0.0)


class InputSpec(BaseModel):
    """Declarative spec for a fundamental input that a category schema expects.

    Decoupled from ``FundamentalInput`` so the integration engine and the
    research brief generator can enumerate "what inputs should exist for this
    category" without needing live values in hand.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    category: str
    mechanism: IntegrationMechanism
    units: str                                 # human-readable: "pp/month", "z-score", ...
    description: str
    expected_source: str                       # "FRED:GASREGW" etc
    # Freshness expectation: if data is older than this, consider it expired.
    freshness_hours: float = 24.0 * 14
    # When this input is scope-sensitive (e.g. CPI MoM vs YoY) the forecaster
    # can inspect ``applies_to`` to decide whether to consume it.
    applies_to: dict[str, Any] = Field(default_factory=dict)
    # Shrinkage prior-n override (see ``InputLoading.shrunk_beta``).
    shrinkage_prior_n: float = 30.0
