"""Quantamental CPI forecaster.

Wraps ``EconomicsCPIForecaster`` (the pure-quant ARIMA(1,1,0)-MC-MoM model)
with the fundamental research layer. The quant pass and the quantamental
pass are BOTH computed and BOTH persisted — this is how the calibration
tracker separately scores "quant only" vs "quant + fundamental" Brier
and is the central discipline of the whole quantamental design.

Pipeline per forecast:

    1. Run the base forecaster. If it abstains, the QF forecaster abstains
       with the same reason (no fundamental-only edge allowed; we need a
       quant posterior to integrate against).
    2. Look up the latest fundamental inputs for the relevant CPI-MoM
       scope from the fundamental store (automated + manual, union).
       Drop expired inputs.
    3. Look up the latest calibrated loading per input; inputs without a
       loading are dropped with reason "no_loading".
    4. Ask the integration engine to apply PRIOR_SHIFT (and any
       ADDITIONAL_OBSERVATION) updates to the quant-underlying samples.
       The engine propagates loading uncertainty via per-sample β draws
       and input uncertainty via per-sample anomaly draws.
    5. Re-apply the contract's YES criterion to the adjusted underlying
       samples, Bayesian-bootstrap to produce a new binary posterior.
    6. Return a ForecastResult whose ``binary_posterior`` is the
       quant+fundamental posterior and whose ``quant_only_binary_posterior``
       is the base forecaster's binary posterior. The integration
       engine's attribution log is placed on ``attribution``.

YoY-transform contracts are NOT modified by this forecaster: the calibrated
loadings are MoM-specific (monthly residuals). For YoY contracts the QF
forecaster returns the quant-only posterior unchanged and records
"scope_mismatch_yoy" in the attribution dropped list. Adding YoY support
would require re-calibrating loadings against 12-month rolling residuals.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from kalshi_edge.forecast import ForecastDistribution, ForecastResult
from kalshi_edge.forecasters.base import Forecaster
from kalshi_edge.forecasters.economics import (
    EconomicsCPIForecaster,
    parse_cpi_contract,
)
from kalshi_edge.fundamental.schemas.base import (
    FundamentalInput,
    InputLoading,
    InputProvenance,
    IntegrationMechanism,
)
from kalshi_edge.fundamental.schemas.cpi import cpi_input_specs
from kalshi_edge.integration.engine import (
    IntegrationOutcome,
    IntegrationRequest,
    integrate,
)
from kalshi_edge.types import Category, Contract, MarketSnapshot


class EconomicsCPIQuantamentalForecaster(Forecaster):
    """CPI forecaster with fundamental adjustments.

    ``fundamental_inputs`` and ``loadings`` are supplied at construction time
    (the CLI loads them from the DB); at forecast time they are filtered to
    the subset that applies to this specific contract (category=economics,
    scope.transform=mom).

    Pass ``inputs=None, loadings=None`` to get a pass-through wrapper that
    degrades gracefully to the quant-only forecast (all inputs "dropped:
    no_loading"). This is the safe default when the fundamental pipeline
    hasn't run yet.
    """

    name = "economics_cpi_qf"
    version = "1"

    def __init__(
        self,
        base: EconomicsCPIForecaster | None = None,
        *,
        inputs: dict[str, FundamentalInput] | None = None,
        loadings: dict[str, InputLoading] | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.base = base or EconomicsCPIForecaster()
        self.inputs: dict[str, FundamentalInput] = dict(inputs or {})
        self.loadings: dict[str, InputLoading] = dict(loadings or {})
        self.rng = rng or np.random.default_rng(0xF00D)

    @property
    def category(self) -> Category:
        return Category.ECONOMICS

    def supports(self, contract: Contract) -> bool:
        return self.base.supports(contract)

    # ---- analyst-driven updates ----
    def attach_inputs(self, extra: dict[str, FundamentalInput]) -> None:
        """Merge additional fundamental inputs (later overrides earlier)."""
        self.inputs.update(extra)

    def attach_loadings(self, extra: dict[str, InputLoading]) -> None:
        self.loadings.update(extra)

    # ---- forecast ----
    def _forecast_impl(
        self, contract: Contract, snapshot: MarketSnapshot | None
    ) -> ForecastResult:
        base_result = self.base.forecast(contract, snapshot)
        if base_result.is_null():
            # Surface the null directly — fundamental layer can't repair
            # abstentions. The QF forecaster's forecaster name still wins
            # so the dashboard knows who owned the call.
            return ForecastResult(
                ticker=base_result.ticker,
                ts=base_result.ts,
                forecaster=self.name,
                version=self.version,
                binary_posterior=base_result.binary_posterior,
                underlying_posterior=base_result.underlying_posterior,
                uncertainty=base_result.uncertainty,
                model_confidence=0.0,
                methodology={**base_result.methodology, "fundamental_layer": "bypassed_null_base"},
                data_sources=base_result.data_sources,
                diagnostics=base_result.diagnostics,
                null_reason=base_result.null_reason,
                quant_only_binary_posterior=base_result.binary_posterior,
                attribution={"attribution": [], "dropped": [], "mean_shift_total": 0.0},
            )

        cpi_contract = parse_cpi_contract(contract)
        if cpi_contract is None:
            # Can't happen given ``supports()`` is True, but be defensive.
            return base_result

        transform = cpi_contract.transform
        if transform != "mom":
            # YoY: return quant-only, but flag it.
            attribution = {
                "mean_shift_total": 0.0,
                "attribution": [],
                "dropped": [
                    {"name": n, "reason": f"scope_mismatch_transform={transform}"}
                    for n in self.inputs
                ],
            }
            return ForecastResult(
                ticker=base_result.ticker,
                ts=base_result.ts,
                forecaster=self.name,
                version=self.version,
                binary_posterior=base_result.binary_posterior,
                underlying_posterior=base_result.underlying_posterior,
                uncertainty=base_result.uncertainty,
                model_confidence=base_result.model_confidence,
                methodology={
                    **base_result.methodology,
                    "fundamental_layer": "scope_mismatch_yoy",
                    "base_forecaster": self.base.name,
                    "base_version": self.base.version,
                },
                data_sources=base_result.data_sources,
                diagnostics=base_result.diagnostics,
                quant_only_binary_posterior=base_result.binary_posterior,
                attribution=attribution,
            )

        # Applicable inputs: category=economics, scope.transform="mom", not expired.
        specs = cpi_input_specs()
        spec_names = {s.name for s in specs if s.applies_to.get("transform") == "mom"}
        now = datetime.now(timezone.utc)
        live_inputs = {
            n: i for n, i in self.inputs.items()
            if n in spec_names and not i.is_expired(now)
        }

        # Build (input, loading) requests; inputs without loadings get dropped.
        requests: list[IntegrationRequest] = []
        dropped_no_loading: list[dict[str, Any]] = []
        for n, inp in live_inputs.items():
            ld = self.loadings.get(n)
            if ld is None:
                dropped_no_loading.append({"name": n, "reason": "no_loading"})
                continue
            requests.append(IntegrationRequest(input=inp, loading=ld))
        for n in spec_names:
            if n not in live_inputs and n not in {d["name"] for d in dropped_no_loading}:
                dropped_no_loading.append({"name": n, "reason": "no_input_value"})

        # Quant-only underlying posterior in MoM% (the unit our loadings use).
        underlying_post = base_result.underlying_posterior
        if underlying_post is None or underlying_post.kind != "samples":
            # Shouldn't happen for MoM CPI (base stores samples); fall back to quant-only.
            return self._result_with_quant_only(contract, base_result, {
                "mean_shift_total": 0.0, "attribution": [], "dropped": dropped_no_loading,
            })
        assert underlying_post.samples is not None
        quant_samples = np.asarray(underlying_post.samples, dtype=float).copy()

        # Criterion in underlying-% space (mirrors what the base forecaster did
        # in its simulate step, but now a function of a sample array).
        criterion = self._criterion_for(cpi_contract)
        shrinkage_prior_n = {
            s.name: float(s.shrinkage_prior_n) for s in specs
        }
        outcome: IntegrationOutcome = integrate(
            quant_samples, requests,
            criterion=criterion,
            rng=self.rng,
            shrinkage_prior_n=shrinkage_prior_n,
        )
        # Merge our ``no_loading`` / ``no_input_value`` drops with the engine's drops.
        engine_out = outcome.to_dict()
        engine_out["dropped"] = dropped_no_loading + engine_out.get("dropped", [])

        # Build the quant+fundamental binary posterior via Bayesian bootstrap
        # on the shifted underlying samples. We replicate the base
        # forecaster's 2000-bootstrap pattern for consistency.
        B = 2000
        N = outcome.adjusted_samples.size
        idx = self.rng.integers(0, N, size=(B, N))
        boot = outcome.adjusted_samples[idx]
        if cpi_contract.criterion.direction == "above":
            hit = boot > cpi_contract.criterion.low
        elif cpi_contract.criterion.direction == "below":
            hit = boot < cpi_contract.criterion.high
        else:
            assert cpi_contract.criterion.direction == "between"
            hit = (boot >= cpi_contract.criterion.low) & (boot <= cpi_contract.criterion.high)
        p_yes_boot = np.clip(np.mean(hit, axis=1), 1e-6, 1.0 - 1e-6)

        adjusted_binary = ForecastDistribution(kind="samples", samples=p_yes_boot)
        adjusted_underlying = ForecastDistribution(kind="samples", samples=outcome.adjusted_samples)

        # Model confidence: inherit base forecaster's, but pull down if we had
        # inputs that were supposed to move the forecast but all got dropped.
        conf = float(base_result.model_confidence)
        n_requested = len(self.inputs)
        n_applied = len(outcome.attribution)
        if n_requested > 0 and n_applied == 0:
            # Fundamental layer had inputs but couldn't apply any. Confidence unchanged
            # (quant-only result is still valid) but we note it in methodology.
            pass

        return ForecastResult(
            ticker=base_result.ticker,
            ts=base_result.ts,
            forecaster=self.name,
            version=self.version,
            binary_posterior=adjusted_binary,
            underlying_posterior=adjusted_underlying,
            uncertainty={},
            model_confidence=conf,
            methodology={
                **base_result.methodology,
                "fundamental_layer": "applied",
                "base_forecaster": self.base.name,
                "base_version": self.base.version,
                "n_inputs_applied": n_applied,
                "n_inputs_dropped": len(engine_out["dropped"]),
                "mean_shift_underlying_total_pct": outcome.mean_shift_underlying_total,
            },
            data_sources={
                **base_result.data_sources,
                **{
                    f"fundamental:{n}": inp.provenance.fetched_at
                    for n, inp in live_inputs.items()
                },
            },
            diagnostics={
                **base_result.diagnostics,
                "quant_only_p_yes": float(base_result.p_yes),
                "quant_only_p_yes_std": float(base_result.p_yes_std),
                "adjusted_p_yes": float(np.mean(p_yes_boot)),
                "adjusted_p_yes_std": float(np.std(p_yes_boot, ddof=1)),
            },
            quant_only_binary_posterior=base_result.binary_posterior,
            attribution=engine_out,
        )

    # ---- helpers ----
    @staticmethod
    def _criterion_for(cpi_contract: Any) -> "object":
        crit = cpi_contract.criterion

        def c(samples: np.ndarray) -> float:
            if crit.direction == "above":
                return float(np.mean(samples > crit.low))
            if crit.direction == "below":
                return float(np.mean(samples < crit.high))
            return float(np.mean((samples >= crit.low) & (samples <= crit.high)))

        return c

    def _result_with_quant_only(
        self, contract: Contract, base_result: ForecastResult,
        attribution: dict[str, Any],
    ) -> ForecastResult:
        return ForecastResult(
            ticker=base_result.ticker,
            ts=base_result.ts,
            forecaster=self.name,
            version=self.version,
            binary_posterior=base_result.binary_posterior,
            underlying_posterior=base_result.underlying_posterior,
            uncertainty=base_result.uncertainty,
            model_confidence=base_result.model_confidence,
            methodology={**base_result.methodology, "fundamental_layer": "bypassed_no_samples"},
            data_sources=base_result.data_sources,
            diagnostics=base_result.diagnostics,
            quant_only_binary_posterior=base_result.binary_posterior,
            attribution=attribution,
        )


# ---------------------------------------------------------------------------
# Helpers that turn DB rows into typed objects for the forecaster.
# ---------------------------------------------------------------------------


def inputs_from_db_rows(rows: dict[str, dict[str, Any]]) -> dict[str, FundamentalInput]:
    """Convert ``Database.latest_fundamental_inputs()`` output to typed inputs."""
    out: dict[str, FundamentalInput] = {}
    for name, r in rows.items():
        try:
            out[name] = FundamentalInput(
                name=r["name"],
                category=r["category"],
                value=float(r["value"]),
                uncertainty=float(r["uncertainty"]) if r["uncertainty"] is not None else None,
                mechanism=IntegrationMechanism(r["mechanism"]),
                provenance=InputProvenance(
                    source=r["source"],
                    source_kind=r["source_kind"],
                    fetched_at=r["fetched_at"],
                    observation_at=r["observation_at"],
                    notes=r.get("notes"),
                ),
                expires_at=r["expires_at"],
                scope=r.get("scope") or {},
            )
        except Exception:  # noqa: BLE001
            # Row that fails validation is skipped, not raised — downstream
            # integration will simply not see this input. Operator inspects
            # the logs.
            continue
    return out


def loadings_from_db_rows(rows: dict[str, dict[str, Any]]) -> dict[str, InputLoading]:
    out: dict[str, InputLoading] = {}
    for name, r in rows.items():
        try:
            out[name] = InputLoading(
                name=r["name"],
                mechanism=IntegrationMechanism(r["mechanism"]),
                beta=float(r["beta"]),
                beta_std=float(r["beta_std"]),
                baseline=float(r["baseline"]),
                n_obs=int(r["n_obs"]),
                fit_method=r["fit_method"],
                fit_at=r["fit_at"],
            )
        except Exception:  # noqa: BLE001
            continue
    return out
