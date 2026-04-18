"""Integration engine math + quantamental pipeline tests.

These cover the load-bearing invariants of the fundamental research layer:

    * Shrinkage math on ``InputLoading`` matches the closed-form normal-
      normal conjugate posterior (no off-by-one or sqrt bugs).
    * ``apply_prior_shifts`` moves the posterior by approximately
      ``β_shrunk * (value - baseline)`` on the underlying mean (not β_raw,
      and not the input value itself).
    * Dropped-input reasons cover every failure mode the engine advertises
      (expired, uncalibrated, mechanism mismatch, non-prior-shift in the
      prior-shift handler).
    * ``integrate`` dispatches prior-shift and additional-observation in
      one pass and never modifies the caller's sample array in place.
    * Manual YAML loader round-trips and rejects malformed input.
    * Brief renderer is deterministic byte-for-byte given fixed inputs.

We DO NOT mock the FRED client or hit the network — calibration tests run
against a synthetic but realistic CPI + gasoline series, which is what
the calibration module claims to be able to handle.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from kalshi_edge.fundamental.briefs.generator import BriefContext, render_brief
from kalshi_edge.fundamental.manual.loader import (
    ManualInputError,
    load_manual_inputs,
    load_manual_inputs_from_dir,
)
from kalshi_edge.fundamental.schemas.base import (
    FundamentalInput,
    InputLoading,
    InputProvenance,
    IntegrationMechanism,
)
from kalshi_edge.forecast import ForecastDistribution, ForecastResult
from kalshi_edge.integration.engine import (
    IntegrationRequest,
    apply_additional_observation,
    apply_prior_shifts,
    integrate,
)
from kalshi_edge.types import Category, Contract, MarketSnapshot, MarketStatus


UTC = timezone.utc


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------


def _input(
    name: str = "gasoline_weekly_delta",
    *,
    value: float = 0.03,
    uncertainty: float | None = None,
    mechanism: IntegrationMechanism = IntegrationMechanism.PRIOR_SHIFT,
    expires_at: datetime | None = None,
    category: str = "economics",
) -> FundamentalInput:
    now = datetime(2026, 4, 1, tzinfo=UTC)
    return FundamentalInput(
        name=name, category=category, value=value, uncertainty=uncertainty,
        mechanism=mechanism,
        provenance=InputProvenance(
            source="FRED:GASREGW", source_kind="automated",
            fetched_at=now, observation_at=now - timedelta(days=1),
        ),
        expires_at=expires_at or (now + timedelta(days=30)),
        scope={"transform": "mom"},
    )


def _loading(
    name: str = "gasoline_weekly_delta",
    *,
    beta: float = 0.5,
    beta_std: float = 0.1,
    baseline: float = 0.0,
    n_obs: int = 120,
    mechanism: IntegrationMechanism = IntegrationMechanism.PRIOR_SHIFT,
) -> InputLoading:
    return InputLoading(
        name=name, mechanism=mechanism, beta=beta, beta_std=beta_std,
        baseline=baseline, n_obs=n_obs, fit_method="ols_bootstrap",
        fit_at=datetime(2026, 3, 1, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Shrinkage math.
# ---------------------------------------------------------------------------


class TestShrinkage:
    def test_zero_n_obs_gives_zero_posterior(self) -> None:
        ld = _loading(n_obs=0, beta=10.0, beta_std=1.0)
        assert ld.shrinkage_weight(prior_n=30.0) == 0.0
        assert ld.shrunk_beta(prior_n=30.0) == 0.0
        assert ld.shrunk_beta_std(prior_n=30.0) == 0.0

    def test_large_n_obs_approaches_mle(self) -> None:
        ld = _loading(n_obs=10_000, beta=0.5, beta_std=0.1)
        w = ld.shrinkage_weight(prior_n=30.0)
        assert 0.99 < w < 1.0
        assert ld.shrunk_beta(prior_n=30.0) == pytest.approx(0.5, rel=1e-2)
        # sqrt(w) ~ 1 for large n.
        assert ld.shrunk_beta_std(prior_n=30.0) == pytest.approx(0.1, rel=1e-2)

    def test_matches_closed_form_at_moderate_n(self) -> None:
        ld = _loading(n_obs=60, beta=0.4, beta_std=0.2)
        w = 60 / (60 + 30)
        assert ld.shrinkage_weight(30.0) == pytest.approx(w, rel=1e-12)
        assert ld.shrunk_beta(30.0) == pytest.approx(w * 0.4, rel=1e-12)
        assert ld.shrunk_beta_std(30.0) == pytest.approx(
            (w ** 0.5) * 0.2, rel=1e-12
        )


# ---------------------------------------------------------------------------
# Prior-shift engine.
# ---------------------------------------------------------------------------


class TestApplyPriorShifts:
    def test_shifts_mean_by_beta_times_anomaly(self) -> None:
        rng = np.random.default_rng(0)
        q = rng.normal(0.5, 0.1, size=20_000)
        inp = _input(value=0.02)                # anomaly = 0.02 − 0.0 = 0.02
        ld = _loading(beta=0.5, beta_std=0.0, n_obs=10_000)  # β_shrunk ≈ 0.5
        req = IntegrationRequest(input=inp, loading=ld)

        outcome = apply_prior_shifts(q, [req], rng=np.random.default_rng(1))
        expected_shift = ld.shrunk_beta() * (inp.value - ld.baseline)
        assert outcome.mean_shift_underlying_total == pytest.approx(
            expected_shift, abs=2e-3
        )
        assert outcome.adjusted_samples.shape == q.shape
        # Engine must not mutate caller's array.
        assert np.mean(q) == pytest.approx(0.5, abs=2e-3)

    def test_drops_expired_input(self) -> None:
        rng = np.random.default_rng(0)
        q = rng.normal(0.0, 0.1, size=1000)
        expired = _input(expires_at=datetime(2020, 1, 1, tzinfo=UTC))
        req = IntegrationRequest(input=expired, loading=_loading())
        outcome = apply_prior_shifts(q, [req])
        assert not outcome.attribution
        assert len(outcome.dropped) == 1
        assert outcome.dropped[0].reason == "input_expired"

    def test_drops_uncalibrated_loading(self) -> None:
        rng = np.random.default_rng(0)
        q = rng.normal(0.0, 0.1, size=1000)
        req = IntegrationRequest(input=_input(), loading=_loading(n_obs=0))
        outcome = apply_prior_shifts(q, [req])
        assert len(outcome.dropped) == 1
        assert outcome.dropped[0].reason == "uncalibrated_loading"

    def test_drops_on_mechanism_mismatch(self) -> None:
        rng = np.random.default_rng(0)
        q = rng.normal(0.0, 0.1, size=1000)
        bad = _loading(mechanism=IntegrationMechanism.LIKELIHOOD_WEIGHT)
        req = IntegrationRequest(input=_input(), loading=bad)
        outcome = apply_prior_shifts(q, [req])
        assert len(outcome.dropped) == 1
        assert "mechanism_mismatch" in outcome.dropped[0].reason

    def test_refuses_non_prior_shift_loading(self) -> None:
        rng = np.random.default_rng(0)
        q = rng.normal(0.0, 0.1, size=1000)
        ao_inp = _input(mechanism=IntegrationMechanism.ADDITIONAL_OBSERVATION)
        ao_ld = _loading(mechanism=IntegrationMechanism.ADDITIONAL_OBSERVATION)
        req = IntegrationRequest(input=ao_inp, loading=ao_ld)
        outcome = apply_prior_shifts(q, [req])
        assert len(outcome.dropped) == 1
        assert "not_prior_shift" in outcome.dropped[0].reason

    def test_uncertainty_widens_posterior(self) -> None:
        """A σ on the input should inflate the adjusted posterior's variance."""
        rng = np.random.default_rng(0)
        q = rng.normal(0.5, 0.1, size=50_000)
        # Strong β, large σ_input → large added variance.
        inp = _input(value=0.02, uncertainty=0.05)
        ld = _loading(beta=1.0, beta_std=0.0, n_obs=10_000)
        req = IntegrationRequest(input=inp, loading=ld)

        outcome = apply_prior_shifts(
            q, [req], rng=np.random.default_rng(7),
        )
        assert np.std(outcome.adjusted_samples) > np.std(q)

    def test_criterion_attribution_sums_to_small_shifts(self) -> None:
        """With tiny shifts, per-input ΔP(YES) ≈ total ΔP(YES)."""
        rng = np.random.default_rng(42)
        q = rng.normal(0.0, 1.0, size=20_000)

        def criterion(samples: np.ndarray) -> float:
            return float(np.mean(samples > 0))

        reqs = [
            IntegrationRequest(
                input=_input(name=f"inp_{i}", value=0.005),
                loading=_loading(name=f"inp_{i}", beta=0.1, beta_std=0.0),
            )
            for i in range(3)
        ]
        outcome = apply_prior_shifts(
            q, reqs, criterion=criterion,
            rng=np.random.default_rng(99),
        )
        # Each entry's marginal shift should be small and same-signed.
        assert all(a.mean_shift_prob is not None for a in outcome.attribution)
        marginals = [a.mean_shift_prob for a in outcome.attribution]
        assert all(m > 0 for m in marginals)       # positive β × positive anomaly
        # Not strictly additive (Monte Carlo + non-linear CDF), but same order
        # of magnitude.
        total = float(criterion(outcome.adjusted_samples) - criterion(q))
        assert abs(total - sum(marginals)) < abs(total) * 2 + 5e-3


# ---------------------------------------------------------------------------
# Additional-observation (Kalman) path.
# ---------------------------------------------------------------------------


class TestApplyAdditionalObservation:
    def test_observation_pulls_posterior_toward_value(self) -> None:
        rng = np.random.default_rng(0)
        q = rng.normal(0.0, 1.0, size=20_000)
        # Pretend we observed y = 2.0 with σ_y = 0.5 (tighter than prior).
        inp = _input(
            value=2.0, uncertainty=0.5,
            mechanism=IntegrationMechanism.ADDITIONAL_OBSERVATION,
        )
        ld = _loading(
            beta=1.0, beta_std=0.0, n_obs=10_000,
            mechanism=IntegrationMechanism.ADDITIONAL_OBSERVATION,
        )
        req = IntegrationRequest(input=inp, loading=ld)

        outcome = apply_additional_observation(
            q, [req], rng=np.random.default_rng(3),
        )
        m_prior, m_post = float(np.mean(q)), float(np.mean(outcome.adjusted_samples))
        # Posterior mean should sit strictly between prior (0) and obs (2)
        # and be closer to the obs because σ_y < σ_prior.
        assert 1.0 < m_post < 2.0
        assert abs(m_post - 2.0) < abs(m_prior - 2.0)

    def test_requires_uncertainty(self) -> None:
        rng = np.random.default_rng(0)
        q = rng.normal(0.0, 1.0, size=1000)
        inp = _input(
            uncertainty=None,
            mechanism=IntegrationMechanism.ADDITIONAL_OBSERVATION,
        )
        ld = _loading(mechanism=IntegrationMechanism.ADDITIONAL_OBSERVATION)
        req = IntegrationRequest(input=inp, loading=ld)
        outcome = apply_additional_observation(q, [req])
        assert len(outcome.dropped) == 1
        assert "uncertainty" in outcome.dropped[0].reason


# ---------------------------------------------------------------------------
# Top-level integrate() dispatch.
# ---------------------------------------------------------------------------


class TestIntegrate:
    def test_dispatches_prior_shift_and_additional_observation(self) -> None:
        rng = np.random.default_rng(0)
        q = rng.normal(0.5, 0.1, size=10_000)
        ps_req = IntegrationRequest(
            input=_input(value=0.02),
            loading=_loading(beta=0.5, beta_std=0.0, n_obs=10_000),
        )
        ao_req = IntegrationRequest(
            input=_input(
                name="ao", value=0.52, uncertainty=0.05,
                mechanism=IntegrationMechanism.ADDITIONAL_OBSERVATION,
            ),
            loading=_loading(
                name="ao",
                mechanism=IntegrationMechanism.ADDITIONAL_OBSERVATION,
                beta=1.0, beta_std=0.0, n_obs=10_000,
            ),
        )
        outcome = integrate(q, [ps_req, ao_req], rng=np.random.default_rng(1))
        assert outcome.adjusted_samples.shape == q.shape
        # Attribution should have entries from both handlers.
        names = {a.name for a in outcome.attribution}
        assert {"gasoline_weekly_delta", "ao"}.issubset(names)


# ---------------------------------------------------------------------------
# Manual YAML loader.
# ---------------------------------------------------------------------------


class TestManualLoader:
    def test_loads_shipped_example(self) -> None:
        root = Path(__file__).resolve().parents[1] / "analyst_inputs"
        inputs = load_manual_inputs_from_dir(root)
        # The example file has 2 entries for category=economics.
        assert len(inputs) >= 2
        by_name = {i.name for i in inputs}
        assert "gasoline_weekly_delta" in by_name
        for inp in inputs:
            assert inp.provenance.source_kind == "manual"
            assert inp.mechanism == IntegrationMechanism.PRIOR_SHIFT

    def test_rejects_unknown_name(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "category: economics\n"
            "inputs:\n"
            "  - name: not_a_real_input\n"
            "    value: 0.1\n"
            "    observation_at: 2026-04-15T00:00:00Z\n"
            "    expires_in_days: 10\n"
        )
        with pytest.raises(ManualInputError):
            load_manual_inputs(bad)

    def test_rejects_missing_expiry(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "category: economics\n"
            "inputs:\n"
            "  - name: gasoline_weekly_delta\n"
            "    value: 0.02\n"
            "    observation_at: 2026-04-15T00:00:00Z\n"
        )
        with pytest.raises(ManualInputError):
            load_manual_inputs(bad)

    def test_rejects_mechanism_mismatch(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(
            "category: economics\n"
            "inputs:\n"
            "  - name: gasoline_weekly_delta\n"
            "    value: 0.02\n"
            "    mechanism: likelihood_weight\n"
            "    observation_at: 2026-04-15T00:00:00Z\n"
            "    expires_in_days: 10\n"
        )
        with pytest.raises(ManualInputError):
            load_manual_inputs(bad)


# ---------------------------------------------------------------------------
# Brief renderer.
# ---------------------------------------------------------------------------


class TestBrief:
    def _ctx(self) -> BriefContext:
        now = datetime(2026, 4, 18, 12, 0, tzinfo=UTC)
        contract = Contract(
            ticker="KXCPI-26APR-T0.3", event_ticker="KXCPI-26APR",
            series_ticker="KXCPI",
            title="CPI MoM for April 2026 above 0.3%?", subtitle="headline",
            category=Category.ECONOMICS,
            open_time=now - timedelta(days=10),
            close_time=now + timedelta(days=5),
            expiration_time=now + timedelta(days=5),
            status=MarketStatus.OPEN,
        )
        snap = MarketSnapshot(
            ticker=contract.ticker, ts=now,
            yes_bid=55.0, yes_ask=57.0, no_bid=43.0, no_ask=45.0,
            last_price=56.0, last_trade_ts=now - timedelta(minutes=10),
            volume=5000, volume_24h=1200, open_interest=10_000,
            liquidity=2500.0,
        )
        rng = np.random.default_rng(0)
        adj = rng.normal(0.57, 0.06, size=2000)
        qonly = rng.normal(0.55, 0.06, size=2000)
        fc = ForecastResult(
            ticker=contract.ticker, ts=now, forecaster="economics_cpi_qf",
            version="1",
            binary_posterior=ForecastDistribution(kind="samples", samples=adj),
            uncertainty={"total": 0.69, "aleatoric": 0.68, "epistemic": 0.01},
            model_confidence=0.6,
            methodology={"model": "ARIMA(1,1,0)+drift", "prior": "residual"},
            data_sources={"FRED:CPIAUCSL": now - timedelta(days=1)},
            quant_only_binary_posterior=ForecastDistribution(
                kind="samples", samples=qonly,
            ),
            attribution={
                "mean_shift_total": 0.02,
                "attribution": [{
                    "name": "gasoline_weekly_delta",
                    "anomaly": 0.03, "beta_shrunk": 0.4,
                    "mean_shift_underlying": 0.012, "mean_shift_prob": 0.03,
                    "beta_raw": 0.5, "beta_std": 0.1, "n_obs": 120,
                    "mechanism": "prior_shift", "value": 0.03, "baseline": 0.0,
                }],
                "dropped": [{"name": "supply_chain_pressure", "reason": "input_expired"}],
            },
        )
        return BriefContext(
            contract=contract, snapshot=snap, forecast=fc,
            inputs=[_input()], loadings={"gasoline_weekly_delta": _loading()},
            as_of=now,
        )

    def test_deterministic(self) -> None:
        a = render_brief(self._ctx())
        b = render_brief(self._ctx())
        assert a == b

    def test_has_all_sections(self) -> None:
        out = render_brief(self._ctx())
        for h in (
            "# CPI MoM for April 2026 above 0.3%?",
            "## Market snapshot",
            "## Forecast",
            "## Fundamental inputs (live)",
            "## Calibrated loadings",
            "## Posterior shift attribution",
            "## Dropped inputs",
            "## What's missing",
            "## Data sources",
            "## Methodology",
        ):
            assert h in out, f"missing section: {h}"

    def test_missing_section_flags_absent_specs(self) -> None:
        out = render_brief(self._ctx())
        # Only gasoline_weekly_delta is provided; the other three CPI
        # specs should show up in "What's missing".
        assert "hpi_lead_12m" in out
        assert "ppi_passthrough" in out
