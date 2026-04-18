"""Economics forecaster — CPI release month (MoM and YoY variants).

DGP: the Consumer Price Index (CPIAUCSL on FRED). We model monthly
log-CPI as ARIMA(1,1,0) with drift — a standard workhorse for CPI
short-horizon nowcasts. Gaussian-residuals on the log-differences;
σ is fit from observed residuals. We simulate N trajectories forward
from the latest observation to the target release month, then compute
P(YES) empirically from the draws.

Two Kalshi series, different underlying transforms — we used to conflate
them and got a ~100% P(YES) on KXCPI markets that should have been
priced at ~5% because we were asking "YoY > 1%" (nearly certain given
~2-3% trend inflation) when the market actually asks "month-over-month
CPI change > 1%" (very rare; MoM typically 0.1-0.4%).

    KXCPI      → "CPI increases by more than X% in {month}" → **MoM**
    KXCPIYOY   → "rate of CPI inflation above X% for year ending {month}" → **YoY**

The YES direction comes from Kalshi's structured ``strike_type`` field
(via market.strikes.parse_yes_criterion), not from a hardcoded
comparator. If the field is missing or a shape we don't handle, abstain.

If FRED isn't configured, or the ticker isn't parseable, we abstain
with a reason. No synthetic data, no fake precision.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone

import numpy as np

from kalshi_edge.data_sources.fred import FredClient, FredSeries
from kalshi_edge.forecast import ForecastDistribution, ForecastResult
from kalshi_edge.forecasters.base import Forecaster
from kalshi_edge.market.strikes import YesCriterion, parse_yes_criterion
from kalshi_edge.types import Category, Contract, MarketSnapshot


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])}

_EVENT_MONTH_RE = re.compile(r"-(\d{2})([A-Z]{3})(?:-|$)")


@dataclass(frozen=True, slots=True)
class CPIContract:
    target_month: date      # first-of-month in the release month
    transform: str          # "mom" (month-over-month) or "yoy" (year-over-year)
    criterion: YesCriterion # direction + bounds from Kalshi's strike_type


def parse_cpi_contract(contract: Contract) -> CPIContract | None:
    ser = contract.series_ticker.upper().strip()
    if ser == "KXCPIYOY":
        transform = "yoy"
    elif ser == "KXCPI":
        transform = "mom"
    else:
        return None
    mm = _EVENT_MONTH_RE.search(contract.event_ticker.upper())
    if not mm:
        return None
    yy = int(mm.group(1))
    mon = _MONTHS.get(mm.group(2))
    if mon is None:
        return None
    target = date(2000 + yy, mon, 1)

    criterion = parse_yes_criterion(contract.raw or {})
    if criterion is None:
        return None
    return CPIContract(target_month=target, transform=transform, criterion=criterion)


class EconomicsCPIForecaster(Forecaster):
    """AR(1)-on-differences CPI-YoY forecaster. Abstains when FRED is unset
    or when we don't recognize the market shape."""

    name = "economics_cpi_ar1"
    version = "1"

    def __init__(self, client: FredClient | None = None, *,
                 n_sims: int = 10_000,
                 rng: np.random.Generator | None = None) -> None:
        self._client = client
        self.n_sims = n_sims
        self.rng = rng or np.random.default_rng(seed=0xCAFE)

    @property
    def category(self) -> Category:
        return Category.ECONOMICS

    def supports(self, contract: Contract) -> bool:
        if contract.category != Category.ECONOMICS:
            return False
        return parse_cpi_contract(contract) is not None

    def _ensure_client(self) -> FredClient | None:
        if self._client is not None:
            return self._client
        key = os.environ.get("FRED_API_KEY")
        if not key:
            return None
        self._client = FredClient(key)
        return self._client

    def _forecast_impl(
        self, contract: Contract, snapshot: MarketSnapshot | None,
    ) -> ForecastResult:
        cpi_contract = parse_cpi_contract(contract)
        if cpi_contract is None:
            return self._null(
                contract,
                "cpi_contract_unparseable (unknown series, month, or strike_type)",
            )

        client = self._ensure_client()
        if client is None:
            return self._null(
                contract,
                "FRED_API_KEY not set — economics forecaster needs CPI history",
            )

        try:
            series = client.observations("CPIAUCSL")
        except Exception as e:  # noqa: BLE001
            return self._null(contract, f"fred_fetch_failed: {e}")

        mask = ~np.isnan(series.values)
        if mask.sum() < 48:
            return self._null(contract, f"cpi_series_too_short: {int(mask.sum())}")

        dates = series.dates[mask]
        values = series.values[mask].astype(float)
        latest_obs_date = dates[-1]
        # Monthly steps needed to reach the target month (which resolves from
        # the CPI release FOR that month — we model the release value).
        steps = _months_between(latest_obs_date, cpi_contract.target_month)
        if steps <= 0:
            return self._null(contract, f"target_month_already_known: {cpi_contract.target_month}")
        if steps > 18:
            return self._null(contract, f"target_month_beyond_horizon: {steps}m")

        p_yes_samples, underlying_samples, fit = self._simulate(
            values=values, steps=steps,
            transform=cpi_contract.transform,
            criterion=cpi_contract.criterion,
        )

        binary = ForecastDistribution(kind="samples", samples=p_yes_samples)
        underlying = ForecastDistribution(kind="samples", samples=underlying_samples)

        return ForecastResult(
            ticker=contract.ticker,
            ts=datetime.now(timezone.utc),
            forecaster=self.name,
            version=self.version,
            binary_posterior=binary,
            underlying_posterior=underlying,
            model_confidence=_confidence_from_spread(p_yes_samples),
            methodology={
                "model": "ARIMA(1,1,0) on log-CPI with drift",
                "likelihood": "Gaussian residuals, σ fit from observed",
                "horizon_months": steps,
                "simulations": self.n_sims,
                "posterior_over_p": "Monte Carlo over simulated trajectories",
                "transform": cpi_contract.transform,
                "latest_obs": latest_obs_date.isoformat(),
                "fit": fit,
            },
            data_sources={"fred_cpiaucsl": _to_utc_midnight(latest_obs_date)},
            diagnostics={
                "p_yes_mean": float(np.mean(p_yes_samples)),
                "p_yes_std": float(np.std(p_yes_samples, ddof=1)),
                "underlying_mean_pct": float(np.mean(underlying_samples)),
                "underlying_std_pct": float(np.std(underlying_samples, ddof=1)),
                "yes_direction": cpi_contract.criterion.direction,
                "yes_low_pct": cpi_contract.criterion.low,
                "yes_high_pct": cpi_contract.criterion.high,
                "transform": cpi_contract.transform,
            },
        )

    def _simulate(self, *, values: np.ndarray, steps: int,
                  transform: str,
                  criterion: YesCriterion,
                  ) -> tuple[np.ndarray, np.ndarray, dict]:
        """ARIMA(1,1,0)-with-drift simulation on log-CPI.

        Returns (p_yes_per_boot, underlying_pct, fit) where
        ``underlying_pct`` is whichever transform the contract asks about
        (MoM% or YoY%). The binary posterior is a Bayesian bootstrap of
        the empirical tail probability under the criterion's direction.
        """
        log_cpi = np.log(values)
        diff = np.diff(log_cpi)                      # monthly log-returns
        # OLS AR(1) with intercept on diff.
        x = diff[:-1]
        y = diff[1:]
        X = np.column_stack([np.ones_like(x), x])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        mu, phi = float(beta[0]), float(beta[1])
        residuals = y - (mu + phi * x)
        sigma = float(np.std(residuals, ddof=2))

        rng = self.rng
        N = self.n_sims

        last = float(log_cpi[-1])
        last_diff = float(diff[-1])
        # Pre-seed the last 12 observed log-CPI values (for YoY denominator
        # when the 12-months-prior point pre-dates the simulation window).
        hist_tail = log_cpi[-12:].copy() if log_cpi.size >= 12 else np.full(12, last)

        shocks = rng.normal(0.0, sigma, size=(N, steps))
        traj = np.empty((N, steps), dtype=float)
        prev_diff = np.full(N, last_diff)
        prev_log = np.full(N, last)
        diffs_out = np.empty((N, steps), dtype=float)
        for t in range(steps):
            new_diff = mu + phi * prev_diff + shocks[:, t]
            prev_log = prev_log + new_diff
            traj[:, t] = prev_log
            diffs_out[:, t] = new_diff
            prev_diff = new_diff

        if transform == "mom":
            # Kalshi rounds to 1 decimal place for MoM. Simulation is
            # continuous; the 1dp rounding shifts the boundary by <0.05pp,
            # which is inside our σ, so we skip explicit quantization.
            underlying_pct = (np.exp(diffs_out[:, steps - 1]) - 1.0) * 100.0
        elif transform == "yoy":
            log_num = traj[:, steps - 1]
            if steps >= 12:
                log_denom = traj[:, steps - 1 - 12]
            else:
                log_denom = np.full(
                    N, hist_tail[max(0, len(hist_tail) - 12 + steps - 1)],
                )
            underlying_pct = (np.exp(log_num - log_denom) - 1.0) * 100.0
        else:
            raise ValueError(f"unknown transform: {transform}")

        # Bayesian bootstrap over the simulation draws for epistemic-uncertainty
        # propagation. Apply the YES criterion element-wise, in the direction
        # Kalshi specified.
        B = 2000
        idx = rng.integers(0, N, size=(B, N))
        boot = underlying_pct[idx]                    # (B, N)

        if criterion.direction == "above":
            assert criterion.low is not None
            hit = boot > criterion.low
        elif criterion.direction == "below":
            assert criterion.high is not None
            hit = boot < criterion.high
        else:
            assert criterion.direction == "between"
            assert criterion.low is not None and criterion.high is not None
            hit = (boot >= criterion.low) & (boot <= criterion.high)

        p_yes_per_boot = np.mean(hit, axis=1)
        p_yes_per_boot = np.clip(p_yes_per_boot, 1e-6, 1.0 - 1e-6)

        fit = {"mu": mu, "phi": phi, "sigma": sigma, "n_obs": int(diff.size)}
        return p_yes_per_boot, underlying_pct, fit


def _months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def _to_utc_midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _confidence_from_spread(p_yes_samples: np.ndarray) -> float:
    s = float(np.std(p_yes_samples))
    return float(max(0.0, min(1.0, np.exp(-4.0 * s))))
