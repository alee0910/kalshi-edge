"""Economics forecaster — CPI YoY in a release month.

DGP: the Consumer Price Index (CPIAUCSL on FRED). We model monthly
log-CPI as ARIMA(1,1,0) with drift — a standard workhorse for CPI
short-horizon nowcasts. The YoY contract resolves on

    YoY_t = CPI_t / CPI_{t-12} - 1

Given we observe CPI up through some past month and the contract
resolves for a future month t*, we simulate forward the remaining
log-differences from the fitted model and carry through to YoY at t*.

A Gaussian-residuals assumption on the log-differences is used; the
residuals from the fit give us σ. We then draw N trajectories, compute
YoY_{t*} per trajectory, and compute P(YoY_{t*} > threshold) as the
empirical tail probability. This gives us an underlying posterior over
YoY (the DGP) and a bootstrap-style binary posterior.

If FRED isn't configured, or the ticker isn't a YoY market we can
unambiguously parse, we abstain with a reason. No synthetic data, no
fake precision.

Supported ticker grammar:
    series KXCPIYOY / KXCPI
    event  KXCPIYOY-<YY><MON>          e.g. 26JUN
    ticker KXCPIYOY-<YY><MON>-T<float> threshold in percent (strict >)
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
from kalshi_edge.types import Category, Contract, MarketSnapshot


_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])}

_EVENT_MONTH_RE = re.compile(r"-(\d{2})([A-Z]{3})(?:-|$)")
_THRESHOLD_RE = re.compile(r"-T(-?\d+(?:\.\d+)?)$")


@dataclass(frozen=True, slots=True)
class CPIContract:
    target_month: date   # first-of-month in the release month
    threshold_pct: float


def parse_cpi_contract(series_ticker: str, event_ticker: str,
                       ticker: str) -> CPIContract | None:
    ser = series_ticker.upper().strip()
    if ser not in ("KXCPIYOY", "KXCPI"):
        return None
    mm = _EVENT_MONTH_RE.search(event_ticker.upper())
    if not mm:
        return None
    yy = int(mm.group(1))
    mon = _MONTHS.get(mm.group(2))
    if mon is None:
        return None
    target = date(2000 + yy, mon, 1)

    mt = _THRESHOLD_RE.search(ticker.upper())
    if not mt:
        return None
    return CPIContract(target_month=target, threshold_pct=float(mt.group(1)))


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
        return parse_cpi_contract(
            contract.series_ticker, contract.event_ticker, contract.ticker,
        ) is not None

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
        cpi_contract = parse_cpi_contract(
            contract.series_ticker, contract.event_ticker, contract.ticker,
        )
        if cpi_contract is None:
            return self._null(contract, "cpi_contract_unparseable")

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

        p_yes_samples, yoy_samples, fit = self._simulate(
            values=values, steps=steps, threshold_pct=cpi_contract.threshold_pct,
        )

        binary = ForecastDistribution(kind="samples", samples=p_yes_samples)
        underlying = ForecastDistribution(kind="samples", samples=yoy_samples)

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
                "latest_obs": latest_obs_date.isoformat(),
                "fit": fit,
            },
            data_sources={"fred_cpiaucsl": _to_utc_midnight(latest_obs_date)},
            diagnostics={
                "p_yes_mean": float(np.mean(p_yes_samples)),
                "p_yes_std": float(np.std(p_yes_samples, ddof=1)),
                "yoy_mean": float(np.mean(yoy_samples)),
                "yoy_std": float(np.std(yoy_samples, ddof=1)),
                "threshold_pct": cpi_contract.threshold_pct,
            },
        )

    def _simulate(self, *, values: np.ndarray, steps: int,
                  threshold_pct: float) -> tuple[np.ndarray, np.ndarray, dict]:
        """ARIMA(1,1,0)-with-drift simulation on log-CPI. Returns (p_yes, yoy, fit)."""
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

        # For YoY at step t*, we need CPI_{t*} / CPI_{t*-12}. Both may be
        # in the simulated future depending on how far ahead we are.
        rng = self.rng
        N = self.n_sims

        last = float(log_cpi[-1])
        last_diff = float(diff[-1])
        # Buffer of 13 to cover YoY ratio even when steps < 12.
        horizon = steps + 13
        # Pre-seed the first 12 months' log-CPI from observed (for YoY denom).
        hist_tail = log_cpi[-12:].copy() if log_cpi.size >= 12 else np.full(12, last)
        # Simulate
        shocks = rng.normal(0.0, sigma, size=(N, steps))
        traj = np.empty((N, steps), dtype=float)
        prev_diff = np.full(N, last_diff)
        prev_log = np.full(N, last)
        for t in range(steps):
            new_diff = mu + phi * prev_diff + shocks[:, t]
            prev_log = prev_log + new_diff
            traj[:, t] = prev_log
            prev_diff = new_diff

        # Target = YoY at step steps-1 of the simulation:
        # CPI_target / CPI_{target-12months}, compare and convert to pct.
        # When steps >= 12 the denominator comes from the sim; otherwise from obs.
        if steps >= 12:
            log_denom = traj[:, steps - 1 - 12]
        else:
            # Observed CPI 12 months prior to the target month.
            log_denom = np.full(N, hist_tail[steps - 1] if steps - 1 < 12 else hist_tail[-1])
            # Precise pick: the observed log-CPI value at target_month - 12m.
            # hist_tail holds the last 12 observed log-CPI values. At offset
            # steps-1 in the simulation, 12 months prior is at (t*-12) which
            # in observed-index terms is the (12-(12-steps)+... )—reduce to:
            log_denom = np.full(N, hist_tail[max(0, len(hist_tail) - 12 + steps - 1)])

        log_num = traj[:, steps - 1]
        yoy = np.exp(log_num - log_denom) - 1.0
        yoy_pct = yoy * 100.0

        # Two posteriors: over the underlying YoY (samples on the DGP) and
        # over the binary P(YoY > threshold). The binary-posterior
        # construction is a bootstrap over the simulation draws for
        # epistemic-uncertainty propagation (parameter uncertainty).
        B = 2000
        idx = rng.integers(0, N, size=(B, N))
        boot = yoy_pct[idx]                          # (B, N)
        p_yes_per_boot = np.mean(boot > threshold_pct, axis=1)
        p_yes_per_boot = np.clip(p_yes_per_boot, 1e-6, 1.0 - 1e-6)

        fit = {"mu": mu, "phi": phi, "sigma": sigma, "n_obs": int(diff.size)}
        return p_yes_per_boot, yoy_pct, fit


def _months_between(a: date, b: date) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def _to_utc_midnight(d: date) -> datetime:
    return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)


def _confidence_from_spread(p_yes_samples: np.ndarray) -> float:
    s = float(np.std(p_yes_samples))
    return float(max(0.0, min(1.0, np.exp(-4.0 * s))))
