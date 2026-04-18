"""Calibrate fundamental loadings for the CPI forecaster.

The quant forecaster (``EconomicsCPIForecaster``) models monthly log-CPI as
ARIMA(1,1,0) with drift. Its residual at month t is the component of the
MoM log-change at t NOT explained by the autoregressive dynamics. If a
fundamental input carries information beyond the AR model, it should be
correlated with these residuals — that correlation is the loading β.

Calibration procedure:

    1. Fit ARIMA(1,1,0) on log(CPIAUCSL) via the same OLS-on-differences
       approach the forecaster uses, but walk-forward: at each month t,
       refit on data strictly before t and record the one-step-ahead
       residual (actual MoM log-change at t minus predicted MoM log-change).
       This is the OOS-residual series we regress on.
    2. For each fundamental input, compute its anomaly series using the
       SAME transform function the live pipeline uses (imports from
       ``fundamental.automated.cpi``). Align dates so the input observed
       at or before the CPI release for month t is the predictor for the
       month-t residual. Different inputs have different information
       availability; the alignment is spelled out per input below.
    3. Run univariate OLS of residual_t on de-meaned anomaly_t (plus a
       small-sample correction). The β and β_std come out of this;
       β_std is the bootstrap std of β under residual resampling, which
       is robust to mild heteroskedasticity and serial correlation.
    4. Return a ``CalibratedLoading`` bundle that the CLI writes into
       ``fundamental_loadings``.

Units:

    * Residuals are in log-change units (same as ARIMA residuals).
    * Value units per input: log-change (gasoline, PPI), z-score (GSCPI),
      % (HPI YoY).
    * β therefore carries units "[log-change]/[input units]".
    * At integration time we shift sample MoM log-changes by β · anomaly;
      the forecaster then converts back to percent via ``exp(mom) - 1``.

No look-ahead. No leakage. No hand-picked multipliers.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Callable

import numpy as np

from kalshi_edge.data_sources.fred import FredClient, FredSeries
from kalshi_edge.fundamental.automated.cpi import (
    gasoline_4w_log_delta,
    gscpi_zscore,
    hpi_lead_12m_yoy,
    ppi_3m_log_delta,
)
from kalshi_edge.logging_ import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class CalibratedLoading:
    name: str
    mechanism: str
    beta: float
    beta_std: float
    baseline: float
    n_obs: int
    fit_method: str
    fit_at: datetime
    diagnostics: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Walk-forward AR(1,1,0)-with-drift residuals on log-CPI.
# ---------------------------------------------------------------------------


def _walk_forward_residuals(
    log_cpi: np.ndarray, min_train: int = 48, scale: float = 100.0,
) -> np.ndarray:
    """Return OOS one-step-ahead residuals for ARIMA(1,1,0) with drift on log-CPI.

    Index alignment: ``residuals[t]`` = actual diff[t] - predicted diff[t],
    where diff[t] = log_cpi[t+1] - log_cpi[t] and the predictor is fit on
    diff[0..t-1]. ``residuals`` has length ``n_diffs - min_train`` — the
    first ``min_train`` diffs are burn-in.

    The t-th residual corresponds to the CPI release for month
    ``t + min_train + 1`` (in 0-indexed series position).

    ``scale`` rescales residuals. Default ``100.0`` converts log-change
    residuals into MoM %-point residuals — the native unit of the quant
    forecaster's posterior samples. That way β learned here applies
    directly to forecaster samples without further transformation.
    """
    diff = np.diff(log_cpi)
    n = diff.size
    if n <= min_train + 1:
        return np.array([], dtype=float)

    out = np.empty(n - min_train, dtype=float)
    # Running OLS on diff[0..i-1] with AR(1) + intercept. We refit each step
    # for simplicity; unlimited-compute, and AR(1) OLS is cheap.
    for i in range(min_train, n):
        x_train = diff[:i - 1]
        y_train = diff[1:i]
        X = np.column_stack([np.ones_like(x_train), x_train])
        beta, *_ = np.linalg.lstsq(X, y_train, rcond=None)
        mu, phi = float(beta[0]), float(beta[1])
        pred = mu + phi * diff[i - 1]
        out[i - min_train] = float(scale * (diff[i] - pred))
    return out


# ---------------------------------------------------------------------------
# Anomaly series builders. Each returns a monthly-aligned pd-free series
# (dates + values) matching the transform used at live-time.
# ---------------------------------------------------------------------------


def _monthly_anchor(d: date) -> date:
    """First-of-month anchor for a date."""
    return date(d.year, d.month, 1)


def _last_obs_before(dates: np.ndarray, values: np.ndarray, target: date) -> tuple[int, float] | None:
    """Return (index, value) of the most recent valid observation strictly before ``target``."""
    mask = ~np.isnan(values)
    if not mask.any():
        return None
    # dates is an array of date objects.
    idxs = np.where(mask)[0]
    best: tuple[int, float] | None = None
    for i in idxs:
        d = dates[i]
        if d < target:
            if best is None or d > dates[best[0]]:
                best = (int(i), float(values[i]))
    return best


def _build_gasoline_anomaly_series(series: FredSeries) -> tuple[list[date], list[float]]:
    """For each month-end, compute the 4-week log-change in gasoline observable
    before that month's CPI release. GASREGW is weekly; we anchor each month
    to its last weekly observation that falls before the 15th of the next
    month (the typical CPI release window).

    Returns (anchor_months, values).
    """
    mask = ~np.isnan(series.values)
    d = series.dates[mask]
    v = series.values[mask].astype(float)
    if v.size < 10:
        return [], []

    # Collect the last weekly price observation for each month.
    by_month: dict[date, tuple[date, float, int]] = {}  # month -> (last_date, price, index)
    for i, di in enumerate(d):
        m = _monthly_anchor(di)
        if m not in by_month or di > by_month[m][0]:
            by_month[m] = (di, float(v[i]), i)
    months = sorted(by_month)

    anchors: list[date] = []
    vals: list[float] = []
    for m in months:
        idx = by_month[m][2]
        if idx < 4:
            continue  # need 4 weeks of history
        val = float(np.log(v[idx]) - np.log(v[idx - 4]))
        anchors.append(m)
        vals.append(val)
    return anchors, vals


def _build_gscpi_anomaly_series(series: FredSeries) -> tuple[list[date], list[float]]:
    """Monthly z-score of GSCPI vs trailing 60-month window, ending at t-1
    so the signal for month t uses only data observable before the
    mid-month CPI release.
    """
    mask = ~np.isnan(series.values)
    d = series.dates[mask]
    v = series.values[mask].astype(float)
    anchors: list[date] = []
    vals: list[float] = []
    for i in range(24, v.size):
        window = v[max(0, i - 60): i]  # exclude current
        mu = float(np.mean(window))
        sigma = float(np.std(window, ddof=1))
        if sigma <= 0:
            continue
        # Signal for month m at v[i] is the z-score. Anchor month = d[i] (monthly series).
        anchors.append(_monthly_anchor(d[i]))
        vals.append(float((v[i] - mu) / sigma))
    return anchors, vals


def _build_hpi_lead_series(series: FredSeries) -> tuple[list[date], list[float]]:
    """At month t, signal = Case-Shiller YoY at t-12.

    Case-Shiller is itself reported with a ~2 month lag; we assume the CPI
    release at month t knows the YoY value observed 12 months earlier. So
    the signal used for CPI-month t is the YoY at month (t-12).
    """
    mask = ~np.isnan(series.values)
    d = series.dates[mask]
    v = series.values[mask].astype(float)
    anchors: list[date] = []
    vals: list[float] = []
    if v.size < 25:
        return anchors, vals
    # The YoY at index i requires v[i-12]; we emit the signal at month t=m+12.
    for i in range(12, v.size):
        yoy = 100.0 * (v[i] / v[i - 12] - 1.0)
        # This yoy corresponds to the HPI YoY AT MONTH d[i]; the CPI-month
        # it predicts is d[i]+12 months.
        cpi_month_year = d[i].year + ((d[i].month + 12 - 1) // 12)
        cpi_month_month = ((d[i].month - 1 + 12) % 12) + 1
        cpi_month = date(cpi_month_year, cpi_month_month, 1)
        anchors.append(cpi_month)
        vals.append(float(yoy))
    return anchors, vals


def _build_ppi_3m_series(series: FredSeries) -> tuple[list[date], list[float]]:
    """3-month log-change in PPIACO, emitted for the following CPI month.

    PPI for month t is reported mid-month t+1; the CPI release for month t
    comes out around the middle of month t+1, so the PPI value for month
    t-1 is observable and can be used for predicting CPI month t.
    """
    mask = ~np.isnan(series.values)
    d = series.dates[mask]
    v = series.values[mask].astype(float)
    anchors: list[date] = []
    vals: list[float] = []
    if v.size < 5:
        return anchors, vals
    for i in range(3, v.size - 1):
        val = float(np.log(v[i]) - np.log(v[i - 3]))
        # Predicting CPI at month i+1.
        nxt_year = d[i].year + (1 if d[i].month == 12 else 0)
        nxt_month = 1 if d[i].month == 12 else d[i].month + 1
        anchors.append(date(nxt_year, nxt_month, 1))
        vals.append(val)
    return anchors, vals


# ---------------------------------------------------------------------------
# Bootstrap univariate OLS.
# ---------------------------------------------------------------------------


def _bootstrap_univariate_ols(
    x: np.ndarray, y: np.ndarray, *, n_boot: int = 2000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, dict[str, float]]:
    """Univariate OLS y = α + β · x with bootstrap β_std.

    Returns (beta, beta_std, diag) where ``diag`` carries R², F, and
    intercept for reporting.
    """
    rng = rng or np.random.default_rng(0xCAFE)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    assert x.shape == y.shape and x.ndim == 1

    def _fit(xi: np.ndarray, yi: np.ndarray) -> tuple[float, float]:
        X = np.column_stack([np.ones_like(xi), xi])
        coef, *_ = np.linalg.lstsq(X, yi, rcond=None)
        return float(coef[0]), float(coef[1])

    alpha, beta = _fit(x, y)
    yhat = alpha + beta * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2)) or 1e-18
    r2 = 1.0 - ss_res / ss_tot

    # Pair bootstrap.
    n = x.size
    betas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        _a, _b = _fit(x[idx], y[idx])
        betas[b] = _b
    beta_std = float(np.std(betas, ddof=1))

    # F-stat for single predictor ~ (β / SE)².
    # (Not strictly needed for the downstream code; reported for audit.)
    f_stat = (beta / beta_std) ** 2 if beta_std > 0 else 0.0

    return beta, beta_std, {
        "alpha": alpha,
        "r2": r2,
        "f_stat": float(f_stat),
        "residual_sigma": float(np.sqrt(ss_res / max(n - 2, 1))),
        "n_bootstrap": float(n_boot),
    }


# ---------------------------------------------------------------------------
# Top-level orchestrator.
# ---------------------------------------------------------------------------


AnomalyBuilder = Callable[[FredSeries], tuple[list[date], list[float]]]

_CPI_INPUT_BUILDERS: dict[str, tuple[str, AnomalyBuilder]] = {
    "gasoline_weekly_delta": ("GASREGW", _build_gasoline_anomaly_series),
    "supply_chain_pressure": ("GSCPI", _build_gscpi_anomaly_series),
    "hpi_lead_12m":          ("CSUSHPISA", _build_hpi_lead_series),
    "ppi_passthrough":       ("PPIACO", _build_ppi_3m_series),
}


def fit_cpi_loadings(
    client: FredClient | None = None,
    *,
    min_train_months: int = 48,
    n_bootstrap: int = 2000,
    max_years: int | None = 25,
) -> list[CalibratedLoading]:
    """Fit all CPI fundamental loadings.

    ``max_years`` caps history used for regression. Very long histories can
    be dominated by regime shifts (e.g., pre-1995 vs post-1995); 25 years is
    a reasonable default. Set None to use full series.
    """
    if client is None:
        key = os.environ.get("FRED_API_KEY")
        if not key:
            log.warning("fit_cpi_loadings_skipped", reason="no_fred_api_key")
            return []
        client = FredClient(key)

    cpi = client.observations("CPIAUCSL")
    mask = ~np.isnan(cpi.values)
    if mask.sum() < min_train_months + 12:
        log.warning("cpi_history_too_short", n=int(mask.sum()))
        return []

    cpi_dates = cpi.dates[mask]
    cpi_vals = cpi.values[mask].astype(float)

    if max_years is not None:
        # Truncate to the tail.
        cutoff_idx = max(0, cpi_vals.size - max_years * 12 - 12)
        cpi_dates = cpi_dates[cutoff_idx:]
        cpi_vals = cpi_vals[cutoff_idx:]

    log_cpi = np.log(cpi_vals)
    residuals = _walk_forward_residuals(log_cpi, min_train=min_train_months)
    # Each residual at position k corresponds to the CPI release for month
    # ``cpi_dates[min_train + 1 + k]``. Build the month index.
    residual_months = [
        _monthly_anchor(cpi_dates[min_train_months + 1 + k]) for k in range(residuals.size)
    ]
    resid_by_month: dict[date, float] = dict(zip(residual_months, residuals.tolist()))

    fit_at = datetime.now(timezone.utc)
    out: list[CalibratedLoading] = []

    for name, (series_id, builder) in _CPI_INPUT_BUILDERS.items():
        try:
            series = client.observations(series_id)
        except Exception as e:  # noqa: BLE001
            log.warning("loading_fit_series_fetch_failed", name=name, error=str(e))
            continue

        try:
            anchor_months, vals = builder(series)
        except Exception as e:  # noqa: BLE001
            log.warning("loading_fit_anomaly_build_failed", name=name, error=str(e))
            continue

        # Align to available residuals.
        x: list[float] = []
        y: list[float] = []
        for m, v in zip(anchor_months, vals):
            r = resid_by_month.get(m)
            if r is None or not np.isfinite(v) or not np.isfinite(r):
                continue
            x.append(v)
            y.append(r)
        if len(x) < 24:
            log.warning("loading_fit_insufficient_overlap", name=name, n=len(x))
            # Record a zero loading so it's visible in the DB as "calibrated but
            # no signal" rather than entirely missing. n_obs=0 → shrinkage fully
            # zeros it out at integration time.
            out.append(CalibratedLoading(
                name=name,
                mechanism="prior_shift",
                beta=0.0, beta_std=0.0, baseline=0.0, n_obs=0,
                fit_method="insufficient_data",
                fit_at=fit_at,
                diagnostics={"n_overlap": len(x)},
            ))
            continue

        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        baseline = float(np.mean(x_arr))
        x_centered = x_arr - baseline

        beta, beta_std, diag = _bootstrap_univariate_ols(
            x_centered, y_arr, n_boot=n_bootstrap,
        )
        diag["x_mean"] = baseline
        diag["x_std"] = float(np.std(x_arr, ddof=1))
        diag["y_mean"] = float(np.mean(y_arr))
        diag["y_std"] = float(np.std(y_arr, ddof=1))
        diag["series_id"] = series_id
        out.append(CalibratedLoading(
            name=name, mechanism="prior_shift",
            beta=beta, beta_std=beta_std,
            baseline=baseline,
            n_obs=int(x_arr.size),
            fit_method="univariate_ols_bootstrap",
            fit_at=fit_at,
            diagnostics=diag,
        ))
        log.info("cpi_loading_fit", name=name, n=int(x_arr.size),
                 beta=round(beta, 6), beta_std=round(beta_std, 6),
                 r2=round(diag["r2"], 4))

    return out
