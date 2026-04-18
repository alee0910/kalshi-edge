"""CPI fundamental input pulls.

Four inputs, all free via FRED:

    gasoline_weekly_delta   — log-change in GASREGW over trailing 4 weeks
    supply_chain_pressure   — GSCPI z-score vs trailing 5-year rolling mean
    hpi_lead_12m            — Case-Shiller YoY %-change 12 months ago
    ppi_passthrough         — 3-month log-change in PPIACO

Each returns a ``FundamentalInput`` with:

    * value = the anomaly in the form the loading calibration consumed
    * uncertainty = 1σ estimate where we have one (measurement noise for
      gasoline via weekly-reporting variance; None for the rest)
    * provenance = "FRED:<series>", fetched_at = now, observation_at =
      the latest observation date of the input series.

If FRED is unavailable we return an empty list with a log note. The
forecaster treats missing inputs the same as expired ones (skipped).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import numpy as np

from kalshi_edge.data_sources.fred import FredClient, FredSeries
from kalshi_edge.fundamental.schemas.base import (
    FundamentalInput,
    InputProvenance,
    IntegrationMechanism,
)
from kalshi_edge.logging_ import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Anomaly transforms — PUBLIC so the calibration module reuses them to
# ensure train-time and live-time inputs are identical. This is the single
# most important invariant of the pipeline.
# ---------------------------------------------------------------------------


def gasoline_4w_log_delta(dates: np.ndarray, values: np.ndarray) -> tuple[datetime, float, float | None]:
    """Trailing 4-week log-change in gasoline retail price.

    Returns (observation_at, value, uncertainty). Uncertainty is the bootstrap
    std of the mean log-return over the trailing 4 weeks — a proxy for
    measurement noise in the weekly reporting.

    If we have fewer than 5 non-NaN weekly observations, returns (latest_date, 0.0, None).
    """
    mask = ~np.isnan(values)
    if mask.sum() < 5:
        latest = _latest_date(dates, values)
        return _to_dt(latest), 0.0, None

    d = dates[mask]
    v = values[mask].astype(float)
    latest_dt = _to_dt(d[-1])
    # 4-week log-change = log(v[-1]) - log(v[-5]) when data is strictly weekly.
    # Guard against short series.
    if v.size < 5:
        return latest_dt, 0.0, None
    val = float(np.log(v[-1]) - np.log(v[-5]))
    # Weekly returns → approximate std via trailing 12 weeks of 4-week-log-returns.
    if v.size >= 16:
        rets = np.log(v[-12:]) - np.log(v[-16:-4])
        unc = float(np.std(rets, ddof=1))
    else:
        unc = None
    return latest_dt, val, unc


def gscpi_zscore(dates: np.ndarray, values: np.ndarray) -> tuple[datetime, float, float | None]:
    """GSCPI z-score vs trailing 60-month rolling mean/std.

    The raw GSCPI is already z-scored against a pre-2012 climatology by
    the NY Fed; we re-z-score against a rolling 60-month window to adapt
    to regime drift. Monthly frequency.
    """
    mask = ~np.isnan(values)
    if mask.sum() < 24:
        latest = _latest_date(dates, values)
        return _to_dt(latest), 0.0, None
    d = dates[mask]
    v = values[mask].astype(float)
    window = min(60, v.size - 1)
    tail = v[-window - 1:-1] if v.size > window else v[:-1]
    mu = float(np.mean(tail))
    sigma = float(np.std(tail, ddof=1))
    latest_dt = _to_dt(d[-1])
    if sigma <= 0:
        return latest_dt, 0.0, None
    z = float((v[-1] - mu) / sigma)
    return latest_dt, z, 1.0 / np.sqrt(window)  # rough SE of the mean for the baseline


def hpi_lead_12m_yoy(dates: np.ndarray, values: np.ndarray) -> tuple[datetime, float, float | None]:
    """Case-Shiller YoY %-change from 12 months ago.

    The signal at time t is the YoY growth observed at t-12. That's the
    lead alignment: home-price growth a year ago predicts today's
    shelter inflation.
    """
    mask = ~np.isnan(values)
    if mask.sum() < 25:
        latest = _latest_date(dates, values)
        return _to_dt(latest), 0.0, None
    d = dates[mask]
    v = values[mask].astype(float)
    # We want: YoY at index i = 100 * (v[i] / v[i-12] - 1); emit the value at i = -13
    if v.size < 25:
        return _to_dt(d[-1]), 0.0, None
    yoy_at_lag_12 = float(100.0 * (v[-13] / v[-25] - 1.0))
    observation_dt = _to_dt(d[-13])
    return observation_dt, yoy_at_lag_12, None


def ppi_3m_log_delta(dates: np.ndarray, values: np.ndarray) -> tuple[datetime, float, float | None]:
    """3-month log-change in PPI all-commodities.

    Monthly series. Log-change over trailing 3 months captures short-run
    producer-price pressure that passes through to CPI.
    """
    mask = ~np.isnan(values)
    if mask.sum() < 4:
        latest = _latest_date(dates, values)
        return _to_dt(latest), 0.0, None
    d = dates[mask]
    v = values[mask].astype(float)
    if v.size < 4:
        return _to_dt(d[-1]), 0.0, None
    val = float(np.log(v[-1]) - np.log(v[-4]))
    return _to_dt(d[-1]), val, None


# ---------------------------------------------------------------------------
# Top-level pull.
# ---------------------------------------------------------------------------


def pull_cpi_fundamentals(client: FredClient | None = None) -> list[FundamentalInput]:
    """Fetch all CPI fundamental inputs from FRED.

    Returns an empty list (and logs) if FRED is not configured or the pulls
    all fail. Partial success is allowed: each input is independent.
    """
    if client is None:
        key = os.environ.get("FRED_API_KEY")
        if not key:
            log.info("pull_cpi_fundamentals_skipped", reason="no_fred_api_key")
            return []
        client = FredClient(key)

    now = datetime.now(timezone.utc)
    out: list[FundamentalInput] = []

    specs = [
        (
            "gasoline_weekly_delta", "GASREGW", gasoline_4w_log_delta,
            timedelta(days=14),
        ),
        (
            "supply_chain_pressure", "GSCPI", gscpi_zscore,
            timedelta(days=45),
        ),
        (
            "hpi_lead_12m", "CSUSHPISA", hpi_lead_12m_yoy,
            timedelta(days=60),
        ),
        (
            "ppi_passthrough", "PPIACO", ppi_3m_log_delta,
            timedelta(days=45),
        ),
    ]

    for name, series_id, transform, freshness in specs:
        try:
            series: FredSeries = client.observations(series_id)
        except Exception as e:  # noqa: BLE001 — network failures are non-fatal per input
            log.warning("fundamental_pull_failed", series=series_id, error=str(e))
            continue

        try:
            observation_at, value, unc = transform(series.dates, series.values)
        except Exception as e:  # noqa: BLE001
            log.warning("fundamental_transform_failed", name=name, error=str(e))
            continue

        out.append(FundamentalInput(
            name=name,
            category="economics",
            value=float(value),
            uncertainty=unc,
            mechanism=IntegrationMechanism.PRIOR_SHIFT,
            provenance=InputProvenance(
                source=f"FRED:{series_id}",
                source_kind="automated",
                fetched_at=now,
                observation_at=observation_at,
                notes=None,
            ),
            expires_at=now + freshness,
            scope={"transform": "mom"},
        ))

    log.info("pull_cpi_fundamentals_done", n_inputs=len(out))
    return out


# ---------------------------------------------------------------------------
# Private helpers.
# ---------------------------------------------------------------------------


def _latest_date(dates: np.ndarray, values: np.ndarray) -> "object":
    mask = ~np.isnan(values) if values.size else np.array([], dtype=bool)
    if mask.any():
        return dates[mask][-1]
    return dates[-1] if len(dates) else datetime.now(timezone.utc).date()


def _to_dt(d: object) -> datetime:
    """Coerce a date-ish object to a UTC midnight datetime."""
    if isinstance(d, datetime):
        return d.astimezone(timezone.utc) if d.tzinfo else d.replace(tzinfo=timezone.utc)
    # numpy.datetime64 or stdlib date
    try:
        # stdlib date
        return datetime(d.year, d.month, d.day, tzinfo=timezone.utc)  # type: ignore[attr-defined]
    except AttributeError:
        pass
    try:
        # numpy.datetime64
        ts = int(np.datetime64(d, "s").astype("int64"))
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:  # noqa: BLE001
        return datetime.now(timezone.utc)
