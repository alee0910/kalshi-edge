"""Open-Meteo ensemble data source.

We fetch the raw ensemble (GEFS 31 + ECMWF IFS 51 = 82 members) and expose
per-member hourly temperature traces for a target date + location. No
silent reduction to a mean/variance summary — the forecaster's job is to
reason over the full ensemble distribution.

Open-Meteo is free, no-key, public. Endpoint:
    https://ensemble-api.open-meteo.com/v1/ensemble

Response shape (confirmed 2026-04-18):
    hourly.temperature_2m              -> control run (1 series)
    hourly.temperature_2m_member01..NN -> perturbed members
    hourly.time                        -> ISO8601 timestamps (local TZ)

We request °F directly via ``temperature_unit=fahrenheit`` so we never
have to reason about unit conversion around the Kalshi threshold.

Caching: requests-cache, 15 min TTL. Ensembles are re-issued every 6h
for GFS, 12h for IFS, so a 15 min local TTL is more than fresh enough
and drastically cuts API load when we re-score a universe.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import numpy as np
import requests
import requests_cache
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from kalshi_edge.logging_ import get_logger

log = get_logger(__name__)

_ENSEMBLE_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Model IDs Open-Meteo accepts. We pull these two; their members are
# independent, perturbed around different control runs, so pooling them
# adds genuine model-uncertainty coverage to what would otherwise be
# within-model sampling variance alone.
_MODELS = ("gfs_seamless", "ecmwf_ifs025")


_RETRYABLE = (requests.ConnectionError, requests.Timeout)


@dataclass(frozen=True, slots=True)
class EnsembleFetch:
    """A per-location ensemble pull. Rectangular: (n_members, n_hours)."""

    location_slug: str
    lat: float
    lon: float
    tz: str
    fetched_at: datetime           # when we hit the API
    issued_at: datetime | None     # model issue time (if present)
    hours_utc: np.ndarray          # (n_hours,) tz-aware UTC datetimes
    temps_f: np.ndarray            # (n_members, n_hours)
    model_labels: tuple[str, ...]  # one label per row of temps_f


class OpenMeteoClient:
    """Thin, cached wrapper over Open-Meteo's ensemble endpoint."""

    def __init__(self, cache_path: str | None = None, ttl_seconds: int = 900) -> None:
        backend: Any = requests_cache.SQLiteCache(cache_path) if cache_path else "memory"
        self.session = requests_cache.CachedSession(
            backend=backend,
            expire_after=ttl_seconds,
            allowable_methods=("GET",),
            cache_control=False,
        )
        self.session.headers["User-Agent"] = "kalshi-edge/0.1 (weather forecaster)"

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=1.0, max=10.0),
        reraise=True,
    )
    def _get(self, params: dict[str, Any]) -> dict[str, Any]:
        r = self.session.get(_ENSEMBLE_URL, params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    def fetch_ensemble(
        self, *, lat: float, lon: float, tz: str, slug: str,
        forecast_days: int = 7,
    ) -> EnsembleFetch:
        """Fetch both models, pool members, return a single rectangular array.

        We request temperatures in °F (so threshold comparison is trivial) and
        hours in the location's local TZ (so the response's "time" column
        corresponds to the local calendar day the Kalshi contract resolves on).
        """
        rows: list[np.ndarray] = []
        labels: list[str] = []
        time_strs: list[str] | None = None
        issue_times: list[datetime] = []

        for model in _MODELS:
            params = {
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m",
                "models": model,
                "timezone": tz,
                "temperature_unit": "fahrenheit",
                "forecast_days": forecast_days,
            }
            data = self._get(params)
            h = data.get("hourly") or {}
            if not h:
                continue
            if time_strs is None:
                time_strs = list(h["time"])
            # Control + each member variable. Sorted so output is deterministic.
            member_keys = [k for k in h.keys()
                           if k == "temperature_2m" or k.startswith("temperature_2m_member")]
            member_keys.sort(key=lambda k: (len(k), k))
            for k in member_keys:
                series = h.get(k)
                if series is None:
                    continue
                rows.append(np.asarray(series, dtype=float))
                labels.append(f"{model}/{k}")

            # Open-Meteo surfaces the model issue time via `hourly` prefix data;
            # it's not a standard key. We approximate by using the fetch time
            # as an upper bound on data age. If a future API exposes the issue
            # time we can plug it in here.

        if not rows or not time_strs:
            raise RuntimeError(f"Open-Meteo returned no ensemble data for {slug}")

        temps = np.vstack(rows)  # (n_members, n_hours)

        # Parse local ISO timestamps and convert to UTC.
        local_tz = ZoneInfo(tz)
        hours_utc = np.asarray([
            datetime.fromisoformat(s).replace(tzinfo=local_tz).astimezone(ZoneInfo("UTC"))
            for s in time_strs
        ])

        return EnsembleFetch(
            location_slug=slug,
            lat=lat, lon=lon, tz=tz,
            fetched_at=datetime.now(ZoneInfo("UTC")),
            issued_at=max(issue_times) if issue_times else None,
            hours_utc=hours_utc,
            temps_f=temps,
            model_labels=tuple(labels),
        )


def daily_extrema(
    fetch: EnsembleFetch, target_date: date, metric: str,
) -> np.ndarray:
    """Reduce an hourly ensemble to per-member daily max or min for one date.

    "Daily" = calendar day in the *location's* local timezone, which is how
    Kalshi weather contracts resolve (e.g. "NYC high on 2026-04-19" spans
    local midnight to midnight, not UTC).
    """
    if metric not in ("high", "low"):
        raise ValueError(f"metric must be 'high' or 'low', got {metric!r}")
    tz = ZoneInfo(fetch.tz)
    local_dates = np.asarray([dt.astimezone(tz).date() for dt in fetch.hours_utc])
    mask = local_dates == target_date
    if not mask.any():
        raise ValueError(
            f"target_date {target_date} not in ensemble window "
            f"({local_dates.min()} → {local_dates.max()})"
        )
    window = fetch.temps_f[:, mask]
    return window.max(axis=1) if metric == "high" else window.min(axis=1)


def pool_members(fetches: Iterable[EnsembleFetch]) -> np.ndarray:
    """Row-stack several fetches' member temps into one pooled ensemble."""
    return np.vstack([f.temps_f for f in fetches])
