"""Minimal FRED client.

FRED is free with a key (fred.stlouisfed.org/docs/api/api_key.html). We use
only the observations endpoint. If no key is configured we raise; the
caller (the economics forecaster) treats that as "abstain with reason".
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import requests
import requests_cache


@dataclass(frozen=True, slots=True)
class FredSeries:
    series_id: str
    dates: np.ndarray   # dtype=object of datetime.date
    values: np.ndarray  # dtype=float, NaN for missing

    def latest(self) -> tuple[date, float]:
        mask = ~np.isnan(self.values)
        if not mask.any():
            raise ValueError(f"FRED series {self.series_id} has no data")
        i = int(np.max(np.where(mask)))
        return self.dates[i], float(self.values[i])


class FredClient:
    """Single-endpoint client over FRED observations with local caching."""

    _BASE = "https://api.stlouisfed.org/fred/series/observations"

    def __init__(self, api_key: str, cache_path: str | None = None,
                 ttl_seconds: int = 6 * 3600) -> None:
        if not api_key:
            raise ValueError("FRED api_key is required")
        self.api_key = api_key
        backend: Any = requests_cache.SQLiteCache(cache_path) if cache_path else "memory"
        self.session = requests_cache.CachedSession(
            backend=backend, expire_after=ttl_seconds, allowable_methods=("GET",),
            cache_control=False,
        )

    def observations(self, series_id: str, *,
                     start: date | None = None, end: date | None = None) -> FredSeries:
        params: dict[str, Any] = {
            "series_id": series_id, "api_key": self.api_key, "file_type": "json",
        }
        if start:
            params["observation_start"] = start.isoformat()
        if end:
            params["observation_end"] = end.isoformat()

        r = self.session.get(self._BASE, params=params, timeout=30)
        r.raise_for_status()
        obs = r.json().get("observations") or []
        dates = np.asarray([datetime.fromisoformat(o["date"]).date() for o in obs])
        vals = np.asarray([
            float(o["value"]) if o["value"] not in (".", "", None) else np.nan
            for o in obs
        ], dtype=float)
        return FredSeries(series_id=series_id, dates=dates, values=vals)
