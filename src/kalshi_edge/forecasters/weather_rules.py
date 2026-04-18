"""Parse a Kalshi weather contract into structured resolution criteria.

We build three things from a Contract:

1. The **location** (from the series_ticker suffix — "NY", "LAX", ...).
2. The **target calendar date** (from the event_ticker suffix — "26APR18").
3. The **YES criterion** (direction + threshold), taken *from Kalshi's
   structured strike fields* — ``strike_type`` / ``cap_strike`` /
   ``floor_strike`` — not from the ticker suffix.

Why not parse the threshold from the ticker?  The ``-T{N}`` ticker suffix
is just a threshold tag — Kalshi uses it for both "high < 61" and
"high > 75" markets within the same series. Historically we defaulted
``-T`` to "above" and got the sign flipped on every "less than" weather
market, producing ~99% "YES" forecasts against markets correctly priced
near 0¢. ``strike_type`` is the authoritative source; ticker-suffix
parsing is a decoy. See tests/test_weather_rules.py for regressions.

Examples observed in live Kalshi data (2026-04-18):

    ticker        = "KXHIGHNY-26APR18-T61"
    strike_type   = "less"
    cap_strike    = 61
    rules_primary = "If the highest temperature recorded in Central Park,
                    New York for April 18, 2026 ... is less than 61°,
                    then the market resolves to Yes."
    → YES iff high < 61

    ticker        = "KXHIGHLAX-26APR19-T75"
    strike_type   = "greater"
    floor_strike  = 75
    → YES iff high > 75
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Literal

from kalshi_edge.market.strikes import YesCriterion, parse_yes_criterion
from kalshi_edge.types import Contract

Metric = Literal["high", "low"]


@dataclass(frozen=True, slots=True)
class WeatherLocation:
    """Resolved Kalshi weather location.

    The station code is what NWS uses (e.g. "KNYC"). Lat/lon match what
    Open-Meteo ensembles take as input. Kalshi weather contracts reference a
    specific canonical station for each city; using the station's exact
    lat/lon for the forecast lookup prevents a whole class of resolution
    drift bugs.
    """

    slug: str          # KXHIGHNY series slug suffix ("NY")
    city: str          # human-readable city ("New York City")
    station: str       # NWS/ICAO station code ("KNYC")
    lat: float
    lon: float
    tz: str            # IANA timezone string, for daily high/low aggregation


# Canonical Kalshi weather cities. Lat/lon and station IDs verified from
# NWS metadata; these are the locations Kalshi's rules_primary explicitly
# names. If Kalshi adds a city, extend this map — the forecaster will
# abstain on unknown cities.
_LOCATIONS: dict[str, WeatherLocation] = {
    "NY":  WeatherLocation("NY",  "New York City",  "KNYC", 40.7794, -73.9692, "America/New_York"),
    "LAX": WeatherLocation("LAX", "Los Angeles",    "KLAX", 33.9381, -118.3889, "America/Los_Angeles"),
    "CHI": WeatherLocation("CHI", "Chicago",        "KMDW", 41.7860, -87.7524, "America/Chicago"),
    "MIA": WeatherLocation("MIA", "Miami",          "KMIA", 25.7906, -80.3163, "America/New_York"),
    "AUS": WeatherLocation("AUS", "Austin",         "KAUS", 30.1833, -97.6800, "America/Chicago"),
    "DEN": WeatherLocation("DEN", "Denver",         "KDEN", 39.8467, -104.6562, "America/Denver"),
    "PHIL":WeatherLocation("PHIL","Philadelphia",   "KPHL", 39.8719, -75.2411, "America/New_York"),
    "PHX": WeatherLocation("PHX", "Phoenix",        "KPHX", 33.4343, -112.0116, "America/Phoenix"),
}


@dataclass(frozen=True, slots=True)
class WeatherContract:
    """Structured resolution criteria for a weather market."""

    location: WeatherLocation
    target_date: date          # calendar date in the location's local TZ
    metric: Metric             # "high" or "low" daily extremum
    criterion: YesCriterion    # direction + bounds, sourced from Kalshi's strike_type


_SERIES_RE = re.compile(r"^KX(HIGH|LOW)([A-Z]+)$")
_EVENT_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})(?:-|$)")

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])}


def parse_weather_contract(contract: Contract) -> WeatherContract | None:
    """Return structured criteria, or None if the ticker isn't parseable.

    Abstaining on unknown locations / missing strike fields is intentional:
    the forecaster must not invent a threshold direction we don't know.
    """
    ser = contract.series_ticker.upper().strip()
    msm = _SERIES_RE.match(ser)
    if not msm:
        return None
    metric: Metric = "high" if msm.group(1) == "HIGH" else "low"
    loc = _LOCATIONS.get(msm.group(2))
    if loc is None:
        return None

    mdate = _EVENT_DATE_RE.search(contract.event_ticker.upper())
    if not mdate:
        return None
    yy = int(mdate.group(1))
    mon = _MONTHS.get(mdate.group(2))
    dd = int(mdate.group(3))
    if mon is None:
        return None
    target_date = date(2000 + yy, mon, dd)

    criterion = parse_yes_criterion(contract.raw or {})
    if criterion is None:
        return None

    return WeatherContract(
        location=loc,
        target_date=target_date,
        metric=metric,
        criterion=criterion,
    )


def target_local_datetime_bounds(wc: WeatherContract) -> tuple[datetime, datetime]:
    """The local-TZ [start, end) that defines the target calendar day.

    We return tz-aware datetimes in UTC for downstream convenience — but they
    correspond to 00:00 and 24:00 in the *location's* local time, because
    Kalshi weather contracts resolve on the local-day max/min.
    """
    from zoneinfo import ZoneInfo
    tz = ZoneInfo(wc.location.tz)
    start_local = datetime.combine(wc.target_date, datetime.min.time(), tzinfo=tz)
    end_local = datetime.combine(
        date.fromordinal(wc.target_date.toordinal() + 1),
        datetime.min.time(), tzinfo=tz,
    )
    return start_local.astimezone(timezone.utc), end_local.astimezone(timezone.utc)
