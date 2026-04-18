"""Parse a Kalshi weather contract into structured resolution criteria.

Weather markets use a small family of series/ticker patterns. We parse them
from ``series_ticker``, ``event_ticker``, and ``ticker`` alone — not from
``rules_primary`` — because the structured fields are stable and the rules
text is prose. If we can't parse confidently, we return None and the
forecaster abstains (no fake precision).

Examples observed in live Kalshi data (2026-04-18):

    series_ticker = "KXHIGHNY"                         -> NYC daily high
    event_ticker  = "KXHIGHNY-26APR19"                 -> 2026-04-19
    ticker        = "KXHIGHNY-26APR19-T61"             -> P(high > 61 °F)

    series_ticker = "KXHIGHLAX"                        -> LAX daily high
    ticker        = "KXHIGHLAX-26APR19-T75"            -> P(high > 75 °F)

Threshold suffix grammar: ``-T<integer>`` or ``-T<integer>.<frac>``.

The comparator is encoded in the prefix: ``-T`` means "above this value"
(strict >). Kalshi sometimes also publishes ``-B`` (below) or ranged markets
but we only keep what we can parse unambiguously. Anything else → abstain.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Literal

Metric = Literal["high", "low"]
Comparator = Literal["above", "below", "equal"]


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
    threshold_f: float         # °F
    comparator: Comparator     # how to interpret threshold for YES


_SERIES_RE = re.compile(r"^KX(HIGH|LOW)([A-Z]+)$")

# 2-digit year + 3-letter month + 2-digit day, e.g. "26APR19".
_EVENT_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})(?:-|$)")

# Threshold grammar in the ticker suffix. Only strictly-"above" (-T) is
# parsed; other comparator conventions abstain rather than guess.
_THRESHOLD_RE = re.compile(r"-T(-?\d+(?:\.\d+)?)$")

_MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"])}


def parse_weather_contract(
    series_ticker: str, event_ticker: str, ticker: str,
) -> WeatherContract | None:
    """Return structured criteria, or None if the ticker isn't parseable.

    Abstaining on unknown locations / malformed tickers is intentional: the
    forecaster must not invent a threshold or a date we don't understand.
    """
    ser = series_ticker.upper().strip()
    msm = _SERIES_RE.match(ser)
    if not msm:
        return None
    metric: Metric = "high" if msm.group(1) == "HIGH" else "low"
    city_slug = msm.group(2)
    loc = _LOCATIONS.get(city_slug)
    if loc is None:
        return None

    mdate = _EVENT_DATE_RE.search(event_ticker.upper())
    if not mdate:
        return None
    yy = int(mdate.group(1))
    mon = _MONTHS.get(mdate.group(2))
    dd = int(mdate.group(3))
    if mon is None:
        return None
    target_date = date(2000 + yy, mon, dd)

    mthr = _THRESHOLD_RE.search(ticker.upper())
    if not mthr:
        return None
    threshold_f = float(mthr.group(1))

    return WeatherContract(
        location=loc,
        target_date=target_date,
        metric=metric,
        threshold_f=threshold_f,
        comparator="above",
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
