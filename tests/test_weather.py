"""Weather forecaster + contract-rule + Open-Meteo parser tests.

No live HTTP in these tests: the Open-Meteo client is replaced by a fake
that returns a handcrafted rectangular ensemble. This exercises the math
(Gaussian fit, variance inflation, bootstrap posterior) with deterministic
inputs so the asserted tolerances are defensible.

Contract shape (post-2026-04-18): YES direction comes from Kalshi's
structured ``strike_type`` + ``cap_strike``/``floor_strike`` fields on the
raw market payload — NOT from ticker-suffix convention. The fixtures below
mirror what the live Kalshi ``/markets`` response looks like.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from kalshi_edge.data_sources.open_meteo import EnsembleFetch, daily_extrema
from kalshi_edge.forecasters.weather import (
    MIN_POSTERIOR_STD_F,
    WeatherForecaster,
)
from kalshi_edge.forecasters.weather_rules import (
    parse_weather_contract,
    target_local_datetime_bounds,
)
from kalshi_edge.types import Category, Contract, MarketStatus


# ---------- fixtures ------------------------------------------------------

def _contract(
    ticker: str,
    event: str,
    series: str,
    *,
    close: datetime | None = None,
    raw: dict[str, Any] | None = None,
) -> Contract:
    close = close or datetime.now(timezone.utc) + timedelta(days=2)
    return Contract(
        ticker=ticker, event_ticker=event, series_ticker=series,
        title="t", subtitle=None, category=Category.WEATHER,
        open_time=None, close_time=close, expiration_time=close,
        status=MarketStatus.OPEN,
        rules_primary=None, rules_secondary=None,
        resolution_criteria={}, raw=raw or {},
    )


def _above(floor: float) -> dict[str, Any]:
    """Kalshi raw payload fragment for a 'YES iff value > floor' contract."""
    return {"strike_type": "greater", "floor_strike": floor}


def _below(cap: float) -> dict[str, Any]:
    """Kalshi raw payload fragment for a 'YES iff value < cap' contract."""
    return {"strike_type": "less", "cap_strike": cap}


# ---------- rule parser ---------------------------------------------------

def test_parse_nyc_high_above_threshold() -> None:
    c = _contract(
        "KXHIGHNY-26APR19-T61", "KXHIGHNY-26APR19", "KXHIGHNY",
        raw=_above(61.0),
    )
    wc = parse_weather_contract(c)
    assert wc is not None
    assert wc.location.slug == "NY"
    assert wc.location.station == "KNYC"
    assert wc.target_date == date(2026, 4, 19)
    assert wc.metric == "high"
    assert wc.criterion.direction == "above"
    assert wc.criterion.low == 61.0


def test_parse_lax_low() -> None:
    c = _contract(
        "KXLOWLAX-26MAR01-T52.5", "KXLOWLAX-26MAR01", "KXLOWLAX",
        raw=_below(52.5),
    )
    wc = parse_weather_contract(c)
    assert wc is not None
    assert wc.metric == "low"
    assert wc.criterion.direction == "below"
    assert wc.criterion.high == 52.5


def test_parse_less_than_ticker_is_below_regardless_of_suffix() -> None:
    # Regression for KXHIGHNY-26APR18-T61: ticker suffix says "T61", but
    # strike_type="less" means YES iff high < 61. The old code hardcoded
    # comparator="above" and produced a sign-flipped forecast.
    c = _contract(
        "KXHIGHNY-26APR18-T61", "KXHIGHNY-26APR18", "KXHIGHNY",
        raw=_below(61.0),
    )
    wc = parse_weather_contract(c)
    assert wc is not None and wc.criterion.direction == "below"
    assert wc.criterion.high == 61.0


def test_parse_unknown_city_abstains() -> None:
    c = _contract(
        "KXHIGHMOSCOW-26APR19-T40", "KXHIGHMOSCOW-26APR19", "KXHIGHMOSCOW",
        raw=_above(40.0),
    )
    assert parse_weather_contract(c) is None


def test_parse_bad_ticker_abstains() -> None:
    # Unparseable series_ticker (doesn't match KX(HIGH|LOW)<city>).
    c1 = _contract("x", "KXHIGHNY-26APR19", "GARBAGE", raw=_above(55.0))
    assert parse_weather_contract(c1) is None
    # Missing date in event_ticker.
    c2 = _contract("KXHIGHNY-T55", "KXHIGHNY", "KXHIGHNY", raw=_above(55.0))
    assert parse_weather_contract(c2) is None


def test_parse_missing_strike_type_abstains() -> None:
    # If Kalshi ever ships a market without strike_type (or with an
    # unrecognized one like "structured"), we abstain rather than guess.
    c = _contract(
        "KXHIGHNY-26APR19-T61", "KXHIGHNY-26APR19", "KXHIGHNY",
        raw={"strike_type": "structured"},
    )
    assert parse_weather_contract(c) is None


def test_target_local_day_bounds_span_24h_local() -> None:
    c = _contract(
        "KXHIGHNY-26APR19-T61", "KXHIGHNY-26APR19", "KXHIGHNY",
        raw=_above(61.0),
    )
    wc = parse_weather_contract(c)
    assert wc is not None
    start_utc, end_utc = target_local_datetime_bounds(wc)
    assert (end_utc - start_utc).total_seconds() == 24 * 3600


# ---------- fake ensemble fetch ------------------------------------------

def _fake_fetch(mu_f: float, sigma_f: float, target: date,
                tz: str = "America/New_York", n_members: int = 40) -> EnsembleFetch:
    """Build a deterministic ensemble fetch centered at (mu, sigma) for target."""
    rng = np.random.default_rng(seed=42)
    local = ZoneInfo(tz)
    hours = [datetime.combine(target, datetime.min.time(), tzinfo=local) + timedelta(hours=h)
             for h in range(48)]
    hours_utc = np.asarray([t.astimezone(ZoneInfo("UTC")) for t in hours])
    diurnal = np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, 48))
    base = mu_f + diurnal * (sigma_f * 0.2)
    shifts = rng.normal(0.0, sigma_f, size=(n_members, 1))
    temps = base[None, :] + shifts
    return EnsembleFetch(
        location_slug="NY",
        lat=40.78, lon=-73.97, tz=tz,
        fetched_at=datetime.now(ZoneInfo("UTC")),
        issued_at=None,
        hours_utc=hours_utc,
        temps_f=temps,
        model_labels=tuple(f"fake/member{i:02d}" for i in range(n_members)),
    )


def test_daily_extrema_picks_local_day_max() -> None:
    target = date(2026, 4, 19)
    fetch = _fake_fetch(mu_f=60.0, sigma_f=2.0, target=target, n_members=10)
    highs = daily_extrema(fetch, target, "high")
    assert highs.shape == (10,)
    assert 50.0 < highs.mean() < 70.0


def test_daily_extrema_rejects_date_outside_window() -> None:
    fetch = _fake_fetch(60.0, 2.0, target=date(2026, 4, 19), n_members=10)
    with pytest.raises(ValueError):
        daily_extrema(fetch, date(2026, 4, 1), "high")


# ---------- forecaster ---------------------------------------------------

class _FakeClient:
    """Stand-in for OpenMeteoClient that returns a prebuilt fetch."""

    def __init__(self, fetch: EnsembleFetch) -> None:
        self.fetch = fetch

    def fetch_ensemble(self, **_kwargs):
        return self.fetch


def test_forecaster_produces_posterior_and_edge() -> None:
    # Target 1 day ahead, threshold well below ensemble mean → P(YES above) ≈ 1.
    today = datetime.now(timezone.utc).date()
    target = today + timedelta(days=1)
    fetch = _fake_fetch(mu_f=70.0, sigma_f=2.0, target=target, n_members=50)
    fc = WeatherForecaster(client=_FakeClient(fetch), variance_inflation=1.3)

    close = datetime.combine(target, datetime.min.time(), tzinfo=timezone.utc)
    c = _contract(
        ticker=f"KXHIGHNY-{target:%y%b%d}".upper() + "-T55",
        event=f"KXHIGHNY-{target:%y%b%d}".upper(),
        series="KXHIGHNY",
        close=close,
        raw=_above(55.0),
    )
    r = fc.forecast(c)
    assert not r.is_null(), r.null_reason
    assert r.p_yes > 0.95
    assert r.p_yes_std >= 0.0
    assert set(r.uncertainty) == {"total", "aleatoric", "epistemic"}
    assert r.underlying_posterior is not None
    assert r.underlying_posterior.kind == "parametric"
    assert r.underlying_posterior.std() >= MIN_POSTERIOR_STD_F
    assert "prior" in r.methodology and "posterior_over_p" in r.methodology


def test_forecaster_below_threshold_flips_sign() -> None:
    # Regression: with strike_type="less", cap=55, and ensemble mean 70,
    # P(YES) = P(high < 55) must be near 0, not near 1.
    today = datetime.now(timezone.utc).date()
    target = today + timedelta(days=1)
    fetch = _fake_fetch(mu_f=70.0, sigma_f=2.0, target=target, n_members=50)
    fc = WeatherForecaster(client=_FakeClient(fetch), variance_inflation=1.3)
    c = _contract(
        ticker=f"KXHIGHNY-{target:%y%b%d}".upper() + "-T55",
        event=f"KXHIGHNY-{target:%y%b%d}".upper(),
        series="KXHIGHNY",
        close=datetime.combine(target, datetime.min.time(), tzinfo=timezone.utc),
        raw=_below(55.0),
    )
    r = fc.forecast(c)
    assert not r.is_null(), r.null_reason
    assert r.p_yes < 0.05


def test_forecaster_near_threshold_has_uncertainty() -> None:
    today = datetime.now(timezone.utc).date()
    target = today + timedelta(days=2)
    fetch = _fake_fetch(mu_f=60.0, sigma_f=3.0, target=target, n_members=50)
    fc = WeatherForecaster(client=_FakeClient(fetch), variance_inflation=1.3)

    c = _contract(
        ticker=f"KXHIGHNY-{target:%y%b%d}".upper() + "-T60",
        event=f"KXHIGHNY-{target:%y%b%d}".upper(),
        series="KXHIGHNY",
        close=datetime.combine(target, datetime.min.time(), tzinfo=timezone.utc),
        raw=_above(60.0),
    )
    r = fc.forecast(c)
    assert not r.is_null()
    assert 0.3 < r.p_yes < 0.7
    assert r.uncertainty["epistemic"] > 0.0


def test_forecaster_rejects_unparseable_ticker() -> None:
    fetch = _fake_fetch(60.0, 2.0, target=date(2026, 4, 19), n_members=30)
    fc = WeatherForecaster(client=_FakeClient(fetch))
    c = _contract(
        ticker="GIBBERISH-X", event="GIBBERISH", series="GIBBERISH",
        close=datetime(2026, 4, 19, tzinfo=timezone.utc),
        raw={},
    )
    r = fc.forecast(c)
    assert r.is_null()


def test_forecaster_rejects_past_target_date() -> None:
    past = datetime.now(timezone.utc).date() - timedelta(days=3)
    fetch = _fake_fetch(60.0, 2.0, target=past, n_members=30)
    fc = WeatherForecaster(client=_FakeClient(fetch))
    c = _contract(
        ticker=f"KXHIGHNY-{past:%y%b%d}".upper() + "-T55",
        event=f"KXHIGHNY-{past:%y%b%d}".upper(),
        series="KXHIGHNY",
        close=datetime.combine(past, datetime.min.time(), tzinfo=timezone.utc),
        raw=_above(55.0),
    )
    r = fc.forecast(c)
    assert r.is_null() and "past" in (r.null_reason or "")


def test_variance_inflation_monotone() -> None:
    today = datetime.now(timezone.utc).date()
    target = today + timedelta(days=1)
    fetch = _fake_fetch(mu_f=60.0, sigma_f=1.5, target=target, n_members=50)

    c = _contract(
        ticker=f"KXHIGHNY-{target:%y%b%d}".upper() + "-T62",
        event=f"KXHIGHNY-{target:%y%b%d}".upper(),
        series="KXHIGHNY",
        close=datetime.combine(target, datetime.min.time(), tzinfo=timezone.utc),
        raw=_above(62.0),
    )
    r_low = WeatherForecaster(client=_FakeClient(fetch), variance_inflation=1.0).forecast(c)
    r_high = WeatherForecaster(client=_FakeClient(fetch), variance_inflation=2.0).forecast(c)
    # Higher inflation → more probability in the upper tail → P(high > 62) larger.
    assert r_high.p_yes > r_low.p_yes
