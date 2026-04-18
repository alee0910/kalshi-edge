"""Weather forecaster + contract-rule + Open-Meteo parser tests.

No live HTTP in these tests: the Open-Meteo client is replaced by a fake
that returns a handcrafted rectangular ensemble. This exercises the math
(Gaussian fit, variance inflation, bootstrap posterior) with deterministic
inputs so the asserted tolerances are defensible.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
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


# ---------- rule parser ---------------------------------------------------

def test_parse_nyc_high_above_threshold() -> None:
    wc = parse_weather_contract(
        "KXHIGHNY", "KXHIGHNY-26APR19", "KXHIGHNY-26APR19-T61",
    )
    assert wc is not None
    assert wc.location.slug == "NY"
    assert wc.location.station == "KNYC"
    assert wc.target_date == date(2026, 4, 19)
    assert wc.metric == "high"
    assert wc.threshold_f == 61.0
    assert wc.comparator == "above"


def test_parse_lax_low() -> None:
    wc = parse_weather_contract(
        "KXLOWLAX", "KXLOWLAX-26MAR01", "KXLOWLAX-26MAR01-T52.5",
    )
    assert wc is not None
    assert wc.metric == "low"
    assert wc.threshold_f == 52.5


def test_parse_unknown_city_abstains() -> None:
    # No KXHIGHMOSCOW in our station map.
    assert parse_weather_contract(
        "KXHIGHMOSCOW", "KXHIGHMOSCOW-26APR19", "KXHIGHMOSCOW-26APR19-T40",
    ) is None


def test_parse_bad_ticker_abstains() -> None:
    assert parse_weather_contract("KXHIGHNY", "KXHIGHNY-26APR19", "garbage") is None
    # Missing date
    assert parse_weather_contract("KXHIGHNY", "KXHIGHNY", "KXHIGHNY-T55") is None


def test_target_local_day_bounds_span_24h_local() -> None:
    wc = parse_weather_contract(
        "KXHIGHNY", "KXHIGHNY-26APR19", "KXHIGHNY-26APR19-T61",
    )
    assert wc is not None
    start_utc, end_utc = target_local_datetime_bounds(wc)
    # Midnight local → midnight next day local is always 24h even across DST
    # for the contract day itself (DST transitions happen in spring/fall, and
    # 2026-04-19 is after the spring US DST transition so it's stable 24h).
    assert (end_utc - start_utc).total_seconds() == 24 * 3600


# ---------- fake ensemble fetch ------------------------------------------

def _fake_fetch(mu_f: float, sigma_f: float, target: date,
                tz: str = "America/New_York", n_members: int = 40) -> EnsembleFetch:
    """Build a deterministic ensemble fetch centered at (mu, sigma) for target."""
    rng = np.random.default_rng(seed=42)
    local = ZoneInfo(tz)
    # 48 hourly points spanning target and the following day.
    hours = [datetime.combine(target, datetime.min.time(), tzinfo=local) + timedelta(hours=h)
             for h in range(48)]
    hours_utc = np.asarray([t.astimezone(ZoneInfo("UTC")) for t in hours])
    # Sine diurnal shape: we'll use the peak of the window as daily high.
    diurnal = np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, 48))  # peaks at hour 12
    base = mu_f + diurnal * (sigma_f * 0.2)
    # Per-member temp trajectory = base shifted by a Gaussian realization.
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
    # The synthetic peak is mu + sigma*0.2, within a few °F for any realization.
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


def _contract(ticker: str, event: str, series: str,
              close: datetime) -> Contract:
    return Contract(
        ticker=ticker, event_ticker=event, series_ticker=series,
        title="t", subtitle=None, category=Category.WEATHER,
        open_time=None, close_time=close, expiration_time=close,
        status=MarketStatus.OPEN,
        rules_primary=None, rules_secondary=None,
        resolution_criteria={}, raw={},
    )


def test_forecaster_produces_posterior_and_edge() -> None:
    # Target 1 day ahead, threshold well below ensemble mean → P(YES) ≈ 1.
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
    )
    r = fc.forecast(c)
    assert not r.is_null(), r.null_reason
    assert r.p_yes > 0.95        # threshold 55 well below 70±(1.3·σ)
    assert r.p_yes_std >= 0.0    # bootstrap spread is nonneg
    # Uncertainty decomposition populated by finalize().
    assert set(r.uncertainty) == {"total", "aleatoric", "epistemic"}
    # Underlying posterior is a Gaussian with inflated sigma.
    assert r.underlying_posterior is not None
    assert r.underlying_posterior.kind == "parametric"
    assert r.underlying_posterior.std() >= MIN_POSTERIOR_STD_F
    # Methodology record present and non-empty.
    assert "prior" in r.methodology and "posterior_over_p" in r.methodology


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
    )
    r = fc.forecast(c)
    assert not r.is_null()
    # Near the mean → p_yes ~ 0.5 with meaningful epistemic uncertainty.
    assert 0.3 < r.p_yes < 0.7
    assert r.uncertainty["epistemic"] > 0.0


def test_forecaster_rejects_unparseable_ticker() -> None:
    fetch = _fake_fetch(60.0, 2.0, target=date(2026, 4, 19), n_members=30)
    fc = WeatherForecaster(client=_FakeClient(fetch))
    c = _contract(
        ticker="GIBBERISH-X", event="GIBBERISH", series="GIBBERISH",
        close=datetime(2026, 4, 19, tzinfo=timezone.utc),
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
    )
    r_low = WeatherForecaster(client=_FakeClient(fetch), variance_inflation=1.0).forecast(c)
    r_high = WeatherForecaster(client=_FakeClient(fetch), variance_inflation=2.0).forecast(c)
    # Higher inflation → more probability pushed toward the tail →
    # probability of "above 62" (right of the 60 mean) should be larger.
    assert r_high.p_yes > r_low.p_yes
