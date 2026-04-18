"""Storage round-trips: contract, snapshot, forecast (samples + parametric)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from kalshi_edge.forecast import ForecastDistribution, ForecastResult
from kalshi_edge.storage import Database
from kalshi_edge.types import (
    Category,
    Contract,
    MarketSnapshot,
    MarketStatus,
    Outcome,
    Resolution,
)


@pytest.fixture
def db(tmp_path: Path) -> Database:
    d = Database(tmp_path / "test.db", wal=False)
    d.init_schema()
    return d


def _contract(ticker: str = "T1") -> Contract:
    return Contract(
        ticker=ticker, event_ticker="E1", series_ticker="S1",
        title="t", subtitle=None, category=Category.WEATHER,
        open_time=None,
        close_time=datetime(2026, 5, 1, tzinfo=timezone.utc),
        expiration_time=datetime(2026, 5, 1, tzinfo=timezone.utc),
        status=MarketStatus.OPEN,
    )


def test_contract_roundtrip(db: Database) -> None:
    c = _contract()
    db.upsert_contract(c)
    got = db.get_contract(c.ticker)
    assert got is not None
    assert got.ticker == c.ticker
    assert got.category == c.category
    assert got.close_time == c.close_time


def test_snapshot_roundtrip(db: Database) -> None:
    c = _contract()
    db.upsert_contract(c)
    snap = MarketSnapshot(
        ticker=c.ticker, ts=datetime(2026, 4, 18, 12, tzinfo=timezone.utc),
        yes_bid=42, yes_ask=46, no_bid=54, no_ask=58,
        last_price=44, last_trade_ts=None,
        volume=100, volume_24h=10, open_interest=5,
    )
    db.insert_snapshot(snap)
    got = db.latest_snapshot(c.ticker)
    assert got is not None and got.yes_bid == 42 and got.yes_ask == 46


def test_forecast_samples_roundtrip(db: Database) -> None:
    c = _contract()
    db.upsert_contract(c)
    rng = np.random.default_rng(0)
    samples = rng.beta(2, 3, size=1000).astype(np.float32)
    fr = ForecastResult(
        ticker=c.ticker,
        ts=datetime(2026, 4, 18, tzinfo=timezone.utc),
        forecaster="test", version="1",
        binary_posterior=ForecastDistribution(kind="samples", samples=samples),
        model_confidence=0.7,
        methodology={"model": "test"},
        data_sources={"src": datetime(2026, 4, 18, tzinfo=timezone.utc)},
        diagnostics={"rhat": 1.01},
    )
    fid = db.insert_forecast(fr)
    assert fid > 0
    got = db.get_forecast(fid)
    assert got is not None
    assert got.binary_posterior.kind == "samples"
    assert got.binary_posterior.samples is not None
    np.testing.assert_allclose(got.binary_posterior.samples, samples, rtol=1e-5)
    assert got.model_confidence == pytest.approx(0.7)


def test_forecast_parametric_roundtrip(db: Database) -> None:
    c = _contract()
    db.upsert_contract(c)
    fr = ForecastResult(
        ticker=c.ticker,
        ts=datetime(2026, 4, 18, tzinfo=timezone.utc),
        forecaster="test", version="1",
        binary_posterior=ForecastDistribution(
            kind="parametric", family="beta", params={"a": 3.0, "b": 5.0}
        ),
        model_confidence=0.5,
    )
    fid = db.insert_forecast(fr)
    got = db.get_forecast(fid)
    assert got is not None
    assert got.binary_posterior.kind == "parametric"
    assert got.binary_posterior.family == "beta"
    assert got.binary_posterior.params == {"a": 3.0, "b": 5.0}


def test_null_forecast_stores_reason(db: Database) -> None:
    c = _contract()
    db.upsert_contract(c)
    fr = ForecastResult(
        ticker=c.ticker,
        ts=datetime(2026, 4, 18, tzinfo=timezone.utc),
        forecaster="test", version="1",
        binary_posterior=ForecastDistribution(
            kind="parametric", family="beta", params={"a": 1.0, "b": 1.0}
        ),
        null_reason="no data source available",
    )
    fid = db.insert_forecast(fr)
    got = db.get_forecast(fid)
    assert got is not None and got.null_reason == "no data source available"


def test_resolution_roundtrip(db: Database) -> None:
    c = _contract()
    db.upsert_contract(c)
    r = Resolution(ticker=c.ticker, resolved_at=datetime(2026, 5, 2, tzinfo=timezone.utc),
                   outcome=Outcome.YES, settled_price=100.0)
    db.upsert_resolution(r)
    row = db.execute("SELECT outcome, settled_price FROM resolutions WHERE ticker=?",
                     (c.ticker,)).fetchone()
    assert row["outcome"] == "YES"
    assert row["settled_price"] == 100.0
