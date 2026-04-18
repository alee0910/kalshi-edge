"""Tests for the NDJSON persistence layer.

The dump layer has two responsibilities: (1) partition rows by their
natural timestamp into YYYY/MM/DD files, and (2) be idempotent within a
single partition (re-running the same dump must not duplicate rows).
Both are covered here with a small in-memory DB populated through the
real Database façade.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kalshi_edge.backtest.persistence import (
    dump_tables_to_ndjson,
    load_table,
    known_tables,
)
from kalshi_edge.storage.db import Database
from kalshi_edge.types import Outcome, Resolution


def _mkdb(tmp_path: Path) -> Database:
    """Real Database instance on a temp sqlite file — no mocking."""
    db = Database(tmp_path / "kalshi.db", wal=False)
    db.init_schema()
    return db


def _insert_contract(db: Database, ticker: str, category: str = "weather") -> None:
    """Minimum-viable contracts row so FK constraints don't blow up."""
    db._conn().execute(
        """
        INSERT OR IGNORE INTO contracts (
            ticker, event_ticker, series_ticker, category, title, status
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (ticker, "E1", "S1", category, "test", "open"),
    )


def _insert_forecast(db: Database, *, ticker: str, ts: datetime, p_yes: float,
                    forecaster: str = "fake_fc", version: str = "v1") -> int:
    """Minimal insert that populates the columns the dumper reads."""
    cur = db._conn().execute(
        """
        INSERT INTO forecasts (
            ticker, ts, forecaster, version, p_yes, p_yes_std,
            p_yes_p05, p_yes_p95, model_confidence,
            methodology, data_sources, diagnostics
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ticker, ts.isoformat(), forecaster, version, p_yes, 0.0,
         p_yes, p_yes, 0.5, "{}", "{}", "{}"),
    )
    return int(cur.lastrowid)


def _insert_snapshot(db: Database, *, ticker: str, ts: datetime,
                    yes_bid: float = 40.0, yes_ask: float = 45.0) -> None:
    db._conn().execute(
        """
        INSERT INTO market_snapshots (
            ticker, ts, yes_bid, yes_ask, no_bid, no_ask,
            last_price, volume, volume_24h, open_interest, liquidity
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (ticker, ts.isoformat(), yes_bid, yes_ask,
         100 - yes_ask, 100 - yes_bid, (yes_bid + yes_ask) / 2,
         100, 500, 50, 100),
    )


class TestDumpAndLoad:
    def test_dumps_known_tables(self, tmp_path):
        db = _mkdb(tmp_path)
        ts = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
        _insert_contract(db, "T1")
        _insert_forecast(db, ticker="T1", ts=ts, p_yes=0.6)
        _insert_snapshot(db, ticker="T1", ts=ts)

        out = tmp_path / "corpus"
        written = dump_tables_to_ndjson(db, out, now=ts)

        # Every known table is represented, even when empty (value 0).
        for name in known_tables():
            assert name in written
        assert written["forecasts"] == 1
        assert written["market_snapshots"] == 1

        # Partition layout: <table>/YYYY/MM/DD.ndjson
        fpath = out / "forecasts" / "2026" / "04" / "18.ndjson"
        assert fpath.exists()
        with fpath.open() as fh:
            rows = [json.loads(l) for l in fh if l.strip()]
        assert len(rows) == 1
        assert rows[0]["ticker"] == "T1"
        assert rows[0]["p_yes"] == 0.6

    def test_partition_by_timestamp(self, tmp_path):
        """Rows with different ts dates land in different partition files."""
        db = _mkdb(tmp_path)
        ts_a = datetime(2026, 4, 17, 23, 0, tzinfo=timezone.utc)
        ts_b = datetime(2026, 4, 18, 1, 0, tzinfo=timezone.utc)
        _insert_contract(db, "A")
        _insert_contract(db, "B")
        _insert_forecast(db, ticker="A", ts=ts_a, p_yes=0.2)
        _insert_forecast(db, ticker="B", ts=ts_b, p_yes=0.8)

        out = tmp_path / "corpus"
        dump_tables_to_ndjson(db, out, now=ts_b)

        assert (out / "forecasts" / "2026" / "04" / "17.ndjson").exists()
        assert (out / "forecasts" / "2026" / "04" / "18.ndjson").exists()

    def test_idempotent_within_partition(self, tmp_path):
        """Dumping the same DB twice to the same dir yields one row per PK."""
        db = _mkdb(tmp_path)
        ts = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
        _insert_contract(db, "T1")
        _insert_forecast(db, ticker="T1", ts=ts, p_yes=0.6)

        out = tmp_path / "corpus"
        first = dump_tables_to_ndjson(db, out, now=ts)
        second = dump_tables_to_ndjson(db, out, now=ts)

        assert first["forecasts"] == 1
        assert second["forecasts"] == 0  # all PKs already present

        loaded = load_table(out, "forecasts")
        assert len(loaded) == 1

    def test_load_table_dedupes_last_write_wins(self, tmp_path):
        """If two files contain the same PK, load_table keeps the last one."""
        out = tmp_path / "corpus" / "resolutions"
        out.mkdir(parents=True)
        # Same ticker appears in two files — later file wins.
        earlier = out / "2026" / "04" / "17.ndjson"
        earlier.parent.mkdir(parents=True)
        earlier.write_text(json.dumps({
            "ticker": "X", "resolved_at": "2026-04-17T10:00:00+00:00",
            "outcome": "YES", "settled_price": 100.0,
        }) + "\n")
        later = out / "2026" / "04" / "18.ndjson"
        later.parent.mkdir(parents=True, exist_ok=True)
        later.write_text(json.dumps({
            "ticker": "X", "resolved_at": "2026-04-18T10:00:00+00:00",
            "outcome": "NO", "settled_price": 0.0,
        }) + "\n")

        rows = load_table(tmp_path / "corpus", "resolutions")
        assert len(rows) == 1
        assert rows[0]["outcome"] == "NO"

    def test_unknown_table_errors(self, tmp_path):
        with pytest.raises(KeyError):
            load_table(tmp_path, "does_not_exist")
