"""SQLite DAL.

We keep this intentionally low-abstraction: thin helpers over ``sqlite3`` that
return dataclasses from ``types.py`` / ``forecast.py``. No ORM. For the
scales we care about (thousands of contracts, millions of snapshots over a
year) SQLite with WAL is more than fast enough and removes a dependency.

Posterior samples are stored as BLOBs via ``numpy.save`` bytes. Parametric
posteriors round-trip through JSON. Either representation round-trips to a
``ForecastDistribution``.
"""

from __future__ import annotations

import io
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np

from kalshi_edge.forecast import ForecastDistribution, ForecastResult
from kalshi_edge.types import (
    Category,
    Contract,
    MarketSnapshot,
    MarketStatus,
    Outcome,
    Resolution,
)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"
_SCHEMA_VERSION = 1


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _parse_iso(s: str | None) -> datetime | None:
    if s is None:
        return None
    # sqlite adapters sometimes round-trip as 'YYYY-MM-DD HH:MM:SS'
    return datetime.fromisoformat(s.replace(" ", "T") if "T" not in s else s)


def _samples_to_blob(x: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, x.astype(np.float32), allow_pickle=False)
    return buf.getvalue()


def _blob_to_samples(b: bytes | None) -> np.ndarray | None:
    if b is None:
        return None
    return np.load(io.BytesIO(b), allow_pickle=False)


def _dist_to_storage(d: ForecastDistribution | None) -> tuple[bytes | None, str | None, str | None]:
    if d is None:
        return None, None, None
    if d.kind == "samples":
        assert d.samples is not None
        return _samples_to_blob(d.samples), "samples", None
    return None, "parametric", json.dumps({"family": d.family, "params": d.params})


def _storage_to_dist(
    blob: bytes | None, kind: str | None, params_json: str | None
) -> ForecastDistribution | None:
    if kind is None:
        return None
    if kind == "samples":
        samples = _blob_to_samples(blob)
        if samples is None:
            return None
        return ForecastDistribution(kind="samples", samples=samples)
    if kind == "parametric":
        p = json.loads(params_json or "{}")
        return ForecastDistribution(kind="parametric", family=p.get("family"), params=p.get("params", {}))
    raise ValueError(f"unknown stored kind: {kind}")


class Database:
    """Thread-local SQLite connection manager + typed CRUD helpers."""

    def __init__(self, path: Path | str, wal: bool = True) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._wal = wal
        self._local = threading.local()

    # ------- connection lifecycle ---------------------------------------
    def _conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            # No PARSE_DECLTYPES: the stdlib default TIMESTAMP converter requires
            # 'YYYY-MM-DD HH:MM:SS' and mangles tz-aware ISO-8601. We store
            # timestamps as TEXT and round-trip them through _iso / _parse_iso.
            c = sqlite3.connect(
                self.path,
                isolation_level=None,  # autocommit; we use explicit BEGIN
                check_same_thread=False,
            )
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA foreign_keys = ON")
            if self._wal:
                c.execute("PRAGMA journal_mode = WAL")
                c.execute("PRAGMA synchronous = NORMAL")
            self._local.conn = c
        return c

    def close(self) -> None:
        c = getattr(self._local, "conn", None)
        if c is not None:
            c.close()
            self._local.conn = None

    # ------- schema ------------------------------------------------------
    def init_schema(self) -> None:
        sql = _SCHEMA_PATH.read_text()
        c = self._conn()
        c.executescript(sql)
        c.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
            (_SCHEMA_VERSION,),
        )

    # ------- contracts ---------------------------------------------------
    def upsert_contract(self, contract: Contract) -> None:
        c = self._conn()
        c.execute(
            """
            INSERT INTO contracts (
                ticker, event_ticker, series_ticker, category, title, subtitle,
                status, open_time, close_time, expiration_time,
                rules_primary, rules_secondary, resolution_criteria, raw
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(ticker) DO UPDATE SET
                event_ticker = excluded.event_ticker,
                series_ticker = excluded.series_ticker,
                category = excluded.category,
                title = excluded.title,
                subtitle = excluded.subtitle,
                status = excluded.status,
                open_time = excluded.open_time,
                close_time = excluded.close_time,
                expiration_time = excluded.expiration_time,
                rules_primary = excluded.rules_primary,
                rules_secondary = excluded.rules_secondary,
                resolution_criteria = excluded.resolution_criteria,
                raw = excluded.raw,
                updated_at = CURRENT_TIMESTAMP
            """,
            (
                contract.ticker,
                contract.event_ticker,
                contract.series_ticker,
                contract.category.value,
                contract.title,
                contract.subtitle,
                contract.status.value,
                _iso(contract.open_time),
                _iso(contract.close_time),
                _iso(contract.expiration_time),
                contract.rules_primary,
                contract.rules_secondary,
                json.dumps(contract.resolution_criteria) if contract.resolution_criteria else None,
                json.dumps(contract.raw) if contract.raw else None,
            ),
        )

    def get_contract(self, ticker: str) -> Contract | None:
        row = self._conn().execute(
            "SELECT * FROM contracts WHERE ticker = ?", (ticker,)
        ).fetchone()
        return _row_to_contract(row) if row else None

    def iter_contracts(self, category: Category | None = None) -> Iterator[Contract]:
        q = "SELECT * FROM contracts"
        args: tuple[Any, ...] = ()
        if category is not None:
            q += " WHERE category = ?"
            args = (category.value,)
        for row in self._conn().execute(q, args):
            yield _row_to_contract(row)

    # ------- market snapshots -------------------------------------------
    def insert_snapshot(self, snap: MarketSnapshot, ob_yes_json: str | None = None,
                        ob_no_json: str | None = None) -> None:
        self._conn().execute(
            """
            INSERT OR IGNORE INTO market_snapshots (
                ticker, ts, yes_bid, yes_ask, no_bid, no_ask,
                last_price, last_trade_ts, volume, volume_24h, open_interest,
                liquidity, orderbook_yes, orderbook_no
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                snap.ticker, _iso(snap.ts),
                snap.yes_bid, snap.yes_ask, snap.no_bid, snap.no_ask,
                snap.last_price, _iso(snap.last_trade_ts),
                snap.volume, snap.volume_24h, snap.open_interest,
                snap.liquidity, ob_yes_json, ob_no_json,
            ),
        )

    def latest_snapshot(self, ticker: str) -> MarketSnapshot | None:
        row = self._conn().execute(
            "SELECT * FROM market_snapshots WHERE ticker = ? ORDER BY ts DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        return _row_to_snapshot(row) if row else None

    # ------- forecasts ---------------------------------------------------
    def insert_forecast(self, result: ForecastResult) -> int:
        binary_blob, binary_kind, binary_params = _dist_to_storage(result.binary_posterior)
        under_blob, under_kind, under_params = _dist_to_storage(result.underlying_posterior)
        q05 = float(result.binary_posterior.quantile(0.05)) if not result.is_null() else None
        q95 = float(result.binary_posterior.quantile(0.95)) if not result.is_null() else None

        cur = self._conn().execute(
            """
            INSERT OR REPLACE INTO forecasts (
                ticker, ts, forecaster, version,
                p_yes, p_yes_std, p_yes_p05, p_yes_p95,
                binary_posterior, binary_posterior_kind, binary_posterior_params,
                underlying_posterior, underlying_posterior_kind, underlying_posterior_params,
                uncertainty, model_confidence, methodology, data_sources, diagnostics,
                null_reason
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                result.ticker, _iso(result.ts), result.forecaster, result.version,
                None if result.is_null() else result.p_yes,
                None if result.is_null() else result.p_yes_std,
                q05, q95,
                binary_blob, binary_kind, binary_params,
                under_blob, under_kind, under_params,
                json.dumps(result.uncertainty),
                result.model_confidence,
                json.dumps(result.methodology, default=str),
                json.dumps({k: _iso(v) for k, v in result.data_sources.items()}),
                json.dumps(result.diagnostics, default=str),
                result.null_reason,
            ),
        )
        return int(cur.lastrowid or 0)

    def get_forecast(self, forecast_id: int) -> ForecastResult | None:
        row = self._conn().execute(
            "SELECT * FROM forecasts WHERE id = ?", (forecast_id,)
        ).fetchone()
        return _row_to_forecast(row) if row else None

    def latest_forecast(self, ticker: str, forecaster: str | None = None) -> ForecastResult | None:
        if forecaster:
            q = "SELECT * FROM forecasts WHERE ticker = ? AND forecaster = ? ORDER BY ts DESC LIMIT 1"
            args: tuple[Any, ...] = (ticker, forecaster)
        else:
            q = "SELECT * FROM forecasts WHERE ticker = ? ORDER BY ts DESC LIMIT 1"
            args = (ticker,)
        row = self._conn().execute(q, args).fetchone()
        return _row_to_forecast(row) if row else None

    # ------- resolutions -------------------------------------------------
    def upsert_resolution(self, res: Resolution) -> None:
        self._conn().execute(
            """
            INSERT INTO resolutions (ticker, resolved_at, outcome, settled_price)
            VALUES (?,?,?,?)
            ON CONFLICT(ticker) DO UPDATE SET
                resolved_at = excluded.resolved_at,
                outcome = excluded.outcome,
                settled_price = excluded.settled_price
            """,
            (res.ticker, _iso(res.resolved_at), res.outcome.value, res.settled_price),
        )

    # ------- calibration records ----------------------------------------
    def insert_calibration_record(
        self,
        *,
        forecast_id: int,
        ticker: str,
        forecaster: str,
        category: str | None,
        p_yes: float,
        outcome: int,
        resolved_at: datetime,
        brier: float,
        log_score: float,
    ) -> None:
        self._conn().execute(
            """
            INSERT OR REPLACE INTO calibration_records
                (forecast_id, ticker, forecaster, category, p_yes, outcome,
                 resolved_at, brier, log_score)
            VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (forecast_id, ticker, forecaster, category, p_yes, outcome,
             _iso(resolved_at), brier, log_score),
        )

    # ------- bulk execute for tests -------------------------------------
    def execute(self, q: str, args: Iterable[Any] = ()) -> sqlite3.Cursor:
        return self._conn().execute(q, tuple(args))


# ---- row-to-type adapters ----------------------------------------------
def _row_to_contract(row: sqlite3.Row) -> Contract:
    return Contract(
        ticker=row["ticker"],
        event_ticker=row["event_ticker"],
        series_ticker=row["series_ticker"],
        category=Category(row["category"]),
        title=row["title"],
        subtitle=row["subtitle"],
        status=MarketStatus(row["status"]),
        open_time=_parse_iso(row["open_time"]),
        close_time=_parse_iso(row["close_time"]),
        expiration_time=_parse_iso(row["expiration_time"]),
        rules_primary=row["rules_primary"],
        rules_secondary=row["rules_secondary"],
        resolution_criteria=json.loads(row["resolution_criteria"]) if row["resolution_criteria"] else {},
        raw=json.loads(row["raw"]) if row["raw"] else {},
    )


def _row_to_snapshot(row: sqlite3.Row) -> MarketSnapshot:
    return MarketSnapshot(
        ticker=row["ticker"],
        ts=_parse_iso(row["ts"]) or datetime.now(timezone.utc),
        yes_bid=row["yes_bid"],
        yes_ask=row["yes_ask"],
        no_bid=row["no_bid"],
        no_ask=row["no_ask"],
        last_price=row["last_price"],
        last_trade_ts=_parse_iso(row["last_trade_ts"]),
        volume=row["volume"] or 0.0,
        volume_24h=row["volume_24h"] or 0.0,
        open_interest=row["open_interest"] or 0.0,
        liquidity=row["liquidity"],
    )


def _row_to_forecast(row: sqlite3.Row) -> ForecastResult:
    binary = _storage_to_dist(
        row["binary_posterior"],
        row["binary_posterior_kind"],
        row["binary_posterior_params"],
    )
    # A null-reason forecast may have no posterior; synthesize a placeholder.
    if binary is None:
        binary = ForecastDistribution(
            kind="parametric", family="beta", params={"a": 1.0, "b": 1.0}
        )
    under = _storage_to_dist(
        row["underlying_posterior"],
        row["underlying_posterior_kind"],
        row["underlying_posterior_params"],
    )
    return ForecastResult(
        ticker=row["ticker"],
        ts=_parse_iso(row["ts"]) or datetime.now(timezone.utc),
        forecaster=row["forecaster"],
        version=row["version"],
        binary_posterior=binary,
        underlying_posterior=under,
        uncertainty=json.loads(row["uncertainty"]) if row["uncertainty"] else {},
        model_confidence=row["model_confidence"] or 0.0,
        methodology=json.loads(row["methodology"]) if row["methodology"] else {},
        data_sources={
            k: _parse_iso(v) or datetime.now(timezone.utc)
            for k, v in (json.loads(row["data_sources"]) if row["data_sources"] else {}).items()
        },
        diagnostics=json.loads(row["diagnostics"]) if row["diagnostics"] else {},
        null_reason=row["null_reason"],
    )
