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
        # Defensive migration for databases created before the quantamental
        # columns were added. SQLite < 3.35 doesn't support "ADD COLUMN IF
        # NOT EXISTS", so we inspect PRAGMA table_info and add any missing
        # columns one at a time. This is idempotent.
        self._migrate_add_columns(c, "forecasts", [
            ("p_yes_quant_only", "REAL"),
            ("p_yes_std_quant_only", "REAL"),
            ("binary_posterior_quant_only", "BLOB"),
            ("binary_posterior_quant_only_kind", "TEXT"),
            ("binary_posterior_quant_only_params", "TEXT"),
        ])
        self._migrate_add_columns(c, "calibration_records", [
            ("p_yes_quant_only", "REAL"),
            ("brier_quant_only", "REAL"),
            ("log_score_quant_only", "REAL"),
        ])
        c.execute(
            "INSERT OR IGNORE INTO schema_version (version) VALUES (?)",
            (_SCHEMA_VERSION,),
        )

    @staticmethod
    def _migrate_add_columns(
        conn: sqlite3.Connection, table: str, cols: list[tuple[str, str]]
    ) -> None:
        existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        for name, ddl in cols:
            if name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")

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

        # Quantamental split. For pure-quant forecasters these are None.
        qo = getattr(result, "quant_only_binary_posterior", None)
        qo_blob, qo_kind, qo_params = _dist_to_storage(qo)
        qo_p_yes = float(qo.mean()) if qo is not None else None
        qo_p_yes_std = float(qo.std()) if qo is not None else None

        cur = self._conn().execute(
            """
            INSERT OR REPLACE INTO forecasts (
                ticker, ts, forecaster, version,
                p_yes, p_yes_std, p_yes_p05, p_yes_p95,
                binary_posterior, binary_posterior_kind, binary_posterior_params,
                underlying_posterior, underlying_posterior_kind, underlying_posterior_params,
                uncertainty, model_confidence, methodology, data_sources, diagnostics,
                null_reason,
                p_yes_quant_only, p_yes_std_quant_only,
                binary_posterior_quant_only, binary_posterior_quant_only_kind,
                binary_posterior_quant_only_params
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                qo_p_yes, qo_p_yes_std,
                qo_blob, qo_kind, qo_params,
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

    # ------- fundamental inputs -----------------------------------------
    def insert_fundamental_input(
        self,
        *,
        name: str,
        category: str,
        value: float,
        uncertainty: float | None,
        mechanism: str,
        source: str,
        source_kind: str,
        fetched_at: datetime,
        observation_at: datetime,
        expires_at: datetime,
        scope: dict[str, Any] | None = None,
        notes: str | None = None,
    ) -> int:
        cur = self._conn().execute(
            """
            INSERT OR IGNORE INTO fundamental_inputs (
                name, category, value, uncertainty, mechanism,
                source, source_kind, fetched_at, observation_at, expires_at,
                scope, notes
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                name, category, float(value),
                None if uncertainty is None else float(uncertainty),
                mechanism, source, source_kind,
                _iso(fetched_at), _iso(observation_at), _iso(expires_at),
                json.dumps(scope or {}), notes,
            ),
        )
        return int(cur.lastrowid or 0)

    def latest_fundamental_inputs(
        self, *, category: str | None = None, as_of: datetime | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Return the most recent non-expired input per name.

        "Most recent" is by ``fetched_at`` DESC. If ``as_of`` is provided,
        only inputs fetched at or before that time are considered — useful
        for rerunning historical forecasts without leaking future data.
        """
        args: list[Any] = []
        q = """
            SELECT name, category, value, uncertainty, mechanism, source,
                   source_kind, fetched_at, observation_at, expires_at, scope, notes
            FROM fundamental_inputs
            WHERE 1=1
        """
        if category is not None:
            q += " AND category = ?"
            args.append(category)
        if as_of is not None:
            q += " AND fetched_at <= ?"
            args.append(_iso(as_of))
        q += " ORDER BY name, fetched_at DESC"

        out: dict[str, dict[str, Any]] = {}
        for r in self._conn().execute(q, tuple(args)):
            if r["name"] in out:
                continue  # take first (most recent) per name
            expires = _parse_iso(r["expires_at"])
            now = as_of or datetime.now(timezone.utc)
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            if expires and expires <= now:
                continue  # expired; skip (the integration engine would drop it anyway)
            out[r["name"]] = {
                "name": r["name"],
                "category": r["category"],
                "value": r["value"],
                "uncertainty": r["uncertainty"],
                "mechanism": r["mechanism"],
                "source": r["source"],
                "source_kind": r["source_kind"],
                "fetched_at": _parse_iso(r["fetched_at"]),
                "observation_at": _parse_iso(r["observation_at"]),
                "expires_at": expires,
                "scope": json.loads(r["scope"]) if r["scope"] else {},
                "notes": r["notes"],
            }
        return out

    # ------- fundamental loadings ---------------------------------------
    def insert_fundamental_loading(
        self,
        *,
        name: str,
        mechanism: str,
        beta: float,
        beta_std: float,
        baseline: float,
        n_obs: int,
        fit_method: str,
        fit_at: datetime,
        diagnostics: dict[str, Any] | None = None,
    ) -> int:
        cur = self._conn().execute(
            """
            INSERT OR REPLACE INTO fundamental_loadings (
                name, mechanism, beta, beta_std, baseline, n_obs,
                fit_method, fit_at, diagnostics
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            (
                name, mechanism, float(beta), float(beta_std), float(baseline),
                int(n_obs), fit_method, _iso(fit_at),
                json.dumps(diagnostics or {}, default=str),
            ),
        )
        return int(cur.lastrowid or 0)

    def latest_loadings(self) -> dict[str, dict[str, Any]]:
        """Return the latest loading per input name."""
        out: dict[str, dict[str, Any]] = {}
        for r in self._conn().execute(
            """
            SELECT name, mechanism, beta, beta_std, baseline, n_obs,
                   fit_method, fit_at, diagnostics
            FROM fundamental_loadings ORDER BY name, fit_at DESC
            """
        ):
            if r["name"] in out:
                continue
            out[r["name"]] = {
                "name": r["name"],
                "mechanism": r["mechanism"],
                "beta": r["beta"],
                "beta_std": r["beta_std"],
                "baseline": r["baseline"],
                "n_obs": r["n_obs"],
                "fit_method": r["fit_method"],
                "fit_at": _parse_iso(r["fit_at"]),
                "diagnostics": json.loads(r["diagnostics"]) if r["diagnostics"] else {},
            }
        return out

    # ------- forecast attribution ---------------------------------------
    def insert_forecast_attribution(
        self,
        *,
        forecast_id: int,
        ticker: str,
        forecaster: str,
        quant_p_yes: float | None,
        adjusted_p_yes: float | None,
        mean_shift_underlying: float | None,
        attribution: list[dict[str, Any]],
        dropped: list[dict[str, Any]],
    ) -> int:
        cur = self._conn().execute(
            """
            INSERT OR REPLACE INTO forecast_attribution (
                forecast_id, ticker, forecaster, quant_p_yes, adjusted_p_yes,
                mean_shift_underlying, attribution, dropped
            ) VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                forecast_id, ticker, forecaster,
                quant_p_yes, adjusted_p_yes, mean_shift_underlying,
                json.dumps(attribution, default=str),
                json.dumps(dropped, default=str),
            ),
        )
        return int(cur.lastrowid or 0)

    def get_attribution_for(self, forecast_id: int) -> dict[str, Any] | None:
        row = self._conn().execute(
            "SELECT * FROM forecast_attribution WHERE forecast_id = ?", (forecast_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "forecast_id": row["forecast_id"],
            "ticker": row["ticker"],
            "forecaster": row["forecaster"],
            "quant_p_yes": row["quant_p_yes"],
            "adjusted_p_yes": row["adjusted_p_yes"],
            "mean_shift_underlying": row["mean_shift_underlying"],
            "attribution": json.loads(row["attribution"]) if row["attribution"] else [],
            "dropped": json.loads(row["dropped"]) if row["dropped"] else [],
        }

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
    # Quant-only columns are new; rows from before the migration won't have
    # keys. Use ``.keys()`` to guard so old fixtures still round-trip.
    qo_kind = _row_col(row, "binary_posterior_quant_only_kind")
    qo_blob = _row_col(row, "binary_posterior_quant_only")
    qo_params = _row_col(row, "binary_posterior_quant_only_params")
    quant_only = _storage_to_dist(qo_blob, qo_kind, qo_params) if qo_kind else None
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
        quant_only_binary_posterior=quant_only,
    )


def _row_col(row: sqlite3.Row, name: str) -> Any:
    """Return row[name] or None if the column doesn't exist (pre-migration fixture)."""
    try:
        return row[name]
    except (IndexError, KeyError):
        return None
