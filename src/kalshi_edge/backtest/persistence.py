"""NDJSON corpus: dump a live SQLite DB out, load it back for analysis.

The four tables we archive per run:

* ``forecasts``        — every P(YES) the stack has ever produced
* ``market_snapshots`` — the market state the forecast saw
* ``resolutions``      — eventual outcomes (ticker → YES/NO/VOID)
* ``contracts_lite``   — (ticker, event_ticker, series_ticker, category,
                          title, close_time) — enough to join forecasts
                          back to a category without the ``raw`` JSON blob.

Partition key is a row-local timestamp so a replayer can rebuild history
out-of-order without re-bucketing::

    forecasts/2026/04/18.ndjson

``dump_tables_to_ndjson`` is append-only within the partition: rows whose
primary key is already on disk for the same day are skipped. That makes
re-runs on the same UTC date idempotent, which matters because GitHub
Actions may replay a cron manually.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kalshi_edge.storage.db import Database


# ----------------------------------------------------------------------
# Table specs
# ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _TableSpec:
    """How to dump one table.

    * ``name``    — NDJSON directory name (also a stable identifier used
                    by ``load_table``; not tied to the SQL table name so we
                    can dump a projection like ``contracts_lite``).
    * ``query``   — SELECT that yields the columns we care about.
    * ``ts_col``  — column whose ISO-8601 string value determines the
                    YYYY/MM/DD partition. Must be present in ``query``.
    * ``pk_cols`` — tuple of column names that identify a row. Used for
                    idempotent appends: if a row with the same PK already
                    sits in today's NDJSON file, we skip it.
    """
    name: str
    query: str
    ts_col: str
    pk_cols: tuple[str, ...]


_FORECASTS = _TableSpec(
    name="forecasts",
    query=(
        "SELECT id, ticker, ts, forecaster, version, p_yes, p_yes_std, "
        "p_yes_p05, p_yes_p95, model_confidence, null_reason, "
        "p_yes_quant_only, methodology, diagnostics "
        "FROM forecasts"
    ),
    ts_col="ts",
    pk_cols=("ticker", "ts", "forecaster", "version"),
)

_SNAPSHOTS = _TableSpec(
    name="market_snapshots",
    query=(
        "SELECT id, ticker, ts, yes_bid, yes_ask, no_bid, no_ask, "
        "last_price, volume, volume_24h, open_interest, liquidity "
        "FROM market_snapshots"
    ),
    ts_col="ts",
    pk_cols=("ticker", "ts"),
)

_RESOLUTIONS = _TableSpec(
    name="resolutions",
    query="SELECT ticker, resolved_at, outcome, settled_price FROM resolutions",
    ts_col="resolved_at",
    pk_cols=("ticker",),
)

# Lite projection of ``contracts``: enough to map ticker → category and
# close_time for the backtest joins, skipping the ``raw`` JSON which is
# large and changes noisily.
_CONTRACTS_LITE = _TableSpec(
    name="contracts_lite",
    query=(
        "SELECT ticker, event_ticker, series_ticker, category, title, "
        "subtitle, close_time, expiration_time, updated_at AS ts "
        "FROM contracts"
    ),
    ts_col="ts",
    pk_cols=("ticker",),
)


_SPECS: tuple[_TableSpec, ...] = (_FORECASTS, _SNAPSHOTS, _RESOLUTIONS, _CONTRACTS_LITE)


# ----------------------------------------------------------------------
# Dumpers
# ----------------------------------------------------------------------


def dump_tables_to_ndjson(
    db: Database,
    out_dir: Path,
    *,
    now: datetime | None = None,
) -> dict[str, int]:
    """Append every tracked table's rows to ``<out_dir>/<table>/YYYY/MM/DD.ndjson``.

    Rows whose primary key is already present in today's file for that
    table are skipped, so a re-run on the same date is a no-op. Returns a
    map ``{table_name: rows_written}``.
    """
    now = now or datetime.now(timezone.utc)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, int] = {}
    for spec in _SPECS:
        written[spec.name] = _dump_one(db, out_dir, spec, fallback_now=now)
    return written


def _dump_one(
    db: Database,
    out_dir: Path,
    spec: _TableSpec,
    *,
    fallback_now: datetime,
) -> int:
    """Stream one table out to day-partitioned NDJSON. Returns rows written."""
    cur = db.execute(spec.query)
    rows = [dict(r) for r in cur.fetchall()]
    if not rows:
        return 0

    # Group by partition day first so we open each file at most once.
    by_day: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for row in rows:
        day = _partition_key(row.get(spec.ts_col), fallback=fallback_now)
        by_day.setdefault(day, []).append(row)

    total = 0
    for (yyyy, mm, dd), day_rows in by_day.items():
        fpath = out_dir / spec.name / f"{yyyy:04d}" / f"{mm:02d}" / f"{dd:02d}.ndjson"
        fpath.parent.mkdir(parents=True, exist_ok=True)

        existing_pks = _read_pks(fpath, spec.pk_cols)
        new_rows = [
            r for r in day_rows if _pk_tuple(r, spec.pk_cols) not in existing_pks
        ]
        if not new_rows:
            continue

        with fpath.open("a", encoding="utf-8") as fh:
            for r in new_rows:
                fh.write(json.dumps(r, sort_keys=True, default=_json_default))
                fh.write("\n")
        total += len(new_rows)

    return total


def _partition_key(ts_val: Any, *, fallback: datetime) -> tuple[int, int, int]:
    """Parse an ISO string or datetime into a (y, m, d) partition key.

    Falls back to ``fallback`` (the current UTC time) if the column is
    null or unparseable — safer than dropping the row. Stored timestamps
    are always ISO-8601 UTC per ``_iso`` in the DB layer.
    """
    if isinstance(ts_val, datetime):
        dt = ts_val if ts_val.tzinfo else ts_val.replace(tzinfo=timezone.utc)
    elif isinstance(ts_val, str) and ts_val:
        try:
            dt = datetime.fromisoformat(ts_val.replace("Z", "+00:00"))
        except ValueError:
            dt = fallback
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = fallback
    dt = dt.astimezone(timezone.utc)
    return (dt.year, dt.month, dt.day)


def _pk_tuple(row: dict[str, Any], pk_cols: tuple[str, ...]) -> tuple[Any, ...]:
    return tuple(row.get(c) for c in pk_cols)


def _read_pks(fpath: Path, pk_cols: tuple[str, ...]) -> set[tuple[Any, ...]]:
    """Return the set of PKs already in a day file (empty if file missing)."""
    if not fpath.exists():
        return set()
    out: set[tuple[Any, ...]] = set()
    with fpath.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            out.add(_pk_tuple(obj, pk_cols))
    return out


def _json_default(o: Any) -> Any:
    if isinstance(o, datetime):
        return o.astimezone(timezone.utc).isoformat()
    raise TypeError(f"not JSON-serializable: {type(o).__name__}")


# ----------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------


def load_table(data_dir: Path, table: str) -> list[dict[str, Any]]:
    """Read every NDJSON partition for ``table`` into a list of dicts.

    Dedup is per-table: the row whose PK appears *last* in the walk order
    wins. That gives us last-write-wins semantics, which is the right
    policy for slowly-changing fields like a resolution that gets revised.
    """
    spec = _spec_by_name(table)
    by_pk: dict[tuple[Any, ...], dict[str, Any]] = {}
    root = Path(data_dir) / table
    if not root.exists():
        return []
    for fpath in sorted(root.rglob("*.ndjson")):
        for row in _iter_ndjson(fpath):
            by_pk[_pk_tuple(row, spec.pk_cols)] = row
    return list(by_pk.values())


def _iter_ndjson(fpath: Path) -> Iterator[dict[str, Any]]:
    with fpath.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if isinstance(obj, dict):
                yield obj


def _spec_by_name(name: str) -> _TableSpec:
    for s in _SPECS:
        if s.name == name:
            return s
    raise KeyError(f"unknown backtest table: {name!r}")


def known_tables() -> tuple[str, ...]:
    """Names of tables the persistence layer dumps."""
    return tuple(s.name for s in _SPECS)
