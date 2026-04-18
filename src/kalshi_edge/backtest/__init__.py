"""Backtest corpus + analysis.

The live pipeline's SQLite DB is ephemeral (rebuilt in every CI runner),
so the backtest layer keeps its own append-only NDJSON archive on a
dedicated ``data-snapshots`` branch. Each cron run dumps new forecasts,
market snapshots, and resolutions; the ``backtest`` CLI later joins
them to produce calibration + hypothetical-PnL metrics.

The archive is organized on disk as::

    <root>/<table>/YYYY/MM/DD.ndjson

One line of JSON per row. Files are append-only; dedup is done on
load by a table-specific primary key.
"""
