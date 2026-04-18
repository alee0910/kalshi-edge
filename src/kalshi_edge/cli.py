"""kalshi-edge entrypoint CLI.

Deliberately thin: the real logic lives in the library modules. This file
only wires click commands to those modules so we can iterate on the internals
without breaking user-facing invocations.
"""

from __future__ import annotations

import json
from pathlib import Path

import click

from kalshi_edge.config import load_config
from kalshi_edge.forecasters import build_default_registry
from kalshi_edge.logging_ import configure, get_logger
from kalshi_edge.market import KalshiClient, UniverseFilter
from kalshi_edge.market.parser import UnsupportedMarket, parse_market
from kalshi_edge.report import write_report
from kalshi_edge.storage import Database
from kalshi_edge.types import Category


@click.group()
@click.option("--config", "config_path", type=click.Path(path_type=Path), default=None)
@click.pass_context
def main(ctx: click.Context, config_path: Path | None) -> None:
    cfg = load_config(config_path)
    configure(level=cfg.logging.level, renderer=cfg.logging.renderer)  # type: ignore[arg-type]
    ctx.obj = cfg


@main.command("init-db")
@click.pass_obj
def init_db(cfg) -> None:  # type: ignore[no-untyped-def]
    db = Database(cfg.storage.sqlite_path, wal=cfg.storage.wal_mode)
    db.init_schema()
    click.echo(f"initialized {cfg.storage.sqlite_path}")


@main.command("pull-universe")
@click.option("--status", default="open", help="Kalshi status filter.")
@click.option("--max-items", type=int, default=None)
@click.option("--series", default=None,
              help="Comma-separated series_ticker list. If set, bypass the full "
                   "scan and fetch only these series. Much faster because "
                   "Kalshi's open-markets feed is dominated by MV parlays.")
@click.option("--persist/--no-persist", default=True)
@click.option("--dump", type=click.Path(path_type=Path), default=None,
              help="Write kept markets to this JSON file.")
@click.pass_obj
def pull_universe(cfg, status: str, max_items: int | None, series: str | None,
                  persist: bool, dump: Path | None) -> None:  # type: ignore[no-untyped-def]
    """Fetch the live market universe, filter, and optionally persist/dump."""
    log = get_logger("cli.pull_universe")
    client = KalshiClient(cfg.kalshi)
    db = Database(cfg.storage.sqlite_path, wal=cfg.storage.wal_mode)
    if persist:
        db.init_schema()
    ufilter = UniverseFilter(cfg.universe_filter)

    if series:
        series_list = [s.strip() for s in series.split(",") if s.strip()]
        def market_source():
            for ser in series_list:
                yield from client.iter_markets(series_ticker=ser, status=status,
                                               max_items=max_items)
    else:
        def market_source():
            yield from client.iter_markets(status=status, max_items=max_items)

    pairs = []
    count = 0
    mv_skipped = 0
    parse_failed = 0
    for m in market_source():
        try:
            contract, snap = parse_market(m, cfg.categories)
        except UnsupportedMarket:
            mv_skipped += 1
            continue
        except Exception as e:  # noqa: BLE001
            parse_failed += 1
            log.warning("parse_failed", ticker=m.get("ticker"), error=str(e))
            continue
        pairs.append((contract, snap))
        count += 1

    kept, stats = ufilter.filter(pairs)

    click.echo(
        f"pulled={count} mv_skipped={mv_skipped} parse_failed={parse_failed} "
        f"kept={stats.kept} "
        f"drop[status={stats.dropped_status} cat={stats.dropped_category} "
        f"dte={stats.dropped_dte} vol={stats.dropped_volume} "
        f"spread={stats.dropped_spread} noprice={stats.dropped_missing_prices}]"
    )

    if persist:
        for fm in kept:
            db.upsert_contract(fm.contract)
            db.insert_snapshot(fm.snapshot)
        click.echo(f"persisted {len(kept)} contracts to {cfg.storage.sqlite_path}")

    if dump:
        dump.write_text(json.dumps([
            {
                "ticker": fm.contract.ticker,
                "category": fm.contract.category.value,
                "title": fm.contract.title,
                "close_time": fm.contract.close_time.isoformat() if fm.contract.close_time else None,
                "dte_days": round(fm.dte_days, 2),
                "yes_bid": fm.snapshot.yes_bid,
                "yes_ask": fm.snapshot.yes_ask,
                "spread_cents": fm.spread_cents,
                "volume_24h": fm.snapshot.volume_24h,
            }
            for fm in kept
        ], indent=2))
        click.echo(f"dumped {len(kept)} markets to {dump}")


@main.command("forecast")
@click.option("--category", type=click.Choice([c.value for c in Category]),
              default=None, help="Limit to a single category.")
@click.option("--ticker", default=None, help="Forecast a single ticker.")
@click.option("--limit", type=int, default=None,
              help="Stop after N contracts (useful for smoke tests).")
@click.pass_obj
def forecast_cmd(cfg, category: str | None, ticker: str | None,
                 limit: int | None) -> None:  # type: ignore[no-untyped-def]
    """Run registered forecasters over persisted contracts; store results."""
    log = get_logger("cli.forecast")
    db = Database(cfg.storage.sqlite_path, wal=cfg.storage.wal_mode)
    registry = build_default_registry()

    cat_filter = Category(category) if category else None
    stats = {"total": 0, "forecasted": 0, "null": 0, "no_forecaster": 0}

    for contract in db.iter_contracts(category=cat_filter):
        if ticker and contract.ticker != ticker:
            continue
        if limit and stats["total"] >= limit:
            break
        stats["total"] += 1
        fc = registry.get(contract)
        if fc is None:
            stats["no_forecaster"] += 1
            continue
        snap = db.latest_snapshot(contract.ticker)
        result = fc.forecast(contract, snap)
        db.insert_forecast(result)
        if result.is_null():
            stats["null"] += 1
            log.info("forecast_null", ticker=contract.ticker,
                     reason=result.null_reason)
        else:
            stats["forecasted"] += 1
            log.info("forecast_ok", ticker=contract.ticker,
                     p_yes=round(result.p_yes, 3),
                     model_confidence=round(result.model_confidence, 3))

    click.echo(
        f"total={stats['total']} forecasted={stats['forecasted']} "
        f"null={stats['null']} no_forecaster={stats['no_forecaster']}"
    )


@main.command("build-report")
@click.option("--out", "out_path", type=click.Path(path_type=Path),
              default=Path("site/index.html"),
              help="Output path for the static HTML report.")
@click.pass_obj
def build_report(cfg, out_path: Path) -> None:  # type: ignore[no-untyped-def]
    """Render the persisted universe as a static HTML page (no JS)."""
    db = Database(cfg.storage.sqlite_path, wal=cfg.storage.wal_mode)
    bytes_written = write_report(db, out_path)
    click.echo(f"wrote {bytes_written} bytes to {out_path}")
