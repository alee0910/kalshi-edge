"""kalshi-edge entrypoint CLI.

Deliberately thin: the real logic lives in the library modules. This file
only wires click commands to those modules so we can iterate on the internals
without breaking user-facing invocations.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
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


# Repo root. Resolve from this file so invocations from any cwd work
# consistently (e.g. GitHub Actions, ad-hoc runs, tests).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_ANALYST_INPUTS = _REPO_ROOT / "analyst_inputs"
_DEFAULT_BRIEFS_DIR = _REPO_ROOT / "site" / "briefs"


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
@click.option("--quantamental/--no-quantamental", "quantamental", default=True,
              help="If True (default) and the category supports it, run the "
                   "quantamental forecaster instead of the pure-quant one so "
                   "fundamental attribution is persisted.")
@click.pass_obj
def forecast_cmd(cfg, category: str | None, ticker: str | None,
                 limit: int | None, quantamental: bool) -> None:  # type: ignore[no-untyped-def]
    """Run registered forecasters over persisted contracts; store results.

    When ``--quantamental`` is set (default), economics contracts are handled
    by the CPI quantamental forecaster wired with the latest fundamental
    inputs + loadings from the DB. The result carries both a quant-only and
    a quant+fundamental posterior, and its attribution log is persisted to
    ``forecast_attribution``.
    """
    log = get_logger("cli.forecast")
    db = Database(cfg.storage.sqlite_path, wal=cfg.storage.wal_mode)
    registry = build_default_registry()

    # Swap in the quantamental CPI forecaster when requested. Defer import so
    # a broken fundamental package doesn't take down pure-quant flows.
    qf_forecaster = None
    if quantamental:
        try:
            from kalshi_edge.forecasters.economics_qf import (
                EconomicsCPIQuantamentalForecaster,
                inputs_from_db_rows,
                loadings_from_db_rows,
            )

            inputs = inputs_from_db_rows(db.latest_fundamental_inputs(category="economics"))
            loadings = loadings_from_db_rows(db.latest_loadings())
            qf_forecaster = EconomicsCPIQuantamentalForecaster(
                inputs=inputs, loadings=loadings,
            )
            registry.register(qf_forecaster)
        except Exception as e:  # noqa: BLE001 — log and fall back to pure-quant
            log.warning("quantamental_init_failed", error=str(e))

    cat_filter = Category(category) if category else None
    stats = {"total": 0, "forecasted": 0, "null": 0, "no_forecaster": 0, "attribution": 0}

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
        forecast_id = db.insert_forecast(result)
        # Persist attribution if the forecaster produced it (quantamental path).
        attribution = getattr(result, "attribution", None) or {}
        if forecast_id and attribution:
            q_only_p = None
            q_only = getattr(result, "quant_only_binary_posterior", None)
            if q_only is not None:
                try:
                    q_only_p = float(q_only.mean())
                except Exception:  # noqa: BLE001
                    q_only_p = None
            db.insert_forecast_attribution(
                forecast_id=forecast_id,
                ticker=contract.ticker,
                forecaster=result.forecaster,
                quant_p_yes=q_only_p,
                adjusted_p_yes=result.p_yes if not result.is_null() else None,
                mean_shift_underlying=attribution.get("mean_shift_total"),
                attribution=attribution.get("attribution", []),
                dropped=attribution.get("dropped", []),
            )
            stats["attribution"] += 1
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
        f"null={stats['null']} no_forecaster={stats['no_forecaster']} "
        f"attribution={stats['attribution']}"
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


@main.command("pull-fundamental")
@click.option(
    "--analyst-dir",
    type=click.Path(path_type=Path),
    default=_DEFAULT_ANALYST_INPUTS,
    help="Root directory of analyst YAML inputs.",
)
@click.option(
    "--skip-automated/--no-skip-automated",
    default=False,
    help="Skip the FRED automated pulls (useful when only manual inputs "
         "have changed and FRED is down / quota-limited).",
)
@click.option(
    "--skip-manual/--no-skip-manual",
    default=False,
    help="Skip the analyst YAML loader (useful when only the automated "
         "pulls need to refresh).",
)
@click.option(
    "--strict/--no-strict",
    default=True,
    help="Fail loud on any malformed YAML input (recommended).",
)
@click.pass_obj
def pull_fundamental(
    cfg, analyst_dir: Path, skip_automated: bool, skip_manual: bool,
    strict: bool,
) -> None:  # type: ignore[no-untyped-def]
    """Fetch automated + analyst fundamental inputs and persist them."""
    log = get_logger("cli.pull_fundamental")
    db = Database(cfg.storage.sqlite_path, wal=cfg.storage.wal_mode)
    db.init_schema()

    n_auto = 0
    n_manual = 0

    if not skip_automated:
        from kalshi_edge.fundamental.automated import pull_cpi_fundamentals
        try:
            inputs = pull_cpi_fundamentals()
        except Exception as e:  # noqa: BLE001
            log.warning("pull_automated_failed", error=str(e))
            inputs = []
        for inp in inputs:
            db.insert_fundamental_input(
                name=inp.name, category=inp.category, value=inp.value,
                uncertainty=inp.uncertainty, mechanism=inp.mechanism.value,
                source=inp.provenance.source, source_kind=inp.provenance.source_kind,
                fetched_at=inp.provenance.fetched_at,
                observation_at=inp.provenance.observation_at,
                expires_at=inp.expires_at, scope=inp.scope,
                notes=inp.provenance.notes,
            )
            n_auto += 1

    if not skip_manual:
        from kalshi_edge.fundamental.manual import (
            ManualInputError,
            load_manual_inputs_from_dir,
        )
        try:
            inputs = load_manual_inputs_from_dir(analyst_dir, strict=strict)
        except ManualInputError as e:
            # In strict mode this is a hard fail so operators notice.
            raise click.ClickException(f"manual input load failed: {e}") from e
        for inp in inputs:
            db.insert_fundamental_input(
                name=inp.name, category=inp.category, value=inp.value,
                uncertainty=inp.uncertainty, mechanism=inp.mechanism.value,
                source=inp.provenance.source, source_kind=inp.provenance.source_kind,
                fetched_at=inp.provenance.fetched_at,
                observation_at=inp.provenance.observation_at,
                expires_at=inp.expires_at, scope=inp.scope,
                notes=inp.provenance.notes,
            )
            n_manual += 1

    click.echo(f"automated={n_auto} manual={n_manual}")


@main.command("calibrate-loadings")
@click.option(
    "--min-train", type=int, default=48,
    help="Minimum months of residuals before emitting a first calibration "
         "point. Lower means earlier history but less stable β.",
)
@click.option(
    "--max-years", type=float, default=25.0,
    help="Cap training window (years) to keep regime-drift bias bounded.",
)
@click.option(
    "--fred-api-key", default=None,
    help="FRED API key override. Defaults to FRED_API_KEY env var.",
)
@click.pass_obj
def calibrate_loadings(
    cfg, min_train: int, max_years: float, fred_api_key: str | None,
) -> None:  # type: ignore[no-untyped-def]
    """Refit CPI fundamental loadings from FRED history.

    Writes one row per input to ``fundamental_loadings``. Existing rows are
    retained; the latest ``fit_at`` wins at forecast time.
    """
    log = get_logger("cli.calibrate_loadings")
    import os
    key = fred_api_key or cfg.fred_api_key or os.environ.get("FRED_API_KEY")
    if not key:
        raise click.ClickException(
            "no FRED API key (set FRED_API_KEY or --fred-api-key)"
        )

    db = Database(cfg.storage.sqlite_path, wal=cfg.storage.wal_mode)
    db.init_schema()

    from kalshi_edge.data_sources.fred import FredClient
    from kalshi_edge.fundamental.calibration import fit_cpi_loadings

    client = FredClient(key)
    try:
        loadings = fit_cpi_loadings(
            client, min_train_months=min_train, max_years=int(max_years),
        )
    except Exception as e:  # noqa: BLE001
        raise click.ClickException(f"calibration failed: {e}") from e

    for ld in loadings:
        db.insert_fundamental_loading(
            name=ld.name, mechanism=ld.mechanism.value,
            beta=ld.beta, beta_std=ld.beta_std, baseline=ld.baseline,
            n_obs=ld.n_obs, fit_method=ld.fit_method, fit_at=ld.fit_at,
            diagnostics=getattr(ld, "diagnostics", None),
        )
        log.info(
            "calibrated",
            name=ld.name,
            beta=round(ld.beta, 4),
            beta_std=round(ld.beta_std, 4),
            n_obs=ld.n_obs,
        )
    click.echo(f"calibrated {len(loadings)} loadings")


@main.command("build-briefs")
@click.option(
    "--out", "out_dir", type=click.Path(path_type=Path),
    default=_DEFAULT_BRIEFS_DIR,
    help="Output directory for Markdown brief files.",
)
@click.option(
    "--category", type=click.Choice([c.value for c in Category]),
    default=None, help="Limit to a single category.",
)
@click.option(
    "--ticker", default=None, help="Generate a brief for one ticker only.",
)
@click.pass_obj
def build_briefs(
    cfg, out_dir: Path, category: str | None, ticker: str | None,
) -> None:  # type: ignore[no-untyped-def]
    """Render Markdown research briefs from persisted forecasts."""
    log = get_logger("cli.build_briefs")
    db = Database(cfg.storage.sqlite_path, wal=cfg.storage.wal_mode)

    from kalshi_edge.fundamental.briefs import BriefContext, write_briefs
    from kalshi_edge.forecasters.economics_qf import (
        inputs_from_db_rows,
        loadings_from_db_rows,
    )

    # Load inputs + loadings once; they're small and used by every brief.
    inputs_by_cat: dict[str, dict] = {}
    for cat in {c.value for c in Category}:
        try:
            rows = db.latest_fundamental_inputs(category=cat)
        except Exception:  # noqa: BLE001
            rows = {}
        if rows:
            inputs_by_cat[cat] = inputs_from_db_rows(rows)
    loadings = loadings_from_db_rows(db.latest_loadings())

    cat_filter = Category(category) if category else None
    contexts: list[BriefContext] = []
    for contract in db.iter_contracts(category=cat_filter):
        if ticker and contract.ticker != ticker:
            continue
        snap = db.latest_snapshot(contract.ticker)
        fc = db.latest_forecast(contract.ticker)
        if fc is None:
            log.debug("brief_skipped_no_forecast", ticker=contract.ticker)
            continue
        live_inputs = list(inputs_by_cat.get(contract.category.value, {}).values())
        contexts.append(BriefContext(
            contract=contract,
            snapshot=snap,
            forecast=fc,
            inputs=live_inputs,
            loadings=loadings,
            as_of=datetime.now(timezone.utc),
        ))

    written = write_briefs(contexts, out_dir)
    click.echo(f"wrote {len(written)} briefs to {out_dir}")
