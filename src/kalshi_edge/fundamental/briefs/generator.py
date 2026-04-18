"""Deterministic Markdown research-brief generator.

A brief describes a single contract's state at one moment in time:

    1. Contract header (ticker, title, close time, status).
    2. Market snapshot (yes/no mid, spread, volume).
    3. Quant-only vs quant+fundamental posteriors on P(YES), with the
       underlying-variable posterior summary when available.
    4. Fundamental inputs feeding the forecast: value, uncertainty,
       provenance, age, expiry, and — if a loading has been fit — the
       posterior shift each one caused in the most recent integration.
    5. Calibrated loadings table: β (MLE), β (shrunk), σ_β, n_obs,
       fit_at, fit_method.
    6. Dropped-input log (why each skipped input was ignored).
    7. Data-source staleness.
    8. "What's missing" prompt: expected InputSpecs with no live input,
       so the analyst knows what manual entry would plug a gap.

The function is pure: same inputs → identical output bytes. Timestamps are
formatted as UTC ISO-8601 with minute precision to keep diffs readable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from kalshi_edge.forecast import ForecastResult
from kalshi_edge.fundamental.schemas.base import (
    FundamentalInput,
    InputLoading,
    InputSpec,
)
from kalshi_edge.fundamental.schemas.cpi import cpi_input_specs
from kalshi_edge.logging_ import get_logger
from kalshi_edge.types import Contract, MarketSnapshot

log = get_logger(__name__)


# Per-category spec providers (mirrors the manual loader registry).
_SPECS_BY_CATEGORY: dict[str, Any] = {
    "economics": cpi_input_specs,
}


# ---------------------------------------------------------------------------
# Context object passed to the renderer.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BriefContext:
    """Everything the renderer needs to produce a brief for one contract.

    Attribution data (mean shifts, dropped inputs) comes from
    ``forecast.attribution`` — we accept the serialized dict form so the
    renderer can work off DB rows directly.
    """

    contract: Contract
    snapshot: MarketSnapshot | None
    forecast: ForecastResult
    inputs: list[FundamentalInput] = field(default_factory=list)
    loadings: dict[str, InputLoading] = field(default_factory=dict)
    as_of: datetime | None = None                   # defaults to forecast.ts


# ---------------------------------------------------------------------------
# Public entry points.
# ---------------------------------------------------------------------------


def render_brief(ctx: BriefContext) -> str:
    """Render a single contract brief to a Markdown string.

    Never raises on missing data — degrades section-by-section.
    """
    as_of = ctx.as_of or ctx.forecast.ts
    lines: list[str] = []
    lines.extend(_header(ctx, as_of))
    lines.extend(_market_section(ctx))
    lines.extend(_forecast_section(ctx))
    lines.extend(_underlying_section(ctx))
    lines.extend(_fundamentals_section(ctx))
    lines.extend(_loadings_section(ctx))
    lines.extend(_attribution_section(ctx))
    lines.extend(_dropped_section(ctx))
    lines.extend(_missing_section(ctx))
    lines.extend(_sources_section(ctx, as_of))
    lines.extend(_methodology_section(ctx))
    return "\n".join(lines).rstrip() + "\n"


def write_briefs(
    contexts: Iterable[BriefContext],
    out_dir: Path | str,
) -> list[Path]:
    """Render each context and write ``<ticker>.md`` under ``out_dir``.

    Returns the list of written paths, sorted. Overwrites existing briefs
    (the intent is "latest snapshot wins"; history lives in git / the DB,
    not the filesystem).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for ctx in contexts:
        body = render_brief(ctx)
        p = out / f"{_safe_filename(ctx.contract.ticker)}.md"
        p.write_text(body)
        written.append(p)
    written.sort()
    log.info("briefs_written", n=len(written), out_dir=str(out))
    return written


# ---------------------------------------------------------------------------
# Section renderers. Each returns a list of lines (no trailing newline).
# ---------------------------------------------------------------------------


def _header(ctx: BriefContext, as_of: datetime) -> list[str]:
    c = ctx.contract
    title = c.title or c.ticker
    close = _fmt_dt(c.close_time) if c.close_time else "—"
    subtitle = f" — {c.subtitle}" if c.subtitle else ""
    return [
        f"# {title}{subtitle}",
        "",
        f"- **Ticker**: `{c.ticker}`",
        f"- **Event**: `{c.event_ticker}`  •  **Series**: `{c.series_ticker}`",
        f"- **Category**: {c.category.value}  •  **Status**: {c.status.value}",
        f"- **Close**: {close}",
        f"- **Brief as-of**: {_fmt_dt(as_of)}",
        "",
    ]


def _market_section(ctx: BriefContext) -> list[str]:
    snap = ctx.snapshot
    out = ["## Market snapshot", ""]
    if snap is None:
        out.append("No snapshot available.")
        out.append("")
        return out
    mid = snap.yes_mid
    spread = snap.yes_spread
    mid_s = _fmt_cents(mid) if mid is not None else "—"
    spread_s = _fmt_cents(spread) if spread is not None else "—"
    last = _fmt_cents(snap.last_price) if snap.last_price is not None else "—"
    lt = _fmt_dt(snap.last_trade_ts) if snap.last_trade_ts else "—"
    out.extend([
        f"- **YES mid**: {mid_s}  •  **spread**: {spread_s}  •  **last**: {last}",
        f"- **Volume**: {snap.volume:,.0f}  •  **24h**: {snap.volume_24h:,.0f}  "
        f"•  **OI**: {snap.open_interest:,.0f}",
        f"- **Last trade**: {lt}",
        "",
    ])
    return out


def _forecast_section(ctx: BriefContext) -> list[str]:
    fc = ctx.forecast
    out = ["## Forecast", ""]
    if fc.is_null():
        out.extend([
            f"Null forecast: `{fc.null_reason}`.",
            "",
            f"Forecaster: `{fc.forecaster}` v{fc.version} • Confidence: {fc.model_confidence:.2f}",
            "",
        ])
        return out

    p = fc.p_yes
    p_std = fc.p_yes_std
    # Bootstrap 80% CI on p_yes via the posterior quantiles.
    try:
        q10 = fc.p_yes_quantile(0.10)
        q90 = fc.p_yes_quantile(0.90)
        ci = f"[{q10:.1%}, {q90:.1%}]"
    except Exception:  # noqa: BLE001 — parametric without quantile support
        ci = "—"

    mkt_s = "—"
    edge_s = "—"
    if ctx.snapshot is not None and ctx.snapshot.yes_mid is not None:
        mkt = ctx.snapshot.yes_mid / 100.0
        mkt_s = f"{mkt:.1%}"
        edge_s = f"{(p - mkt) * 100:+.2f} pp"

    out.extend([
        f"- **Model P(YES)**: **{p:.1%}**  •  σ = {p_std:.3f}  •  80% CI {ci}",
        f"- **Market-implied P(YES)**: {mkt_s}  •  **Edge (model − market)**: {edge_s}",
        f"- **Model confidence**: {fc.model_confidence:.2f}",
    ])

    q_only = fc.quant_only_binary_posterior
    if q_only is not None:
        out.append(
            f"- **Quant-only P(YES)**: {q_only.mean():.1%}  "
            f"•  **Fundamental-driven shift**: "
            f"{(p - q_only.mean()) * 100:+.2f} pp"
        )

    unc = fc.uncertainty or {}
    if unc:
        out.append(
            "- **Uncertainty (nats)**: "
            f"total {unc.get('total', math.nan):.3f}, "
            f"aleatoric {unc.get('aleatoric', math.nan):.3f}, "
            f"epistemic {unc.get('epistemic', math.nan):.3f}"
        )
    out.append("")
    return out


def _underlying_section(ctx: BriefContext) -> list[str]:
    ul = ctx.forecast.underlying_posterior
    if ul is None:
        return []
    out = ["## Underlying posterior", ""]
    try:
        mean = ul.mean()
        std = ul.std()
        q10 = ul.quantile(0.10)
        q50 = ul.quantile(0.50)
        q90 = ul.quantile(0.90)
        out.append(
            f"- **Mean ± σ**: {mean:+.4f} ± {std:.4f}  "
            f"•  **10/50/90%**: {q10:+.4f} / {q50:+.4f} / {q90:+.4f}"
        )
    except Exception as e:  # noqa: BLE001 — degrade gracefully
        out.append(f"- Could not summarize underlying posterior: {e}")
    out.append("")
    return out


def _fundamentals_section(ctx: BriefContext) -> list[str]:
    out = ["## Fundamental inputs (live)", ""]
    if not ctx.inputs:
        out.extend(["_No live fundamental inputs on record._", ""])
        return out
    out.extend([
        "| Input | Value | σ | Source | Observed | Expires | Mechanism |",
        "|---|---|---|---|---|---|---|",
    ])
    for inp in sorted(ctx.inputs, key=lambda i: i.name):
        unc = "—" if inp.uncertainty is None else f"{inp.uncertainty:.4f}"
        out.append(
            f"| `{inp.name}` "
            f"| {inp.value:+.4f} "
            f"| {unc} "
            f"| {inp.provenance.source} "
            f"| {_fmt_dt(inp.provenance.observation_at)} "
            f"| {_fmt_dt(inp.expires_at)} "
            f"| {inp.mechanism.value} |"
        )
    out.append("")
    return out


def _loadings_section(ctx: BriefContext) -> list[str]:
    out = ["## Calibrated loadings", ""]
    if not ctx.loadings:
        out.extend(["_No calibrated loadings on record._", ""])
        return out
    out.extend([
        "| Name | β (MLE) | β (shrunk) | σ_β | baseline | n_obs | method | fit |",
        "|---|---|---|---|---|---|---|---|",
    ])
    for name in sorted(ctx.loadings):
        ld = ctx.loadings[name]
        out.append(
            f"| `{name}` "
            f"| {ld.beta:+.4f} "
            f"| {ld.shrunk_beta():+.4f} "
            f"| {ld.beta_std:.4f} "
            f"| {ld.baseline:+.4f} "
            f"| {ld.n_obs} "
            f"| {ld.fit_method} "
            f"| {_fmt_dt(ld.fit_at)} |"
        )
    out.append("")
    return out


def _attribution_section(ctx: BriefContext) -> list[str]:
    attr = (ctx.forecast.attribution or {}).get("attribution") or []
    if not attr:
        return []
    out = ["## Posterior shift attribution", ""]
    total = (ctx.forecast.attribution or {}).get("mean_shift_total")
    if total is not None:
        out.append(f"**Total mean shift (underlying)**: {total:+.4f}")
        out.append("")
    out.extend([
        "| Input | anomaly | β (shrunk) | Δ underlying | Δ P(YES) |",
        "|---|---|---|---|---|",
    ])
    for a in attr:
        ds_p = a.get("mean_shift_prob")
        ds_p_s = "—" if ds_p is None else f"{ds_p * 100:+.2f} pp"
        out.append(
            f"| `{a.get('name')}` "
            f"| {a.get('anomaly', 0.0):+.4f} "
            f"| {a.get('beta_shrunk', 0.0):+.4f} "
            f"| {a.get('mean_shift_underlying', 0.0):+.4f} "
            f"| {ds_p_s} |"
        )
    out.append("")
    return out


def _dropped_section(ctx: BriefContext) -> list[str]:
    dropped = (ctx.forecast.attribution or {}).get("dropped") or []
    if not dropped:
        return []
    out = ["## Dropped inputs", ""]
    for d in dropped:
        out.append(f"- `{d.get('name')}` — {d.get('reason')}")
    out.append("")
    return out


def _missing_section(ctx: BriefContext) -> list[str]:
    category = ctx.contract.category.value
    specs_fn = _SPECS_BY_CATEGORY.get(category)
    if specs_fn is None:
        return []
    specs: list[InputSpec] = specs_fn()
    have = {i.name for i in ctx.inputs}
    missing = [s for s in specs if s.name not in have]
    if not missing:
        return []
    out = ["## What's missing", ""]
    out.append(
        "The following inputs are declared for this category but not "
        "currently on record (consider a manual entry under "
        "`analyst_inputs/`):"
    )
    out.append("")
    for s in missing:
        out.append(
            f"- `{s.name}` — expects `{s.expected_source}` "
            f"({s.units}). {s.description}"
        )
    out.append("")
    return out


def _sources_section(ctx: BriefContext, as_of: datetime) -> list[str]:
    ds = ctx.forecast.data_sources or {}
    if not ds:
        return []
    out = ["## Data sources", ""]
    for name in sorted(ds):
        ts = ds[name]
        if isinstance(ts, datetime):
            age_h = (as_of - _ensure_utc(ts)).total_seconds() / 3600.0
            out.append(f"- `{name}` — last update {_fmt_dt(ts)} ({age_h:.1f}h ago)")
        else:
            out.append(f"- `{name}` — {ts}")
    out.append("")
    return out


def _methodology_section(ctx: BriefContext) -> list[str]:
    m = ctx.forecast.methodology or {}
    if not m:
        return []
    out = ["## Methodology", ""]
    for key in ("model", "prior", "likelihood", "notes"):
        v = m.get(key)
        if v:
            out.append(f"- **{key.capitalize()}**: {v}")
    assumptions = m.get("assumptions") or []
    if assumptions:
        out.append("- **Assumptions**:")
        for a in assumptions:
            out.append(f"  - {a}")
    sens = m.get("sensitivity") or {}
    if sens:
        out.append("- **Sensitivity**:")
        for k in sorted(sens):
            out.append(f"  - {k}: {sens[k]}")
    out.append("")
    return out


# ---------------------------------------------------------------------------
# Formatting helpers. Centralized so brief output stays byte-stable.
# ---------------------------------------------------------------------------


def _fmt_dt(dt: datetime | None) -> str:
    if dt is None:
        return "—"
    d = _ensure_utc(dt)
    return d.strftime("%Y-%m-%d %H:%MZ")


def _ensure_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _fmt_cents(v: float) -> str:
    return f"{v:.2f}¢"


def _safe_filename(ticker: str) -> str:
    # Kalshi tickers are ASCII + hyphens/slashes. Replace path separators only.
    return ticker.replace("/", "_").replace(" ", "_")
