"""Static HTML report generator.

Renders the current persisted universe (contracts + latest snapshots) as a
single self-contained HTML page. No JS, no external fonts, no external
assets — so the output works offline and is safe to host from any static
server (GitHub Pages, S3, a USB stick).

Intentional minimal surface: one public function, ``build_report_html``,
and a CLI wrapper in ``kalshi_edge.cli``. The scheduler / GitHub Actions
job pulls the universe first, then calls this.

Forecasts are not yet wired in. The report shows the *universe state*:
what markets we're watching, their prices, spreads, volumes. Once the
per-category forecasters land, the report gains edge columns (model p_yes,
edge vs. market, uncertainty band) — keyed off the same DB, so the
hosting pipeline doesn't change.
"""

from __future__ import annotations

import html
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kalshi_edge.storage import Database


@dataclass(frozen=True, slots=True)
class _Row:
    ticker: str
    series_ticker: str
    category: str
    title: str
    close_time: datetime | None
    yes_bid: float | None
    yes_ask: float | None
    volume_24h: float
    liquidity: float | None
    snapshot_ts: datetime | None
    # Forecast fields (None if no forecaster ran or the forecaster abstained).
    fc_p_yes: float | None
    fc_p05: float | None
    fc_p95: float | None
    fc_model_confidence: float | None
    fc_null_reason: str | None
    fc_forecaster: str | None
    fc_ts: datetime | None


_QUERY = """
SELECT
    c.ticker, c.series_ticker, c.category, c.title, c.close_time,
    s.yes_bid, s.yes_ask, s.volume_24h, s.liquidity, s.ts AS snapshot_ts,
    f.p_yes AS fc_p_yes, f.p_yes_p05 AS fc_p05, f.p_yes_p95 AS fc_p95,
    f.model_confidence AS fc_model_confidence,
    f.null_reason AS fc_null_reason,
    f.forecaster AS fc_forecaster,
    f.ts AS fc_ts
FROM contracts c
LEFT JOIN (
    SELECT ticker, MAX(ts) AS max_ts FROM market_snapshots GROUP BY ticker
) m ON m.ticker = c.ticker
LEFT JOIN market_snapshots s
    ON s.ticker = c.ticker AND s.ts = m.max_ts
LEFT JOIN (
    SELECT ticker, MAX(ts) AS max_ts FROM forecasts GROUP BY ticker
) fm ON fm.ticker = c.ticker
LEFT JOIN forecasts f
    ON f.ticker = c.ticker AND f.ts = fm.max_ts
ORDER BY c.category, s.volume_24h DESC NULLS LAST
"""


def _parse_iso(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace(" ", "T") if "T" not in s else s)
    except ValueError:
        return None


def _load_rows(db: Database) -> list[_Row]:
    cur: sqlite3.Cursor = db.execute(_QUERY)
    out: list[_Row] = []
    for r in cur.fetchall():
        out.append(_Row(
            ticker=r["ticker"],
            series_ticker=r["series_ticker"] or "",
            category=r["category"] or "other",
            title=r["title"] or "",
            close_time=_parse_iso(r["close_time"]),
            yes_bid=r["yes_bid"],
            yes_ask=r["yes_ask"],
            volume_24h=r["volume_24h"] or 0.0,
            liquidity=r["liquidity"],
            snapshot_ts=_parse_iso(r["snapshot_ts"]),
            fc_p_yes=r["fc_p_yes"],
            fc_p05=r["fc_p05"],
            fc_p95=r["fc_p95"],
            fc_model_confidence=r["fc_model_confidence"],
            fc_null_reason=r["fc_null_reason"],
            fc_forecaster=r["fc_forecaster"],
            fc_ts=_parse_iso(r["fc_ts"]),
        ))
    return out


def _dte_days(close: datetime | None, now: datetime) -> float | None:
    if close is None:
        return None
    return (close - now).total_seconds() / 86400.0


def _fmt_cents(v: float | None) -> str:
    return "—" if v is None else f"{v:.0f}¢"


def _fmt_int(v: float | None) -> str:
    return "—" if v is None else f"{int(round(v)):,}"


def _fmt_dte(v: float | None) -> str:
    if v is None:
        return "—"
    if v < 1:
        return f"{v * 24:.1f}h"
    return f"{v:.1f}d"


def _fmt_prob(v: float | None) -> str:
    return "—" if v is None else f"{v * 100:.1f}%"


def _fmt_edge_points(model_p: float | None, mid_cents: float | None) -> tuple[str, str]:
    """Return (text, css_class) for an edge cell. Edge = model - market, in pts.

    Positive edge means model thinks YES is underpriced (buy YES).
    Negative edge means YES is overpriced (buy NO).
    """
    if model_p is None or mid_cents is None:
        return "—", ""
    edge = model_p * 100.0 - mid_cents
    sign = "+" if edge > 0 else ""
    cls = "pos" if edge > 3 else ("neg" if edge < -3 else "neu")
    return f"{sign}{edge:.1f}", cls


_CSS = """
:root {
    --bg:#0b0d10; --fg:#e6e6e6; --muted:#9aa0a6; --accent:#7cc4ff;
    --border:#1e2329; --card:#111418; --row-alt:#0f1216;
}
* { box-sizing: border-box; }
body {
    background: var(--bg); color: var(--fg); margin: 0;
    font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
}
.wrap { max-width: 1200px; margin: 0 auto; padding: 32px 24px 80px; }
h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; letter-spacing: -0.01em; }
.sub { color: var(--muted); font-size: 13px; margin-bottom: 28px; }
.tiles { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 28px; }
.tile {
    background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 12px 16px; min-width: 120px;
}
.tile .lbl { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
.tile .val { font-size: 20px; font-weight: 600; margin-top: 2px; font-variant-numeric: tabular-nums; }
h2 { font-size: 15px; font-weight: 600; margin: 32px 0 8px; text-transform: capitalize; }
h2 .count { color: var(--muted); font-weight: 400; font-size: 12px; margin-left: 8px; }
table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 500; font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; }
td.num, th.num { text-align: right; }
tr:nth-child(even) td { background: var(--row-alt); }
.ticker { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: var(--accent); }
.title { max-width: 380px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.edge.pos { color: #4ade80; font-weight: 600; }
.edge.neg { color: #f87171; font-weight: 600; }
.edge.neu { color: var(--muted); }
.ci { color: var(--muted); font-size: 11px; }
.null { color: var(--muted); font-style: italic; font-size: 12px; }
.notice {
    background: #14181e; border: 1px solid var(--border); border-radius: 8px;
    padding: 12px 16px; color: var(--muted); font-size: 13px; margin-bottom: 24px;
}
footer { color: var(--muted); font-size: 12px; margin-top: 48px; border-top: 1px solid var(--border); padding-top: 16px; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
"""


def build_report_html(db: Database, *, now: datetime | None = None,
                      top_n_per_category: int = 50) -> str:
    """Render the persisted universe to a single self-contained HTML string."""
    now = now or datetime.now(timezone.utc)
    rows = _load_rows(db)

    by_cat: dict[str, list[_Row]] = defaultdict(list)
    for r in rows:
        by_cat[r.category].append(r)

    total_mkts = len(rows)
    total_vol = sum(r.volume_24h for r in rows)
    n_cats = len(by_cat)
    fresh_ts = max((r.snapshot_ts for r in rows if r.snapshot_ts), default=None)

    parts: list[str] = []
    parts.append(f"<!doctype html><html lang=en><head><meta charset=utf-8>")
    parts.append("<meta name=viewport content='width=device-width,initial-scale=1'>")
    parts.append("<title>kalshi-edge — universe</title>")
    parts.append(f"<style>{_CSS}</style></head><body><div class=wrap>")

    parts.append("<h1>kalshi-edge — live universe</h1>")
    parts.append("<div class=sub>")
    parts.append("Contracts currently tracked by the forecasting stack. ")
    parts.append(f"Snapshot rendered {html.escape(now.strftime('%Y-%m-%d %H:%M UTC'))}. ")
    if fresh_ts:
        parts.append(f"Latest market data {html.escape(fresh_ts.strftime('%Y-%m-%d %H:%M UTC'))}.")
    parts.append("</div>")

    n_forecasted = sum(1 for r in rows if r.fc_p_yes is not None)
    n_abstained = sum(1 for r in rows
                       if r.fc_null_reason is not None and r.fc_p_yes is None)
    parts.append("<div class=notice>")
    parts.append("<b>model p_yes</b> is this system's Bayesian posterior for YES; ")
    parts.append("<b>edge</b> is model − market (in points). ")
    parts.append("A <b>[p05, p95]</b> band on model p_yes shows posterior uncertainty. ")
    parts.append("Categories without a forecaster, or markets where the forecaster ")
    parts.append("abstained (no fake precision) show no model row.")
    parts.append("</div>")

    parts.append("<div class=tiles>")
    parts.append(f"<div class=tile><div class=lbl>markets</div><div class=val>{total_mkts:,}</div></div>")
    parts.append(f"<div class=tile><div class=lbl>categories</div><div class=val>{n_cats}</div></div>")
    parts.append(f"<div class=tile><div class=lbl>24h volume</div><div class=val>{int(total_vol):,}</div></div>")
    parts.append(f"<div class=tile><div class=lbl>forecasted</div><div class=val>{n_forecasted:,}</div></div>")
    parts.append(f"<div class=tile><div class=lbl>abstained</div><div class=val>{n_abstained:,}</div></div>")
    parts.append("</div>")

    for cat in sorted(by_cat):
        cat_rows = sorted(by_cat[cat], key=lambda r: r.volume_24h, reverse=True)
        shown = cat_rows[:top_n_per_category]
        more = len(cat_rows) - len(shown)
        more_txt = f" (top {len(shown)} of {len(cat_rows)})" if more > 0 else f" ({len(cat_rows)})"
        parts.append(f"<h2>{html.escape(cat)}<span class=count>{html.escape(more_txt)}</span></h2>")
        parts.append("<table><thead><tr>")
        parts.append("<th>ticker</th><th>title</th>")
        parts.append("<th class=num>DTE</th>")
        parts.append("<th class=num>mid</th><th class=num>spread</th>")
        parts.append("<th class=num>vol 24h</th>")
        parts.append("<th class=num>model p</th>")
        parts.append("<th class=num>edge (pts)</th>")
        parts.append("<th class=num>conf</th>")
        parts.append("</tr></thead><tbody>")
        for r in shown:
            dte = _dte_days(r.close_time, now)
            mid = None
            spread = None
            if r.yes_bid is not None and r.yes_ask is not None:
                mid = (r.yes_bid + r.yes_ask) / 2
                spread = r.yes_ask - r.yes_bid
            parts.append("<tr>")
            parts.append(f"<td class=ticker>{html.escape(r.ticker)}</td>")
            parts.append(f"<td class=title title='{html.escape(r.title)}'>{html.escape(r.title)}</td>")
            parts.append(f"<td class=num>{_fmt_dte(dte)}</td>")
            parts.append(f"<td class=num>{_fmt_cents(mid)}</td>")
            parts.append(f"<td class=num>{_fmt_cents(spread)}</td>")
            parts.append(f"<td class=num>{_fmt_int(r.volume_24h)}</td>")
            if r.fc_p_yes is not None:
                ci = ""
                if r.fc_p05 is not None and r.fc_p95 is not None:
                    ci = f"<div class=ci>[{_fmt_prob(r.fc_p05)}, {_fmt_prob(r.fc_p95)}]</div>"
                parts.append(f"<td class=num>{_fmt_prob(r.fc_p_yes)}{ci}</td>")
                edge_txt, edge_cls = _fmt_edge_points(r.fc_p_yes, mid)
                parts.append(f"<td class='num edge {edge_cls}'>{edge_txt}</td>")
                parts.append(f"<td class=num>{_fmt_prob(r.fc_model_confidence)}</td>")
            elif r.fc_null_reason:
                parts.append("<td class=num colspan=3><span class=null>"
                             f"abstained — {html.escape(r.fc_null_reason[:60])}</span></td>")
            else:
                parts.append("<td class=num colspan=3><span class=null>no forecaster</span></td>")
            parts.append("</tr>")
        parts.append("</tbody></table>")

    parts.append("<footer>")
    parts.append("kalshi-edge · research-grade Kalshi mispricing detector · ")
    parts.append("price data via <a href='https://kalshi.com'>Kalshi public API</a>. ")
    parts.append("Not financial advice. No auto-trading. ")
    parts.append(f"Rebuilt at {html.escape(now.strftime('%Y-%m-%d %H:%M UTC'))}.")
    parts.append("</footer></div></body></html>")

    return "".join(parts)


def write_report(db: Database, out_path: Path, *, now: datetime | None = None) -> int:
    """Render and write the report. Returns bytes written."""
    html_out = build_report_html(db, now=now)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path.write_text(html_out, encoding="utf-8")
