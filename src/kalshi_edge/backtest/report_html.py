"""HTML report for the NDJSON-driven backtest.

Lives at ``site/quantamental/backtest.html`` so the quantamental subsite's
nav can link to it. Shares the main report's CSS to keep the look
consistent; the reliability curves are rendered as small inline SVGs so
no JS / external dependencies are needed.

This file is intentionally not dashboard-y. The goal is a static "what
did the forecasters do?" snapshot that updates every cron run.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kalshi_edge.backtest.analysis import ForecasterMetrics, compute_metrics, join_corpus
from kalshi_edge.backtest.retro_market import RetroSummary
from kalshi_edge.report import _CSS, _quantamental_page_shell


def build_backtest_html(
    data_dir: Path,
    *,
    now: datetime | None = None,
    edge_threshold_cents: float = 5.0,
) -> str:
    """Build the HTML body and wrap it in the quantamental subsite chrome."""
    now = now or datetime.now(timezone.utc)
    joined = join_corpus(data_dir)
    metrics = compute_metrics(joined, edge_threshold_cents=edge_threshold_cents)

    body: list[str] = []
    n_total = len(joined)
    n_resolved = sum(1 for j in joined if j.outcome is not None)
    body.append(
        "<div class=sub>"
        "Out-of-sample scoring over the NDJSON corpus of forecasts. "
        f"<b>{n_total:,}</b> forecasts observed, <b>{n_resolved:,}</b> "
        f"resolved. Edge-PnL threshold: <b>{edge_threshold_cents:.0f}¢</b>."
        "</div>"
    )

    if not metrics:
        body.append(
            "<div class=notice>No forecasts on disk yet. The corpus fills "
            "in as <code>dump-history</code> runs in CI and commits to "
            "the <code>data-snapshots</code> branch.</div>"
        )
        return _quantamental_page_shell("Backtest", "".join(body), now=now)

    # -- Per-(forecaster, category) summary table.
    body.append(
        "<h2 style='margin-top:28px'>Per-forecaster scoreboard</h2>"
        "<table><thead><tr>"
        "<th>forecaster</th><th>category</th>"
        "<th class=num>n</th><th class=num>resolved</th>"
        "<th class=num>Brier</th><th class=num>log-loss</th>"
        "<th class=num>edge bets</th><th class=num>edge PnL (¢)</th>"
        "</tr></thead><tbody>"
    )
    for m in metrics:
        body.append(_metrics_row_html(m))
    body.append("</tbody></table>")

    # -- Reliability curves. One small SVG per forecaster with enough resolved data.
    rely_cards: list[str] = []
    for m in metrics:
        if not m.reliability or m.n_resolved < 10:
            continue
        rely_cards.append(_reliability_card_html(m))
    if rely_cards:
        body.append("<h2 style='margin-top:28px'>Reliability curves</h2>")
        body.append(
            "<div style='display:flex;flex-wrap:wrap;gap:18px'>"
            + "".join(rely_cards)
            + "</div>"
        )

    return _quantamental_page_shell("Backtest", "".join(body), now=now)


def _metrics_row_html(m: ForecasterMetrics) -> str:
    return (
        "<tr>"
        f"<td><code>{html.escape(m.forecaster)}</code></td>"
        f"<td>{html.escape(m.category)}</td>"
        f"<td class=num>{m.n_forecasts}</td>"
        f"<td class=num>{m.n_resolved}</td>"
        f"<td class=num>{_fmt_score(m.brier)}</td>"
        f"<td class=num>{_fmt_score(m.log_loss)}</td>"
        f"<td class=num>{m.edge_bets}</td>"
        f"<td class='num edge {_pnl_cls(m.edge_pnl_cents)}'>"
        f"{m.edge_pnl_cents:+.1f}</td>"
        "</tr>"
    )


def _reliability_card_html(m: ForecasterMetrics) -> str:
    """Tiny inline SVG: identity line + bin dots sized by n."""
    w, h = 220, 220
    pad = 28
    inner_w = w - 2 * pad
    inner_h = h - 2 * pad

    def px(p: float) -> float:
        return pad + p * inner_w

    def py(p: float) -> float:
        return (h - pad) - p * inner_h

    parts = [
        f"<svg width='{w}' height='{h}' style='background:var(--row-alt);border:1px solid var(--border);border-radius:4px'>",
        # Identity line (y = x).
        f"<line x1='{px(0)}' y1='{py(0)}' x2='{px(1)}' y2='{py(1)}' "
        "stroke='var(--muted)' stroke-dasharray='4 4' />",
        # Axes.
        f"<line x1='{pad}' y1='{h - pad}' x2='{w - pad}' y2='{h - pad}' stroke='var(--border)' />",
        f"<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{h - pad}' stroke='var(--border)' />",
    ]
    total_n = sum(b["n"] for b in m.reliability) or 1
    for b in m.reliability:
        r = 2.0 + 6.0 * (b["n"] / total_n) ** 0.5
        cx, cy = px(b["mean_p"]), py(b["empirical_rate"])
        parts.append(
            f"<circle cx='{cx:.1f}' cy='{cy:.1f}' r='{r:.1f}' "
            "fill='var(--accent)' fill-opacity='0.75' />"
        )
    parts.append(
        f"<text x='{pad}' y='{pad - 8}' fill='var(--fg)' "
        f"font-size='12' font-family='ui-monospace,monospace'>"
        f"{html.escape(m.forecaster)} · {html.escape(m.category)} "
        f"(n={m.n_resolved})</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def _fmt_score(v: float | None) -> str:
    return "—" if v is None else f"{v:.4f}"


def _pnl_cls(v: float) -> str:
    if v > 0:
        return "pos"
    if v < 0:
        return "neg"
    return "neu"


def write_backtest_report(
    data_dir: Path,
    out_path: Path,
    *,
    now: datetime | None = None,
    edge_threshold_cents: float = 5.0,
) -> int:
    html_out = build_backtest_html(
        data_dir, now=now, edge_threshold_cents=edge_threshold_cents
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path.write_text(html_out, encoding="utf-8")


__all__ = ["build_backtest_html", "write_backtest_report"]


# ``_CSS`` is re-exported just to keep the import from being unused when
# we tweak the layout; the shell already embeds it.
_ = _CSS
