"""Static HTML report generator.

Renders the current persisted universe (contracts + latest snapshots +
latest forecasts) as a single self-contained HTML page. Inline CSS + a
sliver of inline JS for the category-filter pills. No external assets.

Rows are sorted by *edge* (model p_yes − market mid, in points)
descending, so the most YES-underpriced contracts sit at the top and
the most YES-overpriced (best NO buys) at the bottom. Rows where the
forecaster abstained or no forecaster exists fall below all
forecasted rows, sorted by volume — they're not actionable but still
shown for universe transparency.
"""

from __future__ import annotations

import html
import sqlite3
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from kalshi_edge.storage import Database


@dataclass(frozen=True, slots=True)
class _Row:
    ticker: str
    event_ticker: str
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
    c.ticker, c.event_ticker, c.series_ticker, c.category, c.title, c.close_time,
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
            event_ticker=r["event_ticker"] or "",
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


# Bet-recommendation gate. Below these thresholds we print "—" rather
# than a YES/NO — the apparent edge is either too small to beat the
# spread, too uncertain, or the market itself is too thin/pinned to
# trade against. Tunable.
_BET_MIN_EDGE_PTS = 5.0       # edge in probability points (¢-equivalent)
_BET_MIN_CONF = 0.20          # model confidence floor
_BET_MAX_SPREAD_CENTS = 15.0  # matches the universe filter
_BET_MIN_VOL_24H = 100.0


def _is_liquid(r: _Row) -> bool:
    """True iff the quoted market is tradeable and its mid is informative.

    We reject markets pinned to 0¢ or 100¢ on either side: a bid of 0
    means "no one is bidding," an ask of 100 means "no one is offering",
    and either way the ``mid`` we compute is not a fair reflection of
    where size could actually change hands.
    """
    if r.yes_bid is None or r.yes_ask is None:
        return False
    if r.yes_bid <= 0 or r.yes_ask >= 100:
        return False
    if (r.yes_ask - r.yes_bid) > _BET_MAX_SPREAD_CENTS:
        return False
    if r.volume_24h < _BET_MIN_VOL_24H:
        return False
    return True


def _bet_recommendation(r: _Row) -> tuple[str, str, str]:
    """Return (label, css_class, reason) for the bet column.

    ``label`` is 'YES', 'NO', or '—'. ``reason`` is a tooltip fragment
    explaining why we didn't recommend (for '—' cases). The gate is the
    conjunction of:
      - a forecast (not abstained)
      - |edge| ≥ _BET_MIN_EDGE_PTS
      - confidence ≥ _BET_MIN_CONF
      - liquid quoted market (see _is_liquid)
    """
    if r.fc_p_yes is None:
        return "—", "neu", "no model forecast"
    mid = _mid_cents(r)
    if mid is None:
        return "—", "neu", "no quoted mid"
    edge = r.fc_p_yes * 100.0 - mid
    if abs(edge) < _BET_MIN_EDGE_PTS:
        return "—", "neu", f"edge |{edge:.1f}| < {_BET_MIN_EDGE_PTS}¢ threshold"
    conf = r.fc_model_confidence or 0.0
    if conf < _BET_MIN_CONF:
        return "—", "neu", f"model confidence {conf * 100:.0f}% < {_BET_MIN_CONF * 100:.0f}%"
    if not _is_liquid(r):
        return "—", "neu", (
            f"market illiquid: bid/ask {r.yes_bid}/{r.yes_ask}¢, "
            f"vol {int(r.volume_24h)}"
        )
    if edge > 0:
        return "YES", "pos", ""
    return "NO", "neg", ""


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
.bet { text-align: center; font-weight: 700; letter-spacing: 0.04em; font-size: 12px; }
.bet.pos { color: #0b0d10; background: #4ade80; }
.bet.neg { color: #fff; background: #dc2626; }
.bet.neu { color: var(--muted); font-weight: 400; }
.ci { color: var(--muted); font-size: 11px; }
.null { color: var(--muted); font-style: italic; font-size: 12px; }
.notice {
    background: #14181e; border: 1px solid var(--border); border-radius: 8px;
    padding: 12px 16px; color: var(--muted); font-size: 13px; margin-bottom: 24px;
}
.filters { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 14px; align-items: center; }
.filters .lbl { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: 0.04em; margin-right: 6px; }
.pill {
    background: var(--card); color: var(--fg); border: 1px solid var(--border);
    border-radius: 999px; padding: 5px 12px; font-size: 12px; cursor: pointer;
    font: inherit; font-size: 12px; line-height: 1; letter-spacing: 0.01em;
}
.pill:hover { border-color: var(--accent); }
.pill.active { background: var(--accent); color: #0b0d10; border-color: var(--accent); font-weight: 600; }
.pill .n { color: var(--muted); margin-left: 6px; font-variant-numeric: tabular-nums; }
.pill.active .n { color: #0b0d10; opacity: 0.7; }
.cat-tag {
    display: inline-block; font-size: 10px; text-transform: uppercase; letter-spacing: 0.04em;
    color: var(--muted); border: 1px solid var(--border); border-radius: 4px;
    padding: 1px 6px; margin-left: 6px; vertical-align: middle;
}
footer { color: var(--muted); font-size: 12px; margin-top: 48px; border-top: 1px solid var(--border); padding-top: 16px; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
"""


_INLINE_JS = """
(function(){
  var pills = document.querySelectorAll('.pill');
  var rows = document.querySelectorAll('tbody tr[data-cat]');
  function apply(cat){
    for (var i=0;i<rows.length;i++){
      var r = rows[i];
      r.style.display = (cat === 'all' || r.getAttribute('data-cat') === cat) ? '' : 'none';
    }
    for (var j=0;j<pills.length;j++){
      pills[j].classList.toggle('active', pills[j].getAttribute('data-cat') === cat);
    }
  }
  for (var k=0;k<pills.length;k++){
    pills[k].addEventListener('click', function(e){ apply(e.currentTarget.getAttribute('data-cat')); });
  }
})();
"""


def _kalshi_url(r: _Row) -> str:
    # Kalshi event pages live at /events/{event_ticker}; the ?market= param
    # pre-selects the specific threshold within the event.
    if not r.event_ticker:
        return "https://kalshi.com"
    return f"https://kalshi.com/events/{r.event_ticker}?market={r.ticker}"


def _mid_cents(r: _Row) -> float | None:
    if r.yes_bid is None or r.yes_ask is None:
        return None
    return (r.yes_bid + r.yes_ask) / 2.0


def _edge_points(r: _Row) -> float | None:
    mid = _mid_cents(r)
    if r.fc_p_yes is None or mid is None:
        return None
    return r.fc_p_yes * 100.0 - mid


def build_report_html(db: Database, *, now: datetime | None = None) -> str:
    """Render the persisted universe to a single self-contained HTML string."""
    now = now or datetime.now(timezone.utc)
    rows = _load_rows(db)

    total_mkts = len(rows)
    total_vol = sum(r.volume_24h for r in rows)
    cat_counts = Counter(r.category for r in rows)
    n_cats = len(cat_counts)
    fresh_ts = max((r.snapshot_ts for r in rows if r.snapshot_ts), default=None)
    n_forecasted = sum(1 for r in rows if r.fc_p_yes is not None)
    n_abstained = sum(1 for r in rows
                      if r.fc_null_reason is not None and r.fc_p_yes is None)

    # Primary sort key: forecasted rows first (by signed edge DESC), then
    # abstained/no-forecaster rows (by volume DESC). None edge sorts last.
    def sort_key(r: _Row) -> tuple[int, float, float]:
        e = _edge_points(r)
        if e is None:
            return (1, 0.0, -r.volume_24h)
        return (0, -e, -r.volume_24h)

    sorted_rows = sorted(rows, key=sort_key)

    parts: list[str] = []
    parts.append(f"<!doctype html><html lang=en><head><meta charset=utf-8>")
    parts.append("<meta name=viewport content='width=device-width,initial-scale=1'>")
    parts.append("<title>kalshi-edge — universe</title>")
    parts.append(f"<style>{_CSS}</style></head><body><div class=wrap>")

    parts.append("<h1>kalshi-edge — live universe</h1>")
    parts.append("<div class=sub>")
    parts.append("Contracts currently tracked by the forecasting stack, sorted by edge. ")
    parts.append(f"Snapshot rendered {html.escape(now.strftime('%Y-%m-%d %H:%M UTC'))}. ")
    if fresh_ts:
        parts.append(f"Latest market data {html.escape(fresh_ts.strftime('%Y-%m-%d %H:%M UTC'))}.")
    parts.append("</div>")

    parts.append("<div class=notice>")
    parts.append("<b>Market ¢</b> is the current YES mid-price in cents (0 = certain NO, 100 = certain YES). ")
    parts.append("<b>Fair ¢</b> is our model's estimate of the same thing — what the YES contract is <i>really</i> worth. ")
    parts.append("<b>Edge</b> is Fair − Market in cents. <b>Bet</b> says YES / NO / — based on the edge, our confidence, ")
    parts.append(f"and whether the market is liquid enough to trade (we suppress recommendations when |edge| &lt; {_BET_MIN_EDGE_PTS:.0f}¢, ")
    parts.append(f"confidence &lt; {_BET_MIN_CONF*100:.0f}%, spread &gt; {_BET_MAX_SPREAD_CENTS:.0f}¢, ")
    parts.append(f"vol &lt; {_BET_MIN_VOL_24H:.0f}, or either side is pinned to 0¢/100¢). ")
    parts.append("Rows the forecaster couldn't run on sink to the bottom.")
    parts.append("</div>")

    parts.append("<div class=tiles>")
    parts.append(f"<div class=tile><div class=lbl>markets</div><div class=val>{total_mkts:,}</div></div>")
    parts.append(f"<div class=tile><div class=lbl>categories</div><div class=val>{n_cats}</div></div>")
    parts.append(f"<div class=tile><div class=lbl>24h volume</div><div class=val>{int(total_vol):,}</div></div>")
    parts.append(f"<div class=tile><div class=lbl>forecasted</div><div class=val>{n_forecasted:,}</div></div>")
    parts.append(f"<div class=tile><div class=lbl>abstained</div><div class=val>{n_abstained:,}</div></div>")
    parts.append("</div>")

    # Filter pills: "all" + one per category (ordered by count DESC).
    parts.append("<div class=filters>")
    parts.append("<span class=lbl>filter</span>")
    parts.append(f"<button class='pill active' data-cat='all'>all<span class=n>{total_mkts}</span></button>")
    for cat, n in cat_counts.most_common():
        parts.append(
            f"<button class='pill' data-cat='{html.escape(cat)}'>"
            f"{html.escape(cat)}<span class=n>{n}</span></button>"
        )
    parts.append("</div>")

    parts.append("<table><thead><tr>")
    parts.append("<th>ticker</th><th>title</th>")
    parts.append("<th class=num>DTE</th>")
    parts.append("<th class=num>Market ¢</th>")
    parts.append("<th class=num>Fair ¢</th>")
    parts.append("<th class=num>Edge</th>")
    parts.append("<th>Bet</th>")
    parts.append("<th class=num>Conf</th>")
    parts.append("<th class=num>Spread</th>")
    parts.append("<th class=num>Vol 24h</th>")
    parts.append("</tr></thead><tbody>")

    for r in sorted_rows:
        dte = _dte_days(r.close_time, now)
        mid = _mid_cents(r)
        spread = None
        if r.yes_bid is not None and r.yes_ask is not None:
            spread = r.yes_ask - r.yes_bid
        cat_attr = html.escape(r.category)
        parts.append(f"<tr data-cat='{cat_attr}'>")
        parts.append(
            f"<td class=ticker>"
            f"<a href='{html.escape(_kalshi_url(r))}' target=_blank rel=noopener>"
            f"{html.escape(r.ticker)}</a>"
            f"<span class=cat-tag>{cat_attr}</span></td>"
        )
        parts.append(f"<td class=title title='{html.escape(r.title)}'>{html.escape(r.title)}</td>")
        parts.append(f"<td class=num>{_fmt_dte(dte)}</td>")
        parts.append(f"<td class=num>{_fmt_cents(mid)}</td>")
        if r.fc_p_yes is not None:
            fair_cents = r.fc_p_yes * 100.0
            ci = ""
            if r.fc_p05 is not None and r.fc_p95 is not None:
                ci = (
                    f"<div class=ci>"
                    f"[{r.fc_p05 * 100:.0f}¢, {r.fc_p95 * 100:.0f}¢]"
                    f"</div>"
                )
            parts.append(f"<td class=num>{fair_cents:.1f}¢{ci}</td>")
            edge_txt, edge_cls = _fmt_edge_points(r.fc_p_yes, mid)
            parts.append(f"<td class='num edge {edge_cls}'>{edge_txt}</td>")
            bet_label, bet_cls, bet_reason = _bet_recommendation(r)
            bet_attr = f" title='{html.escape(bet_reason)}'" if bet_reason else ""
            parts.append(
                f"<td class='bet {bet_cls}'{bet_attr}>{bet_label}</td>"
            )
            parts.append(f"<td class=num>{_fmt_prob(r.fc_model_confidence)}</td>")
        elif r.fc_null_reason:
            parts.append(
                "<td class=num colspan=4><span class=null>"
                f"abstained — {html.escape(r.fc_null_reason[:100])}</span></td>"
            )
        else:
            parts.append(
                "<td class=num colspan=4><span class=null>no forecaster</span></td>"
            )
        parts.append(f"<td class=num>{_fmt_cents(spread)}</td>")
        parts.append(f"<td class=num>{_fmt_int(r.volume_24h)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")

    parts.append(f"<script>{_INLINE_JS}</script>")
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
