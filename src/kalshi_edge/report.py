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
import json
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kalshi_edge.storage import Database


@dataclass(frozen=True, slots=True)
class _Row:
    ticker: str
    event_ticker: str
    series_ticker: str
    category: str
    title: str
    subtitle: str | None   # Kalshi's yes_sub_title — labels what YES resolves on
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
    fc_version: str | None
    fc_ts: datetime | None
    fc_methodology: dict[str, Any] = field(default_factory=dict)
    fc_data_sources: dict[str, Any] = field(default_factory=dict)
    fc_diagnostics: dict[str, Any] = field(default_factory=dict)


_QUERY = """
SELECT
    c.ticker, c.event_ticker, c.series_ticker, c.category, c.title, c.subtitle, c.close_time,
    s.yes_bid, s.yes_ask, s.volume_24h, s.liquidity, s.ts AS snapshot_ts,
    f.p_yes AS fc_p_yes, f.p_yes_p05 AS fc_p05, f.p_yes_p95 AS fc_p95,
    f.model_confidence AS fc_model_confidence,
    f.null_reason AS fc_null_reason,
    f.forecaster AS fc_forecaster,
    f.version AS fc_version,
    f.ts AS fc_ts,
    f.methodology AS fc_methodology,
    f.data_sources AS fc_data_sources,
    f.diagnostics AS fc_diagnostics
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
            subtitle=r["subtitle"] or None,
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
            fc_version=r["fc_version"],
            fc_ts=_parse_iso(r["fc_ts"]),
            fc_methodology=_parse_json(r["fc_methodology"]),
            fc_data_sources=_parse_json(r["fc_data_sources"]),
            fc_diagnostics=_parse_json(r["fc_diagnostics"]),
        ))
    return out


def _parse_json(s: str | None) -> dict[str, Any]:
    if not s:
        return {}
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else {}
    except (ValueError, TypeError):
        return {}


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
    # Sanity gate: refuse to recommend on extreme disagreements against a
    # liquid market until a human has eyeballed the audit panel. Such edges
    # are more often a sign-flip bug than a real mispricing.
    if abs(edge) >= 40.0:
        return "—", "neu", (
            f"model disagrees with liquid market by {edge:+.0f}¢ — "
            f"review the audit panel before trading"
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
th.sortable { cursor: pointer; user-select: none; }
th.sortable:hover { color: var(--fg); }
th.sortable .arr { color: var(--border); margin-left: 4px; font-size: 9px; display: inline-block; width: 8px; }
th.sortable.sorted .arr { color: var(--accent); }
td.num, th.num { text-align: right; }
tr:nth-child(even) td { background: var(--row-alt); }
.ticker { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; color: var(--accent); }
.title { max-width: 380px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.yes-side { color: var(--muted); font-size: 11px; }
.edge.pos { color: #4ade80; font-weight: 600; }
.edge.neg { color: #f87171; font-weight: 600; }
.edge.neu { color: var(--muted); }
.edge-click { cursor: pointer; user-select: none; }
.edge-click:hover { background: #1a1f26; }
.edge-caret { color: var(--muted); font-size: 10px; margin-left: 2px; transition: transform 0.15s; display: inline-block; }
.edge-click.open .edge-caret { transform: rotate(90deg); }
.detail-row td {
    background: #0a0c0f !important;
    padding: 0 !important;
    border-bottom: 1px solid var(--border);
}
.detail-card {
    padding: 16px 20px 20px;
    border-left: 3px solid var(--accent);
    font-size: 12.5px;
    line-height: 1.55;
}
.detail-meta {
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 12px;
}
.detail-meta b { color: var(--fg); font-weight: 600; text-transform: none; letter-spacing: 0; }
.detail-section { margin-top: 12px; }
.detail-h {
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 6px;
    font-weight: 600;
}
.detail-empty { color: var(--muted); font-style: italic; font-size: 12px; }
.detail-warn {
    background: #2a1510; border: 1px solid #7a2010; border-left: 3px solid #f59e0b;
    color: #fbbf24; padding: 8px 12px; margin: 10px 0 4px;
    border-radius: 4px; font-size: 12px; line-height: 1.5;
}
.detail-warn b { color: #fde68a; }
.kv {
    display: grid;
    grid-template-columns: 220px 1fr;
    gap: 8px 16px;
    padding: 2px 0;
    border-bottom: 1px dashed #171b22;
}
.kv:last-child { border-bottom: none; }
.kv .k { color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 11.5px; }
.kv .v { color: var(--fg); word-break: break-word; }
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
  var activeCat = 'all';
  function mainRows(){ return document.querySelectorAll('tbody tr.main-row'); }
  function allRows(){ return document.querySelectorAll('tbody tr[data-cat]'); }
  function applyFilter(cat){
    activeCat = cat;
    var rows = allRows();
    for (var i=0;i<rows.length;i++){
      var r = rows[i];
      var catMatch = (cat === 'all' || r.getAttribute('data-cat') === cat);
      if (r.classList.contains('detail-row')) {
        r.style.display = (catMatch && !r.hasAttribute('hidden')) ? '' : 'none';
      } else {
        r.style.display = catMatch ? '' : 'none';
      }
    }
    for (var j=0;j<pills.length;j++){
      pills[j].classList.toggle('active', pills[j].getAttribute('data-cat') === cat);
    }
  }
  for (var k=0;k<pills.length;k++){
    pills[k].addEventListener('click', function(e){ applyFilter(e.currentTarget.getAttribute('data-cat')); });
  }
  // Toggle edge-detail rows on click.
  var edges = document.querySelectorAll('.edge-click');
  for (var n=0;n<edges.length;n++){
    edges[n].addEventListener('click', function(e){
      var id = e.currentTarget.getAttribute('data-detail');
      var d = document.getElementById(id);
      if (!d) return;
      var wasHidden = d.hasAttribute('hidden');
      if (wasHidden) {
        d.removeAttribute('hidden');
        e.currentTarget.classList.add('open');
      } else {
        d.setAttribute('hidden', '');
        e.currentTarget.classList.remove('open');
      }
      var catMatch = (activeCat === 'all' || d.getAttribute('data-cat') === activeCat);
      d.style.display = (catMatch && !d.hasAttribute('hidden')) ? '' : 'none';
    });
  }

  // Column sort. Clicking a sortable <th> sorts main rows by that column.
  // Missing values (abstained / no forecaster) always sink to the bottom.
  // The detail row is re-attached immediately after its parent so the
  // audit panel stays paired with its main row.
  var heads = document.querySelectorAll('th.sortable');
  var sortState = { key: null, dir: 'desc' };
  function readVal(row, key, type){
    var cell = row.querySelector("td[data-key='" + key + "']");
    if (!cell) return null;
    var raw = cell.getAttribute('data-s');
    if (raw === null || raw === '' || raw === 'NaN') return null;
    if (type === 'num') {
      var n = parseFloat(raw);
      return isNaN(n) ? null : n;
    }
    return raw.toLowerCase();
  }
  function sortBy(key, type){
    var dir = (sortState.key === key && sortState.dir === 'desc') ? 'asc' : 'desc';
    sortState = { key: key, dir: dir };
    var tbody = document.querySelector('tbody');
    var rows = Array.prototype.slice.call(mainRows());
    rows.sort(function(a, b){
      var va = readVal(a, key, type);
      var vb = readVal(b, key, type);
      var aNull = (va === null), bNull = (vb === null);
      if (aNull && bNull) return 0;
      if (aNull) return 1;  // nulls sink to bottom always
      if (bNull) return -1;
      if (va < vb) return dir === 'asc' ? -1 : 1;
      if (va > vb) return dir === 'asc' ? 1 : -1;
      return 0;
    });
    for (var i=0;i<rows.length;i++){
      var r = rows[i];
      tbody.appendChild(r);
      var detailId = r.getAttribute('data-detail-id');
      if (detailId) {
        var d = document.getElementById(detailId);
        if (d) tbody.appendChild(d);
      }
    }
    for (var j=0;j<heads.length;j++){
      var h = heads[j];
      var isActive = (h.getAttribute('data-key') === key);
      h.classList.toggle('sorted', isActive);
      var arr = h.querySelector('.arr');
      if (arr) arr.textContent = isActive ? (dir === 'asc' ? '\u25B2' : '\u25BC') : '';
    }
    applyFilter(activeCat);  // re-apply visibility
  }
  for (var m=0;m<heads.length;m++){
    (function(h){
      h.addEventListener('click', function(){
        sortBy(h.getAttribute('data-key'), h.getAttribute('data-type') || 'alpha');
      });
    })(heads[m]);
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


def _fmt_diag_value(v: Any) -> str:
    if isinstance(v, float):
        if abs(v) >= 1000 or (v != 0 and abs(v) < 1e-3):
            return f"{v:.4g}"
        return f"{v:.4f}".rstrip("0").rstrip(".")
    if isinstance(v, (list, tuple)):
        return ", ".join(_fmt_diag_value(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, default=str)
    return str(v)


def _kv_block(d: dict[str, Any]) -> str:
    if not d:
        return "<div class=detail-empty>(none)</div>"
    rows: list[str] = []
    for k, v in d.items():
        rows.append(
            f"<div class=kv>"
            f"<span class=k>{html.escape(str(k))}</span>"
            f"<span class=v>{html.escape(_fmt_diag_value(v))}</span>"
            f"</div>"
        )
    return "".join(rows)


def _sanity_warnings(r: _Row, mid: float | None, edge_pts: float | None) -> list[str]:
    """Heuristic sanity flags for extreme model-vs-market disagreement.

    These aren't errors — the model *can* beat a mispriced market — but
    large disagreements with a liquid market are the smoke signal for
    parser bugs (wrong comparator direction, wrong transform, etc.). We
    surface them in the detail panel so the user can eyeball before
    trusting the number. Tuned to fire rarely: both the edge and the
    market must be extreme.
    """
    out: list[str] = []
    if r.fc_p_yes is None or mid is None or edge_pts is None:
        return out
    fair_pts = r.fc_p_yes * 100.0
    # A: model near-certain (>97% or <3%) but market strongly disagrees
    # AND the market is liquid enough to take seriously.
    if _is_liquid(r) and abs(edge_pts) >= 40.0:
        out.append(
            f"Model ({fair_pts:.1f}¢) disagrees with a liquid market ({mid:.1f}¢) "
            f"by {edge_pts:+.1f}¢. In practice this is almost always a parser "
            f"sign-flip or unit-mismatch bug — verify the contract's YES "
            f"direction and the model's underlying transform in the "
            f"Diagnostics section below before trusting this edge."
        )
    # B: market pinned at an extreme (<3¢ or >97¢) but the model is moderate.
    # This can be real (thin liquidity at tails) but is worth a look.
    if mid is not None and (mid <= 3.0 or mid >= 97.0) and 10.0 <= fair_pts <= 90.0:
        out.append(
            f"Market is pinned near {'0¢' if mid <= 3 else '100¢'} but our "
            f"model gives {fair_pts:.1f}¢. If the market has meaningful "
            f"depth this is suspicious — check whether the model's "
            f"threshold/transform matches the contract subtitle."
        )
    return out


def _build_detail_html(r: _Row, mid: float | None) -> str:
    """Render the audit panel that explains how this row's edge was produced."""
    fair_pts = r.fc_p_yes * 100.0 if r.fc_p_yes is not None else None
    edge_pts = _edge_points(r)
    bet_label, _, bet_reason = _bet_recommendation(r)

    spread = None
    if r.yes_bid is not None and r.yes_ask is not None:
        spread = r.yes_ask - r.yes_bid
    ci = ""
    if r.fc_p05 is not None and r.fc_p95 is not None:
        ci = f"  [CI: {r.fc_p05 * 100:.1f}¢ – {r.fc_p95 * 100:.1f}¢]"

    pricing_lines: list[str] = []
    pricing_lines.append(
        f"<div class=kv><span class=k>market mid</span>"
        f"<span class=v>{_fmt_cents(mid)}"
        f" (bid {_fmt_cents(r.yes_bid)} / ask {_fmt_cents(r.yes_ask)}, "
        f"spread {_fmt_cents(spread)})</span></div>"
    )
    if fair_pts is not None:
        pricing_lines.append(
            f"<div class=kv><span class=k>model fair</span>"
            f"<span class=v>{fair_pts:.2f}¢{html.escape(ci)}</span></div>"
        )
    if edge_pts is not None:
        sign = "+" if edge_pts > 0 else ""
        pricing_lines.append(
            f"<div class=kv><span class=k>edge</span>"
            f"<span class=v>{sign}{edge_pts:.2f}¢ "
            f"(= model fair − market mid)</span></div>"
        )
    conf = r.fc_model_confidence
    conf_txt = f"{conf * 100:.1f}%" if conf is not None else "—"
    rec_txt = f"{bet_label}"
    if bet_label == "—" and bet_reason:
        rec_txt += f" — {bet_reason}"
    pricing_lines.append(
        f"<div class=kv><span class=k>recommendation</span>"
        f"<span class=v>{html.escape(rec_txt)} (confidence {conf_txt})</span></div>"
    )
    if r.subtitle:
        pricing_lines.append(
            f"<div class=kv><span class=k>YES resolves on</span>"
            f"<span class=v>{html.escape(r.subtitle)}</span></div>"
        )

    fc_ts_txt = (
        r.fc_ts.strftime("%Y-%m-%d %H:%M UTC") if r.fc_ts else "—"
    )
    meta = (
        f"<div class=detail-meta>"
        f"forecaster <b>{html.escape(r.fc_forecaster or '—')}</b>"
        f" (v{html.escape(r.fc_version or '—')}) · "
        f"ran {html.escape(fc_ts_txt)}"
        f"</div>"
    )

    warn_blocks = [
        f"<div class=detail-warn><b>Heads up:</b> {html.escape(w)}</div>"
        for w in _sanity_warnings(r, mid, edge_pts)
    ]

    return (
        "<div class=detail-card>"
        + meta
        + "".join(warn_blocks)
        + "<div class=detail-section><div class=detail-h>Pricing breakdown</div>"
        + "".join(pricing_lines)
        + "</div>"
        + "<div class=detail-section><div class=detail-h>How this edge was computed</div>"
        + _kv_block(r.fc_methodology)
        + "</div>"
        + "<div class=detail-section><div class=detail-h>Data inputs (with fetch timestamps)</div>"
        + _kv_block(r.fc_data_sources)
        + "</div>"
        + "<div class=detail-section><div class=detail-h>Diagnostics</div>"
        + _kv_block(r.fc_diagnostics)
        + "</div>"
        + "</div>"
    )


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
    parts.append("Click any <b>Edge</b> value to audit how that edge was derived — model, data inputs, and raw diagnostics. ")
    parts.append("Click any column header to re-sort; rows the forecaster couldn't run on always sink to the bottom.")
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
    parts.append(
        "<th class='sortable' data-key='ticker' data-type='alpha'>"
        "ticker<span class=arr></span></th>"
    )
    parts.append(
        "<th class='sortable' data-key='title' data-type='alpha'>"
        "title<span class=arr></span></th>"
    )
    parts.append(
        "<th class='num sortable' data-key='dte' data-type='num'>"
        "DTE<span class=arr></span></th>"
    )
    parts.append(
        "<th class='num sortable' data-key='market' data-type='num'>"
        "Market &cent;<span class=arr></span></th>"
    )
    parts.append(
        "<th class='num sortable' data-key='fair' data-type='num'>"
        "Fair &cent;<span class=arr></span></th>"
    )
    parts.append(
        "<th class='num sortable sorted' data-key='edge' data-type='num'>"
        "Edge<span class='arr'>\u25BC</span></th>"
    )
    parts.append(
        "<th class='sortable' data-key='bet' data-type='alpha'>"
        "Bet<span class=arr></span></th>"
    )
    parts.append(
        "<th class='num sortable' data-key='conf' data-type='num'>"
        "Conf<span class=arr></span></th>"
    )
    parts.append(
        "<th class='num sortable' data-key='spread' data-type='num'>"
        "Spread<span class=arr></span></th>"
    )
    parts.append(
        "<th class='num sortable' data-key='vol' data-type='num'>"
        "Vol 24h<span class=arr></span></th>"
    )
    parts.append("</tr></thead><tbody>")

    for idx, r in enumerate(sorted_rows):
        dte = _dte_days(r.close_time, now)
        mid = _mid_cents(r)
        spread = None
        if r.yes_bid is not None and r.yes_ask is not None:
            spread = r.yes_ask - r.yes_bid
        cat_attr = html.escape(r.category)
        row_id = f"row-{idx}"
        has_forecast = r.fc_p_yes is not None
        detail_attr = f" data-detail-id='detail-{idx}'" if has_forecast else ""
        parts.append(
            f"<tr class='main-row' data-cat='{cat_attr}' id='{row_id}'{detail_attr}>"
        )
        parts.append(
            f"<td class=ticker data-key='ticker' data-s='{html.escape(r.ticker)}'>"
            f"<a href='{html.escape(_kalshi_url(r))}' target=_blank rel=noopener>"
            f"{html.escape(r.ticker)}</a>"
            f"<span class=cat-tag>{cat_attr}</span></td>"
        )
        title_html = html.escape(r.title)
        tooltip = r.title
        if r.subtitle:
            title_html += (
                f" <span class=yes-side>· YES = {html.escape(r.subtitle)}</span>"
            )
            tooltip = f"{r.title} (YES resolves on: {r.subtitle})"
        parts.append(
            f"<td class=title data-key='title' data-s='{html.escape(r.title)}' "
            f"title='{html.escape(tooltip)}'>{title_html}</td>"
        )
        dte_s = f"{dte:.6f}" if dte is not None else ""
        parts.append(
            f"<td class=num data-key='dte' data-s='{dte_s}'>{_fmt_dte(dte)}</td>"
        )
        mid_s = f"{mid:.4f}" if mid is not None else ""
        parts.append(
            f"<td class=num data-key='market' data-s='{mid_s}'>{_fmt_cents(mid)}</td>"
        )
        if has_forecast:
            fair_cents = r.fc_p_yes * 100.0
            ci = ""
            if r.fc_p05 is not None and r.fc_p95 is not None:
                ci = (
                    f"<div class=ci>"
                    f"[{r.fc_p05 * 100:.0f}¢, {r.fc_p95 * 100:.0f}¢]"
                    f"</div>"
                )
            parts.append(
                f"<td class=num data-key='fair' data-s='{fair_cents:.4f}'>"
                f"{fair_cents:.1f}¢{ci}</td>"
            )
            edge_pts = _edge_points(r)
            edge_txt, edge_cls = _fmt_edge_points(r.fc_p_yes, mid)
            edge_s = f"{edge_pts:.4f}" if edge_pts is not None else ""
            parts.append(
                f"<td class='num edge {edge_cls} edge-click' "
                f"data-key='edge' data-s='{edge_s}' "
                f"data-detail='detail-{idx}' "
                f"title='Click to see how this edge was derived'>"
                f"{edge_txt} <span class=edge-caret>▸</span></td>"
            )
            bet_label, bet_cls, bet_reason = _bet_recommendation(r)
            # Sort key: YES first, NO last, — in the middle.
            bet_sort = {"YES": "a-yes", "—": "b-none", "NO": "c-no"}.get(bet_label, "z")
            bet_attr = f" title='{html.escape(bet_reason)}'" if bet_reason else ""
            parts.append(
                f"<td class='bet {bet_cls}' data-key='bet' "
                f"data-s='{bet_sort}'{bet_attr}>{bet_label}</td>"
            )
            conf_s = (
                f"{r.fc_model_confidence:.4f}"
                if r.fc_model_confidence is not None else ""
            )
            parts.append(
                f"<td class=num data-key='conf' data-s='{conf_s}'>"
                f"{_fmt_prob(r.fc_model_confidence)}</td>"
            )
        elif r.fc_null_reason:
            # colspan cell covers fair/edge/bet/conf — no sortable values.
            parts.append(
                "<td class=num colspan=4 data-key='fair' data-s=''>"
                f"<span class=null>abstained — "
                f"{html.escape(r.fc_null_reason[:100])}</span></td>"
            )
        else:
            parts.append(
                "<td class=num colspan=4 data-key='fair' data-s=''>"
                "<span class=null>no forecaster</span></td>"
            )
        spread_s = f"{spread:.4f}" if spread is not None else ""
        parts.append(
            f"<td class=num data-key='spread' data-s='{spread_s}'>"
            f"{_fmt_cents(spread)}</td>"
        )
        parts.append(
            f"<td class=num data-key='vol' data-s='{r.volume_24h:.4f}'>"
            f"{_fmt_int(r.volume_24h)}</td>"
        )
        parts.append("</tr>")
        if has_forecast:
            detail_inner = _build_detail_html(r, mid)
            parts.append(
                f"<tr class='detail-row' id='detail-{idx}' data-cat='{cat_attr}' hidden>"
                f"<td colspan=10>{detail_inner}</td>"
                f"</tr>"
            )
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
