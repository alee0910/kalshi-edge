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
    # Quantamental split, if present. ``fc_p_yes_quant_only`` is P(YES) before
    # fundamental inputs were applied; ``fc_attribution`` unpacks per-input
    # contribution entries (list) and ``fc_dropped`` the reasons we skipped
    # any input. All None/empty for pure-quant forecasters.
    fc_p_yes_quant_only: float | None = None
    fc_attribution: list[dict[str, Any]] = field(default_factory=list)
    fc_dropped: list[dict[str, Any]] = field(default_factory=list)
    fc_mean_shift_underlying: float | None = None
    fc_forecast_id: int | None = None


_QUERY = """
SELECT
    c.ticker, c.event_ticker, c.series_ticker, c.category, c.title, c.subtitle, c.close_time,
    s.yes_bid, s.yes_ask, s.volume_24h, s.liquidity, s.ts AS snapshot_ts,
    f.id AS fc_id,
    f.p_yes AS fc_p_yes, f.p_yes_p05 AS fc_p05, f.p_yes_p95 AS fc_p95,
    f.model_confidence AS fc_model_confidence,
    f.null_reason AS fc_null_reason,
    f.forecaster AS fc_forecaster,
    f.version AS fc_version,
    f.ts AS fc_ts,
    f.methodology AS fc_methodology,
    f.data_sources AS fc_data_sources,
    f.diagnostics AS fc_diagnostics,
    f.p_yes_quant_only AS fc_p_yes_quant_only,
    a.attribution AS fc_attribution,
    a.dropped AS fc_dropped,
    a.mean_shift_underlying AS fc_mean_shift_underlying
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
LEFT JOIN forecast_attribution a
    ON a.forecast_id = f.id
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
            fc_p_yes_quant_only=r["fc_p_yes_quant_only"],
            fc_attribution=_parse_json_list(r["fc_attribution"]),
            fc_dropped=_parse_json_list(r["fc_dropped"]),
            fc_mean_shift_underlying=r["fc_mean_shift_underlying"],
            fc_forecast_id=r["fc_id"],
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


def _parse_json_list(s: str | None) -> list[dict[str, Any]]:
    """Parse a JSON array-of-dicts field; degrade to empty list."""
    if not s:
        return []
    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [x for x in obj if isinstance(x, dict)]
        return []
    except (ValueError, TypeError):
        return []


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
.detail-narrative p {
    margin: 0 0 8px; line-height: 1.6; font-size: 12.5px; color: #d4d8dd;
}
.detail-narrative p b { color: var(--fg); font-weight: 600; }
.detail-narrative p.detail-caveat {
    color: #f8c377; background: #1d1814; border-left: 2px solid #b06a1f;
    padding: 6px 10px; border-radius: 3px; font-size: 12px;
}
.detail-narrative p.detail-caveat b { color: #fde4bd; }
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
table.fund-table { margin-top: 8px; font-size: 12px; }
table.fund-table th, table.fund-table td { padding: 4px 8px; }
table.fund-table code { color: var(--accent); }
ul.fund-dropped { margin: 4px 0 0 0; padding-left: 20px; font-size: 12px; color: var(--muted); }
ul.fund-dropped code { color: var(--fg); }
.nav-sub { margin-bottom: 20px; font-size: 12px; color: var(--muted); }
.nav-sub a { margin-right: 14px; }
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


def _weather_edge_narrative(r: _Row, mid: float | None) -> str:
    """Plain-English 'why this edge?' block for weather forecasts.

    Builds a short paragraph from the enriched ensemble diagnostics so the
    user can see *what* drove P(YES) — the ensemble's central tendency,
    where the YES threshold sits inside the ensemble distribution, the
    raw empirical hit-rate before any correction, and how much the
    variance-inflation step moves the number. Without this, the detail
    panel shows a flat kv grid and the driver of a surprising edge is
    opaque.

    Returns '' when the forecast isn't a weather-ensemble result or the
    diagnostics don't carry the expected keys (older rows, partial
    migrations).
    """
    if r.fc_forecaster != "weather_ensemble":
        return ""
    d = r.fc_diagnostics
    needed = {"ensemble_mean_f", "ensemble_std_f", "yes_direction",
              "raw_empirical_p_yes", "n_members"}
    if not needed.issubset(d.keys()):
        return ""

    try:
        mu = float(d["ensemble_mean_f"])
        sigma = float(d["ensemble_std_f"])
        direction = str(d["yes_direction"])
        raw_p = float(d["raw_empirical_p_yes"])
        n = int(d["n_members"])
    except (TypeError, ValueError):
        return ""

    median = d.get("ensemble_median_f")
    mn = d.get("ensemble_min_f")
    mx = d.get("ensemble_max_f")
    q05 = d.get("ensemble_q05_f")
    q95 = d.get("ensemble_q95_f")
    kappa = d.get("variance_inflation")
    low = d.get("yes_low_f")
    high = d.get("yes_high_f")

    if direction == "above" and low is not None:
        thr_txt = f"&gt; {low:g}&deg;F"
        z = (float(low) - mu) / sigma if sigma > 0 else 0.0
        placement = (
            f"the YES threshold ({low:g}&deg;F) sits <b>{z:+.2f}&sigma;</b> from "
            f"the ensemble mean"
        )
    elif direction == "below" and high is not None:
        thr_txt = f"&lt; {high:g}&deg;F"
        z = (float(high) - mu) / sigma if sigma > 0 else 0.0
        placement = (
            f"the YES threshold ({high:g}&deg;F) sits <b>{z:+.2f}&sigma;</b> "
            f"from the ensemble mean"
        )
    elif direction == "between" and low is not None and high is not None:
        thr_txt = f"in [{low:g}, {high:g}]&deg;F"
        placement = (
            f"the YES band spans {low:g}–{high:g}&deg;F around the ensemble "
            f"mean of {mu:g}&deg;F"
        )
    else:
        return ""

    central = f"mean <b>{mu:g}&deg;F</b>"
    if median is not None:
        central += f", median {median:g}&deg;F"
    central += f", std {sigma:g}&deg;F"

    spread_bits: list[str] = []
    if q05 is not None and q95 is not None:
        spread_bits.append(f"90% of members fall in [{q05:g}, {q95:g}]&deg;F")
    if mn is not None and mx is not None:
        spread_bits.append(f"full range [{mn:g}, {mx:g}]&deg;F")

    fair_p = r.fc_p_yes if r.fc_p_yes is not None else None
    fair_pct_txt = f"{fair_p * 100:.1f}%" if fair_p is not None else "—"
    raw_pct_txt = f"{raw_p * 100:.1f}%"

    model_line: str
    if kappa is not None:
        model_line = (
            f"We widen each member's deviation from the mean by "
            f"<b>&kappa;={kappa:g}</b> (a published correction for NWP "
            f"ensembles being ~20-50% too narrow at 1-7d lead), then "
            f"Bayesian-bootstrap the inflated members {_BOOTSTRAP_DRAWS_TXT} "
            f"times to get a distribution over P(YES). Posterior mean: "
            f"<b>{fair_pct_txt}</b>."
        )
    else:
        model_line = f"Posterior mean P(YES): <b>{fair_pct_txt}</b>."

    market_line = ""
    if mid is not None and fair_p is not None:
        delta = fair_p * 100.0 - mid
        market_line = (
            f" The market prices this at <b>{mid:.0f}&cent;</b>, so the "
            f"edge is <b>{delta:+.1f}&cent;</b> = model − market."
        )

    # The "1-7d lead" correction is the published regime where GEFS/ECMWF
    # 2m-T ensembles are known underdispersive. Beyond that it's less
    # settled — flag when the user is looking at a far-out forecast.
    days_ahead = d.get("days_ahead")
    caveats: list[str] = []
    if isinstance(days_ahead, (int, float)) and days_ahead > 7:
        caveats.append(
            f"Target is <b>{int(days_ahead)} days out</b>, beyond the 1-7d "
            f"window where our &kappa; calibration is strongest — treat the "
            f"number as weaker than it would be at short lead."
        )
    # Big market/model divergence on coastal sites: the 25km grid can't
    # see the marine layer, so "LAX high" forecasts run warm vs actual
    # observations. Only mention when the divergence is notable.
    if mid is not None and fair_p is not None and abs(fair_p * 100 - mid) >= 15:
        slug_hint = ""
        if "LAX" in r.ticker.upper():
            slug_hint = (
                "This is an LAX (coastal) site, where a 25 km grid "
                "can't resolve the marine-layer temperature inversion. "
                "Expect a systematic warm bias on days the layer "
                "persists; treat disagreements against a liquid market "
                "as a flag for this limitation."
            )
        if slug_hint:
            caveats.append(slug_hint)

    bits: list[str] = [
        f"<p><b>Ensemble ({n} members):</b> {central}."
        + (f" {'; '.join(spread_bits)}." if spread_bits else "")
        + "</p>",
        f"<p><b>Threshold placement:</b> {placement}; raw fraction of "
        f"members satisfying YES ({thr_txt}) is <b>{raw_pct_txt}</b> "
        f"(no inflation, no bootstrap — this is the ensemble's vote).</p>",
        f"<p><b>Model step:</b> {model_line}{market_line}</p>",
    ]
    for c in caveats:
        bits.append(f"<p class=detail-caveat><b>Caveat:</b> {c}</p>")
    return "".join(bits)


_BOOTSTRAP_DRAWS_TXT = "2000"


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

    narrative = _weather_edge_narrative(r, mid)
    narrative_block = (
        "<div class=detail-section>"
        "<div class=detail-h>Why this edge?</div>"
        f"<div class=detail-narrative>{narrative}</div>"
        "</div>"
        if narrative else ""
    )

    return (
        "<div class=detail-card>"
        + meta
        + "".join(warn_blocks)
        + "<div class=detail-section><div class=detail-h>Pricing breakdown</div>"
        + "".join(pricing_lines)
        + "</div>"
        + narrative_block
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


def _fundamental_attribution_block(r: _Row) -> str:
    """Render the quant-only-vs-adjusted split and per-input attribution.

    Only rendered for quantamental forecasts (``fc_p_yes_quant_only`` or
    ``fc_attribution`` present). For pure-quant rows this returns ``""``.
    """
    if (
        r.fc_p_yes_quant_only is None
        and not r.fc_attribution
        and not r.fc_dropped
    ):
        return ""

    lines: list[str] = ["<div class=detail-section>"]
    lines.append("<div class=detail-h>Fundamental attribution</div>")

    # Split summary: quant-only vs adjusted P(YES).
    if r.fc_p_yes_quant_only is not None and r.fc_p_yes is not None:
        q_pts = r.fc_p_yes_quant_only * 100.0
        adj_pts = r.fc_p_yes * 100.0
        shift_pts = adj_pts - q_pts
        lines.append(
            f"<div class=kv><span class=k>quant-only P(YES)</span>"
            f"<span class=v>{q_pts:.2f}¢</span></div>"
            f"<div class=kv><span class=k>quant + fundamental</span>"
            f"<span class=v>{adj_pts:.2f}¢ ({shift_pts:+.2f}¢ shift)</span></div>"
        )
    if r.fc_mean_shift_underlying is not None:
        lines.append(
            f"<div class=kv><span class=k>underlying mean shift</span>"
            f"<span class=v>{r.fc_mean_shift_underlying:+.4f}</span></div>"
        )

    # Per-input table. Kept compact; full provenance lives in the brief.
    if r.fc_attribution:
        rows_html = [
            "<table class='fund-table'>",
            "<thead><tr>"
            "<th>input</th>"
            "<th class=num>anomaly</th>"
            "<th class=num>β (shrunk)</th>"
            "<th class=num>σ_β</th>"
            "<th class=num>n_obs</th>"
            "<th class=num>Δ underlying</th>"
            "<th class=num>Δ P(YES)</th>"
            "</tr></thead><tbody>",
        ]
        for a in r.fc_attribution:
            ds_p = a.get("mean_shift_prob")
            ds_p_s = "—" if ds_p is None else f"{ds_p * 100:+.2f}¢"
            rows_html.append(
                f"<tr>"
                f"<td><code>{html.escape(str(a.get('name', '—')))}</code></td>"
                f"<td class=num>{_fmt_float(a.get('anomaly'))}</td>"
                f"<td class=num>{_fmt_float(a.get('beta_shrunk'))}</td>"
                f"<td class=num>{_fmt_float(a.get('beta_std'))}</td>"
                f"<td class=num>{_fmt_int(a.get('n_obs'))}</td>"
                f"<td class=num>{_fmt_float(a.get('mean_shift_underlying'))}</td>"
                f"<td class=num>{html.escape(ds_p_s)}</td>"
                f"</tr>"
            )
        rows_html.append("</tbody></table>")
        lines.extend(rows_html)
    else:
        lines.append(
            "<div class=detail-empty>No active fundamental inputs drove this forecast "
            "(all were missing, expired, or uncalibrated).</div>"
        )

    if r.fc_dropped:
        dropped_items = [
            f"<li><code>{html.escape(str(d.get('name', '?')))}</code> — "
            f"{html.escape(str(d.get('reason', '?')))}</li>"
            for d in r.fc_dropped
        ]
        lines.append(
            "<div class=detail-h style='margin-top:10px'>Dropped inputs</div>"
            f"<ul class='fund-dropped'>{''.join(dropped_items)}</ul>"
        )

    lines.append("</div>")
    return "".join(lines)


def _fmt_float(v: Any) -> str:
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return html.escape(str(v))
    if abs(f) >= 1000 or (f != 0 and abs(f) < 1e-3):
        return f"{f:+.4g}"
    return f"{f:+.4f}"


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


def _quantamental_page_shell(title: str, body: str, *, now: datetime) -> str:
    """Wrap a body fragment in chrome for the /quantamental/ subsite.

    Links are relative and resolve within the subdir. The ``../`` link back
    to the universe assumes the quantamental subsite lives one level below
    the main page (site/ layout: ``/index.html`` + ``/quantamental/*``).
    """
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)} — kalshi-edge quantamental</title>"
        f"<style>{_CSS}</style></head><body><div class=wrap>"
        f"<h1>kalshi-edge — quantamental · {html.escape(title)}</h1>"
        "<div class=nav-sub>"
        "<a href='./index.html'>Overview</a>"
        "<a href='./research_queue.html'>Research queue</a>"
        "<a href='./calibration_split.html'>Calibration split</a>"
        "<a href='./briefs/'>Briefs</a>"
        "<a href='../index.html'>← Universe</a>"
        "</div>"
        + body
        + "<footer>"
        + f"Rebuilt at {html.escape(now.strftime('%Y-%m-%d %H:%M UTC'))}."
        + "</footer></div></body></html>"
    )


def build_research_queue_html(db: Database, *, now: datetime | None = None) -> str:
    """Page listing contracts needing analyst attention.

    A contract lands on the research queue if ANY of the following hold:

        * The forecaster is quantamental (wraps fundamental inputs) and at
          least one declared input for its category has no live entry. The
          analyst can close the gap with a manual YAML.
        * The forecaster is quantamental and at least one live input was
          dropped at integration time (e.g. ``input_expired``,
          ``uncalibrated_loading``).
        * The model shows a large, liquid-market disagreement (|edge| ≥
          15¢ AND |edge| ≥ ``_BET_MIN_EDGE_PTS``) — these deserve a human
          eyeball before the ranker acts.

    Rows are sorted so the gappiest/most-divergent sit at the top. The
    page is self-contained and uses the same CSS as the main report.
    """
    now = now or datetime.now(timezone.utc)
    rows = _load_rows(db)

    # Per-category expected input sets. Kept local so report.py doesn't force
    # a hard import dependency on the fundamental package if it isn't loaded.
    expected_by_cat = _expected_inputs_by_category()

    # Per-category sets of observed input names (from the latest attribution
    # of each forecast in that category). Cheap to compute from the already-
    # loaded rows.
    observed_by_cat: dict[str, set[str]] = {}
    for r in rows:
        names = {a.get("name") for a in r.fc_attribution if a.get("name")}
        if names:
            observed_by_cat.setdefault(r.category, set()).update(
                x for x in names if isinstance(x, str)
            )

    @dataclass(frozen=True, slots=True)
    class _QueueItem:
        row: _Row
        missing_inputs: tuple[str, ...]
        dropped_inputs: tuple[str, ...]
        divergence_pts: float | None
        sort_key: float

    queue: list[_QueueItem] = []
    for r in rows:
        expected = expected_by_cat.get(r.category, set())
        observed_names = {a.get("name") for a in r.fc_attribution if a.get("name")}
        missing = tuple(sorted(expected - observed_names)) if expected else ()
        dropped = tuple(sorted(str(d.get("name", "?")) for d in r.fc_dropped))
        mid = _mid_cents(r)
        edge = _edge_points(r)
        divergence = None
        # Only flag divergence for forecasted rows with a liquid market.
        if (
            r.fc_p_yes is not None and mid is not None and edge is not None
            and _is_liquid(r) and abs(edge) >= max(_BET_MIN_EDGE_PTS, 15.0)
        ):
            divergence = edge

        if not missing and not dropped and divergence is None:
            continue

        # Bigger problems sort higher. Scale: missing-count * 10 + dropped * 5
        # + |divergence|. Negative to sort DESC.
        score = -(
            len(missing) * 10.0
            + len(dropped) * 5.0
            + (abs(divergence) if divergence is not None else 0.0)
        )
        queue.append(_QueueItem(
            row=r,
            missing_inputs=missing,
            dropped_inputs=dropped,
            divergence_pts=divergence,
            sort_key=score,
        ))

    queue.sort(key=lambda q: (q.sort_key, q.row.ticker))

    body: list[str] = []
    body.append(
        "<div class=sub>"
        "Contracts where the system wants the analyst to look. Items fall "
        "into three buckets: missing fundamental inputs (analyst can fill), "
        "dropped inputs (expired or uncalibrated), and large model-vs-market "
        "disagreements on liquid markets."
        "</div>"
    )

    if not queue:
        body.append(
            "<div class=notice>No items on the research queue. Either every "
            "expected input is live and calibrated, or no forecaster has run "
            "yet.</div>"
        )
        return _quantamental_page_shell("Research queue", "".join(body), now=now)

    body.append(
        "<table><thead><tr>"
        "<th>ticker</th><th>title</th>"
        "<th class=num>edge</th>"
        "<th>missing</th><th>dropped</th>"
        "<th>reason</th>"
        "</tr></thead><tbody>"
    )
    for q in queue:
        r = q.row
        mid = _mid_cents(r)
        edge_txt, edge_cls = _fmt_edge_points(r.fc_p_yes, mid)
        reasons: list[str] = []
        if q.missing_inputs:
            reasons.append(f"{len(q.missing_inputs)} missing input(s)")
        if q.dropped_inputs:
            reasons.append(f"{len(q.dropped_inputs)} dropped input(s)")
        if q.divergence_pts is not None:
            reasons.append(f"liquid divergence {q.divergence_pts:+.1f}¢")
        body.append(
            f"<tr>"
            f"<td class=ticker>"
            f"<a href='{html.escape(_kalshi_url(r))}' target=_blank rel=noopener>"
            f"{html.escape(r.ticker)}</a></td>"
            f"<td class=title title='{html.escape(r.title)}'>{html.escape(r.title)}</td>"
            f"<td class='num edge {edge_cls}'>{edge_txt}</td>"
            f"<td>{_fmt_name_list(q.missing_inputs)}</td>"
            f"<td>{_fmt_name_list(q.dropped_inputs)}</td>"
            f"<td>{html.escape('; '.join(reasons))}</td>"
            f"</tr>"
        )
    body.append("</tbody></table>")
    return _quantamental_page_shell("Research queue", "".join(body), now=now)


def _fmt_name_list(names: tuple[str, ...]) -> str:
    if not names:
        return "—"
    inner = ", ".join(f"<code>{html.escape(n)}</code>" for n in names)
    return f"<span class=yes-side>{inner}</span>"


def _expected_inputs_by_category() -> dict[str, set[str]]:
    """Return {category: {input_name, ...}} from declared InputSpecs.

    Imports live inside the function so that the top of report.py doesn't
    force loading the fundamental package (keeps the dashboard usable even
    if fundamentals haven't been initialized yet).
    """
    try:
        from kalshi_edge.fundamental.schemas.cpi import cpi_input_specs
    except ImportError:
        return {}
    out: dict[str, set[str]] = {}
    for spec in cpi_input_specs():
        out.setdefault(spec.category, set()).add(spec.name)
    return out


def build_calibration_split_html(db: Database, *, now: datetime | None = None) -> str:
    """Page showing quant-only vs quant+fundamental Brier / log-loss split.

    Only meaningful once resolved forecasts start accumulating in
    ``calibration_records``. For quantamental forecasters, each row records
    both the quant-only and quant+fundamental Brier; this page aggregates
    them per forecaster and displays the delta. A negative delta (funda-
    mental is better) is the evidence we need that the fundamental layer
    earns its complexity.
    """
    now = now or datetime.now(timezone.utc)
    cur = db.execute(
        """
        SELECT forecaster, category,
               COUNT(*) AS n,
               AVG(brier) AS brier,
               AVG(log_score) AS log_score,
               AVG(brier_quant_only) AS brier_qonly,
               AVG(log_score_quant_only) AS log_qonly,
               SUM(CASE WHEN p_yes_quant_only IS NOT NULL THEN 1 ELSE 0 END) AS n_qonly
        FROM calibration_records
        GROUP BY forecaster, category
        ORDER BY forecaster, category
        """
    )
    rows = list(cur.fetchall())

    body: list[str] = []
    body.append(
        "<div class=sub>"
        "Per-forecaster Brier / log-loss split. For quantamental forecasters "
        "the <b>Δ Brier</b> column (quant+fundamental minus quant-only) says "
        "whether fundamentals are adding alpha: negative means the full model "
        "is scoring better than the quant-only pass."
        "</div>"
    )
    if not rows:
        body.append(
            "<div class=notice>No calibration records yet. This page fills "
            "in once forecasts start resolving.</div>"
        )
        return _quantamental_page_shell("Calibration split", "".join(body), now=now)

    body.append(
        "<table><thead><tr>"
        "<th>forecaster</th><th>category</th>"
        "<th class=num>n</th><th class=num>n (QF)</th>"
        "<th class=num>Brier</th><th class=num>Brier (quant-only)</th>"
        "<th class=num>Δ Brier</th>"
        "<th class=num>log-loss</th><th class=num>log-loss (quant-only)</th>"
        "</tr></thead><tbody>"
    )
    for r in rows:
        brier = r["brier"]
        brier_q = r["brier_qonly"]
        d_brier = (brier - brier_q) if (brier is not None and brier_q is not None) else None
        d_cls = ""
        if d_brier is not None:
            d_cls = "pos" if d_brier < 0 else ("neg" if d_brier > 0 else "neu")
        d_brier_s = _fmt_float(d_brier) if d_brier is not None else "—"
        body.append(
            f"<tr>"
            f"<td><code>{html.escape(r['forecaster'] or '—')}</code></td>"
            f"<td>{html.escape(r['category'] or '—')}</td>"
            f"<td class=num>{r['n']}</td>"
            f"<td class=num>{r['n_qonly'] or 0}</td>"
            f"<td class=num>{_fmt_float(brier)}</td>"
            f"<td class=num>{_fmt_float(brier_q)}</td>"
            f"<td class='num edge {d_cls}'>{d_brier_s}</td>"
            f"<td class=num>{_fmt_float(r['log_score'])}</td>"
            f"<td class=num>{_fmt_float(r['log_qonly'])}</td>"
            f"</tr>"
        )
    body.append("</tbody></table>")
    return _quantamental_page_shell("Calibration split", "".join(body), now=now)


def write_report(db: Database, out_path: Path, *, now: datetime | None = None) -> int:
    """Render and write the universe page. Returns bytes written."""
    html_out = build_report_html(db, now=now)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path.write_text(html_out, encoding="utf-8")


def build_quantamental_index_html(db: Database, *, now: datetime | None = None) -> str:
    """Landing page for the /quantamental/ subsite.

    Explains what the subsite is, summarizes the live fundamental-input
    stock by category, and links out to the analytical pages. Intentionally
    sparse — the per-contract attribution lives in the Markdown briefs and
    the research queue.
    """
    now = now or datetime.now(timezone.utc)

    cur = db.execute(
        """
        SELECT category, COUNT(*) AS n,
               SUM(CASE WHEN expires_at IS NULL OR expires_at > ? THEN 1 ELSE 0 END) AS n_live
        FROM fundamental_inputs
        GROUP BY category
        ORDER BY category
        """,
        (now.isoformat(),),
    )
    input_rows = list(cur.fetchall())

    cur = db.execute(
        """
        SELECT name, MAX(fit_at) AS fit_at, beta, beta_std, n_obs
        FROM fundamental_loadings
        GROUP BY name
        ORDER BY name
        """
    )
    loading_rows = list(cur.fetchall())

    cur = db.execute(
        """
        SELECT COUNT(*) AS n_fc,
               SUM(CASE WHEN p_yes_quant_only IS NOT NULL THEN 1 ELSE 0 END) AS n_qf
        FROM forecasts
        """
    )
    fc_row = cur.fetchone()

    body: list[str] = []
    body.append(
        "<div class=sub>"
        "The quantamental subsite augments the pure-quant universe with "
        "<b>structured fundamental inputs</b> that shape Bayesian priors. "
        "Each input carries a value, uncertainty, provenance, expiration, "
        "and a quantified integration mechanism. No qualitative overlays, "
        "no LLMs in the loop."
        "</div>"
    )

    body.append("<h2 style='margin-top:28px'>Stock of fundamental inputs</h2>")
    if not input_rows:
        body.append(
            "<div class=notice>No fundamental inputs loaded yet. Run "
            "<code>kalshi-edge pull-fundamental</code>.</div>"
        )
    else:
        body.append(
            "<table><thead><tr>"
            "<th>category</th><th class=num>total</th><th class=num>live</th>"
            "</tr></thead><tbody>"
        )
        for r in input_rows:
            body.append(
                f"<tr><td>{html.escape(r['category'] or '—')}</td>"
                f"<td class=num>{r['n']}</td>"
                f"<td class=num>{r['n_live'] or 0}</td></tr>"
            )
        body.append("</tbody></table>")

    body.append("<h2 style='margin-top:28px'>Calibrated loadings</h2>")
    if not loading_rows:
        body.append(
            "<div class=notice>No calibrated loadings yet. Run "
            "<code>kalshi-edge calibrate-loadings</code> (needs "
            "<code>FRED_API_KEY</code>).</div>"
        )
    else:
        body.append(
            "<table><thead><tr>"
            "<th>input</th><th class=num>β</th><th class=num>σ_β</th>"
            "<th class=num>n_obs</th><th>last fit</th>"
            "</tr></thead><tbody>"
        )
        for r in loading_rows:
            body.append(
                f"<tr><td><code>{html.escape(r['name'] or '—')}</code></td>"
                f"<td class=num>{_fmt_float(r['beta'])}</td>"
                f"<td class=num>{_fmt_float(r['beta_std'])}</td>"
                f"<td class=num>{r['n_obs']}</td>"
                f"<td>{html.escape(str(r['fit_at'] or '—'))}</td></tr>"
            )
        body.append("</tbody></table>")

    n_fc = fc_row["n_fc"] if fc_row else 0
    n_qf = (fc_row["n_qf"] or 0) if fc_row else 0
    body.append(
        f"<h2 style='margin-top:28px'>Forecast coverage</h2>"
        f"<div class=kv><span class=k>total forecasts</span>"
        f"<span class=v>{n_fc}</span></div>"
        f"<div class=kv><span class=k>quantamental forecasts</span>"
        f"<span class=v>{n_qf}</span></div>"
    )

    return _quantamental_page_shell("Overview", "".join(body), now=now)


def write_quantamental_report(
    db: Database,
    out_dir: Path,
    *,
    now: datetime | None = None,
) -> int:
    """Render the three quantamental pages into ``out_dir``.

    Writes ``index.html`` (landing page), ``research_queue.html``, and
    ``calibration_split.html``. Returns the bytes written for the landing
    page (simple success signal). Briefs under ``<out_dir>/briefs/`` are
    produced by ``build-briefs`` separately.
    """
    now = now or datetime.now(timezone.utc)
    out_dir.mkdir(parents=True, exist_ok=True)

    idx_path = out_dir / "index.html"
    n = idx_path.write_text(build_quantamental_index_html(db, now=now), encoding="utf-8")

    rq_path = out_dir / "research_queue.html"
    rq_path.write_text(build_research_queue_html(db, now=now), encoding="utf-8")

    cs_path = out_dir / "calibration_split.html"
    cs_path.write_text(build_calibration_split_html(db, now=now), encoding="utf-8")

    return n
