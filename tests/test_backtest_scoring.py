"""Tests for scoring rules and analysis join logic."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from kalshi_edge.backtest.scoring import (
    brier,
    log_loss,
    mean_brier,
    mean_log_loss,
    reliability_bins,
)
from kalshi_edge.backtest.analysis import compute_metrics, join_corpus


class TestScoring:
    def test_brier_matches_squared_error(self):
        assert brier(0.7, 1) == pytest.approx(0.09)
        assert brier(0.7, 0) == pytest.approx(0.49)
        assert brier(0.0, 0) == 0.0
        assert brier(1.0, 1) == 0.0

    def test_log_loss_clipped_at_extremes(self):
        # p=1, outcome=0 would be -log(0) = infty; clip keeps it finite.
        v = log_loss(1.0, 0)
        assert math.isfinite(v)
        assert v > 15  # -log(1e-9) ≈ 20.7 — clip is at 1e-9

    def test_mean_helpers_return_none_on_empty(self):
        assert mean_brier([], []) is None
        assert mean_log_loss([], []) is None

    def test_reliability_bins_basic(self):
        # Perfectly calibrated: empirical_rate == mean_p in each bin.
        ps = [0.1, 0.1, 0.9, 0.9]
        ys = [0, 0, 1, 1]  # 0% in low bin, 100% in high bin
        bins = reliability_bins(ps, ys, n_bins=10)
        assert len(bins) == 2
        lo = next(b for b in bins if b["mean_p"] < 0.5)
        hi = next(b for b in bins if b["mean_p"] > 0.5)
        assert lo["empirical_rate"] == 0.0
        assert hi["empirical_rate"] == 1.0
        assert lo["n"] == 2
        assert hi["n"] == 2

    def test_reliability_skips_empty_bins(self):
        bins = reliability_bins([0.5], [1], n_bins=10)
        assert len(bins) == 1


class TestJoinAndCompute:
    """End-to-end: write an NDJSON corpus by hand and verify the analyzer."""

    def _write_corpus(self, root: Path) -> None:
        def write(path: Path, rows: list[dict]) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")

        # Two forecasts on ticker T1 from a single quant forecaster.
        # Market mid is 40c at both forecast times; forecaster says 60%
        # YES, so edge is +20c both times. Outcome is YES (resolution=YES).
        write(
            root / "forecasts" / "2026" / "04" / "17.ndjson",
            [
                {"ticker": "T1", "ts": "2026-04-17T10:00:00+00:00",
                 "forecaster": "fc", "version": "v1", "p_yes": 0.6},
                {"ticker": "T1", "ts": "2026-04-17T11:00:00+00:00",
                 "forecaster": "fc", "version": "v1", "p_yes": 0.6},
            ],
        )
        write(
            root / "market_snapshots" / "2026" / "04" / "17.ndjson",
            [
                {"ticker": "T1", "ts": "2026-04-17T09:55:00+00:00",
                 "yes_bid": 38.0, "yes_ask": 42.0},
            ],
        )
        write(
            root / "resolutions" / "2026" / "04" / "18.ndjson",
            [
                {"ticker": "T1", "resolved_at": "2026-04-18T00:00:00+00:00",
                 "outcome": "YES", "settled_price": 100.0},
            ],
        )
        write(
            root / "contracts_lite" / "2026" / "04" / "17.ndjson",
            [
                {"ticker": "T1", "event_ticker": "E", "series_ticker": "S",
                 "category": "sports", "title": "T", "subtitle": "",
                 "close_time": "2026-04-17T23:00:00+00:00",
                 "expiration_time": "2026-04-17T23:00:00+00:00",
                 "ts": "2026-04-17T12:00:00+00:00"},
            ],
        )

    def test_join_matches_snapshot_and_resolution(self, tmp_path):
        self._write_corpus(tmp_path)
        joined = join_corpus(tmp_path)
        assert len(joined) == 2
        j = joined[0]
        assert j.ticker == "T1"
        assert j.forecaster == "fc"
        assert j.category == "sports"
        assert j.market_mid_cents == pytest.approx(40.0)
        assert j.outcome == 1

    def test_metrics_resolve_and_edge_pnl(self, tmp_path):
        self._write_corpus(tmp_path)
        joined = join_corpus(tmp_path)
        metrics = compute_metrics(joined, edge_threshold_cents=5.0)
        assert len(metrics) == 1
        m = metrics[0]
        assert m.forecaster == "fc"
        assert m.category == "sports"
        assert m.n_forecasts == 2
        assert m.n_resolved == 2
        # Brier for p=0.6 outcome=1 is (0.4)^2 = 0.16.
        assert m.brier == pytest.approx(0.16)
        # Every forecast has edge=+20c ≥ 5c → 2 bets, bought YES at 40c,
        # paid off at 100c → PnL = 2 * (100 - 40) = 120c.
        assert m.edge_bets == 2
        assert m.edge_pnl_cents == pytest.approx(120.0)

    def test_ignores_forecasts_without_snapshot_or_resolution(self, tmp_path):
        # Forecast on T2 has no snapshot and no resolution — should count
        # toward n_forecasts but not edge_bets or Brier.
        root = tmp_path
        (root / "forecasts" / "2026" / "04" / "17.ndjson").parent.mkdir(
            parents=True, exist_ok=True
        )
        (root / "forecasts" / "2026" / "04" / "17.ndjson").write_text(
            json.dumps({"ticker": "T2", "ts": "2026-04-17T10:00:00+00:00",
                       "forecaster": "fc", "version": "v1", "p_yes": 0.7})
            + "\n"
        )
        joined = join_corpus(tmp_path)
        assert len(joined) == 1
        metrics = compute_metrics(joined)
        assert metrics[0].n_forecasts == 1
        assert metrics[0].n_resolved == 0
        assert metrics[0].edge_bets == 0
        assert metrics[0].brier is None
