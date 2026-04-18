"""Tests for Polymarket Beta prior + title-match cross-market logic."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from kalshi_edge.data_sources.polymarket import PolymarketMarket
from kalshi_edge.forecasters.polymarket_match import (
    default_window,
    find_best_match,
    jaccard,
)
from kalshi_edge.forecasters.polymarket_prior import (
    beta_posterior_from_polymarket,
    ess_from_liquidity,
)


# ---- ess_from_liquidity ----------------------------------------------------

def test_ess_is_monotonic_in_volume() -> None:
    a = ess_from_liquidity(volume=1e3, liquidity=0.0)
    b = ess_from_liquidity(volume=1e6, liquidity=0.0)
    assert a < b


def test_ess_clipped_floor_and_cap() -> None:
    lo = ess_from_liquidity(volume=0.0, liquidity=0.0)
    hi = ess_from_liquidity(volume=1e12, liquidity=1e12)
    # Floor / cap defined as module constants.
    assert 11.9 < lo < 12.1
    assert 239.9 < hi < 240.1


# ---- beta_posterior_from_polymarket ---------------------------------------

def test_beta_mean_matches_yes_price() -> None:
    dist, _ = beta_posterior_from_polymarket(
        yes_price=0.65, volume=1e5, liquidity=5e3,
    )
    assert dist.kind == "parametric"
    assert dist.family == "beta"
    a = dist.params["a"]
    b = dist.params["b"]
    assert a / (a + b) == pytest.approx(0.65, abs=1e-9)


def test_beta_wider_for_thin_market() -> None:
    dist_thin, _ = beta_posterior_from_polymarket(
        yes_price=0.5, volume=100.0, liquidity=10.0,
    )
    dist_deep, _ = beta_posterior_from_polymarket(
        yes_price=0.5, volume=1e6, liquidity=1e5,
    )
    assert dist_thin.std() > dist_deep.std()


def test_beta_proxy_fidelity_widens() -> None:
    d_full, _ = beta_posterior_from_polymarket(
        yes_price=0.5, volume=1e5, liquidity=1e3, proxy_fidelity=1.0,
    )
    d_loose, _ = beta_posterior_from_polymarket(
        yes_price=0.5, volume=1e5, liquidity=1e3, proxy_fidelity=0.5,
    )
    assert d_loose.std() > d_full.std()


def test_beta_rejects_degenerate_price() -> None:
    with pytest.raises(ValueError):
        beta_posterior_from_polymarket(yes_price=0.0, volume=1.0, liquidity=1.0)
    with pytest.raises(ValueError):
        beta_posterior_from_polymarket(yes_price=1.0, volume=1.0, liquidity=1.0)


def test_beta_rejects_bad_fidelity() -> None:
    with pytest.raises(ValueError):
        beta_posterior_from_polymarket(
            yes_price=0.5, volume=1.0, liquidity=1.0, proxy_fidelity=0.0,
        )


# ---- jaccard ---------------------------------------------------------------

def test_jaccard_disjoint_is_zero() -> None:
    assert jaccard({"a"}, {"b"}) == 0.0


def test_jaccard_identical_is_one() -> None:
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0


# ---- find_best_match -------------------------------------------------------

def _pm(slug: str, question: str, end: datetime | None, yes: float = 0.5) -> PolymarketMarket:
    return PolymarketMarket(
        id=slug,
        slug=slug,
        condition_id="c",
        question=question,
        yes_price=yes,
        no_price=1 - yes,
        volume=1000.0,
        liquidity=100.0,
        end_date=end,
        fetched_at=datetime.now(timezone.utc),
    )


def test_find_best_match_picks_high_overlap() -> None:
    client = MagicMock()
    client.search_active.return_value = [
        _pm("a", "Fed rate decision at March 2026 FOMC meeting 25 basis points cut", None),
        _pm("b", "Will Bitcoin hit 100k by end of year", None),
    ]
    m = find_best_match(
        client=client, search_keywords=["fed"],
        title="Will the Fed cut 25 basis points at March 2026 meeting",
    )
    assert m is not None
    assert m.market.slug == "a"


def test_find_best_match_abstains_on_tie() -> None:
    client = MagicMock()
    client.search_active.return_value = [
        _pm("a", "Fed 25bp cut March 2026", None),
        _pm("b", "Fed 25bp cut March 2026", None),
    ]
    m = find_best_match(
        client=client, search_keywords=["fed"],
        title="Fed 25bp cut March 2026",
    )
    # Both tie at 1.0 — ambiguous.
    assert m is None


def test_find_best_match_respects_required_tokens() -> None:
    client = MagicMock()
    client.search_active.return_value = [
        _pm("wrongmonth", "Fed 25bp cut January 2026", None),
    ]
    m = find_best_match(
        client=client, search_keywords=["fed"],
        title="Will the Fed cut 25 basis points at March 2026 meeting",
        extra_required_tokens={"march", "2026"},
    )
    assert m is None  # required tokens not in question


def test_find_best_match_respects_date_window() -> None:
    client = MagicMock()
    out_of_window = datetime(2030, 1, 1, tzinfo=timezone.utc)
    client.search_active.return_value = [
        _pm("a", "Fed 25bp cut March 2026 meeting decision", out_of_window),
    ]
    anchor = datetime(2026, 3, 15, tzinfo=timezone.utc)
    m = find_best_match(
        client=client, search_keywords=["fed"],
        title="Fed 25bp cut March 2026 meeting decision",
        target_window=default_window(anchor, days=45),
    )
    assert m is None


def test_find_best_match_subtitle_disambiguates_opposite_outcome() -> None:
    """Regression: Kalshi KXFEDDECISION-26MAY-T25 (subtitle "25 basis points")
    must not match a Polymarket market about the 50bp outcome, even
    though most of the rest of the title overlaps.

    This is the rates/politics analog of the weather strike_type sign
    flip: we had enough Jaccard overlap on "fed / may / 2026 / basis /
    points" to clear _MIN_SCORE against a 50bp Polymarket market, but
    the two resolve on different binary events.
    """
    client = MagicMock()
    client.search_active.return_value = [
        _pm("t50", "Will the Fed cut 50 basis points at May 2026 meeting", None),
        _pm("t25", "Will the Fed cut 25 basis points at May 2026 meeting", None),
    ]
    m = find_best_match(
        client=client, search_keywords=["fed"],
        title="Fed rate decision May 2026 meeting",
        subtitle="25 basis points",
    )
    assert m is not None
    assert m.market.slug == "t25"


def test_find_best_match_subtitle_filters_all_yields_none() -> None:
    """If no candidate contains the subtitle tokens, abstain."""
    client = MagicMock()
    client.search_active.return_value = [
        _pm("t50", "Will the Fed cut 50 basis points at May 2026 meeting", None),
    ]
    m = find_best_match(
        client=client, search_keywords=["fed"],
        title="Fed rate decision May 2026 meeting",
        subtitle="25 basis points",
    )
    assert m is None


def test_find_best_match_below_min_score_returns_none() -> None:
    client = MagicMock()
    client.search_active.return_value = [
        _pm("a", "Totally unrelated topic", None),
    ]
    m = find_best_match(
        client=client, search_keywords=["fed"],
        title="Will the Fed cut rates?",
    )
    assert m is None
