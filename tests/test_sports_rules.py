"""Tests for sports contract parsing + odds-api game matching + devigging."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from kalshi_edge.data_sources.the_odds_api import (
    BookOutcome,
    Game,
    SERIES_TO_SPORT,
    TheOddsAPIClient,
    devig_book,
)
from kalshi_edge.forecasters.sports_rules import (
    match_game,
    parse_sports_contract,
)


def test_parse_sports_contract_mlb() -> None:
    sc = parse_sports_contract("KXMLBGAME", "KXMLBGAME-26APR181605CWSATH")
    assert sc is not None
    assert sc.sport_key == "baseball_mlb"
    assert sc.target_date == date(2026, 4, 18)


def test_parse_sports_contract_unknown_series() -> None:
    assert parse_sports_contract("KXATPMATCH", "KXATPMATCH-26MAR03-XY") is None


def test_parse_sports_contract_bad_date() -> None:
    assert parse_sports_contract("KXNBAGAME", "KXNBAGAME-99XXX01LAC") is None


def _game(home: str, away: str, on: date) -> Game:
    return Game(
        sport_key="baseball_mlb",
        game_id="g1",
        commence_time=datetime(on.year, on.month, on.day, 20, 0, tzinfo=timezone.utc),
        home_team=home,
        away_team=away,
        outcomes=[],
        fetched_at=datetime.now(timezone.utc),
    )


def test_match_game_picks_yes_team_by_position() -> None:
    sc = parse_sports_contract("KXMLBGAME", "KXMLBGAME-26APR181605CWSATH")
    assert sc is not None
    games = [_game("Chicago White Sox", "Oakland Athletics", date(2026, 4, 18))]
    res = match_game(sc, "Will the White Sox beat the Athletics?", games)
    assert res is not None
    g, team = res
    assert team == "Chicago White Sox"  # earlier mention wins


def test_match_game_away_team_first() -> None:
    sc = parse_sports_contract("KXMLBGAME", "KXMLBGAME-26APR181605CWSATH")
    assert sc is not None
    games = [_game("Chicago White Sox", "Oakland Athletics", date(2026, 4, 18))]
    res = match_game(sc, "Will the Athletics beat the White Sox?", games)
    assert res is not None
    _, team = res
    assert team == "Oakland Athletics"


def test_match_game_nba_city_only_title() -> None:
    """Regression: Kalshi NBA contracts use "City at City Winner?" format.

    The-odds-api returns "Toronto Raptors" / "Cleveland Cavaliers"; the
    Kalshi title mentions only the cities. We need to match by the
    leading-words (city) token, not just the mascot.
    """
    sc = parse_sports_contract("KXNBAGAME", "KXNBAGAME-26APR18TORCLE")
    assert sc is not None
    game = Game(
        sport_key="basketball_nba",
        game_id="g1",
        commence_time=datetime(2026, 4, 18, 17, 0, tzinfo=timezone.utc),
        home_team="Cleveland Cavaliers",
        away_team="Toronto Raptors",
        outcomes=[],
        fetched_at=datetime.now(timezone.utc),
    )
    res = match_game(sc, "Toronto at Cleveland Winner?", [game])
    assert res is not None
    _, team = res
    # "Toronto" is mentioned first → that's the YES team.
    assert team == "Toronto Raptors"


def test_match_game_et_date_wraps_to_next_utc_day() -> None:
    """Regression: NBA 8pm-ET tips convert to 00:00 UTC the *next* day.

    Kalshi ticker date is in America/New_York; matching on UTC date alone
    drops every primetime NBA/NHL game on the east coast.
    """
    sc = parse_sports_contract("KXNBAGAME", "KXNBAGAME-26APR18TORCLE")
    assert sc is not None
    # 2026-04-18 20:00 America/New_York = 2026-04-19 00:00 UTC.
    game = Game(
        sport_key="basketball_nba",
        game_id="g1",
        commence_time=datetime(2026, 4, 19, 0, 0, tzinfo=timezone.utc),
        home_team="Cleveland Cavaliers",
        away_team="Toronto Raptors",
        outcomes=[],
        fetched_at=datetime.now(timezone.utc),
    )
    res = match_game(sc, "Will the Raptors beat the Cavaliers?", [game])
    assert res is not None
    _, team = res
    assert team == "Toronto Raptors"


def test_match_game_date_mismatch_returns_none() -> None:
    sc = parse_sports_contract("KXMLBGAME", "KXMLBGAME-26APR181605CWSATH")
    assert sc is not None
    games = [_game("Chicago White Sox", "Oakland Athletics", date(2026, 4, 19))]
    assert match_game(sc, "Will the White Sox win?", games) is None


def test_match_game_subtitle_agrees_accepts() -> None:
    """When Kalshi's yes_sub_title names the YES team, match_game requires
    the chosen team to overlap with it. Happy path: they agree."""
    sc = parse_sports_contract("KXMLBGAME", "KXMLBGAME-26APR181605CWSATH")
    assert sc is not None
    games = [_game("Chicago White Sox", "Oakland Athletics", date(2026, 4, 18))]
    res = match_game(
        sc, "Will the White Sox beat the Athletics?", games,
        yes_subtitle="White Sox",
    )
    assert res is not None
    _, team = res
    assert team == "Chicago White Sox"


def test_match_game_subtitle_disagrees_abstains() -> None:
    """Regression: if title-position says YES = A but Kalshi subtitle says
    YES = B, we abstain rather than ship a sign-flipped forecast.
    This is the sports analog of the weather T61-less-than bug."""
    sc = parse_sports_contract("KXMLBGAME", "KXMLBGAME-26APR181605CWSATH")
    assert sc is not None
    games = [_game("Chicago White Sox", "Oakland Athletics", date(2026, 4, 18))]
    # Title-position would pick White Sox, but subtitle says YES = Athletics.
    res = match_game(
        sc, "Will the White Sox beat the Athletics?", games,
        yes_subtitle="Oakland Athletics",
    )
    assert res is None


def test_match_game_ambiguous_same_city() -> None:
    sc = parse_sports_contract("KXMLBGAME", "KXMLBGAME-26APR181605CWSATH")
    assert sc is not None
    games = [
        _game("Chicago White Sox", "Oakland Athletics", date(2026, 4, 18)),
        _game("Chicago Cubs", "Los Angeles Dodgers", date(2026, 4, 18)),
    ]
    # "Chicago" hits both — we abstain.
    assert match_game(sc, "Will Chicago win?", games) is None


def test_devig_book_symmetric() -> None:
    # Two even-money outcomes at 1.91 each -> ~52.36% implied each -> devig to 50/50.
    out = devig_book({"A": 1.91, "B": 1.91})
    assert out["A"] == pytest.approx(0.5, abs=1e-9)
    assert out["B"] == pytest.approx(0.5, abs=1e-9)


def test_devig_book_asymmetric() -> None:
    # Favorite at 1.50 (66.67%), dog at 3.00 (33.33%). Sum = 100%: already fair, devig = identity.
    out = devig_book({"A": 1.50, "B": 3.00})
    assert out["A"] == pytest.approx(2 / 3, abs=1e-9)
    assert out["B"] == pytest.approx(1 / 3, abs=1e-9)


def test_devig_book_with_vig() -> None:
    # 2.00 / 2.00 has 100% implied, 1.90/1.90 has more — same fair 50/50 after devig.
    out = devig_book({"A": 1.90, "B": 1.90})
    assert out["A"] == pytest.approx(0.5, abs=1e-9)


def test_devig_book_drops_nonpositive() -> None:
    assert devig_book({"A": 0.0, "B": 2.0}) == {"B": pytest.approx(1.0)}


def test_h2h_odds_uses_events_then_per_event_and_filters_by_commence() -> None:
    """Client should list events, then pull per-event odds only for events
    that commence on/before the window; far-future events are skipped so
    we don't pay quota for games Kalshi isn't trading today."""
    client = TheOddsAPIClient(api_key="k")
    calls: list[tuple[str, dict[str, str]]] = []

    def fake_get(path: str, params: dict[str, str]) -> object:
        calls.append((path, dict(params)))
        if path == "/sports/baseball_mlb/events":
            return [
                {"id": "e1", "commence_time": "2026-04-18T23:00:00Z",
                 "home_team": "Chicago White Sox", "away_team": "Oakland Athletics"},
                {"id": "e2", "commence_time": "2026-04-25T23:00:00Z",
                 "home_team": "Toronto Blue Jays", "away_team": "Detroit Tigers"},
            ]
        if path == "/sports/baseball_mlb/events/e1/odds":
            return {
                "id": "e1", "commence_time": "2026-04-18T23:00:00Z",
                "home_team": "Chicago White Sox", "away_team": "Oakland Athletics",
                "bookmakers": [
                    {"key": "fanduel", "markets": [
                        {"key": "h2h", "outcomes": [
                            {"name": "Chicago White Sox", "price": 1.83},
                            {"name": "Oakland Athletics", "price": 2.05},
                        ]},
                    ]},
                ],
            }
        raise AssertionError(f"unexpected call: {path}")

    client._get = fake_get  # type: ignore[assignment]
    commence_before = datetime(2026, 4, 20, 0, 0, tzinfo=timezone.utc)
    games = client.h2h_odds("baseball_mlb", commence_before=commence_before)

    paths = [c[0] for c in calls]
    assert paths == [
        "/sports/baseball_mlb/events",
        "/sports/baseball_mlb/events/e1/odds",
    ], f"expected exactly one per-event odds call (e1), got: {paths}"
    assert len(games) == 1
    assert games[0].game_id == "e1"
    assert games[0].home_team == "Chicago White Sox"
    assert len(games[0].outcomes) == 2
    assert games[0].outcomes[0].bookmaker == "fanduel"


def test_series_to_sport_keys_stable() -> None:
    assert SERIES_TO_SPORT["KXNBAGAME"] == "basketball_nba"
    assert SERIES_TO_SPORT["KXMLBGAME"] == "baseball_mlb"
    assert SERIES_TO_SPORT["KXNHL"] == "icehockey_nhl"
