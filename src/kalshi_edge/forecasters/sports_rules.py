"""Parse Kalshi sports contracts + match them to the-odds-api games.

Kalshi's event_ticker for a game market looks like
``KXMLBGAME-26APR181605CWSATH`` — series / yy+Mon+dd / HHMM / team-code
salad. Rather than decoding the team codes (which differ per league and
aren't officially documented), we fuzzy-match the contract title against
odds-api's English team names. This is far more robust and falls through
gracefully when titles drift.

We support NBA, NHL, MLB head-to-head markets. ATP tennis is a
per-tournament key family at the-odds-api and too volatile to hardcode;
it abstains in v1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo

from kalshi_edge.data_sources.the_odds_api import Game, SERIES_TO_SPORT

# Kalshi encodes event dates in US/Eastern (the league schedule's local
# calendar day), while the-odds-api returns commence_time in UTC. A
# 20:00 ET tip converts to 00:00 UTC the *next* calendar day, so a naive
# UTC-date comparison drops most primetime NBA/NHL games. We normalize
# by comparing the game's ET-local date to the ticker's encoded date.
_KALSHI_TZ = ZoneInfo("America/New_York")

_DATE_RE = re.compile(r"-(\d{2})([A-Z]{3})(\d{2})")
_MONTHS = {m: i for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
     "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"], start=1)}


@dataclass(frozen=True, slots=True)
class SportsContract:
    sport_key: str
    target_date: date   # UTC date of the scheduled game


def parse_sports_contract(
    series_ticker: str, event_ticker: str,
) -> SportsContract | None:
    sport = SERIES_TO_SPORT.get(series_ticker.upper())
    if sport is None:
        return None
    m = _DATE_RE.search(event_ticker or "")
    if not m:
        return None
    yy, mon, dd = m.groups()
    month = _MONTHS.get(mon)
    if month is None:
        return None
    try:
        d = date(2000 + int(yy), month, int(dd))
    except ValueError:
        return None
    return SportsContract(sport_key=sport, target_date=d)


def _team_tokens(team: str) -> list[str]:
    """Return candidate substrings that a Kalshi title might contain.

    odds-api returns verbose names like "Toronto Raptors" or "Chicago
    White Sox". Kalshi titles come in multiple formats across leagues:

    - NBA: "Toronto at Cleveland Winner?"       (city only)
    - MLB: "Will the White Sox beat the Athletics?"  (mascot only)
    - NHL: occasionally full name.

    So we check the full string, the last 1-2 words (mascot), *and* the
    leading words (city). When both teams in a game share a city
    (Los Angeles Lakers/Clippers, Chicago Cubs/White Sox), match_game's
    ambiguity check still kicks in and we abstain rather than guess.
    """
    t = team.strip()
    if not t:
        return []
    parts = t.split()
    out = [t.lower()]
    if len(parts) >= 2:
        out.append(" ".join(parts[-2:]).lower())   # last 2 words (often "City Mascot")
        out.append(parts[-1].lower())               # mascot alone (e.g. "Raptors")
        out.append(" ".join(parts[:-1]).lower())   # city if mascot is 1 word
        # Also handle 3-word teams where the mascot itself is 2 words
        # ("White Sox", "Red Sox", "Blue Jays"). The single-word "city" is
        # "Chicago" / "Boston" / "Toronto". Skip short leading particles
        # ("Los", "San", "New") — too noisy to match against.
        if len(parts) >= 3:
            city_alt = " ".join(parts[:-2]).lower()
            if len(city_alt) >= 4:
                out.append(city_alt)
    elif parts:
        out.append(parts[0].lower())
    return out


def match_game(
    sc: SportsContract, title: str, games: list[Game],
) -> tuple[Game, str] | None:
    """Pick the odds-api game + YES-team corresponding to a Kalshi contract.

    Returns (game, target_team) or None if no unambiguous match.
    Title-position disambiguates when both teams appear: Kalshi phrasing
    "Will <X> beat <Y>?" puts the YES team first.
    """
    title_l = (title or "").lower()
    candidates: list[tuple[Game, str, int]] = []  # game, team, earliest-hit position
    for g in games:
        ct = g.commence_time
        if ct.tzinfo is None:
            ct = ct.replace(tzinfo=timezone.utc)
        if ct.astimezone(_KALSHI_TZ).date() != sc.target_date:
            continue
        home_pos = _first_hit(title_l, _team_tokens(g.home_team))
        away_pos = _first_hit(title_l, _team_tokens(g.away_team))
        if home_pos is None and away_pos is None:
            continue
        if home_pos is not None and away_pos is None:
            candidates.append((g, g.home_team, home_pos))
        elif away_pos is not None and home_pos is None:
            candidates.append((g, g.away_team, away_pos))
        else:
            # Both teams mentioned. Earlier-mentioned team is YES in Kalshi phrasing.
            assert home_pos is not None and away_pos is not None
            if home_pos < away_pos:
                candidates.append((g, g.home_team, home_pos))
            else:
                candidates.append((g, g.away_team, away_pos))
    if not candidates:
        return None
    # If multiple same-date games both show a hit (e.g. "Chicago" matches
    # two games), we can't disambiguate — abstain rather than guess.
    if len(candidates) > 1:
        return None
    g, team, _ = candidates[0]
    return (g, team)


def _first_hit(haystack: str, needles: list[str]) -> int | None:
    best: int | None = None
    for n in needles:
        if not n:
            continue
        i = haystack.find(n)
        if i >= 0 and (best is None or i < best):
            best = i
    return best
