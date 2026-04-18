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
from datetime import date, datetime

from kalshi_edge.data_sources.the_odds_api import Game, SERIES_TO_SPORT

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

    odds-api returns e.g. "Chicago White Sox"; Kalshi titles may say
    "White Sox", "Sox", or "Chicago". We check the last word (mascot) and
    the full string. Last word is usually the strongest signal and also
    the only one common to both compact and verbose titles.
    """
    t = team.strip()
    if not t:
        return []
    parts = t.split()
    out = [t.lower()]
    if len(parts) >= 2:
        # last 2 words (e.g. "White Sox"), and the last word alone
        out.append(" ".join(parts[-2:]).lower())
        out.append(parts[-1].lower())
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
        if g.commence_time.date() != sc.target_date:
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
