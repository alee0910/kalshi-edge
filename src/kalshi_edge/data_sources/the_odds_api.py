"""the-odds-api.com client.

We treat bookmaker prices as a *market-ensemble* prior: each bookmaker
publishes an implied probability on top of a vig; devigging (via
multiplicative normalization, which is the ML estimator under the
assumption that the vig is distributed symmetrically across outcomes)
gives that book's fair-odds estimate. We then carry the set of per-book
devigged probabilities forward as samples — the weather forecaster's
Bayesian bootstrap is exactly analogous.

Caching: 15 min TTL. Pre-game moneylines move slowly; within-minute
refresh is wasted quota. The free tier is 500 req/month so aggressive
caching also protects the key.

Sport keys we target (stable keys from the-odds-api docs):
    basketball_nba, icehockey_nhl, baseball_mlb

Tennis is per-tournament (tennis_atp_<slug>) and too volatile to hard-code
here; the sports forecaster abstains on ATP for v1.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests
import requests_cache
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from kalshi_edge.logging_ import get_logger

log = get_logger(__name__)

_BASE = "https://api.the-odds-api.com/v4"

# Kalshi series prefix -> odds-api sport key.
SERIES_TO_SPORT = {
    "KXNBAGAME": "basketball_nba",
    "KXNHL": "icehockey_nhl",
    "KXMLBGAME": "baseball_mlb",
}

_RETRYABLE = (requests.ConnectionError, requests.Timeout)


@dataclass(frozen=True, slots=True)
class BookOutcome:
    """One bookmaker's outcome (team + decimal price) for a game."""
    bookmaker: str
    team: str
    decimal_price: float


@dataclass(frozen=True, slots=True)
class Game:
    """One game as reported by the-odds-api, normalized across books."""
    sport_key: str
    game_id: str
    commence_time: datetime       # UTC
    home_team: str
    away_team: str
    outcomes: list[BookOutcome]   # h2h only; both teams across all books
    fetched_at: datetime


class TheOddsAPIClient:
    """Thin, cached wrapper over the-odds-api. No authentication header —
    key is a URL param. We keep the key out of logs by never stringifying
    the full request URL."""

    def __init__(
        self, api_key: str, *, cache_path: str | None = None,
        ttl_seconds: int = 900,
    ) -> None:
        if not api_key:
            raise ValueError("the-odds-api api_key is required")
        self.api_key = api_key
        backend: Any = requests_cache.SQLiteCache(cache_path) if cache_path else "memory"
        self.session = requests_cache.CachedSession(
            backend=backend,
            expire_after=ttl_seconds,
            allowable_methods=("GET",),
            cache_control=False,
        )
        self.session.headers["User-Agent"] = "kalshi-edge/0.1 (sports forecaster)"

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=5),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    def _get(self, path: str, params: dict[str, str]) -> Any:
        full = dict(params)
        full["apiKey"] = self.api_key
        url = f"{_BASE}{path}"
        r = self.session.get(url, params=full, timeout=15)
        r.raise_for_status()
        return r.json()

    def h2h_odds(self, sport_key: str) -> list[Game]:
        """Moneyline (head-to-head) odds for every currently-upcoming game
        in a sport. Returns [] if none are in the API window."""
        raw = self._get(
            f"/sports/{sport_key}/odds",
            params={"regions": "us", "markets": "h2h", "oddsFormat": "decimal"},
        )
        if not isinstance(raw, list):
            log.warning("odds_api_unexpected_shape", sport=sport_key)
            return []
        now = datetime.utcnow().replace(microsecond=0)
        games: list[Game] = []
        for g in raw:
            try:
                outs: list[BookOutcome] = []
                for book in g.get("bookmakers", []):
                    bname = book.get("key") or book.get("title") or "unknown"
                    for mkt in book.get("markets", []):
                        if mkt.get("key") != "h2h":
                            continue
                        for o in mkt.get("outcomes", []):
                            team = str(o.get("name") or "")
                            price = o.get("price")
                            if not team or price is None:
                                continue
                            try:
                                outs.append(BookOutcome(
                                    bookmaker=str(bname),
                                    team=team,
                                    decimal_price=float(price),
                                ))
                            except (TypeError, ValueError):
                                continue
                games.append(Game(
                    sport_key=sport_key,
                    game_id=str(g.get("id") or ""),
                    commence_time=_parse_iso(g.get("commence_time")),
                    home_team=str(g.get("home_team") or ""),
                    away_team=str(g.get("away_team") or ""),
                    outcomes=outs,
                    fetched_at=now,
                ))
            except Exception as e:  # noqa: BLE001
                log.warning("odds_api_game_parse_failed", error=str(e))
                continue
        return games


def _parse_iso(s: Any) -> datetime:
    if not s:
        return datetime.utcnow()
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return datetime.utcnow()


def devig_book(book_prices: dict[str, float]) -> dict[str, float]:
    """Multiplicative devigging: p_i = (1/o_i) / sum(1/o_j).

    This is the simplest and most common method; it's the MLE under the
    assumption that the vig is distributed proportionally to the implied
    probability. More sophisticated alternatives (Shin, power) correct for
    favorite-longshot bias but require additional data. For US moneylines
    the difference is usually < 1pt and not worth the complexity.
    """
    inv = {team: 1.0 / float(o) for team, o in book_prices.items() if o and o > 0}
    s = sum(inv.values())
    if s <= 0:
        return {}
    return {team: v / s for team, v in inv.items()}
