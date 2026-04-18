"""Match a Kalshi contract to a Polymarket market by title + date window.

Used by the rates_ and politics_ forecasters. Both follow the same
pattern: pull a (cached) page of active Polymarket markets filtered by a
small keyword set, then pick at most one whose *question* resembles our
Kalshi contract's *title* and whose end date is close to the Kalshi
close time.

We score similarity with a Jaccard-like token overlap on lowercased
alphanumeric tokens after stopword removal. This is intentionally simple
— cross-venue resolution criteria differ in small ways ("the Fed" vs
"FOMC", "wins the" vs "winner of"), and we'd rather miss than misattribute.

The matcher returns at most one market. If the top match is within
``MARGIN`` of the runner-up, we abstain — ambiguity is a signal the
resolution criteria diverge and the Polymarket prior would be
unreliable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from kalshi_edge.data_sources.polymarket import PolymarketClient, PolymarketMarket


_STOPWORDS = frozenset({
    "a", "an", "the", "will", "be", "is", "are", "at", "in", "on", "of",
    "to", "for", "by", "with", "and", "or", "this", "that", "do", "does",
    "have", "has", "it", "its", "as", "from", "how", "which", "what",
    "when", "where", "who", "whom", "whose", "why", "not", "than",
    "then", "would", "could", "should", "may", "might", "over", "under",
    "between", "above", "below", "up", "down", "into", "out",
})


@dataclass(frozen=True, slots=True)
class PolymarketMatch:
    market: PolymarketMarket
    score: float
    runner_up_score: float


_MIN_SCORE = 0.30    # ≥ 30% Jaccard overlap to count as a hit at all.
_MARGIN = 0.10       # winner must beat runner-up by this much.


def _tokens(s: str) -> set[str]:
    """Lowercased alphanumeric tokens with stopwords removed."""
    out: set[str] = set()
    cur: list[str] = []
    for ch in s.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                w = "".join(cur)
                if w not in _STOPWORDS and len(w) > 1:
                    out.add(w)
                cur = []
    if cur:
        w = "".join(cur)
        if w not in _STOPWORDS and len(w) > 1:
            out.add(w)
    return out


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def find_best_match(
    *,
    client: PolymarketClient,
    search_keywords: list[str],
    title: str,
    target_window: tuple[datetime, datetime] | None = None,
    extra_required_tokens: set[str] | None = None,
    limit: int = 500,
) -> PolymarketMatch | None:
    """Return the uniquely-best Polymarket market for a Kalshi contract.

    ``search_keywords`` — any-match filter used on Polymarket's side.
    ``title``           — the Kalshi contract title we're matching against.
    ``target_window``   — if given, Polymarket endDate must fall inside.
    ``extra_required_tokens`` — tokens that must all appear in the
        Polymarket question (AND). Use this to enforce e.g. the target
        meeting month/year on FOMC markets where the keyword search is
        too permissive by itself.

    Returns ``None`` when no market clears ``_MIN_SCORE`` or when the
    top two are within ``_MARGIN`` of each other.
    """
    pool = client.search_active(keywords=search_keywords, limit=limit)
    if not pool:
        return None

    kalshi_tokens = _tokens(title)
    if not kalshi_tokens:
        return None

    scored: list[tuple[float, PolymarketMarket]] = []
    for m in pool:
        if target_window is not None and m.end_date is not None:
            ed = m.end_date
            if ed.tzinfo is None:
                ed = ed.replace(tzinfo=timezone.utc)
            lo, hi = target_window
            if ed < lo or ed > hi:
                continue
        q_tokens = _tokens(m.question)
        if extra_required_tokens and not extra_required_tokens.issubset(q_tokens):
            continue
        score = jaccard(kalshi_tokens, q_tokens)
        if score > 0.0:
            scored.append((score, m))

    if not scored:
        return None
    scored.sort(key=lambda t: t[0], reverse=True)
    best_score, best = scored[0]
    runner = scored[1][0] if len(scored) > 1 else 0.0
    if best_score < _MIN_SCORE:
        return None
    if best_score - runner < _MARGIN:
        return None
    return PolymarketMatch(market=best, score=best_score, runner_up_score=runner)


def default_window(anchor: datetime, *, days: int = 45) -> tuple[datetime, datetime]:
    if anchor.tzinfo is None:
        anchor = anchor.replace(tzinfo=timezone.utc)
    delta = timedelta(days=days)
    return (anchor - delta, anchor + delta)
