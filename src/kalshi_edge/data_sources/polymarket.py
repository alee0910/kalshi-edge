"""Polymarket gamma-api client.

Polymarket's public gamma-api publishes a searchable list of active
prediction-market contracts with their current YES prices. We use it as
a *cross-market Bayesian prior*: when a Kalshi contract has a
reasonable Polymarket counterpart (same resolution criterion, same
tick-date), Polymarket's price is a second liquid estimator of the true
probability. For categories where we have no first-principles model
(rates/FOMC, US politics), this is the best calibrated source we can
ship without custom scrapers or paid feeds.

Endpoint: ``https://gamma-api.polymarket.com/markets`` — no auth.

Notable field shape (2026 wire):
    id             str
    slug           str
    conditionId    str
    question       str
    outcomes       JSON-string of a list, e.g. '["Yes","No"]'
    outcomePrices  JSON-string of a list of str decimals '["0.56","0.44"]'
    volume         float
    liquidity      float
    closed         bool
    active         bool
    endDate        ISO8601

We depend on very little of this to keep resilience to wire drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests
import requests_cache
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter

from kalshi_edge.logging_ import get_logger

log = get_logger(__name__)

_BASE = "https://gamma-api.polymarket.com"
_RETRYABLE = (requests.ConnectionError, requests.Timeout)


@dataclass(frozen=True, slots=True)
class PolymarketMarket:
    id: str
    slug: str
    condition_id: str
    question: str
    yes_price: float | None     # decimal 0..1 or None if unresolved
    no_price: float | None
    volume: float
    liquidity: float
    end_date: datetime | None
    fetched_at: datetime


class PolymarketClient:
    def __init__(self, cache_path: str | None = None, ttl_seconds: int = 300) -> None:
        backend: Any = requests_cache.SQLiteCache(cache_path) if cache_path else "memory"
        self.session = requests_cache.CachedSession(
            backend=backend,
            expire_after=ttl_seconds,
            allowable_methods=("GET",),
            cache_control=False,
        )
        self.session.headers["User-Agent"] = "kalshi-edge/0.1 (cross-market prior)"

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential_jitter(initial=0.5, max=5),
        retry=retry_if_exception_type(_RETRYABLE),
    )
    def _get(self, path: str, params: dict[str, Any]) -> Any:
        r = self.session.get(f"{_BASE}{path}", params=params, timeout=15)
        r.raise_for_status()
        return r.json()

    def by_slug(self, slug: str) -> PolymarketMarket | None:
        raw = self._get("/markets", {"slug": slug})
        items = raw if isinstance(raw, list) else [raw]
        for it in items:
            if not isinstance(it, dict):
                continue
            m = _parse_market(it)
            if m is not None and m.slug == slug:
                return m
        return None

    def search_active(
        self, keywords: list[str], *, limit: int = 500,
    ) -> list[PolymarketMarket]:
        """Active, non-closed markets matching any keyword in question/slug.

        Polymarket's list endpoint doesn't do full-text search, so we
        pull the active page and filter client-side. Combined with the
        response cache this costs ~0 bandwidth on a warm process.
        """
        raw = self._get("/markets", {
            "closed": "false", "active": "true", "limit": str(limit),
        })
        if not isinstance(raw, list):
            return []
        kws = [k.lower() for k in keywords if k]
        out: list[PolymarketMarket] = []
        for it in raw:
            if not isinstance(it, dict):
                continue
            blob = (
                str(it.get("question") or "") + " " +
                str(it.get("slug") or "")
            ).lower()
            if kws and not any(k in blob for k in kws):
                continue
            m = _parse_market(it)
            if m is not None:
                out.append(m)
        return out


def _parse_market(it: dict[str, Any]) -> PolymarketMarket | None:
    try:
        outcomes = _parse_json_list(it.get("outcomes"))
        prices = _parse_json_list(it.get("outcomePrices"))
        yes_price: float | None = None
        no_price: float | None = None
        if outcomes and prices and len(outcomes) == len(prices):
            for o, p in zip(outcomes, prices):
                olc = str(o).strip().lower()
                try:
                    pv = float(p)
                except (TypeError, ValueError):
                    continue
                if olc in ("yes", "true", "y"):
                    yes_price = pv
                elif olc in ("no", "false", "n"):
                    no_price = pv
        end_date = _parse_iso(it.get("endDate") or it.get("end_date"))
        return PolymarketMarket(
            id=str(it.get("id") or ""),
            slug=str(it.get("slug") or ""),
            condition_id=str(it.get("conditionId") or it.get("condition_id") or ""),
            question=str(it.get("question") or ""),
            yes_price=yes_price,
            no_price=no_price,
            volume=float(it.get("volume") or 0.0),
            liquidity=float(it.get("liquidity") or 0.0),
            end_date=end_date,
            fetched_at=datetime.utcnow(),
        )
    except Exception as e:  # noqa: BLE001
        log.warning("polymarket_parse_failed", error=str(e))
        return None


def _parse_json_list(v: Any) -> list[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    try:
        return list(json.loads(v))
    except (TypeError, ValueError):
        return []


def _parse_iso(s: Any) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except ValueError:
        return None
