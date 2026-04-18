"""Kalshi market-data client.

Read-only endpoints at https://api.elections.kalshi.com/trade-api/v2 do not
require authentication; we still sign when credentials are available so the
same client handles portfolio / trading endpoints later.

Caching via requests-cache with per-endpoint TTLs (short for orderbook,
longer for contract metadata). Rate limited client-side. Retries transient
5xx / 429 with exponential backoff (tenacity).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import requests
import requests_cache
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from kalshi_edge.config import KalshiConfig
from kalshi_edge.logging_ import get_logger
from kalshi_edge.market._rate_limit import RateLimiter
from kalshi_edge.market.auth import KalshiCredentials, sign_request

log = get_logger(__name__)


class KalshiAPIError(RuntimeError):
    def __init__(self, status: int, body: str) -> None:
        super().__init__(f"Kalshi API {status}: {body[:500]}")
        self.status = status
        self.body = body


_RETRYABLE = (requests.ConnectionError, requests.Timeout, KalshiAPIError)


def _is_retryable_status(exc: BaseException) -> bool:
    if isinstance(exc, KalshiAPIError):
        return exc.status == 429 or 500 <= exc.status < 600
    return isinstance(exc, (requests.ConnectionError, requests.Timeout))


@dataclass
class Page:
    items: list[dict[str, Any]]
    cursor: str | None


class KalshiClient:
    """Thin wrapper over Kalshi's REST v2 API.

    Parameters
    ----------
    config:
        KalshiConfig (base URL, rate limit, TTLs).
    credentials:
        Optional. If provided, every request is signed even for public
        endpoints (harmless, and future-proofs against endpoints that
        change access mode).
    cache_path:
        Optional path for an on-disk sqlite cache. In-memory if None.
    """

    _ENDPOINTS = {
        "markets_list": "/markets",
        "market_detail": "/markets/{ticker}",
        "orderbook": "/markets/{ticker}/orderbook",
        "event_detail": "/events/{ticker}",
        "series_detail": "/series/{ticker}",
        # Historical candlesticks live under the series — Kalshi's API
        # requires the series ticker as a path parameter. Used for the
        # retro-market-calibration backtest.
        "candlesticks": "/series/{series}/markets/{ticker}/candlesticks",
    }

    def __init__(
        self,
        config: KalshiConfig,
        credentials: KalshiCredentials | None = None,
        cache_path: Path | str | None = None,
    ) -> None:
        self.config = config
        self.credentials = credentials
        self._limiter = RateLimiter(rate_per_second=config.requests_per_second)

        # requests-cache: per-URL TTL resolved via expire_after + urls_expire_after.
        urls_expire_after = {
            # requests-cache matches patterns as `host/path`.
            "*/trade-api/v2/markets": config.cache_ttl.get("markets_list", 60),
            "*/trade-api/v2/markets/*": config.cache_ttl.get("market_detail", 60),
            "*/trade-api/v2/markets/*/orderbook": config.cache_ttl.get("orderbook", 5),
            "*/trade-api/v2/events/*": config.cache_ttl.get("event_detail", 300),
            "*/trade-api/v2/series/*": config.cache_ttl.get("series_detail", 3600),
        }
        # requests-cache exposes SQLiteCache as a class; "memory" is a string
        # backend id for the default in-process dict-backed cache.
        backend: Any = requests_cache.SQLiteCache(cache_path) if cache_path else "memory"
        self.session = requests_cache.CachedSession(
            backend=backend,
            urls_expire_after=urls_expire_after,
            expire_after=60,
            allowable_methods=("GET",),
            cache_control=False,
            stale_if_error=False,
        )
        self.session.headers.update({"Accept": "application/json", "User-Agent": "kalshi-edge/0.1"})

    # ------- core request pipeline --------------------------------------
    def _url(self, path: str) -> str:
        return f"{self.config.base_url.rstrip('/')}{path}"

    @retry(
        retry=retry_if_exception_type(_RETRYABLE),
        stop=stop_after_attempt(5),
        wait=wait_exponential_jitter(initial=1.0, max=20.0),
        reraise=True,
    )
    def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = self._url(path)
        headers: dict[str, str] = {}
        if self.credentials is not None:
            headers.update(sign_request(self.credentials, "GET", path))

        # We rate-limit before every call. Cache-hit sleeps only occur when the
        # limiter is already depleted; the cost is negligible in our scheduler
        # pattern and the code stays simple.
        self._limiter.acquire()

        resp = self.session.get(url, params=params, headers=headers, timeout=20)
        if not resp.ok:
            err = KalshiAPIError(resp.status_code, resp.text)
            # Non-retryable 4xx raises; retryable (429 / 5xx) re-raises into tenacity.
            raise err
        try:
            return resp.json()
        except ValueError as e:
            raise KalshiAPIError(resp.status_code, resp.text) from e

    # ------- read endpoints ---------------------------------------------
    def list_markets(
        self,
        *,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        status: str | None = "open",
        limit: int = 1000,
        cursor: str | None = None,
    ) -> Page:
        params: dict[str, Any] = {"limit": min(max(limit, 1), 1000)}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if event_ticker:
            params["event_ticker"] = event_ticker
        if status:
            params["status"] = status
        if cursor:
            params["cursor"] = cursor
        data = self._get(self._ENDPOINTS["markets_list"], params=params)
        return Page(items=list(data.get("markets", [])), cursor=data.get("cursor") or None)

    def iter_markets(
        self,
        *,
        series_ticker: str | None = None,
        event_ticker: str | None = None,
        status: str | None = "open",
        max_items: int | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yield every market matching the filters, transparently paginating."""
        cursor: str | None = None
        yielded = 0
        while True:
            page = self.list_markets(
                series_ticker=series_ticker,
                event_ticker=event_ticker,
                status=status,
                cursor=cursor,
            )
            for m in page.items:
                yield m
                yielded += 1
                if max_items is not None and yielded >= max_items:
                    return
            if not page.cursor:
                return
            cursor = page.cursor

    def get_market(self, ticker: str) -> dict[str, Any]:
        data = self._get(self._ENDPOINTS["market_detail"].format(ticker=ticker))
        return dict(data.get("market") or data)

    def get_orderbook(self, ticker: str) -> dict[str, Any]:
        data = self._get(self._ENDPOINTS["orderbook"].format(ticker=ticker))
        return dict(data.get("orderbook") or data)

    def get_event(self, ticker: str) -> dict[str, Any]:
        data = self._get(self._ENDPOINTS["event_detail"].format(ticker=ticker))
        return dict(data)

    def get_series(self, ticker: str) -> dict[str, Any]:
        data = self._get(self._ENDPOINTS["series_detail"].format(ticker=ticker))
        return dict(data)

    def get_candlesticks(
        self,
        series_ticker: str,
        market_ticker: str,
        *,
        start_ts: int | datetime,
        end_ts: int | datetime,
        period_interval: int = 60,
    ) -> list[dict[str, Any]]:
        """Fetch historical candlesticks for one settled/live market.

        Kalshi's candlesticks endpoint returns OHLC + volume bars at a
        chosen interval (minutes). We use it to recover the closing market
        price for a settled contract without needing a live orderbook.

        ``start_ts`` / ``end_ts`` may be given as Unix seconds or datetimes;
        Kalshi caps a single call's window (typically 5000 bars), so the
        caller is expected to size intervals appropriately for long ranges.
        """
        params = {
            "start_ts": _unix_seconds(start_ts),
            "end_ts": _unix_seconds(end_ts),
            "period_interval": int(period_interval),
        }
        data = self._get(
            self._ENDPOINTS["candlesticks"].format(
                series=series_ticker, ticker=market_ticker,
            ),
            params=params,
        )
        return list(data.get("candlesticks") or [])


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _unix_seconds(ts: int | datetime) -> int:
    """Normalize a (timezone-aware) datetime or int to Unix seconds."""
    if isinstance(ts, datetime):
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return int(ts.timestamp())
    return int(ts)
