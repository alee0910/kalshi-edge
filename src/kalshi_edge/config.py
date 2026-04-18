"""Configuration loading.

Merges three sources, in order of increasing precedence:
  1. YAML at ``KALSHI_EDGE_CONFIG`` (default: config/default.yaml).
  2. Environment variables (``KALSHI_EDGE_*``, ``FRED_API_KEY``, Kalshi/Gmail creds).
  3. Explicit overrides passed at call sites (tests mostly).

Pydantic v2 validates everything. A failed validation should fail loud — a
silently-misconfigured forecaster is worse than a crashed one.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator


class KalshiConfig(BaseModel):
    base_url: str = "https://api.elections.kalshi.com/trade-api/v2"
    requests_per_second: float = 5.0
    max_retries: int = 5
    retry_backoff_seconds: float = 1.5
    cache_ttl: dict[str, int] = Field(default_factory=dict)

    api_key_id: str | None = None
    private_key_path: Path | None = None
    env: str = "prod"


class UniverseFilterConfig(BaseModel):
    min_dte_days: float = 0.1
    min_volume: float = 500.0
    max_spread_cents: float = 8.0
    allowed_categories: list[str] = Field(default_factory=list)
    status: str = "open"


class StorageConfig(BaseModel):
    sqlite_path: Path = Path("./data/kalshi_edge.db")
    samples_dir: Path = Path("./data/samples")
    wal_mode: bool = True


class SchedulerConfig(BaseModel):
    market_refresh_minutes: int = 15
    forecast_refresh_minutes: int = 60
    calibration_refresh_hours: int = 24
    digest_cron: str = "0 7 * * *"
    timezone: str = "America/New_York"


class RankerConfig(BaseModel):
    bankroll_dollars: float = 10000.0
    kelly_fraction: float = 0.25
    alert_edge_points: float = 15.0
    alert_model_confidence: float = 0.7
    fee_model: str = "kalshi_default_v1"
    correlation_penalty_lambda: float = 0.25

    @field_validator("kelly_fraction")
    @classmethod
    def _bounded_kelly(cls, v: float) -> float:
        if not 0 < v <= 1:
            raise ValueError("kelly_fraction must be in (0, 1]")
        return v


class CalibrationConfig(BaseModel):
    min_resolved_for_reliability: int = 30
    reliability_bins: int = 10
    isotonic_refit_days: int = 7


class AlertsConfig(BaseModel):
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 587
    use_tls: bool = True
    digest_top_n: int = 5

    gmail_address: str | None = None
    gmail_app_password: str | None = None
    recipient: str | None = None


class LoggingConfig(BaseModel):
    level: str = "INFO"
    renderer: str = "console"


class AppConfig(BaseModel):
    kalshi: KalshiConfig = Field(default_factory=KalshiConfig)
    universe_filter: UniverseFilterConfig = Field(default_factory=UniverseFilterConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    ranker: RankerConfig = Field(default_factory=RankerConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    alerts: AlertsConfig = Field(default_factory=AlertsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    data_dir: Path = Path("./data")
    fred_api_key: str | None = None

    categories: dict[str, dict[str, list[str]]] = Field(default_factory=dict)


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """Overlay environment variables onto the YAML-loaded dict."""
    kalshi = raw.setdefault("kalshi", {})
    if v := os.environ.get("KALSHI_API_KEY_ID"):
        kalshi["api_key_id"] = v
    if v := os.environ.get("KALSHI_PRIVATE_KEY_PATH"):
        kalshi["private_key_path"] = v
    if v := os.environ.get("KALSHI_ENV"):
        kalshi["env"] = v

    alerts = raw.setdefault("alerts", {})
    if v := os.environ.get("GMAIL_ADDRESS"):
        alerts["gmail_address"] = v
    if v := os.environ.get("GMAIL_APP_PASSWORD"):
        alerts["gmail_app_password"] = v
    if v := os.environ.get("ALERT_RECIPIENT"):
        alerts["recipient"] = v

    if v := os.environ.get("FRED_API_KEY"):
        raw["fred_api_key"] = v
    if v := os.environ.get("KALSHI_EDGE_DATA_DIR"):
        raw["data_dir"] = v

    logging_ = raw.setdefault("logging", {})
    if v := os.environ.get("KALSHI_EDGE_LOG_LEVEL"):
        logging_["level"] = v

    return raw


def load_config(
    yaml_path: Path | str | None = None,
    categories_path: Path | str | None = None,
) -> AppConfig:
    yaml_path = Path(yaml_path or os.environ.get("KALSHI_EDGE_CONFIG") or _REPO_ROOT / "config" / "default.yaml")
    categories_path = Path(categories_path or _REPO_ROOT / "config" / "categories.yaml")

    if not yaml_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {yaml_path}")

    with yaml_path.open() as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    raw = _apply_env_overrides(raw)

    if categories_path.exists():
        with categories_path.open() as f:
            raw["categories"] = yaml.safe_load(f) or {}

    return AppConfig.model_validate(raw)
