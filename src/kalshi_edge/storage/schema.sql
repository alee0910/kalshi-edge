-- kalshi-edge SQLite schema v1.
-- Applied by storage.db.Database.init_schema(). Keep idempotent.
-- Keep numeric prices in CENTS to match Kalshi's wire format.

PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- One row per unique contract (ticker).
CREATE TABLE IF NOT EXISTS contracts (
    ticker              TEXT PRIMARY KEY,
    event_ticker        TEXT NOT NULL,
    series_ticker       TEXT NOT NULL,
    category            TEXT NOT NULL,
    title               TEXT NOT NULL,
    subtitle            TEXT,
    status              TEXT NOT NULL,
    open_time           TIMESTAMP,
    close_time          TIMESTAMP,
    expiration_time     TIMESTAMP,
    rules_primary       TEXT,
    rules_secondary     TEXT,
    resolution_criteria TEXT,            -- JSON
    raw                 TEXT,            -- JSON, Kalshi response pass-through
    created_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_contracts_series  ON contracts(series_ticker);
CREATE INDEX IF NOT EXISTS idx_contracts_event   ON contracts(event_ticker);
CREATE INDEX IF NOT EXISTS idx_contracts_close   ON contracts(close_time);
CREATE INDEX IF NOT EXISTS idx_contracts_cat     ON contracts(category);

-- Time-series of market state.
CREATE TABLE IF NOT EXISTS market_snapshots (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker             TEXT NOT NULL REFERENCES contracts(ticker) ON DELETE CASCADE,
    ts                 TIMESTAMP NOT NULL,
    yes_bid            REAL,
    yes_ask            REAL,
    no_bid             REAL,
    no_ask             REAL,
    last_price         REAL,
    last_trade_ts      TIMESTAMP,
    volume             REAL NOT NULL DEFAULT 0,
    volume_24h         REAL NOT NULL DEFAULT 0,
    open_interest      REAL NOT NULL DEFAULT 0,
    liquidity          REAL,
    orderbook_yes      TEXT,              -- JSON list of [price, size]
    orderbook_no       TEXT,
    UNIQUE(ticker, ts)
);
CREATE INDEX IF NOT EXISTS idx_snap_ticker_ts ON market_snapshots(ticker, ts);

-- Every forecast produced, keyed by forecaster + version + ts.
CREATE TABLE IF NOT EXISTS forecasts (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker                TEXT NOT NULL REFERENCES contracts(ticker) ON DELETE CASCADE,
    ts                    TIMESTAMP NOT NULL,
    forecaster            TEXT NOT NULL,
    version               TEXT NOT NULL,
    p_yes                 REAL,           -- posterior mean
    p_yes_std             REAL,           -- epistemic spread
    p_yes_p05             REAL,
    p_yes_p95             REAL,
    binary_posterior      BLOB,           -- np.save bytes or JSON text
    binary_posterior_kind TEXT,           -- 'samples' | 'parametric'
    binary_posterior_params TEXT,         -- JSON if parametric
    underlying_posterior  BLOB,           -- samples of underlying continuous
    underlying_posterior_kind TEXT,
    underlying_posterior_params TEXT,
    uncertainty           TEXT,           -- JSON
    model_confidence      REAL,
    methodology           TEXT,           -- JSON
    data_sources          TEXT,           -- JSON
    diagnostics           TEXT,           -- JSON
    null_reason           TEXT,
    UNIQUE(ticker, ts, forecaster, version)
);
CREATE INDEX IF NOT EXISTS idx_fcst_ticker_ts ON forecasts(ticker, ts);
CREATE INDEX IF NOT EXISTS idx_fcst_fc_ts     ON forecasts(forecaster, ts);

CREATE TABLE IF NOT EXISTS resolutions (
    ticker        TEXT PRIMARY KEY REFERENCES contracts(ticker) ON DELETE CASCADE,
    resolved_at   TIMESTAMP NOT NULL,
    outcome       TEXT NOT NULL,          -- YES | NO | VOID
    settled_price REAL
);

-- Ranker-emitted signals keyed by (ts, ticker). We keep every signal so the
-- dashboard can backtest the ranker against outcomes.
CREATE TABLE IF NOT EXISTS edge_signals (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                    TIMESTAMP NOT NULL,
    ticker                TEXT NOT NULL REFERENCES contracts(ticker) ON DELETE CASCADE,
    forecast_id           INTEGER REFERENCES forecasts(id) ON DELETE SET NULL,
    market_mid            REAL,
    market_spread         REAL,
    edge_points           REAL,           -- sign-aware, side below
    edge_side             TEXT,           -- 'YES' | 'NO'
    edge_sharpe           REAL,
    kelly_fraction        REAL,
    kelly_dollars         REAL,
    fee_adjusted_ev       REAL,
    time_decay            REAL,
    correlation_penalty   REAL,
    composite_score       REAL,
    UNIQUE(ts, ticker)
);
CREATE INDEX IF NOT EXISTS idx_signal_ts ON edge_signals(ts);

-- Materialized calibration records. Can be derived from forecasts + resolutions,
-- but keep this denormalized so reliability-curve queries stay cheap.
CREATE TABLE IF NOT EXISTS calibration_records (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    forecast_id  INTEGER NOT NULL REFERENCES forecasts(id) ON DELETE CASCADE,
    ticker       TEXT NOT NULL,
    forecaster   TEXT NOT NULL,
    category     TEXT,
    p_yes        REAL NOT NULL,
    outcome      INTEGER NOT NULL,        -- 1 YES, 0 NO
    resolved_at  TIMESTAMP NOT NULL,
    brier        REAL NOT NULL,
    log_score    REAL NOT NULL,
    UNIQUE(forecast_id)
);
CREATE INDEX IF NOT EXISTS idx_calib_fc ON calibration_records(forecaster, resolved_at);
