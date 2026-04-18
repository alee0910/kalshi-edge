"""External data-source adapters.

One module per API. Each exposes typed fetch functions with local caching
(sqlite via requests-cache, or a lightweight feather/parquet cache for
large numeric pulls). No module here should touch Kalshi — that lives
under ``market/``.

Planned modules (in build order):
    open_meteo.py   — GEFS + ECMWF ensembles for the weather forecaster
    fred.py         — macro time series for the economic-release forecaster
    cme.py          — SOFR / Fed-funds futures for the rates forecaster
    odds_api.py     — multi-book sports odds (free-tier aware) for sports
    polymarket.py   — cross-venue sanity read for politics / earnings
    ncep.py         — NOMADS fallback if open-meteo is insufficient
"""
