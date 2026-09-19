# Architecture

## Public data layer

`download_binance_spot.py` retrieves unauthenticated spot-market archives from explicitly allowlisted public sources. `import_to_duckdb.py` normalizes them into `btc_1m`; `resample_timeframes.py` derives higher-timeframe OHLCV tables; `validate_data.py` produces integrity summaries. Paths are project-relative.

## Research layer

The research modules consume pandas frames or public DuckDB OHLC tables. They implement indicators, swing detection, trend-clock features, Livermore events, profile selection, walk-forward folds, and robustness summaries. They do not place trades or call private exchange endpoints.

## Presentation layer

`public_dashboard.py` opens the public DuckDB file in read-only mode. Its stable status payload contains only a symbol, latest public timestamp, close, and row count. The standard-library HTTP server exposes GET routes and binds to localhost by default.

## Security boundary

The public tree is built from a narrow allowlist. Authenticated clients, execution paths, secrets, account state, runtime databases, generated data, reports, logs, and local models remain outside it. `scripts/verify_public_tree.py` rejects sensitive path classes, secret-like assignments, token patterns, symlinks, and known private literals. CI runs both the scanner and tests.
