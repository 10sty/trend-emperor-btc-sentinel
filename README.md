# Trend Emperor BTC Sentinel

Trend Emperor BTC Sentinel is a newly public, safety-first toolkit for reproducible Bitcoin market-data research. It imports unauthenticated public spot data into DuckDB, derives higher timeframes, runs structural and Livermore-style analysis, measures walk-forward robustness and overfitting risk, and presents a deliberately limited read-only dashboard.

This repository is the credential-free open-source edition of a longer-running local project. It is intended for research and education and is not financial advice.

## What is included

- Public Binance Vision BTCUSDT spot klines, BitcoinCharts Bitstamp history, and an ff137 Bitstamp fallback.
- A portable DuckDB ingestion, normalization, resampling, and integrity-report pipeline.
- Trend-clock, swing-point, Livermore event, profile optimization, walk-forward, and robustness modules.
- Automated tests and a repository policy scanner.
- A localhost-only dashboard exposing public market fields.

## Security boundary

Authenticated exchange clients, trade execution, private endpoints, credentials, account state, operational records, local databases, and generated reports are outside this repository. The dashboard reads only a local public-data DuckDB file and binds to `127.0.0.1` by default. Never commit `.env` files or secrets.

## Quick start

Requires Python 3.9 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/verify_public_tree.py .
```

Download and import public market history:

```bash
python run_phase1.py
```

Run the local public dashboard:

```bash
python run_public_dashboard.py
```

Then open `http://127.0.0.1:8765`.

## Data and research notes

Public datasets can contain gaps, outages, schema changes, and exchange-specific artifacts. Run the integrity checks before using results. Walk-forward and robustness outputs describe historical behavior; they do not predict future returns.

## Maintenance

The primary maintainer plans to continue issue triage, tests, documentation, dependency updates, security review, and releases as the public project develops. Contributions that preserve the public-data-only security boundary are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache License 2.0. See [LICENSE](LICENSE).
