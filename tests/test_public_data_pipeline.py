from datetime import date
from pathlib import Path

import pandas as pd

import config
from download_binance_spot import binance_monthly_url
from import_to_duckdb import normalize_binance_dataframe


def test_config_is_portable_and_credential_free() -> None:
    assert config.PROJECT_ROOT == Path(__file__).resolve().parents[1]
    assert config.DASHBOARD_HOST == "127.0.0.1"
    assert not any(
        "API_KEY" in name or "API_SECRET" in name or "PASSWORD" in name
        for name in vars(config)
    )


def test_binance_url_is_public_spot_only() -> None:
    url = binance_monthly_url(date(2025, 1, 1))
    assert url.startswith(
        "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/1m/"
    )
    assert "futures" not in url.lower()


def test_normalization_uses_public_schema() -> None:
    raw = pd.DataFrame(
        [[1_700_000_000_000, 1, 2, 0.5, 1.5, 10, 0, 15, 3, 4, 6, 0]]
    )
    row = normalize_binance_dataframe(raw).iloc[0]
    assert row["source"] == "binance_vision_spot"
    assert row["symbol"] == "BTCUSDT"
    assert float(row["close"]) == 1.5
