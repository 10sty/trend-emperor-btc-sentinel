from pathlib import Path

import duckdb

from public_dashboard import public_status_payload


PRIVATE_KEYS = {
    "api_key",
    "api_secret",
    "balance",
    "equity",
    "positions",
    "orders",
    "take_profit",
    "stop_loss",
    "leverage",
    "account",
    "live_unlocked",
}


def assert_public_only(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            assert key.lower() not in PRIVATE_KEYS
            assert_public_only(nested)
    elif isinstance(value, list):
        for nested in value:
            assert_public_only(nested)


def test_missing_database_returns_safe_fallback(tmp_path: Path) -> None:
    payload = public_status_payload(tmp_path / "missing.duckdb")

    assert payload["status"] == "data_unavailable"
    assert payload["market_data"] is None
    assert_public_only(payload)


def test_malformed_database_returns_safe_fallback(tmp_path: Path) -> None:
    malformed = tmp_path / "malformed.duckdb"
    malformed.write_text("not a database", encoding="utf-8")

    payload = public_status_payload(malformed)

    assert payload["status"] == "data_unavailable"
    assert payload["market_data"] is None
    assert_public_only(payload)


def test_valid_database_returns_only_public_market_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "public.duckdb"
    connection = duckdb.connect(str(db_path))
    connection.execute(
        "CREATE TABLE btc_1m (timestamp_utc TIMESTAMP, symbol VARCHAR, close DOUBLE)"
    )
    connection.execute(
        "INSERT INTO btc_1m VALUES (TIMESTAMP '2026-09-19 12:34:00', 'BTCUSDT', 63125.5)"
    )
    connection.close()

    payload = public_status_payload(db_path)

    assert payload["status"] == "ok"
    assert set(payload["market_data"]) == {
        "symbol",
        "timestamp_utc",
        "close",
        "row_count",
    }
    assert payload["market_data"] == {
        "symbol": "BTCUSDT",
        "timestamp_utc": "2026-09-19T12:34:00",
        "close": 63125.5,
        "row_count": 1,
    }
    assert_public_only(payload)
