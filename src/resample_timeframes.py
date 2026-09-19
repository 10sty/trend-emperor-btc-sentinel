from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb

from config import DB_PATH, PHASE2_RESAMPLE_TIMEFRAMES


TIMEFRAME_RULES = {
    "15m": {"interval": "15 minutes", "expected_minutes": 15, "bucket": "time_bucket(INTERVAL '15 minutes', timestamp_utc)"},
    "1h": {"interval": "1 hour", "expected_minutes": 60, "bucket": "date_trunc('hour', timestamp_utc)"},
    "4h": {"interval": "4 hours", "expected_minutes": 240, "bucket": "time_bucket(INTERVAL '4 hours', timestamp_utc)"},
    "1d": {"interval": "1 day", "expected_minutes": 1440, "bucket": "date_trunc('day', timestamp_utc)"},
    "1w": {"interval": "1 week", "expected_minutes": 10080, "bucket": "date_trunc('week', timestamp_utc)"},
    "1M": {"interval": "1 month", "expected_minutes": None, "bucket": "date_trunc('month', timestamp_utc)"},
}


def table_name(timeframe: str) -> str:
    if timeframe not in TIMEFRAME_RULES:
        raise ValueError(f"Unsupported Phase 2 timeframe: {timeframe}")
    # DuckDB identifiers are case-insensitive, so btc_1M would collide with btc_1m.
    return "btc_1mo" if timeframe == "1M" else f"btc_{timeframe}"


def expected_minutes_expression(timeframe: str) -> str:
    expected = TIMEFRAME_RULES[timeframe]["expected_minutes"]
    if expected is not None:
        return str(expected)
    return "date_diff('minute', timestamp_utc, timestamp_utc + INTERVAL '1 month')"


def resample_timeframe(timeframe: str, db_path: Path = DB_PATH) -> dict:
    if timeframe not in TIMEFRAME_RULES:
        raise ValueError(f"Unsupported Phase 2 timeframe: {timeframe}")

    rule = TIMEFRAME_RULES[timeframe]
    target_table = table_name(timeframe)
    bucket = rule["bucket"]
    interval = rule["interval"]
    expected_expr = expected_minutes_expression(timeframe)
    con = duckdb.connect(str(db_path))
    con.execute(
        f"""
        CREATE OR REPLACE TABLE {target_table} AS
        WITH latest_by_symbol AS (
            SELECT symbol, max(timestamp_utc) AS latest_1m
            FROM btc_1m
            GROUP BY symbol
        ),
        bucketed AS (
            SELECT
                {bucket} AS timestamp_utc,
                symbol,
                timestamp_utc AS source_timestamp_utc,
                open,
                high,
                low,
                close,
                volume,
                quote_volume,
                trade_count,
                taker_buy_base_volume,
                taker_buy_quote_volume
            FROM btc_1m
        ),
        complete_windows AS (
            SELECT b.*
            FROM bucketed b
            JOIN latest_by_symbol l USING (symbol)
            WHERE b.timestamp_utc + INTERVAL '{interval}' <= l.latest_1m + INTERVAL '1 minute'
        ),
        ranked AS (
            SELECT
                *,
                first_value(open) OVER w AS bucket_open,
                last_value(close) OVER w AS bucket_close
            FROM complete_windows
            WINDOW w AS (
                PARTITION BY timestamp_utc, symbol
                ORDER BY source_timestamp_utc
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            )
        ),
        aggregated AS (
            SELECT
                timestamp_utc,
                'local_resample_from_1m' AS source,
                symbol,
                any_value(bucket_open) AS open,
                max(high) AS high,
                min(low) AS low,
                any_value(bucket_close) AS close,
                sum(volume) AS volume,
                sum(quote_volume) AS quote_volume,
                sum(trade_count) AS trade_count,
                sum(taker_buy_base_volume) AS taker_buy_base_volume,
                sum(taker_buy_quote_volume) AS taker_buy_quote_volume,
                count(*) AS source_1m_count
            FROM ranked
            GROUP BY timestamp_utc, symbol
        )
        SELECT
            timestamp_utc,
            source,
            symbol,
            open,
            high,
            low,
            close,
            volume,
            quote_volume,
            trade_count,
            taker_buy_base_volume,
            taker_buy_quote_volume
        FROM aggregated
        ORDER BY timestamp_utc, symbol;
        """
    )
    stats = con.execute(
        f"""
        WITH counts AS (
            SELECT
                timestamp_utc,
                symbol,
                count(*) AS source_1m_count
            FROM (
                SELECT {bucket} AS timestamp_utc, symbol, timestamp_utc AS source_timestamp_utc
                FROM btc_1m
            ) b
            JOIN (
                SELECT symbol, max(timestamp_utc) AS latest_1m
                FROM btc_1m
                GROUP BY symbol
            ) l USING (symbol)
            WHERE b.timestamp_utc + INTERVAL '{interval}' <= l.latest_1m + INTERVAL '1 minute'
            GROUP BY timestamp_utc, symbol
        )
        SELECT
            '{target_table}' AS table_name,
            symbol,
            COUNT(*) AS row_count,
            MIN(timestamp_utc) AS earliest_time,
            MAX(timestamp_utc) AS latest_time,
            SUM(CASE WHEN source_1m_count < {expected_expr} THEN 1 ELSE 0 END) AS periods_with_missing_1m,
            SUM(({expected_expr}) - source_1m_count) AS missing_1m_inside_periods
        FROM counts
        GROUP BY symbol
        ORDER BY symbol;
        """
    ).fetchdf().to_dict("records")
    con.close()
    return {"timeframe": timeframe, "table": target_table, "stats": stats}


def resample_phase2_timeframes(db_path: Path = DB_PATH) -> list[dict]:
    return [resample_timeframe(timeframe, db_path=db_path) for timeframe in PHASE2_RESAMPLE_TIMEFRAMES]


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample Phase 2 local timeframes from btc_1m.")
    parser.add_argument("timeframe", nargs="?", choices=PHASE2_RESAMPLE_TIMEFRAMES)
    args = parser.parse_args()
    if args.timeframe:
        result = resample_timeframe(args.timeframe)
    else:
        result = resample_phase2_timeframes()
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
