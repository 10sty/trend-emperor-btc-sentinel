from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from config import DB_PATH, RAW_DATA_DIR, REPORTS_DIR, ensure_directories
from import_to_duckdb import CREATE_TABLE_SQL


def _scalar(value):
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def downloaded_raw_data_summary() -> list[dict]:
    binance_dir = RAW_DATA_DIR / "binance_vision_spot" / "BTCUSDT" / "1m"
    month_pattern = re.compile(r"BTCUSDT-1m-(\d{4}-\d{2})\.zip$")
    months = []
    files = []
    for path in sorted(binance_dir.glob("BTCUSDT-1m-*.zip")):
        match = month_pattern.match(path.name)
        if match:
            months.append(match.group(1))
            files.append(str(path))

    summaries = []
    if months:
        summaries.append(
            {
                "source": "binance_vision_spot",
                "symbol": "BTCUSDT",
                "file_count": len(files),
                "earliest_month": min(months),
                "latest_month": max(months),
                "raw_path": str(binance_dir),
            }
        )

    bitstamp_path = RAW_DATA_DIR / "bitcoincharts" / "bitstampUSD" / "bitstampUSD.csv.gz"
    if bitstamp_path.exists():
        summaries.append(
            {
                "source": "bitcoincharts_bitstamp",
                "symbol": "BTCUSD",
                "file_count": 1,
                "earliest_month": None,
                "latest_month": None,
                "raw_path": str(bitstamp_path),
            }
        )
    ff137_path = RAW_DATA_DIR / "github_ff137_bitstamp" / "btcusd_bitstamp_1min_2012-2025.csv.gz"
    if ff137_path.exists():
        summaries.append(
            {
                "source": "github_ff137_bitstamp",
                "symbol": "BTCUSD",
                "file_count": 1,
                "earliest_month": "2012-01",
                "latest_month": "2025-01",
                "raw_path": str(ff137_path),
            }
        )
    return summaries


def validate(db_path: Path = DB_PATH) -> dict:
    ensure_directories()
    con = duckdb.connect(str(db_path))
    con.execute(CREATE_TABLE_SQL)

    summary = con.execute(
        """
        SELECT
            source,
            symbol,
            COUNT(*) AS row_count,
            COUNT(DISTINCT timestamp_utc) AS distinct_minutes,
            MIN(timestamp_utc) AS earliest_time,
            MAX(timestamp_utc) AS latest_time
        FROM btc_1m
        GROUP BY source, symbol
        ORDER BY source, symbol;
        """
    ).fetchdf()

    duplicates = con.execute(
        """
        SELECT COUNT(*) AS duplicate_keys
        FROM (
            SELECT timestamp_utc, source, symbol, COUNT(*) AS n
            FROM btc_1m
            GROUP BY timestamp_utc, source, symbol
            HAVING COUNT(*) > 1
        );
        """
    ).fetchone()[0]

    missing_rows = []
    gap_rows = []
    for row in summary.to_dict("records"):
        source = row["source"]
        symbol = row["symbol"]
        earliest = row["earliest_time"]
        latest = row["latest_time"]
        expected = con.execute(
            "SELECT date_diff('minute', ?::TIMESTAMP, ?::TIMESTAMP) + 1",
            [earliest, latest],
        ).fetchone()[0]
        distinct_minutes = int(row["distinct_minutes"])
        missing_rows.append(
            {
                "source": source,
                "symbol": symbol,
                "expected_minutes_between_min_max": int(expected),
                "present_distinct_minutes": distinct_minutes,
                "missing_minutes": int(expected - distinct_minutes),
                "missing_ratio": float((expected - distinct_minutes) / expected) if expected else 0.0,
            }
        )
        gaps = con.execute(
            """
            WITH ordered AS (
                SELECT
                    timestamp_utc,
                    lag(timestamp_utc) OVER (ORDER BY timestamp_utc) AS previous_timestamp
                FROM btc_1m
                WHERE source = ? AND symbol = ?
            )
            SELECT
                previous_timestamp,
                timestamp_utc,
                date_diff('minute', previous_timestamp, timestamp_utc) - 1 AS missing_minutes
            FROM ordered
            WHERE previous_timestamp IS NOT NULL
              AND date_diff('minute', previous_timestamp, timestamp_utc) > 1
            ORDER BY previous_timestamp
            LIMIT 100;
            """,
            [source, symbol],
        ).fetchdf()
        gap_rows.extend(gaps.assign(source=source, symbol=symbol).to_dict("records"))

    schema = con.execute("DESCRIBE btc_1m").fetchdf().to_dict("records")
    yearly_counts = con.execute(
        """
        SELECT
            source,
            symbol,
            year(timestamp_utc) AS year,
            COUNT(*) AS row_count
        FROM btc_1m
        GROUP BY source, symbol, year(timestamp_utc)
        ORDER BY source, symbol, year;
        """
    ).fetchdf()
    monthly_counts = con.execute(
        """
        SELECT
            source,
            symbol,
            strftime(timestamp_utc, '%Y-%m') AS month,
            COUNT(*) AS row_count
        FROM btc_1m
        GROUP BY source, symbol, strftime(timestamp_utc, '%Y-%m')
        ORDER BY source, symbol, month;
        """
    ).fetchdf()
    anomaly_summary = con.execute(
        """
        SELECT
            COUNT(*) FILTER (
                WHERE open IS NULL OR high IS NULL OR low IS NULL OR close IS NULL
                   OR open <= 0 OR high <= 0 OR low <= 0 OR close <= 0
            ) AS bad_ohlc_rows,
            COUNT(*) FILTER (
                WHERE high < low
                   OR open > high OR open < low
                   OR close > high OR close < low
            ) AS inconsistent_ohlc_rows,
            COUNT(*) FILTER (WHERE volume IS NULL OR volume < 0) AS bad_volume_rows,
            COUNT(*) FILTER (WHERE quote_volume < 0) AS bad_quote_volume_rows,
            COUNT(*) FILTER (WHERE trade_count < 0) AS bad_trade_count_rows
        FROM btc_1m;
        """
    ).fetchone()
    totals = con.execute(
        """
        SELECT
            COUNT(*) AS total_rows,
            MIN(timestamp_utc) AS earliest_time,
            MAX(timestamp_utc) AS latest_time
        FROM btc_1m;
        """
    ).fetchone()
    con.close()

    run_metadata_path = REPORTS_DIR / "phase1_run_metadata.json"
    run_metadata = {}
    if run_metadata_path.exists():
        run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "database": str(db_path),
        "database_file_size_bytes": db_path.stat().st_size if db_path.exists() else 0,
        "table": "btc_1m",
        "total_rows": int(totals[0]),
        "earliest_time": _scalar(totals[1]),
        "latest_time": _scalar(totals[2]),
        "duplicate_key_count": int(duplicates),
        "run_metadata": run_metadata,
        "downloaded_raw_data": downloaded_raw_data_summary(),
        "by_source": [
            {key: _scalar(value) for key, value in row.items()} for row in summary.to_dict("records")
        ],
        "yearly_counts": [
            {key: _scalar(value) for key, value in row.items()} for row in yearly_counts.to_dict("records")
        ],
        "monthly_counts": [
            {key: _scalar(value) for key, value in row.items()} for row in monthly_counts.to_dict("records")
        ],
        "missing_minutes_by_source": missing_rows,
        "missing_minutes_total": int(sum(row["missing_minutes"] for row in missing_rows)),
        "first_100_gaps": [
            {key: _scalar(value) for key, value in row.items()} for row in gap_rows
        ],
        "anomaly_checks": {
            "bad_ohlc_rows": int(anomaly_summary[0]),
            "inconsistent_ohlc_rows": int(anomaly_summary[1]),
            "bad_volume_rows": int(anomaly_summary[2]),
            "bad_quote_volume_rows": int(anomaly_summary[3]),
            "bad_trade_count_rows": int(anomaly_summary[4]),
        },
        "duckdb_schema": schema,
        "phase2_recommendations": [
            "Add read-only real-time spot quote ingestion to fill the current incomplete month.",
            "Generate all higher timeframes locally from btc_1m only.",
            "Build a local read-only chart service.",
            "Add trend-structure and Livermore-rule recognition after data QA is stable.",
            "Add alerts without any trading, withdrawal, transfer, or account-modification APIs.",
        ],
    }
    return report


def write_reports(report: dict) -> tuple[Path, Path]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = REPORTS_DIR / "phase1_data_integrity_report.json"
    md_path = REPORTS_DIR / "phase1_data_integrity_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# Phase 1 Data Integrity Report",
        "",
        f"- Database: `{report['database']}`",
        f"- Table: `{report['table']}`",
        f"- Total rows: {report['total_rows']}",
        f"- Earliest time: {report['earliest_time']}",
        f"- Latest time: {report['latest_time']}",
        f"- Duplicate logical keys: {report['duplicate_key_count']}",
        f"- Database size bytes: {report['database_file_size_bytes']}",
        f"- Missing minutes total: {report['missing_minutes_total']}",
        "",
        "## Downloaded Raw Data",
        "",
    ]
    if report["downloaded_raw_data"]:
        for row in report["downloaded_raw_data"]:
            lines.append(
                f"- {row['source']} {row['symbol']}: {row['file_count']} file(s), "
                f"{row['earliest_month']} to {row['latest_month']}, `{row['raw_path']}`"
            )
    else:
        lines.append("- No raw files found.")
    lines.extend([
        "",
        "## Missing Minutes",
        "",
    ])
    for row in report["missing_minutes_by_source"]:
        lines.append(
            f"- {row['source']} {row['symbol']}: "
            f"{row['missing_minutes']} missing of {row['expected_minutes_between_min_max']} expected "
            f"({row['missing_ratio']:.6%})"
        )
    lines.extend(["", "## Anomaly Checks", ""])
    for key, value in report["anomaly_checks"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Yearly Counts", ""])
    for row in report["yearly_counts"]:
        lines.append(f"- {row['source']} {row['symbol']} {row['year']}: {row['row_count']}")
    lines.extend(["", "## Monthly Counts", ""])
    for row in report["monthly_counts"]:
        lines.append(f"- {row['source']} {row['symbol']} {row['month']}: {row['row_count']}")
    if report["first_100_gaps"]:
        lines.extend(["", "## First 100 Gaps", ""])
        for row in report["first_100_gaps"]:
            lines.append(
                f"- {row['source']} {row['symbol']}: {row['previous_timestamp']} to "
                f"{row['timestamp_utc']}, missing {row['missing_minutes']}"
            )
    lines.extend(["", "## DuckDB Schema", ""])
    for column in report["duckdb_schema"]:
        lines.append(f"- {column['column_name']}: {column['column_type']}")
    lines.extend(["", "## Phase 2 Suggestions", ""])
    for item in report["phase2_recommendations"]:
        lines.append(f"- {item}")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate BTC 1m DuckDB data.")
    parser.parse_args()
    report = validate()
    write_reports(report)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
