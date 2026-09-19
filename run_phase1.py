from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parent
SRC_DIR = PROJECT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from config import (  # noqa: E402
    BINANCE_FIRST_MONTH,
    DB_PATH,
    LOGS_DIR,
    PROJECT_ROOT,
    ensure_directories,
)
from download_binance_spot import (  # noqa: E402
    download_binance_spot_monthly,
    download_bitcoincharts_bitstamp,
    download_ff137_bitstamp_1m,
    last_complete_month,
    month_starts,
)
from import_to_duckdb import (  # noqa: E402
    connect,
    import_all_binance,
    import_bitcoincharts_bitstamp,
    import_ff137_bitstamp_1m,
)
from validate_data import validate, write_reports  # noqa: E402


def parse_month(value: str) -> date:
    return date.fromisoformat(value + "-01")


def configure_logging() -> None:
    ensure_directories()
    log_path = LOGS_DIR / "phase1.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=[logging.FileHandler(log_path, encoding="utf-8"), logging.StreamHandler()],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Trend Emperor BTC Sentinel Phase 1.")
    parser.add_argument("--binance-start-month", default=f"{BINANCE_FIRST_MONTH:%Y-%m}")
    parser.add_argument("--binance-end-month", default=f"{last_complete_month():%Y-%m}")
    parser.add_argument(
        "--include-early-bitstamp",
        action="store_true",
        help="Download/import BitcoinCharts Bitstamp BTCUSD trade history and aggregate 2012-2017 to 1m.",
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--skip-import", action="store_true")
    return parser.parse_args()


def main() -> None:
    started = time.monotonic()
    configure_logging()
    args = parse_args()
    logging.info("phase1_start project_root=%s db=%s", PROJECT_ROOT, DB_PATH)

    binance_start = parse_month(args.binance_start_month)
    binance_end = parse_month(args.binance_end_month)
    expected_binance_months = [f"{month:%Y-%m}" for month in month_starts(binance_start, binance_end)]
    downloaded_binance_paths = []

    if not args.skip_download:
        downloaded_binance_paths = download_binance_spot_monthly(binance_start, binance_end)
        if args.include_early_bitstamp:
            if download_bitcoincharts_bitstamp() is None:
                download_ff137_bitstamp_1m()

    inserted = 0
    if not args.skip_import:
        with connect() as con:
            inserted += import_all_binance(con)
            if args.include_early_bitstamp:
                early_inserted = import_bitcoincharts_bitstamp(con)
                if early_inserted == 0:
                    early_inserted = import_ff137_bitstamp_1m(con)
                inserted += early_inserted
        logging.info("phase1_import_done inserted=%s", inserted)

    actual_binance_months = {
        path.stem.rsplit("-", 2)[-2] + "-" + path.stem.rsplit("-", 2)[-1]
        for path in downloaded_binance_paths
    }
    if not actual_binance_months:
        raw_dir = PROJECT_ROOT / "raw_data" / "binance_vision_spot" / "BTCUSDT" / "1m"
        actual_binance_months = {
            path.stem.rsplit("-", 2)[-2] + "-" + path.stem.rsplit("-", 2)[-1]
            for path in raw_dir.glob("BTCUSDT-1m-*.zip")
        }
    missing_binance_months = sorted(set(expected_binance_months) - actual_binance_months)
    metadata = {
        "project_root": str(PROJECT_ROOT),
        "db_path": str(DB_PATH),
        "binance_expected_months": expected_binance_months,
        "binance_missing_months": missing_binance_months,
        "include_early_bitstamp": bool(args.include_early_bitstamp),
        "skip_download": bool(args.skip_download),
        "skip_import": bool(args.skip_import),
        "inserted_rows_this_run": inserted,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    (PROJECT_ROOT / "reports" / "phase1_run_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report = validate()
    report["run_metadata"].update({"elapsed_seconds": round(time.monotonic() - started, 3)})
    json_path, md_path = write_reports(report)
    logging.info("phase1_report json=%s markdown=%s", json_path, md_path)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
