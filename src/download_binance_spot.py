from __future__ import annotations

import argparse
import logging
import time
from datetime import date
from pathlib import Path
from typing import Iterable

import requests

from config import (
    BINANCE_FIRST_MONTH,
    BINANCE_INTERVAL,
    BINANCE_SPOT_MONTHLY_PATH,
    BINANCE_SYMBOL,
    BINANCE_VISION_BASE_URL,
    BITCOINCHARTS_BITSTAMP_FALLBACK_URLS,
    BITCOINCHARTS_BITSTAMP_URL,
    FF137_BITSTAMP_1M_URL,
    RAW_DATA_DIR,
    assert_binance_spot_monthly_path,
    ensure_directories,
)


LOGGER = logging.getLogger("btc_sentinel.download")


def month_starts(start: date, end: date) -> Iterable[date]:
    cursor = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while cursor <= last:
        yield cursor
        if cursor.month == 12:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, cursor.month + 1, 1)


def last_complete_month(today: date | None = None) -> date:
    today = today or date.today()
    if today.month == 1:
        return date(today.year - 1, 12, 1)
    return date(today.year, today.month - 1, 1)


def download_file(url: str, destination: Path, retries: int = 4, timeout: int = 60) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        LOGGER.info("skip_existing path=%s bytes=%s", destination, destination.stat().st_size)
        return True

    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            LOGGER.info("download_start url=%s attempt=%s", url, attempt)
            with requests.get(url, stream=True, timeout=timeout) as response:
                if response.status_code == 404:
                    LOGGER.warning("download_missing url=%s", url)
                    return False
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            partial.replace(destination)
            LOGGER.info("download_done path=%s bytes=%s", destination, destination.stat().st_size)
            return True
        except requests.RequestException as exc:
            LOGGER.warning("download_retry url=%s attempt=%s error=%s", url, attempt, exc)
            time.sleep(min(2**attempt, 30))
        finally:
            if partial.exists() and not destination.exists():
                partial.unlink(missing_ok=True)
    LOGGER.error("download_failed url=%s", url)
    return False


def binance_monthly_url(month: date) -> str:
    relative_path = f"{BINANCE_SPOT_MONTHLY_PATH}/{BINANCE_SYMBOL}-{BINANCE_INTERVAL}-{month:%Y-%m}.zip"
    assert_binance_spot_monthly_path(relative_path)
    return f"{BINANCE_VISION_BASE_URL}/{relative_path}"


def download_binance_spot_monthly(start_month: date, end_month: date) -> list[Path]:
    assert_binance_spot_monthly_path(BINANCE_SPOT_MONTHLY_PATH)
    raw_dir = RAW_DATA_DIR / "binance_vision_spot" / BINANCE_SYMBOL / BINANCE_INTERVAL
    downloaded: list[Path] = []
    for month in month_starts(start_month, end_month):
        url = binance_monthly_url(month)
        target = raw_dir / f"{BINANCE_SYMBOL}-{BINANCE_INTERVAL}-{month:%Y-%m}.zip"
        if download_file(url, target):
            downloaded.append(target)
    return downloaded


def download_bitcoincharts_bitstamp() -> Path | None:
    raw_dir = RAW_DATA_DIR / "bitcoincharts" / "bitstampUSD"
    target = raw_dir / "bitstampUSD.csv.gz"
    for url in (BITCOINCHARTS_BITSTAMP_URL, *BITCOINCHARTS_BITSTAMP_FALLBACK_URLS):
        if download_file(url, target, timeout=180):
            return target
    return None


def download_ff137_bitstamp_1m() -> Path | None:
    raw_dir = RAW_DATA_DIR / "github_ff137_bitstamp"
    target = raw_dir / "btcusd_bitstamp_1min_2012-2025.csv.gz"
    return target if download_file(FF137_BITSTAMP_1M_URL, target, timeout=180) else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BTC spot 1m source data.")
    parser.add_argument("--start-month", default=f"{BINANCE_FIRST_MONTH:%Y-%m}")
    parser.add_argument("--end-month", default=f"{last_complete_month():%Y-%m}")
    parser.add_argument("--skip-binance", action="store_true")
    parser.add_argument("--include-early-bitstamp", action="store_true")
    return parser.parse_args()


def main() -> None:
    ensure_directories()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if not args.skip_binance:
        start = date.fromisoformat(args.start_month + "-01")
        end = date.fromisoformat(args.end_month + "-01")
        download_binance_spot_monthly(start, end)
    if args.include_early_bitstamp:
        if download_bitcoincharts_bitstamp() is None:
            download_ff137_bitstamp_1m()


if __name__ == "__main__":
    main()
