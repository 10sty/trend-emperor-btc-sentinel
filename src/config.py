from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = PROJECT_ROOT / "raw_data"
PROCESSED_DATA_DIR = PROJECT_ROOT / "processed_data"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"
RUNTIME_DIR = PROJECT_ROOT / "runtime"
DB_PATH = DATA_DIR / "btc_history.duckdb"

DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8765
DB_WRITE_LOCK_PATH = RUNTIME_DIR / "duckdb_write.lock"
DB_WRITE_LOCK_TIMEOUT_SECONDS = 120
DB_CONNECT_RETRY_ATTEMPTS = 80
DB_CONNECT_RETRY_SECONDS = 0.75

BINANCE_VISION_BASE_URL = "https://data.binance.vision"
BINANCE_SPOT_MONTHLY_PATH = "data/spot/monthly/klines/BTCUSDT/1m"
BINANCE_SYMBOL = "BTCUSDT"
BINANCE_INTERVAL = "1m"
BINANCE_FIRST_MONTH = date(2017, 8, 1)

BITCOINCHARTS_BITSTAMP_URL = (
    "https://api.bitcoincharts.com/v1/csv/bitstampUSD.csv.gz"
)
BITCOINCHARTS_BITSTAMP_FALLBACK_URLS = (
    "http://api.bitcoincharts.com/v1/csv/bitstampUSD.csv.gz",
)
FF137_BITSTAMP_1M_URL = (
    "https://raw.githubusercontent.com/ff137/bitstamp-btcusd-minute-data/main/"
    "data/historical/btcusd_bitstamp_1min_2012-2025.csv.gz"
)
BITCOINCHARTS_SYMBOL = "BTCUSD"
BITCOINCHARTS_SOURCE = "bitcoincharts_bitstamp"
FF137_SOURCE = "github_ff137_bitstamp"
EARLY_DATA_END = date(2017, 7, 31)
START_DATE = date(2012, 1, 1)

PHASE2_RESAMPLE_TIMEFRAMES = ("15m", "1h", "4h", "1d", "1w", "1M")


@dataclass(frozen=True)
class SourceInfo:
    name: str
    symbol: str
    market_type: str
    exchange_or_dataset: str
    url: str
    local_raw_subdir: str
    notes: str


DATA_SOURCES = (
    SourceInfo(
        name="binance_vision_spot",
        symbol=BINANCE_SYMBOL,
        market_type="spot",
        exchange_or_dataset="Binance Vision official spot klines",
        url=f"{BINANCE_VISION_BASE_URL}/{BINANCE_SPOT_MONTHLY_PATH}/",
        local_raw_subdir="binance_vision_spot/BTCUSDT/1m",
        notes="Official public monthly one-minute spot BTCUSDT klines.",
    ),
    SourceInfo(
        name=BITCOINCHARTS_SOURCE,
        symbol=BITCOINCHARTS_SYMBOL,
        market_type="spot",
        exchange_or_dataset="BitcoinCharts Bitstamp BTCUSD trade history",
        url=BITCOINCHARTS_BITSTAMP_URL,
        local_raw_subdir="bitcoincharts/bitstampUSD",
        notes="Spot BTCUSD trade history aggregated locally to one-minute OHLCV.",
    ),
    SourceInfo(
        name=FF137_SOURCE,
        symbol=BITCOINCHARTS_SYMBOL,
        market_type="spot",
        exchange_or_dataset="ff137 Bitstamp BTC/USD one-minute OHLC data",
        url=FF137_BITSTAMP_1M_URL,
        local_raw_subdir="github_ff137_bitstamp",
        notes="Public early-history fallback dataset.",
    ),
)


def ensure_directories() -> None:
    for path in (
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        LOGS_DIR,
        REPORTS_DIR,
        RUNTIME_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def assert_binance_spot_monthly_path(path: str) -> None:
    normalized = path.replace("\\", "/").lower()
    required = "data/spot/monthly/klines/btcusdt/1m"
    forbidden = ("futures", "option", "um", "cm", "perpetual", "delivery")
    if required not in normalized:
        raise ValueError(f"Binance Vision path must include {required!r}: {path}")
    if any(token in normalized for token in forbidden):
        raise ValueError(f"Forbidden derivatives path token in Binance path: {path}")
