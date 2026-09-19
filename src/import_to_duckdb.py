from __future__ import annotations

import argparse
import gzip
import logging
import zipfile
from pathlib import Path

import duckdb
import pandas as pd

from config import (
    BINANCE_INTERVAL,
    BINANCE_SYMBOL,
    BITCOINCHARTS_SOURCE,
    BITCOINCHARTS_SYMBOL,
    DB_PATH,
    EARLY_DATA_END,
    FF137_SOURCE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    START_DATE,
    ensure_directories,
)
from db_runtime import connect_duckdb_writer


LOGGER = logging.getLogger("btc_sentinel.import")

BINANCE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

TARGET_COLUMNS = [
    "timestamp_utc",
    "source",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
]


CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS btc_1m (
    timestamp_utc TIMESTAMP NOT NULL,
    source VARCHAR NOT NULL,
    symbol VARCHAR NOT NULL,
    open DOUBLE NOT NULL,
    high DOUBLE NOT NULL,
    low DOUBLE NOT NULL,
    close DOUBLE NOT NULL,
    volume DOUBLE NOT NULL,
    quote_volume DOUBLE,
    trade_count BIGINT,
    taker_buy_base_volume DOUBLE,
    taker_buy_quote_volume DOUBLE
);
"""


def connect() -> duckdb.DuckDBPyConnection:
    ensure_directories()
    con = connect_duckdb_writer(DB_PATH)
    con.execute(CREATE_TABLE_SQL)
    con.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS btc_1m_unique "
        "ON btc_1m(timestamp_utc, source, symbol);"
    )
    return con


def insert_deduped(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    df = df[TARGET_COLUMNS].drop_duplicates(["timestamp_utc", "source", "symbol"])
    con.register("incoming_btc_1m", df)
    before = con.execute("SELECT COUNT(*) FROM btc_1m").fetchone()[0]
    con.execute(
        """
        INSERT INTO btc_1m
        SELECT i.*
        FROM incoming_btc_1m i
        LEFT JOIN btc_1m b
          ON b.timestamp_utc = i.timestamp_utc
         AND b.source = i.source
         AND b.symbol = i.symbol
        WHERE b.timestamp_utc IS NULL;
        """
    )
    after = con.execute("SELECT COUNT(*) FROM btc_1m").fetchone()[0]
    con.unregister("incoming_btc_1m")
    return int(after - before)


def normalize_binance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = BINANCE_COLUMNS[: len(df.columns)]
    open_time = pd.to_numeric(df["open_time"], errors="coerce")
    max_open_time = open_time.max(skipna=True)
    if max_open_time and max_open_time < 10_000_000_000:
        timestamp = pd.to_datetime(open_time, unit="s", utc=True)
    elif max_open_time and max_open_time > 100_000_000_000_000:
        timestamp = pd.to_datetime(open_time, unit="us", utc=True)
    else:
        timestamp = pd.to_datetime(open_time, unit="ms", utc=True)

    normalized = pd.DataFrame(
        {
            "timestamp_utc": timestamp.dt.tz_localize(None),
            "source": "binance_vision_spot",
            "symbol": BINANCE_SYMBOL,
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce"),
            "quote_volume": pd.to_numeric(df.get("quote_volume"), errors="coerce"),
            "trade_count": pd.to_numeric(df.get("trade_count"), errors="coerce").astype("Int64"),
            "taker_buy_base_volume": pd.to_numeric(df.get("taker_buy_base_volume"), errors="coerce"),
            "taker_buy_quote_volume": pd.to_numeric(df.get("taker_buy_quote_volume"), errors="coerce"),
        }
    )
    return normalized.dropna(subset=["timestamp_utc", "open", "high", "low", "close", "volume"])


def import_binance_zip(con: duckdb.DuckDBPyConnection, zip_path: Path) -> int:
    with zipfile.ZipFile(zip_path) as archive:
        csv_names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_names:
            LOGGER.warning("no_csv_in_zip path=%s", zip_path)
            return 0
        with archive.open(csv_names[0]) as handle:
            df = pd.read_csv(handle, header=None)

    normalized = normalize_binance_dataframe(df)
    processed_path = (
        PROCESSED_DATA_DIR
        / "binance_vision_spot"
        / BINANCE_SYMBOL
        / BINANCE_INTERVAL
        / f"{zip_path.stem}.csv.gz"
    )
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(processed_path, index=False, compression="gzip")
    inserted = insert_deduped(con, normalized)
    LOGGER.info("imported_binance path=%s rows=%s inserted=%s", zip_path, len(normalized), inserted)
    return inserted


def import_all_binance(con: duckdb.DuckDBPyConnection) -> int:
    raw_dir = RAW_DATA_DIR / "binance_vision_spot" / BINANCE_SYMBOL / BINANCE_INTERVAL
    total = 0
    for zip_path in sorted(raw_dir.glob(f"{BINANCE_SYMBOL}-{BINANCE_INTERVAL}-*.zip")):
        total += import_binance_zip(con, zip_path)
    return total


def aggregate_bitstamp_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    if chunk.empty:
        return pd.DataFrame(columns=TARGET_COLUMNS)
    chunk.columns = ["timestamp_unix", "price", "amount"]
    chunk["timestamp_utc"] = pd.to_datetime(chunk["timestamp_unix"], unit="s", utc=True)
    chunk = chunk[
        (chunk["timestamp_utc"].dt.date >= START_DATE)
        & (chunk["timestamp_utc"].dt.date <= EARLY_DATA_END)
    ].copy()
    if chunk.empty:
        return pd.DataFrame(columns=TARGET_COLUMNS)
    chunk["minute"] = chunk["timestamp_utc"].dt.floor("min")
    chunk["price"] = pd.to_numeric(chunk["price"], errors="coerce")
    chunk["amount"] = pd.to_numeric(chunk["amount"], errors="coerce")
    chunk = chunk.dropna(subset=["price", "amount"])
    grouped = chunk.groupby("minute", sort=True)
    out = grouped.agg(
        open=("price", "first"),
        high=("price", "max"),
        low=("price", "min"),
        close=("price", "last"),
        volume=("amount", "sum"),
        quote_volume=("amount", lambda value: float((value * chunk.loc[value.index, "price"]).sum())),
        trade_count=("price", "count"),
    ).reset_index()
    out["timestamp_utc"] = out["minute"].dt.tz_localize(None)
    out["source"] = BITCOINCHARTS_SOURCE
    out["symbol"] = BITCOINCHARTS_SYMBOL
    out["taker_buy_base_volume"] = pd.NA
    out["taker_buy_quote_volume"] = pd.NA
    return out[TARGET_COLUMNS]


def import_bitcoincharts_bitstamp(con: duckdb.DuckDBPyConnection, chunk_size: int = 2_000_000) -> int:
    raw_path = RAW_DATA_DIR / "bitcoincharts" / "bitstampUSD" / "bitstampUSD.csv.gz"
    if not raw_path.exists():
        LOGGER.warning("missing_bitstamp_raw path=%s", raw_path)
        return 0

    total = 0
    processed_path = PROCESSED_DATA_DIR / "bitcoincharts" / "bitstampUSD_1m.csv.gz"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    first_write = True
    carry = pd.DataFrame()

    with gzip.open(raw_path, "rt", encoding="utf-8", newline="") as handle:
        for chunk in pd.read_csv(handle, header=None, chunksize=chunk_size):
            if not carry.empty:
                chunk = pd.concat([carry, chunk], ignore_index=True)
            chunk.columns = ["timestamp_unix", "price", "amount"]
            chunk["minute"] = pd.to_datetime(chunk["timestamp_unix"], unit="s", utc=True).dt.floor("min")
            last_minute = chunk["minute"].max()
            ready = chunk[chunk["minute"] < last_minute].drop(columns=["minute"])
            carry = chunk[chunk["minute"] == last_minute].drop(columns=["minute"])

            normalized = aggregate_bitstamp_chunk(ready)
            if normalized.empty:
                continue
            normalized.to_csv(
                processed_path,
                index=False,
                compression="gzip",
                mode="wt" if first_write else "at",
                header=first_write,
            )
            first_write = False
            total += insert_deduped(con, normalized)
            LOGGER.info("imported_bitstamp_chunk rows=%s total_inserted=%s", len(normalized), total)
    if not carry.empty:
        normalized = aggregate_bitstamp_chunk(carry)
        if not normalized.empty:
            normalized.to_csv(
                processed_path,
                index=False,
                compression="gzip",
                mode="wt" if first_write else "at",
                header=first_write,
            )
            total += insert_deduped(con, normalized)
            LOGGER.info("imported_bitstamp_final rows=%s total_inserted=%s", len(normalized), total)
    return total


def import_ff137_bitstamp_1m(con: duckdb.DuckDBPyConnection) -> int:
    raw_path = RAW_DATA_DIR / "github_ff137_bitstamp" / "btcusd_bitstamp_1min_2012-2025.csv.gz"
    if not raw_path.exists():
        LOGGER.warning("missing_ff137_raw path=%s", raw_path)
        return 0

    df = pd.read_csv(raw_path, compression="gzip")
    if "timestamp" not in df.columns:
        raise ValueError(f"ff137 Bitstamp file missing timestamp column: {raw_path}")

    timestamp = pd.to_datetime(pd.to_numeric(df["timestamp"], errors="coerce"), unit="s", utc=True)
    normalized = pd.DataFrame(
        {
            "timestamp_utc": timestamp.dt.tz_localize(None),
            "source": FF137_SOURCE,
            "symbol": BITCOINCHARTS_SYMBOL,
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce"),
            "quote_volume": pd.NA,
            "trade_count": pd.NA,
            "taker_buy_base_volume": pd.NA,
            "taker_buy_quote_volume": pd.NA,
        }
    )
    normalized = normalized[
        (normalized["timestamp_utc"].dt.date >= START_DATE)
        & (normalized["timestamp_utc"].dt.date <= EARLY_DATA_END)
    ].dropna(subset=["timestamp_utc", "open", "high", "low", "close", "volume"])

    processed_path = PROCESSED_DATA_DIR / "github_ff137_bitstamp" / "btcusd_bitstamp_1m_2012_2017.csv.gz"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    normalized.to_csv(processed_path, index=False, compression="gzip")
    inserted = insert_deduped(con, normalized)
    LOGGER.info("imported_ff137_bitstamp rows=%s inserted=%s", len(normalized), inserted)
    return inserted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import BTC spot history into DuckDB.")
    parser.add_argument("--skip-binance", action="store_true")
    parser.add_argument("--include-early-bitstamp", action="store_true")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    with connect() as con:
        if not args.skip_binance:
            import_all_binance(con)
        if args.include_early_bitstamp:
            early_inserted = import_bitcoincharts_bitstamp(con)
            if early_inserted == 0:
                import_ff137_bitstamp_1m(con)


if __name__ == "__main__":
    main()
