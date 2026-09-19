from __future__ import annotations

import fcntl
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb

from config import (
    DB_CONNECT_RETRY_ATTEMPTS,
    DB_CONNECT_RETRY_SECONDS,
    DB_WRITE_LOCK_PATH,
    DB_WRITE_LOCK_TIMEOUT_SECONDS,
)


class DBWriteLockTimeout(RuntimeError):
    pass


def connect_duckdb_writer(
    db_path: Path,
    attempts: int = DB_CONNECT_RETRY_ATTEMPTS,
    retry_seconds: float = DB_CONNECT_RETRY_SECONDS,
) -> duckdb.DuckDBPyConnection:
    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            return duckdb.connect(str(db_path))
        except duckdb.IOException as exc:
            last_error = exc
            time.sleep(retry_seconds)
    raise last_error or RuntimeError(f"Unable to open DuckDB writer: {db_path}")


@contextmanager
def db_write_lock(
    path: Path = DB_WRITE_LOCK_PATH,
    timeout_seconds: int = DB_WRITE_LOCK_TIMEOUT_SECONDS,
    retry_seconds: float = 0.25,
) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    with path.open("a+", encoding="utf-8") as handle:
        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError as exc:
                if time.monotonic() >= deadline:
                    raise DBWriteLockTimeout(f"Timed out waiting for DuckDB writer lock: {path}") from exc
                time.sleep(retry_seconds)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
