from __future__ import annotations

import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import duckdb


def _unavailable_payload() -> dict[str, Any]:
    return {
        "project": "Trend Emperor BTC Sentinel",
        "mode": "read_only_public_data",
        "status": "data_unavailable",
        "market_data": None,
    }


def public_status_payload(db_path: Path) -> dict[str, Any]:
    if not db_path.is_file():
        return _unavailable_payload()

    try:
        connection = duckdb.connect(str(db_path), read_only=True)
        try:
            row = connection.execute(
                """
                SELECT symbol, timestamp_utc, close, count(*) OVER () AS row_count
                FROM btc_1m
                ORDER BY timestamp_utc DESC
                LIMIT 1
                """
            ).fetchone()
        finally:
            connection.close()
    except (duckdb.Error, OSError):
        return _unavailable_payload()

    if row is None:
        return _unavailable_payload()

    return {
        "project": "Trend Emperor BTC Sentinel",
        "mode": "read_only_public_data",
        "status": "ok",
        "market_data": {
            "symbol": str(row[0]),
            "timestamp_utc": row[1].isoformat(),
            "close": float(row[2]),
            "row_count": int(row[3]),
        },
    }


def make_handler(
    db_path: Path, static_dir: Path
) -> type[SimpleHTTPRequestHandler]:
    class PublicDashboardHandler(SimpleHTTPRequestHandler):
        def do_GET(self) -> None:
            if urlsplit(self.path).path == "/api/status":
                body = json.dumps(public_status_payload(db_path)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

    return partial(PublicDashboardHandler, directory=str(static_dir))


def serve_dashboard(
    db_path: Path, static_dir: Path, host: str, port: int
) -> None:
    server = ThreadingHTTPServer((host, port), make_handler(db_path, static_dir))
    server.serve_forever()
