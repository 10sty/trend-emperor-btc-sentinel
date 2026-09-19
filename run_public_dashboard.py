#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from config import DASHBOARD_HOST, DASHBOARD_PORT, DB_PATH  # noqa: E402
from public_dashboard import serve_dashboard  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the public read-only BTC research dashboard."
    )
    parser.add_argument("--db", type=Path, default=DB_PATH)
    parser.add_argument("--host", default=DASHBOARD_HOST)
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT)
    args = parser.parse_args()
    serve_dashboard(args.db, PROJECT_ROOT / "dashboard", args.host, args.port)


if __name__ == "__main__":
    main()
