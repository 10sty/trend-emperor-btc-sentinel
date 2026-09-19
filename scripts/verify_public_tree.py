from __future__ import annotations

import re
import sys
from pathlib import Path


FORBIDDEN_NAMES = {
    ".env",
    ".env.binance",
    "data",
    "raw_data",
    "processed_data",
    "logs",
    "reports",
    "runtime",
    "backups",
    "archive",
    ".venv",
    "models",
    "__pycache__",
}
FORBIDDEN_SUFFIXES = {
    ".duckdb",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".log",
    ".gguf",
    ".pem",
    ".key",
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".toml",
    ".yml",
    ".yaml",
    ".json",
    ".js",
    ".css",
    ".html",
    ".sh",
}
CREDENTIAL = re.compile(
    r"(?im)^\s*[A-Z0-9_]*(?:API_KEY|API_SECRET|PASSWORD|PASSWD|ACCESS_TOKEN|REFRESH_TOKEN)\s*=\s*[^\s#][^\r\n]*$"
)
TOKENS = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
)
BANNED = (
    "/Users/" + "trendemperor",
    "192" + ".168.",
    "14" + ".5239",
    "4" + ".86018006",
    "BEGIN " + "PRIVATE KEY",
    "I_" + "UNDERSTAND_CONTRACT_RISK",
    "I_" + "ACCEPT_ONE_SHOT_LIVE_BYBIT_ORDER",
)


def scan_tree(root: Path) -> list[str]:
    errors: list[str] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if ".git" in relative.parts:
            continue
        if path.is_symlink():
            errors.append(f"symlink: {relative}")
            continue
        if any(part.startswith(".env") or part in FORBIDDEN_NAMES for part in relative.parts):
            errors.append(f"forbidden path: {relative}")
            continue
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES:
            errors.append(f"forbidden path: {relative}")
            continue
        if not path.is_file() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if CREDENTIAL.search(text):
            errors.append(f"credential assignment: {relative}")
        for pattern in TOKENS:
            if pattern.search(text):
                errors.append(f"token pattern: {relative}")
        for literal in BANNED:
            if literal in text:
                errors.append(f"private literal in {relative}: {literal}")
    return errors


def main() -> int:
    errors = scan_tree(Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve())
    for error in errors:
        print(error)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
