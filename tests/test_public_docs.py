from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]


def section(text: str, heading: str) -> str:
    match = re.search(
        rf"^## {re.escape(heading)}\n(.+?)(?=^## |\Z)", text, re.M | re.S
    )
    assert match is not None
    return match.group(1).strip()


def test_readme_is_portable_and_honest() -> None:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "newly public" in text.lower()
    assert "not financial advice" in text.lower()
    assert "/Users/" not in text
    assert "stars" not in text.lower()
    assert "downloads" not in text.lower()


def test_application_narratives_fit_limits() -> None:
    text = (ROOT / "docs" / "openai-codex-for-oss-application.md").read_text(
        encoding="utf-8"
    )
    assert len(section(text, "Why this repository qualifies")) <= 500
    assert len(section(text, "Anything else")) <= 500
