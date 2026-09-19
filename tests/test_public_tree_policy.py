from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_public_tree import scan_tree


def test_clean_tree_passes(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "safe.py").write_text("MODE = 'read_only'\n", encoding="utf-8")
    assert scan_tree(tmp_path) == []


def test_forbidden_paths_and_symlink_fail(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("EXAMPLE=1\n", encoding="utf-8")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "btc.duckdb").write_bytes(b"db")
    (tmp_path / "escape").symlink_to(Path("/tmp"))
    errors = scan_tree(tmp_path)
    assert any("forbidden path" in error for error in errors)
    assert any("symlink" in error for error in errors)


def test_private_content_fails(tmp_path: Path) -> None:
    secret_name = "API" + "_SECRET"
    content = "\n".join(
        [
            f"{secret_name}=not-a-placeholder",
            "BEGIN " + "PRIVATE KEY",
            "/Users/" + "trendemperor/Projects/private",
            "192" + ".168.31.45",
            "14" + ".5239",
            "I_" + "UNDERSTAND_CONTRACT_RISK",
        ]
    )
    (tmp_path / "bad.txt").write_text(content, encoding="utf-8")
    assert len(scan_tree(tmp_path)) >= 6
