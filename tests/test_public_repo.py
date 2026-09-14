from pathlib import Path

from scripts.check_public_repo import find_violations


def test_find_violations_reports_paths_size_and_secret(tmp_path: Path) -> None:
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "sample.txt").write_text("raw", encoding="utf-8")
    secret_name = "api" + "_key"
    (tmp_path / "notes.txt").write_text(f"{secret_name} = 'secret'", encoding="utf-8")
    (tmp_path / "model.pt").write_bytes(b"checkpoint")
    (tmp_path / "large.bin").write_bytes(b"x" * 32)

    violations = find_violations(
        tmp_path,
        ["data/sample.txt", "notes.txt", "model.pt", "large.bin"],
        max_bytes=24,
    )

    assert any("forbidden directory" in item and "data/sample.txt" in item for item in violations)
    assert any("secret-like content" in item and "notes.txt" in item for item in violations)
    assert any("model/checkpoint suffix" in item and "model.pt" in item for item in violations)
    assert any("too large" in item and "large.bin" in item for item in violations)
