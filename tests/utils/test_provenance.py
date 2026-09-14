from pathlib import Path

from trajectory_prediction.utils.provenance import sha256_file


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_text("trajectory", encoding="utf-8")
    assert sha256_file(path) == "c32662785b597e46c00684652c150705acb460d6c349e2ca831348c9f4f7ce74"
