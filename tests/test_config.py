from pathlib import Path

import pytest

from trajectory_prediction.config import load_yaml


def test_load_yaml_reads_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("version: v1.0-mini\nsplits: [mini_train]\n", encoding="utf-8")

    assert load_yaml(path) == {"version": "v1.0-mini", "splits": ["mini_train"]}


@pytest.mark.parametrize("content", ["- mini_train\n- mini_val\n", "null\n"])
def test_load_yaml_rejects_non_mapping(tmp_path: Path, content: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="configuration must be a mapping"):
        load_yaml(path)
