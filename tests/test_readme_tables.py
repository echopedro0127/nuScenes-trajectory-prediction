import json
from pathlib import Path

import pytest

from scripts.build_readme_tables import build_tables, load_metrics, markdown_row


def test_markdown_row_formats_measured_values() -> None:
    row = markdown_row("CV", {"sample_count": 10, "ade": 1.2345, "fde": 2.3456})
    assert row == "| CV | 10 | 1.234 | 2.346 |"


def test_load_metrics_rejects_nonfinite_values(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({"sample_count": 2, "ade": float("nan"), "fde": 1.0}), encoding="utf-8")
    with pytest.raises(ValueError, match="finite"):
        load_metrics(path)


def test_build_tables_includes_horizon_rows(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"
    path.write_text(
        json.dumps(
            {
                "sample_count": 2,
                "ade": 1.0,
                "fde": 2.0,
                "horizons": {"1.0s": {"ade": 0.5, "fde": 0.8, "miss_rate_2": 0.0}},
            }
        ),
        encoding="utf-8",
    )
    rendered = build_tables([f"CV={path}"], horizon=[f"CV={path}"])
    assert "### Horizon analysis" in rendered
    assert "CV/1.0s" in rendered
