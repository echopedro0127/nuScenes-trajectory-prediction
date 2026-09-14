from pathlib import Path

import numpy as np

from trajectory_prediction.analysis.visualization import render_case


def test_render_case_creates_nonempty_png(tmp_path: Path) -> None:
    output = tmp_path / "case.png"
    render_case(
        np.array([[0, -1], [0, 0]], np.float32),
        np.array([[0, 1], [0, 2]], np.float32),
        {"CV": np.array([[0, 1], [0, 2]], np.float32)},
        "synthetic",
        output,
    )
    assert output.stat().st_size > 1_000
