"""Headless rendering of agent-frame trajectory cases for reports and review."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _points(name: str, value: np.ndarray, *, allow_empty: bool = False) -> np.ndarray:
    points = np.asarray(value)
    if points.ndim != 2 or points.shape[1] != 2 or (not allow_empty and points.shape[0] == 0):
        raise ValueError(f"{name} must have shape [T >= 1, 2]")
    if not np.issubdtype(points.dtype, np.floating) or not np.isfinite(points).all():
        raise ValueError(f"{name} must be finite floating-point data")
    return points.astype(np.float32, copy=False)


def render_case(
    history: np.ndarray,
    target: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    title: str,
    output_path: Path,
) -> None:
    """Render one local-coordinate history/target/prediction figure to PNG."""
    history_array = _points("history", history)
    target_array = _points("target", target)
    if not isinstance(predictions, Mapping):
        raise TypeError("predictions must be a mapping")
    prediction_arrays = {str(name): _points(f"prediction[{name}]", value) for name, value in predictions.items()}
    if not isinstance(title, str) or not title.strip():
        raise ValueError("title must be a non-empty string")

    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    axis.plot(history_array[:, 0], history_array[:, 1], "o-", color="#2563eb", label="History")
    axis.plot(target_array[:, 0], target_array[:, 1], "o-", color="#16a34a", label="Ground truth")
    prediction_colors = ("#dc2626", "#9333ea", "#ea580c", "#0891b2")
    for index, (name, prediction) in enumerate(prediction_arrays.items()):
        axis.plot(
            prediction[:, 0],
            prediction[:, 1],
            "--",
            color=prediction_colors[index % len(prediction_colors)],
            label=name,
        )
    axis.scatter([0.0], [0.0], marker="*", s=120, color="black", label="Current")
    axis.set(
        aspect="equal",
        xlabel="Agent-frame x [m]",
        ylabel="Agent-frame y [m]",
        title=title,
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, format="png")
    plt.close(figure)
