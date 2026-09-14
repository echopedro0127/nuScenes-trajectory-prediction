import math

import torch


def _validate_trajectory_pair(
    prediction: torch.Tensor, target: torch.Tensor
) -> None:
    if (
        prediction.shape != target.shape
        or prediction.ndim != 3
        or prediction.shape[-1] != 2
        or prediction.shape[1] == 0
    ):
        raise ValueError("prediction and target must share shape [B, T, 2] with T >= 1")


def displacement_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    miss_tolerance_m: float = 2.0,
) -> dict[str, torch.Tensor]:
    """Return per-sample displacement errors for single-mode trajectories."""
    _validate_trajectory_pair(prediction, target)
    if (
        isinstance(miss_tolerance_m, bool)
        or not math.isfinite(miss_tolerance_m)
        or miss_tolerance_m <= 0
    ):
        raise ValueError("miss_tolerance_m must be finite and positive")

    distance = torch.linalg.vector_norm(prediction - target, dim=-1)
    return {
        "ade": distance.mean(dim=-1),
        "fde": distance[:, -1],
        "miss": (distance.max(dim=-1).values > miss_tolerance_m).float(),
    }


def horizon_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    frequency_hz: int = 2,
) -> dict[str, dict[str, torch.Tensor]]:
    """Return displacement metrics over 1, 3, and 6 second prefixes."""
    _validate_trajectory_pair(prediction, target)
    if (
        isinstance(frequency_hz, bool)
        or not isinstance(frequency_hz, int)
        or frequency_hz <= 0
    ):
        raise ValueError("frequency_hz must be a positive integer")

    required_steps = 6 * frequency_hz
    if prediction.shape[1] < required_steps:
        raise ValueError(
            f"trajectories need at least {required_steps} steps for a 6-second horizon"
        )

    return {
        f"{seconds}s": displacement_metrics(
            prediction[:, : seconds * frequency_hz],
            target[:, : seconds * frequency_hz],
        )
        for seconds in (1, 3, 6)
    }
