import math

import torch


def _validate_timing(dt: float, future_steps: int) -> None:
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if (
        isinstance(future_steps, bool)
        or not isinstance(future_steps, int)
        or future_steps <= 0
    ):
        raise ValueError("future_steps must be a positive integer")


def finite_difference_cv(
    history_xy: torch.Tensor,
    dt: float = 0.5,
    future_steps: int = 12,
) -> torch.Tensor:
    """Extrapolate the final finite-difference velocity in agent coordinates."""
    if (
        history_xy.ndim != 3
        or history_xy.shape[1] < 2
        or history_xy.shape[2] != 2
    ):
        raise ValueError("history_xy must have shape [B, H>=2, 2]")
    _validate_timing(dt, future_steps)

    velocity = (history_xy[:, -1] - history_xy[:, -2]) / dt
    times = torch.arange(
        1,
        future_steps + 1,
        device=history_xy.device,
        dtype=history_xy.dtype,
    ) * dt
    return history_xy[:, -1, None, :] + velocity[:, None, :] * times[None, :, None]


def constant_velocity_heading(
    speed: torch.Tensor,
    dt: float = 0.5,
    future_steps: int = 12,
) -> torch.Tensor:
    """Predict constant-speed motion along the agent frame's positive y-axis."""
    if speed.ndim != 1:
        raise ValueError("speed must have shape [B]")
    _validate_timing(dt, future_steps)

    times = torch.arange(
        1,
        future_steps + 1,
        device=speed.device,
        dtype=speed.dtype,
    ) * dt
    output = torch.zeros(
        speed.shape[0],
        future_steps,
        2,
        device=speed.device,
        dtype=speed.dtype,
    )
    output[:, :, 1] = speed[:, None] * times[None, :]
    return output
