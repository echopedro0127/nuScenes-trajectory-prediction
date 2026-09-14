import math

import pytest
import torch

from trajectory_prediction.models.physics import (
    constant_velocity_heading,
    finite_difference_cv,
)


def test_finite_difference_cv_preserves_vector_velocity() -> None:
    history = torch.tensor([[[0.0, -1.0], [0.5, -0.5], [1.0, 0.0]]])

    prediction = finite_difference_cv(history, dt=0.5, future_steps=2)

    torch.testing.assert_close(
        prediction, torch.tensor([[[1.5, 0.5], [2.0, 1.0]]])
    )


def test_heading_baseline_moves_along_local_positive_y() -> None:
    speed = torch.tensor([4.0])

    prediction = constant_velocity_heading(speed, dt=0.5, future_steps=2)

    torch.testing.assert_close(
        prediction, torch.tensor([[[0.0, 2.0], [0.0, 4.0]]])
    )


def test_heading_baseline_preserves_speed_dtype() -> None:
    speed = torch.tensor([4.0], dtype=torch.float64)

    prediction = constant_velocity_heading(speed, future_steps=1)

    assert prediction.dtype == speed.dtype
    assert prediction.device == speed.device


@pytest.mark.parametrize(
    "history",
    [
        torch.zeros(2, 2),
        torch.zeros(1, 1, 2),
        torch.zeros(1, 2, 3),
    ],
)
def test_finite_difference_cv_rejects_invalid_history_shape(
    history: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match=r"\[B, H>=2, 2\]"):
        finite_difference_cv(history)


@pytest.mark.parametrize("dt", [0.0, -0.5, math.inf, math.nan])
def test_physics_baselines_reject_invalid_dt(dt: float) -> None:
    history = torch.zeros(1, 2, 2)
    speed = torch.zeros(1)

    with pytest.raises(ValueError, match="dt"):
        finite_difference_cv(history, dt=dt)
    with pytest.raises(ValueError, match="dt"):
        constant_velocity_heading(speed, dt=dt)


@pytest.mark.parametrize("future_steps", [0, -1, 1.5])
def test_physics_baselines_reject_invalid_future_steps(
    future_steps: float,
) -> None:
    history = torch.zeros(1, 2, 2)
    speed = torch.zeros(1)

    with pytest.raises(ValueError, match="future_steps"):
        finite_difference_cv(history, future_steps=future_steps)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="future_steps"):
        constant_velocity_heading(speed, future_steps=future_steps)  # type: ignore[arg-type]


def test_heading_baseline_rejects_non_vector_speed() -> None:
    with pytest.raises(ValueError, match=r"shape \[B\]"):
        constant_velocity_heading(torch.zeros(1, 1))


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool])
def test_finite_difference_cv_rejects_non_floating_history(
    dtype: torch.dtype,
) -> None:
    history = torch.zeros(1, 2, 2, dtype=dtype)

    with pytest.raises(ValueError, match="floating-point"):
        finite_difference_cv(history)


@pytest.mark.parametrize("dtype", [torch.int64, torch.bool])
def test_heading_baseline_rejects_non_floating_speed(dtype: torch.dtype) -> None:
    speed = torch.zeros(1, dtype=dtype)

    with pytest.raises(ValueError, match="floating-point"):
        constant_velocity_heading(speed)


def test_physics_baselines_reject_boolean_dt() -> None:
    history = torch.zeros(1, 2, 2)
    speed = torch.zeros(1)

    with pytest.raises(ValueError, match="dt"):
        finite_difference_cv(history, dt=True)
    with pytest.raises(ValueError, match="dt"):
        constant_velocity_heading(speed, dt=True)
