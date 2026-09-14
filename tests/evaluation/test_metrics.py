import math

import pytest
import torch

from trajectory_prediction.evaluation.metrics import (
    displacement_metrics,
    horizon_metrics,
)


def test_displacement_metrics_match_known_distances() -> None:
    target = torch.zeros(2, 12, 2)
    prediction = torch.zeros_like(target)
    prediction[0, :, 0] = 1.0

    values = displacement_metrics(prediction, target)

    torch.testing.assert_close(values["ade"], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(values["fde"], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(values["miss"], torch.tensor([0.0, 0.0]))


def test_displacement_metrics_use_strict_max_over_horizon_miss() -> None:
    target = torch.zeros(2, 2, 2)
    prediction = torch.zeros_like(target)
    prediction[0, :, 0] = torch.tensor([3.0, 1.0])
    prediction[1, 0, 0] = 2.0

    values = displacement_metrics(prediction, target)

    torch.testing.assert_close(values["miss"], torch.tensor([1.0, 0.0]))


def test_horizon_metrics_slice_one_three_and_six_seconds() -> None:
    target = torch.zeros(1, 12, 2)
    prediction = torch.arange(1, 13).view(1, 12, 1).repeat(1, 1, 2).float()

    values = horizon_metrics(prediction, target, frequency_hz=2)

    assert set(values) == {"1s", "3s", "6s"}
    torch.testing.assert_close(
        values["1s"]["fde"], torch.tensor([2.0 * math.sqrt(2.0)])
    )


@pytest.mark.parametrize(
    ("prediction", "target"),
    [
        (torch.zeros(1, 2, 2), torch.zeros(2, 2, 2)),
        (torch.zeros(2, 2), torch.zeros(2, 2)),
        (torch.zeros(1, 2, 3), torch.zeros(1, 2, 3)),
        (torch.zeros(1, 0, 2), torch.zeros(1, 0, 2)),
    ],
)
def test_displacement_metrics_reject_invalid_shapes(
    prediction: torch.Tensor, target: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match=r"\[B, T, 2\]"):
        displacement_metrics(prediction, target)


@pytest.mark.parametrize("tolerance", [0.0, -1.0, math.inf, math.nan])
def test_displacement_metrics_reject_invalid_miss_tolerance(tolerance: float) -> None:
    points = torch.zeros(1, 1, 2)

    with pytest.raises(ValueError, match="miss_tolerance_m"):
        displacement_metrics(points, points, miss_tolerance_m=tolerance)


def test_displacement_metrics_reject_boolean_miss_tolerance() -> None:
    points = torch.zeros(1, 1, 2)

    with pytest.raises(ValueError, match="miss_tolerance_m"):
        displacement_metrics(points, points, miss_tolerance_m=True)


@pytest.mark.parametrize("frequency_hz", [0, -1, 1.5])
def test_horizon_metrics_reject_invalid_frequency(frequency_hz: float) -> None:
    points = torch.zeros(1, 12, 2)

    with pytest.raises(ValueError, match="frequency_hz"):
        horizon_metrics(points, points, frequency_hz=frequency_hz)  # type: ignore[arg-type]


def test_horizon_metrics_reject_insufficient_future_steps() -> None:
    points = torch.zeros(1, 11, 2)

    with pytest.raises(ValueError, match="6-second horizon"):
        horizon_metrics(points, points, frequency_hz=2)
