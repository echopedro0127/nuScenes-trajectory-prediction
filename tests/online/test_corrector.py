from __future__ import annotations

import numpy as np
import pytest
import torch

from trajectory_prediction.online.corrector import ResidualCorrector, smoothness_loss
from trajectory_prediction.online.dataset import CorrectionDataset


def test_corrector_masks_elapsed_steps() -> None:
    model = ResidualCorrector(8, 64)
    features = torch.randn(3, 12, 8)
    mask = torch.tensor([[0, 0] + [1] * 10] * 3).bool()

    offsets = model(features, mask)

    assert offsets.shape == (3, 12, 2)
    torch.testing.assert_close(offsets[:, :2], torch.zeros_like(offsets[:, :2]))


def test_constant_offset_has_zero_smoothness() -> None:
    assert smoothness_loss(torch.ones(2, 12, 2), torch.ones(2, 12).bool()).item() == 0.0


def test_smoothness_ignores_invalid_adjacent_pairs() -> None:
    offsets = torch.zeros(1, 4, 2)
    offsets[0, 2] = 10.0
    mask = torch.tensor([[True, False, True, False]])

    assert smoothness_loss(offsets, mask).item() == 0.0


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base = np.zeros((1, 12, 2), dtype=np.float32)
    future = np.arange(24, dtype=np.float32).reshape(1, 12, 2)
    history = np.zeros((1, 5, 2), dtype=np.float32)
    return base, future, history


def test_correction_dataset_features_are_causal() -> None:
    base, future, history = _arrays()
    dataset = CorrectionDataset(base, future, history, update_step=2)
    mutated_future = future.copy()
    mutated_future[:, 4:] += 1000.0
    mutated = CorrectionDataset(base, mutated_future, history, update_step=2)

    first = dataset[0]
    changed = mutated[0]

    np.testing.assert_array_equal(first["features"], changed["features"])
    assert first["features"].shape == (12, 8)
    assert first["mask"].shape == (12,)
    assert not first["mask"][:2].any()
    np.testing.assert_allclose(first["target_offset"], future[0] - base[0])


def test_correction_dataset_rejects_invalid_update() -> None:
    base, future, history = _arrays()
    with pytest.raises(ValueError):
        CorrectionDataset(base, future, history, update_step=0)
