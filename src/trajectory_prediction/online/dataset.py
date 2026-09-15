from __future__ import annotations

import math

import numpy as np
import torch
from torch.utils.data import Dataset


def _points(name: str, value: np.ndarray, count: int | None = None) -> np.ndarray:
    points = np.asarray(value)
    if points.ndim != 3 or points.shape[1:] != (12, 2):
        raise ValueError(f"{name} must have shape [N, 12, 2]")
    if count is not None and points.shape[0] != count:
        raise ValueError(f"{name} sample count mismatch")
    if not np.issubdtype(points.dtype, np.floating) or not np.isfinite(points).all():
        raise ValueError(f"{name} must be finite floating-point data")
    return points.astype(np.float32, copy=True)


class CorrectionDataset(Dataset):
    """Causal residual examples for one streaming update.

    ``update_step`` is one-indexed. At step k, only ``future[:k]`` is used to
    derive the residual features; target offsets for the remaining horizon are
    still retained for supervised training and must be selected with ``mask``.
    """

    def __init__(
        self,
        base_prediction: np.ndarray,
        future_xy: np.ndarray,
        initial_history: np.ndarray,
        update_step: int,
        dt: float = 0.5,
    ) -> None:
        base = _points("base_prediction", base_prediction)
        future = _points("future_xy", future_xy, count=base.shape[0])
        history = np.asarray(initial_history)
        if history.ndim != 3 or history.shape[0] != base.shape[0] or history.shape[2] != 2 or history.shape[1] < 1:
            raise ValueError("initial_history must have shape [N, H >= 1, 2]")
        if not np.issubdtype(history.dtype, np.floating) or not np.isfinite(history).all():
            raise ValueError("initial_history must be finite floating-point data")
        if isinstance(update_step, bool) or not isinstance(update_step, int) or not 1 <= update_step < 12:
            raise ValueError("update_step must be an integer in [1, 11]")
        if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(float(dt)) or dt <= 0:
            raise ValueError("dt must be a finite positive number")
        self.base = base
        self.future = future
        self.history = history.astype(np.float32, copy=True)
        self.update_step = update_step
        self.dt = float(dt)
        self._features, self._mask, self._target_offset = self._build_cached_examples()

    def _build_cached_examples(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        current = self.future[:, self.update_step - 1]
        previous = (
            self.history[:, -1]
            if self.update_step == 1
            else self.future[:, self.update_step - 2]
        )
        actual_velocity = (current - previous) / np.float32(self.dt)
        base_previous = (
            np.zeros((self.base.shape[0], 2), dtype=np.float32)
            if self.update_step == 1
            else self.base[:, self.update_step - 2]
        )
        base_velocity = (self.base[:, self.update_step - 1] - base_previous) / np.float32(self.dt)
        position_residual = current - self.base[:, self.update_step - 1]
        velocity_residual = actual_velocity - base_velocity
        actual_heading = np.arctan2(actual_velocity[:, 1], actual_velocity[:, 0])
        base_heading = np.arctan2(base_velocity[:, 1], base_velocity[:, 0])
        heading_residual = (actual_heading - base_heading + np.pi) % (2.0 * np.pi) - np.pi

        horizon_index = np.arange(12, dtype=np.float32)
        remaining_fraction = (12.0 - horizon_index - 1.0) / 12.0
        sample_count = self.base.shape[0]
        features = np.empty((sample_count, 12, 8), dtype=np.float32)
        features[:, :, :2] = self.base
        features[:, :, 2:4] = position_residual[:, None, :]
        features[:, :, 4:6] = velocity_residual[:, None, :]
        features[:, :, 6] = heading_residual[:, None]
        features[:, :, 7] = remaining_fraction[None, :]
        mask = np.arange(12) >= self.update_step
        features[:, ~mask] = 0.0
        target_offset = self.future - self.base
        return (
            torch.from_numpy(features),
            torch.from_numpy(np.broadcast_to(mask, (sample_count, 12)).copy()),
            torch.from_numpy(target_offset),
        )

    def __len__(self) -> int:
        return int(self.base.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("dataset index must be an integer")
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("correction dataset index out of range")

        return {
            "features": self._features[index],
            "mask": self._mask[index],
            "target_offset": self._target_offset[index],
            "update_step": torch.tensor(self.update_step, dtype=torch.long),
            "index": torch.tensor(index, dtype=torch.long),
        }
