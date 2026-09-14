from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    """Expose validated trajectory cache arrays as tensors.

    The cache stores the complete two-second history. ``history_steps`` selects
    a suffix so the current sample is always the final element and no future
    target values can enter the model input.
    """

    _REQUIRED = ("history_features", "history_xy", "history_mask", "future_xy", "current_speed")

    def __init__(self, arrays: Mapping[str, np.ndarray], history_steps: int = 5) -> None:
        if isinstance(history_steps, bool) or not isinstance(history_steps, int) or history_steps <= 0:
            raise ValueError("history_steps must be a positive integer")
        missing = set(self._REQUIRED).difference(arrays)
        if missing:
            raise ValueError(f"trajectory arrays missing: {sorted(missing)}")

        self._arrays = {name: np.asarray(arrays[name]) for name in self._REQUIRED}
        features = self._arrays["history_features"]
        history_xy = self._arrays["history_xy"]
        history_mask = self._arrays["history_mask"]
        future_xy = self._arrays["future_xy"]
        speed = self._arrays["current_speed"]
        if features.ndim != 3 or history_xy.ndim != 3 or history_mask.ndim != 2:
            raise ValueError("history arrays must have shapes [N, H, F], [N, H, 2], and [N, H]")
        if features.shape[:2] != history_xy.shape[:2] or features.shape[:2] != history_mask.shape:
            raise ValueError("history arrays must share batch and history dimensions")
        if future_xy.ndim != 3 or future_xy.shape[0] != features.shape[0] or future_xy.shape[-1] != 2:
            raise ValueError("future_xy must have shape [N, T, 2] matching history arrays")
        if speed.ndim != 1 or speed.shape[0] != features.shape[0]:
            raise ValueError("current_speed must have shape [N] matching history arrays")
        if features.shape[-1] <= 0 or history_xy.shape[-1] != 2:
            raise ValueError("history features and coordinates have invalid dimensions")
        if history_mask.dtype != np.bool_:
            raise ValueError("history_mask must have bool dtype")
        if history_steps > features.shape[1]:
            raise ValueError(
                f"history_steps={history_steps} exceeds cache history length {features.shape[1]}"
            )
        selected_mask = history_mask[:, -history_steps:]
        if selected_mask.shape[1] == 0 or not selected_mask.any(axis=1).all():
            raise ValueError("history_steps selects an empty history for at least one sample")
        self.history_steps = history_steps

    def __len__(self) -> int:
        return int(self._arrays["future_xy"].shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if not isinstance(index, (int, np.integer)):
            raise TypeError("dataset index must be an integer")
        index = int(index)
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError("trajectory dataset index out of range")
        suffix = slice(-self.history_steps, None)
        return {
            "history_features": torch.from_numpy(self._arrays["history_features"][index, suffix]).float(),
            "history_xy": torch.from_numpy(self._arrays["history_xy"][index, suffix]).float(),
            "history_mask": torch.from_numpy(self._arrays["history_mask"][index, suffix]).bool(),
            "future_xy": torch.from_numpy(self._arrays["future_xy"][index]).float(),
            "current_speed": torch.as_tensor(self._arrays["current_speed"][index], dtype=torch.float32),
            "index": torch.tensor(index, dtype=torch.long),
        }
