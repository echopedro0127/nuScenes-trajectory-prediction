"""Vectorized causal inputs; evaluation/training labels are obtained separately."""
import hashlib

import numpy as np
import torch

from trajectory_prediction.online.interaction_corrector import InteractionCorrector


def residual_batch(current, previous, base, steps):
    current, previous = np.asarray(current, np.float32), np.asarray(previous, np.float32)
    base, steps = np.asarray(base, np.float32), np.asarray(steps)
    n = len(base)
    if base.shape != (n, 12, 2) or current.shape != (n, 2) or previous.shape != (n, 2) or steps.shape != (n,) or not np.issubdtype(steps.dtype, np.integer) or np.any((steps < 1) | (steps > 11)):
        raise ValueError("batch residual shape/step mismatch")
    if not all(np.isfinite(value).all() for value in (current, previous, base)):
        raise ValueError("batch residual inputs must be finite")
    rows = np.arange(n)
    velocity = (current - previous) / np.float32(.5)
    base_previous = base[rows, np.maximum(steps - 2, 0)].copy()
    base_previous[steps == 1] = 0
    base_velocity = (base[rows, steps - 1] - base_previous) / np.float32(.5)
    heading = (np.arctan2(velocity[:, 1], velocity[:, 0]) - np.arctan2(base_velocity[:, 1], base_velocity[:, 0]) + np.pi) % (2 * np.pi) - np.pi
    features = np.empty((n, 12, 8), np.float32)
    features[..., :2] = base
    features[..., 2:4] = (current - base[rows, steps - 1])[:, None]
    features[..., 4:6] = (velocity - base_velocity)[:, None]
    features[..., 6] = heading[:, None]
    features[..., 7] = (12. - np.arange(12, dtype=np.float32) - 1.) / 12.
    mask = np.arange(12)[None] >= steps[:, None]
    features[~mask] = 0
    return features, mask


class BatchedContext:
    def __init__(self, arrays, maps, neighbors, base):
        self.arrays, self.maps, self.neighbors = arrays, maps, neighbors
        self.base = base
        self.size = len(base) * 11

    def fingerprint(self):
        """Bind reuse to actual model inputs and labels, with bounded copies."""
        digest = hashlib.sha256()
        for group, values in (("trajectory", self.arrays), ("map", self.maps),
                              ("neighbors", self.neighbors), ("base", {"base": self.base})):
            for name, value in sorted(values.items()):
                array = np.asarray(value)
                digest.update(str((group, name, array.shape, array.dtype.str)).encode())
                rows = max(1, (8 << 20) // max(1, array[0:1].nbytes))
                for start in range(0, len(array), rows):
                    digest.update(memoryview(np.ascontiguousarray(array[start:start + rows])).cast("B"))
        return digest.hexdigest()

    def _indices(self, indices):
        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or np.any((indices < 0) | (indices >= self.size)):
            raise ValueError("batch indices out of range")
        return indices // 11, indices % 11, indices % 11 + 1

    def inputs(self, indices, device, *, context=True):
        rows, updates, steps = self._indices(indices)
        future = self.arrays["future_xy"]
        current = future[rows, steps - 1]
        previous = future[rows, np.maximum(steps - 2, 0)].copy()
        previous[steps == 1] = self.arrays["history_xy"][rows[steps == 1], -1]
        features, mask = residual_batch(current, previous, self.base[rows], steps)
        values = {"features": features, "mask": mask}
        if context:
            batch = len(rows)
            if context == "map":
                points = np.zeros((batch, 1, 1, 8), np.float32)
                valid = np.zeros((batch, 1, 1), bool)
            else:
                neighbor = self.neighbors
                valid = neighbor["mask"][rows, updates]
                source = neighbor["source_timestamp_us"][rows, updates]
                timestamps = neighbor["timestamp_us"][rows, updates]
                age = np.where(valid, (timestamps[:, None, None] - source) / 2_000_000, 0).astype(np.float32)
                points = np.concatenate((neighbor["features"][rows, updates],
                    neighbor["velocity_mask"][rows, updates, :, :, None].astype(np.float32), age[..., None]), -1)
            if context == "neighbors":
                map_points, map_mask = np.zeros((batch, 1, 1, 4), np.float32), np.zeros((batch, 1, 1), bool)
            else:
                map_points, map_mask = self.maps["features"][rows, updates], self.maps["mask"][rows, updates]
            values.update(map_features=map_points, map_mask=map_mask, neighbor_features=points, neighbor_mask=valid)
        return {name: torch.from_numpy(np.ascontiguousarray(value)).to(device) for name, value in values.items()}

    def labels(self, indices, device):
        rows, _, _ = self._indices(indices)
        return torch.from_numpy(self.arrays["future_xy"][rows] - self.base[rows]).to(device)


def forward_inputs(model, inputs):
    names = ("features", "mask", "map_features", "map_mask", "neighbor_features", "neighbor_mask") if isinstance(model, InteractionCorrector) else ("features", "mask")
    return model(*(inputs[name] for name in names))
