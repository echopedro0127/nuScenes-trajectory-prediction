"""Radar summary conditioning with an exact frozen-control fallback."""
from copy import deepcopy

import numpy as np
import torch
from torch import nn

from trajectory_prediction.online.batched_context import BatchedContext, residual_batch
from trajectory_prediction.online.corrector import _positive_int, _validate_inputs


class RadarCorrector(nn.Module):
    def __init__(self, fallback, hidden_dim=64):
        super().__init__()
        hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.fallback = fallback.requires_grad_(False)
        self.head = nn.Sequential(nn.Linear(21, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, mask, radar, present):
        _validate_inputs(features, mask, 8)
        if radar.shape != (len(features), 13) or present.shape != (len(features),) or present.dtype != torch.bool or radar.device != features.device or present.device != features.device:
            raise ValueError("radar tensor shape/device mismatch")
        context = radar[:, None].expand(-1, 12, -1)
        offsets = self.head(torch.cat((features, context), -1))
        offsets = offsets.masked_fill(~(mask & present[:, None])[..., None], 0)
        return self.fallback(features, mask) + offsets


class RadarDataset:
    def __init__(self, arrays, cache, base, sample_indices=None):
        self.arrays, self.cache, self.base = arrays, cache, base
        self.sample_indices = np.asarray(sample_indices if sample_indices is not None else np.arange(len(base)), dtype=np.int64)
        if self.sample_indices.ndim != 1 or not len(self.sample_indices) or len(np.unique(self.sample_indices)) != len(self.sample_indices) or np.any((self.sample_indices < 0) | (self.sample_indices >= len(base))):
            raise ValueError("invalid radar dataset sample indices")
        self.size = len(self.sample_indices) * 55

    @property
    def scenes(self):
        return sorted(set(self.arrays["scene_name"][self.sample_indices].tolist()))

    def indices(self, indices):
        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or np.any((indices < 0) | (indices >= self.size)):
            raise ValueError("radar indices out of bounds")
        return indices // (len(self.sample_indices) * 11), self.sample_indices[(indices // 11) % len(self.sample_indices)], indices % 11

    def inputs(self, indices, device):
        c, i, k = self.indices(indices)
        position, velocity = self.cache["position"][c, i, k], self.cache["velocity"][c, i, k]
        features, mask = residual_batch(position, position - .5 * velocity, self.base[i], k + 1)
        return {name: torch.from_numpy(np.ascontiguousarray(value)).to(device) for name, value in
            (("features", features), ("mask", mask), ("radar", self.cache["radar_features"][c, i, k]), ("present", self.cache["present"][c, i, k]))}

    def labels(self, indices, device):
        _, i, _ = self.indices(indices)
        return torch.from_numpy(np.ascontiguousarray(self.arrays["future_xy"][i] - self.base[i])).to(device)

    def fingerprint(self):
        cache = {**self.cache, "selected_samples": self.sample_indices}
        return BatchedContext(self.arrays, cache, {}, self.base).fingerprint()


def forward_model(model, inputs):
    if isinstance(model, RadarCorrector):
        return model(inputs["features"], inputs["mask"], inputs["radar"], inputs["present"])
    return model(inputs["features"], inputs["mask"])


def clone_fallback(model, hidden_dim):
    return RadarCorrector(deepcopy(model), hidden_dim)
