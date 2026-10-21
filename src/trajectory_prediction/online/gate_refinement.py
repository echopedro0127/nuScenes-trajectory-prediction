"""Read-only P6 candidate adaptation and causal cross-update regularization."""

import math

import numpy as np
import torch

from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.online.gate_dataset import GateDataset


class RefinementDataset(GateDataset):
    def __init__(self, arrays, candidate="map"):
        super().__init__(arrays)
        if candidate not in ("map", "original"):
            raise ValueError("unsupported refinement candidate")
        self.candidate = candidate

    def fingerprint(self):
        return digest_json({"candidate": self.candidate, "arrays": super().fingerprint()})

    def inputs(self, indices, device):
        values = super().inputs(indices, device)
        if self.candidate == "original":
            _, i, k = self.indices(indices)
            candidate = np.ascontiguousarray(self.arrays["original"][i, k])
            values["map"] = torch.from_numpy(candidate).to(device)
            quality = values["quality"].clone()
            difference = torch.linalg.vector_norm(values["map"] - values["cv"], dim=-1)
            quality[:, 7] = (
                (difference * values["mask"]).double().sum(1) / values["mask"].sum(1) / 20.0
            ).to(quality.dtype)
            values["quality"] = quality
        return values

    def previous_indices(self, indices):
        _, _, k = self.indices(indices)
        return np.asarray(indices) - (k > 0)


def shrink_weights(weights, remaining_s, tau):
    if (
        isinstance(tau, bool)
        or not isinstance(tau, (int, float))
        or not math.isfinite(tau)
        or tau < 0
    ):
        raise ValueError("finite nonnegative shrink time required")
    if (
        weights.shape != remaining_s.shape
        or weights.device != remaining_s.device
        or not torch.isfinite(weights).all()
        or not torch.isfinite(remaining_s).all()
        or (weights < 0).any()
        or (weights > 1).any()
        or (remaining_s < 0.5).any()
    ):
        raise ValueError("invalid shrink inputs")
    return weights if tau == 0 else weights * torch.clamp((remaining_s - 0.5) / tau, 0, 1)


def paired_revision(now, previous, mask, has_previous):
    if (
        now.shape != previous.shape
        or now.ndim != 3
        or now.shape[1:] != (12, 2)
        or mask.shape != now.shape[:2]
        or has_previous.shape != (len(now),)
        or mask.dtype != torch.bool
        or has_previous.dtype != torch.bool
        or len({value.device for value in (now, previous, mask, has_previous)}) != 1
    ):
        raise ValueError("invalid paired revision shapes/device")
    if not torch.isfinite(now).all() or not torch.isfinite(previous).all():
        raise ValueError("nonfinite paired revision")
    valid = mask & has_previous[:, None]
    if not valid.any():
        return (now.sum() + previous.sum()) * 0.0
    distances = torch.linalg.vector_norm(now - previous, dim=-1)
    # Give each paired update equal weight, matching the reported per-update jitter.
    count = valid.sum(1)
    rows = count > 0
    return ((distances * valid).sum(1)[rows] / count[rows]).mean()
