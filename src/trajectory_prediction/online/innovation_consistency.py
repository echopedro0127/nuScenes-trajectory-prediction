"""Observed-prefix innovation weights and differentiable absolute-time revision loss."""

import hashlib

import numpy as np
import torch

from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.online.horizon_features import HorizonDataset

ARMS = ("revision_0", "revision_fixed", "revision_adaptive", "revision_mean")
INNOVATION_SCALE = .5


class InnovationDataset(HorizonDataset):
    def __init__(self, arrays):
        super().__init__(arrays)
        self.innovation = np.zeros((self.n, 11), np.float32)
        self.innovation[:, 1:] = np.linalg.norm(
            arrays["position"][:, 1:]-arrays["cv"][:, np.arange(10), np.arange(1, 11)], axis=-1)
        self.revision_weights = np.exp(-(self.innovation/INNOVATION_SCALE)**2).astype(np.float32)
        self.revision_weights[:, 0] = 0
        if not np.isfinite(self.innovation).all() or not np.isfinite(self.revision_weights).all():
            raise ValueError("nonfinite observed innovation")
        self.train_mean_weight = float(self.revision_weights[:, 1:].mean(dtype=np.float64))

    def fingerprint(self):
        return digest_json({"parent": super().fingerprint(), "scale": INNOVATION_SCALE,
            "innovation": hashlib.sha256(self.innovation.tobytes()).hexdigest(),
            "weights": hashlib.sha256(self.revision_weights.tobytes()).hexdigest(),
            "train_mean_weight": self.train_mean_weight})

    def inputs(self, indices, device):
        values = super().inputs(indices, device)
        _, i, k = self.indices(indices)
        values["revision_weights"] = torch.from_numpy(np.ascontiguousarray(self.revision_weights[i, k])).to(device)
        return values


def weighted_revision(now, previous, mask, has_previous, weights):
    if (now.shape != previous.shape or now.ndim != 3 or now.shape[1:] != (12, 2)
        or mask.shape != now.shape[:2] or mask.dtype != torch.bool
        or has_previous.shape != (len(now),) or has_previous.dtype != torch.bool
        or weights.shape != (len(now),) or not weights.is_floating_point()
        or len({v.device for v in (now, previous, mask, has_previous, weights)}) != 1
        or not all(torch.isfinite(v).all() for v in (now, previous, weights))
        or ((weights < 0) | (weights > 1)).any()):
        raise ValueError("invalid weighted revision shapes/device/weights")
    valid = mask & has_previous[:, None]
    count = valid.sum(1)
    rows = count > 0
    if not rows.any():
        return (now.sum()+previous.sum())*0.
    distances = torch.linalg.vector_norm(now-previous, dim=-1)
    return (((distances*valid).sum(1)[rows]/count[rows])*weights[rows]).mean()
