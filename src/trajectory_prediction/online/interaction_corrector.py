"""Independent map, neighbors and fusion heads with a common frozen control."""
import math
import time

import numpy as np
import torch
from torch import nn

from trajectory_prediction.online.context_corrector import _sync
from trajectory_prediction.online.context_features import residual_features
from trajectory_prediction.online.corrector import _positive_int, _validate_inputs

MODES = ("map", "neighbors", "fusion")


def neighbor_features(view):
    age = np.where(view.mask, (view.timestamp_us - view.source_timestamp_us) / 2_000_000, 0)
    return np.concatenate((view.features, view.velocity_mask[..., None].astype(np.float32),
                           age[..., None].astype(np.float32)), axis=-1)


def matched_width(parameter_count):
    if type(parameter_count) is not int or parameter_count < 15:
        raise ValueError("parameter count must be positive residual network size")
    root = (-12 + math.sqrt(144 + 4 * (parameter_count - 2))) / 2
    return min({max(1, math.floor(root)), max(1, math.ceil(root))},
               key=lambda width: (abs(width * width + 12 * width + 2 - parameter_count), width))


def _encoder(dim, hidden):
    return nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden))


def _pool(encoder, features, mask, dim):
    if features.ndim != 4 or features.shape[-1] != dim or mask.shape != features.shape[:3] or mask.dtype != torch.bool:
        raise ValueError("context features/mask shape or dtype mismatch")
    if features.device != mask.device:
        raise ValueError("context features and masks must share device")
    safe = features.masked_fill(~mask[..., None], 0)
    encoded = encoder(safe).masked_fill(~mask[..., None], 0)
    count = mask.sum((1, 2))
    return encoded.sum((1, 2)) / count.clamp_min(1)[:, None], count > 0


class InteractionCorrector(nn.Module):
    def __init__(self, fallback, mode="fusion", hidden_dim=64):
        super().__init__()
        if mode not in MODES:
            raise ValueError("mode must be map, neighbors or fusion")
        hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.mode, self.hidden_dim = mode, hidden_dim
        self.fallback = fallback
        self.fallback.requires_grad_(False)
        self.map_encoder = _encoder(4, hidden_dim) if mode in ("map", "fusion") else None
        self.neighbor_encoder = _encoder(8, hidden_dim) if mode in ("neighbors", "fusion") else None
        modalities = 2 if mode == "fusion" else 1
        self.head = nn.Sequential(nn.Linear(8 + modalities * (hidden_dim + 1), hidden_dim),
                                  nn.GELU(), nn.Linear(hidden_dim, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, mask, map_features, map_mask, neighbor_features, neighbor_mask):
        _validate_inputs(features, mask, 8)
        pooled, present = [], []
        for encoder, points, valid, dim in ((self.map_encoder, map_features, map_mask, 4),
                                            (self.neighbor_encoder, neighbor_features, neighbor_mask, 8)):
            if encoder is not None:
                value, available = _pool(encoder, points, valid, dim)
                if value.shape[0] != features.shape[0] or value.device != features.device:
                    raise ValueError("context and trajectory batch/device mismatch")
                pooled.extend((value, available[:, None].to(value.dtype)))
                present.append(available)
        context = torch.cat(pooled, -1)[:, None].expand(-1, features.shape[1], -1)
        offsets = self.head(torch.cat((features, context), -1))
        available = torch.stack(present).any(0)
        offsets = offsets.masked_fill(~(mask & available[:, None])[..., None], 0)
        return self.fallback(features, mask) + offsets


def predict_interaction(model, observation, base, map_view, neighbor_view, device):
    """Production boundary; no evaluation target can enter the predictor."""
    _sync(device)
    start = time.perf_counter_ns()
    residual, horizon = residual_features(observation, base)
    args = [torch.from_numpy(array[None].copy()).to(device) for array in
            (residual, horizon, map_view.features, map_view.mask,
             neighbor_features(neighbor_view), neighbor_view.mask)]
    _sync(device)
    prepared = time.perf_counter_ns()
    with torch.inference_mode():
        result = model(*args)[0].detach().cpu().numpy()
    _sync(device)
    finished = time.perf_counter_ns()
    if not np.isfinite(result).all():
        raise ValueError("interaction model produced nonfinite predictions")
    return result, {"feature_ms": (prepared - start) / 1e6, "encoder_head_ms": (finished - prepared) / 1e6}
