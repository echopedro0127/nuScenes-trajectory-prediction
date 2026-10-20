"""Point-map conditioning with exact fallback to the original residual network."""
import numpy as np
import torch
from torch import nn

from trajectory_prediction.online.context_features import residual_features
from trajectory_prediction.online.corrector import (
    ResidualCorrector,
    _positive_int,
    _validate_inputs,
)


class MapCorrector(nn.Module):
    def __init__(self, fallback: ResidualCorrector, hidden_dim: int = 64):
        super().__init__()
        hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.hidden_dim = hidden_dim
        self.fallback = fallback
        self.fallback.requires_grad_(False)
        self.encoder = nn.Sequential(nn.Linear(4, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.head = nn.Sequential(nn.Linear(8 + hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 2))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, features, mask, map_features, map_mask):
        _validate_inputs(features, mask, 8)
        if map_features.ndim != 4 or map_features.shape[-1] != 4 or map_features.shape[0] != features.shape[0]:
            raise ValueError("map features must be [B, K, P, 4]")
        if map_mask.shape != map_features.shape[:3] or map_mask.dtype != torch.bool:
            raise ValueError("map mask must be bool [B, K, P]")
        if map_mask.device != features.device or map_features.device != features.device:
            raise ValueError("map and trajectory must be on same device")
        safe_points = map_features.masked_fill(~map_mask[..., None], 0)
        encoded = self.encoder(safe_points)
        encoded = encoded.masked_fill(~map_mask[..., None], 0)
        count = map_mask.sum(dim=(1, 2))
        pooled = encoded.sum(dim=(1, 2)) / count.clamp_min(1)[:, None]
        expanded = pooled[:, None].expand(-1, features.shape[1], -1)
        delta = self.head(torch.cat((features, expanded), dim=-1))
        delta = delta.masked_fill(~(mask & (count > 0)[:, None])[..., None], 0)
        return self.fallback(features, mask) + delta


def predict_offsets(model, observation, base, map_view, device):
    """Production prediction boundary; evaluation labels are never accepted."""
    features, mask = residual_features(observation, base)
    with torch.inference_mode():
        offsets = model(torch.from_numpy(features[None]).to(device),
                        torch.from_numpy(mask[None]).to(device),
                        torch.from_numpy(map_view.features[None].copy()).to(device),
                        torch.from_numpy(map_view.mask[None].copy()).to(device))
    result = offsets[0].detach().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError("model produced nonfinite offsets")
    return result
