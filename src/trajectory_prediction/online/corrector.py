from __future__ import annotations

import torch
from torch import nn


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_inputs(features: torch.Tensor, mask: torch.Tensor, feature_dim: int) -> None:
    if features.ndim != 3 or features.shape[-1] != feature_dim:
        raise ValueError(f"features must have shape [B, T, {feature_dim}]")
    if mask.ndim != 2 or mask.shape != features.shape[:2]:
        raise ValueError("mask must have shape [B, T] matching features")
    if mask.dtype is not torch.bool:
        raise ValueError("mask must be a boolean tensor")
    if mask.device != features.device:
        raise ValueError("features and mask must be on the same device")


class ResidualCorrector(nn.Module):
    """Predict per-horizon residual offsets only for not-yet-elapsed points."""

    def __init__(self, feature_dim: int = 8, hidden_dim: int = 64) -> None:
        super().__init__()
        feature_dim = _positive_int("feature_dim", feature_dim)
        hidden_dim = _positive_int("hidden_dim", hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        self.feature_dim = feature_dim

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _validate_inputs(features, mask, self.feature_dim)
        return self.network(features) * mask.unsqueeze(-1).to(features.dtype)


def smoothness_loss(offsets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Penalize changes between adjacent valid correction offsets."""
    if offsets.ndim != 3 or offsets.shape[-1] != 2:
        raise ValueError("offsets must have shape [B, T, 2]")
    if mask.ndim != 2 or mask.shape != offsets.shape[:2] or mask.dtype is not torch.bool:
        raise ValueError("mask must be boolean with shape [B, T] matching offsets")
    if offsets.device != mask.device:
        raise ValueError("offsets and mask must be on the same device")
    if offsets.shape[1] < 2:
        return offsets.sum() * 0.0
    pair_mask = mask[:, 1:] & mask[:, :-1]
    delta = offsets[:, 1:] - offsets[:, :-1]
    if not pair_mask.any():
        return offsets.sum() * 0.0
    return delta.square().sum(dim=-1)[pair_mask].mean()
