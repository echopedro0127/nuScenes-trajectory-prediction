"""Causal map-packet timing and quality-only mixtures with an exact CV fallback."""
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

CONDITIONS = ("clean", "delay_500ms", "delay_1000ms", "dropout", "missing", "stale")


@dataclass(frozen=True)
class GateConfig:
    max_age_s: float = 2.
    fixed_scale: float = .5
    decay_s: float = 1.

    def __post_init__(self):
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) for value in (self.max_age_s, self.fixed_scale, self.decay_s)) or not 0 < self.max_age_s <= 2 or not 0 <= self.fixed_scale <= 1 or self.decay_s <= 0:
            raise ValueError("invalid gate packet policy")


def packet_schedule(condition, count=11, config=None):
    config = config or GateConfig()
    if condition not in CONDITIONS or type(count) is not int or not 1 <= count <= 11:
        raise ValueError("invalid packet condition/count")
    delay = {"delay_500ms": 1, "delay_1000ms": 2}.get(condition, 0)
    packets, availability, ages = [], [], []
    for step in range(1, count + 1):
        allowed = [j for j in range(1, step + 1) if j + delay <= step and condition != "missing"
                   and not (condition == "dropout" and j in (4, 5, 6)) and not (condition == "stale" and j != 1)]
        source = max(allowed, default=0)
        packets.append(source)
        availability.append(source + delay if source else 0)
        ages.append((step - source) * .5 if source else 0.)
    packet, age = np.array(packets, np.int64), np.array(ages, np.float32)
    return {"packet_step": packet, "available_step": np.array(availability, np.int64), "age_s": age,
            "age_valid": (packet > 0) & (age <= config.max_age_s)}


def quality_features(position, velocity, previous_velocity, base, steps, coverage, ages, cv, candidate):
    position, velocity, previous_velocity, base = (np.asarray(value, np.float32) for value in (position, velocity, previous_velocity, base))
    steps, coverage, ages = np.asarray(steps), np.asarray(coverage), np.asarray(ages)
    cv, candidate = np.asarray(cv, np.float32), np.asarray(candidate, np.float32)
    n = len(position)
    if position.shape != (n, 2) or velocity.shape != position.shape or previous_velocity.shape != position.shape or base.shape != (n, 12, 2) or cv.shape != base.shape or candidate.shape != base.shape or steps.shape != (n,) or coverage.shape != (n,) or ages.shape != (n,) or not np.issubdtype(steps.dtype, np.integer) or np.any((steps < 1) | (steps > 11)):
        raise ValueError("invalid quality input shapes")
    if not all(np.isfinite(value).all() for value in (position, velocity, previous_velocity, base, coverage, ages, cv, candidate)) or np.any((coverage < 0) | (coverage > 1)) or np.any(ages < 0):
        raise ValueError("nonfinite or invalid quality inputs")
    rows = np.arange(n)
    previous_base = base[rows, np.maximum(steps - 2, 0)].copy()
    previous_base[steps == 1] = 0
    base_velocity = (base[rows, steps - 1] - previous_base) / .5
    mask = np.arange(12)[None] >= steps[:, None]
    disagreement = (np.linalg.norm(candidate - cv, axis=-1) * mask).sum(1, dtype=np.float64) / mask.sum(1)
    return np.column_stack((coverage, np.clip(ages / 2., 0, 1), np.linalg.norm(position - base[rows, steps - 1], axis=-1) / 20.,
        np.linalg.norm(velocity - base_velocity, axis=-1) / 20., np.linalg.norm(velocity, axis=-1) / 20.,
        np.linalg.norm((velocity - previous_velocity) / .5, axis=-1) / 10., (12 - steps) / 12., disagreement / 20.)).astype(np.float32)


def fixed_weights(age_s, valid, config=None):
    config = config or GateConfig()
    if age_s.shape != valid.shape or valid.dtype != torch.bool or not torch.isfinite(age_s).all() or (age_s < 0).any():
        raise ValueError("invalid fixed-gate quality")
    return config.fixed_scale * torch.exp(-age_s / config.decay_s) * (valid & (age_s <= config.max_age_s)).to(age_s.dtype)


class QualityGate(nn.Module):
    def __init__(self, hidden_dim=16):
        super().__init__()
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("positive gate width required")
        self.network = nn.Sequential(nn.Linear(8, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.constant_(self.network[-1].bias, -2.)

    def forward(self, quality, valid):
        if quality.ndim != 2 or quality.shape[1] != 8 or valid.shape != (len(quality),) or valid.dtype != torch.bool or valid.device != quality.device or not torch.isfinite(quality).all():
            raise ValueError("invalid learned-gate quality inputs")
        return torch.sigmoid(self.network(quality).squeeze(-1)) * valid.to(quality.dtype)


def blend_forecasts(cv, candidate, weights):
    if cv.shape != candidate.shape or cv.ndim != 3 or cv.shape[1:] != (12, 2) or weights.shape != (len(cv),) or cv.device != candidate.device or weights.device != cv.device or not all(torch.isfinite(value).all() for value in (cv, candidate, weights)) or (weights < 0).any() or (weights > 1).any():
        raise ValueError("invalid forecast mixture inputs")
    weight = weights[:, None, None]
    mixed = cv + weight * (candidate - cv)
    return torch.where(weight == 0, cv, torch.where(weight == 1, candidate, mixed))
