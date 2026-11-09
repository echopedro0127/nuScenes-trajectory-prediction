"""Keyed degradation of all historical and newly released positions."""
import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Packet:
    source_step: int
    available_step: int
    position: np.ndarray

    def __post_init__(self):
        if type(self.source_step) is not int or type(self.available_step) is not int or self.source_step < -4 or self.available_step < self.source_step:
            raise ValueError("invalid source/availability steps")
        value = np.array(self.position, np.float64, copy=True)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError("packet position must be finite XY")
        value.setflags(write=False)
        object.__setattr__(self, "position", value)


@dataclass(frozen=True)
class Profile:
    name: str
    sigma: float
    delay_steps: int = 0
    drop_steps: tuple = ()
    outlier_steps: tuple = ()
    bias_m: float = 0.

    def __post_init__(self):
        if (not self.name or not np.isfinite(self.sigma) or self.sigma < 0 or not np.isfinite(self.bias_m)
                or self.bias_m < 0 or type(self.delay_steps) is not int or self.delay_steps < 0):
            raise ValueError("invalid degradation profile")


PROFILES = {p.name: p for p in (
    Profile("clean", 0.), Profile("noise_05", .5), Profile("noise_10", 1.), Profile("noise_20", 2.),
    Profile("noise_bias", .5, bias_m=2.), Profile("noise_outlier", .5, outlier_steps=(4, 8)),
    Profile("noise_dropout", .5, drop_steps=(4, 5, 6)),
    Profile("noise_delay_500ms", .5, delay_steps=1), Profile("noise_delay_1000ms", .5, delay_steps=2))}


def _normal(seed, key, step, channel):
    digest = hashlib.sha256(f"v2:{seed}:{key}:{step}:{channel}".encode()).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:8], "big"))).normal(size=2)


def make_packets(history, revealed, key, profile, seed):
    history, revealed = np.asarray(history, np.float64), np.asarray(revealed, np.float64)
    if (history.shape != (5, 2) or revealed.ndim != 2 or revealed.shape[1:] != (2,) or not 1 <= len(revealed) <= 11
            or not np.isfinite(history).all() or not np.isfinite(revealed).all() or not isinstance(key, str) or not key
            or type(seed) is not int or seed < 0):
        raise ValueError("finite history and causal release positions required")
    direction = _normal(seed, key, 0, "bias")
    bias = profile.bias_m * direction / max(np.linalg.norm(direction), 1e-12)
    packets = []
    for step, truth in zip(range(-4, len(revealed) + 1), np.concatenate((history, revealed)), strict=True):
        if step in profile.drop_steps:
            continue
        value = truth + profile.sigma * _normal(seed, key, step, "position") + bias
        if step in profile.outlier_steps:
            direction = _normal(seed, key, step, "outlier")
            value = value + 5. * direction / max(np.linalg.norm(direction), 1e-12)
        packets.append(Packet(step, step + profile.delay_steps, value))
    return tuple(sorted(packets, key=lambda p: (p.available_step, p.source_step)))
