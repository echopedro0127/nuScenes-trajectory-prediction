"""Keyed position degradation; all consumers share one timestamp-aware causal state."""
import hashlib
from dataclasses import dataclass

import numpy as np

from trajectory_prediction.context.schema import validate_timestamp_us
from trajectory_prediction.online.batched_context import residual_batch

CONDITIONS = ("clean", "noise", "delay_500ms", "delay_1000ms", "dropout")


@dataclass(frozen=True)
class CausalState:
    update_step: int
    timestamp_us: int
    source_us: int
    available_us: int
    position: np.ndarray
    velocity: np.ndarray
    fresh: bool

    def __post_init__(self):
        for name in ("timestamp_us", "source_us", "available_us"):
            validate_timestamp_us(getattr(self, name))
        if not self.source_us <= self.available_us <= self.timestamp_us or type(self.update_step) is not int or not 1 <= self.update_step <= 11:
            raise ValueError("invalid causal state timeline")
        for name in ("position", "velocity"):
            value = np.array(getattr(self, name), np.float32, copy=True)
            if value.shape != (2,) or not np.isfinite(value).all():
                raise ValueError("state vectors must be finite XY")
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def age_s(self):
        return (self.timestamp_us - self.source_us) / 1e6


def replay_states(initial_history, revealed_positions, anchor_us, key, condition, seed=2027):
    history, updates = np.asarray(initial_history, np.float32), np.asarray(revealed_positions, np.float32)
    validate_timestamp_us(anchor_us)
    if condition not in CONDITIONS or history.shape != (5, 2) or updates.ndim != 2 or updates.shape[1] != 2 or not 1 <= len(updates) <= 11 or not np.isfinite(history).all() or not np.isfinite(updates).all() or anchor_us < 2000000 or not isinstance(key, str) or not key or type(seed) is not int or seed < 0:
        raise ValueError("invalid observation replay inputs")
    delay = {"delay_500ms": 500000, "delay_1000ms": 1000000}.get(condition, 0)
    delivered = [(anchor_us - 500000, history[-2]), (anchor_us, history[-1])]
    available, queue = anchor_us, []
    for step, clean in enumerate(updates, 1):
        now = anchor_us + step * 500000
        if not (condition == "dropout" and step in (4, 5, 6)):
            value = clean.copy()
            if condition == "noise":
                token = hashlib.sha256(f"{seed}:{key}:{step}".encode()).digest()
                value += np.random.default_rng(int.from_bytes(token[:8], "big")).normal(0., 1., 2).astype(np.float32)
            queue.append((now, now + delay, value))
        fresh = False
        while queue and queue[0][1] <= now:
            source, available, value = queue.pop(0)
            delivered = [delivered[-1], (source, value)]
            fresh = True
        previous, latest = delivered
        velocity = (latest[1] - previous[1]) / np.float32((latest[0] - previous[0]) / 1e6)
        position = latest[1] + velocity * np.float32((now - latest[0]) / 1e6)
        yield CausalState(step, now, latest[0], available, position, velocity, fresh)


def state_residual(state, base):
    features, mask = residual_batch(state.position[None], (state.position - .5 * state.velocity)[None],
                                    np.asarray(base)[None], np.array([state.update_step]))
    return features[0], mask[0]


def filtered_positions(initial_history, anchor_us, states, alpha=.6, beta=.2):
    history = np.asarray(initial_history, np.float32)
    if history.shape != (5, 2) or not np.isfinite(history).all() or not 0 < alpha <= 1 or not 0 < beta <= 1:
        raise ValueError("invalid alpha-beta filter inputs")
    position, velocity = history[-1].copy(), (history[-1] - history[-2]) / .5
    source, output, velocities = anchor_us, [], []
    for state in states:
        if state.source_us > source:
            dt = (state.source_us - source) / 1e6
            measured = state.position - state.velocity * state.age_s
            prediction = position + velocity * dt
            residual = measured - prediction
            position, velocity = prediction + alpha * residual, velocity + beta / dt * residual
            source = state.source_us
        output.append(position + velocity * ((state.timestamp_us - source) / 1e6))
        velocities.append(velocity.copy())
    return np.asarray(output, np.float32), np.asarray(velocities, np.float32)
