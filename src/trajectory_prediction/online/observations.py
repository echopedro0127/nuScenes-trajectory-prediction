from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from trajectory_prediction.context.schema import FrameRef, validate_timestamp_us
from trajectory_prediction.context.timeline import latest_available
from trajectory_prediction.online.replay import replay_updates


def _snapshot(value: np.ndarray) -> np.ndarray:
    result = np.array(value, dtype=np.float32, copy=True)
    if not np.isfinite(result).all():
        raise ValueError("observation arrays must be finite")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ObservationView:
    """Causal model input with copied, read-only trajectory snapshots."""

    update_step: int
    timestamp_us: int
    observed_history: np.ndarray
    observed_position: np.ndarray
    frames: tuple[FrameRef, ...]

    def __post_init__(self) -> None:
        validate_timestamp_us(self.timestamp_us)
        if type(self.update_step) is not int or not 1 <= self.update_step <= 11:
            raise ValueError("update step must be in [1, 11]")
        history = _snapshot(self.observed_history)
        position = _snapshot(self.observed_position)
        if history.ndim != 2 or history.shape[1] != 2 or len(history) == 0:
            raise ValueError("history must have shape [H >= 1, 2]")
        if position.shape != (2,):
            raise ValueError("position must have shape [2]")
        frames = tuple(self.frames)
        if any(item.available_timestamp_us > self.timestamp_us for item in frames):
            raise ValueError("observation cannot contain an unavailable frame")
        object.__setattr__(self, "observed_history", history)
        object.__setattr__(self, "observed_position", position)
        object.__setattr__(self, "frames", frames)


@dataclass(frozen=True)
class EvaluationLabel:
    """Remaining trajectory for loss/evaluation, separate from model input."""

    remaining_target: np.ndarray

    def __post_init__(self) -> None:
        target = _snapshot(self.remaining_target)
        if target.ndim != 2 or target.shape[1] != 2 or len(target) == 0:
            raise ValueError("target must have shape [T >= 1, 2]")
        object.__setattr__(self, "remaining_target", target)


def replay_observations(
    initial_history: np.ndarray,
    future: np.ndarray,
    *,
    anchor_timestamp_us: int,
    history_steps: int = 5,
    frames: tuple[FrameRef, ...] = (),
) -> Iterator[tuple[ObservationView, EvaluationLabel]]:
    """Adapt the fixed 12-point replay without extending its original endpoint.

    Anchor time is t0; update k reveals point k at t0 + k * 500,000 us.
    Each channel supplies its latest available frame no older than two seconds.
    Sensor references are selected without performing sensor I/O.
    """
    validate_timestamp_us(anchor_timestamp_us)
    if np.asarray(future).shape != (12, 2):
        raise ValueError("context replay requires exactly 12 future XY points")
    records = tuple(frames)
    channels = sorted({item.channel for item in records})
    for update in replay_updates(initial_history, future, history_steps):
        timestamp_us = anchor_timestamp_us + update.update_step * 500_000
        selected = tuple(
            item for channel in channels
            if (item := latest_available(records, channel, timestamp_us)) is not None
        )
        yield (
            ObservationView(
                update.update_step, timestamp_us, update.observed_history,
                update.observed_position, selected,
            ),
            EvaluationLabel(update.remaining_target),
        )
