from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReplayUpdate:
    """One causal update in a streaming trajectory replay.

    Arrays are read-only snapshots. ``update_step`` is one-indexed: update 1 reveals
    the first future point and leaves points 2..T hidden from the model.
    """

    update_step: int
    observed_history: np.ndarray
    observed_position: np.ndarray
    remaining_target: np.ndarray


def _validate_points(name: str, value: np.ndarray, *, minimum_steps: int) -> np.ndarray:
    points = np.asarray(value)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < minimum_steps:
        raise ValueError(f"{name} must have shape [T >= {minimum_steps}, 2]")
    if not np.issubdtype(points.dtype, np.floating):
        raise ValueError(f"{name} must be a floating-point array")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} must contain only finite values")
    return points.astype(np.float32, copy=True)


def _readonly(array: np.ndarray) -> np.ndarray:
    snapshot = np.asarray(array, dtype=np.float32).copy()
    snapshot.setflags(write=False)
    return snapshot


def replay_updates(
    initial_history: np.ndarray,
    future: np.ndarray,
    history_steps: int = 5,
) -> Iterator[ReplayUpdate]:
    """Yield causal sliding-window observations as future points are revealed.

    At update ``k`` (1 <= k < len(future)), only ``future[:k]`` has been revealed.
    The returned ``remaining_target`` starts at ``future[k]`` and is never exposed
    through ``observed_history`` or ``observed_position``.
    """
    if (
        isinstance(history_steps, bool)
        or not isinstance(history_steps, int)
        or history_steps <= 0
    ):
        raise ValueError("history_steps must be a positive integer")
    observed = _validate_points("initial_history", initial_history, minimum_steps=1)
    target = _validate_points("future", future, minimum_steps=2)

    for index in range(target.shape[0] - 1):
        observed = np.concatenate((observed, target[index : index + 1]), axis=0)
        yield ReplayUpdate(
            update_step=index + 1,
            observed_history=_readonly(observed[-history_steps:]),
            observed_position=_readonly(target[index]),
            remaining_target=_readonly(target[index + 1 :]),
        )
