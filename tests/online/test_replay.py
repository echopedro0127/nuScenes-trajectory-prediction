from __future__ import annotations

import numpy as np
import pytest

from trajectory_prediction.online.replay import ReplayUpdate, replay_updates


def test_update_never_exposes_later_observations() -> None:
    history = np.arange(10, dtype=np.float32).reshape(5, 2)
    future = np.arange(24, dtype=np.float32).reshape(12, 2) + 100

    updates = list(replay_updates(history, future, history_steps=5))

    first = updates[0]
    np.testing.assert_array_equal(first.observed_history[-1], future[0])
    assert first.remaining_target.shape == (11, 2)
    assert not np.isin(future[1:].ravel(), first.observed_history.ravel()).any()
    for update in updates:
        visible_future = future[: update.update_step]
        assert not np.shares_memory(update.observed_history, future)
        assert not np.isin(future[update.update_step + 1 :].ravel(), update.observed_history.ravel()).any()
        np.testing.assert_array_equal(update.observed_position, visible_future[-1])


def test_replay_updates_are_read_only_snapshots() -> None:
    updates = list(
        replay_updates(
            np.zeros((5, 2), dtype=np.float32),
            np.ones((4, 2), dtype=np.float32),
        )
    )

    assert isinstance(updates[0], ReplayUpdate)
    with pytest.raises(ValueError, match="read-only"):
        updates[0].observed_history[0, 0] = 1.0


@pytest.mark.parametrize(
    ("history", "future", "history_steps"),
    [
        (np.zeros((0, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32), 5),
        (np.zeros((2, 3), dtype=np.float32), np.ones((2, 2), dtype=np.float32), 5),
        (np.zeros((2, 2), dtype=np.int32), np.ones((2, 2), dtype=np.float32), 5),
        (np.zeros((2, 2), dtype=np.float32), np.ones((1, 2), dtype=np.float32), 5),
        (np.zeros((2, 2), dtype=np.float32), np.ones((2, 2), dtype=np.float32), 0),
    ],
)
def test_replay_rejects_invalid_inputs(
    history: np.ndarray, future: np.ndarray, history_steps: int
) -> None:
    with pytest.raises(ValueError):
        list(replay_updates(history, future, history_steps))
