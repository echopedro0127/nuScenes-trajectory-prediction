import numpy as np
import pytest
from nuscenes import prediction
from nuscenes.eval.prediction import splits
from pyquaternion import Quaternion

from trajectory_prediction.data.nuscenes_adapter import (
    assemble_history,
    extract_split,
    quaternion_yaw,
)


def test_assemble_history_reverses_devkit_past_and_appends_current() -> None:
    newest_first = np.array([[0.0, -0.5], [0.0, -1.0]], dtype=np.float32)
    current = np.array([0.0, 0.0], dtype=np.float32)

    history = assemble_history(newest_first, current, expected_past_steps=2)

    np.testing.assert_allclose(history, [[0, -1], [0, -0.5], [0, 0]])
    assert history.dtype == np.float32


@pytest.mark.parametrize(
    ("past", "current"),
    [
        (np.zeros((1, 2)), np.zeros(2)),
        (np.zeros((2, 3)), np.zeros(2)),
        (np.zeros((2, 2)), np.zeros((1, 2))),
    ],
)
def test_assemble_history_rejects_incomplete_or_invalid_xy_shapes(
    past: np.ndarray, current: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        assemble_history(past, current, expected_past_steps=2)


def test_quaternion_yaw_returns_rotation_about_z() -> None:
    rotation = Quaternion(axis=[0, 0, 1], angle=0.75).elements

    assert quaternion_yaw(rotation) == pytest.approx(0.75)


class _FakeNuScenes:
    dataroot = "unused-test-root"

    def get(self, table: str, token: str) -> dict:
        if table == "sample":
            return {"scene_token": f"scene-for-{token}"}
        if table == "scene":
            return {"name": f"name-{token}"}
        raise AssertionError(f"unexpected table: {table}")


class _FakePredictHelper:
    future_offset = 0.0

    def __init__(self, nusc: _FakeNuScenes) -> None:
        self.nusc = nusc

    def get_sample_annotation(self, instance_token: str, sample_token: str) -> dict:
        return {
            "translation": [10.0, 5.0, 0.0],
            "rotation": [1.0, 0.0, 0.0, 0.0],
        }

    def get_past_for_agent(
        self,
        instance_token: str,
        sample_token: str,
        seconds: float,
        in_agent_frame: bool,
    ) -> np.ndarray:
        assert seconds == 2.0
        assert in_agent_frame is False
        steps = 3 if instance_token == "short-history" else 4
        return np.array(
            [[10.0 - 0.5 * step, 5.0] for step in range(1, steps + 1)],
            dtype=np.float64,
        )

    def get_future_for_agent(
        self,
        instance_token: str,
        sample_token: str,
        seconds: float,
        in_agent_frame: bool,
    ) -> np.ndarray:
        assert seconds == 6.0
        assert in_agent_frame is False
        steps = 11 if instance_token == "short-future" else 12
        return np.array(
            [
                [10.0 + self.future_offset + 0.5 * step, 5.0]
                for step in range(1, steps + 1)
            ],
            dtype=np.float64,
        )

    def get_velocity_for_agent(self, instance_token: str, sample_token: str) -> float:
        return np.inf


def test_extract_split_filters_incomplete_tracks_and_builds_local_float32_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = ["good_sample-a", "short-history_sample-b", "short-future_sample-c"]
    monkeypatch.setattr(splits, "get_prediction_challenge_split", lambda split, dataroot: keys)
    monkeypatch.setattr(prediction, "PredictHelper", _FakePredictHelper)

    arrays, counts = extract_split(_FakeNuScenes(), "mini_train")

    assert counts == {
        "requested": 3,
        "accepted": 1,
        "incomplete_history": 1,
        "incomplete_future": 1,
    }
    assert arrays["history_xy"].shape == (1, 5, 2)
    assert arrays["history_features"].shape == (1, 5, 6)
    assert arrays["history_mask"].shape == (1, 5)
    assert arrays["future_xy"].shape == (1, 12, 2)
    for name in (
        "history_xy",
        "history_features",
        "future_xy",
        "anchor_xy",
        "anchor_yaw",
        "current_speed",
    ):
        assert arrays[name].dtype == np.float32
    assert arrays["history_mask"].dtype == np.bool_
    np.testing.assert_allclose(arrays["history_xy"][0, :, 0], 0.0, atol=1e-6)
    np.testing.assert_allclose(arrays["history_xy"][0, :, 1], [-2, -1.5, -1, -0.5, 0])
    np.testing.assert_allclose(arrays["future_xy"][0, :, 1], np.arange(0.5, 6.5, 0.5))
    assert arrays["instance_token"].tolist() == ["good"]
    assert arrays["sample_token"].tolist() == ["sample-a"]
    assert arrays["scene_name"].tolist() == ["name-scene-for-sample-a"]
    assert arrays["current_speed"].tolist() == [0.0]


def test_extract_split_history_features_do_not_depend_on_future_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        splits,
        "get_prediction_challenge_split",
        lambda split, dataroot: ["good_sample-a"],
    )
    monkeypatch.setattr(prediction, "PredictHelper", _FakePredictHelper)

    first, _ = extract_split(_FakeNuScenes(), "mini_train")
    monkeypatch.setattr(_FakePredictHelper, "future_offset", 100.0)
    second, _ = extract_split(_FakeNuScenes(), "mini_train")

    np.testing.assert_array_equal(first["history_features"], second["history_features"])
    assert not np.array_equal(first["future_xy"], second["future_xy"])
