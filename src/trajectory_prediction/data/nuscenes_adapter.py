from collections.abc import Iterable
from typing import Any

import numpy as np
from pyquaternion import Quaternion

from trajectory_prediction.data.cache import REQUIRED_ARRAYS
from trajectory_prediction.data.geometry import build_motion_features, global_to_agent


def assemble_history(
    past_newest_first: np.ndarray,
    current_xy: np.ndarray,
    expected_past_steps: int,
) -> np.ndarray:
    past = np.asarray(past_newest_first, dtype=np.float32)
    current = np.asarray(current_xy, dtype=np.float32)
    if past.ndim != 2 or past.shape[1] != 2:
        raise ValueError("past_newest_first must have shape [T, 2]")
    if past.shape[0] != expected_past_steps:
        raise ValueError(
            f"expected {expected_past_steps} past points, received {past.shape[0]}"
        )
    if current.shape != (2,):
        raise ValueError("current_xy must have shape (2,)")
    return np.concatenate((past[::-1], current[None, :]), axis=0)


def quaternion_yaw(rotation: Iterable[float]) -> float:
    return float(Quaternion(rotation).yaw_pitch_roll[0])


def _records_to_arrays(records: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    arrays = {
        "history_xy": np.asarray(records["history_xy"], dtype=np.float32).reshape(-1, 5, 2),
        "history_features": np.asarray(
            records["history_features"], dtype=np.float32
        ).reshape(-1, 5, 6),
        "history_mask": np.asarray(records["history_mask"], dtype=np.bool_).reshape(-1, 5),
        "future_xy": np.asarray(records["future_xy"], dtype=np.float32).reshape(-1, 12, 2),
        "anchor_xy": np.asarray(records["anchor_xy"], dtype=np.float32).reshape(-1, 2),
        "anchor_yaw": np.asarray(records["anchor_yaw"], dtype=np.float32),
        "current_speed": np.asarray(records["current_speed"], dtype=np.float32),
        "instance_token": np.asarray(records["instance_token"], dtype=np.str_),
        "sample_token": np.asarray(records["sample_token"], dtype=np.str_),
        "scene_name": np.asarray(records["scene_name"], dtype=np.str_),
    }
    return arrays


def extract_split(
    nusc: Any, split: str, history_seconds: float = 2.0
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    from nuscenes.eval.prediction.splits import get_prediction_challenge_split
    from nuscenes.prediction import PredictHelper

    helper = PredictHelper(nusc)
    keys = get_prediction_challenge_split(split, dataroot=nusc.dataroot)
    records: dict[str, list[Any]] = {name: [] for name in REQUIRED_ARRAYS}
    rejected = {"incomplete_history": 0, "incomplete_future": 0}

    for key in keys:
        instance_token, sample_token = key.split("_", maxsplit=1)
        annotation = helper.get_sample_annotation(instance_token, sample_token)
        past = helper.get_past_for_agent(
            instance_token,
            sample_token,
            history_seconds,
            in_agent_frame=False,
        )
        future = helper.get_future_for_agent(
            instance_token,
            sample_token,
            6.0,
            in_agent_frame=False,
        )
        if len(past) != 4:
            rejected["incomplete_history"] += 1
            continue
        if len(future) != 12:
            rejected["incomplete_future"] += 1
            continue

        anchor_xy = np.asarray(annotation["translation"][:2], dtype=np.float32)
        anchor_yaw = quaternion_yaw(annotation["rotation"])
        history_global = assemble_history(past, anchor_xy, expected_past_steps=4)
        history_local = global_to_agent(history_global, anchor_xy, anchor_yaw)
        future_local = global_to_agent(future, anchor_xy, anchor_yaw)
        sample = nusc.get("sample", sample_token)
        scene_name = nusc.get("scene", sample["scene_token"])["name"]
        speed = helper.get_velocity_for_agent(instance_token, sample_token)

        records["history_xy"].append(history_local)
        records["history_features"].append(build_motion_features(history_local, dt=0.5))
        records["history_mask"].append(np.ones(5, dtype=np.bool_))
        records["future_xy"].append(future_local)
        records["anchor_xy"].append(anchor_xy)
        records["anchor_yaw"].append(anchor_yaw)
        records["current_speed"].append(float(speed) if np.isfinite(speed) else 0.0)
        records["instance_token"].append(instance_token)
        records["sample_token"].append(sample_token)
        records["scene_name"].append(scene_name)

    arrays = _records_to_arrays(records)
    counts = {
        "requested": len(keys),
        "accepted": len(arrays["future_xy"]),
        **rejected,
    }
    return arrays, counts
