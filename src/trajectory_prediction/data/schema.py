from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectorySample:
    instance_token: str
    sample_token: str
    scene_name: str
    history_xy: np.ndarray
    history_features: np.ndarray
    history_mask: np.ndarray
    future_xy: np.ndarray
    anchor_xy: np.ndarray
    anchor_yaw: float
    current_speed: float
