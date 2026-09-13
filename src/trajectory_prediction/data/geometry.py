import numpy as np
from numpy.typing import ArrayLike, NDArray


def _rotation(angle: float) -> NDArray[np.float64]:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def global_to_agent(
    points: ArrayLike, anchor_xy: ArrayLike, anchor_yaw: float
) -> NDArray[np.float32]:
    points_array = np.asarray(points, dtype=np.float64)
    anchor_array = np.asarray(anchor_xy, dtype=np.float64)
    rotation = _rotation(np.pi / 2 - anchor_yaw)
    return ((points_array - anchor_array) @ rotation.T).astype(np.float32)


def agent_to_global(
    points: ArrayLike, anchor_xy: ArrayLike, anchor_yaw: float
) -> NDArray[np.float32]:
    points_array = np.asarray(points, dtype=np.float64)
    anchor_array = np.asarray(anchor_xy, dtype=np.float64)
    rotation = _rotation(np.pi / 2 - anchor_yaw)
    return (points_array @ rotation + anchor_array).astype(np.float32)


def build_motion_features(
    history_xy: ArrayLike, dt: float
) -> NDArray[np.float32]:
    xy = np.asarray(history_xy, dtype=np.float32)
    displacement = np.zeros_like(xy)
    displacement[1:] = xy[1:] - xy[:-1]
    velocity = displacement / np.float32(dt)
    return np.concatenate((xy, displacement, velocity), axis=-1)
