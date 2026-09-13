import numpy as np
from numpy.typing import ArrayLike, NDArray


def _rotation(angle: float) -> NDArray[np.float64]:
    cosine = np.cos(angle)
    sine = np.sin(angle)
    return np.array([[cosine, -sine], [sine, cosine]], dtype=np.float64)


def _coordinate_arrays(
    points: ArrayLike, anchor_xy: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    points_array = np.asarray(points, dtype=np.float64)
    anchor_array = np.asarray(anchor_xy, dtype=np.float64)
    if points_array.ndim == 0 or points_array.shape[-1] != 2:
        raise ValueError('points must have final dimension 2')
    if anchor_array.shape != (2,):
        raise ValueError('anchor_xy must have shape (2,)')
    return points_array, anchor_array


def global_to_agent(
    points: ArrayLike, anchor_xy: ArrayLike, anchor_yaw: float
) -> NDArray[np.float32]:
    points_array, anchor_array = _coordinate_arrays(points, anchor_xy)
    rotation = _rotation(np.pi / 2 - anchor_yaw)
    return ((points_array - anchor_array) @ rotation.T).astype(np.float32)


def agent_to_global(
    points: ArrayLike, anchor_xy: ArrayLike, anchor_yaw: float
) -> NDArray[np.float32]:
    points_array, anchor_array = _coordinate_arrays(points, anchor_xy)
    rotation = _rotation(np.pi / 2 - anchor_yaw)
    return (points_array @ rotation + anchor_array).astype(np.float32)


def build_motion_features(
    history_xy: ArrayLike, dt: float
) -> NDArray[np.float32]:
    xy = np.asarray(history_xy, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[0] < 1 or xy.shape[1] != 2:
        raise ValueError('history_xy must have shape [T, 2] with T >= 1')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and greater than zero')
    displacement = np.zeros_like(xy)
    displacement[1:] = xy[1:] - xy[:-1]
    velocity = displacement / np.float32(dt)
    return np.concatenate((xy, displacement, velocity), axis=-1)
