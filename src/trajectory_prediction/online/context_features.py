"""Residual features derived solely from the causal observation and fixed forecast."""
import numpy as np

from trajectory_prediction.online.observations import ObservationView


def residual_features(observation: ObservationView, base: np.ndarray):
    base = np.asarray(base, dtype=np.float32)
    if base.shape != (12, 2) or not np.isfinite(base).all():
        raise ValueError("base must be finite [12, 2]")
    if len(observation.observed_history) < 2:
        raise ValueError("residual velocity requires two observed history points")
    k = observation.update_step
    current = observation.observed_position
    velocity = (current - observation.observed_history[-2]) / np.float32(0.5)
    previous = np.zeros(2, np.float32) if k == 1 else base[k - 2]
    base_velocity = (base[k - 1] - previous) / np.float32(0.5)
    heading = (np.arctan2(velocity[1], velocity[0]) - np.arctan2(base_velocity[1], base_velocity[0])
               + np.pi) % (2 * np.pi) - np.pi
    features = np.empty((12, 8), np.float32)
    features[:, :2] = base
    features[:, 2:4] = current - base[k - 1]
    features[:, 4:6] = velocity - base_velocity
    features[:, 6] = heading
    features[:, 7] = (12. - np.arange(12, dtype=np.float32) - 1.) / 12.
    mask = np.arange(12) >= k
    features[~mask] = 0
    return features, mask
