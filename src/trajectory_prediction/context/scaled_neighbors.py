"""Vectorized affine transforms with the unchanged P3 selection and view validation."""
import numpy as np

from trajectory_prediction.context.neighbors import CATEGORY_PREFIXES, NeighborView
from trajectory_prediction.data.geometry import global_to_agent


def _current_arrays(index, sample):
    if not hasattr(index, "_scaled_candidates"):
        index._scaled_candidates = {}
    if sample not in index._scaled_candidates:
        tokens, points, categories = [], [], []
        for token, annotation in index.annotations.get(sample, {}).items():
            category = index.categories.get(token, "")
            group = next((i for i, prefix in enumerate(CATEGORY_PREFIXES) if category.startswith(prefix)), None)
            if group is not None:
                tokens.append(token)
                points.append(annotation["translation"][:2])
                categories.append(group)
        index._scaled_candidates[sample] = (np.asarray(tokens), np.asarray(points, dtype=float).reshape(-1, 2), np.asarray(categories))
    return index._scaled_candidates[sample]


def query_neighbors(index, observation, target_instance, scene, anchor_xy, anchor_yaw):
    config, timestamp = index.config, observation.timestamp_us
    sample = index._sample(scene, timestamp, timestamp)
    tokens, points, categories = _current_arrays(index, sample)
    relative = global_to_agent(points, anchor_xy, anchor_yaw) - observation.observed_position
    distances = np.linalg.norm(relative, axis=-1)
    candidates = np.flatnonzero((tokens != target_instance) & (distances <= config.radius_m))
    candidates = sorted(candidates, key=lambda i: (float(distances[i]), str(tokens[i])))[:config.max_neighbors]
    n = config.max_neighbors
    features, mask = np.zeros((n, 5, 6), np.float32), np.zeros((n, 5), bool)
    source, available = np.zeros((n, 5), np.int64), np.zeros((n, 5), np.int64)
    annotation_tokens = np.full((n, 5), "", dtype="U64")
    instances, history_xy = [""] * n, np.zeros((n, 5, 2), float)
    history = [index._sample(scene, timestamp - j * 500_000, timestamp) for j in range(4, -1, -1)]
    for j, token in enumerate(history):
        if token in history[j + 1:]:
            history[j] = None
    for i, candidate in enumerate(candidates):
        token, group = str(tokens[candidate]), int(categories[candidate])
        instances[i] = token
        for j, historical in enumerate(history):
            annotation = index.annotations.get(historical, {}).get(token)
            if annotation is not None:
                mask[i, j] = True
                history_xy[i, j] = annotation["translation"][:2]
                source[i, j] = index.samples[historical]["timestamp"]
                available[i, j] = source[i, j] + config.latency_us
                annotation_tokens[i, j] = annotation["token"]
                features[i, j, 4 + group] = 1
    features[..., :2] = global_to_agent(history_xy, anchor_xy, anchor_yaw) - observation.observed_position
    features[~mask] = 0
    delta = source[:, 1:] - source[:, :-1]
    velocity = np.zeros_like(mask)
    velocity[:, 1:] = mask[:, 1:] & mask[:, :-1] & (delta > 0) & (delta <= 1_000_000)
    rates = np.diff(features[..., :2], axis=1) / (np.maximum(delta, 1)[..., None] / 1e6)
    features[:, 1:, 2:4] = np.where(velocity[:, 1:, None], rates, 0)
    return NeighborView(features, mask, velocity, tuple(instances), annotation_tokens, source, available, timestamp)
