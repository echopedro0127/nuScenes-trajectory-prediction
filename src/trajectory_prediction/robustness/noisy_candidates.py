"""Rebuild frozen experts from regular arrived packets; no evaluation labels."""
import numpy as np
import torch

from trajectory_prediction.data.geometry import build_motion_features
from trajectory_prediction.online.batched_context import residual_batch
from trajectory_prediction.online.observations import ObservationView
from trajectory_prediction.robustness.filters import STRATEGIES, Estimator, _transition, run_packets

CANDIDATES = (*STRATEGIES, 'gru_base', 'original', 'map')
FEATURE_GROUPS = {'constant': 0, 'motion': 6, 'covariance': 12, 'innovation': 18}


def build_candidates(packets, gru, original, map_model, lane_map, anchor_xy, anchor_yaw, timestamp_us):
    packets = tuple(packets)
    if ([p.source_step for p in packets] != list(range(-4, 12))
            or any(p.available_step != p.source_step for p in packets)):
        raise ValueError('Phase 2A requires regular complete zero-delay packets')
    for model in (gru, original, map_model):
        model.eval()
    history = np.array([p.position for p in packets[:5]], np.float32)
    origin = history[-1].copy()
    with torch.inference_mode():
        base = gru(torch.from_numpy(build_motion_features(history-origin, .5)[None]),
                   torch.ones((1, 5), dtype=torch.bool)).cpu().numpy()[0] + origin
    physical = [run_packets(packets, name) for name in STRATEGIES]
    candidates = np.empty((11, len(CANDIDATES), 12, 2), np.float32)
    for j, result in enumerate(physical):
        candidates[:, j] = result['forecasts']
    candidates[:, 4] = base
    current = np.array([p.position for p in packets[5:]], np.float32)
    previous = np.concatenate((history[-1:], current[:-1]))
    raw_velocity = (current-previous) / .5
    features = np.zeros((11, 18), np.float32)
    points, masks, coverage = [], [], []
    kf = Estimator('kf_cv', packets[0], packets[1])
    for packet in packets[2:5]:
        kf.receive(packet)
    previous_residual = np.zeros(2)
    previous_velocity = (history[-1]-history[-2]) / .5
    ema = 0.
    for k, packet in enumerate(packets[5:]):
        predicted = _transition(.5)[0] @ kf.x
        residual = packet.position - predicted[:2]
        kf.receive(packet)
        ema = .8 * ema + .2 * np.log1p(kf.nis)
        observation = ObservationView(k+1, timestamp_us+(k+1)*500000,
                                      np.concatenate((history, current[:k+1]))[-5:], current[k], ())
        view = lane_map.query(observation, anchor_xy, anchor_yaw)
        map_points = view.features.copy()
        map_points[..., :2] -= origin
        map_points[~view.mask] = 0
        points.append(map_points)
        masks.append(view.mask)
        cov = float(view.mask.mean())
        coverage.append(cov)
        velocity = [result['forecasts'][k, 1]-result['forecasts'][k, 0] for result in physical[:3]]
        features[k, :6] = [*[np.linalg.norm(v)*2/20 for v in velocity],
                           np.linalg.norm(physical[0]['positions'][k]-physical[2]['positions'][k])/5,
                           np.linalg.norm(raw_velocity[k]-previous_velocity)/5, cov]
        pos_trace, vel_trace = np.trace(kf.p[:2, :2]), np.trace(kf.p[2:, 2:])
        features[k, 6:12] = [np.log1p(pos_trace), np.log1p(vel_trace),
                             np.linalg.slogdet(kf.p[:2, :2])[1], np.linalg.norm(kf.p[:2, 2:]),
                             np.sqrt(pos_trace/vel_trace), pos_trace/vel_trace]
        cosine = np.dot(residual, previous_residual) / max(1e-8, np.linalg.norm(residual)*np.linalg.norm(previous_residual))
        features[k, 12:] = [np.log1p(kf.nis), ema, np.log1p(np.linalg.norm(residual)),
                            residual[0]/5, residual[1]/5, cosine]
        previous_residual, previous_velocity = residual, raw_velocity[k]
    inputs, mask = residual_batch(current-origin, previous-origin, np.repeat((base-origin)[None], 11, 0),
                                  np.arange(1, 12))
    with torch.inference_mode():
        x, valid = torch.from_numpy(inputs), torch.from_numpy(mask)
        candidates[:, 5] = base + original(x, valid).cpu().numpy()
        candidates[:, 6] = base + map_model(x, valid, torch.from_numpy(np.array(points)),
                                           torch.from_numpy(np.array(masks)),
                                           torch.zeros((11, 1, 1, 8)),
                                           torch.zeros((11, 1, 1), dtype=torch.bool)).cpu().numpy()
    missing = np.array(coverage) == 0
    candidates[missing, 6] = candidates[missing, 5]
    if not np.isfinite(candidates).all() or not np.isfinite(features).all():
        raise ValueError('nonfinite rebuilt noisy candidate/features')
    return {'candidates': candidates, 'features': np.clip(features, -10, 10),
            'base': base, 'map_coverage': np.array(coverage, np.float32)}
