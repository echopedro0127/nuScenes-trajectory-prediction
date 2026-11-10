"""Fixed feature/expert interventions with legacy initialization and optimizer order."""
import numpy as np
import torch

from trajectory_prediction.robustness.filters import Estimator
from trajectory_prediction.robustness.packets import PROFILES, make_packets
from trajectory_prediction.robustness.reliability_learning import (
    TRAINING,
    ReliabilityGate,
    _validate,
    masked_features,
)

CONTROLS = ('full', 'motion_time', 'time_innovation', 'physical_candidates', 'no_map_candidate')


def transform_features(features, variant):
    if variant not in CONTROLS:
        raise ValueError('unknown diagnostic control')
    result = np.array(features, dtype=np.float32, copy=True)
    if result.ndim != 3 or result.shape[1:] != (11, 18) or not np.isfinite(result).all():
        raise ValueError('finite [N,11,18] causal features required')
    if variant in ('motion_time', 'time_innovation'):
        t = np.arange(1, 12, dtype=np.float32)/11
        result[:, :, 6:12] = t[:, None] ** np.arange(1, 7, dtype=np.float32)[None]
    if variant == 'motion_time':
        result[:, :, 12:] = 0
    return result


def expert_mask(variant):
    if variant not in CONTROLS:
        raise ValueError('unknown diagnostic control')
    count = 4 if variant == 'physical_candidates' else 6 if variant == 'no_map_candidate' else 7
    return np.arange(7) < count


def _weights(model, x, variant):
    mask = torch.from_numpy(expert_mask(variant))
    return model.network(x).masked_fill(~mask, -torch.inf).softmax(-1)


def fit_control(features, candidates, target, variant, *, epochs=20):
    _validate(features, candidates)
    n = len(features)
    if target.shape != (n, 12, 2) or not np.isfinite(target).all():
        raise ValueError('finite matching training target required')
    if type(epochs) is not int or epochs < 1:
        raise ValueError('positive epoch budget required')
    values = transform_features(features, variant)
    flat = values.reshape(-1, 18)
    mean, std = flat.mean(0, dtype=np.float64), np.maximum(flat.std(0, dtype=np.float64), 1e-4)
    stats = {'mean': mean.tolist(), 'std': std.tolist(), 'fit_streams': n}
    x = torch.from_numpy(masked_features(values, mean, std, 'innovation').reshape(-1, 18))
    experts = torch.from_numpy(np.asarray(candidates, np.float32).reshape(-1, 7, 12, 2))
    truth = torch.from_numpy(np.repeat(np.asarray(target, np.float32)[:, None], 11, 1).reshape(-1, 12, 2))
    masks = torch.from_numpy(np.tile(np.arange(12)[None] >= np.arange(1, 12)[:, None], (n, 1)))
    model = ReliabilityGate().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING['learning_rate'], weight_decay=TRAINING['weight_decay'])
    rng, log = np.random.default_rng(TRAINING['seed']), []
    batch_size = TRAINING['batch_size']
    for epoch in range(epochs):
        order, total = rng.permutation(len(x)), 0.
        for start in range(0, len(x), batch_size):
            i = order[start:start+batch_size]
            optimizer.zero_grad(set_to_none=True)
            weight = _weights(model, x[i], variant)
            forecast = (experts[i]*weight[..., None, None]).sum(1)
            error = torch.linalg.vector_norm(forecast-truth[i], dim=-1)
            loss = ((error*masks[i]).sum(-1)/masks[i].sum(-1)).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(i)
        log.append({'epoch': epoch+1, 'fit_ade_loss': total/len(x)})
    return model.eval(), stats, log


def predict_control(model, stats, features, candidates, variant):
    _validate(features, candidates)
    transformed = transform_features(features, variant)
    x = masked_features(transformed, stats['mean'], stats['std'], 'innovation').reshape(-1, 18)
    weights = []
    with torch.inference_mode():
        for start in range(0, len(x), 4096):
            weights.append(_weights(model, torch.from_numpy(x[start:start+4096]), variant).numpy())
    weights = np.concatenate(weights).reshape(len(features), 11, 7)
    return (candidates*weights[..., None, None]).sum(2), weights


def covariance_lookup():
    packets = make_packets(np.zeros((5, 2)), np.zeros((11, 2)), 'time_only', PROFILES['clean'], 2027)
    model = Estimator('kf_cv', packets[0], packets[1])
    for packet in packets[2:5]:
        model.receive(packet)
    table = []
    for packet in packets[5:]:
        model.receive(packet)
        pos, vel = np.trace(model.p[:2, :2]), np.trace(model.p[2:, 2:])
        table.append([np.log1p(pos), np.log1p(vel), np.linalg.slogdet(model.p[:2, :2])[1],
                      np.linalg.norm(model.p[:2, 2:]), np.sqrt(pos/vel), pos/vel])
    return np.clip(np.array(table, np.float32), -10, 10)
