"""Matched-width feature ablations, fit-only normalization and fixed-budget training."""
import numpy as np
import torch
from torch import nn

from trajectory_prediction.robustness.noisy_candidates import FEATURE_GROUPS

TRAINING = {'epochs': 20, 'batch_size': 1024, 'seed': 2027,
            'learning_rate': .001, 'weight_decay': .0001, 'hidden_dim': 32}


class ReliabilityGate(nn.Module):
    def __init__(self):
        super().__init__()
        with torch.random.fork_rng():
            torch.manual_seed(TRAINING['seed'])
            self.network = nn.Sequential(nn.Linear(18, 32), nn.GELU(), nn.Linear(32, 7))

    def forward(self, features):
        return self.network(features).softmax(-1)


def masked_features(features, mean, std, variant):
    if variant not in FEATURE_GROUPS:
        raise ValueError('unknown reliability feature ablation')
    x = (np.asarray(features, np.float32)-np.asarray(mean, np.float32))/np.asarray(std, np.float32)
    if x.shape[-1] != 18 or not np.isfinite(x).all():
        raise ValueError('finite standardized 18-dimensional features required')
    x = np.clip(x, -10, 10)
    x[..., FEATURE_GROUPS[variant]:] = 0
    return x


def _validate(features, candidates):
    n = len(features)
    if features.shape != (n, 11, 18) or candidates.shape != (n, 11, 7, 12, 2):
        raise ValueError('candidate/feature shapes differ')
    if n < 1 or not np.isfinite(features).all() or not np.isfinite(candidates).all():
        raise ValueError('nonempty finite candidate features required')


def fit_gate(features, candidates, target, variant, *, epochs=20, batch_size=1024):
    _validate(features, candidates)
    n = len(features)
    if target.shape != (n, 12, 2) or not np.isfinite(target).all():
        raise ValueError('finite per-trajectory training target required')
    if type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError('positive training budget required')
    flat = features.reshape(-1, 18)
    mean, std = flat.mean(0, dtype=np.float64), flat.std(0, dtype=np.float64)
    std = np.maximum(std, 1e-4)
    stats = {'mean': mean.tolist(), 'std': std.tolist(), 'fit_streams': n}
    x = torch.from_numpy(masked_features(features, mean, std, variant).reshape(-1, 18))
    experts = torch.from_numpy(np.asarray(candidates, np.float32).reshape(-1, 7, 12, 2))
    truth = torch.from_numpy(np.repeat(np.asarray(target, np.float32)[:, None], 11, 1).reshape(-1, 12, 2))
    masks = torch.from_numpy(np.tile(np.arange(12)[None] >= np.arange(1, 12)[:, None], (n, 1)))
    model = ReliabilityGate().train()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING['learning_rate'], weight_decay=TRAINING['weight_decay'])
    rng = np.random.default_rng(TRAINING['seed'])
    log = []
    for epoch in range(epochs):
        order = rng.permutation(len(x))
        total = 0.
        for start in range(0, len(x), batch_size):
            i = order[start:start+batch_size]
            optimizer.zero_grad(set_to_none=True)
            weight = model(x[i])
            forecast = (experts[i] * weight[..., None, None]).sum(1)
            error = torch.linalg.vector_norm(forecast-truth[i], dim=-1)
            loss = ((error*masks[i]).sum(-1)/masks[i].sum(-1)).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach()) * len(i)
        log.append({'epoch': epoch+1, 'fit_ade_loss': total/len(x)})
    return model.eval(), stats, log


def predict_gate(model, stats, features, candidates, variant):
    _validate(features, candidates)
    x = masked_features(features, stats['mean'], stats['std'], variant).reshape(-1, 18)
    weights = []
    with torch.inference_mode():
        for start in range(0, len(x), 4096):
            weights.append(model(torch.from_numpy(x[start:start+4096])).cpu().numpy())
    weights = np.concatenate(weights).reshape(len(features), 11, 7)
    forecast = (candidates * weights[..., None, None]).sum(2)
    return forecast, weights
