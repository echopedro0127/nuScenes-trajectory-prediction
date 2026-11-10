"""Matched motion-head controls; the seven-head baseline keeps legacy arithmetic."""
import numpy as np
import torch
from torch import nn

from trajectory_prediction.robustness.reliability_learning import (
    TRAINING,
    ReliabilityGate,
    masked_features,
)

CONTROLS = ('baseline7', 'plus_ca', 'plus_stop', 'all_motion')


def active_experts(variant):
    if variant == 'guarded_ca':
        variant = 'plus_ca'
    if variant not in CONTROLS:
        raise ValueError('unknown motion control')
    mask = np.ones(10, dtype=bool)
    mask[7:] = False
    if variant in ('plus_ca', 'all_motion'):
        mask[7] = True
    if variant in ('plus_stop', 'all_motion'):
        mask[8:] = True
    return mask


class MotionGate(nn.Module):
    def __init__(self, seed=2027):
        super().__init__()
        if type(seed) is not int or seed < 0:
            raise ValueError('nonnegative model seed required')
        self.legacy = ReliabilityGate()
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            if seed != TRAINING['seed']:
                for module in self.legacy.modules():
                    if isinstance(module, nn.Linear):
                        module.reset_parameters()
            self.extra = nn.Linear(32, 3)
            nn.init.zeros_(self.extra.weight)
            nn.init.constant_(self.extra.bias, -2.)

    def weights(self, features, variant, ca_allowed=None):
        if variant == 'baseline7':
            return self.legacy(features)
        hidden = self.legacy.network[1](self.legacy.network[0](features))
        logits = torch.cat((self.legacy.network[2](hidden), self.extra(hidden)), -1)
        selected = logits[:, torch.from_numpy(active_experts(variant))]
        if variant == 'guarded_ca':
            if ca_allowed is None:
                raise ValueError('causal guard mask required')
            allowed = torch.ones_like(selected, dtype=torch.bool)
            allowed[:, -1] = ca_allowed
            selected = selected.masked_fill(~allowed, -torch.inf)
        return selected.softmax(-1)


def _validate(features, candidates):
    n = len(features)
    if (n < 1 or features.shape != (n, 11, 18) or candidates.shape != (n, 11, 10, 12, 2)
            or not np.isfinite(features).all() or not np.isfinite(candidates).all()):
        raise ValueError('finite matching ten-expert feature/candidate arrays required')


def _guard_trace(ca_allowed, n, variant):
    if variant != 'guarded_ca':
        return None
    if ca_allowed is None or np.asarray(ca_allowed).shape != (n, 11) or np.asarray(ca_allowed).dtype != np.bool_:
        raise ValueError('boolean [N,11] causal guard trace required')
    return torch.from_numpy(np.ascontiguousarray(ca_allowed).reshape(-1))


def fit_motion(features, candidates, target, variant, *, epochs=20, batch_size=1024, seed=2027, ca_allowed=None):
    _validate(features, candidates)
    n = len(features)
    if target.shape != (n, 12, 2) or not np.isfinite(target).all():
        raise ValueError('finite matching training target required')
    if type(epochs) is not int or epochs < 1 or type(batch_size) is not int or batch_size < 1:
        raise ValueError('positive training budget required')
    active = active_experts(variant)
    guard = _guard_trace(ca_allowed, n, variant)
    flat = features.reshape(-1, 18)
    mean, std = flat.mean(0, dtype=np.float64), np.maximum(flat.std(0, dtype=np.float64), 1e-4)
    stats = {'mean': mean.tolist(), 'std': std.tolist(), 'fit_streams': n}
    x = torch.from_numpy(masked_features(features, mean, std, 'innovation').reshape(-1, 18))
    experts = torch.from_numpy(np.ascontiguousarray(candidates[:, :, active], dtype=np.float32).reshape(-1, int(active.sum()), 12, 2))
    truth = torch.from_numpy(np.repeat(np.asarray(target, np.float32)[:, None], 11, 1).reshape(-1, 12, 2))
    masks = torch.from_numpy(np.tile(np.arange(12)[None] >= np.arange(1, 12)[:, None], (n, 1)))
    model = MotionGate(seed=seed).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=TRAINING['learning_rate'], weight_decay=TRAINING['weight_decay'])
    rng, log = np.random.default_rng(seed), []
    for epoch in range(epochs):
        order, total = rng.permutation(len(x)), 0.
        for start in range(0, len(x), batch_size):
            i = order[start:start+batch_size]
            optimizer.zero_grad(set_to_none=True)
            weight = model.weights(x[i], variant, None if guard is None else guard[i])
            forecast = (experts[i]*weight[..., None, None]).sum(1)
            error = torch.linalg.vector_norm(forecast-truth[i], dim=-1)
            loss = ((error*masks[i]).sum(-1)/masks[i].sum(-1)).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())*len(i)
        log.append({'epoch': epoch+1, 'fit_ade_loss': total/len(x)})
    return model.eval(), stats, log


def predict_motion(model, stats, features, candidates, variant, *, ca_allowed=None):
    _validate(features, candidates)
    active = active_experts(variant)
    guard = _guard_trace(ca_allowed, len(features), variant)
    x = masked_features(features, stats['mean'], stats['std'], 'innovation').reshape(-1, 18)
    parts = []
    with torch.inference_mode():
        for start in range(0, len(x), 4096):
            parts.append(model.weights(torch.from_numpy(x[start:start+4096]), variant,
                                       None if guard is None else guard[start:start+4096]).numpy())
    weights = np.concatenate(parts).reshape(len(features), 11, int(active.sum()))
    full = np.zeros((len(features), 11, 10), np.float32)
    full[:, :, active] = weights
    return (candidates[:, :, active]*weights[..., None, None]).sum(2), full
