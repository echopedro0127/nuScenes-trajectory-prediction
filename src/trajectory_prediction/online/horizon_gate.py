"""Shared lead-time gate and full-sequence metrics with exact CV fallback."""

import numpy as np
import torch
from torch import nn

from trajectory_prediction.online.batched_evaluation import forecast_metrics
from trajectory_prediction.online.quality_gate import QualityGate

VARIANTS = ("B", "D", "R", "H", "DR", "DH", "RH", "DRH")


def feature_mask(variant):
    if variant not in VARIANTS:
        raise ValueError("unknown horizon variant")
    result = torch.ones(19)
    for name, start, stop in (("D", 8, 12), ("R", 12, 16), ("H", 16, 19)):
        if name not in variant:
            result[start:stop] = 0
    return result


class HorizonGate(nn.Module):
    def __init__(self, variant, initial=None):
        super().__init__()
        self.variant = variant
        self.register_buffer("feature_mask", feature_mask(variant))
        self.network = nn.Sequential(nn.Linear(19, 16), nn.GELU(), nn.Linear(16, 1))
        initial = QualityGate().state_dict() if initial is None else initial
        expected = QualityGate().state_dict()
        if set(initial) != set(expected) or any(initial[k].shape != v.shape for k, v in expected.items()):
            raise ValueError("invalid P6 horizon initializer")
        with torch.no_grad():
            self.network[0].weight.zero_()
            self.network[0].weight[:, :8].copy_(initial["network.0.weight"])
            self.network[0].bias.copy_(initial["network.0.bias"])
            self.network[2].weight.copy_(initial["network.2.weight"])
            self.network[2].bias.copy_(initial["network.2.bias"])

    def forward(self, features, valid, mask):
        if (features.ndim != 3 or features.shape[1:] != (12, 19)
            or valid.shape != (len(features),) or valid.dtype != torch.bool
            or mask.shape != features.shape[:2] or mask.dtype != torch.bool
            or len({x.device for x in (features, valid, mask, self.feature_mask)}) != 1
            or not torch.isfinite(features).all()):
            raise ValueError("invalid horizon gate features/mask")
        weights = torch.sigmoid(self.network(features*self.feature_mask).squeeze(-1))
        return weights * (valid[:, None] & mask).to(weights.dtype)


def blend_horizon(cv, candidate, weights):
    if (cv.ndim != 3 or cv.shape[1:] != (12, 2) or candidate.shape != cv.shape
        or weights.shape != cv.shape[:2] or len({x.device for x in (cv, candidate, weights)}) != 1
        or not all(torch.isfinite(x).all() for x in (cv, candidate, weights))
        or ((weights < 0) | (weights > 1)).any()):
        raise ValueError("invalid horizon forecast mixture")
    weight = weights[..., None]
    mixed = cv + weight*(candidate-cv)
    return torch.where(weight == 0, cv, torch.where(weight == 1, candidate, mixed))


def measure_horizon(data, model, *, batch_size=4096, device=None):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive horizon evaluation batch required")
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    metrics = np.empty((6, data.n, 11, 3), np.float64)
    weights = np.empty((6, data.n, 11, 12), np.float32)
    for c in range(6):
        forecast = np.empty((data.n, 11, 12, 2), np.float32)
        for start in range(0, data.n*11, batch_size):
            local = np.arange(start, min(start+batch_size, data.n*11))
            values = data.inputs(c*data.n*11+local, device)
            with torch.inference_mode():
                weight = model(values["features"], values["valid"], values["mask"])
                predicted = blend_horizon(values["cv"], values["map"], weight)
            i, k = local//11, local%11
            forecast[i, k], weights[c, i, k] = predicted.cpu().numpy(), weight.cpu().numpy()
        invalid = ~data.arrays["valid"][c]
        if np.any(weights[c][invalid] != 0) or not np.array_equal(forecast[invalid], data.arrays["cv"][invalid]):
            raise ValueError("horizon hard fallback mismatch")
        metrics[c] = forecast_metrics(forecast, data.arrays["target"])
    return metrics, weights
