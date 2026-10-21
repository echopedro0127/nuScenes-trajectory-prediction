"""Read-only causal direction and realized expert-error features for P6.3."""

import hashlib

import numpy as np
import torch

from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.online.gate_refinement import RefinementDataset

FEATURE_PROTOCOL = {"speed_threshold": .5, "ema_decay": .8, "acceleration_scale": 10.,
    "correction_scale": 20., "horizon_scale": 6., "projection_clip": 5., "feedback_clip": 5.}


class HorizonDataset(RefinementDataset):
    def __init__(self, arrays):
        super().__init__(arrays)
        self.heading = np.zeros((self.n, 11, 2), np.float32)
        latest = np.zeros((self.n, 2), np.float32)
        initial = np.diff(arrays["history"], axis=1) / np.float32(.5)
        for velocity in [*np.moveaxis(initial, 1, 0), *np.moveaxis(arrays["velocity"], 1, 0)]:
            if not np.isfinite(velocity).all():
                raise ValueError("nonfinite observed velocity")
        for k in range(initial.shape[1]):
            velocity = initial[:, k]
            speed = np.linalg.norm(velocity, axis=-1)
            good = speed >= FEATURE_PROTOCOL["speed_threshold"]
            latest[good] = velocity[good] / speed[good, None]
        for k in range(11):
            velocity = arrays["velocity"][:, k]
            speed = np.linalg.norm(velocity, axis=-1)
            good = speed >= FEATURE_PROTOCOL["speed_threshold"]
            latest[good] = velocity[good] / speed[good, None]
            self.heading[:, k] = latest
        lateral = np.stack((-self.heading[..., 1], self.heading[..., 0]), -1)
        acceleration = (arrays["velocity"]-arrays["previous_velocity"])/np.float32(.5)
        motion = np.stack(((acceleration*self.heading).sum(-1), (acceleration*lateral).sum(-1)), -1) / 10.
        self.direction = np.empty((6, self.n, 11, 4), np.float32)
        mask = np.arange(12)[None] >= np.arange(1, 12)[:, None]
        for c in range(6):
            delta = arrays["map"][c]-arrays["cv"]
            mean = (delta*mask[None, :, :, None]).sum(2, dtype=np.float64)/mask.sum(1)[None, :, None]
            projected = np.stack(((mean*self.heading).sum(-1), (mean*lateral).sum(-1)), -1)/20.
            self.direction[c] = np.clip(np.concatenate((motion, projected), -1), -5, 5)
        self.feedback = np.zeros((6, self.n, 11, 4), np.float32)
        cv_error = np.zeros(self.n, np.float64)
        map_error = np.zeros((6, self.n), np.float64)
        map_has = np.zeros((6, self.n), bool)
        for k in range(1, 11):
            observed = arrays["position"][:, k]
            error = np.linalg.norm(arrays["cv"][:, k-1, k]-observed, axis=-1)
            cv_error = error if k == 1 else .8*cv_error+.2*error
            for c in range(6):
                active = arrays["valid"][c, :, k-1] & arrays["valid"][c, :, k]
                error = np.linalg.norm(arrays["map"][c, :, k-1, k]-observed, axis=-1)
                fresh = active & ~map_has[c]
                old = active & map_has[c]
                map_error[c, fresh] = error[fresh]
                map_error[c, old] = .8*map_error[c, old]+.2*error[old]
                map_has[c, active] = True
                invalid = ~arrays["valid"][c, :, k]
                map_has[c, invalid], map_error[c, invalid] = False, 0
                self.feedback[c, :, k] = np.column_stack((np.clip(np.log1p(cv_error), 0, 5),
                    np.clip(np.log1p(map_error[c]), 0, 5), np.ones(self.n), map_has[c]))
        if not all(np.isfinite(value).all() for value in (self.heading, self.direction, self.feedback)):
            raise ValueError("nonfinite causal horizon features")

    def fingerprint(self):
        def checksum(value):
            return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        return digest_json({"parent": super().fingerprint(), "protocol": FEATURE_PROTOCOL,
            "heading": checksum(self.heading), "direction": checksum(self.direction), "feedback": checksum(self.feedback)})

    def inputs(self, indices, device):
        values = super().inputs(indices, device)
        c, i, k = self.indices(indices)
        features = np.empty((len(i), 12, 19), np.float32)
        features[..., :8] = self.arrays["quality"][c, i, k, None, :]
        features[..., 8:12] = self.direction[c, i, k, None, :]
        features[..., 12:16] = self.feedback[c, i, k, None, :]
        delta = self.arrays["map"][c, i, k]-self.arrays["cv"][i, k]
        heading = self.heading[i, k]
        lateral = np.stack((-heading[:, 1], heading[:, 0]), -1)
        features[..., 16] = np.maximum((np.arange(12)[None]-k[:, None])*.5, 0)/6.
        features[..., 17] = np.clip((delta*heading[:, None]).sum(-1)/20., -5, 5)
        features[..., 18] = np.clip((delta*lateral[:, None]).sum(-1)/20., -5, 5)
        values["features"] = torch.from_numpy(features).to(device)
        return values
