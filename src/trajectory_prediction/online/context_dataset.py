"""Training labels stay separate from explicit model argument dictionaries."""
import numpy as np
import torch
from torch.utils.data import Dataset

from trajectory_prediction.online.context_features import residual_features
from trajectory_prediction.online.observations import replay_observations


class ContextDataset(Dataset):
    def __init__(self, arrays, context, base, indices):
        self.examples = []
        for i in indices:
            for k, (observation, label) in enumerate(replay_observations(
                arrays["history_xy"][i], arrays["future_xy"][i],
                anchor_timestamp_us=int(context["anchor_timestamp_us"][i])
            )):
                features, mask = residual_features(observation, base[i])
                target = np.zeros((12, 2), np.float32)
                step = observation.update_step
                target[step:] = label.remaining_target - base[i, step:]
                self.examples.append({"features": torch.from_numpy(features), "mask": torch.from_numpy(mask),
                    "map_features": torch.from_numpy(context["features"][i, k].copy()),
                    "map_mask": torch.from_numpy(context["mask"][i, k].copy()),
                    "target": torch.from_numpy(target)})
        if not self.examples:
            raise ValueError("training partition has no examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]
