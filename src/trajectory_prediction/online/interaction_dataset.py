"""P2 residual/label separation extended with causal neighbor inputs."""
import torch

from trajectory_prediction.context.neighbor_cache import neighbor_view
from trajectory_prediction.online.context_dataset import ContextDataset
from trajectory_prediction.online.interaction_corrector import neighbor_features


class InteractionDataset(ContextDataset):
    def __init__(self, arrays, maps, neighbors, base, indices):
        super().__init__(arrays, maps, base, indices)
        position = 0
        for i in indices:
            for k in range(11):
                view = neighbor_view(neighbors, i, k)
                self.examples[position].update(neighbor_features=torch.from_numpy(neighbor_features(view)),
                                                neighbor_mask=torch.from_numpy(view.mask.copy()))
                position += 1
