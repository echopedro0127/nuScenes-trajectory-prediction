import torch
from torch import nn


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _validate_inputs(
    features: torch.Tensor,
    mask: torch.Tensor,
    input_dim: int,
) -> None:
    if features.ndim != 3:
        raise ValueError("features must have shape [B, H, input_dim]")
    if features.shape[2] != input_dim:
        raise ValueError(f"features input_dim must equal {input_dim}")
    if features.shape[1] < 1:
        raise ValueError("features history dimension H >= 1 is required")
    if mask.ndim != 2:
        raise ValueError("mask must have shape [B, H]")
    if mask.shape != features.shape[:2]:
        raise ValueError("features and mask batch/history dimensions must match")
    if mask.dtype != torch.bool:
        raise ValueError("mask must be a boolean tensor")
    if mask.device != features.device:
        raise ValueError("features and mask must be on the same device")
    if not mask.any(dim=1).all():
        raise ValueError("each sample must contain at least one valid history point")


class GRUPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        future_steps: int,
    ) -> None:
        super().__init__()
        input_dim = _positive_int("input_dim", input_dim)
        hidden_dim = _positive_int("hidden_dim", hidden_dim)
        layers = _positive_int("layers", layers)
        future_steps = _positive_int("future_steps", future_steps)
        self.encoder = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=layers,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_steps * 2),
        )
        self.input_dim = input_dim
        self.future_steps = future_steps

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _validate_inputs(features, mask, self.input_dim)

        sequences = [
            sample_features[sample_mask]
            for sample_features, sample_mask in zip(features, mask, strict=True)
        ]
        lengths = torch.tensor(
            [sequence.shape[0] for sequence in sequences],
            dtype=torch.long,
        )
        compacted = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        packed = nn.utils.rnn.pack_padded_sequence(
            compacted,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.encoder(packed)
        output = self.head(hidden[-1])
        return output.view(features.shape[0], self.future_steps, 2)
