import torch
from torch import nn


def _validate_inputs(
    features: torch.Tensor,
    mask: torch.Tensor,
    input_dim: int,
    max_history_steps: int,
) -> None:
    if features.ndim != 3:
        raise ValueError("features must have shape [B, H, input_dim]")
    if features.shape[2] != input_dim:
        raise ValueError(f"features input_dim must equal {input_dim}")
    if features.shape[1] < 1:
        raise ValueError("features history dimension H >= 1 is required")
    if features.shape[1] > max_history_steps:
        raise ValueError(
            f"history length cannot exceed max_history_steps={max_history_steps}"
        )
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


class TransformerPredictor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        heads: int,
        layers: int,
        future_steps: int,
        max_history_steps: int = 5,
    ) -> None:
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Parameter(
            torch.zeros(1, max_history_steps, hidden_dim)
        )
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            block,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, future_steps * 2)
        self.input_dim = input_dim
        self.future_steps = future_steps
        self.max_history_steps = max_history_steps

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        _validate_inputs(
            features,
            mask,
            self.input_dim,
            self.max_history_steps,
        )

        history_steps = features.shape[1]
        hidden = self.embedding(features) + self.position[:, :history_steps]
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)

        positions = torch.arange(history_steps, device=features.device)
        positions = positions.expand(features.shape[0], history_steps)
        last_valid_index = positions.masked_fill(~mask, -1).max(dim=1).values
        batch_index = torch.arange(features.shape[0], device=features.device)
        pooled = hidden[batch_index, last_valid_index]

        output = self.head(self.norm(pooled))
        return output.view(features.shape[0], self.future_steps, 2)
