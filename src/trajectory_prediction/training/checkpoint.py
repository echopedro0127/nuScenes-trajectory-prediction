from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from torch import nn


def save_checkpoint(
    path: str | os.PathLike[str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_ade: float,
    bad_epochs: int = 0,
) -> None:
    """Atomically save model, optimizer, progress, and CPU RNG state."""
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
        raise ValueError("epoch must be a non-negative integer")
    if isinstance(bad_epochs, bool) or not isinstance(bad_epochs, int) or bad_epochs < 0:
        raise ValueError("bad_epochs must be a non-negative integer")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=target.parent, prefix=f".{target.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "best_ade": float(best_ade),
        "bad_epochs": bad_epochs,
        "torch_rng": torch.get_rng_state().clone(),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | os.PathLike[str],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    include_patience: bool = False,
) -> dict[str, int | float]:
    """Restore a checkpoint using CPU deserialization and return progress."""
    payload: Any = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError("checkpoint payload must be a mapping")
    for key in ("model", "optimizer", "epoch", "best_ade", "torch_rng"):
        if key not in payload:
            raise ValueError(f"checkpoint missing {key}")
    if isinstance(payload["epoch"], bool) or not isinstance(payload["epoch"], int):
        raise TypeError("checkpoint epoch must be an integer")
    if not isinstance(payload["best_ade"], (int, float)) or isinstance(payload["best_ade"], bool):
        raise TypeError("checkpoint best_ade must be numeric")
    bad_epochs = payload.get("bad_epochs", 0)
    if isinstance(bad_epochs, bool) or not isinstance(bad_epochs, int) or bad_epochs < 0:
        raise ValueError("checkpoint bad_epochs must be a non-negative integer")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    state: dict[str, int | float] = {
        "epoch": int(payload["epoch"]),
        "best_ade": float(payload["best_ade"]),
    }
    if include_patience:
        state["bad_epochs"] = bad_epochs
    return state
