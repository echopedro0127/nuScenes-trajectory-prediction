from __future__ import annotations

import csv
import math
from collections.abc import Iterable
from pathlib import Path

import torch
from torch.nn import functional as F

from trajectory_prediction.evaluation.metrics import displacement_metrics
from trajectory_prediction.training.checkpoint import load_checkpoint, save_checkpoint


def _move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> tuple[torch.Tensor, ...]:
    return (
        batch["history_features"].to(device),
        batch["history_mask"].to(device),
        batch["future_xy"].to(device),
    )


def optimize_batch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    features: torch.Tensor,
    mask: torch.Tensor,
    target: torch.Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    prediction = model(features, mask)
    loss = F.smooth_l1_loss(prediction, target)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.detach())


def evaluate_epoch(
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, float | int]:
    was_training = model.training
    model.eval()
    loss_total = 0.0
    ade_total = 0.0
    fde_total = 0.0
    samples = 0
    try:
        with torch.no_grad():
            for batch in loader:
                features, mask, target = _move_batch(batch, device)
                prediction = model(features, mask)
                batch_size = int(target.shape[0])
                metrics = displacement_metrics(prediction, target)
                loss_total += float(F.smooth_l1_loss(prediction, target)) * batch_size
                ade_total += float(metrics["ade"].sum())
                fde_total += float(metrics["fde"].sum())
                samples += batch_size
    finally:
        model.train(was_training)

    if samples == 0:
        raise ValueError("cannot evaluate an empty loader")
    return {
        "loss": loss_total / samples,
        "ade": ade_total / samples,
        "fde": fde_total / samples,
        "samples": samples,
    }


def _train_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
) -> tuple[float, int]:
    loss_total = 0.0
    samples = 0
    for batch in loader:
        features, mask, target = _move_batch(batch, device)
        batch_size = int(target.shape[0])
        loss_total += optimize_batch(model, optimizer, features, mask, target) * batch_size
        samples += batch_size
    if samples == 0:
        raise ValueError("cannot train on an empty loader")
    return loss_total / samples, samples


def fit(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: Iterable[dict[str, torch.Tensor]],
    val_loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    output_dir: Path,
    epochs: int,
    patience: int,
    resume: bool = False,
) -> list[dict[str, float | int]]:
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
        raise ValueError("epochs must be a positive integer")
    if isinstance(patience, bool) or not isinstance(patience, int) or patience <= 0:
        raise ValueError("patience must be a positive integer")
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 1
    best_ade = math.inf
    if resume:
        last_path = output_dir / "last.pt"
        if not last_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {last_path}")
        state = load_checkpoint(last_path, model, optimizer)
        start_epoch = int(state["epoch"]) + 1
        best_ade = float(state["best_ade"])
    if start_epoch > epochs:
        return []

    metrics_path = output_dir / "metrics.csv"
    fieldnames = ["epoch", "train_loss", "val_loss", "val_ade", "val_fde", "learning_rate"]
    write_header = not metrics_path.exists() or not resume
    history: list[dict[str, float | int]] = []
    bad_epochs = 0
    with metrics_path.open("a" if resume and metrics_path.exists() else "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for epoch in range(start_epoch, epochs + 1):
            train_loss, _ = _train_epoch(model, optimizer, train_loader, device)
            validation = evaluate_epoch(model, val_loader, device)
            row: dict[str, float | int] = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": float(validation["loss"]),
                "val_ade": float(validation["ade"]),
                "val_fde": float(validation["fde"]),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
            writer.writerow(row)
            stream.flush()
            history.append(row)
            save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, best_ade)
            if row["val_ade"] < best_ade:
                best_ade = float(row["val_ade"])
                save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_ade)
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break
    return history


__all__ = ["evaluate_epoch", "fit", "optimize_batch"]
