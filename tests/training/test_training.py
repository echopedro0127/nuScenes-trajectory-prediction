from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from trajectory_prediction.models.gru import GRUPredictor
from trajectory_prediction.training.checkpoint import load_checkpoint, save_checkpoint
from trajectory_prediction.training.dataset import TrajectoryDataset
from trajectory_prediction.training.engine import evaluate_epoch, fit, optimize_batch
from trajectory_prediction.utils.seed import seed_everything


def _arrays(samples: int = 4, history: int = 5) -> dict[str, np.ndarray]:
    return {
        "history_features": np.arange(samples * history * 6, dtype=np.float32).reshape(samples, history, 6),
        "history_xy": np.arange(samples * history * 2, dtype=np.float32).reshape(samples, history, 2),
        "history_mask": np.ones((samples, history), dtype=np.bool_),
        "future_xy": np.ones((samples, 12, 2), dtype=np.float32),
        "current_speed": np.arange(samples, dtype=np.float32),
    }


def _batch(target: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.zeros(8, 5, 6)
    mask = torch.ones(8, 5, dtype=torch.bool)
    if target is None:
        target = torch.ones(8, 12, 2)
    return features, mask, target


def test_dataset_slices_last_history_steps_and_returns_tensors() -> None:
    dataset = TrajectoryDataset(_arrays(), history_steps=3)

    sample = dataset[0]

    assert len(dataset) == 4
    assert set(sample) == {
        "history_features",
        "history_xy",
        "history_mask",
        "future_xy",
        "current_speed",
        "index",
    }
    assert sample["history_features"].shape == (3, 6)
    torch.testing.assert_close(sample["history_features"], torch.tensor(_arrays()["history_features"][0, -3:]))
    assert sample["history_xy"].dtype == torch.float32
    assert sample["history_mask"].dtype == torch.bool
    assert sample["current_speed"].shape == ()
    assert sample["index"].dtype == torch.long
    assert sample["index"].item() == 0


@pytest.mark.parametrize("history_steps", [0, -1, 6])
def test_dataset_rejects_invalid_history_steps(history_steps: int) -> None:
    with pytest.raises(ValueError, match="history_steps"):
        TrajectoryDataset(_arrays(), history_steps=history_steps)


def test_seed_everything_repeats_python_numpy_and_torch_streams() -> None:
    seed_everything(123)
    first = (random.random(), np.random.rand(), torch.rand(3))
    seed_everything(123)
    second = (random.random(), np.random.rand(), torch.rand(3))

    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2])


def test_checkpoint_roundtrip_restores_model_optimizer_and_rng(tmp_path: Path) -> None:
    model = GRUPredictor(6, 16, 1, 12)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    features, mask, target = _batch()
    optimize_batch(model, optimizer, features, mask, target)
    expected_model = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_rng = torch.get_rng_state().clone()
    path = tmp_path / "nested" / "checkpoint.pt"

    save_checkpoint(path, model, optimizer, epoch=9, best_ade=1.25)
    model.load_state_dict({name: torch.zeros_like(value) for name, value in model.state_dict().items()})
    restored_optimizer = torch.optim.Adam(model.parameters(), lr=0.5)
    torch.manual_seed(999)
    result = load_checkpoint(path, model, restored_optimizer)

    assert result == {"epoch": 9, "best_ade": 1.25}
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected_model[name])
    torch.testing.assert_close(torch.get_rng_state(), expected_rng)
    assert restored_optimizer.param_groups[0]["lr"] == 1e-3


def test_checkpoint_atomic_failure_preserves_previous_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model = GRUPredictor(6, 16, 1, 12)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, model, optimizer, epoch=1, best_ade=3.0)
    original = path.read_bytes()

    def fail_replace(*args: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr("trajectory_prediction.training.checkpoint.os.replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        save_checkpoint(path, model, optimizer, epoch=2, best_ade=2.0)
    assert path.read_bytes() == original


def test_optimize_batch_overfits_tiny_batch() -> None:
    torch.manual_seed(0)
    model = GRUPredictor(6, 16, 1, 12)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    features, mask, target = _batch()

    initial = optimize_batch(model, optimizer, features, mask, target)
    for _ in range(99):
        final = optimize_batch(model, optimizer, features, mask, target)

    assert final <= initial * 0.2


def test_evaluate_epoch_aggregates_metrics_and_restores_mode() -> None:
    model = GRUPredictor(6, 16, 1, 12)
    loader = DataLoader(TrajectoryDataset(_arrays(3)), batch_size=2)
    model.train()

    metrics = evaluate_epoch(model, loader, torch.device("cpu"))

    assert set(metrics) == {"loss", "ade", "fde", "samples"}
    assert metrics["samples"] == 3
    assert model.training


def test_fit_writes_history_checkpoints_and_resumes(tmp_path: Path) -> None:
    seed_everything(7)
    model = GRUPredictor(6, 8, 1, 12)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = TrajectoryDataset(_arrays(4))
    loader = DataLoader(dataset, batch_size=2)

    history = fit(
        model,
        optimizer,
        loader,
        loader,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        epochs=2,
        patience=4,
    )

    assert len(history) == 2
    assert (tmp_path / "metrics.csv").exists()
    assert (tmp_path / "last.pt").exists()
    assert (tmp_path / "best.pt").exists()
    assert history[-1]["epoch"] == 2

    resumed_model = GRUPredictor(6, 8, 1, 12)
    resumed_optimizer = torch.optim.Adam(resumed_model.parameters(), lr=1e-3)
    resumed = fit(
        resumed_model,
        resumed_optimizer,
        loader,
        loader,
        device=torch.device("cpu"),
        output_dir=tmp_path,
        epochs=3,
        patience=4,
        resume=True,
    )

    assert [row["epoch"] for row in resumed] == [3]
    assert resumed[-1]["epoch"] == 3


def test_empty_loader_has_clear_error() -> None:
    model = GRUPredictor(6, 16, 1, 12)
    with pytest.raises(ValueError, match="empty"):
        evaluate_epoch(model, [], torch.device("cpu"))
