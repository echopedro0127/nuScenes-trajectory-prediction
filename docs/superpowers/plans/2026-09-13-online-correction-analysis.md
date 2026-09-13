# Online Correction and Analysis Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add maneuver-conditioned failure analysis and a causally valid online trajectory-correction benchmark without mixing its results with the official static protocol.

**Architecture:** Analysis consumes static prediction artifacts rather than model internals. A streaming replay reveals observations one step at a time; frozen, reforecast and learned correction strategies share a causal interface.

**Tech Stack:** Python 3.12, PyTorch, NumPy, pandas, Matplotlib, pytest.

---

## Locked file map

- `src/trajectory_prediction/analysis/` owns maneuver, failure and visualization logic.
- `src/trajectory_prediction/online/` owns causal replay, correction and online evaluation.
- `scripts/analyze_maneuvers.py`, `render_cases.py`, `train_corrector.py` and `evaluate_online.py` are thin CLIs.

### Task 1: Compute motion statistics and maneuver tags

**Files:**
- Create: `src/trajectory_prediction/analysis/__init__.py`
- Create: `src/trajectory_prediction/analysis/maneuvers.py`
- Create: `configs/analysis/maneuvers.yaml`
- Create: `tests/analysis/test_maneuvers.py`

- [ ] **Step 1: Write failing maneuver tests**

~~~python
# tests/analysis/test_maneuvers.py
import numpy as np

from trajectory_prediction.analysis.maneuvers import describe_motion, tag_maneuvers


def test_stationary_trajectory_is_low_speed_and_straight() -> None:
    future = np.zeros((12, 2), dtype=np.float32)
    stats = describe_motion(future, dt=0.5)
    tags = tag_maneuvers(stats, {
        low_speed_mps: 2.0, turn_yaw_deg: 15.0,
        high_curvature_yaw_deg: 45.0, acceleration_mps2: 1.0,
    })
    assert {low_speed, straight} <= set(tags)


def test_quarter_turn_is_high_curvature() -> None:
    angles = np.linspace(0, np.pi / 2, 12)
    future = np.stack([10 * np.sin(angles), 10 * (1 - np.cos(angles))], axis=1)
    stats = describe_motion(future.astype(np.float32), dt=0.5)
    assert abs(stats[yaw_change_deg]) >= 45.0
def tag_maneuvers(stats: dict[str, float], thresholds: dict) -> list[str]:
    tags: list[str] = []
    yaw = abs(stats[yaw_change_deg])
    if stats[mean_speed_mps] < thresholds[low_speed_mps]:
        tags.append(low_speed)
    if yaw >= thresholds[high_curvature_yaw_deg]:
        tags.append(high_curvature)
    elif yaw >= thresholds[turn_yaw_deg]:
        tags.append(turn)
    else:
        tags.append(straight)
    accel = stats[mean_acceleration_mps2]
    if accel >= thresholds[acceleration_mps2]:
        tags.append(acceleration)
    elif accel <= -thresholds[acceleration_mps2]:
        tags.append(deceleration)
    return tags
~~~

- [ ] **Step 2: Run and verify failure**

Run: `.\.venv\Scripts\python -m pytest tests/analysis/test_maneuvers.py -v`

Expected: import fails for `analysis.maneuvers`.

- [ ] **Step 3: Implement motion statistics and tags**

~~~python
# src/trajectory_prediction/analysis/maneuvers.py
import numpy as np


def _wrap(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + np.pi) % (2 * np.pi) - np.pi


def describe_motion(future_xy: np.ndarray, dt: float) -> dict[str, float]:
    points = np.concatenate([np.zeros((1, 2), np.float32), future_xy], axis=0)
    delta = np.diff(points, axis=0)
    speed = np.linalg.norm(delta, axis=1) / dt
    valid = speed > 0.2
    heading_delta = _wrap(np.diff(np.arctan2(delta[:, 1], delta[:, 0])))
    valid_turn = valid[1:] & valid[:-1]
    yaw_change = float(heading_delta[valid_turn].sum()) if valid_turn.any() else 0.0
    acceleration = np.diff(speed) / dt
    return {
        "mean_speed_mps": float(speed.mean()),
        "max_speed_mps": float(speed.max(initial=0.0)),
        "mean_acceleration_mps2": float(acceleration.mean()) if len(acceleration) else 0.0,
        "yaw_change_deg": float(np.degrees(yaw_change)),
        "displacement_m": float(np.linalg.norm(future_xy[-1])),
    }


def tag_maneuvers(stats: dict[str, float], thresholds: dict) -> list[str]:
    tags: list[str] = []
    yaw = abs(stats["yaw_change_deg"])
    if stats["mean_speed_mps"] < thresholds["low_speed_mps"]:
        tags.append("low_speed")
    if yaw >= thresholds["high_curvature_yaw_deg"]:
        tags.append("high_curvature")
    elif yaw >= thresholds["turn_yaw_deg"]:
        tags.append("turn")
    else:
        tags.append("straight")
    accel = stats["mean_acceleration_mps2"]
    if accel >= thresholds["acceleration_mps2"]:
        tags.append("acceleration")
    elif accel <= -thresholds["acceleration_mps2"]:
        tags.append("deceleration")
    return tags
~~~

~~~yaml
# configs/analysis/maneuvers.yaml
low_speed_mps: 2.0
turn_yaw_deg: 15.0
high_curvature_yaw_deg: 45.0
acceleration_mps2: 1.0
~~~

Create empty `src/trajectory_prediction/analysis/__init__.py`.

- [ ] **Step 4: Test and commit**

Run: `.\.venv\Scripts\python -m pytest tests/analysis/test_maneuvers.py -v`

Expected: both tests pass.

~~~powershell
git add configs/analysis src/trajectory_prediction/analysis tests/analysis
git commit -m "feat: add interpretable maneuver grouping"
~~~

### Task 2: Export failure records and maneuver tables

**Files:**
- Create: `src/trajectory_prediction/analysis/failures.py`
- Create: `scripts/analyze_maneuvers.py`
- Create: `tests/analysis/test_failures.py`

- [ ] **Step 1: Write the selection test**

~~~python
# tests/analysis/test_failures.py
import pandas as pd

from trajectory_prediction.analysis.failures import select_worst_cases


def test_worst_case_selection_deduplicates_rankings() -> None:
    frame = pd.DataFrame({
        "instance_token": ["a", "b", "c"], "sample_token": ["1", "2", "3"],
        "ade": [9.0, 8.0, 1.0], "fde": [2.0, 10.0, 1.0],
    })
    selected = select_worst_cases(frame, top_k=1)
    assert set(selected["instance_token"]) == {"a", "b"}
~~~

- [ ] **Step 2: Run and verify failure**

Run: `.\.venv\Scripts\python -m pytest tests/analysis/test_failures.py -v`

Expected: import fails for `analysis.failures`.

- [ ] **Step 3: Implement selection**

~~~python
# src/trajectory_prediction/analysis/failures.py
import pandas as pd


def select_worst_cases(frame: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    identity = ["instance_token", "sample_token"]
    ade = frame.sort_values(["ade", *identity], ascending=[False, True, True]).head(top_k)
    fde = frame.sort_values(["fde", *identity], ascending=[False, True, True]).head(top_k)
    return (
        pd.concat([ade.assign(rank_source="ade"), fde.assign(rank_source="fde")])
        .drop_duplicates(identity).reset_index(drop=True)
    )
~~~

- [ ] **Step 4: Add the analysis CLI and commit**

The CLI joins cache and prediction artifacts by tokens, calls `describe_motion`, explodes overlapping tags, aggregates sample count/ADE/FDE, and writes `per_sample_analysis.csv`, `maneuver_metrics.csv` and `worst_cases.json`. JSON values are converted with `.tolist()` and `float`.

Run:

~~~powershell
.\.venv\Scripts\python -m pytest tests/analysis -v
.\.venv\Scripts\ruff check src/trajectory_prediction/analysis scripts/analyze_maneuvers.py
git add src/trajectory_prediction/analysis scripts/analyze_maneuvers.py tests/analysis
git commit -m "feat: export maneuver metrics and worst cases"
~~~

Expected: tests and lint pass; the commit contains no generated outputs.

### Task 3: Render clean single-case BEV figures

**Files:**
- Create: `src/trajectory_prediction/analysis/visualization.py`
- Create: `scripts/render_cases.py`
- Create: `tests/analysis/test_visualization.py`

- [ ] **Step 1: Write a headless rendering test**

~~~python
# tests/analysis/test_visualization.py
from pathlib import Path
import numpy as np
from trajectory_prediction.analysis.visualization import render_case


def test_render_case_creates_nonempty_png(tmp_path: Path) -> None:
    output = tmp_path / "case.png"
    render_case(
        np.array([[0, -1], [0, 0]], np.float32),
        np.array([[0, 1], [0, 2]], np.float32),
        {"CV": np.array([[0, 1], [0, 2]], np.float32)}, "synthetic", output,
    )
    assert output.stat().st_size > 1_000
~~~

- [ ] **Step 2: Run and verify failure**

Run: `.\.venv\Scripts\python -m pytest tests/analysis/test_visualization.py -v`

Expected: import fails for `analysis.visualization`.

- [ ] **Step 3: Implement one-case rendering**

~~~python
# src/trajectory_prediction/analysis/visualization.py
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render_case(history: np.ndarray, target: np.ndarray,
                predictions: dict[str, np.ndarray], title: str,
                output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    axis.plot(history[:, 0], history[:, 1], "o-", label="History")
    axis.plot(target[:, 0], target[:, 1], "o-", label="Ground truth")
    for name, prediction in predictions.items():
        axis.plot(prediction[:, 0], prediction[:, 1], "--", label=name)
    axis.scatter([0], [0], marker="*", s=120, color="black", label="Current")
    axis.set(aspect="equal", xlabel="Agent-frame x [m]", ylabel="Agent-frame y [m]", title=title)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
~~~

- [ ] **Step 4: Add CLI, test and commit**

`render_cases.py` reads worst-case JSON, supports `--limit`, `--model`, `--best` and `--output-dir`, and uses token-derived filenames.

~~~powershell
.\.venv\Scripts\python -m pytest tests/analysis/test_visualization.py -v
git add src/trajectory_prediction/analysis/visualization.py scripts/render_cases.py tests/analysis/test_visualization.py
git commit -m "feat: render interview-ready BEV cases"
~~~

Expected: the PNG test passes.

### Task 4: Implement causal streaming replay

**Files:**
- Create: `src/trajectory_prediction/online/__init__.py`
- Create: `src/trajectory_prediction/online/replay.py`
- Create: `tests/online/test_replay.py`

- [ ] **Step 1: Write the causality test**

~~~python
# tests/online/test_replay.py
import numpy as np
from trajectory_prediction.online.replay import replay_updates


def test_update_never_exposes_later_observations() -> None:
    history = np.arange(10, dtype=np.float32).reshape(5, 2)
    future = np.arange(24, dtype=np.float32).reshape(12, 2) + 100
    first = list(replay_updates(history, future, history_steps=5))[0]
    np.testing.assert_array_equal(first.observed_history[-1], future[0])
    assert first.remaining_target.shape == (11, 2)
    assert not np.isin(future[1:].ravel(), first.observed_history.ravel()).any()
~~~

- [ ] **Step 2: Run and verify failure**

Run: `.\.venv\Scripts\python -m pytest tests/online/test_replay.py -v`

Expected: import fails for `online.replay`.

- [ ] **Step 3: Implement immutable updates**

~~~python
# src/trajectory_prediction/online/replay.py
from collections.abc import Iterator
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ReplayUpdate:
    update_step: int
    observed_history: np.ndarray
    observed_position: np.ndarray
    remaining_target: np.ndarray


def replay_updates(initial_history: np.ndarray, future: np.ndarray,
                   history_steps: int = 5) -> Iterator[ReplayUpdate]:
    observed = np.asarray(initial_history, dtype=np.float32).copy()
    target = np.asarray(future, dtype=np.float32)
    for step in range(len(target) - 1):
        observed = np.concatenate([observed, target[step:step + 1]], axis=0)
        yield ReplayUpdate(step + 1, observed[-history_steps:].copy(),
                           target[step].copy(), target[step + 1:].copy())
~~~

- [ ] **Step 4: Test and commit**

~~~powershell
.\.venv\Scripts\python -m pytest tests/online/test_replay.py -v
git add src/trajectory_prediction/online tests/online
git commit -m "feat: add causal trajectory replay"
~~~

Expected: causality test passes.

### Task 5: Implement the residual correction head

**Files:**
- Create: `src/trajectory_prediction/online/corrector.py`
- Create: `src/trajectory_prediction/online/dataset.py`
- Create: `configs/models/corrector.yaml`
- Create: `scripts/train_corrector.py`
- Create: `tests/online/test_corrector.py`

- [ ] **Step 1: Write correction tests**

~~~python
# tests/online/test_corrector.py
import torch
from trajectory_prediction.online.corrector import ResidualCorrector, smoothness_loss


def test_corrector_masks_elapsed_steps() -> None:
    model = ResidualCorrector(8, 64)
    features = torch.randn(3, 12, 8)
    mask = torch.tensor([[0, 0] + [1] * 10] * 3).bool()
    offsets = model(features, mask)
    assert offsets.shape == (3, 12, 2)
    torch.testing.assert_close(offsets[:, :2], torch.zeros_like(offsets[:, :2]))


def test_constant_offset_has_zero_smoothness() -> None:
    assert smoothness_loss(torch.ones(2, 12, 2), torch.ones(2, 12).bool()) == 0
~~~

- [ ] **Step 2: Run and verify failure**

Run: `.\.venv\Scripts\python -m pytest tests/online/test_corrector.py -v`

Expected: import fails for `online.corrector`.

- [ ] **Step 3: Implement the corrector**

~~~python
# src/trajectory_prediction/online/corrector.py
import torch
from torch import nn


class ResidualCorrector(nn.Module):
    def __init__(self, feature_dim: int = 8, hidden_dim: int = 64) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                     nn.Linear(hidden_dim, 2))

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.network(features) * mask.unsqueeze(-1)


def smoothness_loss(offsets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    delta = offsets[:, 1:] - offsets[:, :-1]
    pair_mask = mask[:, 1:] & mask[:, :-1]
    return delta.square().sum(-1)[pair_mask].mean() if pair_mask.any() else offsets.sum() * 0
~~~

- [ ] **Step 4: Build causal correction examples**

`online/dataset.py` returns `features [12,8]`, `mask [12]` and `target_offset [12,2]`. Features are prior XY, broadcast position-residual XY, broadcast velocity-residual XY, heading residual and remaining-time fraction. Elapsed steps are zero. Unit tests mutate unrevealed future values and assert features remain byte-identical.

- [ ] **Step 5: Add config and training CLI**

~~~yaml
# configs/models/corrector.yaml
model: {feature_dim: 8, hidden_dim: 64}
training:
  seed: 2027
  epochs: 50
  batch_size: 512
  learning_rate: 0.001
  weight_decay: 0.0001
  smoothness_weight: 0.05
  patience: 10
~~~

`train_corrector.py` trains on base predictions from `train`, selects by rolling ADE on `train_val`, and uses `SmoothL1(corrected[mask], target[mask]) + 0.05 * smoothness_loss(offsets, mask)`.

- [ ] **Step 6: Test and commit**

~~~powershell
.\.venv\Scripts\python -m pytest tests/online -v
.\.venv\Scripts\ruff check src/trajectory_prediction/online scripts/train_corrector.py
git add configs/models/corrector.yaml src/trajectory_prediction/online scripts/train_corrector.py tests/online
git commit -m "feat: train causal residual trajectory corrector"
~~~

Expected: mask, causality and smoothness tests pass.

### Task 6: Evaluate online strategies

**Files:**
- Create: `src/trajectory_prediction/online/evaluator.py`
- Create: `scripts/evaluate_online.py`
- Create: `tests/online/test_evaluator.py`

- [ ] **Step 1: Write and run the metric test**

~~~python
# tests/online/test_evaluator.py
import numpy as np
from trajectory_prediction.online.evaluator import update_metrics


def test_identical_update_has_zero_error_and_jitter() -> None:
    target = np.array([[0, 1], [0, 2]], np.float32)
    values = update_metrics(target.copy(), target, target.copy(), 1.5)
    assert values == {"ade": 0.0, "fde": 0.0, "jitter": 0.0, "latency_ms": 1.5}
~~~

Run: `.\.venv\Scripts\python -m pytest tests/online/test_evaluator.py -v`

Expected: import fails for `online.evaluator`.

- [ ] **Step 2: Implement metrics**

~~~python
# src/trajectory_prediction/online/evaluator.py
import numpy as np


def update_metrics(current: np.ndarray, target: np.ndarray,
                   previous_overlap: np.ndarray, latency_ms: float) -> dict[str, float]:
    error = np.linalg.norm(current - target, axis=-1)
    jitter = np.linalg.norm(current - previous_overlap, axis=-1)
    return {"ade": float(error.mean()), "fde": float(error[-1]),
            "jitter": float(jitter.mean()), "latency_ms": float(latency_ms)}
~~~

- [ ] **Step 3: Add three-strategy CLI**

`evaluate_online.py` compares trimmed frozen forecasts, sliding-window reforecast and residual correction for identical sample/update pairs. It times batch-one updates with `perf_counter_ns`, synchronizes CUDA around timing, asserts equal pair counts, and writes `online_per_update.csv`, `online_metrics.json` and `online_maneuver_metrics.csv` with p50/p95 latency.

- [ ] **Step 4: Validate and commit**

Use `train_val` while developing; run locked `val` only after configuration freezes.

~~~powershell
.\.venv\Scripts\python -m pytest tests/online -v
.\.venv\Scripts\ruff check src scripts tests
git add src/trajectory_prediction/online/evaluator.py scripts/evaluate_online.py tests/online/test_evaluator.py
git commit -m "feat: evaluate causal online trajectory updates"
~~~

Expected: all strategies have identical finite result counts and causality checks pass.

### Task 7: Complete manual failure analysis

**Files:**
- Create: `docs/FAILURE_ANALYSIS.md`
- Create selected assets: `assets/figures/success/*.png`
- Create selected assets: `assets/figures/failure/*.png`

- [ ] **Step 1:** Render at least 15 high-error and 10 low-error candidates; select 3–5 readable figures in each category.
- [ ] **Step 2:** Review 20–30 cases and record tokens, metrics, tags, affected models and evidence-based failure category.
- [ ] **Step 3:** Write method, category counts, representative cases, counterexamples, static/online comparison and limitations. Every number references a generated artifact and run ID.
- [ ] **Step 4:** Run the full tests and lint, then commit only selected figures and the analysis.

~~~powershell
.\.venv\Scripts\python -m pytest -v
.\.venv\Scripts\ruff check src scripts tests
git add docs/FAILURE_ANALYSIS.md assets/figures/success assets/figures/failure
git commit -m "docs: add maneuver-conditioned failure analysis"
~~~

Expected: tests and lint pass and no intermediate render directory is tracked.
