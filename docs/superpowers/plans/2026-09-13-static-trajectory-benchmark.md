# Static Trajectory Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a tested, configuration-driven nuScenes trajectory pipeline that compares finite-difference CV, official-style CV+Heading, GRU and Transformer predictors under the official 2-second-history/6-second-future protocol.

**Architecture:** nuScenes metadata is converted once into a versioned NPZ cache containing agent-centric histories, future targets and provenance. Physics and learned models share a `predict(history_features, history_mask) -> [B, 12, 2]` contract, while evaluation operates on exported per-sample predictions and never depends on training internals.

**Tech Stack:** Python 3.12, uv, PyTorch 2.6 CUDA 12.4 wheels, nuscenes-devkit 1.2.0 source install, NumPy 1.26, PyYAML, Matplotlib, pandas, pytest, Ruff.

---

## Locked file map

- `pyproject.toml` — package metadata, runtime and development dependencies.
- `.python-version` — project interpreter version.
- `.gitignore` — excludes data, caches, environments, checkpoints and generated outputs.
- `configs/data/mini.yaml` and `configs/data/trainval.yaml` — official split and cache settings.
- `configs/models/*.yaml` — model/training settings.
- `src/trajectory_prediction/config.py` — strict YAML loading and path resolution.
- `src/trajectory_prediction/data/schema.py` — typed in-memory batch/cache contracts.
- `src/trajectory_prediction/data/geometry.py` — coordinate transforms and motion features.
- `src/trajectory_prediction/data/nuscenes_adapter.py` — only module importing nuScenes APIs.
- `src/trajectory_prediction/data/cache.py` — NPZ serialization and manifest validation.
- `src/trajectory_prediction/models/physics.py` — deterministic physics predictors.
- `src/trajectory_prediction/models/gru.py` and `transformer.py` — learned predictors.
- `src/trajectory_prediction/models/factory.py` — one model-construction entry point.
- `src/trajectory_prediction/evaluation/metrics.py` — per-sample ADE/FDE/MissRate and horizons.
- `src/trajectory_prediction/training/dataset.py` — cache-backed PyTorch dataset.
- `src/trajectory_prediction/training/checkpoint.py` — atomic, resumable checkpoints.
- `src/trajectory_prediction/training/engine.py` — train/validation loop and CSV logging.
- `src/trajectory_prediction/utils/seed.py` — deterministic random-state setup.
- `scripts/prepare_data.py`, `train.py` and `evaluate.py` — thin CLI entry points.
- `tests/` — unit and integration tests mirroring package responsibilities.

### Task 1: Scaffold a reproducible Python project

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `src/trajectory_prediction/__init__.py`
- Create: `tests/test_package.py`

- [ ] **Step 1: Write the package smoke test**

~~~python
# tests/test_package.py
from trajectory_prediction import __version__


def test_package_version() -> None:
    assert __version__ == "0.1.0"
~~~

- [ ] **Step 2: Create project metadata**

~~~toml
# pyproject.toml
[build-system]
requires = ["hatchling>=1.27,<2"]
build-backend = "hatchling.build"

[project]
name = "nuscenes-trajectory-prediction"
version = "0.1.0"
description = "Physics and lightweight learned trajectory prediction on nuScenes"
readme = "README.md"
requires-python = ">=3.12,<3.13"
dependencies = [
  "matplotlib>=3.9,<4",
  "numpy>=1.26,<2",
  "pandas>=2.2,<3",
  "pyquaternion>=0.9.9,<1",
  "pyyaml>=6,<7",
  "tqdm>=4.66,<5",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.3,<9",
  "pytest-cov>=6,<7",
  "ruff>=0.9,<1",
]

[tool.hatch.build.targets.wheel]
packages = ["src/trajectory_prediction"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"

[tool.ruff]
line-length = 100
target-version = "py312"
~~~

~~~text
# .python-version
3.12
~~~

~~~python
# src/trajectory_prediction/__init__.py
__version__ = "0.1.0"
~~~

~~~gitignore
# .gitignore
.venv/
.uv-cache/
.python/
__pycache__/
*.py[cod]
.pytest_cache/
.ruff_cache/
.coverage
data/
outputs/
checkpoints/
*.pt
*.pth
*.ckpt
wandb/
.env
~~~

- [ ] **Step 3: Create the Python 3.12 environment on E:**

Run in PowerShell:

~~~powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
$env:UV_PYTHON_INSTALL_DIR = "$PWD\.python"
uv python install 3.12
uv venv --python 3.12
uv pip install -e ".[dev]"
uv pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install "nuscenes-devkit @ git+https://github.com/nutonomy/nuscenes-devkit.git@v1.2.0#subdirectory=setup"
~~~

Expected: `.venv\Scripts\python.exe` exists; no environment or package cache is created on C: by these commands.

- [ ] **Step 4: Run the smoke test and environment checks**

Run:

~~~powershell
.\.venv\Scripts\python -m pytest tests/test_package.py -v
.\.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.\.venv\Scripts\python -c "from nuscenes.prediction import PredictHelper; from nuscenes.eval.prediction.splits import get_prediction_challenge_split; print('prediction imports OK')"
~~~

Expected: one test passes, CUDA availability is `True` with RTX 3060, and the prediction imports print `prediction imports OK`.

- [ ] **Step 5: Commit**

~~~powershell
git add pyproject.toml .python-version .gitignore src tests/test_package.py
git commit -m "build: scaffold reproducible Python project"
~~~

### Task 2: Implement trajectory schema and coordinate geometry

**Files:**
- Create: `src/trajectory_prediction/data/__init__.py`
- Create: `src/trajectory_prediction/data/schema.py`
- Create: `src/trajectory_prediction/data/geometry.py`
- Create: `tests/data/test_geometry.py`

- [ ] **Step 1: Write failing geometry tests**

~~~python
# tests/data/test_geometry.py
import numpy as np

from trajectory_prediction.data.geometry import (
    build_motion_features,
    global_to_agent,
    agent_to_global,
)


def test_yaw_zero_maps_global_forward_to_local_positive_y() -> None:
    points = np.array([[11.0, 5.0]], dtype=np.float32)
    local = global_to_agent(points, anchor_xy=np.array([10.0, 5.0]), anchor_yaw=0.0)
    np.testing.assert_allclose(local, [[0.0, 1.0]], atol=1e-6)


def test_coordinate_transform_round_trip() -> None:
    points = np.array([[4.0, -2.0], [7.0, 8.0]], dtype=np.float32)
    anchor = np.array([2.0, 3.0], dtype=np.float32)
    local = global_to_agent(points, anchor, anchor_yaw=0.37)
    restored = agent_to_global(local, anchor, anchor_yaw=0.37)
    np.testing.assert_allclose(restored, points, atol=1e-5)


def test_motion_features_have_position_displacement_and_velocity() -> None:
    xy = np.array([[0, -2], [0, -1], [0, 0]], dtype=np.float32)
    features = build_motion_features(xy, dt=0.5)
    assert features.shape == (3, 6)
    np.testing.assert_allclose(features[-1], [0, 0, 0, 1, 0, 2])
~~~

- [ ] **Step 2: Run tests and verify the expected failure**

Run: `.\.venv\Scripts\python -m pytest tests/data/test_geometry.py -v`

Expected: collection fails because `trajectory_prediction.data.geometry` does not exist.

- [ ] **Step 3: Add the immutable sample schema**

~~~python
# src/trajectory_prediction/data/schema.py
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrajectorySample:
    instance_token: str
    sample_token: str
    scene_name: str
    history_xy: np.ndarray
    history_features: np.ndarray
    history_mask: np.ndarray
    future_xy: np.ndarray
    anchor_xy: np.ndarray
    anchor_yaw: float
    current_speed: float
~~~

- [ ] **Step 4: Implement coordinate transforms and features**

~~~python
# src/trajectory_prediction/data/geometry.py
import math

import numpy as np


def _rotation(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def global_to_agent(
    points: np.ndarray, anchor_xy: np.ndarray, anchor_yaw: float
) -> np.ndarray:
    points64 = np.asarray(points, dtype=np.float64)
    anchor64 = np.asarray(anchor_xy, dtype=np.float64)
    rotation = _rotation(math.pi / 2.0 - float(anchor_yaw))
    return ((points64 - anchor64) @ rotation.T).astype(np.float32)


def agent_to_global(
    points: np.ndarray, anchor_xy: np.ndarray, anchor_yaw: float
) -> np.ndarray:
    points64 = np.asarray(points, dtype=np.float64)
    anchor64 = np.asarray(anchor_xy, dtype=np.float64)
    rotation = _rotation(math.pi / 2.0 - float(anchor_yaw))
    return (points64 @ rotation + anchor64).astype(np.float32)


def build_motion_features(history_xy: np.ndarray, dt: float) -> np.ndarray:
    xy = np.asarray(history_xy, dtype=np.float32)
    displacement = np.zeros_like(xy)
    displacement[1:] = xy[1:] - xy[:-1]
    velocity = displacement / np.float32(dt)
    return np.concatenate([xy, displacement, velocity], axis=-1)
~~~

Create an empty `src/trajectory_prediction/data/__init__.py`.

- [ ] **Step 5: Run the tests**

Run: `.\.venv\Scripts\python -m pytest tests/data/test_geometry.py -v`

Expected: all three tests pass.

- [ ] **Step 6: Commit**

~~~powershell
git add src/trajectory_prediction/data tests/data/test_geometry.py
git commit -m "feat: add trajectory schema and agent-frame geometry"
~~~

### Task 3: Implement deterministic metrics and physics baselines

**Files:**
- Create: `src/trajectory_prediction/evaluation/__init__.py`
- Create: `src/trajectory_prediction/evaluation/metrics.py`
- Create: `src/trajectory_prediction/models/__init__.py`
- Create: `src/trajectory_prediction/models/physics.py`
- Create: `tests/evaluation/test_metrics.py`
- Create: `tests/models/test_physics.py`

- [ ] **Step 1: Write failing metric tests**

~~~python
# tests/evaluation/test_metrics.py
import torch

from trajectory_prediction.evaluation.metrics import displacement_metrics, horizon_metrics


def test_displacement_metrics_match_known_distances() -> None:
    target = torch.zeros(2, 12, 2)
    prediction = torch.zeros_like(target)
    prediction[0, :, 0] = 1.0
    values = displacement_metrics(prediction, target)
    torch.testing.assert_close(values["ade"], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(values["fde"], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(values["miss"], torch.tensor([0.0, 0.0]))


def test_horizon_metrics_slice_one_three_and_six_seconds() -> None:
    target = torch.zeros(1, 12, 2)
    prediction = torch.arange(1, 13).view(1, 12, 1).repeat(1, 1, 2).float()
    values = horizon_metrics(prediction, target, frequency_hz=2)
    assert set(values) == {"1s", "3s", "6s"}
    torch.testing.assert_close(values["1s"]["fde"], torch.tensor([2.0 * 2**0.5]))
~~~

- [ ] **Step 2: Write failing physics tests**

~~~python
# tests/models/test_physics.py
import torch

from trajectory_prediction.models.physics import (
    finite_difference_cv,
    constant_velocity_heading,
)


def test_finite_difference_cv_preserves_vector_velocity() -> None:
    history = torch.tensor([[[0.0, -1.0], [0.5, -0.5], [1.0, 0.0]]])
    prediction = finite_difference_cv(history, dt=0.5, future_steps=2)
    torch.testing.assert_close(
        prediction, torch.tensor([[[1.5, 0.5], [2.0, 1.0]]])
    )


def test_heading_baseline_moves_along_local_positive_y() -> None:
    speed = torch.tensor([4.0])
    prediction = constant_velocity_heading(speed, dt=0.5, future_steps=2)
    torch.testing.assert_close(
        prediction, torch.tensor([[[0.0, 2.0], [0.0, 4.0]]])
    )
~~~

- [ ] **Step 3: Run tests and verify failure**

Run: `.\.venv\Scripts\python -m pytest tests/evaluation/test_metrics.py tests/models/test_physics.py -v`

Expected: imports fail because metrics and physics modules do not exist.

- [ ] **Step 4: Implement deterministic metrics**

~~~python
# src/trajectory_prediction/evaluation/metrics.py
import torch


def displacement_metrics(
    prediction: torch.Tensor, target: torch.Tensor, miss_tolerance_m: float = 2.0
) -> dict[str, torch.Tensor]:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must share shape [B, T, 2]")
    distance = torch.linalg.vector_norm(prediction - target, dim=-1)
    return {
        "ade": distance.mean(dim=-1),
        "fde": distance[:, -1],
        "miss": (distance.max(dim=-1).values > miss_tolerance_m).float(),
    }


def horizon_metrics(
    prediction: torch.Tensor, target: torch.Tensor, frequency_hz: int = 2
) -> dict[str, dict[str, torch.Tensor]]:
    output: dict[str, dict[str, torch.Tensor]] = {}
    for seconds in (1, 3, 6):
        stop = seconds * frequency_hz
        output[f"{seconds}s"] = displacement_metrics(
            prediction[:, :stop], target[:, :stop]
        )
    return output
~~~

- [ ] **Step 5: Implement physics predictors**

~~~python
# src/trajectory_prediction/models/physics.py
import torch


def finite_difference_cv(
    history_xy: torch.Tensor, dt: float = 0.5, future_steps: int = 12
) -> torch.Tensor:
    if history_xy.ndim != 3 or history_xy.shape[1] < 2:
        raise ValueError("history_xy must have shape [B, H>=2, 2]")
    velocity = (history_xy[:, -1] - history_xy[:, -2]) / dt
    times = torch.arange(
        1, future_steps + 1, device=history_xy.device, dtype=history_xy.dtype
    ) * dt
    return history_xy[:, -1, None, :] + velocity[:, None, :] * times[None, :, None]


def constant_velocity_heading(
    speed: torch.Tensor, dt: float = 0.5, future_steps: int = 12
) -> torch.Tensor:
    times = torch.arange(
        1, future_steps + 1, device=speed.device, dtype=speed.dtype
    ) * dt
    output = torch.zeros(speed.shape[0], future_steps, 2, device=speed.device)
    output[:, :, 1] = speed[:, None] * times[None, :]
    return output
~~~

Create empty `__init__.py` files under `models` and `evaluation`.

- [ ] **Step 6: Run tests**

Run: `.\.venv\Scripts\python -m pytest tests/evaluation/test_metrics.py tests/models/test_physics.py -v`

Expected: all four tests pass.

- [ ] **Step 7: Commit**

~~~powershell
git add src/trajectory_prediction/models src/trajectory_prediction/evaluation tests/models tests/evaluation
git commit -m "feat: add physics baselines and trajectory metrics"
~~~

### Task 4: Build the official nuScenes metadata adapter and cache

**Files:**
- Create: `src/trajectory_prediction/data/nuscenes_adapter.py`
- Create: `src/trajectory_prediction/data/cache.py`
- Create: `src/trajectory_prediction/config.py`
- Create: `configs/data/mini.yaml`
- Create: `configs/data/trainval.yaml`
- Create: `scripts/prepare_data.py`
- Create: `tests/data/test_cache.py`
- Create: `tests/data/test_nuscenes_adapter.py`

- [ ] **Step 1: Write a cache round-trip test**

~~~python
# tests/data/test_cache.py
from pathlib import Path

import numpy as np

from trajectory_prediction.data.cache import load_cache, save_cache


def test_cache_round_trip_and_manifest(tmp_path: Path) -> None:
    arrays = {
        "history_xy": np.zeros((2, 5, 2), np.float32),
        "history_features": np.zeros((2, 5, 6), np.float32),
        "history_mask": np.ones((2, 5), np.bool_),
        "future_xy": np.zeros((2, 12, 2), np.float32),
        "anchor_xy": np.zeros((2, 2), np.float32),
        "anchor_yaw": np.zeros(2, np.float32),
        "current_speed": np.ones(2, np.float32),
        "instance_token": np.array(["a", "b"]),
        "sample_token": np.array(["c", "d"]),
        "scene_name": np.array(["s1", "s2"]),
    }
    save_cache(tmp_path, arrays, {"schema_version": 1, "split": "mini_train"})
    loaded, manifest = load_cache(tmp_path, expected_split="mini_train")
    np.testing.assert_array_equal(loaded["future_xy"], arrays["future_xy"])
    assert manifest["schema_version"] == 1
~~~

- [ ] **Step 2: Write adapter helper tests without real data**

~~~python
# tests/data/test_nuscenes_adapter.py
import numpy as np

from trajectory_prediction.data.nuscenes_adapter import assemble_history


def test_assemble_history_reverses_devkit_past_and_appends_current() -> None:
    newest_first = np.array([[0.0, -0.5], [0.0, -1.0]], dtype=np.float32)
    current = np.array([0.0, 0.0], dtype=np.float32)
    history = assemble_history(newest_first, current, expected_past_steps=2)
    np.testing.assert_allclose(history, [[0, -1], [0, -0.5], [0, 0]])
~~~

- [ ] **Step 3: Run tests and verify failure**

Run: `.\.venv\Scripts\python -m pytest tests/data/test_cache.py tests/data/test_nuscenes_adapter.py -v`

Expected: module import failures.

- [ ] **Step 4: Implement versioned cache IO**

~~~python
# src/trajectory_prediction/data/cache.py
import hashlib
import json
from pathlib import Path

import numpy as np


REQUIRED_ARRAYS = {
    "history_xy", "history_features", "history_mask", "future_xy",
    "anchor_xy", "anchor_yaw", "current_speed", "instance_token",
    "sample_token", "scene_name",
}


def save_cache(root: Path, arrays: dict[str, np.ndarray], manifest: dict) -> None:
    missing = REQUIRED_ARRAYS - arrays.keys()
    if missing:
        raise ValueError(f"cache arrays missing: {sorted(missing)}")
    root.mkdir(parents=True, exist_ok=True)
    npz_path = root / "trajectories.npz"
    np.savez_compressed(npz_path, **arrays)
    payload = dict(manifest)
    payload["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_cache(
    root: Path, expected_split: str
) -> tuple[dict[str, np.ndarray], dict]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported cache schema")
    if manifest.get("split") != expected_split:
        raise ValueError("cache split does not match requested split")
    npz_path = root / "trajectories.npz"
    digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    if digest != manifest.get("npz_sha256"):
        raise ValueError("cache checksum mismatch")
    with np.load(npz_path) as data:
        arrays = {name: data[name] for name in data.files}
    return arrays, manifest
~~~

- [ ] **Step 5: Implement the metadata adapter core**

~~~python
# src/trajectory_prediction/data/nuscenes_adapter.py
from collections.abc import Iterable

import numpy as np
from pyquaternion import Quaternion

from trajectory_prediction.data.geometry import build_motion_features, global_to_agent


def assemble_history(
    past_newest_first: np.ndarray,
    current_xy: np.ndarray,
    expected_past_steps: int,
) -> np.ndarray:
    past = np.asarray(past_newest_first, dtype=np.float32)
    if len(past) != expected_past_steps:
        raise ValueError(f"expected {expected_past_steps} past points, received {len(past)}")
    return np.concatenate([past[::-1], np.asarray(current_xy)[None]], axis=0)


def quaternion_yaw(rotation: Iterable[float]) -> float:
    quaternion = Quaternion(rotation)
    return float(quaternion.yaw_pitch_roll[0])


def extract_split(nusc, split: str, history_seconds: float = 2.0) -> tuple[dict, dict]:
    from nuscenes.eval.prediction.splits import get_prediction_challenge_split
    from nuscenes.prediction import PredictHelper

    helper = PredictHelper(nusc)
    keys = get_prediction_challenge_split(split, dataroot=nusc.dataroot)
    records: dict[str, list] = {
        name: [] for name in (
            "history_xy", "history_features", "history_mask", "future_xy",
            "anchor_xy", "anchor_yaw", "current_speed", "instance_token",
            "sample_token", "scene_name",
        )
    }
    rejected = {"incomplete_history": 0, "incomplete_future": 0}
    for key in keys:
        instance_token, sample_token = key.split("_")
        annotation = helper.get_sample_annotation(instance_token, sample_token)
        past = helper.get_past_for_agent(
            instance_token, sample_token, history_seconds, in_agent_frame=False
        )
        future = helper.get_future_for_agent(
            instance_token, sample_token, 6.0, in_agent_frame=False
        )
        if len(past) != 4:
            rejected["incomplete_history"] += 1
            continue
        if len(future) != 12:
            rejected["incomplete_future"] += 1
            continue
        anchor_xy = np.asarray(annotation["translation"][:2], dtype=np.float32)
        anchor_yaw = quaternion_yaw(annotation["rotation"])
        history_global = assemble_history(past, anchor_xy, expected_past_steps=4)
        history_local = global_to_agent(history_global, anchor_xy, anchor_yaw)
        future_local = global_to_agent(future, anchor_xy, anchor_yaw)
        sample = nusc.get("sample", sample_token)
        scene_name = nusc.get("scene", sample["scene_token"])["name"]
        speed = helper.get_velocity_for_agent(instance_token, sample_token)
        records["history_xy"].append(history_local)
        records["history_features"].append(build_motion_features(history_local, 0.5))
        records["history_mask"].append(np.ones(5, dtype=np.bool_))
        records["future_xy"].append(future_local)
        records["anchor_xy"].append(anchor_xy)
        records["anchor_yaw"].append(anchor_yaw)
        records["current_speed"].append(0.0 if np.isnan(speed) else speed)
        records["instance_token"].append(instance_token)
        records["sample_token"].append(sample_token)
        records["scene_name"].append(scene_name)
    arrays = {name: np.asarray(values) for name, values in records.items()}
    return arrays, {"requested": len(keys), "accepted": len(arrays["future_xy"]), **rejected}
~~~

- [ ] **Step 6: Add strict data configuration**

~~~yaml
# configs/data/mini.yaml
version: v1.0-mini
dataroot: data/nuscenes
splits: [mini_train, mini_val]
history_seconds: 2.0
future_seconds: 6.0
frequency_hz: 2
cache_root: data/cache/mini
~~~

~~~yaml
# configs/data/trainval.yaml
version: v1.0-trainval
dataroot: data/nuscenes
splits: [train, train_val, val]
history_seconds: 2.0
future_seconds: 6.0
frequency_hz: 2
cache_root: data/cache/trainval
~~~

~~~python
# src/trajectory_prediction/config.py
from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return content
~~~

- [ ] **Step 7: Add the preparation CLI**

~~~python
# scripts/prepare_data.py
import argparse
from pathlib import Path

from nuscenes import NuScenes

from trajectory_prediction.config import load_yaml
from trajectory_prediction.data.cache import save_cache
from trajectory_prediction.data.nuscenes_adapter import extract_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)
    nusc = NuScenes(
        version=config["version"], dataroot=config["dataroot"], verbose=True
    )
    for split in config["splits"]:
        arrays, counts = extract_split(nusc, split, config["history_seconds"])
        save_cache(
            Path(config["cache_root"]) / split,
            arrays,
            {
                "schema_version": 1,
                "split": split,
                "dataset_version": config["version"],
                "history_seconds": config["history_seconds"],
                "future_seconds": config["future_seconds"],
                "frequency_hz": config["frequency_hz"],
                "filter_counts": counts,
            },
        )


if __name__ == "__main__":
    main()
~~~

- [ ] **Step 8: Run unit tests**

Run: `.\.venv\Scripts\python -m pytest tests/data -v`

Expected: all data and geometry tests pass.

- [ ] **Step 9: Commit**

~~~powershell
git add configs/data scripts/prepare_data.py src/trajectory_prediction/config.py src/trajectory_prediction/data tests/data
git commit -m "feat: prepare official nuScenes trajectory caches"
~~~

### Task 5: Implement GRU and Transformer predictors

**Files:**
- Create: `src/trajectory_prediction/models/gru.py`
- Create: `src/trajectory_prediction/models/transformer.py`
- Create: `src/trajectory_prediction/models/factory.py`
- Create: `configs/models/gru.yaml`
- Create: `configs/models/transformer.yaml`
- Create: `tests/models/test_learned_models.py`

- [ ] **Step 1: Write model contract tests**

~~~python
# tests/models/test_learned_models.py
import pytest
import torch

from trajectory_prediction.models.gru import GRUPredictor
from trajectory_prediction.models.transformer import TransformerPredictor


@pytest.mark.parametrize(
    "model",
    [
        GRUPredictor(input_dim=6, hidden_dim=64, layers=1, future_steps=12),
        TransformerPredictor(
            input_dim=6, hidden_dim=128, heads=4, layers=2, future_steps=12
        ),
    ],
)
def test_predictor_contract_and_backward(model: torch.nn.Module) -> None:
    features = torch.randn(4, 5, 6)
    mask = torch.tensor([[1, 1, 1, 1, 1]] * 4, dtype=torch.bool)
    output = model(features, mask)
    assert output.shape == (4, 12, 2)
    output.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
~~~

- [ ] **Step 2: Run and verify failure**

Run: `.\.venv\Scripts\python -m pytest tests/models/test_learned_models.py -v`

Expected: imports fail for both learned model modules.

- [ ] **Step 3: Implement the GRU**

~~~python
# src/trajectory_prediction/models/gru.py
import torch
from torch import nn


class GRUPredictor(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int, layers: int, future_steps: int
    ) -> None:
        super().__init__()
        self.encoder = nn.GRU(
            input_dim, hidden_dim, num_layers=layers, batch_first=True
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, future_steps * 2),
        )
        self.future_steps = future_steps

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            features, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.encoder(packed)
        return self.head(hidden[-1]).view(features.shape[0], self.future_steps, 2)
~~~

- [ ] **Step 4: Implement the Transformer**

~~~python
# src/trajectory_prediction/models/transformer.py
import math

import torch
from torch import nn


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
        self.position = nn.Parameter(torch.zeros(1, max_history_steps, hidden_dim))
        block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, future_steps * 2)
        self.future_steps = future_steps

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(features) + self.position[:, : features.shape[1]]
        hidden = self.encoder(hidden, src_key_padding_mask=~mask)
        last_index = mask.sum(dim=1) - 1
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_index]
        return self.head(self.norm(pooled)).view(features.shape[0], self.future_steps, 2)
~~~

- [ ] **Step 5: Add model configs and factory**

~~~yaml
# configs/models/gru.yaml
model:
  name: gru
  input_dim: 6
  hidden_dim: 64
  layers: 1
  future_steps: 12
training:
  seed: 2027
  epochs: 80
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.0001
  patience: 12
~~~

~~~yaml
# configs/models/transformer.yaml
model:
  name: transformer
  input_dim: 6
  hidden_dim: 128
  heads: 4
  layers: 2
  future_steps: 12
training:
  seed: 2027
  epochs: 80
  batch_size: 256
  learning_rate: 0.0005
  weight_decay: 0.0001
  patience: 12
~~~

~~~python
# src/trajectory_prediction/models/factory.py
from torch import nn

from trajectory_prediction.models.gru import GRUPredictor
from trajectory_prediction.models.transformer import TransformerPredictor


def build_model(config: dict) -> nn.Module:
    name = config["name"]
    kwargs = {key: value for key, value in config.items() if key != "name"}
    if name == "gru":
        return GRUPredictor(**kwargs)
    if name == "transformer":
        return TransformerPredictor(**kwargs)
    raise ValueError(f"unknown learned model: {name}")
~~~

- [ ] **Step 6: Run tests**

Run: `.\.venv\Scripts\python -m pytest tests/models -v`

Expected: physics and learned model tests pass.

- [ ] **Step 7: Commit**

~~~powershell
git add configs/models src/trajectory_prediction/models tests/models
git commit -m "feat: add lightweight GRU and Transformer predictors"
~~~

### Task 6: Implement dataset loading, deterministic training and resume

**Files:**
- Create: `src/trajectory_prediction/training/__init__.py`
- Create: `src/trajectory_prediction/training/dataset.py`
- Create: `src/trajectory_prediction/training/checkpoint.py`
- Create: `src/trajectory_prediction/training/engine.py`
- Create: `src/trajectory_prediction/utils/__init__.py`
- Create: `src/trajectory_prediction/utils/seed.py`
- Create: `scripts/train.py`
- Create: `tests/training/test_training.py`

- [ ] **Step 1: Write the tiny-batch overfit and checkpoint test**

~~~python
# tests/training/test_training.py
from pathlib import Path

import torch

from trajectory_prediction.models.gru import GRUPredictor
from trajectory_prediction.training.checkpoint import load_checkpoint, save_checkpoint
from trajectory_prediction.training.engine import optimize_batch


def test_model_overfits_repeated_tiny_batch_and_resumes(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = GRUPredictor(6, 16, 1, 12)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    features = torch.randn(4, 5, 6).repeat(4, 1, 1)
    mask = torch.ones(16, 5, dtype=torch.bool)
    target = torch.randn(4, 12, 2).repeat(4, 1, 1)
    first = optimize_batch(model, optimizer, features, mask, target)
    for _ in range(100):
        last = optimize_batch(model, optimizer, features, mask, target)
    assert last < first * 0.2
    path = tmp_path / "state.pt"
    save_checkpoint(path, model, optimizer, epoch=9, best_ade=1.25)
    state = load_checkpoint(path, model, optimizer)
    assert state == {"epoch": 9, "best_ade": 1.25}
~~~

- [ ] **Step 2: Run and verify failure**

Run: `.\.venv\Scripts\python -m pytest tests/training/test_training.py -v`

Expected: training module imports fail.

- [ ] **Step 3: Implement cache-backed dataset and deterministic setup**

~~~python
# src/trajectory_prediction/training/dataset.py
import torch
from torch.utils.data import Dataset


class TrajectoryDataset(Dataset):
    def __init__(self, arrays: dict, history_steps: int = 5) -> None:
        self.arrays = arrays
        self.history_steps = history_steps

    def __len__(self) -> int:
        return len(self.arrays["future_xy"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = self.arrays["history_features"].shape[1] - self.history_steps
        return {
            "history_features": torch.from_numpy(
                self.arrays["history_features"][index, start:]
            ).float(),
            "history_xy": torch.from_numpy(
                self.arrays["history_xy"][index, start:]
            ).float(),
            "history_mask": torch.from_numpy(
                self.arrays["history_mask"][index, start:]
            ).bool(),
            "future_xy": torch.from_numpy(self.arrays["future_xy"][index]).float(),
            "current_speed": torch.tensor(
                self.arrays["current_speed"][index], dtype=torch.float32
            ),
            "index": torch.tensor(index),
        }
~~~

~~~python
# src/trajectory_prediction/utils/seed.py
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
~~~

- [ ] **Step 4: Implement atomic checkpoints**

~~~python
# src/trajectory_prediction/training/checkpoint.py
import os
from pathlib import Path

import torch


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_ade: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_ade": best_ade,
            "torch_rng": torch.get_rng_state(),
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint(
    path: Path, model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> dict[str, float | int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"])
    return {"epoch": int(payload["epoch"]), "best_ade": float(payload["best_ade"])}
~~~

- [ ] **Step 5: Implement one optimization primitive**

~~~python
# src/trajectory_prediction/training/engine.py
import torch
from torch.nn import functional as F


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
~~~

Extend this file during the same step with `evaluate_epoch` and `fit` functions that:

- move batches to the configured device;
- average loss, ADE and FDE weighted by sample count;
- write one CSV row containing epoch, train loss, val ADE, val FDE and learning rate;
- save `last.pt` every epoch and replace `best.pt` only when validation ADE improves;
- stop after the configured patience;
- resume from `last.pt` using the next epoch number.

The functions must call `displacement_metrics` and `save_checkpoint` defined in earlier tasks; they must not duplicate those implementations.

- [ ] **Step 6: Add the training CLI**

Implement `scripts/train.py` to accept:

~~~text
--data-config configs/data/mini.yaml
--model-config configs/models/gru.yaml
--train-split mini_train
--val-split mini_val
--history-seconds 2.0
--output-dir outputs/mini/gru
--resume
~~~

It must load both YAML files, convert history seconds to `history_steps = seconds * 2 + 1`, load the two caches, seed before constructing data loaders and models, select CUDA when available, and call `fit`. Reject history outside `{0.5, 1.0, 2.0}`.

- [ ] **Step 7: Run tests and lint**

Run:

~~~powershell
.\.venv\Scripts\python -m pytest tests/training/test_training.py -v
.\.venv\Scripts\ruff check src scripts tests
~~~

Expected: tiny-batch loss falls by at least 80%, checkpoint state restores, and Ruff reports no errors.

- [ ] **Step 8: Commit**

~~~powershell
git add src/trajectory_prediction/training src/trajectory_prediction/utils scripts/train.py tests/training
git commit -m "feat: add deterministic resumable training"
~~~

### Task 7: Implement unified static evaluation and exports

**Files:**
- Create: `src/trajectory_prediction/evaluation/runner.py`
- Create: `configs/models/cv.yaml`
- Create: `configs/models/cv_heading.yaml`
- Create: `scripts/evaluate.py`
- Create: `tests/evaluation/test_runner.py`

- [ ] **Step 1: Write an export test**

~~~python
# tests/evaluation/test_runner.py
import json
from pathlib import Path

import numpy as np

from trajectory_prediction.evaluation.runner import write_evaluation


def test_evaluation_writes_summary_and_per_sample_rows(tmp_path: Path) -> None:
    prediction = np.zeros((2, 12, 2), np.float32)
    target = prediction.copy()
    target[1, :, 0] = 1.0
    identifiers = {
        "instance_token": np.array(["i1", "i2"]),
        "sample_token": np.array(["s1", "s2"]),
        "scene_name": np.array(["scene-a", "scene-b"]),
    }
    write_evaluation(tmp_path, prediction, target, identifiers)
    summary = json.loads((tmp_path / "metrics.json").read_text())
    assert summary["sample_count"] == 2
    assert summary["ade"] == 0.5
    assert (tmp_path / "per_sample.csv").exists()
~~~

- [ ] **Step 2: Run and verify failure**

Run: `.\.venv\Scripts\python -m pytest tests/evaluation/test_runner.py -v`

Expected: import fails for `evaluation.runner`.

- [ ] **Step 3: Implement the evaluation writer**

~~~python
# src/trajectory_prediction/evaluation/runner.py
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajectory_prediction.evaluation.metrics import displacement_metrics, horizon_metrics


def write_evaluation(
    output_dir: Path,
    prediction: np.ndarray,
    target: np.ndarray,
    identifiers: dict[str, np.ndarray],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred_tensor = torch.from_numpy(prediction)
    target_tensor = torch.from_numpy(target)
    metrics = displacement_metrics(pred_tensor, target_tensor)
    horizons = horizon_metrics(pred_tensor, target_tensor)
    summary = {
        "sample_count": len(prediction),
        "ade": float(metrics["ade"].mean()),
        "fde": float(metrics["fde"].mean()),
        "miss_rate_2": float(metrics["miss"].mean()),
        "horizons": {
            name: {
                "ade": float(values["ade"].mean()),
                "fde": float(values["fde"].mean()),
            }
            for name, values in horizons.items()
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    frame = pd.DataFrame(identifiers)
    frame["ade"] = metrics["ade"].numpy()
    frame["fde"] = metrics["fde"].numpy()
    frame["miss"] = metrics["miss"].numpy()
    frame.to_csv(output_dir / "per_sample.csv", index=False)
    np.savez_compressed(
        output_dir / "predictions.npz", prediction=prediction, target=target
    )
~~~

- [ ] **Step 4: Add model configs**

~~~yaml
# configs/models/cv.yaml
model:
  name: finite_difference_cv
  future_steps: 12
~~~

~~~yaml
# configs/models/cv_heading.yaml
model:
  name: constant_velocity_heading
  future_steps: 12
~~~

- [ ] **Step 5: Add evaluation CLI**

Implement `scripts/evaluate.py` with arguments `--data-config`, `--model-config`, `--split`, `--checkpoint`, `--history-seconds` and `--output-dir`. It must:

- load a cache and preserve its original ordering;
- run either physics functions or a learned model built by `build_model`;
- require a checkpoint only for GRU/Transformer;
- use inference mode and configured device;
- save agent-local predictions through `write_evaluation`;
- additionally inverse-transform every prediction to global coordinates and write official-format JSON with one mode and probability 1.0.

- [ ] **Step 6: Run all static unit tests**

Run:

~~~powershell
.\.venv\Scripts\python -m pytest -v
.\.venv\Scripts\ruff check src scripts tests
~~~

Expected: all tests pass and Ruff reports no errors.

- [ ] **Step 7: Commit**

~~~powershell
git add configs/models scripts/evaluate.py src/trajectory_prediction/evaluation tests/evaluation
git commit -m "feat: add unified static trajectory evaluation"
~~~

### Task 8: Run mini smoke tests and the controlled main protocol

**Files:**
- Modify: `README.md`
- Create: `docs/DATA_SETUP.md`
- Generated but ignored: `data/nuscenes/`, `data/cache/`, `outputs/`

- [ ] **Step 1: Document licensed data acquisition**

Document the exact official archive names, expected directory tree and checks:

~~~text
data/nuscenes/
├── maps/
│   ├── expansion/
│   └── prediction/prediction_scenes.json
├── v1.0-mini/
└── v1.0-trainval/
~~~

State that users obtain trainval metadata from the official nuScenes download page after accepting its terms. Do not automate credentials, cookies or signed URLs.

- [ ] **Step 2: Download only mini and map expansion**

Use official URLs in a user-approved network step, place archives under `data/downloads`, verify archive names, and extract into `data/nuscenes`. Do not download trainval sensor blobs.

- [ ] **Step 3: Prepare mini caches**

Run:

~~~powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
.\.venv\Scripts\python scripts/prepare_data.py --config configs/data/mini.yaml
~~~

Expected: manifests for `mini_train` and `mini_val` show nonzero accepted samples and no split overlap.

- [ ] **Step 4: Train mini GRU and Transformer**

Run:

~~~powershell
.\.venv\Scripts\python scripts/train.py --data-config configs/data/mini.yaml --model-config configs/models/gru.yaml --train-split mini_train --val-split mini_val --history-seconds 2.0 --output-dir outputs/mini/gru
.\.venv\Scripts\python scripts/train.py --data-config configs/data/mini.yaml --model-config configs/models/transformer.yaml --train-split mini_train --val-split mini_val --history-seconds 2.0 --output-dir outputs/mini/transformer
~~~

Expected: each run produces `best.pt`, `last.pt` and `metrics.csv`; validation ADE and FDE are finite.

- [ ] **Step 5: Evaluate all four mini models**

Run `scripts/evaluate.py` for CV, CV+Heading, GRU and Transformer on `mini_val`. Expected: each output contains `metrics.json`, `per_sample.csv`, `predictions.npz` and official-format prediction JSON. Label all mini results as smoke tests.

- [ ] **Step 6: Acquire metadata-only trainval and prepare official splits**

After the user downloads `v1.0-trainval_meta.tgz`, extract it into `data/nuscenes` and reuse map expansion. Run `prepare_data.py --config configs/data/trainval.yaml`.

Expected: `train`, `train_val` and `val` caches contain nonzero samples; their scene sets are pairwise disjoint.

- [ ] **Step 7: Run controlled main experiments**

Train GRU and Transformer on `train` with `train_val` model selection. Evaluate all four models on locked `val`. Repeat learned-model training for history values 0.5, 1.0 and 2.0 seconds without changing other hyperparameters.

Expected: real-valued model-comparison and history/horizon tables exist under `outputs/main`; no result is copied from mini.

- [ ] **Step 8: Run final verification**

Run:

~~~powershell
.\.venv\Scripts\python -m pytest -v
.\.venv\Scripts\ruff check src scripts tests
git status --short
~~~

Expected: tests and lint pass; Git shows only intentional documentation/result-summary changes and never raw data, caches or checkpoints.

- [ ] **Step 9: Commit**

~~~powershell
git add README.md docs/DATA_SETUP.md
git commit -m "docs: add reproducible nuScenes data workflow"
~~~
