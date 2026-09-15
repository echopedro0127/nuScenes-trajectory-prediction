# Reproducibility checklist

This document separates data-independent verification from licensed-data experiments.
No nuScenes archive, cache, checkpoint, or measured benchmark result is committed to this
repository.

## Clean environment

Use Python 3.12 on a drive with enough space. The project declares PyTorch 2.6 and
`nuscenes-devkit==1.2.0` in `pyproject.toml`; the default setup supports CPU smoke tests.
Keep the uv cache on the project drive:

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
$env:CUBLAS_WORKSPACE_CONFIG = ":4096:8"
uv sync --extra dev
.\.venv\Scripts\python.exe -c "import torch, nuscenes; print(torch.__version__)"
```

For GPU training, install the matching CUDA 12.4 PyTorch wheel from the official PyTorch
index after the base environment is created. Record the output of
`torch.cuda.is_available()` and the GPU model in the experiment log.

## Data and provenance

1. Download only the official metadata archives described in [DATA_SETUP.md](DATA_SETUP.md).
2. Extract them to the configured `dataroot`; never place archives under tracked paths.
3. Run `scripts/prepare_data.py` for the selected YAML configuration.
4. Keep each cache manifest beside its NPZ. It records dataset/devkit versions and the
   normalized data-configuration SHA-256.
5. Do not mix `mini_*` caches with `train`, `train_val`, or `val` outputs.

Every training and evaluation command validates the split and configuration hash before
model execution. Preserve the resulting `metrics.json`, `per_sample.csv`, and
`predictions.npz` together with the commit ID and model YAML used to create them.

For the scene-level online experiment, generate GRU predictions for `train`, `train_val`,
and `val`; train the residual corrector on `train`, select its checkpoint on `train_val`,
and pass only the locked `val` prediction file to `evaluate_online.py`. The evaluator
rejects a checkpoint whose training or selection split equals the requested evaluation
split. The mini smoke configuration has only `mini_train` and `mini_val`, so its corrector
selection explicitly reuses `mini_train`; treat those numbers as a pipeline check only.

## Verification gates

Run these commands from the repository root before opening a pull request:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check src scripts tests
.\.venv\Scripts\python.exe scripts\check_public_repo.py
git diff --check
git status --short
```

The tests exercise geometry, physics baselines, learned-model dimensions, deterministic
training/resume, static evaluation, maneuver/failure analysis, causal replay, online
correction, release scripts, and CI configuration without requiring licensed data.

## Experiment record

For each public result, record:

- Git commit and Python/package versions;
- data YAML path, split names, cache manifest hashes, random seed, history length and device;
- model YAML, checkpoint provenance and training command;
- output-directory paths and SHA-256 hashes for the published metric files;
- whether the result is a mini smoke test or the locked scene-level train/train_val/val run.

Only the latter supports final resume claims. If an experiment cannot be reproduced from
the recorded command and artifacts, label it exploratory and keep it out of README tables.
