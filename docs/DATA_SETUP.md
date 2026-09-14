# nuScenes data setup

This repository consumes public nuScenes metadata and map annotations. It does not
redistribute the dataset or automate account credentials, cookies, signed URLs, or
acceptance of the nuScenes license. Download files yourself from the official source.

## What to download

For a one-week smoke test, download only:

- `v1.0-mini.tgz`
- `nuScenes-map-expansion-v1.3.zip`

For the metadata-only main experiment, download `v1.0-trainval_meta.tgz` and reuse the
map expansion archive. Do not download `v1.0-trainvalXX_blobs.tgz` sensor archives. The
adapter reads annotations, timestamps, scene metadata, and map metadata; camera, LiDAR,
and radar payloads are outside this MVP.

## Expected layout

After extraction, the relevant tree is:

```text
data/
|-- downloads/                 # optional, ignored; original archives
`-- nuscenes/                  # ignored; never commit this directory
    |-- maps/
    |   `-- expansion/prediction/prediction_scenes.json
    |-- v1.0-mini/
    |   |-- scene.json
    |   |-- sample.json
    |   |-- sample_annotation.json
    |   `-- ... metadata tables
    `-- v1.0-trainval/
        |-- scene.json
        |-- sample.json
        |-- sample_annotation.json
        `-- ... metadata tables
```

The extraction directory must match `dataroot` in the selected YAML. Checked-in configs
use `data/nuscenes` and relative cache roots so commands replay from the repository root.

## Environment and storage

Use Python 3.12 and keep the virtual environment, uv cache, archives, data, and outputs
on a drive with sufficient free space. This project was placed on `E:` because the system
`C:` drive was nearly full:

```powershell
$env:UV_CACHE_DIR = "$PWD\.uv-cache"
uv sync --extra dev
.\.venv\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

The project pins `nuscenes-devkit==1.2.0` through PyPI-compatible dependency resolution.
The base dependency includes PyTorch 2.6 so a fresh checkout is importable. The tested
GPU environment uses the CUDA 12.4 wheel; if `uv sync` selected a CPU wheel, reinstall the
matching CUDA wheel from the official PyTorch index before training. CPU execution remains
supported for smoke tests and CI.

## Prepare deterministic caches

Run from the repository root:

```powershell
.\.venv\Scripts\python scripts\prepare_data.py --config configs\data\mini.yaml
```

The script loads the official `mini_train` and `mini_val` prediction splits, enforces the
2-second history / 6-second future / 2 Hz protocol, rejects incomplete trajectories, and
writes one content-addressed NPZ plus `manifest.json` per split. The manifest records the
dataset version, devkit version, filter counts, and a SHA-256 hash of the normalized data
configuration. Loading later with `scripts/train.py` or `scripts/evaluate.py` rejects a
split or configuration hash mismatch before model execution.

For the main metadata-only experiment, first extract `v1.0-trainval_meta.tgz` and run:

```powershell
.\.venv\Scripts\python scripts\prepare_data.py --config configs\data\trainval.yaml
```

The `train`, `train_val`, and `val` caches must be non-empty and scene-disjoint. Never
copy mini results into the main result directory.

## Train and evaluate

Smoke-test learned models on mini data:

```powershell
.\.venv\Scripts\python scripts\train.py `
  --data-config configs\data\mini.yaml `
  --model-config configs\models\gru.yaml `
  --train-split mini_train --val-split mini_val `
  --history-seconds 2.0 --output-dir outputs\mini\gru

.\.venv\Scripts\python scripts\train.py `
  --data-config configs\data\mini.yaml `
  --model-config configs\models\transformer.yaml `
  --train-split mini_train --val-split mini_val `
  --history-seconds 2.0 --output-dir outputs\mini\transformer
```

Evaluate all four static models on `mini_val`:

```powershell
.\.venv\Scripts\python scripts\evaluate.py --split mini_val `
  --model-config configs\models\cv.yaml --output-dir outputs\mini\cv
.\.venv\Scripts\python scripts\evaluate.py --split mini_val `
  --model-config configs\models\cv_heading.yaml --output-dir outputs\mini\cv_heading
.\.venv\Scripts\python scripts\evaluate.py --split mini_val `
  --model-config configs\models\gru.yaml `
  --checkpoint outputs\mini\gru\best.pt --output-dir outputs\mini\gru_eval
.\.venv\Scripts\python scripts\evaluate.py --split mini_val `
  --model-config configs\models\transformer.yaml `
  --checkpoint outputs\mini\transformer\best.pt --output-dir outputs\mini\transformer_eval
```

The allowed history values are `0.5`, `1.0`, and `2.0` seconds. The full protocol uses
five observed points and twelve future points at `+0.5 ... +6.0` seconds. All metrics are
computed in the agent frame; the export additionally applies the cached anchor position
and yaw to write global-coordinate predictions.

## Prediction JSON compatibility

`official_predictions.json` intentionally uses a stable token-key mapping:

```json
{
  "instanceToken_sampleToken": {
    "prediction": [[0.0, 1.0], "... 12 points ..."],
    "probabilities": [1.0]
  }
}
```

This is convenient for joining predictions back to cache rows. The devkit 1.2.0
`compute_metrics.py` API instead deserializes a JSON list whose records contain
`instance`, `sample`, `prediction`, and `probabilities`. Keep the mapping as the project
artifact and convert it to the devkit list shape in a small, reviewable adapter before
running the official evaluator; do not silently claim that local metrics are challenge
submission metrics.

## Reproducibility and safety checklist

Before publishing a branch or GitHub release:

```powershell
git status --short
git ls-files data outputs .venv .uv-cache
git diff --check
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\ruff check src scripts tests
```

The expected result of `git ls-files` is empty for data, outputs, environments, and uv
caches. Do not commit raw archives, extracted tables, generated caches, checkpoints,
tokens, credentials, or absolute local paths. Mini numbers are smoke-test evidence only;
main claims require the locked scene-level train/train_val/val protocol and a documented
rerun.
