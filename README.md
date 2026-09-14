# nuScenes Trajectory Prediction

This project studies physics baselines, lightweight learned models, causal online
correction, and motion-conditioned analysis under the nuScenes prediction protocol.

## Status

The repository is under active development. Quantitative results will be published only
after reproducible experiments on official scene-level splits. It contains no placeholder
metrics or state-of-the-art claims.

The current static benchmark compares finite-difference constant velocity,
constant-velocity-plus-heading, GRU, and lightweight Transformer predictors. The planned
supplementary contribution is causal online trajectory correction: reforecasting only with
observations available at each 0.5-second update, and measuring correction gain, jitter,
and latency separately from the official static protocol.

## Project documents

- [Project plan](PROJECT_PLAN.md)
- [Approved technical design](docs/superpowers/specs/2026-09-13-nuscenes-trajectory-prediction-design.md)
- [Implementation roadmap](docs/superpowers/plans/2026-09-13-project-roadmap.md)

## Data and licensing

nuScenes data is not redistributed by this repository. Users must obtain the dataset from
the official nuScenes source and comply with its applicable terms. This project uses no
company-internal data, code, configuration, checkpoints, or preprocessing materials.

See [DATA_SETUP.md](docs/DATA_SETUP.md) for the exact metadata-only download workflow,
expected directory tree, cache provenance checks, and reproducible commands. Keep all
archives, extracted data, caches, outputs, and checkpoints under the project drive; the
repository intentionally tracks none of them.

## Quick start

```powershell
uv sync --extra dev
.\.venv\Scripts\python scripts\prepare_data.py --config configs\data\mini.yaml
.\.venv\Scripts\python scripts\evaluate.py --split mini_val --model-config configs\models\cv.yaml --output-dir outputs\mini\cv
```

The first command that touches `data/nuscenes` requires a user-downloaded nuScenes archive;
no credentials, cookies, signed URLs, or raw data are automated here. Mini metrics are
smoke-test results only and must not be presented as final benchmark claims.

## Static evaluation outputs

Each evaluation directory contains `metrics.json`, `per_sample.csv`, `predictions.npz`,
and `official_predictions.json`. The latter is the repository's order-preserving
`instance_token_sample_token` mapping with one deterministic mode and probability 1.0.
nuScenes-devkit 1.2.0's `compute_metrics.py` consumes a list of dictionaries with
`instance`, `sample`, `prediction`, and `probabilities` fields; convert the mapping to
that devkit submission shape before calling the devkit evaluator. The distinction is
documented explicitly so a project artifact is not mistaken for a directly consumable
challenge submission.
