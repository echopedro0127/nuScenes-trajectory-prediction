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
- [Data setup and licensing workflow](docs/DATA_SETUP.md)
- [Reproducibility checklist](docs/REPRODUCIBILITY.md)
- [Failure and maneuver analysis](docs/FAILURE_ANALYSIS.md)

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

For a fail-fast end-to-end mini run (static baselines, GRU/Transformer, online correction
and maneuver/failure exports), use the PowerShell orchestrator:

```powershell
.\scripts\run_smoke.ps1
```

After the metadata-only train/validation caches are prepared, the full scene-level run is:

```powershell
.\scripts\run_main.ps1
```

Both scripts accept `-SkipTraining` only when the expected `best.pt` checkpoints already
exist. They keep generated artifacts under `outputs/`, which is intentionally ignored.

Measured result tables can be regenerated without editing numbers by passing artifact paths
to `scripts/build_readme_tables.py`, for example:

```powershell
.\.venv\Scripts\python scripts\build_readme_tables.py `
  --static CV=outputs\mini\cv\metrics.json GRU=outputs\mini\gru_eval\metrics.json `
  --horizon CV=outputs\mini\cv\metrics.json `
  --online MINI=outputs\mini\online\online_metrics.json `
  --output outputs\README_TABLES.md
```

## Mini smoke results (not final benchmark claims)

The first metadata-only run completed on `v1.0-mini` with 47 accepted `mini_val` agents.
These numbers are a pipeline smoke test, not a substitute for the scene-level train /
train_val / val experiment:

| Model | ADE | FDE |
| --- | ---: | ---: |
| Finite-difference CV | 5.320 | 13.095 |
| CV + heading | 5.450 | 13.285 |
| GRU | 5.334 | 8.946 |
| Lightweight Transformer | 3.677 | 7.826 |

The causal online protocol on the same 47 agents reports rolling-row means:

| Strategy | ADE | FDE | Jitter | p95 latency (ms) |
| --- | ---: | ---: | ---: | ---: |
| Frozen forecast | 7.239 | 8.946 | 0.000 | 0.011 |
| Sliding CV | 2.203 | 5.005 | 1.019 | 0.522 |
| Residual corrected | 3.366 | 5.156 | 1.587 | 0.889 |

The complete generated tables, horizon breakdown, maneuver groups and provenance remain
under ignored `outputs/mini/README_TABLES.md`. Re-run the smoke command after any code or
configuration change instead of hand-editing these values.

## Trainval validation results

The metadata-only scene-level run used `train` for fitting, `train_val` for model selection,
and the locked `val` split for reporting. It contains 7,201 accepted validation agents;
sensor blobs were not downloaded. These are the first reproducible project results, not a
claim of state-of-the-art performance:

| Model | ADE | FDE | MissRate@2m |
| --- | ---: | ---: | ---: |
| Finite-difference CV | 4.668 | 11.280 | 0.910 |
| CV + heading | 4.562 | 11.104 | 0.904 |
| GRU | 3.766 | 8.950 | 0.914 |
| Lightweight Transformer | 3.761 | 8.918 | 0.916 |

At the 1-second horizon, the Transformer reaches ADE `0.440` and FDE `0.595`; at 3 seconds
it reaches ADE `1.386` and FDE `2.817`. The complete 1/3/6-second table is regenerated
from `outputs/trainval/README_TABLES.md`.

The supplementary causal online protocol reports 79,211 rolling update rows:

| Strategy | ADE | FDE | Jitter | p95 latency (ms) |
| --- | ---: | ---: | ---: | ---: |
| Frozen forecast | 6.297 | 8.950 | 0.000 | 0.008 |
| Sliding CV | 1.938 | 4.236 | 1.108 | 0.285 |
| Residual corrected | 2.239 | 4.113 | 1.554 | 0.517 |

The residual corrector improves the frozen endpoint error but does not beat the sliding
baseline on ADE; this mixed result is reported explicitly rather than overstated.

## Static evaluation outputs

Each evaluation directory contains `metrics.json`, `per_sample.csv`, `predictions.npz`,
and `official_predictions.json`. The latter is the repository's order-preserving
`instance_token_sample_token` mapping with one deterministic mode and probability 1.0.
nuScenes-devkit 1.2.0's `compute_metrics.py` consumes a list of dictionaries with
`instance`, `sample`, `prediction`, and `probabilities` fields; convert the mapping to
that devkit submission shape before calling the devkit evaluator. The distinction is
documented explicitly so a project artifact is not mistaken for a directly consumable
challenge submission.
