# nuScenes Trajectory Prediction

This project studies physics baselines, lightweight learned models, causal online
correction, and motion-conditioned analysis under the nuScenes prediction protocol.

## Status

The metadata-only pipeline is implemented end to end: official scene-level splits,
physics baselines, GRU/Transformer predictors, causal online correction, maneuver/failure
analysis, and reproducible trajectory figures. It contains no placeholder metrics or
state-of-the-art claims.

The current static benchmark compares finite-difference constant velocity,
constant-velocity-plus-heading, GRU, and lightweight Transformer predictors. The
supplementary contribution is causal online trajectory correction: reforecasting only with
observations available at each 0.5-second update, and measuring correction gain, jitter,
and latency separately from the official static protocol.

## Latest progress — 2026-11-25

The current real-clock track studies **causal online trajectory prediction and robust
correction on nuScenes**. It uses metadata-derived XY observations, shared source-keyed
synthetic perturbations, actual update timestamps, and pristine ground truth. The older
static and online benchmark tables below use separate protocols.

| Completed comparison | Select development results at 1 s |
| --- | --- |
| Clean CV, fixed versus actual assimilation dt | 0.711013 → 0.633016 m; 10.97% lower error |
| Gaussian 0.5 m, actual CV versus KF | 2.400904 → 1.468641 m; 38.83% lower error |
| Gaussian 1.0 m, actual CV versus KF | 4.596162 → 2.256338 m; 50.91% lower error |
| Gaussian 0.5 m + sparse 5 m outliers, KF versus NIS-KF | 1.846843 → 1.684352 m; 8.80% lower error |

These are window-macro position errors, not official challenge ADE/FDE or new holdout
scores. Noisy rows average three observation-noise seeds. Each combination shares
22,231 windows and 511,854 valid target comparisons. Outlier P95 falls 28.76%, while
NIS slightly worsens clean/Gaussian mean error by 0.47%/0.54% and can worsen the first
update after an outlier. These tradeoffs are retained in the reports.

The latest code passed **1,289 tests**, independent semantic replay, saved-prediction
rescoring and scalar-reference checks. Recovery diagnostics are complete; the proposed
21 new recovery runs are **designed but not implemented or evaluated**. Historical mini
Radar experiments are documented separately and are not inputs to these real-clock
results.

- [Current project progress and improvement summary](docs/PROJECT_PROGRESS_2026_11_25.md)
- [Clean real-clock results](docs/REALCLOCK_PHYSICS_CLEAN_B3B1_RESULTS.md)
- [Source-keyed Gaussian results](docs/REALCLOCK_PHYSICS_GAUSSIAN_B3B2A_RESULTS.md)
- [Outlier and NIS results](docs/REALCLOCK_PHYSICS_OUTLIER_B3B2B_RESULTS.md)
- [Recovery diagnostics](docs/REALCLOCK_RECOVERY_FORENSICS_B3B2C.md)
- [Proposed causal recovery experiment](docs/superpowers/specs/2026-11-25-causal-innovation-recovery-b3b2d-design.md)

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
| Frozen forecast | 7.239 | 8.946 | 0.000 | 0.008 |
| Sliding CV | 2.203 | 5.005 | 1.019 | 0.374 |
| Residual corrected | 5.182 | 7.611 | 4.175 | 0.685 |

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
| Frozen forecast | 6.297 | 8.950 | 0.000 | 0.032 |
| Sliding CV | 1.938 | 4.236 | 1.108 | 1.318 |
| Residual corrected | 2.204 | 4.074 | 1.549 | 2.430 |

The residual corrector is trained on `train`, selected on `train_val`, and evaluated only
on the locked `val` split. It improves frozen endpoint error but does not beat the sliding
baseline on ADE; this mixed result is reported explicitly rather than overstated. The mini
smoke run has no third split, so it selects on `mini_train` and remains a pipeline check.

## Failure analysis and trajectory figures

`run_main.ps1` also writes maneuver summaries and the 20 highest-error agent-frame
figures under `outputs/trainval/analysis` and `outputs/trainval/figures`. Each PNG shows
blue observed history, green ground truth, and the dashed model forecast; the accompanying
`render_manifest.json` records the cache hash and verified checkpoint. Outputs stay ignored
so the licensed nuScenes metadata and generated artifacts are never committed; rerun the
documented command in [FAILURE_ANALYSIS.md](docs/FAILURE_ANALYSIS.md) to reproduce them.

## Static evaluation outputs

Each evaluation directory contains `metrics.json`, `per_sample.csv`, `predictions.npz`,
and `official_predictions.json`. The latter is the repository's order-preserving
`instance_token_sample_token` mapping with one deterministic mode and probability 1.0.
nuScenes-devkit 1.2.0's `compute_metrics.py` consumes a list of dictionaries with
`instance`, `sample`, `prediction`, and `probabilities` fields; convert the mapping to
that devkit submission shape before calling the devkit evaluator. The distinction is
documented explicitly so a project artifact is not mistaken for a directly consumable
challenge submission.
