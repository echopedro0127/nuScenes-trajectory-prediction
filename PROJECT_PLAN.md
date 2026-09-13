# Trajectory Prediction on nuScenes — Project Plan

## Project objective

Build a public, reproducible and interview-ready trajectory-prediction benchmark that answers:

> How much can lightweight learned trajectory models improve over simple physics-based motion baselines, where do they fail, and can causal online correction reduce those failures?

The project prioritizes controlled experiments and complete engineering over SOTA model complexity. It uses only public nuScenes data and public documentation. Company data, code, configuration, checkpoints and proprietary preprocessing are explicitly out of scope.

## A. Environment review

Environment inspected on 2026-09-13:

| Item | Detected | Decision |
| --- | --- | --- |
| OS | Windows 11 Home, 64-bit, build 26200 | Use native Windows for the one-week MVP |
| CPU | Intel i7-12700H, 14 cores / 20 threads | Sufficient for metadata preprocessing |
| RAM | 15.6 GiB | Sufficient with cached, compact trajectory arrays |
| GPU | RTX 3060 Laptop, 6 GiB VRAM | Sufficient for GRU and 2-layer Transformer |
| NVIDIA driver | 566.07; reports CUDA 12.7 capability | Install a compatible CUDA-enabled PyTorch wheel |
| Python | 3.13.12 currently on PATH | Create an isolated Python 3.12 environment |
| Git | 2.47.1 | Available |
| uv | 0.10.12 | Use for environment management |
| FFmpeg | Installed | Available for optional media export |
| WSL | Feature present, no Linux distribution installed | Do not add WSL setup risk to the MVP |

Disk availability at review time:

- `C:` 2.7 GiB free — do not use for environments, caches or data.
- `D:` 102.8 GiB free.
- `E:` 322.3 GiB free — use for project, environment, cache, data and outputs.

The official devkit v1.2.0 is tested with Python 3.9 and 3.12 and recommends Python 3.12. Its current base requirements constrain NumPy below 2.0. See the [official installation guide](https://github.com/nutonomy/nuscenes-devkit/blob/master/docs/installation.md) and [base requirements](https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/setup/requirements/requirements_base.txt).

## B. nuScenes task review

The project follows the official prediction protocol:

- At most 2 seconds of observed history.
- 6 seconds of future prediction.
- 2 Hz sampling, or one point every 0.5 seconds.
- 12 future XY points at `+0.5, +1.0, ..., +6.0` seconds.
- Agents identified by `instance_token` and `sample_token`.
- Official submission predictions expressed in the global coordinate frame.
- Internally, trajectories represented in an agent-centric frame with the current agent at the origin and its heading aligned with positive y, matching the devkit convention.
- Past records returned newest-first by `PredictHelper` are reversed before model input.
- Local predictions are inverse-transformed to global coordinates for official-format evaluation and export.

The standard full-history input contains four past positions plus the current anchor. History sensitivity uses `0.5 s`, `1.0 s` and `2.0 s`; `3.0 s` is excluded because it violates the official two-second history limit.

Primary deterministic metrics are ADE and FDE, equivalent to `minADE_1` and `minFDE_1`. `MissRate_2_1` is secondary. Off-road Rate is a map-dependent diagnostic. A deterministic trajectory is never duplicated to manufacture K-mode metrics. See the [official task definition](https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/python-sdk/nuscenes/eval/prediction/README.md) and [official evaluation configuration](https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/python-sdk/nuscenes/eval/prediction/configs/predict_2020_icra.json).

## C. Dataset download recommendation

Use a two-stage, metadata-first workflow:

### Stage 1: smoke test

Download:

- `v1.0-mini.tgz`.
- `nuScenes-map-expansion-v1.3.zip`.

Use the official `mini_train` and `mini_val` split to validate parsing, coordinate transforms, baselines, model training, evaluation and visualization. The mini split contains only 8 training scenes and 2 validation scenes, so its metrics must be labelled as smoke-test results and must not support final resume claims.

### Stage 2: main experiments

Download:

- `v1.0-trainval_meta.tgz`.
- Reuse the map expansion.

Do not download the ten `v1.0-trainvalXX_blobs.tgz` archives. This project reads annotations and map metadata, not camera, LiDAR or radar payloads. This reduces the required download from roughly 293 GB to under 1 GB of compressed metadata and maps.

The official prediction split is scene-level. Use `train` (500 scenes) for fitting, `train_val` (200 scenes) for checkpoint selection and configuration decisions, and `val` (150 scenes) once for the final report. The official implementation is documented in [prediction splits](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/splits.py) and [scene splits](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/splits.py).

## D. Revised MVP scope

### Models

1. Finite-difference Constant Velocity: estimate the 2D velocity vector from the latest valid observations and extrapolate it.
2. Official-style Constant Velocity + Heading: use current annotation yaw and devkit-derived scalar velocity. Clearly attribute the definition to the nuScenes devkit.
3. GRU predictor: one or two recurrent layers, hidden dimension 64 or 128, direct regression of 12 local XY positions.
4. Lightweight Transformer: linear feature embedding, temporal encoding, two encoder layers, hidden dimension 128, four attention heads and a 12-point regression head.

All learned models use the same inputs, targets, splits and evaluation code. Smooth L1 is the initial training loss; ADE and FDE remain evaluation metrics.

### Controlled experiments

- Four-model comparison under one fixed protocol.
- History sensitivity at 0.5, 1.0 and 2.0 seconds.
- Horizon metrics at 1, 3 and 6 seconds.
- Motion-conditioned metrics for straight, turn, high curvature, low speed, acceleration and deceleration.
- Intersection/non-intersection analysis only if the simple map query is reliable; it does not block MVP completion.
- Automatic top-ADE and top-FDE case export, followed by manual review of 20–30 cases.

### Project contribution: causal online trajectory correction

Treat online correction as a supplementary protocol, separate from the official static benchmark:

1. Generate an initial six-second trajectory.
2. Simulate streaming by revealing one new ground-truth observation every 0.5 seconds.
3. At each update, expose only observations available at that timestamp.
4. Compare a frozen initial forecast, sliding-window reforecast and a lightweight residual correction head.
5. Condition the correction head on position residual, velocity residual, heading residual, remaining horizon and the remaining original trajectory.
6. Predict offsets for the remaining trajectory and regularize temporal smoothness to avoid discontinuous updates.

Report rolling ADE/FDE, endpoint improvement, correction gain, prediction jitter, and p50/p95 inference latency. Break correction results down by maneuver. Do not call the method academically novel without a literature review; describe it as a project contribution.

## E. Expected resource usage

| Component | Disk | System RAM | GPU VRAM |
| --- | ---: | ---: | ---: |
| Python environment and CUDA PyTorch | 4–8 GiB | — | — |
| mini archive plus extraction peak | 8–10 GiB | 1–3 GiB | — |
| map expansion | About 0.4 GiB compressed; about 1 GiB extracted | Query-dependent | — |
| trainval metadata only | About 0.46 GiB compressed; about 1–3 GiB extracted | About 4–8 GiB while loaded | — |
| Processed trajectory cache | 0.1–1 GiB | 2–6 GiB during creation | — |
| GRU or Transformer training | Checkpoints normally under 100 MiB | 1–4 GiB | Target below 2 GiB |
| Tables, case JSON and figures | Normally under 1 GiB | Low | — |

All generated paths must live under `E:`. Environment setup will explicitly redirect uv and package caches away from the nearly full `C:` drive.

## F. Day-by-day plan

### Day 1 — Environment and data contract

- Create project-local Python 3.12 environment and cache locations.
- Pin dependencies and initialize public repository structure.
- Download mini and map expansion only.
- Load the official mini prediction split.
- Render one history/current/future trajectory.
- Test local/global coordinate round trips and timestamp alignment.

Acceptance: one official prediction agent produces a valid 2-second history and 12-point future; coordinate round-trip numerical error is negligible.

### Day 2 — Data pipeline and physics baselines

- Build deterministic preprocessing and compact cached samples.
- Log exclusion counts for missing or incomplete trajectories.
- Implement the two physics baselines.
- Implement ADE, FDE and MissRate.
- Export mini baseline metrics and BEV cases.

### Day 3 — GRU baseline

- Implement dataset and data loader.
- Implement GRU and Smooth L1 training.
- Add seed control, checkpointing, resume, best-model selection and CSV/JSON logging.
- Verify the model can overfit a tiny synthetic batch before full training.

### Day 4 — Lightweight Transformer

- Implement temporal encoding, masks, encoder and trajectory head.
- Train under the same protocol as the GRU.
- Produce the four-model mini smoke-test table.

### Day 5 — Main metadata-only experiments

- Import trainval metadata without sensor blobs.
- Verify scene disjointness across `train`, `train_val` and `val`.
- Run the controlled model comparison.
- Run history and prediction-horizon analyses.

### Day 6 — Online correction and failure analysis

- Implement causal streaming replay and the three online strategies.
- Measure correction accuracy, jitter and latency.
- Group errors by maneuver.
- Export worst cases and 3–5 clean success and failure figures.
- Manually annotate 20–30 failures.

### Day 7 — Reproduction and GitHub handoff

- Rerun core experiments from documented commands.
- Complete README, result tables, limitations and data-license instructions.
- Verify no data, large checkpoints, secrets, caches or absolute paths are tracked.
- Write resume bullets using measured values only.

## Success criteria

The MVP is complete when a clean environment can reproduce data preparation, four-model evaluation, horizon and maneuver tables, online-correction results, failure exports and README figures through documented commands. A Transformer win is not required; a defensible negative or mixed result is acceptable.

Optional map encoders, multimodal forecasting, camera context, diffusion and flow matching remain outside the one-week MVP.
