# nuScenes Trajectory Prediction Benchmark Design

## 1. Purpose and scope

This repository will provide a public, reproducible benchmark for deterministic single-agent trajectory prediction on nuScenes. It compares two physics baselines, a GRU and a lightweight Transformer under one controlled protocol, then explains their behavior through horizon-, maneuver- and failure-conditioned analysis.

The distinguishing project contribution is a causal online residual-correction experiment. It is supplementary to the official static nuScenes protocol and is reported separately.

Out of scope for the one-week MVP are image and LiDAR encoders, neural HD-map encoders, multimodal K-mode prediction, diffusion, flow matching and SOTA claims.

## 2. Experimental contract

### 2.1 Samples and splits

Each sample is keyed by the official `{instance_token}_{sample_token}` pair. Mini uses `mini_train` and `mini_val` only for smoke testing. Main results use:

- `train`: fitting.
- `train_val`: model selection, early stopping and configuration decisions.
- `val`: final locked evaluation.

Splits remain scene-level and come directly from the devkit. The data preparation command will assert that scene-token sets are pairwise disjoint. No random sample-level split is permitted.

### 2.2 Time and coordinates

The anchor time is `t=0`. The maximum history is two seconds at 2 Hz, with the current point explicitly included. The target contains 12 points from `t=0.5` through `t=6.0` seconds.

Raw annotation translations are global XY. Before training, the current translation is subtracted and points are rotated into the devkit-compatible agent frame where current yaw aligns with positive y. Targets and predictions remain in meters. Normalization statistics, if used, are fitted on the training split only and stored in cache metadata. Inverse transformation is mandatory for official-format export.

Required invariants:

- The current local point is numerically `[0, 0]`.
- Global-to-local-to-global round trip stays within a small floating-point tolerance.
- Time indices are strictly increasing after past-history reversal.
- Samples without the requested complete history and 12 future points are excluded consistently, with counts and reasons logged per split.

### 2.3 Features and targets

The common history representation contains local XY position, displacement, finite-difference velocity and relative heading. Acceleration may be derived from timestamps and velocity when valid. Missing derived values use an explicit validity mask rather than silently fabricated values.

The learned-model target is future local XY with shape `[12, 2]`. The physics baselines and learned models return the same shape and units.

## 3. Component boundaries

The repository will separate the following responsibilities:

- Data adapter: reads official tokens, annotations, timestamps and maps without touching sensor payloads.
- Geometry: owns coordinate transformations and motion-feature derivation.
- Cache builder: produces versioned, split-aware trajectory records and a manifest containing source versions, filtering counts and checksums.
- Models: expose one prediction interface from history tensors to `[batch, 12, 2]` trajectories.
- Training engine: owns optimization, logging, checkpoint selection and resume behavior.
- Evaluation: computes per-sample and aggregate deterministic metrics at 1, 3 and 6 seconds.
- Maneuver analysis: assigns transparent geometry-derived tags and aggregates metrics.
- Online evaluation: simulates causal observation arrival and evaluates update strategies.
- Visualization: consumes exported case records rather than re-running models.

Cached records and exported JSON form stable interfaces between these units. This prevents plotting, failure analysis or metric aggregation from depending on model internals.

## 4. Static model design

### 4.1 Finite-difference Constant Velocity

Estimate a two-dimensional velocity vector from the last two valid history positions and their timestamps, then linearly extrapolate all 12 future points. If only one valid observation remains after validation, emit a stationary forecast and record the fallback.

### 4.2 Constant Velocity + Heading

Reproduce the behavior and definition of the nuScenes devkit `ConstantVelocityHeading` baseline: use current annotated yaw and a scalar speed estimate to produce a heading-aligned velocity. Cite the official implementation in the README and do not present it as an original method.

### 4.3 GRU

Embed per-step features, encode the ordered history with a one- or two-layer GRU, and map the final valid hidden state to all 12 XY positions. Padding is masked; the final state must correspond to the last valid timestamp.

### 4.4 Transformer

Project features to 128 dimensions, add learned or sinusoidal temporal encoding, apply two four-head encoder layers, pool the current/last valid token and regress 24 output values reshaped to `[12, 2]`. The design target is well below one million trainable parameters unless measurements justify a small increase.

Both learned models initially use Smooth L1 loss on local XY. Training configuration, optimizer, batch size, scheduler, patience and seed are captured in versioned YAML files.

## 5. Evaluation and analysis

For predicted trajectory `p_t` and target `g_t` over `T` steps:

```text
ADE = (1/T) * sum_t ||p_t - g_t||_2
FDE = ||p_T - g_T||_2
```

For the deterministic setting these correspond to `minADE_1` and `minFDE_1`. MissRate uses the official two-meter, maximum-pointwise-error definition. Aggregate results include sample count, mean and bootstrap confidence intervals where runtime permits.

Horizon evaluation slices the same six-second prediction at 1, 3 and 6 seconds; it does not train separate models unless the experiment explicitly says so.

Maneuver tags derive from interpretable thresholds over displacement, speed, acceleration, yaw change and curvature. Tags may overlap for state categories such as acceleration and turning. Thresholds are fixed using training data statistics and recorded in configuration before final validation evaluation.

Per-sample export contains identifiers, history, target, each model prediction, ADE, FDE, speed, acceleration, yaw change, curvature and maneuver tags. Worst cases are ranked independently by ADE and FDE, deduplicated, then rendered in BEV.

## 6. Causal online correction design

### 6.1 Protocol

At anchor `t=0`, the base model emits a six-second forecast. A replay loop reveals the observation at `t=0.5`, updates the available two-second sliding history and evaluates the remaining trajectory. It repeats at 0.5-second intervals. Future observations beyond the update timestamp are inaccessible to every strategy.

Results from this protocol are not placed in the official static benchmark table because the information set differs.

### 6.2 Compared strategies

- Frozen forecast: trim the original trajectory without updating it.
- Reforecast: run the base model again on the latest causal history.
- Residual correction: adjust the remaining prior prediction with a small learned head.

The residual head receives the observed position error at the newly revealed step, velocity and heading residuals, normalized remaining time and the remaining prior trajectory. It outputs an XY offset sequence. A smoothness penalty discourages large changes between consecutive offsets. Training examples are generated only from training scenes; correction checkpoint selection uses `train_val`.

### 6.3 Online metrics

- Rolling ADE and rolling FDE over the remaining valid horizon.
- Absolute and relative improvement over frozen forecast.
- Endpoint improvement.
- Update jitter, measured as displacement between overlapping predictions before and after an update.
- p50 and p95 end-to-end update latency with batch size one on the declared hardware.

Metrics are reported overall and by update time and maneuver. Failed or unavailable updates must be counted rather than dropped silently.

## 7. Reliability and error handling

Data preparation fails fast for absent metadata directories, incompatible dataset versions, absent prediction map files, non-monotonic timestamps or split overlap. Individual incomplete trajectories are filtered with explicit reason codes and summary counts.

Every cache includes a schema version, preprocessing configuration hash, devkit version and source split. Training refuses a cache whose schema or configuration does not match the run configuration. Evaluation records checkpoint hash, configuration and dataset-manifest hash in every result export.

Resume restores model, optimizer, scheduler, epoch, best metric and random-number-generator states. Atomic checkpoint replacement prevents interrupted writes from corrupting the last valid checkpoint.

## 8. Verification strategy

Unit tests cover coordinate round trips, time ordering, feature derivation, physics trajectories, ADE/FDE/MissRate definitions, horizon slicing, masking, model output shapes, maneuver rules and online causality.

Integration tests use synthetic trajectories to verify stationary, straight constant-speed, turning, acceleration and braking behavior. Each learned model must overfit a tiny deterministic batch before dataset-scale training. A mini end-to-end test covers preparation, one short training run, resume, evaluation, case export and rendering.

Reproducibility verification runs the same seeded mini experiment twice and compares cache hashes, sample ordering and metrics within documented numerical tolerance.

## 9. Deliverables and acceptance

The completed repository includes configuration-based commands for preparation, training, evaluation, maneuver analysis, online correction and rendering; deterministic seeds; resumable checkpoints; CSV/JSON metrics; failure records; README-ready figures; and documentation of data acquisition and limitations.

Acceptance requires:

- Four static models evaluated under one shared protocol.
- Scene-disjoint split assertions passing.
- Main ADE/FDE, horizon and maneuver tables generated from real runs.
- Causal online comparison with accuracy, jitter and latency results.
- At least 20 manually reviewed failures and 3–5 clean success/failure figures.
- A clean-environment reproduction of the documented core workflow.
- No raw data, large checkpoints, secrets, private paths or proprietary material tracked by Git.

The project is successful even when learned models do not dominate every baseline, provided the result is reproducible and the failure analysis explains the observed behavior.
