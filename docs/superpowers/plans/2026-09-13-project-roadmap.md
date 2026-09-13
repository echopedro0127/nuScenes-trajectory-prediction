# nuScenes Trajectory Prediction Project Roadmap

Execute the plans in this order:

1. [Static Benchmark Implementation Plan](./2026-09-13-static-trajectory-benchmark.md) — environment, trajectory cache, physics baselines, GRU, Transformer, training and deterministic evaluation.
2. [Online Correction and Analysis Implementation Plan](./2026-09-13-online-correction-analysis.md) — maneuver grouping, causal replay, learned correction, failures and BEV rendering.
3. [Reproducibility and Release Implementation Plan](./2026-09-13-reproducibility-release.md) — end-to-end commands, clean rerun, documentation, result provenance and GitHub hygiene.

Each plan ends in a runnable, testable repository state. Do not start the next plan until the preceding plan's verification commands pass and its changes are committed.

The three plans jointly cover every requirement in [the approved design](../specs/2026-09-13-nuscenes-trajectory-prediction-design.md).
