# B3b2b implementation plan

> Execute inline with superpowers:executing-plans; no implementation delegation.

**Goal:** reproduce 28 source-consistent outlier/NIS physics comparisons under the approved B3b2b protocol, retaining negative outcomes.

**Architecture:** immutable original Gaussian/clean owner proofs; independent source outlier channels; actual/fixed-dt NIS kernels; pristine GT frame; paired detection/recovery metrics; streaming output and replay plus saved-prediction rescore.

**Tech stack:** Python, NumPy float64, pytest, existing artifact/hash infrastructure.

## Task 1: source noise and observation frame

- [x] Add `tests/robustness/test_source_outlier.py` with independent canonical SHA256 reference, occurrence boundary, magnitude, channel/seed separation and global RNG preservation. Add `tests/evaluation/test_outlier_data.py` covering first-observation outlier, pristine world GT and future isolation.
- [x] Run these tests before adding modules; require missing-module failure.
- [x] Implement `robustness/source_outlier.py`: `packet_outlier(digest,seed)` and `outlier_table(packets,seed)`; exact integer 2**64//20 cutoff and independent direction channel. Implement `evaluation/outlier_data.py`: `outlier_frame(base,normals,outliers,condition)`; reuse Gaussian-only path exactly, add D0 origin for outlier condition.
- [x] Run `python -m pytest tests/robustness/test_source_outlier.py tests/evaluation/test_outlier_data.py -q`; require all pass, then commit.

## Task 2: NIS physics and independent scalar reference

- [x] Add `tests/robustness/test_realclock_nis.py`: no-alarm parent equality, causal prefix, cap/threshold/negative tolerance, irregular time, finite/PSD covariance and source validation.
- [x] Run RED before implementing `robustness/realclock_nis.py`; public `replay_nis(times,positions,strategy)` accepts only observed [N,16] clock/XY and one of two frozen strategies. Save all 14 update diagnostics and all states/covs, query slice 5..15. `nis_scale` enforces -1e-12 tolerance and strict threshold.
- [x] Add independent `evaluation/outlier_reference.py` scalar H/inverse/Joseph path; test batched vs scalar on acceleration, turning, first-two-point and consecutive spikes, actual and fixed dt at 1e-10.
- [x] Run focused tests; require pass and commit.

## Task 3: metrics

- [x] Add `tests/evaluation/test_outlier_metrics.py` with hand-counted TP/FP/FN/TN, empty denominator, overlapping events, interrupted recovery, invalid GT and late event cases; run RED.
- [x] Implement `evaluation/outlier_metrics.py`: diagnostic groups by role/log, 14-update confusion counts, source vs exposure denominators, three exposure flags/intersections, k0..3 recovery at 1s using valid mask, no-new-spike eligibility and window-first reduction. Event labels enter evaluator only.
- [x] Verify test counts/means exactly; commit.

## Task 4: owner proof, streaming pipeline and integration

- [x] Add `tests/evaluation/test_physics_outlier_pipeline.py` using existing tiny Gaussian parent fixture: 28 combinations, exact CV/KF parent regression, semantic replay, saved-P independent rescore, regenerated-hash tamper rejection for source tables/observations/diagnostics/metrics/aggregates/proofs, source changes and incomplete output.
- [x] Run RED. Implement `evaluation/outlier_sources.py` with original Gaussian-owner CLI proof, exact binding/flags/refs/runtime verification and nested clean owner lookup. Implement `evaluation/physics_outlier.py` and `scripts/run_realclock_physics_outlier.py` with `--parent`, `--parent-owner`, `--output-root`, `--verify-only`.
- [x] Stream clean once then two conditions × three seeds × four strategies, include per-seed and aggregate paired signed deltas. Independent rescore loads saved predictions and pristine GT and rebuilds score/paired/detection/recovery/aggregate groups.
- [x] Verify scalar samples selected by existing fixed hash; complete only after sources/refs pre/post checks, exact immutable files and completed-last write. Preserve original-owner proof, packet table, all diagnostic masks/times, context offsets and scores.
- [x] Run all new tests and affected baseline tests, Ruff and public repo check. Commit implementation only after success.

## Task 5: formal execution and acceptance

- [x] Use sibling shared venv with PYTHONPATH=src and OMP/MKL/OPENBLAS_NUM_THREADS=2. Build new `outputs/realclock-physics-outlier/b3b2b-v1` against untouched sibling Gaussian parent.
- [x] Independently invoke CLI `--verify-only`. Require semantic replay, scalar reference and saved-P rescore flags, all28 common masks/denominators, full parent CV/KF regression and immutable parent checks.
- [x] Run full pytest once; save full output. If failures, fix via failing regression before retrying.
- [x] Write aggregate results/limits into `docs/REALCLOCK_PHYSICS_OUTLIER_B3B2B_RESULTS.md`, update roadmap and approved-spec status. Include clean/Gaussian costs, outlier gains or failures, per-seed/tails, timing control, source unique-vs-exposure counts, diagnostics/recovery and test evidence.
- [x] Public scan and diff check; commit locally, leave all original owners unchanged, do not push. Report exact stage completion and remaining limits.
