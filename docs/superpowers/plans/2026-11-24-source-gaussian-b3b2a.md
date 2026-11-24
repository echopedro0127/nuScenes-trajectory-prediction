# B3b2a Source-keyed Gaussian Implementation Plan

> **For agentic workers:** Use executing-plans inline and test-driven-development. User approved the 2026-11-24 B3b2a spec with “继续”. Implementation stays local; only read-only code review may be delegated under requesting-code-review.

**Goal:** Compare four frozen CV/KF methods under two source-consistent Gaussian amplitudes and three noise seeds, preserving every shared GT target and negative result.

**Architecture:** Strict source manifest and keyed PCG64 normals → noisy first-observation frame with untouched GT → per-seed scoring and seed-then-window reductions → streaming immutable shards, saved-prediction rescore and scalar replay.

**Tech Stack:** Existing float64 Python/NumPy and provenance helpers; no new dependencies, training, downloads, delay/dropout or holdout.

## T1 RNG and channel separation

- [x] Create ignored `.worktrees/realclock-physics-gaussian`, branch feature/realclock-physics-gaussian, base1464a06.
- [x] RED→GREEN `tests/robustness/test_source_gaussian.py`, `robustness/source_gaussian.py`: normal_vector(source_digest,seed), strict source manifest ordering/keys, stable table and window token lookup. Freeze canonical JSON/SHA256/PCG64 and literal golden vector before scores.
- [x] RED→GREEN `tests/evaluation/test_gaussian_data.py`, `evaluation/gaussian_data.py`: noisy-frame(shared, point_normals, sigma), all16 history/query observations noisy, first noisy observation origin, clean GT shifted by origin only, sigma0 parent reproduction, causal prefix and cross-window shared noise.

## T2 Seed metrics

- [x] RED→GREEN `tests/evaluation/test_gaussian_metrics.py`, `evaluation/gaussian_metrics.py`: exact seeds2027/28/29 and identical masks/counts; explicit per-window seed means before window macro and log macro. Keep every per-seed tail/proportion and label seed-mean/std without pooled-quantile confusion.
- [x] Aggregate actual/fixed and KF/CV paired gains plus noisy-minus-clean extra error, signed and with matching GT. Store seed/window arrays, counts and null reasons.

## T3 Sources and streaming publication

- [x] Add `evaluation/gaussian_sources.py`: native original B3b1 verifier, strict proof schema/flags, frozen owner/source/ref coverage; read source_packets only from bound original B3a manifest.
- [x] Add `evaluation/physics_gaussian.py`, CLI `scripts/run_realclock_physics_gaussian.py`, tests `test_physics_gaussian_pipeline.py`. Generate Z once per seed, share across sigma/strategy/windows. Stream condition/seed/strategy files, do not retain24 covariance/prediction sets in RAM.
- [x] Save shared clean GT/clocks/mapping, source manifest/point indices, standard normals, noisy observations/origin offsets,24 state/prediction/error shards and per-seed/aggregate/paired/clean-degradation results. All immutable, output nonoverwrite, completed last, source checks before/after.
- [x] Verify re-generates all files from frozen keys/observations, scalar-checks fixed hash windows for all24 cases and independently rescores saved predictions/GT. Test common vector bias, noise/GT mutation, rehash tamper, source/proof/policy drift and interruption.
- [x] Extend saved-prediction rescore to all seed macro/paired/clean-degradation groups. Add sigma0 exact regression against frozen parent states/covariances/predictions/errors and publish clean_regression.json; do not count it as new noise repeats.

## T4 Review and formal experiment

- [x] Read-only spec/quality review, focused tests, full Ruff/pytest/public scan/diff; resolve findings before source freeze. Use workspace basetemp and OMP/MKL/OPENBLAS_NUM_THREADS=2.
- [x] Commit stable source; formal parent `../realclock-physics-clean/outputs/realclock-physics-clean/b3b1-v1`, owner `../realclock-physics-clean`, new output `outputs/realclock-physics-gaussian/b3b2a-v1`.
- [x] All24 combinations:22231 windows,511854 valid pairs per combination; all3 seeds retained. Verify independently, audit actual BLAS2 before/after and preserve nine prior frozen stages.
- [x] Report full fit/select noise/seed/strategy/lead results, Gaussian versus clean and all regressions in `docs/REALCLOCK_PHYSICS_GAUSSIAN_B3B2A_RESULTS.md`; update roadmap/spec/plan and commit locally, no push.

Runtime uses sibling `../trajectory-mvp/.venv/Scripts/python.exe`, PYTHONPATH=src. Source code stays unchanged after publication. Gaussian is synthetic position degradation; no claim about real detection, official challenge ADE/FDE, new blind tests or calibrated uncertainty. Outliers/NIS remain B3b2b.

Acceptance: source73070d2;1256 passed in954.45s(43 new). Formal identity6eca3e0cabfd216d00d007ac3f3673a52aedb52866344ef112d3c9595c073551;24 combinations,22231 windows/511854 valid pairs each. Independent replay/rescore,8448 sampled states/4224 KF covariances,96 RNG vectors and17736 global-frame predictions verified; nine prior stages preserved. Actual BLAS2 before/after. Full positive and negative results: ../../REALCLOCK_PHYSICS_GAUSSIAN_B3B2A_RESULTS.md.
