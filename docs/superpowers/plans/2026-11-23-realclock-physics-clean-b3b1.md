# B3b1 Clean Real-dt Physics Implementation Plan

> **For agentic workers:** Execute inline using executing-plans and test-driven-development. User approved the 2026-11-18 B3b1 spec with “继续实验”. Only read-only review is delegated under requesting-code-review.

**Goal:** Run four causal CV/KF strategies on identical clean B3a queries and publish reproducible paired errors.

**Architecture:** Batched float64 physics states → shared source-GT/identity assembly and masked metrics → explicit original B3a proof → per-strategy immutable state/prediction shards and independent replay. Future target times enter only the evaluator after states are frozen.

**Tech Stack:** Existing Python/NumPy, standard JSON and provenance utilities, CPU two threads. No new dependencies, training, downloads, noise or holdout.

## T1 State machine

- [x] Create ignored `.worktrees/realclock-physics-clean`, branch feature/realclock-physics-clean, base 1fb9a01; main user files untouched.
- [ ] Add failing `tests/robustness/test_realclock_physics.py` for `replay_states(times[N,16], positions[N,16,2], strategy)` and implement `robustness/realclock_physics.py`.
- [ ] Cover irregular analytic CV, fixed-vs-actual equivalence at .5s, independent scalar KF, first-two initialization plus historical warmup, future-perturbation invariance, covariance symmetry/PSD, bad clock/dtype/nonfinite rejection. Only indices0..15 enter state replay; index16 never assimilated.

## T2 Sources and metrics

- [ ] Add RED tests then implement `evaluation/physics_clean_data.py`: load parent stable windows/clocks and original B0b world_xy by role/token; verify source identity/clock; subtract first arrived point float64. Save origin, shared full GT and parent mapping, never pass GT targets to physics state code.
- [ ] Implement evaluator and `evaluation/physics_clean_metrics.py` with per-window means, role/log groups, pooled-query P50/P95 and E>2 proportion, actual-lead diagnostics, explicit zero-denominator reasons, and same-mask paired differences. Save per-window results and per-query errors; invalid values are zero placeholders with mandatory valid mask.
- [ ] Verify hand-computable unequal-query windows, log macro versus query pooling, empty lead windows, paired sets and independent error reconstruction.

## T3 Immutable release

- [ ] Add `evaluation/physics_clean_sources.py`: verify original B3a CLI in explicit owner; bind all parent immutable artifacts, owner sources and refs, exact proof fields/payload; subsequent verify rechecks frozen refs and saved proof. Parent must be metadata-only B3a.
- [ ] Add `evaluation/physics_clean.py` and CLI `scripts/run_realclock_physics_clean.py`; source/runtime/protocol binding, directory nonoverwrite, completed last, failure marker. Publish shared.npz, windows.json, each strategy state/covariance/prediction/error shard, grouped/per-window metrics and paired deltas.
- [ ] Add integration RED→GREEN tests with frozen synthetic B3a owner and real assembly; verify all predictions/states/covariances/metrics from original observations plus a second metrics reconstruction from saved predictions/shared GT. Reject rehashed tampering, extra files, source/proof/policy changes and interrupted release.
- [x] Add `evaluation/physics_clean_reference.py`: fixed SHA256 sample of at most16 windows per role, independent scalar H/inverse Joseph replay, checked in both prepare/verify; publish scalar_reference.json. A finite common-bias regression test must reject completion.

## T4 Review, experiment and report

- [ ] Read-only review against approved spec, resolve evidenced issues with regression tests. Focused tests, Ruff, public-file scan, full pytest with workspace basetemp and diff check. Never relax protocol to favor results.
- [ ] Commit stable source before formal run. Parent is `../realclock-query-audit/outputs/realclock-query-audit/b3a-v1`, owner `../realclock-query-audit`; output new `outputs/realclock-physics-clean/b3b1-v1`.
- [ ] Verify-only independently replays all four strategies and recomputes scores; fixed-hash sample slow scalar CV/KF at pre-frozen atol/rtol1e-10, verify original source and all previous phase manifests. Assert 22231 windows, 244541 queries and 511854 valid pairs per strategy.
- [ ] Record actual fit/select results for all four strategies and signed improvements/regressions in `docs/REALCLOCK_PHYSICS_CLEAN_B3B1_RESULTS.md`; update roadmap/spec/plan. Commit locally, preserve branch, do not push.

Commands use sibling `../trajectory-mvp/.venv/Scripts/python.exe`, PYTHONPATH=src, OMP/MKL/OPENBLAS_NUM_THREADS=2 at process startup. The run binds requested environment, not an unobserved actual thread count; formal execution additionally records external threadpool_info before/after. Finish source edits before formal publication. The physical strategies have no training seeds; clean is ideal annotation-derived observation, not detection/tracking accuracy. B3b2 noise and learned-model adaptation are separate.
