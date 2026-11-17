# B2b2a Event Boundary Corrections Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, with test-driven-development. User approved the design by saying “继续”; no further design permission is needed.

**Goal:** Publish strict token-bound declarations for existing needs_correction candidates, preserving original evidence and independently replaying every artifact.

**Architecture:** Pure correction semantics → explicit frozen B2b1 owner validation and references → immutable template/import packages with independent verification. Original owners remain read-only. No new events or prediction scores.

**Tech Stack:** Existing Python, csv/json, NumPy/runtime provenance; no added dependencies.

## T1 Workspace and baseline

- [x] Create `.worktrees/event-boundary-corrections`, branch `feature/event-boundary-corrections`, based on `7a2c9e6`; verify `.worktrees` ignored.
- [x] Run existing candidate-verdict tests as baseline using sibling trajectory-mvp environment.

## T2 Pure semantics

Files: `src/trajectory_prediction/evaluation/boundary_corrections.py`, `tests/evaluation/test_boundary_corrections.py`.

- [x] Write failing tests for `make_template(overlay_identity, review_identity, data, verdicts)`, `parse_corrections(text, baseline)`, `derive_corrections(overlay_identity, review_identity, data, verdicts, rows)`.
- [x] Run `python -m pytest tests/evaluation/test_boundary_corrections.py -q`; observe missing functionality.
- [x] Implement full eligible denominator (only needs_correction), locked fields, pending/propose declarations, exact columns, bounded Unicode CSV, local token clocks, unchanged censor flags, relation and overlap checks.
- [x] Verify valid boundary edits, censored originals, empty eligibility, original immutability, malformed/forged records, touching versus overlapping intervals, paired edits and preserved unresolved states.
- [x] Run focused tests to green, then commit pure semantics.

## T3 Frozen source and publication

Files: `evaluation/correction_sources.py`, `evaluation/correction_overlay.py`, `scripts/make_correction_template.py`, `scripts/import_boundary_corrections.py`, `tests/evaluation/test_correction_overlay_pipeline.py` (evaluation paths relative to `src/trajectory_prediction`).

- [x] Add failing integration tests for template → empty/proposed import → verify, source/copy/rehashed artifact tampering, source owner mismatch, template-as-parent rejection, mutable reading progress, and interrupted publication.
- [x] Native parent command: `[sys.executable, owner/'scripts/import_candidate_verdicts.py', '--output-root', parent, '--verify-only']`, with cwd=owner and PYTHONPATH=owner/src. Require frozen source hashes and exact static summary/identity/semantic proof. Only declared reading counts may vary.
- [x] Bind B2b1 static artifacts, completed marker, parent source files and full references; load review_data and atomic_verdicts only after validation. Never import the new clone's B2b1 verifier as substitute for old source identity.
- [x] Implement `make_template_pack(parent, owner, output)`, `import_overlay(parent, owner, annotations, output)`, `verify_pack(output, expected_mode=None)` with exact immutable file coverage, before/after checks, completed last, no overwrite. Verify reconstructs documents from source plus original CSV; compare bytes and binding digest.
- [x] CLI mirrors B2b1 (`--parent`, `--parent-owner`, `--annotations`, `--output-root`, `--verify-only`); template cannot accept annotations, import requires explicit CSV.
- [x] Run integration and semantic tests to green, then commit.

## T4 Verification and real acceptance

- [x] `python -m ruff check .`; `python -m pytest -q` (capture full log); `python scripts/check_public_repo.py`; `git diff --check`.
- [x] Build real template against `../candidate-verdicts/outputs/candidate-verdicts/empty-v1`, owner `../candidate-verdicts`. Copy CSV into distinct editable draft and immutable input snapshot.
- [x] Import the empty snapshot into a new immutable output, verify template and overlay separately. Expected eligible=0, no_eligible_corrections, proposed=0; do not fabricate human entries to exercise real data.
- [x] Recheck frozen parent sources/artifacts via bound references. Write `docs/EVENT_BOUNDARY_CORRECTIONS_B2B2A_RESULTS.md` and update roadmap/status. Record actual test/acceptance counts; no predicted gains.
- [x] Commit locally; leave branch intact; do not push.

All commands use `PYTHONPATH=src`, OMP/MKL threads=2 and `../trajectory-mvp/.venv/Scripts/python.exe`. Later source edits invalidate newly generated packages, so finish code before real publication. B2b2b new-event/relation editing and B3 query/GT/noise protocols remain separate stages.

Acceptance: source commit 632c3b2; 1117 passed in 502.05s; 48 new tests; native template and empty overlay independently replayed; eligible=proposed=0. See ../../EVENT_BOUNDARY_CORRECTIONS_B2B2A_RESULTS.md and outputs/boundary-corrections/b2b2a-acceptance-v1.json. Frozen sources remained unchanged after publication.
