# B3a Real-clock Query Audit Implementation Plan

> **For agentic workers:** Execute inline using superpowers:executing-plans and test-driven-development. User approved the B3a design with “继续”. Delegate only read-only code review when required by requesting-code-review.

**Goal:** Publish independently replayable query/GT mapping and rule-candidate coverage for all B0b windows, without inference or training.

**Architecture:** Pure integer-clock matcher → validated B0b/B1a assembly and coverage → explicit frozen-owner proofs and immutable artifact publication. Arrays are compressed NPZ with explicit status codes and source-index tables to avoid hundreds of thousands of redundant JSON objects. JSON manifests retain identities, windows, events and namespace keys.

**Tech Stack:** Existing Python/NumPy, csv/json and provenance helpers; no new packages.

## T1 Isolation and semantics

- [x] Create ignored `.worktrees/realclock-query-audit`, branch feature/realclock-query-audit, base fe8224c.
- [x] Create `evaluation/query_clock.py` and `tests/evaluation/test_query_clock.py` through RED→GREEN. API: `match_target(times, query, desired)` takes strict positive integer clocks and desired target times; returns index/status/offset; rejects malformed clocks, preserves horizon/tie/tolerance reasons. `window_queries(times)` returns [N,11] actual queries and [N,11,3] mapping arrays.
- [x] Test exact, inclusive 50000us tolerance, 1us outside, tie, no future, desired beyond horizon, strict int64 clocks, independent scalar reference, and nominal-grid conservation.

## T2 Assembly and coverage

- [x] Create `evaluation/query_audit_data.py`, tests `test_query_audit_data.py`: validate windows/track points/blocks/events; unique role/instance/sample IDs, row lookup, source-token/clock identity, segment continuity, finite coordinates and role/log separation.
- [x] Assemble actual query arrays, GT target source indices and reason masks; namespace source-packet keys exclude window/strategy/model/event, deduplicate overlapping windows. Fixed seeds are descriptions only; no RNG.
- [x] Event coverage joins same block and keeps all events even with zero windows; query-to-onset leads .5/1s inclusive 50ms, strict pre-onset; confirmation coverage independent; retain censoring and rule_candidate provenance. Test overlap-window dedup and complete denominators.

## T3 Provenance and publication

- [x] Create `evaluation/query_audit_sources.py`: explicit original B0b/B1a native commands, exact immutable coverage/source hashes/static proof payload; require B1a parent identity/path match the supplied B0b. Bind all upstream refs, native proofs, runtime and strict frozen protocol.
- [x] Create `evaluation/query_audit.py` and `scripts/run_realclock_query_audit.py`. APIs `prepare_audit(tracks, tracks_owner, events, events_owner, output)` and `verify_audit(output)`. Completed last, output nonoverwrite, before/after source/ref checks; no inference.
- [x] Create `tests/evaluation/test_query_audit_pipeline.py` with frozen synthetic owners and actual assembly. Test publish/replay, policy/rehashed arrays/doc tamper, source change, owner mismatch, extra files and interruption. Verify regenerates all NPZ and JSON content, not just checksum.

## T4 Review and acceptance

- [x] Read-only review against approved spec; fix evidenced issues with regression tests. Run focused tests, Ruff, full pytest using workspace basetemp, public scan and diff check.
- [x] Commit stable source locally before real run. Publish B3a using B0b `../scenario-tracks-full/outputs/scenario-tracks/b0b-v1` and B1a `../scenario-stop-events/outputs/scenario-events/b1a-v1` in their explicit owners, then independent verify-only.
- [x] Assert 22231 windows, 244541 queries and 733623 matching attempts. Record actual matched/reason counts, per-role/lead/log/event coverage, immutable input hashes and no prediction-score flags. Preserve all old worktrees.
- [x] Write `docs/REALCLOCK_QUERY_AUDIT_B3A_RESULTS.md`, update roadmap/spec/plan; commit locally, keep branch, no push.

Commands use `../trajectory-mvp/.venv/Scripts/python.exe`, PYTHONPATH=src, OMP/MKL=2. Complete source before publication; subsequent documentation changes do not alter source identity. Next B3b is actual-dt physical replay, separately designed after these coverage results.

Acceptance: source 6ccc8b0; 1168 passed in 505.99s (51 new); real full-cache run and independent verify passed, identity e52bac9e3312f2902d529c6c609a696b18c6e5aa62c37957e0a02892de501fa7. 64 actual windows / 2112 independent scalar checks passed; seven old stages preserved. See ../../REALCLOCK_QUERY_AUDIT_B3A_RESULTS.md.
