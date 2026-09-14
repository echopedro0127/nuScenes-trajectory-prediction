# Failure and maneuver analysis

The analysis CLI joins an evaluation artifact with the matching deterministic cache. It
does not access camera, LiDAR, or radar payloads and does not alter the official static
metrics.

## Run

```powershell
.\.venv\Scripts\python scripts\analyze_maneuvers.py `
  --data-config configs\data\mini.yaml `
  --split mini_val `
  --evaluation-dir outputs\mini\gru_eval `
  --output-dir outputs\mini\analysis `
  --top-k 15
```

The same command works with `configs/data/trainval.yaml` and the locked `val` split after
metadata-only caches are prepared. The evaluator checks the cache configuration hash and
joins rows by `instance_token`/`sample_token`; silently mismatched predictions are rejected.

## Outputs

- `per_sample_analysis.csv`: ADE, FDE, miss flag, kinematic descriptors, and overlapping
  maneuver tags for every matched sample;
- `maneuver_metrics.csv`: sample counts and mean errors for each tag; one trajectory can
  contribute to more than one tag;
- `worst_cases.json`: deterministic union of the top-ADE and top-FDE cases, with the
  trajectory identifiers and descriptors needed for manual review.

Render the first 20 cases as agent-frame PNGs with a verified best checkpoint:

```powershell
.\.venv\Scripts\python scripts\render_cases.py `
  --data-config configs\data\trainval.yaml `
  --split val `
  --evaluation-dir outputs\trainval\gru_eval `
  --model GRU --best --limit 20 `
  --output-dir outputs\trainval\figures
```

Each image uses blue history, green ground truth, and a dashed model forecast. The output
directory also contains `render_manifest.json`, including the cache hash and the list of
rendered files. No map or sensor payload is copied into the figures.

Tags are interpretable threshold groups (`straight`, `turn`, `high_curvature`, `low_speed`,
`acceleration`, and `deceleration`). They are not mutually exclusive classes. Thresholds
live in `configs/analysis/maneuvers.yaml` and must be versioned with the result.

## Manual review protocol

For a final report, inspect 2030 cases from `worst_cases.json` and record the dominant
cause using a small fixed vocabulary: missing/short history, stationary or near-stationary
agent, sharp turn, acceleration change, interaction ambiguity, coordinate/data issue, or
model failure. Keep those annotations outside the raw dataset and publish only aggregate
counts or anonymized plots.

The online-correction protocol is reported separately. Use
`online_per_update.csv`, `online_metrics.json`, and `online_maneuver_metrics.csv` from
`scripts/evaluate_online.py`; never combine rolling online rows with the official static
ADE/FDE table.
