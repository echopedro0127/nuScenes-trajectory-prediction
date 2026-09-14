# Reproducibility and Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the tested benchmark into a clean public GitHub project whose commands and quantitative claims are traceable to real artifacts.

**Architecture:** PowerShell entry points orchestrate experiments, provenance hashes connect configs/caches/checkpoints/results, README tables are generated from JSON, and CI tests all code paths that do not require licensed data.

**Tech Stack:** Python 3.12, PowerShell, pytest, Ruff, GitHub Actions, Markdown.

---

### Task 1: Add provenance and repository hygiene checks

**Files:**
- Create: `src/trajectory_prediction/utils/provenance.py`
- Create: `scripts/check_public_repo.py`
- Create: `tests/utils/test_provenance.py`

- [ ] **Step 1: Write a failing hash test**

~~~python
# tests/utils/test_provenance.py
from pathlib import Path
from trajectory_prediction.utils.provenance import sha256_file


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_text("trajectory", encoding="utf-8")
    assert sha256_file(path) == "d72e5fac65cda164657fab5cc773d91b65071f21b9e557f92be81bfbfcd9db64"
~~~

- [ ] **Step 2: Run and verify failure**

Run: `.\.venv\Scripts\python -m pytest tests/utils/test_provenance.py -v`

Expected: import fails for `utils.provenance`.

- [ ] **Step 3: Implement hashing**

~~~python
# src/trajectory_prediction/utils/provenance.py
import hashlib
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()
~~~

- [ ] **Step 4: Add public repository checks**

`check_public_repo.py` reads `git ls-files -z` and exits nonzero if a tracked path begins with `data/`, `outputs/` or `checkpoints/`; has a model/archive/secret suffix; exceeds 10 MiB; or contains a user-profile path, API-key assignment, bearer token or private-key header. It prints every violation or `public repository checks passed`.

- [ ] **Step 5: Test and commit**

~~~powershell
.\.venv\Scripts\python -m pytest tests/utils/test_provenance.py -v
.\.venv\Scripts\python scripts/check_public_repo.py
git add src/trajectory_prediction/utils/provenance.py scripts/check_public_repo.py tests/utils
git commit -m "chore: validate public repository hygiene"
~~~

Expected: test passes and the public check reports success.

### Task 2: Add reproducible orchestration

**Files:**
- Create: `scripts/run_smoke.ps1`
- Create: `scripts/run_main.ps1`
- Create: `Makefile`
- Create: `tests/test_commands.py`

- [ ] **Step 1: Write the command-surface test**

~~~python
# tests/test_commands.py
from pathlib import Path


def test_documented_entrypoints_exist() -> None:
    expected = ["prepare_data.py", "train.py", "evaluate.py", "analyze_maneuvers.py",
                "render_cases.py", "train_corrector.py", "evaluate_online.py",
                "check_public_repo.py"]
    assert all((Path("scripts") / name).is_file() for name in expected)
~~~

- [ ] **Step 2: Run the test**

Run: `.\.venv\Scripts\python -m pytest tests/test_commands.py -v`

Expected: pass because preceding plans created every entry point.

- [ ] **Step 3: Implement smoke and main orchestration**

`run_smoke.ps1` redirects uv caches, validates mini metadata/maps, then prepares data, performs one-epoch learned-model checks, evaluates four static models, analyzes maneuvers and renders cases. `run_main.ps1` requires trainval metadata and runs static experiments, three history lengths, corrector training, online evaluation, analysis and rendering. Each command checks `$LASTEXITCODE` and throws immediately on failure. `-SkipTraining` is accepted only when checkpoints and provenance sidecars exist.

- [ ] **Step 4: Add cross-platform aliases**

~~~makefile
.PHONY: test lint public-check
test:
	python -m pytest -v
lint:
	ruff check src scripts tests
public-check:
	python scripts/check_public_repo.py
~~~

- [ ] **Step 5: Verify and commit**

~~~powershell
.\.venv\Scripts\python -m pytest tests/test_commands.py -v
.\.venv\Scripts\ruff check src scripts tests
git add scripts/run_smoke.ps1 scripts/run_main.ps1 Makefile tests/test_commands.py
git commit -m "build: add reproducible experiment entry points"
~~~

Expected: tests and lint pass.

### Task 3: Add data-independent CI

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Create the workflow**

~~~yaml
name: ci
on: [push, pull_request]
jobs:
  test:
    runs-on: windows-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v5
        with: {python-version: "3.12"}
      - run: uv pip install --system -e ".[dev]"
      - run: uv pip install --system torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
      - run: uv pip install --system nuscenes-devkit==1.2.0
      - run: python -m pytest -v
      - run: ruff check src scripts tests
      - run: python scripts/check_public_repo.py
~~~

- [ ] **Step 2: Validate and commit**

~~~powershell
.\.venv\Scripts\python -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml', encoding='utf-8')); print('workflow YAML OK')"
.\.venv\Scripts\python -m pytest -v
git add .github/workflows/ci.yml
git commit -m "ci: test data-independent pipeline"
~~~

Expected: YAML parses and all tests pass.

### Task 4: Generate README tables from artifacts

**Files:**
- Create: `scripts/build_readme_tables.py`
- Create: `tests/test_readme_tables.py`
- Modify: `README.md`

- [ ] **Step 1: Write the formatting test**

~~~python
# tests/test_readme_tables.py
from scripts.build_readme_tables import markdown_row


def test_markdown_row_formats_measured_values() -> None:
    row = markdown_row("CV", {"sample_count": 10, "ade": 1.2345, "fde": 2.3456})
    assert row == "| CV | 10 | 1.234 | 2.346 |"
~~~

- [ ] **Step 2: Run and verify failure**

Run: `.\.venv\Scripts\python -m pytest tests/test_readme_tables.py -v`

Expected: import fails for `scripts.build_readme_tables`.

- [ ] **Step 3: Implement table formatting**

~~~python
# scripts/build_readme_tables.py
import json
from pathlib import Path


def markdown_row(name: str, metrics: dict) -> str:
    return f"| {name} | {metrics['sample_count']} | {metrics['ade']:.3f} | {metrics['fde']:.3f} |"


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
~~~

The CLI accepts named `MODEL=metrics.json` inputs, rejects missing/non-finite/unequal-sample inputs, and emits static, horizon, maneuver and online Markdown tables.

- [ ] **Step 4: Complete README and commit**

Populate Motivation, Task Definition, Dataset, Coordinate System, Models, Training, Metrics, Main Results, Horizon Analysis, Maneuver Analysis, Online Correction, Failure Analysis, Visualization, Reproduction and Limitations. Insert only generated tables and measured figures; label mini and online protocols explicitly.

~~~powershell
.\.venv\Scripts\python -m pytest tests/test_readme_tables.py -v
.\.venv\Scripts\python scripts/check_public_repo.py
git add scripts/build_readme_tables.py tests/test_readme_tables.py README.md
git commit -m "docs: publish measured benchmark results"
~~~

Expected: tests and public checks pass.

### Task 5: Perform clean reproduction and final audit

**Files:**
- Modify: `README.md`
- Modify: `docs/FAILURE_ANALYSIS.md`
- Create: `docs/REPRODUCIBILITY.md`

- [ ] **Step 1:** Record Git commit, Python/dependency versions, GPU, seed, exact commands and hashes for caches, checkpoints and results.
- [ ] **Step 2:** Create a fresh project-local environment and rerun the complete mini smoke workflow without reusing caches.
- [ ] **Step 3:** Recreate locked val static, maneuver and online results from licensed metadata and frozen checkpoints; require metric agreement within `1e-6`.
- [ ] **Step 4:** Run the release gate.

~~~powershell
.\.venv\Scripts\python -m pytest -v
.\.venv\Scripts\ruff check src scripts tests
.\.venv\Scripts\python scripts/check_public_repo.py
git diff --check
git status --short
~~~

- [ ] **Step 5:** Write two truthful resume bullets using measured values, exact split and model size without SOTA or academic-novelty claims.
- [ ] **Step 6:** Commit the final report.

~~~powershell
git add README.md docs/FAILURE_ANALYSIS.md docs/REPRODUCIBILITY.md
git commit -m "docs: finalize reproducible project report"
~~~

Expected: all gates pass, documented metrics reproduce, and the worktree is clean.
