from pathlib import Path

import yaml


def test_ci_workflow_runs_data_independent_release_gates() -> None:
    workflow = yaml.safe_load(Path(".github/workflows/ci.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["test"]["steps"]
    commands = "\n".join(str(step.get("run", "")) for step in steps)
    assert "pytest" in commands
    assert "ruff.exe check src scripts tests" in commands
    assert "check_public_repo.py" in commands
