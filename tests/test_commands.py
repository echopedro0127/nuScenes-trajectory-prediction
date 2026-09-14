from pathlib import Path


def test_documented_entrypoints_exist() -> None:
    expected = [
        "prepare_data.py",
        "train.py",
        "evaluate.py",
        "analyze_maneuvers.py",
        "train_corrector.py",
        "evaluate_online.py",
        "check_public_repo.py",
        "run_smoke.ps1",
        "run_main.ps1",
    ]
    assert all((Path("scripts") / name).is_file() for name in expected)


def test_orchestration_scripts_fail_fast_and_support_skip_training() -> None:
    for name in ("run_smoke.ps1", "run_main.ps1"):
        content = (Path("scripts") / name).read_text(encoding="utf-8")
        assert "$ErrorActionPreference = \"Stop\"" in content
        assert "SkipTraining" in content
