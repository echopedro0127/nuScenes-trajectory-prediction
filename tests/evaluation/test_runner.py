from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from trajectory_prediction.evaluation.runner import (
    _dispatch_physics,
    build_evaluation_parser,
    write_evaluation,
    write_official_predictions,
)


def _identifiers(count: int = 2) -> dict[str, np.ndarray]:
    return {
        "instance_token": np.asarray([f"i{index}" for index in range(count)]),
        "sample_token": np.asarray([f"s{index}" for index in range(count)]),
        "scene_name": np.asarray([f"scene-{index}" for index in range(count)]),
    }


def test_evaluation_writes_summary_rows_and_arrays(tmp_path: Path) -> None:
    prediction = np.zeros((2, 12, 2), dtype=np.float32)
    target = prediction.copy()
    target[1, :, 0] = 1.0

    write_evaluation(
        tmp_path,
        prediction,
        target,
        {
            "instance_token": np.asarray(["i1", "i2"]),
            "sample_token": np.asarray(["s1", "s2"]),
            "scene_name": np.asarray(["scene-a", "scene-b"]),
        },
        provenance={"split": "mini_val"},
    )

    summary = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert summary["sample_count"] == 2
    assert summary["ade"] == 0.5
    assert summary["fde"] == 0.5
    assert summary["miss_rate_2"] == 0.0
    assert set(summary["horizons"]) == {"1s", "3s", "6s"}
    assert summary["provenance"] == {"split": "mini_val"}

    rows = (tmp_path / "per_sample.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "instance_token,sample_token,scene_name,ade,fde,miss"
    assert rows[1].startswith("i1,s1,scene-a,0.0,0.0,0.0")
    assert rows[2].startswith("i2,s2,scene-b,1.0,1.0,0.0")

    with np.load(tmp_path / "predictions.npz") as arrays:
        np.testing.assert_array_equal(arrays["prediction"], prediction)
        np.testing.assert_array_equal(arrays["target"], target)


@pytest.mark.parametrize(
    "prediction,target",
    [
        (np.zeros((2, 11, 2), dtype=np.float32), np.zeros((2, 11, 2), dtype=np.float32)),
        (np.zeros((2, 12, 3), dtype=np.float32), np.zeros((2, 12, 3), dtype=np.float32)),
        (np.zeros((2, 12, 2), dtype=np.int32), np.zeros((2, 12, 2), dtype=np.float32)),
        (np.zeros((2, 12, 2), dtype=np.float32), np.zeros((3, 12, 2), dtype=np.float32)),
    ],
)
def test_evaluation_rejects_invalid_trajectory_arrays(
    tmp_path: Path, prediction: np.ndarray, target: np.ndarray
) -> None:
    with pytest.raises(ValueError, match=r"(\[N, 12, 2\]|floating-point)"):
        write_evaluation(tmp_path, prediction, target, _identifiers())


def test_evaluation_rejects_identifier_length_mismatch(tmp_path: Path) -> None:
    prediction = np.zeros((2, 12, 2), dtype=np.float32)
    identifiers = _identifiers()
    identifiers["sample_token"] = np.asarray(["only-one"])

    with pytest.raises(ValueError, match="identifier"):
        write_evaluation(tmp_path, prediction, prediction, identifiers)


def test_evaluation_reports_exact_horizon_metrics(tmp_path: Path) -> None:
    prediction = np.zeros((1, 12, 2), dtype=np.float32)
    prediction[0, :2, 0] = 1.0
    prediction[0, 2:6, 0] = 2.0
    prediction[0, 6:, 0] = 3.0

    write_evaluation(tmp_path, prediction, np.zeros_like(prediction), _identifiers(1))

    summary = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert summary["horizons"] == {
        "1s": {"ade": 1.0, "fde": 1.0, "miss_rate_2": 0.0},
        "3s": {"ade": 5.0 / 3.0, "fde": 2.0, "miss_rate_2": 0.0},
        "6s": {"ade": 7.0 / 3.0, "fde": 3.0, "miss_rate_2": 1.0},
    }


def test_official_predictions_preserve_order_and_format(tmp_path: Path) -> None:
    prediction = np.arange(48, dtype=np.float32).reshape(2, 12, 2)
    output = tmp_path / "official.json"

    write_official_predictions(output, prediction, _identifiers())

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert list(payload) == ["i0_s0", "i1_s1"]
    assert payload["i0_s0"]["prediction"] == prediction[0].tolist()
    assert payload["i0_s0"]["probabilities"] == [1.0]
    assert output.read_text(encoding="utf-8").startswith("{\n  \"i0_s0\"")


def test_official_predictions_reject_duplicate_keys(tmp_path: Path) -> None:
    identifiers = _identifiers()
    identifiers["instance_token"][1] = identifiers["instance_token"][0]
    identifiers["sample_token"][1] = identifiers["sample_token"][0]

    with pytest.raises(ValueError, match="duplicate"):
        write_official_predictions(
            tmp_path / "official.json",
            np.zeros((2, 12, 2), dtype=np.float32),
            identifiers,
        )


def test_physics_dispatch_uses_expected_coordinate_contract() -> None:
    arrays = {
        "history_xy": np.asarray(
            [[[0.0, -1.0], [0.0, 0.0]]], dtype=np.float32
        ),
        "current_speed": np.asarray([4.0], dtype=np.float32),
    }

    finite_difference = _dispatch_physics("finite_difference_cv", arrays, 2)
    heading = _dispatch_physics("constant_velocity_heading", arrays, 2)

    np.testing.assert_allclose(finite_difference[0, :2, 1], [1.0, 2.0])
    np.testing.assert_allclose(heading[0, :2, 1], [2.0, 4.0])


def test_physics_dispatch_rejects_unknown_model() -> None:
    with pytest.raises(ValueError, match="unknown physics model"):
        _dispatch_physics(
            "gru",
            {
                "history_xy": np.zeros((1, 5, 2), dtype=np.float32),
                "current_speed": np.zeros(1, dtype=np.float32),
            },
            5,
        )


def test_evaluation_parser_has_required_split_and_defaults() -> None:
    parser = build_evaluation_parser()
    args = parser.parse_args(["--split", "mini_val"])

    assert args.split == "mini_val"
    assert args.history_seconds == 2.0
    assert args.data_config == Path("configs/data/mini.yaml")
    assert args.output_dir == Path("outputs/eval")


def test_learned_model_requires_checkpoint() -> None:
    from trajectory_prediction.evaluation.runner import validate_checkpoint_requirement

    with pytest.raises(ValueError, match="checkpoint"):
        validate_checkpoint_requirement("gru", None)

    validate_checkpoint_requirement("finite_difference_cv", None)


def test_imported_torch_is_available_for_metrics_contract() -> None:
    assert torch.__version__
