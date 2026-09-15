from __future__ import annotations

import pytest

from scripts.train_corrector import build_parser, validate_split_names
from trajectory_prediction.online.evaluator import validate_corrector_metadata


def test_corrector_parser_requires_independent_split_inputs() -> None:
    args = build_parser().parse_args(
        [
            "--train-split",
            "train",
            "--val-split",
            "train_val",
            "--train-base-predictions",
            "train.npz",
            "--val-base-predictions",
            "train_val.npz",
        ]
    )

    assert args.train_split == "train"
    assert args.val_split == "train_val"
    assert args.train_base_predictions.name == "train.npz"
    assert args.val_base_predictions.name == "train_val.npz"


def test_corrector_rejects_equal_split_roles() -> None:
    with pytest.raises(ValueError, match="different"):
        validate_split_names("val", "val")


def test_corrector_allows_explicit_train_selection_for_smoke() -> None:
    validate_split_names("mini_train", "mini_train", allow_same=True)


def test_online_evaluator_rejects_corrector_fit_on_eval_split() -> None:
    metadata = {
        "data_config_sha256": "config",
        "train_split": "train",
        "val_split": "val",
    }

    with pytest.raises(ValueError, match="evaluation split"):
        validate_corrector_metadata(metadata, "config", "val")


def test_online_evaluator_accepts_disjoint_corrector_splits() -> None:
    metadata = {
        "data_config_sha256": "config",
        "train_split": "train",
        "val_split": "train_val",
    }

    validate_corrector_metadata(metadata, "config", "val")


def test_online_evaluator_accepts_explicit_same_train_selection() -> None:
    metadata = {
        "data_config_sha256": "config",
        "train_split": "mini_train",
        "val_split": "mini_train",
        "same_split_validation": True,
    }

    validate_corrector_metadata(metadata, "config", "mini_val")
