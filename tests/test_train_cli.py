import sys

import pytest

from scripts import train


def test_train_cli_rejects_identical_train_and_validation_splits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = {
        "version": "v1.0-mini",
        "dataroot": "data/nuscenes",
        "splits": ["mini_train", "mini_val"],
        "history_seconds": 2.0,
        "future_seconds": 6.0,
        "frequency_hz": 2,
        "cache_root": "data/cache/mini",
    }
    monkeypatch.setattr(train, "load_yaml", lambda path: config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--train-split",
            "mini_train",
            "--val-split",
            "mini_train",
        ],
    )

    with pytest.raises(ValueError, match="different"):
        train.main()
