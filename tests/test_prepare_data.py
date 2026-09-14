import sys
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest

from scripts import prepare_data
from trajectory_prediction.config import data_config_sha256


def _config() -> dict:
    return {
        "version": "v1.0-mini",
        "dataroot": "data/nuscenes",
        "splits": ["mini_train", "mini_val"],
        "history_seconds": 2.0,
        "future_seconds": 6.0,
        "frequency_hz": 2,
        "cache_root": "data/cache/mini",
    }


def _install_fakes(
    monkeypatch: pytest.MonkeyPatch,
    extracted: dict[str, tuple[list[str], int]],
) -> tuple[dict, list[str], list[tuple[Path, dict, dict]]]:
    config = _config()
    extracted_splits: list[str] = []
    saved: list[tuple[Path, dict, dict]] = []

    monkeypatch.setattr(prepare_data, "load_yaml", lambda path: config)
    monkeypatch.setattr(prepare_data, "NuScenes", lambda **kwargs: object())

    def fake_extract(nusc, split: str, history_seconds: float):
        extracted_splits.append(split)
        scenes, accepted = extracted[split]
        return {"scene_name": np.asarray(scenes, dtype=np.str_)}, {
            "requested": max(accepted, 1),
            "accepted": accepted,
            "incomplete_history": 0,
            "incomplete_future": 0,
        }

    def fake_save(root: Path, arrays: dict, manifest: dict) -> None:
        saved.append((root, arrays, manifest))

    monkeypatch.setattr(prepare_data, "extract_split", fake_extract)
    monkeypatch.setattr(prepare_data, "save_cache", fake_save)
    monkeypatch.setattr(sys, "argv", ["prepare_data.py", "--config", "config.yaml"])
    return config, extracted_splits, saved


def test_prepare_rejects_all_filtered_split_before_any_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, extracted_splits, saved = _install_fakes(
        monkeypatch,
        {
            "mini_train": ([], 0),
            "mini_val": (["scene-val"], 1),
        },
    )

    with pytest.raises(ValueError, match="mini_train.*zero accepted"):
        prepare_data.main()

    assert extracted_splits == ["mini_train", "mini_val"]
    assert saved == []


def test_prepare_rejects_scene_overlap_before_any_save(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, extracted_splits, saved = _install_fakes(
        monkeypatch,
        {
            "mini_train": (["scene-shared"], 1),
            "mini_val": (["scene-shared"], 1),
        },
    )

    with pytest.raises(ValueError, match="scene overlap.*mini_train.*mini_val"):
        prepare_data.main()

    assert extracted_splits == ["mini_train", "mini_val"]
    assert saved == []


def test_prepare_saves_unique_splits_with_complete_provenance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, extracted_splits, saved = _install_fakes(
        monkeypatch,
        {
            "mini_train": (["scene-train"], 1),
            "mini_val": (["scene-val"], 1),
        },
    )

    prepare_data.main()

    assert extracted_splits == ["mini_train", "mini_val"]
    assert [root for root, _, _ in saved] == [
        Path("data/cache/mini/mini_train"),
        Path("data/cache/mini/mini_val"),
    ]
    for (_, _, manifest), split in zip(saved, config["splits"], strict=True):
        assert manifest == {
            "schema_version": 1,
            "split": split,
            "dataset_version": "v1.0-mini",
            "history_seconds": 2.0,
            "future_seconds": 6.0,
            "frequency_hz": 2,
            "filter_counts": {
                "requested": 1,
                "accepted": 1,
                "incomplete_history": 0,
                "incomplete_future": 0,
            },
            "preprocessing_config_sha256": data_config_sha256(config),
            "nuscenes_devkit_version": version("nuscenes-devkit"),
        }
