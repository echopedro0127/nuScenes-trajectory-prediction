from dataclasses import asdict

import numpy as np
import pytest
import torch

from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.context.maps import MapConfig
from trajectory_prediction.online.context_training import (
    load_map_checkpoint,
    scene_partition,
    shared_contract,
    train_models,
)


def test_scene_partition_is_deterministic_disjoint_and_rejects_overlap():
    train, selection = scene_partition(["c", "a", "b", "a"])
    assert train == ["a", "b"] and selection == ["c"]
    with pytest.raises(ValueError):
        scene_partition(["a"])


def synthetic():
    n = 3
    future = np.tile(np.arange(24, dtype=np.float32).reshape(1, 12, 2), (n, 1, 1))
    arrays = {"history_xy": np.zeros((n, 5, 2), np.float32), "future_xy": future,
              "scene_name": np.array(["a", "b", "c"])}
    context = {"features": np.zeros((n, 11, 1, 2, 4), np.float32),
               "mask": np.ones((n, 11, 1, 2), bool), "anchor_timestamp_us": np.zeros(n, np.int64)}
    contract = {"schema_version": 1, "dataset_version": "v1.0-mini", "config": asdict(MapConfig()),
                "source": {"map": "a" * 64}, "preprocessing_sha256": "b" * 64,
                "keys_sha256": "c" * 64, "trajectory_sha256": "d" * 64,
                "time_policy": "fixed_2hz_11_updates", "oracle": "target_position_annotations",
                "coordinate_policy": "fixed_anchor_forward_positive_y"}
    manifest = {"contract": {**contract, "split": "mini_train"}, "npz_sha256": "e" * 64}
    return arrays, context, manifest, future + 1


def test_short_training_checkpoint_provenance_and_disjoint_evaluation(tmp_path):
    torch.set_num_threads(1)
    arrays, context, manifest, base = synthetic()
    identity = {"model_name": "gru", "checkpoint_sha256": "f" * 64,
                "model_config_sha256": "a" * 64, "data_config_sha256": "b" * 64,
                "history_seconds": 2.0, "history_steps": 5}
    artifact = {"context_sha256": "e" * 64, "contract": manifest["contract"],
                "base": {"npz_sha256": "f" * 64, "keys_sha256": "a" * 64,
                         "metrics_sha256": "b" * 64, "model_identity": identity}}
    metadata = {"shared_contract": shared_contract(manifest["contract"]),
                "base_model": identity,
                "train_scenes": ["a", "b"], "selection_scenes": ["c"],
                "train_scenes_sha256": digest_json(["a", "b"]),
                "selection_scenes_sha256": digest_json(["c"]),
                "train_provenance": artifact,
                "selection_provenance": artifact}
    model = train_models(arrays, context, base, [0, 1], arrays, context, base, [2],
                         metadata, tmp_path, epochs=1, batch_size=16, hidden_dim=8,
                         device=torch.device("cpu"), seed=7, patience=2)
    assert model is not None
    restored, payload = load_map_checkpoint(tmp_path / "best.pt", manifest["contract"], ["heldout"])
    assert payload["metadata"]["seed"] == 7
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="overlap"):
        load_map_checkpoint(tmp_path / "best.pt", manifest["contract"], ["c"])
    with pytest.raises(ValueError, match="contract"):
        load_map_checkpoint(tmp_path / "best.pt", {**manifest["contract"], "source": {"map": "wrong"}}, ["heldout"])

    with pytest.raises(ValueError, match="base model"):
        load_map_checkpoint(tmp_path / "best.pt", manifest["contract"], ["heldout"],
                            expected_base_identity={"checkpoint_sha256": "different"})

    from copy import deepcopy
    mutations = [
        lambda m: m["train_provenance"]["base"].pop("keys_sha256"),
        lambda m: m["selection_provenance"].pop("contract"),
        lambda m: m["train_provenance"]["contract"].update(source={"map": "0" * 64}),
        lambda m: m["selection_provenance"]["contract"].update(split="mini_val"),
        lambda m: m["train_provenance"].update(context_sha256="not-a-hash"),
    ]
    for mutate in mutations:
        changed = deepcopy(payload)
        mutate(changed["metadata"])
        torch.save(changed, tmp_path / "invalid.pt")
        with pytest.raises(ValueError):
            load_map_checkpoint(tmp_path / "invalid.pt", manifest["contract"], ["heldout"])



def test_training_cli_rejects_mini_val_before_reading_data():
    from trajectory_prediction.online.context_training import main
    with pytest.raises(SystemExit):
        main(["--split", "mini_val", "--trajectory-cache", "missing", "--context-cache", "missing",
              "--base-predictions", "missing", "--dataroot", "missing", "--output-dir", "missing"])
