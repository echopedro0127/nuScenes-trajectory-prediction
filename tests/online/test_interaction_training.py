from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest
import torch

from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.context.maps import MapConfig
from trajectory_prediction.context.neighbors import NeighborConfig
from trajectory_prediction.online.context_training import shared_contract
from trajectory_prediction.online.interaction_provenance import (
    load_interaction_checkpoint,
    shared_neighbor_contract,
)
from trajectory_prediction.online.interaction_training import train_suite


def fixture():
    n = 3
    future = np.tile(np.arange(24, dtype=np.float32).reshape(1, 12, 2), (n, 1, 1))
    arrays = {"history_xy": np.zeros((n, 5, 2), np.float32), "future_xy": future,
              "scene_name": np.array(["a", "b", "c"])}
    maps = {"features": np.zeros((n, 11, 1, 2, 4), np.float32),
            "mask": np.ones((n, 11, 1, 2), bool), "anchor_timestamp_us": np.zeros(n, np.int64)}
    neighbors = {"features": np.zeros((n, 11, 8, 5, 6), np.float32),
        "mask": np.zeros((n, 11, 8, 5), bool), "velocity_mask": np.zeros((n, 11, 8, 5), bool),
        "instance_tokens": np.full((n, 11, 8), "", dtype="U64"),
        "annotation_tokens": np.full((n, 11, 8, 5), "", dtype="U64"),
        "source_timestamp_us": np.zeros((n, 11, 8, 5), np.int64),
        "available_timestamp_us": np.zeros((n, 11, 8, 5), np.int64),
        "timestamp_us": np.tile(np.arange(1, 12, dtype=np.int64) * 500_000, (n, 1))}
    map_contract = {"schema_version": 1, "split": "mini_train", "dataset_version": "v1.0-mini",
        "config": asdict(MapConfig(max_lanes=1, points_per_lane=2)), "source": {"map": "a" * 64},
        "keys_sha256": "b" * 64, "trajectory_sha256": "c" * 64, "preprocessing_sha256": "d" * 64}
    neighbor_contract = {"schema_version": 1, "modality": "oracle_neighbors", "split": "mini_train",
        "config": asdict(NeighborConfig()), "source": {"map": "a" * 64, "category": "e" * 64},
        "keys_sha256": "b" * 64, "trajectory_sha256": "c" * 64,
        "map_contract": map_contract, "map_npz_sha256": "f" * 64,
        "category_prefixes": ["vehicle.", "human.pedestrian."]}
    identity = {"model_name": "gru", "checkpoint_sha256": "a" * 64, "model_config_sha256": "b" * 64,
                "data_config_sha256": "d" * 64, "history_seconds": 2., "history_steps": 5}
    artifact = {"context_sha256": "f" * 64, "contract": map_contract,
        "base": {"npz_sha256": "a" * 64, "keys_sha256": "b" * 64, "metrics_sha256": "c" * 64,
                 "model_identity": identity}}
    neighbor_artifact = {"context_sha256": "e" * 64, "contract": neighbor_contract}
    metadata = {"shared_contract": shared_contract(map_contract),
        "neighbor_shared_contract": shared_neighbor_contract(neighbor_contract), "base_model": identity,
        "train_scenes": ["a", "b"], "selection_scenes": ["c"],
        "train_scenes_sha256": digest_json(["a", "b"]), "selection_scenes_sha256": digest_json(["c"]),
        "train_provenance": artifact, "selection_provenance": artifact,
        "train_neighbor_provenance": neighbor_artifact, "selection_neighbor_provenance": neighbor_artifact}
    return (arrays, maps, neighbors, future + 1), metadata, map_contract, neighbor_contract


def test_short_ablation_training_restores_suite_and_rejects_invalid_provenance(tmp_path):
    torch.set_num_threads(1)
    bundle, metadata, map_contract, neighbor_contract = fixture()
    models = train_suite(bundle, [0, 1], bundle, [2], metadata, tmp_path,
                         epochs=1, batch_size=16, hidden_dim=8, seed=7)
    restored, payload = load_interaction_checkpoint(tmp_path / "best.pt", map_contract,
        neighbor_contract, ["heldout"], metadata["base_model"])
    assert all(0 < progress["best_epoch"] <= progress["completed_epochs"] == 1
               for progress in payload["metadata"]["stage_progress"].values())
    assert set(restored) == {"original", "map", "neighbors", "fusion", "matched_map", "matched_neighbors", "matched_fusion"}
    for name in models:
        for a, b in zip(models[name].parameters(), restored[name].parameters(), strict=True):
            torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="overlap"):
        load_interaction_checkpoint(tmp_path / "best.pt", map_contract, neighbor_contract, ["c"], metadata["base_model"])
    with pytest.raises(ValueError, match="base model"):
        load_interaction_checkpoint(tmp_path / "best.pt", map_contract, neighbor_contract, ["heldout"],
                                    {**metadata["base_model"], "checkpoint_sha256": "0" * 64})
    changed = deepcopy(neighbor_contract)
    changed["config"]["radius_m"] = 10
    with pytest.raises(ValueError, match="neighbor"):
        load_interaction_checkpoint(tmp_path / "best.pt", map_contract, changed, ["heldout"], metadata["base_model"])
    for mutation in (
        lambda m: m.pop("train_neighbor_provenance"),
        lambda m: m["selection_neighbor_provenance"]["contract"].update(map_npz_sha256="0" * 64),
        lambda m: m["train_neighbor_provenance"]["contract"].update(source={"map": "0" * 64}),
    ):
        changed = deepcopy(payload)
        mutation(changed["metadata"])
        torch.save(changed, tmp_path / "bad.pt")
        with pytest.raises(ValueError):
            load_interaction_checkpoint(tmp_path / "bad.pt", map_contract, neighbor_contract, ["heldout"], metadata["base_model"])
