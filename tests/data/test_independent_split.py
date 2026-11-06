import json

import numpy as np
import pytest
import torch

from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.data import independent_split as phase
from trajectory_prediction.data.cache import load_cache, save_cache
from trajectory_prediction.utils.provenance import sha256_file


def records():
    return [{"name": f"scene-{i:02d}", "log_token": f"log-{i // 2}"} for i in range(14)]


def arrays():
    n = 14
    return {
        "history_xy": np.zeros((n, 5, 2), np.float32),
        "history_features": np.zeros((n, 5, 6), np.float32),
        "history_mask": np.ones((n, 5), bool),
        "future_xy": np.broadcast_to(np.arange(n, dtype=np.float32)[:, None, None], (n, 12, 2)).copy(),
        "anchor_xy": np.zeros((n, 2), np.float32), "anchor_yaw": np.zeros(n, np.float32),
        "current_speed": np.ones(n, np.float32),
        "instance_token": np.array([f"instance-{i}" for i in range(n)]),
        "sample_token": np.array([f"sample-{i}" for i in range(n)]),
        "scene_name": np.array([row["name"] for row in records()]),
    }


def inputs(tmp_path):
    source = tmp_path / "source"
    save_cache(source, arrays(), {"schema_version": 1, "split": "train",
                                  "dataset_version": "v1.0-trainval",
                                  "preprocessing_config_sha256": "a" * 64})
    data = tmp_path / "nuscenes"
    (data / "v1.0-trainval").mkdir(parents=True)
    (data / "v1.0-trainval" / "scene.json").write_text(json.dumps(records()))
    checkpoint = tmp_path / "old.pt"
    metadata = {"train_split": "train", "val_split": "train_val", "history_steps": 5,
                "data_config_sha256": "a" * 64, "model_config_sha256": "b" * 64}
    torch.save({"checkpoint_schema": 1, "metadata": metadata, "model": {}}, checkpoint)
    contract = {"split": "train", "base_hashes": {"model_identity": {
        "model_name": "gru", "checkpoint_sha256": sha256_file(checkpoint),
        "model_config_sha256": "b" * 64, "data_config_sha256": "a" * 64,
        "history_seconds": 2., "history_steps": 5}}}
    gate = tmp_path / "gate.json"
    gate.write_text(json.dumps({"schema_version": 1, "contract": contract,
                                "contract_sha256": digest_json(contract)}))
    return source, data, checkpoint, gate


def test_log_partition_is_order_independent_and_keeps_adjacent_scenes_together():
    scene_records = records()
    names = [row["name"] for row in scene_records]
    chosen = phase.partition_scene_groups(names, scene_records)
    assert chosen == phase.partition_scene_groups(list(reversed(names)), list(reversed(scene_records)))
    assert set(chosen) == set(phase.ROLES)
    assert sorted(name for group in chosen.values() for name in group) == sorted(names)
    for i in range(0, 14, 2):
        assert next(k for k, v in chosen.items() if names[i] in v) == next(k for k, v in chosen.items() if names[i + 1] in v)


def test_unknown_scene_conflicting_mapping_and_too_few_logs_are_rejected():
    with pytest.raises(ValueError):
        phase.partition_scene_groups(["missing"], records())
    with pytest.raises(ValueError):
        phase.partition_scene_groups(["scene-00"], records() + [{"name": "scene-00", "log_token": "different"}])
    with pytest.raises(ValueError):
        phase.partition_scene_groups(["scene-00"], records())


def test_real_cache_roundtrip_config_and_old_upstream_exposure(tmp_path):
    source, data, checkpoint, gate = inputs(tmp_path)
    output = tmp_path / "new"
    report = phase.prepare(source, data, checkpoint, gate, output)
    verified = phase.verify_prepared(output)
    assert verified == report and not report["holdout_evaluated"]
    assert report["old_upstream_reusable"] is False
    assert report["old_gru_exposed_holdout_scenes"] == report["roles"]["independent_holdout"]["scene_count"]
    assert sum(row["sample_count"] for row in report["roles"].values()) == 14
    original = arrays()
    for role in phase.ROLES:
        cached, manifest = load_cache(output / "cache" / role, role)
        indices = np.flatnonzero(np.isin(original["scene_name"], report["partitions"][role]))
        for key in cached:
            np.testing.assert_array_equal(cached[key], original[key][indices])
        assert manifest["split"] == role
    with pytest.raises(FileExistsError):
        phase.prepare(source, data, checkpoint, gate, output)


@pytest.mark.parametrize("field,value", [("train_split", "other"), ("data_config_sha256", "x" * 64)])
def test_old_checkpoint_cannot_claim_incorrect_training_source(tmp_path, field, value):
    source, data, checkpoint, gate = inputs(tmp_path)
    payload = torch.load(checkpoint, weights_only=True)
    payload["metadata"][field] = value
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="upstream"):
        phase.prepare(source, data, checkpoint, gate, tmp_path / "bad")


def test_duplicate_keys_and_changed_output_are_rejected(tmp_path):
    source, data, checkpoint, gate = inputs(tmp_path)
    duplicate = arrays()
    duplicate["instance_token"][1] = duplicate["instance_token"][0]
    duplicate["sample_token"][1] = duplicate["sample_token"][0]
    save_cache(source, duplicate, {"schema_version": 1, "split": "train",
               "dataset_version": "v1.0-trainval", "preprocessing_config_sha256": "a" * 64})
    with pytest.raises(ValueError, match="duplicate"):
        phase.prepare(source, data, checkpoint, gate, tmp_path / "bad")
    save_cache(source, arrays(), {"schema_version": 1, "split": "train",
               "dataset_version": "v1.0-trainval", "preprocessing_config_sha256": "a" * 64})
    root = tmp_path / "good"
    phase.prepare(source, data, checkpoint, gate, root)
    (root / "partitions.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        phase.verify_prepared(root)


def test_rehashed_report_cannot_claim_old_upstream_is_independent(tmp_path):
    source, data, checkpoint, gate = inputs(tmp_path)
    root = tmp_path / "new"
    phase.prepare(source, data, checkpoint, gate, root)
    report = json.loads((root / "report.json").read_text())
    report["old_upstream_reusable"] = True
    report["old_gru_exposed_holdout_scenes"] = 0
    (root / "report.json").write_text(json.dumps(report))
    marker = json.loads((root / "completed.json").read_text())
    marker["sha256"]["report.json"] = sha256_file(root / "report.json")
    (root / "completed.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="exposure"):
        phase.verify_prepared(root)
