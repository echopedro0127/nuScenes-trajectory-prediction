from copy import deepcopy

import numpy as np
import pytest
import torch

from trajectory_prediction.context.gate_preparation import generate_candidates
from trajectory_prediction.online.corrector import ResidualCorrector
from trajectory_prediction.online.gate_dataset import GateDataset, load_gate_cache, save_gate_cache
from trajectory_prediction.online.interaction_corrector import InteractionCorrector


def fixture():
    torch.set_num_threads(1)
    n = 2
    future = np.tile(np.arange(24, dtype=np.float32).reshape(1, 12, 2) / 10, (n, 1, 1))
    arrays = {"history_xy": np.zeros((n, 5, 2), np.float32), "future_xy": future,
        "instance_token": np.array(["i0", "i1"]), "sample_token": np.array(["s0", "s1"]), "scene_name": np.array(["a", "b"])}
    maps = {"anchor_timestamp_us": np.full(n, 2000000, np.int64), "timestamp_us": np.tile(2000000 + np.arange(1, 12, dtype=np.int64) * 500000, (n, 1)),
        "features": np.ones((n, 11, 1, 2, 4), np.float32), "mask": np.ones((n, 11, 1, 2), bool)}
    original = ResidualCorrector(hidden_dim=8)
    model = InteractionCorrector(deepcopy(original), "map", 8)
    torch.nn.init.constant_(model.head[-1].weight, .1)
    return arrays, maps, future + 1, original.eval(), model.eval()


def test_packet_candidates_never_use_future_packets_and_cache_checks_inputs(tmp_path):
    arrays, maps, base, original, model = fixture()
    values = generate_candidates(arrays, maps, base, original, model, batch_size=7)
    assert values["map"].shape == (6, 2, 11, 12, 2)
    assert not values["valid"][4].any()
    np.testing.assert_array_equal(values["map"][4], values["original"])
    np.testing.assert_array_equal(values["source_us"][2, :, :2], 0)
    maps["features"][:, 3:] += 1000
    changed = generate_candidates(arrays, maps, base, original, model, batch_size=7)
    np.testing.assert_allclose(changed["map"][:, :, :3], values["map"][:, :, :3], atol=0, rtol=0)
    contract = {"gate_config": {"max_age_s": 2., "fixed_scale": .5, "decay_s": 1.}, "split": "train", "seed": 2027}
    save_gate_cache(tmp_path, values, contract)
    loaded, manifest = load_gate_cache(tmp_path, contract)
    data = GateDataset(loaded)
    assert data.size == 132
    inputs = data.inputs(np.array([0, 44, 88]), torch.device("cpu"))
    assert set(inputs) == {"cv", "map", "quality", "valid", "mask", "age_s"}
    assert data.labels(np.array([0]), torch.device("cpu")).shape == (1, 12, 2)
    with pytest.raises(ValueError, match="contract"):
        load_gate_cache(tmp_path, {**contract, "seed": 2028})
    for value in loaded.values():
        if isinstance(value, np.memmap): value._mmap.close()
    with (tmp_path / "quality.npy").open("ab") as stream: stream.write(b"bad")
    with pytest.raises(ValueError, match="checksum"):
        load_gate_cache(tmp_path, contract)
    assert manifest["contract"]["seed"] == 2027


def test_schema_rebuilds_quality_and_validity(tmp_path):
    values = generate_candidates(*fixture(), batch_size=7)
    contract = {"gate_config": {"max_age_s": 2., "fixed_scale": .5, "decay_s": 1.}, "split": "train", "seed": 2027}
    values["quality"][0, 0, 0, 7] += 100
    with pytest.raises(ValueError, match="quality"):
        save_gate_cache(tmp_path, values, contract)


def test_full_source_reference_set_and_cv_reconstruction_are_required(tmp_path):
    from trajectory_prediction.context.gate_preparation import verify_references
    values = generate_candidates(*fixture(), batch_size=7)
    contract = {"gate_config": {"max_age_s": 2., "fixed_scale": .5, "decay_s": 1.}, "split": "train", "seed": 2027}
    values["cv"][:] += 1000
    from trajectory_prediction.online.quality_gate import quality_features
    i, k = np.repeat(np.arange(2), 11), np.tile(np.arange(11), 2)
    for c in range(6):
        values["quality"][c] = quality_features(values["position"][i, k], values["velocity"][i, k], values["previous_velocity"][i, k],
            values["base"][i], k + 1, values["coverage"][c, i, k], values["age_s"][c, i, k], values["cv"][i, k], values["map"][c, i, k]).reshape(2, 11, 8)
    with pytest.raises(ValueError, match="CV"):
        save_gate_cache(tmp_path, values, contract)
    with pytest.raises(ValueError, match="references"):
        verify_references({"references": {}, "map_contract": {"source": {"metadata.json": "a" * 64}}})


def test_frozen_candidate_audit_rejects_self_consistent_future_and_shifted_timeline():
    from trajectory_prediction.context.gate_preparation import audit_candidates
    from trajectory_prediction.online.quality_gate import quality_features
    arrays, maps, base, original, model = fixture()
    values = generate_candidates(arrays, maps, base, original, model, batch_size=7)
    audit_candidates(arrays, maps, base, original, model, values, batch_size=7)
    i, k = np.repeat(np.arange(2), 11), np.tile(np.arange(11), 2)
    for c in range(6):
        valid = values["valid"][c]
        values["map"][c][valid] = np.broadcast_to(values["target"][:, None], (2, 11, 12, 2))[valid]
        values["quality"][c] = quality_features(values["position"][i,k], values["velocity"][i,k], values["previous_velocity"][i,k],
            base[i], k+1, values["coverage"][c,i,k], values["age_s"][c,i,k], values["cv"][i,k], values["map"][c,i,k]).reshape(2,11,8)
    with pytest.raises(ValueError, match="candidate reconstruction"):
        audit_candidates(arrays, maps, base, original, model, values, batch_size=7)
    values["anchor_timestamp_us"] += 1000000000
    with pytest.raises(ValueError, match="parent timeline"):
        audit_candidates(arrays, maps, base, original, model, values, batch_size=7)
