import json
from copy import deepcopy

import pytest
import torch
from test_gate_training import datasets


def metadata_fixture():
    shared = {"schema": "independent_candidates_v1", "seed": 2027, "partition_identity": "a" * 64,
              "map_training_identity": "b" * 64, "gate_config": {}, "conditions": [], "sources": {},
              "base_hashes": {"model_identity": {"checkpoint_sha256": "c" * 64}},
              "references": {"checkpoint": {"path": "new.pt", "sha256": "d" * 64}}}
    return {"partition_identity": "a" * 64,
            "candidate_contracts": {role: {**deepcopy(shared), "split": role} for role in ("independent_fit", "independent_select")},
            "candidate_manifest_sha256": {"independent_fit": "e" * 64, "independent_select": "f" * 64},
            "train_scenes": ["a"], "selection_scenes": ["b"],
            "role_logs": {"independent_fit": ["fit-log"], "independent_select": ["select-log"]}}


@pytest.mark.parametrize("kind", ["roles", "parent", "logs", "scenes", "manifest"])
def test_gate_rejects_mismatched_independent_inputs(kind):
    from trajectory_prediction.online.independent_gate_training import validate_inputs
    metadata = metadata_fixture()
    if kind == "roles":
        metadata["candidate_contracts"]["independent_select"]["split"] = "val"
    elif kind == "parent":
        metadata["candidate_contracts"]["independent_select"]["map_training_identity"] = "0" * 64
    elif kind == "logs":
        metadata["role_logs"]["independent_select"] = ["fit-log"]
    elif kind == "scenes":
        metadata["selection_scenes"] = ["a"]
    else:
        metadata["candidate_manifest_sha256"]["independent_fit"] = "bad"
    with pytest.raises(ValueError):
        validate_inputs(metadata)


def test_gate_short_training_reload_and_source_identity(tmp_path):
    from trajectory_prediction.online.independent_gate_training import (
        load_independent_gate,
        train_independent_gate,
    )
    torch.set_num_threads(1)
    train, selection = datasets()
    metadata = metadata_fixture()
    output = tmp_path / "new"
    model = train_independent_gate(train, selection, metadata, output, epochs=2, batch_size=32)
    restored, payload = load_independent_gate(output / "best.pt", metadata)
    assert payload["metadata"]["architecture"] == "independent_quality_gate_v1"
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
    assert (output / "completed.json").exists()
    diagnostics = json.loads((output / "report.json").read_text())["diagnostics"]
    assert diagnostics["missing"]["mean_gate_weight"] == 0
    assert diagnostics["missing"]["strategies"]["gate"] == diagnostics["missing"]["strategies"]["cv"]
    with pytest.raises(FileExistsError):
        train_independent_gate(train, selection, metadata, output, epochs=2, batch_size=32)
    bad = deepcopy(metadata)
    bad["candidate_manifest_sha256"]["independent_fit"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        load_independent_gate(output / "best.pt", bad)
    payload["metadata"]["independent_sources"] = {}
    torch.save(payload, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="source"):
        load_independent_gate(tmp_path / "bad.pt", metadata)
