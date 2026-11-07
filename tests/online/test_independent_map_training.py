from copy import deepcopy

import pytest
import torch
from test_interaction_training import fixture
from test_scaled_training import subset


def independent_fixture():
    bundle, metadata, _, _ = fixture()
    metadata = deepcopy(metadata)
    for field, role in (("train_provenance", "independent_fit"),
                        ("selection_provenance", "independent_select")):
        metadata[field] = deepcopy(metadata[field])
        metadata[field]["contract"]["split"] = role
    metadata["partition_identity"] = "e" * 64
    metadata["role_logs"] = {"independent_fit": ["fit-log"], "independent_select": ["select-log"]}
    metadata["upstream_training"] = {"train_split": "independent_fit", "val_split": "independent_select",
        "data_config_sha256": "d" * 64, "history_steps": 5}
    return bundle, metadata


def test_accepts_real_independent_roles_without_rewriting_recorded_metadata():
    from trajectory_prediction.online.independent_map_training import validate_metadata
    _, metadata = independent_fixture()
    before = deepcopy(metadata)
    validate_metadata(metadata)
    assert metadata == before


@pytest.mark.parametrize("mutation", [
    lambda m: m["train_provenance"]["contract"].update(split="train"),
    lambda m: m["role_logs"].update(independent_select=["fit-log"]),
    lambda m: m["upstream_training"].update(train_split="train"),
    lambda m: m["upstream_training"].update(data_config_sha256="0" * 64),
    lambda m: m.update(partition_identity="invalid"),
    lambda m: m["selection_provenance"]["base"]["model_identity"].update(checkpoint_sha256="0" * 64),
])
def test_rejects_leaking_or_mismatched_sources(mutation):
    from trajectory_prediction.online.independent_map_training import validate_metadata
    _, metadata = independent_fixture()
    mutation(metadata)
    with pytest.raises(ValueError):
        validate_metadata(metadata)


def test_short_training_reload_and_overwrite_refusal(tmp_path):
    from trajectory_prediction.online.batched_context import BatchedContext
    from trajectory_prediction.online.independent_map_training import load_pair, train_pair
    torch.set_num_threads(1)
    bundle, metadata = independent_fixture()
    train = BatchedContext(*subset(bundle, [0, 1]))
    selection = BatchedContext(*subset(bundle, [2]))
    output = tmp_path / "run"
    models = train_pair(train, selection, metadata, output, epochs=1, hidden_dim=8, batch_size=16)
    restored, payload = load_pair(output / "best.pt", metadata)
    assert set(models) == {"original", "map"}
    assert payload["metadata"]["architecture"] == "independent_map_pair_v1"
    for name in models:
        for key, value in models[name].state_dict().items():
            torch.testing.assert_close(value, restored[name].state_dict()[key], rtol=0, atol=0)
    for key, value in restored["original"].state_dict().items():
        torch.testing.assert_close(value, restored["map"].fallback.state_dict()[key], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        train_pair(train, selection, metadata, output, epochs=1, hidden_dim=8, batch_size=16)
    bad = deepcopy(metadata)
    bad["partition_identity"] = "f" * 64
    with pytest.raises(ValueError, match="identity"):
        load_pair(output / "best.pt", bad)
    payload["model"]["map.fallback.network.0.weight"] += 1
    torch.save(payload, tmp_path / "bad.pt")
    with pytest.raises(ValueError, match="fallback"):
        load_pair(tmp_path / "bad.pt", metadata)
