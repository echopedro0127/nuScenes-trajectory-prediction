from dataclasses import asdict

import pytest
from test_gate_preparation import fixture

from trajectory_prediction.context.independent_candidates import publish_candidates, sources
from trajectory_prediction.online.gate_dataset import load_gate_cache
from trajectory_prediction.online.quality_gate import GateConfig


def contract(role="independent_fit"):
    return {"split": role, "seed": 2027, "sources": sources(), "gate_config": asdict(GateConfig())}


def test_candidate_publisher_refuses_holdout_before_writing(tmp_path):
    arrays, maps, base, original, model = fixture()
    output = tmp_path / "holdout"
    with pytest.raises(ValueError, match="fit/select"):
        publish_candidates(output, arrays, maps, base, {"original": original, "map": model}, contract("independent_holdout"))
    assert not output.exists()


def test_candidate_publisher_audits_roundtrip_and_refuses_overwrite(tmp_path):
    arrays, maps, base, original, model = fixture()
    output = tmp_path / "cache"
    expected = contract()
    report = publish_candidates(output, arrays, maps, base, {"original": original, "map": model}, expected)
    data, _ = load_gate_cache(output, expected)
    assert report["condition_updates"] == 132
    assert not data["map"].flags.writeable
    with pytest.raises(FileExistsError):
        publish_candidates(output, arrays, maps, base, {"original": original, "map": model}, expected)
    for value in data.values():
        value._mmap.close()
    with (output / "map.npy").open("ab") as stream:
        stream.write(b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        load_gate_cache(output, expected)
