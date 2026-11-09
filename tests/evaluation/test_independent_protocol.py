import json

import pytest


def test_protocol_roundtrip_tamper_and_overwrite_refusal(tmp_path):
    from trajectory_prediction.evaluation.independent_protocol import publish_lock, verify_lock
    root = tmp_path / "experiment"
    root.mkdir()
    (root / "model.pt").write_bytes(b"frozen test model")
    source = tmp_path / "source.py"
    source.write_text("# frozen test source\n")
    output = root / "protocol"
    marker = publish_lock(root, ["model.pt"], {"test_source": source}, {"primary": "RH"}, output)
    assert verify_lock(root, output)["identity"] == marker["identity"]
    with pytest.raises(FileExistsError):
        publish_lock(root, ["model.pt"], {"test_source": source}, {}, output)
    protocol = json.loads((output / "protocol.json").read_text())
    protocol["body"]["policy"]["primary"] = "adaptive"
    (output / "protocol.json").write_text(json.dumps(protocol))
    with pytest.raises(ValueError):
        verify_lock(root, output)


@pytest.mark.parametrize("changed", ["model", "source"])
def test_protocol_verifier_rejects_changed_parent(tmp_path, changed):
    from trajectory_prediction.evaluation.independent_protocol import publish_lock, verify_lock
    (tmp_path / "model.pt").write_bytes(b"model")
    source = tmp_path / "source.py"
    source.write_text("# source")
    output = tmp_path / "protocol"
    publish_lock(tmp_path, ["model.pt"], {"source": source}, {}, output)
    (tmp_path / "model.pt" if changed == "model" else source).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        verify_lock(tmp_path, output)


@pytest.mark.parametrize("name", ["../outside.pt", "nested/../../outside.pt"])
def test_protocol_rejects_reference_escape(tmp_path, name):
    from trajectory_prediction.evaluation.independent_protocol import publish_lock
    with pytest.raises(ValueError, match="within"):
        publish_lock(tmp_path, [name], {}, {}, tmp_path / "protocol")
    assert not (tmp_path / "protocol").exists()


def test_freeze_refuses_holdout_artifacts_before_reading_models(tmp_path):
    from trajectory_prediction.evaluation.independent_protocol import freeze
    (tmp_path / "predictions/independent_holdout").mkdir(parents=True)
    with pytest.raises(ValueError, match="holdout artifacts"):
        freeze(tmp_path, tmp_path / "protocol")
    assert not (tmp_path / "protocol").exists()


def test_all_selected_models_are_rechecked_even_without_extra_references(tmp_path):
    from trajectory_prediction.evaluation.independent_protocol import publish_lock, verify_lock
    from trajectory_prediction.utils.provenance import sha256_file
    model = tmp_path / "gru.pt"
    model.write_bytes(b"selected model")
    policy = {"models": {"gru": {"path": "gru.pt", "sha256": sha256_file(model)}}}
    output = tmp_path / "protocol"
    publish_lock(tmp_path, [], {}, policy, output)
    model.write_bytes(b"different weights")
    with pytest.raises(ValueError, match="checksum"):
        verify_lock(tmp_path, output)
