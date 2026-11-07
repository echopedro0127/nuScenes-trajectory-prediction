from copy import deepcopy

import pytest
from test_independent_map_training import independent_fixture


def parent_fixture():
    bundle, metadata = independent_fixture()
    arrays = {"scene_name": bundle[0]["scene_name"][:2]}
    provenance = metadata["train_provenance"]
    return metadata, arrays, deepcopy({"contract": provenance["contract"], "npz_sha256": provenance["context_sha256"]}), deepcopy(provenance["base"])


def test_accepts_only_matching_independent_parent():
    from trajectory_prediction.context.independent_candidates import validate_parent
    metadata, arrays, manifest, base = parent_fixture()
    validate_parent(metadata, arrays, manifest, base, "independent_fit")


@pytest.mark.parametrize("kind", ["holdout", "base", "map", "scenes", "role"])
def test_rejects_wrong_parent_before_candidate_generation(kind):
    from trajectory_prediction.context.independent_candidates import validate_parent
    metadata, arrays, manifest, base = deepcopy(parent_fixture())
    role = "independent_fit"
    if kind == "holdout":
        role = "independent_holdout"
    elif kind == "base":
        base["model_identity"]["checkpoint_sha256"] = "0" * 64
    elif kind == "map":
        manifest["npz_sha256"] = "0" * 64
    elif kind == "scenes":
        arrays["scene_name"][0] = "c"
    else:
        manifest["contract"]["split"] = "train"
    with pytest.raises(ValueError):
        validate_parent(metadata, arrays, manifest, base, role)
