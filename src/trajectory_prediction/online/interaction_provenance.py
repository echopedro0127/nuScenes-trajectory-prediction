"""Checkpoint contracts for the complete map/neighbor ablation suite."""
import re

import torch
from torch import nn

from trajectory_prediction.context.neighbors import CATEGORY_PREFIXES, NeighborConfig
from trajectory_prediction.online.context_training import shared_contract, validate_scene_metadata
from trajectory_prediction.online.corrector import ResidualCorrector
from trajectory_prediction.online.interaction_corrector import (
    MODES,
    InteractionCorrector,
    matched_width,
)

SUITE_NAMES = ("original", *MODES, *("matched_" + mode for mode in MODES))


def shared_neighbor_contract(contract):
    return {key: contract.get(key) for key in ("schema_version", "modality", "config", "source",
        "category_prefixes", "coordinate_policy", "time_policy", "velocity_policy", "oracle")}


def _hash(value):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("neighbor provenance requires complete source hashes")


def validate_interaction_metadata(metadata):
    validate_scene_metadata(metadata)
    for field, parent in (("train_neighbor_provenance", "train_provenance"),
                          ("selection_neighbor_provenance", "selection_provenance")):
        provenance = metadata.get(field, {})
        _hash(provenance.get("context_sha256"))
        contract = provenance.get("contract")
        if not isinstance(contract, dict) or shared_neighbor_contract(contract) != metadata.get("neighbor_shared_contract"):
            raise ValueError("neighbor shared source contract mismatch")
        if contract.get("schema_version") != 1 or contract.get("modality") != "oracle_neighbors" or contract.get("category_prefixes") != list(CATEGORY_PREFIXES):
            raise ValueError("neighbor schema/category policy mismatch")
        expected = metadata[parent]
        if contract.get("map_contract") != expected["contract"] or contract.get("map_npz_sha256") != expected["context_sha256"]:
            raise ValueError("neighbor/map provenance binding mismatch")
        for key in ("split", "keys_sha256", "trajectory_sha256"):
            if contract.get(key) != expected["contract"].get(key):
                raise ValueError("neighbor key/split/trajectory mismatch")
        source = contract.get("source")
        if not isinstance(source, dict) or not source:
            raise ValueError("neighbor source files missing")
        for key, value in expected["contract"]["source"].items():
            if source.get(key) != value:
                raise ValueError("neighbor/map metadata source mismatch")
        for value in source.values():
            _hash(value)
        NeighborConfig(**contract["config"])


def load_interaction_checkpoint(path, map_contract, neighbor_contract, evaluation_scenes, base_identity):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    if payload.get("checkpoint_schema") != 1 or metadata.get("architecture") != "interaction_suite_v1":
        raise ValueError("unsupported interaction checkpoint")
    validate_interaction_metadata(metadata)
    if metadata["shared_contract"] != shared_contract(map_contract):
        raise ValueError("checkpoint map contract mismatch")
    if metadata["neighbor_shared_contract"] != shared_neighbor_contract(neighbor_contract):
        raise ValueError("checkpoint neighbor contract mismatch")
    if metadata["base_model"] != base_identity:
        raise ValueError("checkpoint base model identity mismatch")
    if set(evaluation_scenes) & set(metadata["train_scenes"] + metadata["selection_scenes"]):
        raise ValueError("evaluation scene overlap with training/selection")
    hidden = metadata["hidden_dim"]
    models = nn.ModuleDict({"original": ResidualCorrector(hidden_dim=hidden)})
    for mode in MODES:
        models[mode] = InteractionCorrector(ResidualCorrector(hidden_dim=hidden), mode, hidden)
        width = matched_width(sum(p.numel() for p in models[mode].parameters()))
        models["matched_" + mode] = ResidualCorrector(hidden_dim=width)
    models.load_state_dict(payload["model"], strict=True)
    original = models["original"].state_dict()
    for mode in MODES:
        if any(not torch.equal(value, models[mode].fallback.state_dict()[key]) for key, value in original.items()):
            raise ValueError("context fallback does not match original control")
    return models.eval(), payload
