"""Checksum-bound NPY sidecars for bounded-memory access to existing context schemas."""
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np

from trajectory_prediction.context.cache import key_hash, validate_context
from trajectory_prediction.context.neighbor_cache import validate_neighbors
from trajectory_prediction.utils.provenance import sha256_file


def _validate(root, arrays, manifest):
    """Compare archive members without loading the archive into RAM."""
    if "npz_file" in manifest:
        name = manifest["npz_file"]
        if Path(name).name != name or sha256_file(root / name) != manifest["npz_sha256"]:
            raise ValueError("mmap archive checksum mismatch")
        with zipfile.ZipFile(root / name) as archive:
            if sorted(archive.namelist()) != sorted(name + ".npy" for name in arrays):
                raise ValueError("mmap archive array schema mismatch")
            for name in arrays:
                digest = hashlib.sha256()
                with archive.open(name + ".npy") as member:
                    while block := member.read(1 << 20):
                        digest.update(block)
                if digest.hexdigest() != sha256_file(root / (name + ".npy")):
                    raise ValueError("mmap array differs from bound archive")
    contract = manifest["contract"]
    if "config" in contract:
        validator = validate_neighbors if contract.get("modality") == "oracle_neighbors" else validate_context
        # Existing canonical validators check every view; each chunk stays bounded.
        n = len(arrays["instance_token"])
        if not n:
            raise ValueError("empty mmap schema")
        for start in range(0, n, 64):
            chunk = {name: value[start:start + 64] for name, value in arrays.items()}
            validator(chunk, {**contract, "keys_sha256": key_hash(chunk)})
    if "keys_sha256" in contract and key_hash(arrays) != contract["keys_sha256"]:
        raise ValueError("mmap key order mismatch")


def publish_sidecar(root, arrays):
    root = Path(root)
    entries = {}
    for name, array in arrays.items():
        if hasattr(array, "flush"):
            array.flush()
        path = root / (name + ".npy")
        entries[name] = {"file": path.name, "sha256": sha256_file(path),
                         "shape": list(array.shape), "dtype": array.dtype.str}
    archive = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    _validate(root, arrays, archive)
    manifest = {"schema_version": 1, "archive_manifest_sha256": sha256_file(root / "manifest.json"),
                "arrays": entries}
    temporary = root / ".mmap-manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(root / "mmap_manifest.json")


def open_sidecar(root, expected_contract):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    sidecar = json.loads((root / "mmap_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("contract") != expected_contract or sidecar.get("schema_version") != 1 or sidecar.get("archive_manifest_sha256") != sha256_file(root / "manifest.json"):
        raise ValueError("mmap source contract mismatch")
    if "npz_file" in manifest:
        name = manifest["npz_file"]
        if Path(name).name != name or sha256_file(root / name) != manifest["npz_sha256"]:
            raise ValueError("mmap archive checksum mismatch")
    arrays = {}
    for name, entry in sidecar["arrays"].items():
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or entry["file"] != name + ".npy":
            raise ValueError("invalid mmap array filename")
        path = root / entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("mmap array checksum mismatch")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(array.shape) != entry["shape"] or array.dtype.str != entry["dtype"]:
            raise ValueError("mmap shape/dtype mismatch")
        arrays[name] = array
    if "keys_sha256" in expected_contract and key_hash(arrays) != expected_contract["keys_sha256"]:
        raise ValueError("mmap key order mismatch")
    _validate(root, arrays, manifest)
    return arrays
