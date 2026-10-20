"""Checksum-bound NPY sidecars for bounded-memory access to existing context schemas."""
import json
import re
from pathlib import Path

import numpy as np

from trajectory_prediction.context.cache import key_hash
from trajectory_prediction.utils.provenance import sha256_file


def publish_sidecar(root, arrays):
    root = Path(root)
    entries = {}
    for name, array in arrays.items():
        if hasattr(array, "flush"):
            array.flush()
        path = root / (name + ".npy")
        entries[name] = {"file": path.name, "sha256": sha256_file(path),
                         "shape": list(array.shape), "dtype": array.dtype.str}
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
    return arrays
