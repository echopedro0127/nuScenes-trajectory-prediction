"""Metadata-only anchor index with explicit source-file provenance."""
import json
from pathlib import Path

import numpy as np

from trajectory_prediction.context.schema import validate_timestamp_us
from trajectory_prediction.data.nuscenes_adapter import quaternion_yaw
from trajectory_prediction.utils.provenance import sha256_file


class MetadataIndex:
    def __init__(self, dataroot: Path, version: str):
        self.root = Path(dataroot)
        self.version = version
        self.tables = {}
        self.source_hashes = {}
        for name in ("sample", "scene", "log", "sample_annotation"):
            path = self.root / version / f"{name}.json"
            records = json.loads(path.read_text(encoding="utf-8"))
            self.source_hashes[f"{version}/{name}.json"] = sha256_file(path)
            self.tables[name] = {record["token"]: record for record in records}
        self.annotations = {(r["instance_token"], r["sample_token"]): r
                            for r in self.tables["sample_annotation"].values()}

    def anchor(self, instance, sample, scene, xy, yaw):
        record = self.tables["sample"][sample]
        scene_record = self.tables["scene"][record["scene_token"]]
        annotation = self.annotations[(instance, sample)]
        if scene_record["name"] != scene:
            raise ValueError("anchor scene mismatch")
        if not np.allclose(annotation["translation"][:2], xy, atol=1e-3, rtol=0):
            raise ValueError("anchor position mismatch")
        delta = (quaternion_yaw(annotation["rotation"]) - yaw + np.pi) % (2 * np.pi) - np.pi
        if abs(delta) > 1e-4:
            raise ValueError("anchor heading mismatch")
        timestamp = record["timestamp"]
        validate_timestamp_us(timestamp)
        location = self.tables["log"][scene_record["log_token"]]["location"]
        return timestamp, location

    def provenance(self):
        hashes = dict(self.source_hashes)
        for path in sorted((self.root / "maps" / "expansion").glob("*.json")):
            hashes[f"maps/expansion/{path.name}"] = sha256_file(path)
        if len(hashes) == len(self.source_hashes):
            raise ValueError("no expansion maps found")
        return hashes
