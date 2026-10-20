"""Native Radar timestamps, explicit filters, past poses, and auditable file reads."""
import json
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud

from trajectory_prediction.context.radar import RadarConfig, aggregate_returns, transform_returns
from trajectory_prediction.context.schema import FrameRef
from trajectory_prediction.utils.provenance import sha256_file

CHANNELS = ("RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT")
TABLES = ("sample", "scene", "sample_data", "sensor", "calibrated_sensor", "ego_pose")


class RadarIndex:
    def __init__(self, dataroot, version="v1.0-mini", config=None, decoded_cache=True):
        self.root, self.version, self.config = Path(dataroot), version, config or RadarConfig()
        self.decoded_cache, self.decoded = decoded_cache, OrderedDict()
        tables, self.source_hashes, self.used_files = {}, {}, {}
        for name in TABLES:
            path = self.root / version / (name + ".json")
            tables[name] = {row["token"]: row for row in json.loads(path.read_text(encoding="utf-8"))}
            self.source_hashes[version + "/" + name + ".json"] = sha256_file(path)
        self.samples, self.scenes, self.poses, self.calibrations = (tables[name] for name in ("sample", "scene", "ego_pose", "calibrated_sensor"))
        self.frames = {}
        for row in tables["sample_data"].values():
            calibration = self.calibrations[row["calibrated_sensor_token"]]
            sensor = tables["sensor"][calibration["sensor_token"]]
            if sensor["modality"] != "radar":
                continue
            channel = sensor["channel"]
            if channel not in CHANNELS:
                raise ValueError("unexpected radar channel")
            scene = self.scenes[self.samples[row["sample_token"]]["scene_token"]]["name"]
            self.frames.setdefault((scene, channel), []).append(row)
        self.times = {}
        for key, frames in self.frames.items():
            frames.sort(key=lambda row: (row["timestamp"], row["token"]))
            self.times[key] = np.array([row["timestamp"] for row in frames], np.int64)
        self.last_profile = {}

    def query(self, state, scene, anchor_xy, anchor_yaw):
        started = time.perf_counter_ns()
        points, sources, channels, refs = [], [], [], []
        read_ms, transform_ms = 0., 0.
        for channel_index, channel in enumerate(CHANNELS):
            key = scene, channel
            records = self.frames.get(key, [])
            end = int(np.searchsorted(self.times.get(key, []), state.timestamp_us, side="right"))
            for row in records[max(0, end - self.config.sweeps_per_channel):end][::-1]:
                timestamp = row["timestamp"]
                if state.timestamp_us - timestamp > self.config.window_us:
                    continue
                pose = self.poses[row["ego_pose_token"]]
                if pose["timestamp"] > timestamp:
                    raise ValueError("radar pose unavailable at source time")
                name = row["filename"]
                path = (self.root / name).resolve()
                if not path.is_relative_to(self.root.resolve()):
                    raise ValueError("radar path leaves dataset")
                before = time.perf_counter_ns()
                if self.decoded_cache and name in self.decoded:
                    cloud = self.decoded.pop(name)
                else:
                    cloud = RadarPointCloud.from_file(str(path), invalid_states=[0], dynprop_states=list(range(7)), ambig_states=[3])
                if self.decoded_cache:
                    self.decoded[name] = cloud
                    if len(self.decoded) > 256:
                        self.decoded.popitem(last=False)
                read_ms += (time.perf_counter_ns() - before) / 1e6
                if name not in self.used_files:
                    self.used_files[name] = sha256_file(path)
                before = time.perf_counter_ns()
                transformed = transform_returns(cloud.points, self.calibrations[row["calibrated_sensor_token"]], pose, anchor_xy, anchor_yaw)
                transform_ms += (time.perf_counter_ns() - before) / 1e6
                points.append(transformed)
                sources.append(np.full(len(transformed), timestamp, np.int64))
                channels.append(np.full(len(transformed), channel_index, np.int64))
                refs.append(FrameRef(row["token"], channel, timestamp, timestamp))
        before = time.perf_counter_ns()
        view = aggregate_returns(np.concatenate(points) if points else np.zeros((0, 7), np.float32),
            np.concatenate(sources) if sources else np.zeros(0, np.int64),
            np.concatenate(channels) if channels else np.zeros(0, np.int64), state.position, state.timestamp_us, self.config, refs)
        self.last_profile = {"read_decode_ms": read_ms, "transform_ms": transform_ms,
            "aggregate_ms": (time.perf_counter_ns() - before) / 1e6, "radar_total_ms": (time.perf_counter_ns() - started) / 1e6}
        return view
