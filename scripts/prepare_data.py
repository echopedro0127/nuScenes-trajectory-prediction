import argparse
from importlib.metadata import version as distribution_version
from pathlib import Path

from nuscenes import NuScenes

from trajectory_prediction.config import (
    data_config_sha256,
    load_yaml,
    validate_data_config,
)
from trajectory_prediction.data.cache import save_cache
from trajectory_prediction.data.nuscenes_adapter import extract_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = validate_data_config(load_yaml(args.config))
    config_sha256 = data_config_sha256(config)

    nusc = NuScenes(
        version=config["version"], dataroot=config["dataroot"], verbose=True
    )
    prepared: dict[str, tuple[dict, dict]] = {}
    for split in config["splits"]:
        prepared[split] = extract_split(
            nusc, split, history_seconds=config["history_seconds"]
        )

    for split, (_, counts) in prepared.items():
        if counts["accepted"] == 0:
            raise ValueError(f"split {split} has zero accepted trajectories")

    scene_owners: dict[str, str] = {}
    for split, (arrays, _) in prepared.items():
        scenes = set(arrays["scene_name"].tolist())
        overlap = scenes.intersection(scene_owners)
        if overlap:
            previous_split = scene_owners[next(iter(overlap))]
            raise ValueError(
                f"scene overlap between {previous_split} and {split}: {sorted(overlap)}"
            )
        scene_owners.update({scene_name: split for scene_name in scenes})

    devkit_version = distribution_version("nuscenes-devkit")
    for split, (arrays, counts) in prepared.items():
        save_cache(
            Path(config["cache_root"]) / split,
            arrays,
            {
                "schema_version": 1,
                "split": split,
                "dataset_version": config["version"],
                "history_seconds": config["history_seconds"],
                "future_seconds": config["future_seconds"],
                "frequency_hz": config["frequency_hz"],
                "filter_counts": counts,
                "preprocessing_config_sha256": config_sha256,
                "nuscenes_devkit_version": devkit_version,
            },
        )


if __name__ == "__main__":
    main()
