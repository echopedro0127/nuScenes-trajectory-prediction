import argparse
from pathlib import Path

from nuscenes import NuScenes

from trajectory_prediction.config import load_yaml
from trajectory_prediction.data.cache import save_cache
from trajectory_prediction.data.nuscenes_adapter import extract_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_yaml(args.config)

    nusc = NuScenes(
        version=config["version"], dataroot=config["dataroot"], verbose=True
    )
    for split in config["splits"]:
        arrays, counts = extract_split(nusc, split, config["history_seconds"])
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
            },
        )


if __name__ == "__main__":
    main()
