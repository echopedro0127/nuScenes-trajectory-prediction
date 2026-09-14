import hashlib
import json
import math
from collections.abc import Mapping
from numbers import Real
from pathlib import Path

import yaml

REQUIRED_DATA_CONFIG_KEYS = {
    "version",
    "dataroot",
    "splits",
    "history_seconds",
    "future_seconds",
    "frequency_hz",
    "cache_root",
}


def load_yaml(path: Path) -> dict:
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"configuration must be a mapping: {path}")  # noqa: TRY004
    return content


def _normalized_string(config: Mapping[str, object], key: str) -> str:
    value = config[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a nonempty string")
    return value.strip()


def _protocol_number(
    config: Mapping[str, object], key: str, expected: float
) -> float:
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{key} must be numeric and equal to {expected}")  # noqa: TRY004
    numeric = float(value)
    if not math.isfinite(numeric) or numeric != expected:
        raise ValueError(f"{key} must equal {expected}")
    return numeric


def validate_data_config(config: Mapping[str, object]) -> dict:
    missing = REQUIRED_DATA_CONFIG_KEYS.difference(config)
    if missing:
        raise ValueError(f"data configuration missing keys: {sorted(missing)}")

    normalized = dict(config)
    for key in ("version", "dataroot", "cache_root"):
        normalized[key] = _normalized_string(config, key)

    raw_splits = config["splits"]
    if not isinstance(raw_splits, list) or not raw_splits:
        raise ValueError("splits must be a nonempty list of unique strings")
    splits: list[str] = []
    for split in raw_splits:
        if not isinstance(split, str) or not split.strip():
            raise ValueError("splits must contain nonempty strings")
        splits.append(split.strip())
    if len(splits) != len(set(splits)):
        raise ValueError("splits must be unique")
    normalized["splits"] = splits

    normalized["history_seconds"] = _protocol_number(
        config, "history_seconds", 2.0
    )
    normalized["future_seconds"] = _protocol_number(
        config, "future_seconds", 6.0
    )
    frequency_hz = _protocol_number(config, "frequency_hz", 2.0)
    normalized["frequency_hz"] = int(frequency_hz)
    return normalized


def data_config_sha256(config: Mapping[str, object]) -> str:
    normalized = validate_data_config(config)
    canonical = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
