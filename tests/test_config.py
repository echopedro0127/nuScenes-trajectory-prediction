import hashlib
import json
from pathlib import Path

import pytest

from trajectory_prediction.config import (
    data_config_sha256,
    load_yaml,
    validate_data_config,
)


def _valid_data_config() -> dict:
    return {
        "version": "v1.0-mini",
        "dataroot": "data/nuscenes",
        "splits": ["mini_train", "mini_val"],
        "history_seconds": 2.0,
        "future_seconds": 6.0,
        "frequency_hz": 2,
        "cache_root": "data/cache/mini",
    }


def test_load_yaml_reads_mapping(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("version: v1.0-mini\nsplits: [mini_train]\n", encoding="utf-8")

    assert load_yaml(path) == {"version": "v1.0-mini", "splits": ["mini_train"]}


@pytest.mark.parametrize("content", ["- mini_train\n- mini_val\n", "null\n"])
def test_load_yaml_rejects_non_mapping(tmp_path: Path, content: str) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="configuration must be a mapping"):
        load_yaml(path)


def test_validate_data_config_normalizes_protocol_values() -> None:
    config = _valid_data_config()
    config["history_seconds"] = 2
    config["future_seconds"] = 6
    config["frequency_hz"] = 2.0

    validated = validate_data_config(config)

    assert type(validated) is dict
    assert validated["history_seconds"] == 2.0
    assert validated["future_seconds"] == 6.0
    assert validated["frequency_hz"] == 2


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("history_seconds", 1.0),
        ("future_seconds", 3.0),
        ("frequency_hz", 1),
        ("frequency_hz", True),
        ("history_seconds", "2.0"),
    ],
)
def test_validate_data_config_rejects_non_protocol_sampling(key: str, value: object) -> None:
    config = _valid_data_config()
    config[key] = value

    with pytest.raises(ValueError, match=key):
        validate_data_config(config)


def test_validate_data_config_requires_all_keys() -> None:
    config = _valid_data_config()
    config.pop("cache_root")

    with pytest.raises(ValueError, match="missing.*cache_root"):
        validate_data_config(config)


@pytest.mark.parametrize("key", ["version", "dataroot", "cache_root"])
def test_validate_data_config_rejects_empty_strings(key: str) -> None:
    config = _valid_data_config()
    config[key] = "  "

    with pytest.raises(ValueError, match=key):
        validate_data_config(config)


@pytest.mark.parametrize(
    "splits",
    [[], ["mini_train", "mini_train"], ["mini_train", ""], "mini_train"],
)
def test_validate_data_config_rejects_invalid_or_duplicate_splits(splits: object) -> None:
    config = _valid_data_config()
    config["splits"] = splits

    with pytest.raises(ValueError, match="splits"):
        validate_data_config(config)


def test_data_config_hash_is_stable_under_mapping_key_order() -> None:
    config = _valid_data_config()
    reordered = dict(reversed(list(config.items())))

    digest = data_config_sha256(config)

    expected_json = json.dumps(
        validate_data_config(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    assert digest == hashlib.sha256(expected_json).hexdigest()
    assert data_config_sha256(reordered) == digest
