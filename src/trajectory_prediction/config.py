from pathlib import Path

import yaml


def load_yaml(path: Path) -> dict:
    content = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(content, dict):
        raise ValueError(f"configuration must be a mapping: {path}")  # noqa: TRY004
    return content
