"""Render deterministic worst-case trajectory figures from evaluation artifacts."""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from trajectory_prediction.analysis.visualization import render_case
from trajectory_prediction.config import data_config_sha256, load_yaml, validate_data_config
from trajectory_prediction.data.cache import load_cache
from trajectory_prediction.utils.provenance import sha256_file


def _project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if resolved.parent.name == "data" and resolved.parent.parent.name == "configs":
        return resolved.parents[2]
    return resolved.parent


def _safe_token(value: object) -> str:
    token = str(value)
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", token).strip("-")
    return cleaned[:48] or "unknown"


def _load_cases(path: Path, limit: int) -> list[dict[str, object]]:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
        raise ValueError("limit must be a positive integer")
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read worst-case JSON: {path}") from exc
    if not isinstance(value, list):
        raise TypeError("worst-case JSON must contain a list")
    cases: list[dict[str, object]] = []
    for item in value[:limit]:
        if not isinstance(item, Mapping):
            raise TypeError("each worst case must be an object")
        if not isinstance(item.get("instance_token"), str) or not isinstance(item.get("sample_token"), str):
            raise TypeError("worst cases require string instance_token and sample_token")
        cases.append(dict(item))
    if not cases:
        raise ValueError("worst-case JSON contains no cases")
    return cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render worst-case trajectory figures")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/mini.yaml"))
    parser.add_argument("--split", required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--model", default="prediction")
    parser.add_argument("--best", action="store_true", help="verify the evaluation uses model/best.pt")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/figures"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_config_path = args.data_config.resolve()
    project_root = _project_root(data_config_path)
    config = validate_data_config(load_yaml(data_config_path))
    if args.split not in config["splits"]:
        raise ValueError(f"split {args.split!r} is not listed in data configuration")
    cache_root = Path(config["cache_root"])
    if not cache_root.is_absolute():
        cache_root = project_root / cache_root
    arrays, manifest = load_cache(
        cache_root / args.split,
        args.split,
        expected_config_sha256=data_config_sha256(config),
    )
    evaluation_dir = args.evaluation_dir if args.evaluation_dir.is_absolute() else project_root / args.evaluation_dir
    prediction_path = evaluation_dir / "predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as payload:
        if "prediction" not in payload:
            raise ValueError("evaluation archive must contain prediction")
        prediction = np.asarray(payload["prediction"])
    if prediction.shape != arrays["future_xy"].shape:
        raise ValueError("prediction and cached future arrays must share shape")

    metrics_path = evaluation_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if not isinstance(metrics, Mapping):
        raise TypeError("metrics.json must contain an object")
    provenance = metrics.get("provenance", {})
    if args.best:
        checkpoint = evaluation_dir.parent / args.model.lower() / "best.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"best checkpoint not found: {checkpoint}")
        if not isinstance(provenance, Mapping) or provenance.get("checkpoint_sha256") != sha256_file(checkpoint):
            raise ValueError("evaluation provenance does not match model/best.pt")

    cases_path = args.cases
    if cases_path is None:
        cases_path = evaluation_dir.parent / "analysis" / "worst_cases.json"
    if not cases_path.is_absolute():
        cases_path = project_root / cases_path
    cases = _load_cases(cases_path, args.limit)
    indices = {
        (str(instance), str(sample)): index
        for index, (instance, sample) in enumerate(zip(arrays["instance_token"], arrays["sample_token"], strict=True))
    }
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rendered: list[str] = []
    for rank, case in enumerate(cases, start=1):
        key = (case["instance_token"], case["sample_token"])
        if key not in indices:
            raise ValueError(f"worst case is not present in cache: {key}")
        index = indices[key]
        title = f"{args.model} | rank {rank} | ADE {float(case.get('ade', float('nan'))):.2f} | FDE {float(case.get('fde', float('nan'))):.2f}"
        output_path = output_dir / f"{rank:03d}_{_safe_token(case['instance_token'])}_{_safe_token(case['sample_token'])}.png"
        render_case(
            arrays["history_xy"][index],
            arrays["future_xy"][index],
            {args.model: prediction[index]},
            title,
            output_path,
        )
        rendered.append(str(output_path))
    manifest_payload = {
        "model": args.model,
        "best_checkpoint_verified": bool(args.best),
        "split": args.split,
        "cache_npz_sha256": manifest["npz_sha256"],
        "cases_json": str(cases_path),
        "rendered_files": rendered,
    }
    (output_dir / "render_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
