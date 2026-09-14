"""Build Markdown result tables from measured JSON/CSV artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path


def _finite_metric(metrics: Mapping[str, object], name: str) -> float:
    value = metrics.get(name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _sample_count(metrics: Mapping[str, object]) -> int:
    value = metrics.get("sample_count")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError("sample_count must be a positive integer")
    return value


def _validate_metrics(metrics: Mapping[str, object]) -> dict[str, object]:
    sample_count = _sample_count(metrics)
    return {"sample_count": sample_count, "ade": _finite_metric(metrics, "ade"), "fde": _finite_metric(metrics, "fde")}


def load_metrics(path: Path) -> dict[str, object]:
    """Load and validate a static metrics JSON artifact."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read metrics JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise TypeError("metrics JSON must contain an object")
    return _validate_metrics(value) | {key: item for key, item in value.items() if key not in {"sample_count", "ade", "fde"}}


def markdown_row(name: str, metrics: Mapping[str, object]) -> str:
    """Format one measured ADE/FDE row with stable three-decimal precision."""
    checked = _validate_metrics(metrics)
    return f"| {name} | {checked['sample_count']} | {checked['ade']:.3f} | {checked['fde']:.3f} |"


def _table(title: str, rows: Sequence[str]) -> str:
    if not rows:
        return ""
    return "\n".join([f"### {title}", "", "| Model | Samples | ADE | FDE |", "| --- | ---: | ---: | ---: |", *rows])


def _named(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise ValueError(f"expected NAME=PATH, got {value!r}")
    return name, Path(path)


def build_static_table(entries: Sequence[str]) -> str:
    """Load ``NAME=metrics.json`` entries and return a Markdown table."""
    parsed = [(_named(item)[0], load_metrics(_named(item)[1])) for item in entries]
    if not parsed:
        raise ValueError("at least one static metrics artifact is required")
    counts = {_sample_count(metrics) for _, metrics in parsed}
    if len(counts) != 1:
        raise ValueError("all static metrics must use the same sample_count")
    return _table("Static benchmark", [markdown_row(name, metrics) for name, metrics in parsed])


def _online_rows(entries: Sequence[str]) -> list[str]:
    rows: list[str] = []
    for item in entries:
        name, path = _named(item)
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise TypeError(f"online metrics must be an object: {path}")
        for strategy, metrics in value.items():
            if not isinstance(metrics, Mapping):
                raise TypeError(f"online strategy metrics must be objects: {path}")
            rows.append(markdown_row(f"{name}/{strategy}", metrics))
    return rows


def _horizon_rows(entries: Sequence[str]) -> list[str]:
    rows: list[str] = []
    for item in entries:
        name, path = _named(item)
        metrics = load_metrics(path)
        horizons = metrics.get("horizons")
        if not isinstance(horizons, Mapping):
            raise TypeError(f"metrics JSON has no horizons object: {path}")
        for horizon, values in horizons.items():
            if not isinstance(values, Mapping):
                raise TypeError(f"horizon metrics must be objects: {path}")
            rows.append(markdown_row(f"{name}/{horizon}", {"sample_count": metrics["sample_count"], **values}))
    return rows


def _maneuver_rows(entries: Sequence[str]) -> list[str]:
    rows: list[str] = []
    for item in entries:
        name, path = _named(item)
        with path.open(newline="", encoding="utf-8") as stream:
            for record in csv.DictReader(stream):
                metrics = {key: float(record[key]) if key != "sample_count" else int(record[key]) for key in ("sample_count", "ade", "fde")}
                rows.append(markdown_row(f"{name}/{record['maneuver']}", metrics))
    return rows


def build_tables(
    static: Sequence[str],
    online: Sequence[str] = (),
    maneuver: Sequence[str] = (),
    horizon: Sequence[str] = (),
) -> str:
    sections = [build_static_table(static)]
    if horizon:
        sections.append(_table("Horizon analysis", _horizon_rows(horizon)))
    if online:
        sections.append(_table("Online correction", _online_rows(online)))
    if maneuver:
        sections.append(_table("Maneuver analysis", _maneuver_rows(maneuver)))
    return "\n\n".join(section for section in sections if section) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Markdown tables from measured artifacts")
    parser.add_argument("--static", nargs="+", required=True, metavar="NAME=METRICS_JSON")
    parser.add_argument("--online", nargs="*", default=(), metavar="NAME=ONLINE_JSON")
    parser.add_argument("--maneuver", nargs="*", default=(), metavar="NAME=MANEUVER_CSV")
    parser.add_argument("--horizon", nargs="*", default=(), metavar="NAME=METRICS_JSON")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    rendered = build_tables(args.static, args.online, args.maneuver, args.horizon)
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
