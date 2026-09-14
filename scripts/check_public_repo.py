"""Reject common data, secret, and machine-specific leaks before publication."""

from __future__ import annotations

import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path

FORBIDDEN_PREFIXES = ("data/", "outputs/", "checkpoints/")
MODEL_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors", ".onnx")
ARCHIVE_SUFFIXES = (".zip", ".tgz", ".tar", ".tar.gz", ".rar", ".7z")
SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx")
SECRET_PATTERNS = (
    re.compile(rb"(?i)\b(?:api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]"),
    re.compile(rb"(?i)\bauthorization\s*:\s*bearer\s+[a-z0-9._~+/=-]{12,}"),
    re.compile(rb"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----"),
    re.compile(rb"(?i)(?:[A-Z]:\\Users\\|/Users/|/home/)[^\r\n\"']+"),
)


def _normalise(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _readable_file(root: Path, relative_path: str) -> Path | None:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def find_violations(root: Path, tracked_paths: Iterable[str], max_bytes: int = 10 * 1024 * 1024) -> list[str]:
    """Return deterministic, human-readable violations for tracked repository paths."""
    if isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes <= 0:
        raise ValueError("max_bytes must be a positive integer")
    violations: list[str] = []
    root = Path(root).resolve()
    for raw_path in sorted(tracked_paths):
        relative = _normalise(str(raw_path))
        lower = relative.casefold()
        if lower.startswith(FORBIDDEN_PREFIXES):
            violations.append(f"{relative}: forbidden directory")
        if lower.endswith(MODEL_SUFFIXES):
            violations.append(f"{relative}: model/checkpoint suffix")
        if lower.endswith(ARCHIVE_SUFFIXES):
            violations.append(f"{relative}: dataset/archive suffix")
        if lower.endswith(SECRET_SUFFIXES):
            violations.append(f"{relative}: secret/key suffix")
        file_path = _readable_file(root, relative)
        if file_path is None:
            continue
        size = file_path.stat().st_size
        if size > max_bytes:
            violations.append(f"{relative}: too large ({size} bytes > {max_bytes})")
            continue
        content = file_path.read_bytes()
        if any(pattern.search(content) for pattern in SECRET_PATTERNS):
            violations.append(f"{relative}: secret-like content")
    return violations


def _git_root() -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def _tracked_paths(root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def main() -> int:
    try:
        root = _git_root()
        violations = find_violations(root, _tracked_paths(root))
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        print(f"public repository check failed to run: {exc}", file=sys.stderr)
        return 2
    if violations:
        print("public repository checks failed:")
        for violation in violations:
            print(f"- {violation}")
        return 1
    print("public repository checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
