"""Run every quality gate the CI pipeline runs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKS: list[tuple[str, list[str]]] = [
    ("format", [sys.executable, "-m", "ruff", "format", "--check", "."]),
    ("lint", [sys.executable, "-m", "ruff", "check", "."]),
    ("types", [sys.executable, "-m", "mypy"]),
    ("tests", [sys.executable, "-m", "pytest"]),
]


def main() -> int:
    failed: list[str] = []
    for name, command in CHECKS:
        print(f"\n=== {name} ===", flush=True)
        result = subprocess.run(command, cwd=PROJECT_ROOT)
        if result.returncode != 0:
            failed.append(name)

    print("\n=== summary ===")
    for name, _ in CHECKS:
        print(f"{name}: {'FAILED' if name in failed else 'passed'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
