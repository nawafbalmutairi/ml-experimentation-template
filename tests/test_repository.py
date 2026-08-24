from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def is_ignored(path: str) -> bool:
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    return subprocess.run(["git", "check-ignore", "-q", path], cwd=REPO_ROOT).returncode == 0


def test_written_reports_are_tracked() -> None:
    """The write-up is the workflow's deliverable; ignoring it loses it in every copy."""
    assert not is_ignored("reports/churn-analysis.md")


@pytest.mark.parametrize("artefact", ["reports/predictions.csv", "reports/lift.png"])
def test_generated_report_artefacts_stay_untracked(artefact: str) -> None:
    assert is_ignored(artefact)
