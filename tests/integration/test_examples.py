"""Run the example scripts, so the documented usage can never silently break."""

from __future__ import annotations

import runpy
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def test_examples_folder_is_found() -> None:
    assert sorted(path.name for path in EXAMPLES.glob("*.py")) == [
        "process_packets.py",
        "udp_runtime.py",
    ]


def test_process_packets_reports_gap_and_glitch(capsys: pytest.CaptureFixture[str]) -> None:
    runpy.run_path(str(EXAMPLES / "process_packets.py"), run_name="__main__")
    output = capsys.readouterr().out
    assert "degraded frames:    2" in output
    assert "{'ANGULAR_VELOCITY_OUT_OF_RANGE': 1}" in output
    assert '"status":"ok"' in output


def test_udp_runtime_reports_stale_frames() -> None:
    # A separate process gives the example its own event loop, as when a user runs it.
    completed = subprocess.run(
        [sys.executable, str(EXAMPLES / "udp_runtime.py")],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert "went STALE after sequence 249" in completed.stdout
    assert "'ok': 250" in completed.stdout
