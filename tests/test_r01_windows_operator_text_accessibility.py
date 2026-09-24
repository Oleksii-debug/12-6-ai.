from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from twelve_six.windows_operator_preflight import EXIT_BLOCKED, EXIT_ERROR, _render_text

ROOT = Path(__file__).resolve().parents[1]


def _assert_line_oriented_printable(text: str) -> None:
    assert all(char == "\n" or char.isprintable() for char in text)
    assert all(all(char.isprintable() for char in line) for line in text.splitlines())


def test_text_renderer_exposes_safe_stop_status_and_validation_errors() -> None:
    rendered = _render_text(
        {
            "status": "BLOCKED",
            "target": "20m",
            "launch_authorized": False,
            "training_authorized": False,
            "safe_stop": {
                "status": "INVALID",
                "marker": None,
                "errors": [
                    "safe_stop_marker_sha256_invalid",
                    "safe_stop_target_mismatch",
                ],
            },
        }
    )

    assert "SAFE_STOP_STATUS: INVALID" in rendered.splitlines()
    assert "SAFE_STOP_ERROR: safe_stop_marker_sha256_invalid" in rendered.splitlines()
    assert "SAFE_STOP_ERROR: safe_stop_target_mismatch" in rendered.splitlines()
    _assert_line_oriented_printable(rendered)


def test_text_renderer_exposes_actionable_error_and_escapes_controls() -> None:
    rendered = _render_text(
        {
            "status": "ERROR",
            "error": "bad\nFORGED: yes\x1b[31m\r",
            "launch_authorized": False,
            "training_authorized": False,
        }
    )

    assert "ERROR: bad\\x0aFORGED: yes\\x1b[31m\\x0d" in rendered.splitlines()
    assert "\x1b" not in rendered
    assert "\r" not in rendered
    _assert_line_oriented_printable(rendered)


def test_text_cli_error_reports_reason_without_json(tmp_path: Path) -> None:
    missing_profile = tmp_path / "missing-profile.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "twelve_six.windows_operator_preflight",
            "--profile",
            str(missing_profile),
            "verify",
            "--target",
            "20m",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    assert proc.returncode == EXIT_ERROR
    assert "OPERATOR_STATUS: ERROR" in proc.stdout.splitlines()
    assert any(
        line.startswith("ERROR: invalid_or_unreadable_json:")
        for line in proc.stdout.splitlines()
    )
    assert "LAUNCH_AUTHORIZED: false" in proc.stdout.splitlines()
    assert "TRAINING_AUTHORIZED: false" in proc.stdout.splitlines()
    _assert_line_oriented_printable(proc.stdout.rstrip("\n"))


def test_text_cli_status_announces_not_requested_safe_stop(tmp_path: Path) -> None:
    state_dir = tmp_path / "operator-state"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "twelve_six.windows_operator_preflight",
            "--state-dir",
            str(state_dir),
            "status",
            "--target",
            "20m",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )

    if sys.platform == "win32":
        assert proc.returncode in (0, EXIT_BLOCKED)
    else:
        assert proc.returncode == EXIT_BLOCKED
    assert "SAFE_STOP_STATUS: NOT_REQUESTED" in proc.stdout.splitlines()
    assert "LAUNCH_AUTHORIZED: false" in proc.stdout.splitlines()
    assert "TRAINING_AUTHORIZED: false" in proc.stdout.splitlines()
    _assert_line_oriented_printable(proc.stdout.rstrip("\n"))
