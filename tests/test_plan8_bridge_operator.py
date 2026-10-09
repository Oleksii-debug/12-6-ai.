"""Plan 8 Section 5 text bridge CLI negative/restart smoke tests."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CLI = _ROOT / "tools" / "plan8_evidence_bridge.py"


def _invoke(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT / "src")
    return subprocess.run(
        [sys.executable, str(_CLI), *args],
        cwd=_ROOT, env=env, text=True,
        capture_output=True, check=False, timeout=30,
    )


def test_cli_help_is_keyboard_text_accessible() -> None:
    result = _invoke("--help")
    assert result.returncode == 0
    for command in ("stage", "execute", "verify", "publish", "status", "stop"):
        assert command in result.stdout


def test_cli_status_unstaged_never_reports_pass(tmp_path: Path) -> None:
    result = _invoke(
        "status", "--spool", str(tmp_path / "spool"),
        "--dispatch-sha256", "a" * 64,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "state": "NOT_STAGED", "verification": "NOT_CHECKED"
    }


def test_cli_rejects_bad_identity_and_unknown_stop(tmp_path: Path) -> None:
    bad = _invoke(
        "status", "--spool", str(tmp_path),
        "--dispatch-sha256", "../../etc/passwd",
    )
    assert bad.returncode == 2
    assert "DENIED" in bad.stderr
    stop = _invoke(
        "stop", "--spool", str(tmp_path),
        "--dispatch-sha256", "b" * 64,
    )
    assert stop.returncode == 2
    assert "unknown" in stop.stderr


@pytest.mark.parametrize("command", ["stage", "execute", "verify", "publish"])
def test_cli_requires_preverified_authority_fields(command: str) -> None:
    result = _invoke(command)
    assert result.returncode != 0
    assert "--packet" in result.stderr
    assert "--authority-keys" in result.stderr
