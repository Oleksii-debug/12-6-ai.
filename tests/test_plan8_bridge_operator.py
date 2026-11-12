"""Plan 8 Section 5 text bridge CLI negative/restart smoke tests."""
from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CLI = _ROOT / "tools" / "plan8_evidence_bridge.py"


def _invoke(*args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(_ROOT / "src")
    env["PYTHONSAFEPATH"] = "1"
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


def test_execute_records_verified_failure_as_defect(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.syspath_prepend(str(_CLI.parent))
    cli = importlib.import_module("plan8_evidence_bridge")
    identity = "a" * 64
    dispatch = SimpleNamespace(identity_sha256=lambda: identity)
    monkeypatch.setattr(cli, "_load_dispatch", lambda args: (dispatch, object(), {}))
    claimed = []
    monkeypatch.setattr(cli, "claim_bridge_dispatch", lambda _, value: claimed.append(value))
    monkeypatch.setattr(cli, "_public_key_verifier", lambda _: object())
    observed = []

    def physical_operator(argv, *, return_verified_failure):
        assert argv[0] == "run"
        assert return_verified_failure is True
        return 0

    def record(spool, **arguments):
        assert spool == tmp_path
        assert arguments["dispatch"] is dispatch
        observed.append(identity)
        return {"verdict": "FAIL", "defect": {"kind": "REPRODUCIBLE_TEST_FAILURE"}}

    monkeypatch.setattr(cli, "physical_operator", physical_operator)
    monkeypatch.setattr(cli, "record_bridge_return", record)
    result = cli.main([
        "execute", "--spool", str(tmp_path), "--packet", "packet.json",
        "--authority-keys", "authority.json", "--repo-root", str(_ROOT),
        "--dispatch-id", "dispatch", "--profile-id", "SCALE-500M",
        "--scenario-id", "scenario", "--physical-gate-id", "gate",
        "--host-keys", "host.json", "--receipt", "receipt.json", "--log", "log.bin",
        "--host-private-key", "host.raw", "--host-key-id", "host",
    ])
    output = json.loads(capsys.readouterr().out)
    assert result == 1
    assert claimed == observed == [identity]
    assert output["state"] == "VERIFIED_RETURN"
    assert output["physical_verified"] is True
    assert output["defect"]["kind"] == "REPRODUCIBLE_TEST_FAILURE"
