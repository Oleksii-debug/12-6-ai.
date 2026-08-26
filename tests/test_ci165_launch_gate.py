from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six import launch_gate
from twelve_six.milestone150_entrypoint import enforce_launch_gate


def test_historical_missing_pytest_fails_before_heavy_project_setup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regress SCALE-141's exact historical `No module named pytest` class."""
    request = {
        "schema": launch_gate.REQUEST_SCHEMA,
        "required_modules": ["pytest"],
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")

    monkeypatch.setattr(
        launch_gate.importlib.util,
        "find_spec",
        lambda name: None if name == "pytest" else object(),
    )
    heavy_called = False

    def fail_if_heavy(
        _: dict[str, object],
    ) -> tuple[dict[str, object], dict[str, object], list[str]]:
        nonlocal heavy_called
        heavy_called = True
        raise AssertionError("heavy project contracts must not run when pytest is missing")

    monkeypatch.setattr(launch_gate, "_verify_project_contracts", fail_if_heavy)
    with pytest.raises(launch_gate.LaunchGateError, match="required module unavailable: pytest"):
        launch_gate.create_launch_envelope(tmp_path, request_path, tmp_path / "envelope.json")
    assert heavy_called is False
    assert not (tmp_path / "envelope.json").exists()


def test_invalid_run_budget_is_fail_closed() -> None:
    with pytest.raises(launch_gate.LaunchGateError, match="positive integer"):
        launch_gate._verify_budget({"budget": {"optimizer_steps": 0}})
    with pytest.raises(launch_gate.LaunchGateError, match="missing steps/token target"):
        launch_gate._verify_budget({"budget": {"max_wall_minutes": 10}})


def test_envelope_hash_signature_source_config_and_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = {
        "schema": launch_gate.REQUEST_SCHEMA,
        "binding": {"workflow": "fixture", "scale": "500k"},
        "purpose_profile": {"profile_id": "fixture", "path": "profile.json"},
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    profile = {
        "profile_id": "fixture",
        "profile_path": "profile.json",
        "profile_file_sha256": "p",
        "profile_semantic_sha256": None,
        "base_profile_id": "fixture",
        "base_profile_path": "profile.json",
        "base_profile_file_sha256": "p",
        "base_manifest_sha256": "m",
        "locks": {"runtime": {"path": "runtime.lock.txt", "sha256": "r"}},
    }
    monkeypatch.setattr(launch_gate, "_git_head", lambda _: "a" * 40)
    monkeypatch.setattr(launch_gate, "_verify_lock_profile", lambda _repo, _request: profile)

    unsigned = {
        "schema": launch_gate.ENVELOPE_SCHEMA,
        "source_sha": "a" * 40,
        "request_sha256": launch_gate._hash_json(request),
        "binding": request["binding"],
        "python": {
            "implementation": "cpython",
            "version": launch_gate._python_version(),
            "executable": "fixture-python",
        },
        "purpose_profile": profile,
        "checks": {},
        "check_order": [],
        "training_performed": False,
    }
    envelope = dict(unsigned)
    envelope["envelope_sha256"] = launch_gate._hash_json(unsigned)
    envelope_path = tmp_path / "envelope.json"
    envelope_path.write_text(json.dumps(envelope), encoding="utf-8")

    verified = launch_gate.verify_launch_envelope(
        tmp_path,
        request_path,
        envelope_path,
        expected_binding={"workflow": "fixture", "scale": "500k"},
    )
    assert verified["envelope_sha256"] == envelope["envelope_sha256"]

    with pytest.raises(launch_gate.LaunchGateError, match="bound to another workflow/config"):
        launch_gate.verify_launch_envelope(
            tmp_path,
            request_path,
            envelope_path,
            expected_binding={"workflow": "fixture", "scale": "1m"},
        )

    monkeypatch.setattr(launch_gate, "_git_head", lambda _: "b" * 40)
    with pytest.raises(launch_gate.LaunchGateError, match="source SHA mismatch"):
        launch_gate.verify_launch_envelope(tmp_path, request_path, envelope_path)
    monkeypatch.setattr(launch_gate, "_git_head", lambda _: "a" * 40)

    tampered = dict(envelope)
    tampered["training_performed"] = True
    envelope_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(launch_gate.LaunchGateError, match="hash signature mismatch"):
        launch_gate.verify_launch_envelope(tmp_path, request_path, envelope_path)
    envelope_path.write_text(json.dumps(envelope), encoding="utf-8")

    request["binding"]["scale"] = "1m"
    request_path.write_text(json.dumps(request), encoding="utf-8")
    with pytest.raises(launch_gate.LaunchGateError, match="launch config mismatch"):
        launch_gate.verify_launch_envelope(tmp_path, request_path, envelope_path)


def test_m150_training_entrypoint_refuses_absent_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TWELVE_SIX_LAUNCH_REQUEST", raising=False)
    monkeypatch.delenv("TWELVE_SIX_LAUNCH_ENVELOPE", raising=False)
    with pytest.raises(launch_gate.LaunchGateError, match="mandatory"):
        enforce_launch_gate(
            [
                "phase1",
                "--repo-root",
                ".",
                "--source-sha",
                "0" * 40,
                "--output-dir",
                "out",
                "--scale",
                "500k",
            ]
        )


def test_m150_non_training_commands_do_not_require_envelope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TWELVE_SIX_LAUNCH_REQUEST", raising=False)
    monkeypatch.delenv("TWELVE_SIX_LAUNCH_ENVELOPE", raising=False)
    enforce_launch_gate(["prepare", "--source-sha", "0" * 40])
    enforce_launch_gate(["verify-scale", "--scale", "500k"])
