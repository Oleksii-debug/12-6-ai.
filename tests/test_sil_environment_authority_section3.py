from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from twelve_six import sil_qualification
from twelve_six.capability_map import load_capability_registry
from twelve_six.sil_qualification import (
    CommandExecution,
    GitState,
    canonical_sil_environment_receipt_v1,
    load_sil_scenario,
    qualify_sil,
    verify_sil_evidence,
)

_ROOT = Path(__file__).parents[1]
_REGISTRY = _ROOT / "configs" / "control" / "product_capabilities_v1.json"
_SCENARIO = _ROOT / "configs" / "control" / "sil_scenario_v1.json"
_GIT_SHA = "a" * 40


def _pass_runner(
    argv: tuple[str, ...],
    cwd: Path,
    timeout_seconds: int,
    input_envelope_bytes: bytes,
    expected_input_identity_sha256: str,
) -> CommandExecution:
    assert cwd
    assert timeout_seconds > 0
    assert hashlib.sha256(input_envelope_bytes).hexdigest() == (
        expected_input_identity_sha256
    )
    return CommandExecution(
        return_code=0,
        stdout="PASS " + " ".join(argv),
        stderr="",
        duration_ms=1,
        consumed_input_identity_sha256=expected_input_identity_sha256,
    )


def _git_probe(_: str | Path) -> GitState:
    return GitState(sha=_GIT_SHA, tracked_clean=True)


def test_public_sil_environment_validation_ignores_module_global_rebinding(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    registry = load_capability_registry(_REGISTRY)
    scenario = load_sil_scenario(_SCENARIO)
    canonical_environment = canonical_sil_environment_receipt_v1()
    package_bytes = sil_qualification.build_package_manifest_bytes(_ROOT)

    evidence, log_text = sil_qualification._qualify_sil_with_backends(
        repo_root=_ROOT,
        expected_git_sha=_GIT_SHA,
        registry=registry,
        scenario=scenario,
        package_bytes=package_bytes,
        environment_receipt=canonical_environment,
        command_runner=_pass_runner,
        git_probe=_git_probe,
    )
    evidence_path = tmp_path / "environment-authority-evidence.json"
    log_path = tmp_path / "environment-authority.log"
    evidence_path.write_bytes(
        sil_qualification._canonical_json_bytes(evidence) + b"\n"
    )
    log_path.write_text(log_text, encoding="utf-8")

    monkeypatch.setattr(
        sil_qualification,
        "_validate_sil_environment_receipt",
        lambda _payload: canonical_environment,
    )
    forged_environment = {"forged": True}

    with pytest.raises(ValueError, match="exact pinned lock-source contract"):
        qualify_sil(
            repo_root=_ROOT,
            expected_git_sha=_GIT_SHA,
            registry=registry,
            scenario=scenario,
            package_bytes=b"non-empty",
            environment_receipt=forged_environment,
        )

    with pytest.raises(ValueError, match="exact pinned lock-source contract"):
        verify_sil_evidence(
            evidence_path,
            log_path,
            expected_package_bytes=package_bytes,
            expected_environment_receipt=forged_environment,
            expected_registry=registry,
            expected_scenario=scenario,
            expected_git_sha=_GIT_SHA,
        )
