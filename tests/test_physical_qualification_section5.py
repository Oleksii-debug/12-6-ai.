from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import pytest

from twelve_six.physical_qualification import (
    ActionExecution,
    ExecutionMode,
    HostInventory,
    QualificationAction,
    QualificationPacket,
    QualificationVerdict,
    ResourceKind,
    execute_qualification,
    load_verified_signed_packet,
    verify_qualification_evidence,
    write_evidence_bundle,
)
from twelve_six.sil_qualification import GitState


_GIT_SHA = "a" * 40
_AGENT_BYTES = b"section-5-agent-source"
_AGENT_SHA = hashlib.sha256(_AGENT_BYTES).hexdigest()
_KEY_ID = "physical-root-1"
_HOST_KEY_ID = "physical-host-1"


def _fake_signature(message: bytes) -> bytes:
    left = hashlib.sha256(_KEY_ID.encode("ascii") + message).digest()
    right = hashlib.sha256(message + _KEY_ID.encode("ascii")).digest()
    return left + right


def _fake_verify(key_id: str, message: bytes, signature: bytes) -> bool:
    return key_id == _KEY_ID and signature == _fake_signature(message)


def _host_signature(message: bytes) -> bytes:
    left = hashlib.sha256(_HOST_KEY_ID.encode("ascii") + message).digest()
    right = hashlib.sha256(message + _HOST_KEY_ID.encode("ascii")).digest()
    return left + right


def _fake_evidence_signer(key_id: str, message: bytes) -> bytes:
    assert key_id == _HOST_KEY_ID
    return _host_signature(message)


def _fake_evidence_verify(key_id: str, message: bytes, signature: bytes) -> bool:
    return key_id == _HOST_KEY_ID and signature == _host_signature(message)


def _action(
    *,
    action_id: str = "cpu-regression",
    resources: tuple[ResourceKind, ...] = (ResourceKind.CPU,),
    max_output_bytes: int = 1024,
) -> QualificationAction:
    return QualificationAction(
        action_id=action_id,
        pytest_targets=("tests/test_physical_qualification_section5.py",),
        timeout_seconds=60,
        max_output_bytes=max_output_bytes,
        required_resources=resources,
    )


def _packet(
    *,
    mode: ExecutionMode = ExecutionMode.REAL_HOST,
    actions: tuple[QualificationAction, ...] | None = None,
    artifacts: tuple[str, ...] = (),
) -> QualificationPacket:
    return QualificationPacket(
        schema_version="12-6.physical-qualification-packet.v1",
        packet_id="section5-test",
        target_git_sha=_GIT_SHA,
        agent_source_sha256=_AGENT_SHA,
        execution_mode=mode,
        allowed_os_families=("LINUX", "WINDOWS"),
        not_before_epoch_seconds=100,
        expires_epoch_seconds=200,
        actions=(_action(),) if actions is None else actions,
        artifact_paths=artifacts,
    )


def _write_signed_packet(path: Path, packet: QualificationPacket, *, corrupt: bool = False) -> None:
    message = json.dumps(
        packet.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    signature = _fake_signature(message)
    if corrupt:
        signature = b"x" * 64
    payload = {
        "schema_version": "12-6.signed-physical-qualification-packet.v1",
        "packet": packet.to_dict(),
        "signature": {
            "algorithm": "ED25519",
            "key_id": _KEY_ID,
            "signature_b64": base64.b64encode(signature).decode("ascii"),
        },
    }
    path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _load(tmp_path: Path, packet: QualificationPacket):
    path = tmp_path / "packet.json"
    _write_signed_packet(path, packet)
    return load_verified_signed_packet(
        path,
        signature_verifier=_fake_verify,
        now_epoch_seconds=150,
    )


def _inventory(*, cuda: bool = False) -> HostInventory:
    return HostInventory(
        os_family="LINUX",
        platform_system="Linux",
        platform_release="test",
        machine="x86_64",
        python_version="3.13.0",
        cpu_logical_count=8,
        ram_total_bytes=16 * 1024**3,
        disk_total_bytes=100 * 1024**3,
        disk_free_bytes=80 * 1024**3,
        torch_version="2.8.0",
        cuda_available=cuda,
        cuda_device_names=("Test GPU",) if cuda else (),
    )


def _pass_runner(action: QualificationAction, root: Path) -> ActionExecution:
    del action, root
    return ActionExecution(return_code=0, stdout=b"ok\n", stderr=b"", duration_ms=7)


def _git_probe(_: Path) -> GitState:
    return GitState(sha=_GIT_SHA, tracked_clean=True)


def test_signed_packet_rejects_bad_signature_and_expiry(tmp_path: Path) -> None:
    path = tmp_path / "packet.json"
    packet = _packet()
    _write_signed_packet(path, packet, corrupt=True)
    with pytest.raises(ValueError, match="signature verification failed"):
        load_verified_signed_packet(
            path,
            signature_verifier=_fake_verify,
            now_epoch_seconds=150,
        )

    _write_signed_packet(path, packet)
    with pytest.raises(ValueError, match="validity window"):
        load_verified_signed_packet(
            path,
            signature_verifier=_fake_verify,
            now_epoch_seconds=201,
        )


def test_action_rejects_shell_or_path_escape() -> None:
    with pytest.raises(ValueError, match="under tests"):
        QualificationAction(
            action_id="bad",
            pytest_targets=("src/twelve_six/foo.py",),
            timeout_seconds=10,
            max_output_bytes=100,
            required_resources=(ResourceKind.CPU,),
        )
    with pytest.raises(ValueError, match="inside the repository"):
        QualificationAction(
            action_id="bad",
            pytest_targets=("tests/../secrets.py",),
            timeout_seconds=10,
            max_output_bytes=100,
            required_resources=(ResourceKind.CPU,),
        )


def test_simulation_can_never_claim_physical_pass(tmp_path: Path) -> None:
    verified = _load(tmp_path, _packet(mode=ExecutionMode.SIMULATION))
    evidence, _ = execute_qualification(
        verified,
        repo_root=tmp_path,
        host_inventory=_inventory(),
        action_runner=_pass_runner,
        git_probe=_git_probe,
        agent_source_bytes=_AGENT_BYTES,
        evidence_signing_key_id=_HOST_KEY_ID,
        evidence_signer=_fake_evidence_signer,
    )
    assert evidence["verdict"] == QualificationVerdict.SIMULATION_PASS.value
    assert set(evidence["resource_observations"].values()) == {"SIMULATED"}


def test_real_host_pass_requires_exact_proven_resources(tmp_path: Path) -> None:
    action = _action(
        resources=(ResourceKind.CPU, ResourceKind.DISK, ResourceKind.RAM),
    )
    verified = _load(tmp_path, _packet(actions=(action,)))
    evidence, _ = execute_qualification(
        verified,
        repo_root=tmp_path,
        host_inventory=_inventory(),
        action_runner=_pass_runner,
        git_probe=_git_probe,
        agent_source_bytes=_AGENT_BYTES,
        evidence_signing_key_id=_HOST_KEY_ID,
        evidence_signer=_fake_evidence_signer,
    )
    assert evidence["verdict"] == QualificationVerdict.PASS.value
    assert evidence["resource_observations"]["CPU"] == "REAL_PROBED"
    assert evidence["resource_observations"]["RAM"] == "REAL_PROBED"
    assert evidence["resource_observations"]["DISK"] == "REAL_PROBED"


def test_unproven_network_or_absent_gpu_blocks_real_pass(tmp_path: Path) -> None:
    actions = (
        _action(
            action_id="gpu",
            resources=(ResourceKind.GPU,),
        ),
        _action(
            action_id="network",
            resources=(ResourceKind.NETWORK,),
        ),
    )
    verified = _load(tmp_path, _packet(actions=actions))
    evidence, _ = execute_qualification(
        verified,
        repo_root=tmp_path,
        host_inventory=_inventory(cuda=False),
        action_runner=_pass_runner,
        git_probe=_git_probe,
        agent_source_bytes=_AGENT_BYTES,
        evidence_signing_key_id=_HOST_KEY_ID,
        evidence_signer=_fake_evidence_signer,
    )
    assert evidence["verdict"] == QualificationVerdict.FAIL.value
    assert "required real resource is not proven: GPU" in evidence["reasons"]
    assert "required real resource is not proven: NETWORK" in evidence["reasons"]


def test_agent_source_and_exact_clean_checkout_are_fail_closed(tmp_path: Path) -> None:
    verified = _load(tmp_path, _packet())
    with pytest.raises(ValueError, match="does not authorize"):
        execute_qualification(
            verified,
            repo_root=tmp_path,
            host_inventory=_inventory(),
            action_runner=_pass_runner,
            git_probe=_git_probe,
            agent_source_bytes=b"wrong-agent",
            evidence_signing_key_id=_HOST_KEY_ID,
            evidence_signer=_fake_evidence_signer,
        )

    with pytest.raises(ValueError, match="dirty before qualification"):
        execute_qualification(
            verified,
            repo_root=tmp_path,
            host_inventory=_inventory(),
            action_runner=_pass_runner,
            git_probe=lambda _: GitState(sha=_GIT_SHA, tracked_clean=False),
            agent_source_bytes=_AGENT_BYTES,
        )


def test_post_action_tree_mutation_or_output_overflow_fails(tmp_path: Path) -> None:
    verified = _load(tmp_path, _packet(actions=(_action(max_output_bytes=2),)))
    states = iter(
        (
            GitState(sha=_GIT_SHA, tracked_clean=True),
            GitState(sha=_GIT_SHA, tracked_clean=True),
            GitState(sha=_GIT_SHA, tracked_clean=False),
        )
    )
    evidence, _ = execute_qualification(
        verified,
        repo_root=tmp_path,
        host_inventory=_inventory(),
        action_runner=lambda action, root: ActionExecution(
            return_code=0,
            stdout=b"too long",
            stderr=b"",
            duration_ms=1,
        ),
        git_probe=lambda _: next(states),
        agent_source_bytes=_AGENT_BYTES,
        evidence_signing_key_id=_HOST_KEY_ID,
        evidence_signer=_fake_evidence_signer,
    )
    assert evidence["verdict"] == QualificationVerdict.FAIL.value
    assert evidence["actions"][0]["stdout_truncated"] is True
    assert evidence["actions"][0]["post_tracked_clean"] is False


def test_artifact_hashes_and_log_are_independently_verified(tmp_path: Path) -> None:
    artifact = tmp_path / "evidence.bin"
    artifact.write_bytes(b"physical-result")
    verified = _load(tmp_path, _packet(artifacts=("evidence.bin",)))
    evidence, log_bytes = execute_qualification(
        verified,
        repo_root=tmp_path,
        host_inventory=_inventory(),
        action_runner=_pass_runner,
        git_probe=_git_probe,
        agent_source_bytes=_AGENT_BYTES,
        evidence_signing_key_id=_HOST_KEY_ID,
        evidence_signer=_fake_evidence_signer,
    )
    evidence_path = tmp_path / "evidence.json"
    log_path = tmp_path / "run.jsonl"
    write_evidence_bundle(
        evidence_path,
        log_path,
        evidence=evidence,
        log_bytes=log_bytes,
    )
    checked = verify_qualification_evidence(
        evidence_path,
        log_path,
        verified_packet=verified,
        agent_source_bytes=_AGENT_BYTES,
        artifact_root=tmp_path,
        require_real_pass=True,
        evidence_signature_verifier=_fake_evidence_verify,
    )
    assert checked["verdict"] == "PASS"

    log_path.write_bytes(log_bytes + b"x")
    with pytest.raises(ValueError, match="log hash mismatch"):
        verify_qualification_evidence(
            evidence_path,
            log_path,
            verified_packet=verified,
            agent_source_bytes=_AGENT_BYTES,
            artifact_root=tmp_path,
            require_real_pass=True,
            evidence_signature_verifier=_fake_evidence_verify,
        )


def test_artifact_substitution_and_simulation_real_gate_are_rejected(tmp_path: Path) -> None:
    artifact = tmp_path / "evidence.bin"
    artifact.write_bytes(b"physical-result")
    verified = _load(
        tmp_path,
        _packet(
            mode=ExecutionMode.SIMULATION,
            artifacts=("evidence.bin",),
        ),
    )
    evidence, log_bytes = execute_qualification(
        verified,
        repo_root=tmp_path,
        host_inventory=_inventory(),
        action_runner=_pass_runner,
        git_probe=_git_probe,
        agent_source_bytes=_AGENT_BYTES,
        evidence_signing_key_id=_HOST_KEY_ID,
        evidence_signer=_fake_evidence_signer,
    )
    evidence_path = tmp_path / "evidence.json"
    log_path = tmp_path / "run.jsonl"
    write_evidence_bundle(
        evidence_path,
        log_path,
        evidence=evidence,
        log_bytes=log_bytes,
    )
    with pytest.raises(ValueError, match="real physical PASS is required"):
        verify_qualification_evidence(
            evidence_path,
            log_path,
            verified_packet=verified,
            agent_source_bytes=_AGENT_BYTES,
            artifact_root=tmp_path,
            require_real_pass=True,
            evidence_signature_verifier=_fake_evidence_verify,
        )

    artifact.write_bytes(b"substituted")
    with pytest.raises(ValueError, match="artifact hash/size mismatch"):
        verify_qualification_evidence(
            evidence_path,
            log_path,
            verified_packet=verified,
            agent_source_bytes=_AGENT_BYTES,
            artifact_root=tmp_path,
            require_real_pass=False,
            evidence_signature_verifier=_fake_evidence_verify,
        )


def test_forged_self_consistent_evidence_without_host_signature_is_rejected(
    tmp_path: Path,
) -> None:
    verified = _load(tmp_path, _packet())
    evidence, log_bytes = execute_qualification(
        verified,
        repo_root=tmp_path,
        host_inventory=_inventory(),
        action_runner=_pass_runner,
        git_probe=_git_probe,
        agent_source_bytes=_AGENT_BYTES,
        evidence_signing_key_id=_HOST_KEY_ID,
        evidence_signer=_fake_evidence_signer,
    )
    evidence_path = tmp_path / "evidence.json"
    log_path = tmp_path / "run.jsonl"
    write_evidence_bundle(
        evidence_path,
        log_path,
        evidence=evidence,
        log_bytes=log_bytes,
    )

    payload = json.loads(evidence_path.read_text(encoding="utf-8"))
    payload["resource_observations"]["PROVIDER"] = "REAL_PROBED"
    attestation = payload.pop("evidence_attestation")
    payload.pop("evidence_identity_sha256")
    message = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["evidence_identity_sha256"] = hashlib.sha256(message).hexdigest()
    payload["evidence_attestation"] = attestation
    evidence_path.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="host attestation verification failed"):
        verify_qualification_evidence(
            evidence_path,
            log_path,
            verified_packet=verified,
            agent_source_bytes=_AGENT_BYTES,
            artifact_root=tmp_path,
            require_real_pass=False,
            evidence_signature_verifier=_fake_evidence_verify,
        )


def test_duplicate_json_members_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "packet.json"
    path.write_text(
        '{"schema_version":"x","schema_version":"y","packet":{},"signature":{}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="strict unambiguous"):
        load_verified_signed_packet(
            path,
            signature_verifier=_fake_verify,
            now_epoch_seconds=150,
        )
