from __future__ import annotations

import base64
import hashlib
import inspect
import json
import subprocess

import pytest

import twelve_six.evidence_control_bridge as bridge_module
import twelve_six.physical_qualification as physical_module
from twelve_six.evidence_control_bridge import (
    CanonicalEvidenceRecord,
    HostDispatch,
    PhysicalExecutionReceipt,
    RequalificationReceipt,
    build_canonical_evidence_record,
    build_host_dispatch,
    build_requalification_requirement,
    physical_failure_observation,
    write_canonical_evidence_record,
)
from twelve_six.physical_qualification import (
    ExecutionMode,
    QualificationAction,
    QualificationPacket,
    VerifiedSignedPacket,
)
from twelve_six.sil_qualification import build_package_manifest_bytes


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _git(char: str) -> str:
    return char * 40


def _dispatch(
    *,
    git_sha: str = _git("a"),
    package: str = _sha("package-a"),
    scenario: str = "host-smoke",
    gate: str = "windows-host",
) -> HostDispatch:
    return HostDispatch(
        schema_version="12-6.host-dispatch.v1",
        dispatch_id="dispatch-a",
        target_git_sha=git_sha,
        package_identity_sha256=package,
        packet_identity_sha256=_sha("packet-a"),
        signed_bundle_identity_sha256=_sha("bundle-a"),
        scenario_id=scenario,
        physical_gate_id=gate,
    )


def _receipt(
    dispatch: HostDispatch,
    *,
    verdict: str,
    evidence: str,
    reproducer: str | None = None,
) -> PhysicalExecutionReceipt:
    return PhysicalExecutionReceipt(
        schema_version="12-6.physical-execution-receipt.v1",
        dispatch_identity_sha256=dispatch.identity_sha256(),
        target_git_sha=dispatch.target_git_sha,
        package_identity_sha256=dispatch.package_identity_sha256,
        packet_identity_sha256=dispatch.packet_identity_sha256,
        signed_bundle_identity_sha256=dispatch.signed_bundle_identity_sha256,
        physical_evidence_identity_sha256=_sha(evidence),
        host_inventory_identity_sha256=_sha("host-inventory"),
        verdict=verdict,
        scenario_id=dispatch.scenario_id,
        physical_gate_id=dispatch.physical_gate_id,
        reproducer_command=reproducer,
        _verification_token=bridge_module._VERIFIED_PHYSICAL_RECEIPT,
    )


def _verified_packet(target_git_sha: str, *, mode: ExecutionMode) -> VerifiedSignedPacket:
    action = QualificationAction(
        action_id="smoke",
        pytest_targets=("tests/test_smoke.py",),
        timeout_seconds=60,
        max_output_bytes=4096,
        required_resources=(),
    )
    packet = QualificationPacket(
        schema_version="12-6.physical-qualification-packet.v1",
        packet_id="packet-a",
        target_git_sha=target_git_sha,
        agent_source_sha256=_sha("agent-source"),
        execution_mode=mode,
        allowed_os_families=("LINUX",),
        not_before_epoch_seconds=1,
        expires_epoch_seconds=61,
        actions=(action,),
        artifact_paths=(),
    )
    return VerifiedSignedPacket(
        packet=packet,
        signing_key_id="packet-key",
        signature_sha256=_sha("packet-signature"),
        signed_bundle_identity_sha256=_sha("packet-bundle"),
        _verification_token=physical_module._VERIFIED_PACKET_TOKEN,
    )


def _clean_package_repo(tmp_path) -> str:
    (tmp_path / "src" / "twelve_six").mkdir(parents=True)
    (tmp_path / "configs" / "research").mkdir(parents=True)
    (tmp_path / "tests").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(
        "[build-system]\\nrequires=[]\\nbuild-backend='setuptools.build_meta'\\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "twelve_six" / "__init__.py").write_text(
        "__all__ = []\\n",
        encoding="utf-8",
    )
    (tmp_path / "configs" / "research" / "smoke.json").write_text(
        "{}\\n",
        encoding="utf-8",
    )
    (tmp_path / "tests" / "test_smoke.py").write_text(
        "def test_smoke():\\n    assert True\\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "section6@example.invalid"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Section 6 Test"],
        cwd=tmp_path,
        check=True,
    )
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "fixture"],
        cwd=tmp_path,
        check=True,
    )
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_host_dispatch_derives_package_from_exact_clean_candidate(tmp_path) -> None:
    git_sha = _clean_package_repo(tmp_path)
    verified = _verified_packet(git_sha, mode=ExecutionMode.REAL_HOST)

    dispatch = build_host_dispatch(
        verified,
        repo_root=tmp_path,
        dispatch_id="dispatch-clean",
        scenario_id="host-smoke",
        physical_gate_id="linux-host",
    )

    assert dispatch.target_git_sha == git_sha
    assert dispatch.package_identity_sha256 == hashlib.sha256(
        build_package_manifest_bytes(tmp_path)
    ).hexdigest()


def test_host_dispatch_rejects_dirty_or_simulated_candidate(tmp_path) -> None:
    git_sha = _clean_package_repo(tmp_path)
    simulated = _verified_packet(git_sha, mode=ExecutionMode.SIMULATION)
    with pytest.raises(ValueError, match="REAL_HOST"):
        build_host_dispatch(
            simulated,
            repo_root=tmp_path,
            dispatch_id="dispatch-sim",
            scenario_id="host-smoke",
            physical_gate_id="linux-host",
        )

    (tmp_path / "src" / "twelve_six" / "__init__.py").write_text(
        "__all__ = ['dirty']\\n",
        encoding="utf-8",
    )
    real = _verified_packet(git_sha, mode=ExecutionMode.REAL_HOST)
    with pytest.raises(ValueError, match="dirty"):
        build_host_dispatch(
            real,
            repo_root=tmp_path,
            dispatch_id="dispatch-dirty",
            scenario_id="host-smoke",
            physical_gate_id="linux-host",
        )


def test_dispatch_identity_binds_package_candidate_and_scenario() -> None:
    base = _dispatch()
    changed_package = _dispatch(package=_sha("package-b"))
    changed_candidate = _dispatch(git_sha=_git("b"))
    changed_scenario = _dispatch(scenario="server-smoke")

    assert base.identity_sha256() != changed_package.identity_sha256()
    assert base.identity_sha256() != changed_candidate.identity_sha256()
    assert base.identity_sha256() != changed_scenario.identity_sha256()


def test_physical_receipt_cannot_be_forged_without_verification_token() -> None:
    dispatch = _dispatch()
    with pytest.raises(ValueError, match="must come from evidence verification"):
        PhysicalExecutionReceipt(
            schema_version="12-6.physical-execution-receipt.v1",
            dispatch_identity_sha256=dispatch.identity_sha256(),
            target_git_sha=dispatch.target_git_sha,
            package_identity_sha256=dispatch.package_identity_sha256,
            packet_identity_sha256=dispatch.packet_identity_sha256,
            signed_bundle_identity_sha256=(
                dispatch.signed_bundle_identity_sha256
            ),
            physical_evidence_identity_sha256=_sha("evidence"),
            host_inventory_identity_sha256=_sha("host"),
            verdict="PASS",
            scenario_id=dispatch.scenario_id,
            physical_gate_id=dispatch.physical_gate_id,
            reproducer_command=None,
        )


def test_pass_receipt_cannot_carry_failure_reproducer() -> None:
    dispatch = _dispatch()
    with pytest.raises(ValueError, match="PASS physical receipt"):
        _receipt(
            dispatch,
            verdict="PASS",
            evidence="pass-with-reproducer",
            reproducer="pytest -q tests/test_x.py",
        )


def test_canonical_record_cross_binds_dispatch_and_verified_receipt() -> None:
    dispatch = _dispatch()
    receipt = _receipt(dispatch, verdict="PASS", evidence="pass")
    record = build_canonical_evidence_record(dispatch, receipt)

    assert isinstance(record, CanonicalEvidenceRecord)
    assert record.target_git_sha == dispatch.target_git_sha
    assert record.package_identity_sha256 == dispatch.package_identity_sha256
    assert record.receipt_identity_sha256 == receipt.identity_sha256()


def test_canonical_record_rejects_receipt_from_other_dispatch() -> None:
    dispatch = _dispatch()
    other = _dispatch(git_sha=_git("b"))
    receipt = _receipt(other, verdict="PASS", evidence="other-pass")

    with pytest.raises(ValueError, match="different dispatch"):
        build_canonical_evidence_record(dispatch, receipt)


def test_canonical_evidence_write_is_create_only(tmp_path) -> None:
    dispatch = _dispatch()
    receipt = _receipt(dispatch, verdict="PASS", evidence="write-pass")
    record = build_canonical_evidence_record(dispatch, receipt)
    target = tmp_path / "evidence" / "receipt.json"

    write_canonical_evidence_record(target, record)
    first = target.read_bytes()
    assert first.endswith(b"\n")
    assert record.identity_sha256()

    with pytest.raises(FileExistsError):
        write_canonical_evidence_record(target, record)


def test_physical_failure_converts_to_aiqa_observation_only_with_reproducer() -> None:
    dispatch = _dispatch()
    receipt = _receipt(
        dispatch,
        verdict="FAIL",
        evidence="test-failure",
        reproducer="pytest -q tests/test_bridge.py",
    )
    observation = physical_failure_observation(receipt)

    assert observation.git_sha == dispatch.target_git_sha
    assert observation.physical_gate_id == dispatch.physical_gate_id
    assert observation.evidence_identity_sha256 == (
        receipt.physical_evidence_identity_sha256
    )


def test_infrastructure_failure_does_not_invent_pytest_reproducer() -> None:
    receipt = _receipt(
        _dispatch(),
        verdict="FAIL",
        evidence="infrastructure-failure",
        reproducer=None,
    )
    with pytest.raises(ValueError, match="no pytest reproducer"):
        physical_failure_observation(receipt)


def test_requalification_requires_changed_candidate_or_package() -> None:
    dispatch = _dispatch()
    failed = _receipt(
        dispatch,
        verdict="FAIL",
        evidence="old-failure",
        reproducer="pytest -q tests/test_bridge.py",
    )
    unchanged_package = b"same-package"
    same_dispatch = _dispatch(
        package=hashlib.sha256(unchanged_package).hexdigest(),
    )
    failed_same = _receipt(
        same_dispatch,
        verdict="FAIL",
        evidence="same-failure",
        reproducer="pytest -q tests/test_bridge.py",
    )

    with pytest.raises(ValueError, match="new candidate Git SHA"):
        build_requalification_requirement(
            failed_same,
            repaired_candidate_git_sha=same_dispatch.target_git_sha,
            repaired_package_manifest_bytes=unchanged_package,
        )

    requirement = build_requalification_requirement(
        failed,
        repaired_candidate_git_sha=_git("b"),
        repaired_package_manifest_bytes=b"repaired-package",
    )
    assert requirement.repaired_candidate_git_sha == _git("b")


def test_requalification_rejects_unchanged_package_even_on_new_sha() -> None:
    package_bytes = b"same-package"
    old_dispatch = _dispatch(
        package=hashlib.sha256(package_bytes).hexdigest(),
    )
    failed = _receipt(
        old_dispatch,
        verdict="FAIL",
        evidence="old-failure",
        reproducer="pytest -q tests/test_bridge.py",
    )

    with pytest.raises(ValueError, match="changed package identity"):
        build_requalification_requirement(
            failed,
            repaired_candidate_git_sha=_git("b"),
            repaired_package_manifest_bytes=package_bytes,
        )


def test_physical_failure_reproducer_is_derived_from_verified_action_argv() -> None:
    evidence = {
        "actions": [
            {
                "verdict": "FAIL",
                "argv": [
                    "/usr/bin/python3",
                    "-m",
                    "pytest",
                    "-q",
                    "tests/test_alpha.py",
                    "tests/test_beta.py",
                ],
            }
        ]
    }

    assert bridge_module._physical_failure_reproducer(evidence) == (
        "pytest -q tests/test_alpha.py tests/test_beta.py"
    )


def test_physical_failure_reproducer_rejects_noncanonical_argv() -> None:
    evidence = {
        "actions": [
            {
                "verdict": "FAIL",
                "argv": [
                    "/usr/bin/python3",
                    "-m",
                    "pytest",
                    "--maxfail=1",
                    "tests/test_alpha.py",
                ],
            }
        ]
    }

    with pytest.raises(ValueError, match="canonical pytest argv"):
        bridge_module._physical_failure_reproducer(evidence)


def test_requalification_receipt_cannot_be_forged_without_fresh_verification() -> None:
    with pytest.raises(ValueError, match="requires fresh verification"):
        RequalificationReceipt(
            schema_version="12-6.requalification-receipt.v1",
            requirement_identity_sha256=_sha("requirement"),
            repaired_candidate_git_sha=_git("b"),
            repaired_package_identity_sha256=_sha("package-b"),
            sil_evidence_identity_sha256=_sha("sil"),
            physical_evidence_identity_sha256=_sha("physical"),
            repaired_dispatch_identity_sha256=_sha("dispatch"),
            scenario_id="host-smoke",
            physical_gate_id="windows-host",
        )


def test_requalification_starts_only_from_physical_failure() -> None:
    passed = _receipt(_dispatch(), verdict="PASS", evidence="old-pass")
    with pytest.raises(ValueError, match="starts from a physical FAIL"):
        build_requalification_requirement(
            passed,
            repaired_candidate_git_sha=_git("b"),
            repaired_package_manifest_bytes=b"new-package",
        )


def test_requalification_preflight_accepts_fresh_same_scenario_physical_pass() -> None:
    old_dispatch = _dispatch()
    failed = _receipt(
        old_dispatch,
        verdict="FAIL",
        evidence="old-failure",
        reproducer="pytest -q tests/test_bridge.py",
    )
    package_bytes = b"repaired-package"
    requirement = build_requalification_requirement(
        failed,
        repaired_candidate_git_sha=_git("b"),
        repaired_package_manifest_bytes=package_bytes,
    )
    repaired_dispatch = HostDispatch(
        schema_version="12-6.host-dispatch.v1",
        dispatch_id="dispatch-b",
        target_git_sha=requirement.repaired_candidate_git_sha,
        package_identity_sha256=requirement.repaired_package_identity_sha256,
        packet_identity_sha256=_sha("packet-b"),
        signed_bundle_identity_sha256=_sha("bundle-b"),
        scenario_id=requirement.scenario_id,
        physical_gate_id=requirement.physical_gate_id,
    )
    physical_pass = _receipt(
        repaired_dispatch,
        verdict="PASS",
        evidence="fresh-physical-pass",
    )

    bridge_module._require_requalification_bindings(
        requirement,
        repaired_dispatch=repaired_dispatch,
        repaired_physical_receipt=physical_pass,
        expected_package_bytes=package_bytes,
    )


def test_old_physical_pass_cannot_transfer_to_repaired_dispatch() -> None:
    old_dispatch = _dispatch()
    failed = _receipt(
        old_dispatch,
        verdict="FAIL",
        evidence="old-failure",
        reproducer="pytest -q tests/test_bridge.py",
    )
    package_bytes = b"new-package"
    requirement = build_requalification_requirement(
        failed,
        repaired_candidate_git_sha=_git("b"),
        repaired_package_manifest_bytes=package_bytes,
    )
    repaired_dispatch = HostDispatch(
        schema_version="12-6.host-dispatch.v1",
        dispatch_id="dispatch-b",
        target_git_sha=_git("b"),
        package_identity_sha256=hashlib.sha256(package_bytes).hexdigest(),
        packet_identity_sha256=_sha("packet-b"),
        signed_bundle_identity_sha256=_sha("bundle-b"),
        scenario_id=old_dispatch.scenario_id,
        physical_gate_id=old_dispatch.physical_gate_id,
    )
    stale_pass = _receipt(old_dispatch, verdict="PASS", evidence="stale-pass")

    with pytest.raises(ValueError, match="fresh dispatch"):
        bridge_module._require_requalification_bindings(
            requirement,
            repaired_dispatch=repaired_dispatch,
            repaired_physical_receipt=stale_pass,
            expected_package_bytes=package_bytes,
        )


def test_repaired_dispatch_must_preserve_same_physical_scenario() -> None:
    old_dispatch = _dispatch()
    failed = _receipt(
        old_dispatch,
        verdict="FAIL",
        evidence="old-failure",
        reproducer="pytest -q tests/test_bridge.py",
    )
    package_bytes = b"new-package"
    requirement = build_requalification_requirement(
        failed,
        repaired_candidate_git_sha=_git("b"),
        repaired_package_manifest_bytes=package_bytes,
    )
    changed_scenario = HostDispatch(
        schema_version="12-6.host-dispatch.v1",
        dispatch_id="dispatch-b",
        target_git_sha=_git("b"),
        package_identity_sha256=hashlib.sha256(package_bytes).hexdigest(),
        packet_identity_sha256=_sha("packet-b"),
        signed_bundle_identity_sha256=_sha("bundle-b"),
        scenario_id="different-scenario",
        physical_gate_id=old_dispatch.physical_gate_id,
    )
    fresh_pass = _receipt(
        changed_scenario,
        verdict="PASS",
        evidence="fresh-pass",
    )

    with pytest.raises(ValueError, match="changes the physical scenario"):
        bridge_module._require_requalification_bindings(
            requirement,
            repaired_dispatch=changed_scenario,
            repaired_physical_receipt=fresh_pass,
            expected_package_bytes=package_bytes,
        )


def test_identity_hashing_is_sealed_against_helper_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dispatch = _dispatch()
    receipt = _receipt(dispatch, verdict="PASS", evidence="sealed-pass")
    record = build_canonical_evidence_record(dispatch, receipt)
    expected = (
        dispatch.identity_sha256(),
        receipt.identity_sha256(),
        record.identity_sha256(),
    )

    monkeypatch.setattr(
        bridge_module,
        "_canonical_identity",
        lambda _payload: "0" * 64,
    )

    assert (
        dispatch.identity_sha256(),
        receipt.identity_sha256(),
        record.identity_sha256(),
    ) == expected


def test_verification_entrypoints_expose_no_verifier_override_bypass() -> None:
    physical_parameters = inspect.signature(
        bridge_module.verify_physical_execution
    ).parameters
    requalification_parameters = inspect.signature(
        bridge_module.qualify_repaired_candidate
    ).parameters

    assert "qualification_verifier" not in physical_parameters
    assert "sil_verifier" not in requalification_parameters

    for schema in (
        HostDispatch,
        PhysicalExecutionReceipt,
        CanonicalEvidenceRecord,
        bridge_module.RequalificationRequirement,
        RequalificationReceipt,
    ):
        assert tuple(inspect.signature(schema.identity_sha256).parameters) == (
            "instance",
        )


def _signed_bridge_fixture():
    raw_signature = bytes(range(64))
    provisional = _verified_packet(_git("a"), mode=ExecutionMode.REAL_HOST)
    verified = VerifiedSignedPacket(
        packet=provisional.packet,
        signing_key_id=provisional.signing_key_id,
        signature_sha256=hashlib.sha256(raw_signature).hexdigest(),
        signed_bundle_identity_sha256=provisional.signed_bundle_identity_sha256,
        _verification_token=physical_module._VERIFIED_PACKET_TOKEN,
    )
    dispatch = HostDispatch(
        schema_version="12-6.host-dispatch.v1",
        dispatch_id="signed-dispatch",
        target_git_sha=verified.packet.target_git_sha,
        package_identity_sha256=_sha("package-a"),
        packet_identity_sha256=verified.packet.identity_sha256(),
        signed_bundle_identity_sha256=verified.signed_bundle_identity_sha256,
        scenario_id="host-smoke",
        physical_gate_id="linux-host",
    )
    encoded = json.dumps(
        {
            "schema_version": "12-6.signed-physical-qualification-packet.v1",
            "packet": verified.packet.to_dict(),
            "signature": {
                "algorithm": "ED25519",
                "key_id": verified.signing_key_id,
                "signature_b64": base64.b64encode(raw_signature).decode("ascii"),
            },
        },
        sort_keys=True,
    ).encode()
    return dispatch, verified, encoded


def test_bridge_handoff_claim_restart_and_stop(tmp_path) -> None:
    dispatch, verified, signed = _signed_bridge_fixture()
    first = bridge_module.stage_bridge_dispatch(
        tmp_path, dispatch=dispatch, verified_packet=verified, signed_packet_bytes=signed
    )
    assert first["automatic_retry_allowed"] is False
    assert bridge_module.stage_bridge_dispatch(
        tmp_path, dispatch=dispatch, verified_packet=verified, signed_packet_bytes=signed
    ) == first
    key = dispatch.identity_sha256()
    assert bridge_module.bridge_status(tmp_path, key)["state"] == "READY"
    bridge_module.claim_bridge_dispatch(tmp_path, key)
    assert bridge_module.bridge_status(tmp_path, key)["state"] == "CLAIMED_OUTCOME_UNKNOWN"
    with pytest.raises(ValueError, match="no automatic retry"):
        bridge_module.claim_bridge_dispatch(tmp_path, key)
    bridge_module.stop_bridge_dispatch(tmp_path, key)
    bridge_module.stop_bridge_dispatch(tmp_path, key)
    assert bridge_module.bridge_status(tmp_path, key)["state"] == "STOP_REQUESTED"
    with pytest.raises(ValueError, match="no automatic retry"):
        bridge_module.claim_bridge_dispatch(tmp_path, key)


def test_bridge_different_packet_replay_and_corrupt_handoff_fail_closed(tmp_path) -> None:
    dispatch, verified, signed = _signed_bridge_fixture()
    bridge_module.stage_bridge_dispatch(
        tmp_path, dispatch=dispatch, verified_packet=verified, signed_packet_bytes=signed
    )
    bad = signed.replace(b"ED25519", b"UNKNOWN")
    with pytest.raises(ValueError, match="does not bind"):
        bridge_module.stage_bridge_dispatch(
            tmp_path, dispatch=dispatch, verified_packet=verified, signed_packet_bytes=bad
        )
    packet_file = tmp_path / "packets" / (dispatch.identity_sha256() + ".json")
    packet_file.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="different bytes"):
        bridge_module.stage_bridge_dispatch(
            tmp_path, dispatch=dispatch, verified_packet=verified, signed_packet_bytes=signed
        )
    with pytest.raises(ValueError, match="handoff mismatch"):
        bridge_module.bridge_status(tmp_path, dispatch.identity_sha256())


def test_bridge_forged_signed_packet_and_symlink_denied(tmp_path) -> None:
    dispatch, verified, signed = _signed_bridge_fixture()
    with pytest.raises(ValueError, match="malformed"):
        bridge_module.stage_bridge_dispatch(
            tmp_path, dispatch=dispatch, verified_packet=verified,
            signed_packet_bytes=b'{"signature":{},"signature":{}}',
        )
    with pytest.raises(ValueError, match="bind"):
        bridge_module.stage_bridge_dispatch(
            tmp_path, dispatch=dispatch, verified_packet=verified,
            signed_packet_bytes=signed.replace(b"packet-a", b"packet-b"),
        )
    spool_link = tmp_path / "spool-link"
    spool_link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        bridge_module.stage_bridge_dispatch(
            spool_link, dispatch=dispatch, verified_packet=verified,
            signed_packet_bytes=signed,
        )


def test_bridge_missing_claim_cannot_publish_unverified_host_result(tmp_path) -> None:
    dispatch, verified, signed = _signed_bridge_fixture()
    bridge_module.stage_bridge_dispatch(
        tmp_path, dispatch=dispatch, verified_packet=verified, signed_packet_bytes=signed
    )
    evidence = tmp_path / "receipt.json"
    log = tmp_path / "log.bin"
    evidence.write_text("{}", encoding="utf-8")
    log.write_bytes(b"log")
    with pytest.raises(ValueError, match="prior claimed attempt"):
        bridge_module.record_bridge_return(
            tmp_path, dispatch=dispatch, verified_packet=verified,
            evidence_path=evidence, log_path=log,
            agent_source_bytes=b"source", artifact_root=tmp_path,
            evidence_signature_verifier=lambda *_args: False,
        )


def test_bridge_return_refuses_forged_signature_even_after_claim(tmp_path) -> None:
    dispatch, verified, signed = _signed_bridge_fixture()
    bridge_module.stage_bridge_dispatch(
        tmp_path, dispatch=dispatch, verified_packet=verified, signed_packet_bytes=signed
    )
    bridge_module.claim_bridge_dispatch(tmp_path, dispatch.identity_sha256())
    evidence = tmp_path / "receipt.json"
    log = tmp_path / "log.bin"
    evidence.write_text("{}", encoding="utf-8")
    log.write_bytes(b"log")
    with pytest.raises((ValueError, KeyError, RuntimeError)):
        bridge_module.record_bridge_return(
            tmp_path, dispatch=dispatch, verified_packet=verified,
            evidence_path=evidence, log_path=log,
            agent_source_bytes=b"source", artifact_root=tmp_path,
            evidence_signature_verifier=lambda *_args: False,
        )
    assert bridge_module.bridge_status(
        tmp_path, dispatch.identity_sha256()
    )["state"] == "CLAIMED_OUTCOME_UNKNOWN"


@pytest.mark.parametrize("reproducer,expected_kind", [
    ("pytest -q tests/test_smoke.py", "REPRODUCIBLE_TEST_FAILURE"),
    (None, "PHYSICAL_INFRASTRUCTURE_FAILURE"),
])
def test_bridge_signed_failure_return_is_durable_and_idempotent(
    tmp_path, monkeypatch, reproducer, expected_kind,
) -> None:
    dispatch, verified, signed = _signed_bridge_fixture()
    bridge_module.stage_bridge_dispatch(
        tmp_path, dispatch=dispatch, verified_packet=verified, signed_packet_bytes=signed
    )
    bridge_module.claim_bridge_dispatch(tmp_path, dispatch.identity_sha256())
    sealed = _receipt(
        dispatch, verdict="FAIL", evidence="signed-physical-failure",
        reproducer=reproducer,
    )
    # Transport test double only: independent S4 tests execute the real signer verifier.
    monkeypatch.setattr(
        bridge_module, "verify_physical_execution", lambda *_args, **_kwargs: sealed
    )
    evidence = tmp_path / "receipt.json"
    log = tmp_path / "log.bin"
    evidence.write_text('{"signed":true}', encoding="utf-8")
    log.write_bytes(b"physical log")
    options = {
        "dispatch": dispatch, "verified_packet": verified, "evidence_path": evidence,
        "log_path": log, "agent_source_bytes": b"source", "artifact_root": tmp_path,
        "evidence_signature_verifier": lambda *_args: False,
    }
    first = bridge_module.record_bridge_return(tmp_path, **options)
    assert first["defect"]["kind"] == expected_kind
    assert bridge_module.record_bridge_return(tmp_path, **options) == first
    assert bridge_module.bridge_status(
        tmp_path, dispatch.identity_sha256()
    )["state"] == "EVIDENCE_RECORDED"
    evidence.write_text('{"signed":false}', encoding="utf-8")
    with pytest.raises(ValueError, match="different bytes"):
        bridge_module.record_bridge_return(tmp_path, **options)


def test_github_bridge_publish_is_create_only_and_hash_only(tmp_path, monkeypatch) -> None:
    dispatch, verified, signed = _signed_bridge_fixture()
    bridge_module.stage_bridge_dispatch(
        tmp_path, dispatch=dispatch, verified_packet=verified, signed_packet_bytes=signed
    )
    bridge_module.claim_bridge_dispatch(tmp_path, dispatch.identity_sha256())
    sealed = _receipt(dispatch, verdict="FAIL", evidence="physical", reproducer=None)
    monkeypatch.setattr(
        bridge_module, "verify_physical_execution", lambda *_args, **_kwargs: sealed
    )
    receipt = tmp_path / "receipt.json"
    log = tmp_path / "private-log.bin"
    receipt.write_bytes(b'{"sealed":true}')
    log.write_bytes(b"SECRET_LOCAL_LOG_NOT_FOR_GITHUB")
    remote: dict[str, bytes] = {}
    calls = []

    def pretend_github(method, path, *, token, branch, upload=None):
        calls.append((method, path, branch))
        assert token == "dummy-secret-token"
        assert branch == "plan8-evidence"
        if method == "GET":
            if path not in remote:
                return 404, {}
            return 200, {
                "type": "file", "sha": bridge_module._git_blob_identity(remote[path])
            }
        assert method == "PUT" and path not in remote
        remote[path] = upload
        return 201, {
            "content": {"sha": bridge_module._git_blob_identity(upload)}
        }

    monkeypatch.setattr(bridge_module, "_github_evidence_request", pretend_github)
    args = {
        "dispatch": dispatch, "verified_packet": verified, "evidence_path": receipt,
        "log_path": log, "agent_source_bytes": b"source", "artifact_root": tmp_path,
        "evidence_signature_verifier": lambda *_args: False,
        "github_token": "dummy-secret-token",
    }
    first = bridge_module.publish_bridge_return_github(tmp_path, **args)
    assert len(first["publication"]) == 2
    assert all(x["state"] == "VERIFIED_CREATED" for x in first["publication"])
    assert len(remote) == 2
    assert all(b"SECRET_LOCAL_LOG_NOT_FOR_GITHUB" not in b for b in remote.values())
    second = bridge_module.publish_bridge_return_github(tmp_path, **args)
    assert all(x["state"] == "ALREADY_IDENTICAL" for x in second["publication"])
    assert len([x for x in calls if x[0] == "PUT"]) == 2


def test_github_bridge_denies_main_and_ambiguous_remote_conflict(tmp_path, monkeypatch):
    dispatch, verified, signed = _signed_bridge_fixture()
    bridge_module.stage_bridge_dispatch(
        tmp_path, dispatch=dispatch, verified_packet=verified, signed_packet_bytes=signed
    )
    bridge_module.claim_bridge_dispatch(tmp_path, dispatch.identity_sha256())
    sealed = _receipt(dispatch, verdict="PASS", evidence="physical")
    monkeypatch.setattr(
        bridge_module, "verify_physical_execution", lambda *_args, **_kwargs: sealed
    )
    receipt = tmp_path / "receipt.json"
    log = tmp_path / "log.bin"
    receipt.write_bytes(b'{"sealed":true}')
    log.write_bytes(b"local log")
    kwargs = {
        "dispatch": dispatch, "verified_packet": verified, "evidence_path": receipt,
        "log_path": log, "agent_source_bytes": b"source", "artifact_root": tmp_path,
        "evidence_signature_verifier": lambda *_args: False,
        "github_token": "dummy-secret-token",
    }
    with pytest.raises(ValueError, match="dedicated evidence branch"):
        bridge_module.publish_bridge_return_github(
            tmp_path, evidence_branch="main", **kwargs
        )
    monkeypatch.setattr(
        bridge_module, "_github_evidence_request",
        lambda method, path, **_other: (404, {}) if method == "GET" else (422, {}),
    )
    with pytest.raises(ValueError, match="conflict cannot be reconciled"):
        bridge_module.publish_bridge_return_github(tmp_path, **kwargs)


def test_github_bridge_rejects_missing_secret_without_network(monkeypatch):
    def no_http(_request, **_options):
        raise AssertionError("must never contact internet with empty credential")

    monkeypatch.setattr(bridge_module, "build_opener", lambda *_args: no_http)
    with pytest.raises(ValueError, match="token"):
        bridge_module._github_evidence_request(
            "GET", "evidence/plan8/test/bridge-return.json",
            token="", branch="plan8-evidence",
        )


def test_bridge_stop_interleaved_claim_fences_execution(tmp_path, monkeypatch) -> None:
    dispatch, verified, signed = _signed_bridge_fixture()
    bridge_module.stage_bridge_dispatch(
        tmp_path, dispatch=dispatch, verified_packet=verified,
        signed_packet_bytes=signed,
    )
    real_write = bridge_module._bridge_write_once
    raced = False

    def race_stop(path, payload):
        nonlocal raced
        if path.parent.name == "claims" and not raced:
            raced = True
            bridge_module.stop_bridge_dispatch(tmp_path, dispatch.identity_sha256())
        return real_write(path, payload)

    monkeypatch.setattr(bridge_module, "_bridge_write_once", race_stop)
    with pytest.raises(ValueError, match="STOP_REQUESTED"):
        bridge_module.claim_bridge_dispatch(tmp_path, dispatch.identity_sha256())
    assert raced
    assert bridge_module.bridge_status(
        tmp_path, dispatch.identity_sha256()
    )["state"] == "STOP_REQUESTED"
    with pytest.raises(ValueError, match="no automatic retry"):
        bridge_module.claim_bridge_dispatch(tmp_path, dispatch.identity_sha256())


def test_bridge_stop_after_claim_preserves_verified_receipt_recovery(
    tmp_path, monkeypatch,
) -> None:
    dispatch, verified, signed = _signed_bridge_fixture()
    bridge_module.stage_bridge_dispatch(
        tmp_path, dispatch=dispatch, verified_packet=verified,
        signed_packet_bytes=signed,
    )
    bridge_module.claim_bridge_dispatch(tmp_path, dispatch.identity_sha256())
    bridge_module.stop_bridge_dispatch(tmp_path, dispatch.identity_sha256())
    receipt = _receipt(dispatch, verdict="PASS", evidence="post-stop-pass")
    # Transport test double only; S4 tests independently test signature validation.
    monkeypatch.setattr(
        bridge_module, "verify_physical_execution", lambda *_a, **_k: receipt,
    )
    host_receipt = tmp_path / "host-receipt.json"
    host_log = tmp_path / "host-log.bin"
    host_receipt.write_bytes(b'{"signed":true}')
    host_log.write_bytes(b"host evidence")
    kwargs = {
        "dispatch": dispatch, "verified_packet": verified,
        "evidence_path": host_receipt, "log_path": host_log,
        "agent_source_bytes": b"agent", "artifact_root": tmp_path,
        "evidence_signature_verifier": lambda *_a: False,
    }
    result = bridge_module.record_bridge_return(tmp_path, **kwargs)
    assert result["verdict"] == "PASS"
    assert bridge_module.record_bridge_return(tmp_path, **kwargs) == result
    assert (
        tmp_path / "outbound" / dispatch.identity_sha256() / "evidence-record.json"
    ).is_file()
    with pytest.raises(ValueError, match="no automatic retry"):
        bridge_module.claim_bridge_dispatch(tmp_path, dispatch.identity_sha256())
