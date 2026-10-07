from __future__ import annotations

import hashlib
import inspect

import pytest

import twelve_six.evidence_control_bridge as bridge_module
from twelve_six.evidence_control_bridge import (
    CanonicalEvidenceRecord,
    HostDispatch,
    PhysicalExecutionReceipt,
    RequalificationReceipt,
    build_canonical_evidence_record,
    build_requalification_requirement,
    physical_failure_observation,
    write_canonical_evidence_record,
)


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
