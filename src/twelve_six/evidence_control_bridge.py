from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from twelve_six.ai_qa_control import ExternalObservation, FailureSource
from twelve_six.capability_map import CapabilityRegistry
from twelve_six.physical_qualification import (
    ExecutionMode,
    ExternalResourceVerifier,
    ResourceKind,
    SignatureVerifier,
    VerifiedSignedPacket,
    verify_qualification_evidence,
)
from twelve_six.sil_qualification import (
    SILScenario,
    build_package_manifest_bytes,
    parse_vector_command,
    require_exact_clean_git_state,
    verify_sil_evidence,
)


_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_VERIFIED_PHYSICAL_RECEIPT = object()
_VERIFIED_REQUALIFICATION_RECEIPT = object()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_identity(value: Any) -> str:
    return _sha256_bytes(_canonical_json_bytes(value))


def _sealed_identity_method(
    identity_func: Callable[[Any], str],
) -> Callable[[Any], str]:
    def identity_sha256(instance: Any) -> str:
        return identity_func(instance.to_dict())

    return identity_sha256


def _is_exact_type(value: object, expected: type[object]) -> bool:
    return type(value) is expected  # noqa: E721


def _require_text(name: str, value: object) -> str:
    if not _is_exact_type(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty text")
    return value


def _require_id(name: str, value: object) -> str:
    if not _is_exact_type(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical identifier")
    return value


def _require_sha40(name: str, value: object) -> str:
    if not _is_exact_type(value, str) or _SHA40_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase 40-hex Git SHA")
    return value


def _require_sha256(name: str, value: object) -> str:
    if not _is_exact_type(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return value


@dataclass(frozen=True, slots=True)
class HostDispatch:
    schema_version: str
    dispatch_id: str
    target_git_sha: str
    package_identity_sha256: str
    packet_identity_sha256: str
    signed_bundle_identity_sha256: str
    scenario_id: str
    physical_gate_id: str

    def __post_init__(self) -> None:
        if (
            not _is_exact_type(self.schema_version, str)
            or self.schema_version != "12-6.host-dispatch.v1"
        ):
            raise ValueError("unsupported host dispatch schema")
        _require_id("dispatch_id", self.dispatch_id)
        _require_sha40("target_git_sha", self.target_git_sha)
        _require_sha256("package_identity_sha256", self.package_identity_sha256)
        _require_sha256("packet_identity_sha256", self.packet_identity_sha256)
        _require_sha256(
            "signed_bundle_identity_sha256",
            self.signed_bundle_identity_sha256,
        )
        _require_id("scenario_id", self.scenario_id)
        _require_id("physical_gate_id", self.physical_gate_id)

    def to_dict(self) -> dict[str, Any]:
        HostDispatch.__post_init__(self)
        return {
            "schema_version": self.schema_version,
            "dispatch_id": self.dispatch_id,
            "target_git_sha": self.target_git_sha,
            "package_identity_sha256": self.package_identity_sha256,
            "packet_identity_sha256": self.packet_identity_sha256,
            "signed_bundle_identity_sha256": self.signed_bundle_identity_sha256,
            "scenario_id": self.scenario_id,
            "physical_gate_id": self.physical_gate_id,
        }

    identity_sha256 = _sealed_identity_method(_canonical_identity)


def build_host_dispatch(
    verified_packet: VerifiedSignedPacket,
    *,
    repo_root: str | Path,
    dispatch_id: str,
    scenario_id: str,
    physical_gate_id: str,
) -> HostDispatch:
    if not _is_exact_type(verified_packet, VerifiedSignedPacket):
        raise ValueError("verified_packet must be an exact VerifiedSignedPacket")
    VerifiedSignedPacket.__post_init__(verified_packet)
    if verified_packet.packet.execution_mode is not ExecutionMode.REAL_HOST:
        raise ValueError("Section 6 physical dispatch requires REAL_HOST mode")
    require_exact_clean_git_state(
        repo_root,
        verified_packet.packet.target_git_sha,
    )
    package_manifest_bytes = build_package_manifest_bytes(repo_root)
    if not package_manifest_bytes:
        raise ValueError("canonical package manifest cannot be empty")
    return HostDispatch(
        schema_version="12-6.host-dispatch.v1",
        dispatch_id=dispatch_id,
        target_git_sha=verified_packet.packet.target_git_sha,
        package_identity_sha256=_sha256_bytes(package_manifest_bytes),
        packet_identity_sha256=verified_packet.packet.identity_sha256(),
        signed_bundle_identity_sha256=(
            verified_packet.signed_bundle_identity_sha256
        ),
        scenario_id=scenario_id,
        physical_gate_id=physical_gate_id,
    )


@dataclass(frozen=True, slots=True)
class PhysicalExecutionReceipt:
    schema_version: str
    dispatch_identity_sha256: str
    target_git_sha: str
    package_identity_sha256: str
    packet_identity_sha256: str
    signed_bundle_identity_sha256: str
    physical_evidence_identity_sha256: str
    host_inventory_identity_sha256: str
    verdict: str
    scenario_id: str
    physical_gate_id: str
    reproducer_command: str | None
    _verification_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._verification_token is not _VERIFIED_PHYSICAL_RECEIPT:
            raise ValueError("physical receipt must come from evidence verification")
        if (
            not _is_exact_type(self.schema_version, str)
            or self.schema_version != "12-6.physical-execution-receipt.v1"
        ):
            raise ValueError("unsupported physical execution receipt schema")
        _require_sha256(
            "dispatch_identity_sha256",
            self.dispatch_identity_sha256,
        )
        _require_sha40("target_git_sha", self.target_git_sha)
        for name, value in (
            ("package_identity_sha256", self.package_identity_sha256),
            ("packet_identity_sha256", self.packet_identity_sha256),
            (
                "signed_bundle_identity_sha256",
                self.signed_bundle_identity_sha256,
            ),
            (
                "physical_evidence_identity_sha256",
                self.physical_evidence_identity_sha256,
            ),
            (
                "host_inventory_identity_sha256",
                self.host_inventory_identity_sha256,
            ),
        ):
            _require_sha256(name, value)
        if not _is_exact_type(self.verdict, str) or self.verdict not in {
            "PASS",
            "FAIL",
        }:
            raise ValueError("physical receipt verdict must be PASS or FAIL")
        _require_id("scenario_id", self.scenario_id)
        _require_id("physical_gate_id", self.physical_gate_id)
        if self.reproducer_command is not None:
            _require_text("reproducer_command", self.reproducer_command)
            parse_vector_command(self.reproducer_command)
        if self.verdict == "PASS" and self.reproducer_command is not None:
            raise ValueError("PASS physical receipt cannot carry a failure reproducer")

    def to_dict(self) -> dict[str, Any]:
        PhysicalExecutionReceipt.__post_init__(self)
        return {
            "schema_version": self.schema_version,
            "dispatch_identity_sha256": self.dispatch_identity_sha256,
            "target_git_sha": self.target_git_sha,
            "package_identity_sha256": self.package_identity_sha256,
            "packet_identity_sha256": self.packet_identity_sha256,
            "signed_bundle_identity_sha256": self.signed_bundle_identity_sha256,
            "physical_evidence_identity_sha256": (
                self.physical_evidence_identity_sha256
            ),
            "host_inventory_identity_sha256": (
                self.host_inventory_identity_sha256
            ),
            "verdict": self.verdict,
            "scenario_id": self.scenario_id,
            "physical_gate_id": self.physical_gate_id,
            "reproducer_command": self.reproducer_command,
        }

    identity_sha256 = _sealed_identity_method(_canonical_identity)


def _physical_failure_reproducer(evidence: dict[str, Any]) -> str | None:
    actions = evidence.get("actions")
    if not _is_exact_type(actions, list):
        raise ValueError("physical evidence actions must be an array")
    for action in actions:
        if not _is_exact_type(action, dict):
            raise ValueError("physical evidence action must be an exact object")
        if action.get("verdict") != "FAIL":
            continue
        argv = action.get("argv")
        if (
            not _is_exact_type(argv, list)
            or len(argv) < 5
            or argv[1:4] != ["-m", "pytest", "-q"]
            or any(not _is_exact_type(item, str) for item in argv)
        ):
            raise ValueError("physical FAIL has no canonical pytest argv")
        command = "pytest -q " + " ".join(shlex.quote(item) for item in argv[4:])
        parse_vector_command(command)
        return command
    return None


def verify_physical_execution(
    dispatch: HostDispatch,
    evidence_path: str | Path,
    log_path: str | Path,
    *,
    verified_packet: VerifiedSignedPacket,
    agent_source_bytes: bytes,
    artifact_root: str | Path,
    evidence_signature_verifier: SignatureVerifier,
    resource_probe_verifiers: (
        dict[ResourceKind, ExternalResourceVerifier] | None
    ) = None,
) -> PhysicalExecutionReceipt:
    if not _is_exact_type(dispatch, HostDispatch):
        raise ValueError("dispatch must be an exact HostDispatch")
    HostDispatch.__post_init__(dispatch)
    if not _is_exact_type(verified_packet, VerifiedSignedPacket):
        raise ValueError("verified_packet must be an exact VerifiedSignedPacket")
    VerifiedSignedPacket.__post_init__(verified_packet)
    packet = verified_packet.packet
    if packet.execution_mode is not ExecutionMode.REAL_HOST:
        raise ValueError("physical bridge accepts only REAL_HOST packets")
    if packet.target_git_sha != dispatch.target_git_sha:
        raise ValueError("dispatch Git SHA differs from signed packet")
    if packet.identity_sha256() != dispatch.packet_identity_sha256:
        raise ValueError("dispatch packet identity differs from signed packet")
    if (
        verified_packet.signed_bundle_identity_sha256
        != dispatch.signed_bundle_identity_sha256
    ):
        raise ValueError("dispatch signed bundle differs from verified packet")

    evidence = verify_qualification_evidence(
        evidence_path,
        log_path,
        verified_packet=verified_packet,
        agent_source_bytes=agent_source_bytes,
        artifact_root=artifact_root,
        require_real_pass=False,
        evidence_signature_verifier=evidence_signature_verifier,
        resource_probe_verifiers=resource_probe_verifiers,
    )
    if not _is_exact_type(evidence, dict):
        raise ValueError("physical verifier must return an exact evidence object")
    if evidence.get("execution_mode") != "REAL_HOST":
        raise ValueError("simulation evidence cannot enter the physical bridge")
    if evidence.get("target_git_sha") != dispatch.target_git_sha:
        raise ValueError("physical evidence Git SHA differs from dispatch")
    if evidence.get("packet_identity_sha256") != dispatch.packet_identity_sha256:
        raise ValueError("physical evidence packet identity differs from dispatch")
    if (
        evidence.get("signed_bundle_identity_sha256")
        != dispatch.signed_bundle_identity_sha256
    ):
        raise ValueError("physical evidence signed bundle differs from dispatch")
    verdict = evidence.get("verdict")
    if not _is_exact_type(verdict, str) or verdict not in {"PASS", "FAIL"}:
        raise ValueError("REAL_HOST evidence must have PASS or FAIL verdict")

    reproducer = (
        _physical_failure_reproducer(evidence) if verdict == "FAIL" else None
    )
    return PhysicalExecutionReceipt(
        schema_version="12-6.physical-execution-receipt.v1",
        dispatch_identity_sha256=dispatch.identity_sha256(),
        target_git_sha=dispatch.target_git_sha,
        package_identity_sha256=dispatch.package_identity_sha256,
        packet_identity_sha256=dispatch.packet_identity_sha256,
        signed_bundle_identity_sha256=(
            dispatch.signed_bundle_identity_sha256
        ),
        physical_evidence_identity_sha256=_require_sha256(
            "physical evidence identity",
            evidence.get("evidence_identity_sha256"),
        ),
        host_inventory_identity_sha256=_require_sha256(
            "host inventory identity",
            evidence.get("host_inventory_identity_sha256"),
        ),
        verdict=verdict,
        scenario_id=dispatch.scenario_id,
        physical_gate_id=dispatch.physical_gate_id,
        reproducer_command=reproducer,
        _verification_token=_VERIFIED_PHYSICAL_RECEIPT,
    )


def physical_failure_observation(
    receipt: PhysicalExecutionReceipt,
) -> ExternalObservation:
    if not _is_exact_type(receipt, PhysicalExecutionReceipt):
        raise ValueError("receipt must be an exact PhysicalExecutionReceipt")
    PhysicalExecutionReceipt.__post_init__(receipt)
    if receipt.verdict != "FAIL":
        raise ValueError("only a physical FAIL can become an AI QA observation")
    if receipt.reproducer_command is None:
        raise ValueError(
            "physical infrastructure FAIL has no pytest reproducer; "
            "preserve it as a physical defect packet instead of fabricating one"
        )
    return ExternalObservation(
        schema_version="12-6.aiqa-observation.v1",
        source=FailureSource.PHYSICAL,
        git_sha=receipt.target_git_sha,
        evidence_identity_sha256=receipt.physical_evidence_identity_sha256,
        failure_summary=(
            "physical qualification failed for scenario "
            f"{receipt.scenario_id}"
        ),
        reproducer_command=receipt.reproducer_command,
        physical_gate_id=receipt.physical_gate_id,
    )


@dataclass(frozen=True, slots=True)
class CanonicalEvidenceRecord:
    schema_version: str
    dispatch_identity_sha256: str
    receipt_identity_sha256: str
    target_git_sha: str
    package_identity_sha256: str
    physical_evidence_identity_sha256: str
    verdict: str
    scenario_id: str
    physical_gate_id: str

    def __post_init__(self) -> None:
        if (
            not _is_exact_type(self.schema_version, str)
            or self.schema_version != "12-6.canonical-evidence-record.v1"
        ):
            raise ValueError("unsupported canonical evidence record schema")
        for name, value in (
            ("dispatch_identity_sha256", self.dispatch_identity_sha256),
            ("receipt_identity_sha256", self.receipt_identity_sha256),
            (
                "physical_evidence_identity_sha256",
                self.physical_evidence_identity_sha256,
            ),
            ("package_identity_sha256", self.package_identity_sha256),
        ):
            _require_sha256(name, value)
        _require_sha40("target_git_sha", self.target_git_sha)
        if not _is_exact_type(self.verdict, str) or self.verdict not in {
            "PASS",
            "FAIL",
        }:
            raise ValueError("canonical evidence verdict must be PASS or FAIL")
        _require_id("scenario_id", self.scenario_id)
        _require_id("physical_gate_id", self.physical_gate_id)

    def to_dict(self) -> dict[str, Any]:
        CanonicalEvidenceRecord.__post_init__(self)
        return {
            "schema_version": self.schema_version,
            "dispatch_identity_sha256": self.dispatch_identity_sha256,
            "receipt_identity_sha256": self.receipt_identity_sha256,
            "target_git_sha": self.target_git_sha,
            "package_identity_sha256": self.package_identity_sha256,
            "physical_evidence_identity_sha256": (
                self.physical_evidence_identity_sha256
            ),
            "verdict": self.verdict,
            "scenario_id": self.scenario_id,
            "physical_gate_id": self.physical_gate_id,
        }

    identity_sha256 = _sealed_identity_method(_canonical_identity)


def build_canonical_evidence_record(
    dispatch: HostDispatch,
    receipt: PhysicalExecutionReceipt,
) -> CanonicalEvidenceRecord:
    if not _is_exact_type(dispatch, HostDispatch):
        raise ValueError("dispatch must be an exact HostDispatch")
    if not _is_exact_type(receipt, PhysicalExecutionReceipt):
        raise ValueError("receipt must be an exact PhysicalExecutionReceipt")
    HostDispatch.__post_init__(dispatch)
    PhysicalExecutionReceipt.__post_init__(receipt)
    if receipt.dispatch_identity_sha256 != dispatch.identity_sha256():
        raise ValueError("physical receipt belongs to a different dispatch")
    if receipt.target_git_sha != dispatch.target_git_sha:
        raise ValueError("physical receipt belongs to a different candidate")
    if receipt.package_identity_sha256 != dispatch.package_identity_sha256:
        raise ValueError("physical receipt belongs to a different package")
    if (
        receipt.scenario_id != dispatch.scenario_id
        or receipt.physical_gate_id != dispatch.physical_gate_id
    ):
        raise ValueError("physical receipt belongs to a different scenario")
    return CanonicalEvidenceRecord(
        schema_version="12-6.canonical-evidence-record.v1",
        dispatch_identity_sha256=dispatch.identity_sha256(),
        receipt_identity_sha256=receipt.identity_sha256(),
        target_git_sha=dispatch.target_git_sha,
        package_identity_sha256=dispatch.package_identity_sha256,
        physical_evidence_identity_sha256=(
            receipt.physical_evidence_identity_sha256
        ),
        verdict=receipt.verdict,
        scenario_id=dispatch.scenario_id,
        physical_gate_id=dispatch.physical_gate_id,
    )


def write_canonical_evidence_record(
    path: str | Path,
    record: CanonicalEvidenceRecord,
) -> None:
    if not _is_exact_type(record, CanonicalEvidenceRecord):
        raise ValueError("record must be an exact CanonicalEvidenceRecord")
    CanonicalEvidenceRecord.__post_init__(record)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("xb") as stream:
        stream.write(_canonical_json_bytes(record.to_dict()) + b"\n")


@dataclass(frozen=True, slots=True)
class RequalificationRequirement:
    schema_version: str
    prior_dispatch_identity_sha256: str
    prior_candidate_git_sha: str
    prior_package_identity_sha256: str
    prior_physical_evidence_identity_sha256: str
    repaired_candidate_git_sha: str
    repaired_package_identity_sha256: str
    scenario_id: str
    physical_gate_id: str

    def __post_init__(self) -> None:
        if (
            not _is_exact_type(self.schema_version, str)
            or self.schema_version != "12-6.requalification-requirement.v1"
        ):
            raise ValueError("unsupported requalification requirement schema")
        _require_sha256(
            "prior_dispatch_identity_sha256",
            self.prior_dispatch_identity_sha256,
        )
        _require_sha40("prior_candidate_git_sha", self.prior_candidate_git_sha)
        _require_sha256(
            "prior_package_identity_sha256",
            self.prior_package_identity_sha256,
        )
        _require_sha256(
            "prior_physical_evidence_identity_sha256",
            self.prior_physical_evidence_identity_sha256,
        )
        _require_sha40(
            "repaired_candidate_git_sha",
            self.repaired_candidate_git_sha,
        )
        _require_sha256(
            "repaired_package_identity_sha256",
            self.repaired_package_identity_sha256,
        )
        _require_id("scenario_id", self.scenario_id)
        _require_id("physical_gate_id", self.physical_gate_id)
        if self.prior_candidate_git_sha == self.repaired_candidate_git_sha:
            raise ValueError(
                "AI repair requalification requires a new candidate Git SHA"
            )
        if (
            self.prior_package_identity_sha256
            == self.repaired_package_identity_sha256
        ):
            raise ValueError(
                "AI repair requalification requires a changed package identity"
            )

    def to_dict(self) -> dict[str, Any]:
        RequalificationRequirement.__post_init__(self)
        return {
            "schema_version": self.schema_version,
            "prior_dispatch_identity_sha256": (
                self.prior_dispatch_identity_sha256
            ),
            "prior_candidate_git_sha": self.prior_candidate_git_sha,
            "prior_package_identity_sha256": (
                self.prior_package_identity_sha256
            ),
            "prior_physical_evidence_identity_sha256": (
                self.prior_physical_evidence_identity_sha256
            ),
            "repaired_candidate_git_sha": self.repaired_candidate_git_sha,
            "repaired_package_identity_sha256": (
                self.repaired_package_identity_sha256
            ),
            "scenario_id": self.scenario_id,
            "physical_gate_id": self.physical_gate_id,
        }

    identity_sha256 = _sealed_identity_method(_canonical_identity)


def build_requalification_requirement(
    failed_receipt: PhysicalExecutionReceipt,
    *,
    repaired_candidate_git_sha: str,
    repaired_package_manifest_bytes: bytes,
) -> RequalificationRequirement:
    if not _is_exact_type(failed_receipt, PhysicalExecutionReceipt):
        raise ValueError("failed_receipt must be an exact PhysicalExecutionReceipt")
    PhysicalExecutionReceipt.__post_init__(failed_receipt)
    if failed_receipt.verdict != "FAIL":
        raise ValueError("AI repair requalification starts from a physical FAIL")
    if (
        not _is_exact_type(repaired_package_manifest_bytes, bytes)
        or not repaired_package_manifest_bytes
    ):
        raise ValueError("repaired package manifest must be non-empty exact bytes")
    return RequalificationRequirement(
        schema_version="12-6.requalification-requirement.v1",
        prior_dispatch_identity_sha256=failed_receipt.dispatch_identity_sha256,
        prior_candidate_git_sha=failed_receipt.target_git_sha,
        prior_package_identity_sha256=failed_receipt.package_identity_sha256,
        prior_physical_evidence_identity_sha256=(
            failed_receipt.physical_evidence_identity_sha256
        ),
        repaired_candidate_git_sha=repaired_candidate_git_sha,
        repaired_package_identity_sha256=_sha256_bytes(
            repaired_package_manifest_bytes
        ),
        scenario_id=failed_receipt.scenario_id,
        physical_gate_id=failed_receipt.physical_gate_id,
    )


@dataclass(frozen=True, slots=True)
class RequalificationReceipt:
    schema_version: str
    requirement_identity_sha256: str
    repaired_candidate_git_sha: str
    repaired_package_identity_sha256: str
    sil_evidence_identity_sha256: str
    physical_evidence_identity_sha256: str
    repaired_dispatch_identity_sha256: str
    scenario_id: str
    physical_gate_id: str
    _verification_token: object | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if self._verification_token is not _VERIFIED_REQUALIFICATION_RECEIPT:
            raise ValueError("requalification receipt requires fresh verification")
        if (
            not _is_exact_type(self.schema_version, str)
            or self.schema_version != "12-6.requalification-receipt.v1"
        ):
            raise ValueError("unsupported requalification receipt schema")
        _require_sha256(
            "requirement_identity_sha256",
            self.requirement_identity_sha256,
        )
        _require_sha40(
            "repaired_candidate_git_sha",
            self.repaired_candidate_git_sha,
        )
        for name, value in (
            (
                "repaired_package_identity_sha256",
                self.repaired_package_identity_sha256,
            ),
            ("sil_evidence_identity_sha256", self.sil_evidence_identity_sha256),
            (
                "physical_evidence_identity_sha256",
                self.physical_evidence_identity_sha256,
            ),
            (
                "repaired_dispatch_identity_sha256",
                self.repaired_dispatch_identity_sha256,
            ),
        ):
            _require_sha256(name, value)
        _require_id("scenario_id", self.scenario_id)
        _require_id("physical_gate_id", self.physical_gate_id)

    def to_dict(self) -> dict[str, Any]:
        RequalificationReceipt.__post_init__(self)
        return {
            "schema_version": self.schema_version,
            "requirement_identity_sha256": self.requirement_identity_sha256,
            "repaired_candidate_git_sha": self.repaired_candidate_git_sha,
            "repaired_package_identity_sha256": (
                self.repaired_package_identity_sha256
            ),
            "sil_evidence_identity_sha256": self.sil_evidence_identity_sha256,
            "physical_evidence_identity_sha256": (
                self.physical_evidence_identity_sha256
            ),
            "repaired_dispatch_identity_sha256": (
                self.repaired_dispatch_identity_sha256
            ),
            "scenario_id": self.scenario_id,
            "physical_gate_id": self.physical_gate_id,
        }

    identity_sha256 = _sealed_identity_method(_canonical_identity)


def _require_requalification_bindings(
    requirement: RequalificationRequirement,
    *,
    repaired_dispatch: HostDispatch,
    repaired_physical_receipt: PhysicalExecutionReceipt,
    expected_package_bytes: bytes,
) -> None:
    if not _is_exact_type(requirement, RequalificationRequirement):
        raise ValueError("requirement must be an exact RequalificationRequirement")
    if not _is_exact_type(repaired_dispatch, HostDispatch):
        raise ValueError("repaired_dispatch must be an exact HostDispatch")
    if not _is_exact_type(repaired_physical_receipt, PhysicalExecutionReceipt):
        raise ValueError(
            "repaired_physical_receipt must be an exact PhysicalExecutionReceipt"
        )
    RequalificationRequirement.__post_init__(requirement)
    HostDispatch.__post_init__(repaired_dispatch)
    PhysicalExecutionReceipt.__post_init__(repaired_physical_receipt)
    if not _is_exact_type(expected_package_bytes, bytes) or not expected_package_bytes:
        raise ValueError("expected_package_bytes must be non-empty exact bytes")
    if _sha256_bytes(expected_package_bytes) != requirement.repaired_package_identity_sha256:
        raise ValueError("repaired package bytes do not match requirement")
    if repaired_dispatch.target_git_sha != requirement.repaired_candidate_git_sha:
        raise ValueError("fresh dispatch does not target repaired candidate")
    if (
        repaired_dispatch.package_identity_sha256
        != requirement.repaired_package_identity_sha256
    ):
        raise ValueError("fresh dispatch does not bind repaired package")
    if (
        repaired_dispatch.scenario_id != requirement.scenario_id
        or repaired_dispatch.physical_gate_id != requirement.physical_gate_id
    ):
        raise ValueError("fresh dispatch changes the physical scenario")
    if (
        repaired_physical_receipt.dispatch_identity_sha256
        != repaired_dispatch.identity_sha256()
    ):
        raise ValueError("physical PASS does not belong to fresh dispatch")
    if repaired_physical_receipt.verdict != "PASS":
        raise ValueError("repaired candidate requires a fresh physical PASS")
    if (
        repaired_physical_receipt.physical_evidence_identity_sha256
        == requirement.prior_physical_evidence_identity_sha256
    ):
        raise ValueError("prior physical evidence cannot transfer to repaired candidate")
    if (
        repaired_physical_receipt.target_git_sha
        != requirement.repaired_candidate_git_sha
        or repaired_physical_receipt.package_identity_sha256
        != requirement.repaired_package_identity_sha256
    ):
        raise ValueError("physical PASS is not bound to repaired candidate/package")


def qualify_repaired_candidate(
    requirement: RequalificationRequirement,
    *,
    repaired_dispatch: HostDispatch,
    repaired_physical_receipt: PhysicalExecutionReceipt,
    sil_evidence_path: str | Path,
    sil_log_path: str | Path,
    expected_package_bytes: bytes,
    expected_environment_receipt: dict[str, Any],
    expected_registry: CapabilityRegistry,
    expected_scenario: SILScenario,
) -> RequalificationReceipt:
    _require_requalification_bindings(
        requirement,
        repaired_dispatch=repaired_dispatch,
        repaired_physical_receipt=repaired_physical_receipt,
        expected_package_bytes=expected_package_bytes,
    )

    sil_evidence = verify_sil_evidence(
        sil_evidence_path,
        sil_log_path,
        expected_package_bytes=expected_package_bytes,
        expected_environment_receipt=expected_environment_receipt,
        expected_registry=expected_registry,
        expected_scenario=expected_scenario,
        expected_git_sha=requirement.repaired_candidate_git_sha,
        require_pass=True,
    )
    if not _is_exact_type(sil_evidence, dict):
        raise ValueError("SIL verifier must return an exact evidence object")
    sil_identity = _require_sha256(
        "SIL evidence identity",
        sil_evidence.get("evidence_identity_sha256"),
    )
    if (
        sil_evidence.get("git_sha")
        != requirement.repaired_candidate_git_sha
    ):
        raise ValueError("SIL PASS is not bound to repaired candidate")
    if (
        sil_evidence.get("package_identity_sha256")
        != requirement.repaired_package_identity_sha256
    ):
        raise ValueError("SIL PASS is not bound to repaired package")

    return RequalificationReceipt(
        schema_version="12-6.requalification-receipt.v1",
        requirement_identity_sha256=requirement.identity_sha256(),
        repaired_candidate_git_sha=requirement.repaired_candidate_git_sha,
        repaired_package_identity_sha256=(
            requirement.repaired_package_identity_sha256
        ),
        sil_evidence_identity_sha256=sil_identity,
        physical_evidence_identity_sha256=(
            repaired_physical_receipt.physical_evidence_identity_sha256
        ),
        repaired_dispatch_identity_sha256=repaired_dispatch.identity_sha256(),
        scenario_id=requirement.scenario_id,
        physical_gate_id=requirement.physical_gate_id,
        _verification_token=_VERIFIED_REQUALIFICATION_RECEIPT,
    )


# Plan 8 / Section 5: immutable, replay-resistant host/GitHub handoff.
# This file transport is deliberately not a remote shell or automatic retry worker.
# Files in outbound/ are uploadable as GitHub Actions artifacts; they are not
# automatically promoted to a real physical PASS.

_BRIDGE_MAX_PACKET_BYTES = 4 * 1024 * 1024
_BRIDGE_MAX_RECEIPT_BYTES = 4 * 1024 * 1024
_BRIDGE_MAX_LOG_BYTES = 32 * 1024 * 1024


def _bridge_regular_bytes(path: Path, max_bytes: int) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("bridge input must be a regular non-symlink file")
    with path.open("rb") as stream:
        payload = stream.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError("bridge input exceeds its bounded size")
    return payload


def _bridge_write_once(path: Path, payload: bytes) -> bool:
    """Atomic exclusive claim/publication; replay is safe only for identical bytes."""
    if type(payload) is not bytes:
        raise ValueError("bridge publication must contain bytes")
    if path.parent.is_symlink() or path.is_symlink():
        raise ValueError("bridge publication refuses symlink paths")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.parent.is_symlink():
        raise ValueError("bridge publication parent is a symlink")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if _bridge_regular_bytes(path, max(len(payload), _BRIDGE_MAX_PACKET_BYTES)) != payload:
            raise ValueError("bridge idempotency key already has different bytes")
        return False
    with os.fdopen(fd, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return True


def _bridge_spool(root: str | Path) -> Path:
    path = Path(root)
    if path.is_symlink():
        raise ValueError("bridge spool cannot be a symlink")
    path.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or not path.is_dir():
        raise ValueError("bridge spool is not a directory")
    return path


def stage_bridge_dispatch(
    spool_root: str | Path,
    *,
    dispatch: HostDispatch,
    verified_packet: VerifiedSignedPacket,
    signed_packet_bytes: bytes,
) -> dict[str, Any]:
    """Transfer a preverified bounded signed packet with exact host dispatch.

    Retry publishes byte-identical envelopes. A changed packet never replaces
    the already staged dispatch, even after a crash or a reconnect.
    """
    if type(dispatch) is not HostDispatch or type(verified_packet) is not VerifiedSignedPacket:
        raise ValueError("bridge staging requires exact verified dispatch and packet")
    HostDispatch.__post_init__(dispatch)
    VerifiedSignedPacket.__post_init__(verified_packet)
    if verified_packet.packet.execution_mode is not ExecutionMode.REAL_HOST:
        raise ValueError("bridge staging requires REAL_HOST")
    if verified_packet.packet.target_git_sha != dispatch.target_git_sha:
        raise ValueError("bridge packet candidate mismatch")
    if verified_packet.packet.identity_sha256() != dispatch.packet_identity_sha256:
        raise ValueError("bridge packet identity mismatch")
    if (
        verified_packet.signed_bundle_identity_sha256
        != dispatch.signed_bundle_identity_sha256
    ):
        raise ValueError("bridge packet signed bundle mismatch")
    if (
        type(signed_packet_bytes) is not bytes
        or not 0 < len(signed_packet_bytes) <= _BRIDGE_MAX_PACKET_BYTES
    ):
        raise ValueError("bridge signed packet byte bound exceeded")
    # The transport must seal the same bytes that the authority verified.
    # A signed packet has no standing merely because its file is called signed.
    def no_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        parsed: dict[str, Any] = {}
        for key, item in pairs:
            if key in parsed:
                raise ValueError("duplicate signed packet JSON member")
            parsed[key] = item
        return parsed

    try:
        signed = json.loads(
            signed_packet_bytes.decode("utf-8"),
            object_pairs_hook=no_duplicate_fields,
        )
        signature = signed["signature"]
        raw_signature = base64.b64decode(signature["signature_b64"], validate=True)
    except (UnicodeError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ValueError("bridge signed packet is malformed") from exc
    if (
        type(signed) is not dict
        or set(signed) != {"schema_version", "packet", "signature"}
        or signed["schema_version"] != "12-6.signed-physical-qualification-packet.v1"
        or signed["packet"] != verified_packet.packet.to_dict()
        or type(signature) is not dict
        or set(signature) != {"algorithm", "key_id", "signature_b64"}
        or signature["algorithm"] != "ED25519"
        or signature["key_id"] != verified_packet.signing_key_id
        or len(raw_signature) != 64
        or _sha256_bytes(raw_signature) != verified_packet.signature_sha256
    ):
        raise ValueError("bridge signed packet does not bind verified authority bytes")
    spool = _bridge_spool(spool_root)
    identity = dispatch.identity_sha256()
    packet_hash = _sha256_bytes(signed_packet_bytes)
    manifest = {
        "schema_version": "12-6.bridge-handoff.v1",
        "dispatch": dispatch.to_dict(),
        "dispatch_identity_sha256": identity,
        "signed_packet_sha256": packet_hash,
        "packet_relative_path": f"packets/{identity}.json",
        "execution_authority": "SIGNED_PACKET_ONLY",
        "automatic_retry_allowed": False,
    }
    _bridge_write_once(spool / "packets" / f"{identity}.json", signed_packet_bytes)
    _bridge_write_once(
        spool / "dispatches" / f"{identity}.json",
        _canonical_json_bytes(manifest) + b"\n",
    )
    return manifest


def bridge_status(spool_root: str | Path, dispatch_identity_sha256: str) -> dict[str, str]:
    """Text-friendly status; neither a receipt nor a self-asserted physical PASS."""
    identity = _require_sha256("dispatch identity", dispatch_identity_sha256)
    spool = _bridge_spool(spool_root)
    staged = spool / "dispatches" / f"{identity}.json"
    if not staged.exists():
        return {"state": "NOT_STAGED", "verification": "NOT_CHECKED"}
    packet = spool / "packets" / f"{identity}.json"
    raw = _bridge_regular_bytes(staged, _BRIDGE_MAX_PACKET_BYTES)
    try:
        envelope = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ValueError("bridge handoff envelope is corrupted") from exc
    if (
        type(envelope) is not dict
        or envelope.get("schema_version") != "12-6.bridge-handoff.v1"
        or envelope.get("dispatch_identity_sha256") != identity
        or _sha256_bytes(_bridge_regular_bytes(packet, _BRIDGE_MAX_PACKET_BYTES))
        != envelope.get("signed_packet_sha256")
    ):
        raise ValueError("bridge dispatch/packet handoff mismatch")
    if (spool / "stops" / f"{identity}.json").exists():
        return {"state": "STOP_REQUESTED", "verification": "NOT_CHECKED"}
    if (spool / "outbound" / f"{identity}" / "evidence-record.json").exists():
        return {"state": "EVIDENCE_RECORDED", "verification": "NOT_CHECKED"}
    if (spool / "claims" / f"{identity}.json").exists():
        return {"state": "CLAIMED_OUTCOME_UNKNOWN", "verification": "NOT_CHECKED"}
    return {"state": "READY", "verification": "NOT_CHECKED"}


def claim_bridge_dispatch(spool_root: str | Path, dispatch_identity_sha256: str) -> None:
    """Fence a host effect before execution, never replay an ambiguous attempt."""
    identity = _require_sha256("dispatch identity", dispatch_identity_sha256)
    spool = _bridge_spool(spool_root)
    status = bridge_status(spool, identity)
    if status["state"] != "READY":
        raise ValueError(f"bridge run denied: {status['state']}; no automatic retry")
    claim = _canonical_json_bytes(
        {"schema_version": "12-6.bridge-attempt.v1", "dispatch": identity}
    ) + b"\n"
    if not _bridge_write_once(spool / "claims" / f"{identity}.json", claim):
        raise ValueError("bridge attempt already claimed; outcome unknown")


def stop_bridge_dispatch(spool_root: str | Path, dispatch_identity_sha256: str) -> None:
    """Durably prevent future starts; running subprocesses still require Ctrl+C."""
    identity = _require_sha256("dispatch identity", dispatch_identity_sha256)
    spool = _bridge_spool(spool_root)
    if bridge_status(spool, identity)["state"] == "NOT_STAGED":
        raise ValueError("cannot stop an unknown dispatch")
    marker = _canonical_json_bytes(
        {"schema_version": "12-6.bridge-stop.v1", "dispatch": identity}
    ) + b"\n"
    _bridge_write_once(spool / "stops" / f"{identity}.json", marker)


def record_bridge_return(
    spool_root: str | Path,
    *,
    dispatch: HostDispatch,
    verified_packet: VerifiedSignedPacket,
    evidence_path: str | Path,
    log_path: str | Path,
    agent_source_bytes: bytes,
    artifact_root: str | Path,
    evidence_signature_verifier: SignatureVerifier,
    resource_probe_verifiers: dict[ResourceKind, ExternalResourceVerifier] | None = None,
) -> dict[str, Any]:
    """Verify host signature before durable evidence/issue export.

    Interrupted partial publication is recoverable only with identical bytes.
    A failure without a checked-in pytest reproducer remains INFRASTRUCTURE,
    never a fabricated code-defect assertion.
    """
    if type(dispatch) is not HostDispatch:
        raise ValueError("bridge return requires an exact dispatch")
    HostDispatch.__post_init__(dispatch)
    identity = dispatch.identity_sha256()
    spool = _bridge_spool(spool_root)
    status = bridge_status(spool, identity)
    if status["state"] not in {"CLAIMED_OUTCOME_UNKNOWN", "EVIDENCE_RECORDED"}:
        raise ValueError("bridge return requires exactly one prior claimed attempt")
    evidence_bytes = _bridge_regular_bytes(Path(evidence_path), _BRIDGE_MAX_RECEIPT_BYTES)
    log_bytes = _bridge_regular_bytes(Path(log_path), _BRIDGE_MAX_LOG_BYTES)
    sealed = verify_physical_execution(
        dispatch,
        evidence_path,
        log_path,
        verified_packet=verified_packet,
        agent_source_bytes=agent_source_bytes,
        artifact_root=artifact_root,
        evidence_signature_verifier=evidence_signature_verifier,
        resource_probe_verifiers=resource_probe_verifiers,
    )
    record = build_canonical_evidence_record(dispatch, sealed)
    evidence_report: dict[str, Any] = {
        "schema_version": "12-6.bridge-return.v1",
        "dispatch_identity_sha256": identity,
        "record": record.to_dict(),
        "receipt_sha256": _sha256_bytes(evidence_bytes),
        "log_sha256": _sha256_bytes(log_bytes),
        "verdict": sealed.verdict,
        "effect": "ONE_CLAIM_NO_AUTO_RETRY",
    }
    if sealed.verdict == "FAIL":
        if sealed.reproducer_command:
            observation = physical_failure_observation(sealed)
            evidence_report["defect"] = {
                "kind": "REPRODUCIBLE_TEST_FAILURE",
                "reproducer_command": observation.reproducer_command,
                "candidate_sha": observation.git_sha,
                "evidence_sha256": observation.evidence_identity_sha256,
            }
        else:
            evidence_report["defect"] = {
                "kind": "PHYSICAL_INFRASTRUCTURE_FAILURE",
                "candidate_sha": sealed.target_git_sha,
                "evidence_sha256": sealed.physical_evidence_identity_sha256,
                "reproducer_command": None,
            }
    out = spool / "outbound" / identity
    _bridge_write_once(out / "host-evidence.json", evidence_bytes)
    _bridge_write_once(out / "host-log.bin", log_bytes)
    _bridge_write_once(
        out / "bridge-return.json",
        _canonical_json_bytes(evidence_report) + b"\n",
    )
    _bridge_write_once(
        out / "evidence-record.json",
        _canonical_json_bytes(record.to_dict()) + b"\n",
    )
    return evidence_report


class _NoGitHubRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise ValueError("GitHub evidence API redirects are not permitted")


def _github_evidence_request(
    method: str,
    path: str,
    *,
    token: str,
    branch: str,
    upload: bytes | None = None,
) -> tuple[int, dict[str, Any]]:
    """One bounded request to a fixed trusted host; never follow redirects."""
    if type(token) is not str or not 10 <= len(token) <= 4096:
        raise ValueError("GitHub evidence token is missing or malformed")
    if method not in {"GET", "PUT"}:
        raise ValueError("unsupported evidence publication HTTP method")
    uri = (
        "https://api.github.com/repos/Oleksii-debug/12-6-ai./contents/"
        + quote(path, safe="/")
    )
    if method == "GET":
        uri += "?ref=" + quote(branch, safe="")
    else:
        if type(upload) is not bytes or len(upload) > _BRIDGE_MAX_RECEIPT_BYTES:
            raise ValueError("bridge upload has invalid bytes")
    payload = None
    if method == "PUT":
        payload = _canonical_json_bytes({
            "message": "evidence(plan8-s5): immutable signed host return (hash-only)",
            "branch": branch,
            "content": base64.b64encode(upload).decode("ascii"),
        })
    req = Request(
        uri, method=method, data=payload,
        headers={
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "12-6-plan8-evidence-bridge-v1",
            "Content-Type": "application/json",
        },
    )
    opener = build_opener(_NoGitHubRedirects())
    try:
        with opener.open(req, timeout=15) as response:
            content = response.read(16385)
            if len(content) > 16384:
                raise ValueError("GitHub evidence response exceeds limit")
            return response.status, json.loads(content)
    except HTTPError as exc:
        if exc.code in {404, 422}:
            return exc.code, {}
        raise RuntimeError("GitHub evidence publication failed with HTTP error") from exc


def _git_blob_identity(payload: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(payload)).encode("ascii") + b"\x00" + payload
    ).hexdigest()


def publish_bridge_return_github(
    spool_root: str | Path,
    *,
    dispatch: HostDispatch,
    verified_packet: VerifiedSignedPacket,
    evidence_path: str | Path,
    log_path: str | Path,
    agent_source_bytes: bytes,
    artifact_root: str | Path,
    evidence_signature_verifier: SignatureVerifier,
    github_token: str,
    evidence_branch: str = "plan8-evidence",
    resource_probe_verifiers: dict[ResourceKind, ExternalResourceVerifier] | None = None,
) -> dict[str, Any]:
    """Return safe metadata to the project GitHub evidence branch, create-only.

    Always reverify the host-signed evidence before transfer. Never upload raw
    host logs, private keys, environment variables, or unverified result claims.
    A 422 conflict is resolved by a readback; no blind re-execution or writes
    to main are possible through this adapter.
    """
    if evidence_branch != "plan8-evidence":
        raise ValueError("bridge may publish only to dedicated evidence branch")
    report = record_bridge_return(
        spool_root, dispatch=dispatch, verified_packet=verified_packet,
        evidence_path=evidence_path, log_path=log_path,
        agent_source_bytes=agent_source_bytes, artifact_root=artifact_root,
        evidence_signature_verifier=evidence_signature_verifier,
        resource_probe_verifiers=resource_probe_verifiers,
    )
    identity = dispatch.identity_sha256()
    spool = _bridge_spool(spool_root)
    out = spool / "outbound" / identity
    results = []
    for filename in ("evidence-record.json", "bridge-return.json"):
        payload = _bridge_regular_bytes(out / filename, _BRIDGE_MAX_RECEIPT_BYTES)
        expected_blob = _git_blob_identity(payload)
        remote = "evidence/plan8/" + identity + "/" + filename
        status, existing = _github_evidence_request(
            "GET", remote, token=github_token, branch=evidence_branch
        )
        if status == 200:
            if existing.get("type") != "file" or existing.get("sha") != expected_blob:
                raise ValueError("GitHub evidence path contains different bytes")
            results.append({"path": remote, "state": "ALREADY_IDENTICAL"})
            continue
        if status != 404:
            raise ValueError("GitHub evidence readback is inconclusive")
        code, created = _github_evidence_request(
            "PUT", remote, token=github_token, branch=evidence_branch, upload=payload
        )
        if code == 422:
            # A raced or timed-out create is not blindly repeated.
            code, created = _github_evidence_request(
                "GET", remote, token=github_token, branch=evidence_branch
            )
            if (
                code != 200 or created.get("type") != "file"
                or created.get("sha") != expected_blob
            ):
                raise ValueError("GitHub evidence conflict cannot be reconciled")
        elif (
            code != 201 or type(created.get("content")) is not dict
            or created["content"].get("sha") != expected_blob
        ):
            raise ValueError("GitHub evidence create readback is invalid")
        results.append({"path": remote, "state": "VERIFIED_CREATED"})
    return {
        "schema_version": "12-6.github-evidence-publish.v1",
        "dispatch_identity_sha256": identity,
        "verdict": report["verdict"],
        "publication": results,
        "raw_private_logs_uploaded": False,
    }
