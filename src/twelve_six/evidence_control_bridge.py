from __future__ import annotations

import hashlib
import json
import re
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
from twelve_six.sil_qualification import SILScenario, parse_vector_command, verify_sil_evidence


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

    def identity_sha256(
        self,
        _sealed_identity: Callable[[Any], str] = _canonical_identity,
    ) -> str:
        return _sealed_identity(self.to_dict())


def build_host_dispatch(
    verified_packet: VerifiedSignedPacket,
    *,
    package_manifest_bytes: bytes,
    dispatch_id: str,
    scenario_id: str,
    physical_gate_id: str,
) -> HostDispatch:
    if not _is_exact_type(verified_packet, VerifiedSignedPacket):
        raise ValueError("verified_packet must be an exact VerifiedSignedPacket")
    VerifiedSignedPacket.__post_init__(verified_packet)
    if not _is_exact_type(package_manifest_bytes, bytes) or not package_manifest_bytes:
        raise ValueError("package_manifest_bytes must be non-empty exact bytes")
    if verified_packet.packet.execution_mode is not ExecutionMode.REAL_HOST:
        raise ValueError("Section 6 physical dispatch requires REAL_HOST mode")
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

    def identity_sha256(
        self,
        _sealed_identity: Callable[[Any], str] = _canonical_identity,
    ) -> str:
        return _sealed_identity(self.to_dict())


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


PhysicalVerifier = Callable[..., dict[str, Any]]
SILVerifier = Callable[..., dict[str, Any]]


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
    qualification_verifier: PhysicalVerifier = verify_qualification_evidence,
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

    evidence = qualification_verifier(
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

    def identity_sha256(
        self,
        _sealed_identity: Callable[[Any], str] = _canonical_identity,
    ) -> str:
        return _sealed_identity(self.to_dict())


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

    def identity_sha256(
        self,
        _sealed_identity: Callable[[Any], str] = _canonical_identity,
    ) -> str:
        return _sealed_identity(self.to_dict())


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

    def identity_sha256(
        self,
        _sealed_identity: Callable[[Any], str] = _canonical_identity,
    ) -> str:
        return _sealed_identity(self.to_dict())


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
    sil_verifier: SILVerifier = verify_sil_evidence,
) -> RequalificationReceipt:
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

    sil_evidence = sil_verifier(
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
