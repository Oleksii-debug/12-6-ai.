"""Capability evidence requirements and exact-candidate promotion readiness."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from twelve_six.ai_qa_control import (
    AIQAPolicy,
    FailurePacket,
    GateReceipt,
    RepairCandidate,
    evaluate_promotion,
)
from twelve_six.capability_map import CapabilityRegistry
from twelve_six.evidence_control_bridge import (
    _SCALE_PROFILE_FILES,
    HostDispatch,
    PhysicalExecutionReceipt,
    verify_physical_execution,
)
from twelve_six.model import load_stage_config
from twelve_six.physical_qualification import (
    ExecutionMode,
    ExternalResourceVerifier,
    ResourceKind,
    SignatureVerifier,
    VerifiedSignedPacket,
    load_verified_signed_packet,
)
from twelve_six.scenario_harness import Scenario, coverage_map
from twelve_six.sil_qualification import (
    SILScenario,
    build_package_manifest_bytes,
    require_exact_clean_git_state,
    verify_sil_evidence,
)

_SHA40 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_KEY_ID = re.compile(r"[a-z][a-z0-9_.-]{0,95}")
_VERIFIED = object()


class EvidenceLevel(str, Enum):
    Q0 = "Q0"
    Q1 = "Q1"
    Q2 = "Q2"
    Q3 = "Q3"
    Q4 = "Q4"


class ScaleReadiness(str, Enum):
    CONFIG_DEFINED = "CONFIG_DEFINED"
    PROXY_VERIFIED = "PROXY_VERIFIED"
    HOST_VERIFIED = "HOST_VERIFIED"
    TRAINING_AUTHORIZED = "TRAINING_AUTHORIZED"
    LEARNED_QUALIFIED = "LEARNED_QUALIFIED"


_LEVELS = frozenset(level.value for level in EvidenceLevel)
_SCALE_PROOFS = frozenset(state.value for state in ScaleReadiness if state is not ScaleReadiness.CONFIG_DEFINED)
_SIGNED_KINDS = _SCALE_PROOFS | {"Q2", "Q3", "Q4"}
_PHYSICAL_KINDS = {"Q2", "HOST_VERIFIED"}


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _exact(value: object, pattern: re.Pattern[str], label: str) -> None:
    if type(value) is not str or not pattern.fullmatch(value):
        raise ValueError(f"invalid {label}")


def _strict_object(raw: bytes) -> dict[str, Any]:
    if type(raw) is not bytes or not 0 < len(raw) <= 65_536:
        raise ValueError("evidence JSON exceeds its byte bound")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate evidence JSON field")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("invalid evidence JSON") from exc
    if type(value) is not dict:
        raise ValueError("evidence JSON must be an object")
    return value


@dataclass(frozen=True, slots=True)
class EvidenceTarget:
    git_sha: str
    package_sha256: str | None

    def __post_init__(self) -> None:
        _exact(self.git_sha, _SHA40, "candidate Git SHA")
        if self.package_sha256 is not None:
            _exact(self.package_sha256, _SHA256, "candidate package SHA-256")


def _seal(*values: object) -> tuple[object, str]:
    return _VERIFIED, _hash(_canonical(values))


@dataclass(frozen=True, slots=True)
class EvidencePolicy:
    registry_sha256: str
    policy_sha256: str
    requirements: tuple[tuple[str, EvidenceLevel], ...]
    _verification: tuple[object, str] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        _exact(self.registry_sha256, _SHA256, "capability registry SHA-256")
        _exact(self.policy_sha256, _SHA256, "evidence policy SHA-256")
        rows = [(key, level.value) for key, level in self.requirements]
        if self._verification != _seal(self.registry_sha256, self.policy_sha256, rows):
            raise ValueError("evidence policy requires verified ingress")

    def required(self, capability_id: str) -> EvidenceLevel:
        for identifier, level in self.requirements:
            if identifier == capability_id:
                return level
        raise ValueError("unknown capability in evidence policy")


def load_evidence_policy(
    path: str | Path, registry: CapabilityRegistry, *, expected_sha256: str,
) -> EvidencePolicy:
    _exact(expected_sha256, _SHA256, "trusted evidence policy SHA-256")
    with Path(path).open("rb") as stream:
        raw = stream.read(65_537)
    if _hash(raw) != expected_sha256:
        raise ValueError("evidence policy differs from its trusted SHA-256")
    doc = _strict_object(raw)
    if set(doc) != {"schema_version", "capability_registry_sha256", "requirements"}:
        raise ValueError("invalid evidence policy fields")
    if doc["schema_version"] != "12-6.plan8-evidence-policy.v1":
        raise ValueError("unsupported evidence policy version")
    identity = CapabilityRegistry.identity_sha256(registry)
    if doc["capability_registry_sha256"] != identity:
        raise ValueError("evidence policy registry identity mismatch")
    rows = doc["requirements"]
    if type(rows) is not dict or set(rows) != {item.capability_id for item in registry.capabilities}:
        raise ValueError("evidence policy must declare every registry capability exactly once")
    if any(type(value) is not str or value not in _LEVELS for value in rows.values()):
        raise ValueError("invalid capability evidence level")
    levels = tuple((key, EvidenceLevel(rows[key])) for key in sorted(rows))
    return EvidencePolicy(identity, expected_sha256, levels, _verification=_seal(
        identity, expected_sha256, [(key, level.value) for key, level in levels],
    ))


@dataclass(frozen=True, slots=True)
class VerifiedEvidence:
    kind: str
    subject: str
    target: EvidenceTarget
    artifact_sha256: str
    config_sha256: str | None = None
    registry_sha256: str | None = None
    backend: str | None = None
    _verification: tuple[object, str] | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if type(self.target) is not EvidenceTarget:
            raise ValueError("evidence requires an exact target")
        EvidenceTarget.__post_init__(self.target)
        _exact(self.artifact_sha256, _SHA256, "evidence SHA-256")
        if type(self.kind) is not str or self.kind not in _LEVELS | _SCALE_PROOFS:
            raise ValueError("unknown evidence kind")
        if self.kind in _SCALE_PROOFS:
            if type(self.subject) is not str or self.subject not in _SCALE_PROFILE_FILES:
                raise ValueError("unknown scale profile")
            _exact(self.config_sha256, _SHA256, "profile config SHA-256")
            if self.registry_sha256 is not None:
                raise ValueError("scale evidence cannot carry a capability registry")
        else:
            _exact(self.subject, _KEY_ID, "capability ID")
            _exact(self.registry_sha256, _SHA256, "capability registry SHA-256")
            if self.config_sha256 is not None:
                raise ValueError("capability evidence cannot carry a scale config")
        if (self.target.package_sha256 is None) != (self.kind == "Q0"):
            raise ValueError("only Q0 has source-only evidence scope")
        if self.kind == "HOST_VERIFIED":
            if type(self.backend) is not str or self.backend not in {"CPU", "CUDA"}:
                raise ValueError("host evidence must declare its verified backend")
        elif self.backend is not None:
            raise ValueError("backend scope applies only to host evidence")
        if self._verification != _seal(
            self.kind, self.subject, self.target.git_sha, self.target.package_sha256,
            self.artifact_sha256, self.config_sha256, self.registry_sha256, self.backend,
        ):
            raise ValueError("evidence requires verified ingress")


def _admitted(
    kind: str,
    subject: str,
    target: EvidenceTarget,
    artifact_sha256: str,
    config_sha256: str | None = None,
    registry_sha256: str | None = None,
    backend: str | None = None,
) -> VerifiedEvidence:
    return VerifiedEvidence(
        kind, subject, target, artifact_sha256, config_sha256, registry_sha256, backend,
        _verification=_seal(
            kind, subject, target.git_sha, target.package_sha256,
            artifact_sha256, config_sha256, registry_sha256, backend,
        ),
    )


def admit_component_receipt(
    scenario: Scenario,
    receipt: dict[str, object],
    *,
    capability_id: str,
    source_sha: str,
    trusted_scenario_sha256: str,
    registry: CapabilityRegistry,
) -> VerifiedEvidence:
    _exact(trusted_scenario_sha256, _SHA256, "trusted scenario SHA-256")
    coverage = coverage_map((scenario,), (receipt,), expected_source_sha=source_sha)
    if scenario.identity_sha256() != trusted_scenario_sha256:
        raise ValueError("component scenario differs from its trusted authority")
    if capability_id not in coverage:
        raise ValueError("component scenario does not cover capability")
    registry.capability(capability_id)
    return _admitted("Q0", capability_id, EvidenceTarget(source_sha, None), receipt["identity_sha256"],
                     registry_sha256=CapabilityRegistry.identity_sha256(registry))


def admit_sil_receipt(
    evidence_path: str | Path,
    log_path: str | Path,
    *,
    capability_id: str,
    target: EvidenceTarget,
    package_bytes: bytes,
    environment_receipt: dict[str, Any],
    registry: CapabilityRegistry,
    scenario: SILScenario,
) -> VerifiedEvidence:
    if type(package_bytes) is not bytes or _hash(package_bytes) != target.package_sha256:
        raise ValueError("SIL package differs from the exact target")
    result = verify_sil_evidence(
        evidence_path, log_path, expected_package_bytes=package_bytes,
        expected_environment_receipt=environment_receipt, expected_registry=registry,
        expected_scenario=scenario, expected_git_sha=target.git_sha,
    )
    if not any(
        item.journey_id in result["available_journey_ids"] and capability_id in item.capability_ids
        for item in registry.journeys
    ):
        raise ValueError("verified SIL does not cover capability")
    return _admitted("Q1", capability_id, target, result["evidence_identity_sha256"],
                     registry_sha256=CapabilityRegistry.identity_sha256(registry))


@dataclass(frozen=True, slots=True)
class PhysicalEvidenceSource:
    repo_root: Path
    dispatch: HostDispatch
    packet_path: Path
    evidence_path: Path
    log_path: Path
    packet_verifier: SignatureVerifier
    host_verifier: SignatureVerifier
    resource_verifiers: dict[ResourceKind, ExternalResourceVerifier] | None = None


def _verify_physical_source(
    claim: VerifiedEvidence,
    gate_id: str,
    source: PhysicalEvidenceSource | None,
) -> None:
    if type(source) is not PhysicalEvidenceSource:
        raise ValueError("evidence requires an independently verified physical receipt")
    root = source.repo_root.resolve(strict=True)
    require_exact_clean_git_state(root, claim.target.git_sha)
    if _hash(build_package_manifest_bytes(root)) != claim.target.package_sha256:
        raise ValueError("physical package differs from the exact checkout")
    packet = load_verified_signed_packet(source.packet_path, signature_verifier=source.packet_verifier)
    physical = verify_physical_execution(
        source.dispatch, source.evidence_path, source.log_path,
        verified_packet=packet,
        agent_source_bytes=(root / "src/twelve_six/physical_qualification.py").read_bytes(),
        artifact_root=root, evidence_signature_verifier=source.host_verifier,
        resource_probe_verifiers=source.resource_verifiers,
    )
    if physical.physical_gate_id != gate_id:
        raise ValueError("physical gate differs from the independently signed scope")
    _verify_physical_binding(claim, physical, packet)
    require_exact_clean_git_state(root, claim.target.git_sha)


def _verify_physical_binding(
    claim: VerifiedEvidence,
    physical: PhysicalExecutionReceipt | None,
    packet: VerifiedSignedPacket | None,
) -> None:
    if type(physical) is not PhysicalExecutionReceipt:
        raise ValueError("evidence requires an independently verified physical receipt")
    PhysicalExecutionReceipt.__post_init__(physical)
    if (
        physical.verdict != "PASS"
        or physical.target_git_sha != claim.target.git_sha
        or physical.package_identity_sha256 != claim.target.package_sha256
        or physical.physical_evidence_identity_sha256 != claim.artifact_sha256
    ):
        raise ValueError("physical receipt differs from the signed evidence binding")
    if claim.kind == "HOST_VERIFIED":
        if type(packet) is not VerifiedSignedPacket:
            raise ValueError("host readiness requires a verified scale packet")
        VerifiedSignedPacket.__post_init__(packet)
        required = {ResourceKind.CPU, ResourceKind.RAM, ResourceKind.DISK}
        if claim.backend == "CUDA":
            required.add(ResourceKind.GPU)
        observed_scope = {
            resource for action in packet.packet.actions for resource in action.required_resources
        }
        if (
            packet.packet.execution_mode is not ExecutionMode.REAL_HOST
            or packet.packet.target_git_sha != claim.target.git_sha
            or packet.packet.packet_id != f"{claim.subject.lower()}.{claim.config_sha256}"
            or packet.packet.identity_sha256() != physical.packet_identity_sha256
            or packet.signed_bundle_identity_sha256 != physical.signed_bundle_identity_sha256
            or not required.issubset(observed_scope)
        ):
            raise ValueError("verified physical packet lacks the exact scale or hardware scope")


def admit_signed_evidence(
    raw: bytes,
    *,
    target: EvidenceTarget,
    signature_verifier: SignatureVerifier,
    trusted_keys: Mapping[tuple[str, str], frozenset[str]],
    physical_source: PhysicalEvidenceSource | None = None,
) -> VerifiedEvidence:
    """Verify an independent role's signed exact-target qualification."""
    doc = _strict_object(raw)
    if set(doc) != {"statement", "signature"}:
        raise ValueError("invalid signed evidence envelope")
    body, signature = doc["statement"], doc["signature"]
    if type(body) is not dict or set(body) != {
        "schema_version", "kind", "subject", "git_sha", "package_sha256",
        "artifact_sha256", "config_sha256", "registry_sha256", "physical_gate_id", "backend",
    } or body["schema_version"] != "12-6.plan8-evidence-attestation.v1":
        raise ValueError("invalid signed evidence statement")
    kind = body["kind"]
    if type(kind) is not str or kind not in _SIGNED_KINDS:
        raise ValueError("unsupported signed evidence kind")
    subject = body["subject"]
    if type(subject) is not str:
        raise ValueError("invalid signed evidence subject")
    if type(signature) is not dict or set(signature) != {"algorithm", "key_id", "signature_b64"}:
        raise ValueError("invalid evidence signature fields")
    key_id = signature["key_id"]
    _exact(key_id, _KEY_ID, "evidence authority key")
    if signature["algorithm"] != "ED25519" or key_id not in trusted_keys.get((kind, subject), frozenset()):
        raise ValueError("signing key is not trusted for this evidence role")
    try:
        signed = base64.b64decode(signature["signature_b64"], validate=True)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid evidence signature encoding") from exc
    if len(signed) != 64 or signature_verifier(key_id, _canonical(body), signed) is not True:
        raise ValueError("evidence signature verification failed")
    if EvidenceTarget(body["git_sha"], body["package_sha256"]) != target:
        raise ValueError("signed evidence target is stale")
    claim = _admitted(
        kind, subject, target, body["artifact_sha256"], body["config_sha256"], body["registry_sha256"],
        body["backend"],
    )
    if kind in _PHYSICAL_KINDS:
        _exact(body["physical_gate_id"], _KEY_ID, "physical gate ID")
        _verify_physical_source(claim, body["physical_gate_id"], physical_source)
    elif body["physical_gate_id"] is not None:
        raise ValueError("this evidence kind does not bind a physical gate")
    return claim


def _matching_evidence(
    evidence: tuple[VerifiedEvidence, ...], target: EvidenceTarget,
) -> tuple[VerifiedEvidence, ...]:
    if type(target) is not EvidenceTarget:
        raise ValueError("exact evidence target required")
    EvidenceTarget.__post_init__(target)
    if type(evidence) is not tuple or any(type(item) is not VerifiedEvidence for item in evidence):
        raise ValueError("verified evidence must be an immutable tuple")
    for item in evidence:
        VerifiedEvidence.__post_init__(item)
    return tuple(
        item for item in evidence
        if item.target == target or (item.kind == "Q0" and item.target.git_sha == target.git_sha)
    )


def capability_readiness(
    policy: EvidencePolicy,
    capability_id: str,
    target: EvidenceTarget,
    evidence: tuple[VerifiedEvidence, ...],
) -> dict[str, object]:
    if type(policy) is not EvidencePolicy:
        raise ValueError("verified evidence policy required")
    EvidencePolicy.__post_init__(policy)
    required = policy.required(capability_id).value
    admitted = {
        item.kind for item in _matching_evidence(evidence, target)
        if item.subject == capability_id and item.kind in _LEVELS
        and item.registry_sha256 == policy.registry_sha256
    }
    return {"capability_id": capability_id, "required_level": required,
            "admitted_levels": sorted(admitted), "package_bound_levels": sorted(admitted - {"Q0"}),
            "evidence_complete": required in admitted}


def scale_readiness(
    repo_root: str | Path,
    target: EvidenceTarget,
    evidence: tuple[VerifiedEvidence, ...],
) -> tuple[dict[str, object], ...]:
    current = _matching_evidence(evidence, target)
    root = Path(repo_root)
    require_exact_clean_git_state(root, target.git_sha)
    if _hash(build_package_manifest_bytes(root)) != target.package_sha256:
        raise ValueError("scale package differs from the exact checkout")
    results = []
    for profile, filename in _SCALE_PROFILE_FILES.items():
        path = root / "configs/stages" / filename
        raw = path.read_bytes()
        config = load_stage_config(path)
        if config.stage != profile or json.loads(raw).get("qualification") != "CONFIG_DEFINED":
            raise ValueError("scale configuration identity mismatch")
        identity = _hash(raw)
        proofs = tuple(item for item in current if item.subject == profile and item.config_sha256 == identity)
        states = {"CONFIG_DEFINED": True}
        for state in ScaleReadiness:
            if state is not ScaleReadiness.CONFIG_DEFINED:
                states[state.value] = any(item.kind == state.value for item in proofs)
        results.append({"profile_id": profile, "config_sha256": identity,
                        "parameters": config.expected_parameters, "states": states,
                        "host_backends": sorted({item.backend for item in proofs if item.backend is not None})})
    require_exact_clean_git_state(root, target.git_sha)
    return tuple(results)


def evaluate_qualified_repair(
    failure: FailurePacket,
    candidate: RepairCandidate,
    receipts: tuple[GateReceipt, ...],
    *,
    certifier_actor_id: str,
    repair_policy: AIQAPolicy,
    evidence_policy: EvidencePolicy,
    capability_id: str,
    target: EvidenceTarget,
    evidence: tuple[VerifiedEvidence, ...],
) -> dict[str, object]:
    repair = evaluate_promotion(
        failure, candidate, receipts, certifier_actor_id=certifier_actor_id, policy=repair_policy,
    )
    if candidate.candidate_git_sha != target.git_sha:
        raise ValueError("repair candidate differs from the evidence target")
    qualification = capability_readiness(evidence_policy, capability_id, target, evidence)
    eligible = (
        repair.decision == "READY_FOR_INDEPENDENT_PROMOTION"
        and qualification["evidence_complete"]
        and bool(qualification["package_bound_levels"])
    )
    return {"decision": "READY_FOR_INDEPENDENT_PROMOTION" if eligible else "BLOCK",
            "repair_gate": repair.to_dict(), "capability_evidence": qualification}
