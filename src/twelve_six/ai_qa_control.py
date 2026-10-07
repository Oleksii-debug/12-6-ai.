from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from twelve_six.capability_map import CapabilityRegistry, load_capability_registry
from twelve_six.sil_qualification import (
    CommandExecution,
    GitProbe,
    SILScenario,
    load_sil_scenario,
    parse_vector_command,
    probe_git_state,
    run_command,
    verify_sil_evidence,
)


_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_MAX_JSON_BYTES = 4 * 1024 * 1024


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return value


def _require_git_sha(name: str, value: object) -> str:
    if not isinstance(value, str) or _SHA40_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase 40-hex Git SHA")
    return value


def _require_id(name: str, value: object) -> str:
    if not isinstance(value, str) or _ID_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a canonical identifier")
    return value


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object member")
        value[key] = item
    return value


def _strict_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    if len(raw) > _MAX_JSON_BYTES:
        raise ValueError(f"{label} exceeds maximum encoded size")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_unique_json_object,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant is not allowed: {item}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError, RecursionError) as exc:
        raise ValueError(f"{label} is not strict unambiguous UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} root must be a JSON object")
    return value


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(_canonical_json_bytes(payload) + b"\n")


class FailureSource(str, Enum):
    CI = "CI"
    SIL = "SIL"
    PHYSICAL = "PHYSICAL"


class FailureClass(str, Enum):
    TEST = "TEST"
    TIMEOUT = "TIMEOUT"
    ENVIRONMENT = "ENVIRONMENT"
    INTEGRITY = "INTEGRITY"
    PHYSICAL = "PHYSICAL"
    UNKNOWN = "UNKNOWN"


class GateKind(str, Enum):
    COMPONENT = "component"
    ADVERSARIAL = "adversarial"
    SIL = "sil"
    PHYSICAL = "physical"


class GateVerdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class PhysicalScope(str, Enum):
    REQUIRED = "REQUIRED"
    NONE = "NONE"


@dataclass(frozen=True, slots=True)
class AIQAPolicy:
    schema_version: int
    automated_gate_order: tuple[GateKind, ...]
    promotion_gate_order: tuple[GateKind, ...]
    require_independent_certifier: bool
    require_exact_candidate_sha: bool
    physical_not_applicable_requires_explicit_scope: bool
    max_failure_summary_bytes: int
    max_patch_bytes: int

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported AI QA policy schema_version")
        if self.automated_gate_order != (
            GateKind.COMPONENT,
            GateKind.ADVERSARIAL,
        ):
            raise ValueError("automated gate order is non-canonical")
        if self.promotion_gate_order != (
            GateKind.COMPONENT,
            GateKind.ADVERSARIAL,
            GateKind.SIL,
            GateKind.PHYSICAL,
        ):
            raise ValueError("promotion gate order is non-canonical")
        for name, value in (
            ("require_independent_certifier", self.require_independent_certifier),
            ("require_exact_candidate_sha", self.require_exact_candidate_sha),
            (
                "physical_not_applicable_requires_explicit_scope",
                self.physical_not_applicable_requires_explicit_scope,
            ),
        ):
            if type(value) is not bool or value is not True:
                raise ValueError(f"{name} must remain fail-closed true")
        for name, value in (
            ("max_failure_summary_bytes", self.max_failure_summary_bytes),
            ("max_patch_bytes", self.max_patch_bytes),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def load_ai_qa_policy(path: str | Path) -> AIQAPolicy:
    payload = _strict_json_object(path, label="AI QA policy")
    expected = {
        "schema_version",
        "automated_gate_order",
        "promotion_gate_order",
        "require_independent_certifier",
        "require_exact_candidate_sha",
        "physical_not_applicable_requires_explicit_scope",
        "max_failure_summary_bytes",
        "max_patch_bytes",
    }
    if set(payload) != expected:
        raise ValueError("AI QA policy fields are non-canonical")
    automated = payload["automated_gate_order"]
    promotion = payload["promotion_gate_order"]
    if not isinstance(automated, list) or not isinstance(promotion, list):
        raise ValueError("AI QA gate orders must be arrays")
    return AIQAPolicy(
        schema_version=payload["schema_version"],
        automated_gate_order=tuple(GateKind(item) for item in automated),
        promotion_gate_order=tuple(GateKind(item) for item in promotion),
        require_independent_certifier=payload["require_independent_certifier"],
        require_exact_candidate_sha=payload["require_exact_candidate_sha"],
        physical_not_applicable_requires_explicit_scope=payload[
            "physical_not_applicable_requires_explicit_scope"
        ],
        max_failure_summary_bytes=payload["max_failure_summary_bytes"],
        max_patch_bytes=payload["max_patch_bytes"],
    )


def classify_failure(source: FailureSource, summary: str) -> FailureClass:
    if not isinstance(source, FailureSource):
        raise ValueError("source must be a FailureSource")
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("failure summary must be non-empty")
    if source is FailureSource.PHYSICAL:
        return FailureClass.PHYSICAL
    lowered = summary.lower()
    if any(marker in lowered for marker in ("timeout", "timed out", "deadline exceeded")):
        return FailureClass.TIMEOUT
    if any(
        marker in lowered
        for marker in (
            "sha mismatch",
            "hash mismatch",
            "checksum",
            "corrupt",
            "identity mismatch",
            "reseal",
        )
    ):
        return FailureClass.INTEGRITY
    if any(
        marker in lowered
        for marker in (
            "modulenotfounderror",
            "importerror",
            "dependency",
            "no space left",
            "installation failed",
            "runner environment",
        )
    ):
        return FailureClass.ENVIRONMENT
    if any(marker in lowered for marker in ("assert", "failed", "failure", "error")):
        return FailureClass.TEST
    return FailureClass.UNKNOWN


@dataclass(frozen=True, slots=True)
class FailurePacket:
    schema_version: int
    defect_id: str
    source: FailureSource
    failure_class: FailureClass
    failing_git_sha: str
    source_evidence_identity_sha256: str
    failure_summary: str
    failure_summary_sha256: str
    reproducer_argv: tuple[str, ...]
    physical_scope: PhysicalScope
    physical_gate_id: str | None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported FailurePacket schema_version")
        _require_id("defect_id", self.defect_id)
        _require_git_sha("failing_git_sha", self.failing_git_sha)
        _require_sha256(
            "source_evidence_identity_sha256",
            self.source_evidence_identity_sha256,
        )
        if not isinstance(self.failure_summary, str) or not self.failure_summary:
            raise ValueError("failure_summary must be non-empty")
        if _sha256_bytes(self.failure_summary.encode("utf-8")) != self.failure_summary_sha256:
            raise ValueError("failure_summary_sha256 does not match failure_summary")
        if not self.reproducer_argv:
            raise ValueError("a minimal checked-in reproducer is required")
        if self.reproducer_argv[1:4] != ("-m", "pytest", "-q"):
            raise ValueError("reproducer_argv is not a canonical pytest vector")
        if self.physical_scope is PhysicalScope.REQUIRED:
            if self.physical_gate_id is None:
                raise ValueError("physical_gate_id is required for physical scope")
            _require_id("physical_gate_id", self.physical_gate_id)
        elif self.physical_gate_id is not None:
            raise ValueError("physical_gate_id must be null when physical scope is NONE")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "defect_id": self.defect_id,
            "source": self.source.value,
            "failure_class": self.failure_class.value,
            "failing_git_sha": self.failing_git_sha,
            "source_evidence_identity_sha256": self.source_evidence_identity_sha256,
            "failure_summary": self.failure_summary,
            "failure_summary_sha256": self.failure_summary_sha256,
            "reproducer_argv": list(self.reproducer_argv),
            "physical_scope": self.physical_scope.value,
            "physical_gate_id": self.physical_gate_id,
        }

    def identity_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class ExternalObservation:
    schema_version: str
    source: FailureSource
    git_sha: str
    evidence_identity_sha256: str
    failure_summary: str
    reproducer_command: str
    physical_gate_id: str | None

    def __post_init__(self) -> None:
        if self.schema_version != "12-6.aiqa-observation.v1":
            raise ValueError("unsupported external observation schema")
        if self.source not in {FailureSource.CI, FailureSource.PHYSICAL}:
            raise ValueError("external observations support CI or PHYSICAL only")
        _require_git_sha("git_sha", self.git_sha)
        _require_sha256("evidence_identity_sha256", self.evidence_identity_sha256)
        if not isinstance(self.failure_summary, str) or not self.failure_summary:
            raise ValueError("failure_summary must be non-empty")
        parse_vector_command(self.reproducer_command)
        if self.source is FailureSource.PHYSICAL:
            if self.physical_gate_id is None:
                raise ValueError("physical observation requires physical_gate_id")
            _require_id("physical_gate_id", self.physical_gate_id)
        elif self.physical_gate_id is not None:
            raise ValueError("CI observation must not claim a physical gate")


def load_external_observation(path: str | Path) -> ExternalObservation:
    payload = _strict_json_object(path, label="AI QA external observation")
    expected = {
        "schema_version",
        "source",
        "git_sha",
        "evidence_identity_sha256",
        "failure_summary",
        "reproducer_command",
        "physical_gate_id",
    }
    if set(payload) != expected:
        raise ValueError("external observation fields are non-canonical")
    return ExternalObservation(
        schema_version=payload["schema_version"],
        source=FailureSource(payload["source"]),
        git_sha=payload["git_sha"],
        evidence_identity_sha256=payload["evidence_identity_sha256"],
        failure_summary=payload["failure_summary"],
        reproducer_command=payload["reproducer_command"],
        physical_gate_id=payload["physical_gate_id"],
    )


def failure_packet_from_observation(
    observation: ExternalObservation,
    *,
    defect_id: str,
    policy: AIQAPolicy,
) -> FailurePacket:
    if len(observation.failure_summary.encode("utf-8")) > policy.max_failure_summary_bytes:
        raise ValueError("failure summary exceeds AI QA policy bound")
    physical_scope = (
        PhysicalScope.REQUIRED
        if observation.source is FailureSource.PHYSICAL
        else PhysicalScope.NONE
    )
    return FailurePacket(
        schema_version=1,
        defect_id=defect_id,
        source=observation.source,
        failure_class=classify_failure(observation.source, observation.failure_summary),
        failing_git_sha=observation.git_sha,
        source_evidence_identity_sha256=observation.evidence_identity_sha256,
        failure_summary=observation.failure_summary,
        failure_summary_sha256=_sha256_bytes(observation.failure_summary.encode("utf-8")),
        reproducer_argv=parse_vector_command(observation.reproducer_command),
        physical_scope=physical_scope,
        physical_gate_id=observation.physical_gate_id,
    )


def failure_packet_from_sil(
    evidence_path: str | Path,
    log_path: str | Path,
    *,
    defect_id: str,
    policy: AIQAPolicy,
    expected_package_bytes: bytes,
    expected_registry: CapabilityRegistry,
    expected_scenario: SILScenario,
    physical_scope: PhysicalScope = PhysicalScope.NONE,
    physical_gate_id: str | None = None,
) -> FailurePacket:
    evidence = verify_sil_evidence(
        evidence_path,
        log_path,
        expected_package_bytes=expected_package_bytes,
        expected_registry=expected_registry,
        expected_scenario=expected_scenario,
        expected_git_sha=None,
        require_pass=False,
    )
    if evidence["verdict"] != "FAIL":
        raise ValueError("SIL evidence is not a failure")
    failed = [
        item
        for item in evidence["executions"]
        if type(item) is dict and item.get("return_code") != 0
    ]
    if not failed:
        raise ValueError("SIL FAIL has no failing integration execution")
    argv = failed[0].get("argv")
    if (
        not isinstance(argv, list)
        or len(argv) < 5
        or argv[1:4] != ["-m", "pytest", "-q"]
    ):
        raise ValueError("SIL failure has no canonical pytest reproducer")
    test_paths = argv[4:]
    if not all(isinstance(item, str) for item in test_paths):
        raise ValueError("SIL reproducer paths are malformed")
    reproducer = parse_vector_command("pytest -q " + " ".join(test_paths))

    log_text = Path(log_path).read_text(encoding="utf-8", errors="strict")
    summary = log_text[-policy.max_failure_summary_bytes :]
    if not summary.strip():
        summary = (
            f"SIL vector {failed[0].get('vector_id')} failed with "
            f"return code {failed[0].get('return_code')}"
        )
    return FailurePacket(
        schema_version=1,
        defect_id=defect_id,
        source=FailureSource.SIL,
        failure_class=classify_failure(FailureSource.SIL, summary),
        failing_git_sha=evidence["git_sha"],
        source_evidence_identity_sha256=evidence["evidence_identity_sha256"],
        failure_summary=summary,
        failure_summary_sha256=_sha256_bytes(summary.encode("utf-8")),
        reproducer_argv=reproducer,
        physical_scope=physical_scope,
        physical_gate_id=physical_gate_id,
    )


@dataclass(frozen=True, slots=True)
class RepairCandidate:
    schema_version: int
    defect_id: str
    base_git_sha: str
    candidate_git_sha: str
    patch_sha256: str
    proposer_actor_id: str
    failure_packet_identity_sha256: str

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported RepairCandidate schema_version")
        _require_id("defect_id", self.defect_id)
        _require_git_sha("base_git_sha", self.base_git_sha)
        _require_git_sha("candidate_git_sha", self.candidate_git_sha)
        _require_sha256("patch_sha256", self.patch_sha256)
        _require_id("proposer_actor_id", self.proposer_actor_id)
        _require_sha256(
            "failure_packet_identity_sha256",
            self.failure_packet_identity_sha256,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "defect_id": self.defect_id,
            "base_git_sha": self.base_git_sha,
            "candidate_git_sha": self.candidate_git_sha,
            "patch_sha256": self.patch_sha256,
            "proposer_actor_id": self.proposer_actor_id,
            "failure_packet_identity_sha256": self.failure_packet_identity_sha256,
        }

    def identity_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


def build_repair_candidate(
    failure: FailurePacket,
    *,
    base_git_sha: str,
    candidate_git_sha: str,
    patch_bytes: bytes,
    proposer_actor_id: str,
    policy: AIQAPolicy,
) -> RepairCandidate:
    if not isinstance(patch_bytes, bytes) or not patch_bytes:
        raise ValueError("isolated repair patch must be non-empty bytes")
    if len(patch_bytes) > policy.max_patch_bytes:
        raise ValueError("isolated repair patch exceeds AI QA policy bound")
    return RepairCandidate(
        schema_version=1,
        defect_id=failure.defect_id,
        base_git_sha=base_git_sha,
        candidate_git_sha=candidate_git_sha,
        patch_sha256=_sha256_bytes(patch_bytes),
        proposer_actor_id=proposer_actor_id,
        failure_packet_identity_sha256=failure.identity_sha256(),
    )


@dataclass(frozen=True, slots=True)
class RegressionChain:
    schema_version: int
    defect_id: str
    candidate_identity_sha256: str
    candidate_git_sha: str
    component_argv: tuple[str, ...]
    adversarial_argv: tuple[str, ...]
    physical_scope: PhysicalScope
    physical_gate_id: str | None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported RegressionChain schema_version")
        _require_id("defect_id", self.defect_id)
        _require_sha256("candidate_identity_sha256", self.candidate_identity_sha256)
        _require_git_sha("candidate_git_sha", self.candidate_git_sha)
        for argv in (self.component_argv, self.adversarial_argv):
            if not argv or argv[1:4] != ("-m", "pytest", "-q"):
                raise ValueError("regression chain commands must be canonical pytest vectors")
        if self.physical_scope is PhysicalScope.REQUIRED:
            if self.physical_gate_id is None:
                raise ValueError("physical regression gate identity is required")
        elif self.physical_gate_id is not None:
            raise ValueError("physical gate must be null for physical scope NONE")


def build_regression_chain(
    failure: FailurePacket,
    candidate: RepairCandidate,
    *,
    adversarial_command: str,
) -> RegressionChain:
    if candidate.defect_id != failure.defect_id:
        raise ValueError("repair candidate defect identity does not match failure packet")
    if candidate.failure_packet_identity_sha256 != failure.identity_sha256():
        raise ValueError("repair candidate is not bound to this failure packet")
    return RegressionChain(
        schema_version=1,
        defect_id=failure.defect_id,
        candidate_identity_sha256=candidate.identity_sha256(),
        candidate_git_sha=candidate.candidate_git_sha,
        component_argv=failure.reproducer_argv,
        adversarial_argv=parse_vector_command(adversarial_command),
        physical_scope=failure.physical_scope,
        physical_gate_id=failure.physical_gate_id,
    )


@dataclass(frozen=True, slots=True)
class GateReceipt:
    gate: GateKind
    verdict: GateVerdict
    git_sha: str
    evidence_identity_sha256: str
    actor_id: str
    reason: str | None = None

    def __post_init__(self) -> None:
        _require_git_sha("gate receipt git_sha", self.git_sha)
        _require_sha256("gate receipt evidence_identity_sha256", self.evidence_identity_sha256)
        _require_id("gate receipt actor_id", self.actor_id)
        if self.verdict is GateVerdict.NOT_APPLICABLE:
            if self.gate is not GateKind.PHYSICAL:
                raise ValueError("NOT_APPLICABLE is allowed only for a physical gate")
            if not isinstance(self.reason, str) or not self.reason.strip():
                raise ValueError("NOT_APPLICABLE physical gate needs an explicit reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "gate": self.gate.value,
            "verdict": self.verdict.value,
            "git_sha": self.git_sha,
            "evidence_identity_sha256": self.evidence_identity_sha256,
            "actor_id": self.actor_id,
            "reason": self.reason,
        }


CommandRunner = Callable[
    [tuple[str, ...], Path, int, bytes, str],
    CommandExecution,
]


def execute_automated_regressions(
    chain: RegressionChain,
    *,
    repo_root: str | Path,
    actor_id: str,
    timeout_seconds: int = 300,
    command_runner: CommandRunner = run_command,
    git_probe: GitProbe = probe_git_state,
) -> tuple[GateReceipt, GateReceipt]:
    _require_id("actor_id", actor_id)
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive integer")
    root = Path(repo_root)
    state = git_probe(root)
    if state.sha != chain.candidate_git_sha:
        raise ValueError("regression chain exact candidate SHA mismatch")
    if not state.tracked_clean:
        raise ValueError("regression candidate checkout is dirty")

    receipts: list[GateReceipt] = []
    for gate, argv in (
        (GateKind.COMPONENT, chain.component_argv),
        (GateKind.ADVERSARIAL, chain.adversarial_argv),
    ):
        input_envelope = {
            "schema_version": "12-6.aiqa-regression-input.v1",
            "defect_id": chain.defect_id,
            "candidate_identity_sha256": chain.candidate_identity_sha256,
            "candidate_git_sha": chain.candidate_git_sha,
            "gate": gate.value,
            "argv": list(argv),
        }
        input_envelope_bytes = _canonical_json_bytes(input_envelope)
        input_identity = _sha256_bytes(input_envelope_bytes)
        result = command_runner(
            argv,
            root,
            timeout_seconds,
            input_envelope_bytes,
            input_identity,
        )
        evidence = {
            "gate": gate.value,
            "git_sha": chain.candidate_git_sha,
            "argv": list(argv),
            "expected_input_identity_sha256": input_identity,
            "consumed_input_identity_sha256": result.consumed_input_identity_sha256,
            "return_code": result.return_code,
            "stdout_sha256": _sha256_bytes(result.stdout.encode("utf-8")),
            "stderr_sha256": _sha256_bytes(result.stderr.encode("utf-8")),
            "duration_ms": result.duration_ms,
        }
        receipts.append(
            GateReceipt(
                gate=gate,
                verdict=(
                    GateVerdict.PASS
                    if result.return_code == 0
                    and result.consumed_input_identity_sha256 == input_identity
                    else GateVerdict.FAIL
                ),
                git_sha=chain.candidate_git_sha,
                evidence_identity_sha256=_canonical_sha256(evidence),
                actor_id=actor_id,
                reason=None if result.return_code == 0 else "automated regression failed",
            )
        )
    return receipts[0], receipts[1]


@dataclass(frozen=True, slots=True)
class PromotionDecision:
    decision: str
    candidate_identity_sha256: str
    certifier_actor_id: str
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.decision not in {"PROMOTE", "BLOCK"}:
            raise ValueError("promotion decision must be PROMOTE or BLOCK")
        _require_sha256("candidate_identity_sha256", self.candidate_identity_sha256)
        _require_id("certifier_actor_id", self.certifier_actor_id)
        if self.decision == "PROMOTE" and self.reasons:
            raise ValueError("PROMOTE cannot contain blocker reasons")
        if self.decision == "BLOCK" and not self.reasons:
            raise ValueError("BLOCK needs at least one reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "candidate_identity_sha256": self.candidate_identity_sha256,
            "certifier_actor_id": self.certifier_actor_id,
            "reasons": list(self.reasons),
        }

    def identity_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


def evaluate_promotion(
    failure: FailurePacket,
    candidate: RepairCandidate,
    receipts: tuple[GateReceipt, ...],
    *,
    certifier_actor_id: str,
    policy: AIQAPolicy,
) -> PromotionDecision:
    _require_id("certifier_actor_id", certifier_actor_id)
    if candidate.defect_id != failure.defect_id:
        raise ValueError("candidate/failure defect mismatch")
    if candidate.failure_packet_identity_sha256 != failure.identity_sha256():
        raise ValueError("candidate/failure packet binding mismatch")

    reasons: list[str] = []
    if (
        policy.require_independent_certifier
        and certifier_actor_id == candidate.proposer_actor_id
    ):
        reasons.append("certifier is the repair proposer")

    by_gate: dict[GateKind, GateReceipt] = {}
    for receipt in receipts:
        if receipt.gate in by_gate:
            raise ValueError(f"duplicate gate receipt: {receipt.gate.value}")
        by_gate[receipt.gate] = receipt
        if policy.require_exact_candidate_sha and receipt.git_sha != candidate.candidate_git_sha:
            reasons.append(f"{receipt.gate.value} receipt is bound to a different Git SHA")

    for gate in policy.promotion_gate_order:
        if gate not in by_gate:
            reasons.append(f"missing {gate.value} gate receipt")

    for gate in (GateKind.COMPONENT, GateKind.ADVERSARIAL, GateKind.SIL):
        receipt = by_gate.get(gate)
        if receipt is not None and receipt.verdict is not GateVerdict.PASS:
            reasons.append(f"{gate.value} gate is not PASS")

    physical = by_gate.get(GateKind.PHYSICAL)
    if physical is not None:
        if failure.physical_scope is PhysicalScope.REQUIRED:
            if physical.verdict is not GateVerdict.PASS:
                reasons.append("required physical gate is not PASS")
        else:
            if physical.verdict not in {GateVerdict.PASS, GateVerdict.NOT_APPLICABLE}:
                reasons.append("physical gate is neither PASS nor explicit NOT_APPLICABLE")
            if (
                policy.physical_not_applicable_requires_explicit_scope
                and physical.verdict is GateVerdict.NOT_APPLICABLE
                and failure.physical_scope is not PhysicalScope.NONE
            ):
                reasons.append("physical NOT_APPLICABLE conflicts with failure scope")

    for gate in (GateKind.SIL, GateKind.PHYSICAL):
        receipt = by_gate.get(gate)
        if receipt is not None and receipt.actor_id == candidate.proposer_actor_id:
            reasons.append(f"{gate.value} evidence is not independent of repair proposer")

    if reasons:
        return PromotionDecision(
            decision="BLOCK",
            candidate_identity_sha256=candidate.identity_sha256(),
            certifier_actor_id=certifier_actor_id,
            reasons=tuple(reasons),
        )
    return PromotionDecision(
        decision="PROMOTE",
        candidate_identity_sha256=candidate.identity_sha256(),
        certifier_actor_id=certifier_actor_id,
        reasons=(),
    )



def load_failure_packet(path: str | Path) -> FailurePacket:
    payload = _strict_json_object(path, label="AI QA failure packet")
    expected = {
        "schema_version",
        "defect_id",
        "source",
        "failure_class",
        "failing_git_sha",
        "source_evidence_identity_sha256",
        "failure_summary",
        "failure_summary_sha256",
        "reproducer_argv",
        "physical_scope",
        "physical_gate_id",
        "failure_packet_identity_sha256",
    }
    if set(payload) != expected:
        raise ValueError("failure packet fields are non-canonical")
    argv = payload["reproducer_argv"]
    if not isinstance(argv, list) or not all(isinstance(item, str) for item in argv):
        raise ValueError("failure packet reproducer_argv must be a string array")
    packet = FailurePacket(
        schema_version=payload["schema_version"],
        defect_id=payload["defect_id"],
        source=FailureSource(payload["source"]),
        failure_class=FailureClass(payload["failure_class"]),
        failing_git_sha=payload["failing_git_sha"],
        source_evidence_identity_sha256=payload["source_evidence_identity_sha256"],
        failure_summary=payload["failure_summary"],
        failure_summary_sha256=payload["failure_summary_sha256"],
        reproducer_argv=tuple(argv),
        physical_scope=PhysicalScope(payload["physical_scope"]),
        physical_gate_id=payload["physical_gate_id"],
    )
    identity = _require_sha256(
        "failure_packet_identity_sha256",
        payload["failure_packet_identity_sha256"],
    )
    if packet.identity_sha256() != identity:
        raise ValueError("failure packet identity mismatch")
    return packet


def load_repair_candidate(path: str | Path) -> RepairCandidate:
    payload = _strict_json_object(path, label="AI QA repair candidate")
    expected = {
        "schema_version",
        "defect_id",
        "base_git_sha",
        "candidate_git_sha",
        "patch_sha256",
        "proposer_actor_id",
        "failure_packet_identity_sha256",
        "candidate_identity_sha256",
    }
    if set(payload) != expected:
        raise ValueError("repair candidate fields are non-canonical")
    candidate = RepairCandidate(
        schema_version=payload["schema_version"],
        defect_id=payload["defect_id"],
        base_git_sha=payload["base_git_sha"],
        candidate_git_sha=payload["candidate_git_sha"],
        patch_sha256=payload["patch_sha256"],
        proposer_actor_id=payload["proposer_actor_id"],
        failure_packet_identity_sha256=payload["failure_packet_identity_sha256"],
    )
    identity = _require_sha256(
        "candidate_identity_sha256",
        payload["candidate_identity_sha256"],
    )
    if candidate.identity_sha256() != identity:
        raise ValueError("repair candidate identity mismatch")
    return candidate


def _receipt_from_dict(value: object) -> GateReceipt:
    if not isinstance(value, dict) or set(value) != {
        "gate",
        "verdict",
        "git_sha",
        "evidence_identity_sha256",
        "actor_id",
        "reason",
    }:
        raise ValueError("gate receipt fields are non-canonical")
    return GateReceipt(
        gate=GateKind(value["gate"]),
        verdict=GateVerdict(value["verdict"]),
        git_sha=value["git_sha"],
        evidence_identity_sha256=value["evidence_identity_sha256"],
        actor_id=value["actor_id"],
        reason=value["reason"],
    )


def load_gate_receipt_bundle(
    path: str | Path,
    *,
    expected_candidate_identity_sha256: str,
) -> tuple[GateReceipt, ...]:
    payload = _strict_json_object(path, label="AI QA gate receipt bundle")
    if set(payload) != {
        "schema_version",
        "candidate_identity_sha256",
        "receipts",
    }:
        raise ValueError("gate receipt bundle fields are non-canonical")
    if payload["schema_version"] != "12-6.aiqa-gate-receipts.v1":
        raise ValueError("unsupported gate receipt bundle schema")
    expected_identity = _require_sha256(
        "expected_candidate_identity_sha256",
        expected_candidate_identity_sha256,
    )
    if payload["candidate_identity_sha256"] != expected_identity:
        raise ValueError("gate receipt bundle candidate identity mismatch")
    receipts = payload["receipts"]
    if not isinstance(receipts, list) or not receipts:
        raise ValueError("gate receipt bundle must contain receipts")
    return tuple(_receipt_from_dict(item) for item in receipts)


def _write_receipt_bundle(
    path: str | Path,
    candidate: RepairCandidate,
    receipts: tuple[GateReceipt, ...],
) -> None:
    if not receipts:
        raise ValueError("receipt bundle cannot be empty")
    _write_json(
        path,
        {
            "schema_version": "12-6.aiqa-gate-receipts.v1",
            "candidate_identity_sha256": candidate.identity_sha256(),
            "receipts": [receipt.to_dict() for receipt in receipts],
        },
    )


def _candidate_cli(args: argparse.Namespace) -> int:
    policy = load_ai_qa_policy(args.policy)
    failure = load_failure_packet(args.failure)
    candidate = build_repair_candidate(
        failure,
        base_git_sha=args.base_git_sha,
        candidate_git_sha=args.candidate_git_sha,
        patch_bytes=Path(args.patch_file).read_bytes(),
        proposer_actor_id=args.proposer_actor_id,
        policy=policy,
    )
    payload = candidate.to_dict()
    payload["candidate_identity_sha256"] = candidate.identity_sha256()
    _write_json(args.output, payload)
    print(
        json.dumps(
            {
                "defect_id": candidate.defect_id,
                "candidate_git_sha": candidate.candidate_git_sha,
                "candidate_identity_sha256": candidate.identity_sha256(),
            },
            sort_keys=True,
        )
    )
    return 0


def _regression_cli(args: argparse.Namespace) -> int:
    failure = load_failure_packet(args.failure)
    candidate = load_repair_candidate(args.candidate)
    chain = build_regression_chain(
        failure,
        candidate,
        adversarial_command=args.adversarial_command,
    )
    receipts = execute_automated_regressions(
        chain,
        repo_root=args.repo_root,
        actor_id=args.actor_id,
        timeout_seconds=args.timeout_seconds,
    )
    _write_receipt_bundle(args.output, candidate, receipts)
    passed = all(receipt.verdict is GateVerdict.PASS for receipt in receipts)
    print(
        json.dumps(
            {
                "candidate_git_sha": candidate.candidate_git_sha,
                "component": receipts[0].verdict.value,
                "adversarial": receipts[1].verdict.value,
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 1


def _sil_receipt_cli(args: argparse.Namespace) -> int:
    candidate = load_repair_candidate(args.candidate)
    root = Path(args.repo_root).resolve()
    registry = load_capability_registry(args.capability_registry)
    scenario = load_sil_scenario(args.scenario)
    evidence = verify_sil_evidence(
        args.evidence,
        args.log,
        expected_package_bytes=(root / "pyproject.toml").read_bytes(),
        expected_registry=registry,
        expected_scenario=scenario,
        expected_git_sha=candidate.candidate_git_sha,
        require_pass=False,
    )
    receipt = GateReceipt(
        gate=GateKind.SIL,
        verdict=(
            GateVerdict.PASS
            if evidence["verdict"] == "PASS"
            else GateVerdict.FAIL
        ),
        git_sha=candidate.candidate_git_sha,
        evidence_identity_sha256=evidence["evidence_identity_sha256"],
        actor_id=args.actor_id,
        reason=None if evidence["verdict"] == "PASS" else "SIL evidence verdict is FAIL",
    )
    _write_receipt_bundle(args.output, candidate, (receipt,))
    return 0 if receipt.verdict is GateVerdict.PASS else 1


def _physical_scope_receipt_cli(args: argparse.Namespace) -> int:
    failure = load_failure_packet(args.failure)
    candidate = load_repair_candidate(args.candidate)
    if failure.physical_scope is not PhysicalScope.NONE:
        raise ValueError("required physical scope cannot be replaced by NOT_APPLICABLE")
    receipt_payload = {
        "candidate_identity_sha256": candidate.identity_sha256(),
        "candidate_git_sha": candidate.candidate_git_sha,
        "failure_packet_identity_sha256": failure.identity_sha256(),
        "physical_scope": failure.physical_scope.value,
        "reason": args.reason,
        "actor_id": args.actor_id,
    }
    receipt = GateReceipt(
        gate=GateKind.PHYSICAL,
        verdict=GateVerdict.NOT_APPLICABLE,
        git_sha=candidate.candidate_git_sha,
        evidence_identity_sha256=_canonical_sha256(receipt_payload),
        actor_id=args.actor_id,
        reason=args.reason,
    )
    _write_receipt_bundle(args.output, candidate, (receipt,))
    return 0


def _assess_cli(args: argparse.Namespace) -> int:
    policy = load_ai_qa_policy(args.policy)
    failure = load_failure_packet(args.failure)
    candidate = load_repair_candidate(args.candidate)
    receipts: list[GateReceipt] = []
    for bundle in args.receipt_bundle:
        receipts.extend(
            load_gate_receipt_bundle(
                bundle,
                expected_candidate_identity_sha256=candidate.identity_sha256(),
            )
        )
    decision = evaluate_promotion(
        failure,
        candidate,
        tuple(receipts),
        certifier_actor_id=args.certifier_actor_id,
        policy=policy,
    )
    payload = decision.to_dict()
    payload["decision_identity_sha256"] = decision.identity_sha256()
    _write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0 if decision.decision == "PROMOTE" else 1


def _sil_failure_cli(args: argparse.Namespace) -> int:
    policy = load_ai_qa_policy(args.policy)
    root = Path(args.repo_root).resolve()
    registry = load_capability_registry(args.capability_registry)
    scenario = load_sil_scenario(args.scenario)
    packet = failure_packet_from_sil(
        args.evidence,
        args.log,
        defect_id=args.defect_id,
        policy=policy,
        expected_package_bytes=(root / "pyproject.toml").read_bytes(),
        expected_registry=registry,
        expected_scenario=scenario,
        physical_scope=PhysicalScope(args.physical_scope),
        physical_gate_id=args.physical_gate_id,
    )
    payload = packet.to_dict()
    payload["failure_packet_identity_sha256"] = packet.identity_sha256()
    _write_json(args.output, payload)
    print(json.dumps({"defect_id": packet.defect_id, "failure_class": packet.failure_class.value}))
    return 0


def _observation_cli(args: argparse.Namespace) -> int:
    policy = load_ai_qa_policy(args.policy)
    packet = failure_packet_from_observation(
        load_external_observation(args.observation),
        defect_id=args.defect_id,
        policy=policy,
    )
    payload = packet.to_dict()
    payload["failure_packet_identity_sha256"] = packet.identity_sha256()
    _write_json(args.output, payload)
    print(json.dumps({"defect_id": packet.defect_id, "failure_class": packet.failure_class.value}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="12-6 AI QA failure classification and repair-control plane."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    sil = subparsers.add_parser("sil-failure")
    sil.add_argument("--repo-root", required=True)
    sil.add_argument("--capability-registry", required=True)
    sil.add_argument("--scenario", required=True)
    sil.add_argument("--evidence", required=True)
    sil.add_argument("--log", required=True)
    sil.add_argument("--defect-id", required=True)
    sil.add_argument("--policy", required=True)
    sil.add_argument("--physical-scope", choices=("NONE", "REQUIRED"), default="NONE")
    sil.add_argument("--physical-gate-id")
    sil.add_argument("--output", required=True)
    sil.set_defaults(func=_sil_failure_cli)

    observation = subparsers.add_parser("observation")
    observation.add_argument("--observation", required=True)
    observation.add_argument("--defect-id", required=True)
    observation.add_argument("--policy", required=True)
    observation.add_argument("--output", required=True)
    observation.set_defaults(func=_observation_cli)

    candidate = subparsers.add_parser("candidate")
    candidate.add_argument("--failure", required=True)
    candidate.add_argument("--policy", required=True)
    candidate.add_argument("--base-git-sha", required=True)
    candidate.add_argument("--candidate-git-sha", required=True)
    candidate.add_argument("--patch-file", required=True)
    candidate.add_argument("--proposer-actor-id", required=True)
    candidate.add_argument("--output", required=True)
    candidate.set_defaults(func=_candidate_cli)

    regression = subparsers.add_parser("run-regression")
    regression.add_argument("--failure", required=True)
    regression.add_argument("--candidate", required=True)
    regression.add_argument("--repo-root", required=True)
    regression.add_argument("--adversarial-command", required=True)
    regression.add_argument("--actor-id", required=True)
    regression.add_argument("--timeout-seconds", type=int, default=300)
    regression.add_argument("--output", required=True)
    regression.set_defaults(func=_regression_cli)

    sil_receipt = subparsers.add_parser("sil-receipt")
    sil_receipt.add_argument("--candidate", required=True)
    sil_receipt.add_argument("--repo-root", required=True)
    sil_receipt.add_argument("--capability-registry", required=True)
    sil_receipt.add_argument("--scenario", required=True)
    sil_receipt.add_argument("--evidence", required=True)
    sil_receipt.add_argument("--log", required=True)
    sil_receipt.add_argument("--actor-id", required=True)
    sil_receipt.add_argument("--output", required=True)
    sil_receipt.set_defaults(func=_sil_receipt_cli)

    physical_scope = subparsers.add_parser("physical-scope-receipt")
    physical_scope.add_argument("--failure", required=True)
    physical_scope.add_argument("--candidate", required=True)
    physical_scope.add_argument("--actor-id", required=True)
    physical_scope.add_argument("--reason", required=True)
    physical_scope.add_argument("--output", required=True)
    physical_scope.set_defaults(func=_physical_scope_receipt_cli)

    assess = subparsers.add_parser("assess")
    assess.add_argument("--failure", required=True)
    assess.add_argument("--candidate", required=True)
    assess.add_argument("--policy", required=True)
    assess.add_argument("--receipt-bundle", action="append", required=True)
    assess.add_argument("--certifier-actor-id", required=True)
    assess.add_argument("--output", required=True)
    assess.set_defaults(func=_assess_cli)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
