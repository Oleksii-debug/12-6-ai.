from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any

from twelve_six.capability_map import CapabilityRegistry, load_capability_registry
from twelve_six.sil_qualification import (
    CommandExecution,
    GitProbe,
    GitState,
    SILScenario,
    build_package_manifest_bytes,
    load_sil_environment_receipt,
    load_sil_scenario,
    parse_vector_command,
    probe_git_state,
    require_exact_clean_git_state,
    run_command,
    verify_sil_evidence,
)

_SHA40_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,95}$")
_MAX_JSON_BYTES = 4 * 1024 * 1024

# Autonomous repair must not be allowed to rewrite the machinery that judges
# whether the repair is acceptable.  These paths are qualification/control
# trust roots; changing them requires a separate reviewed lineage.
_PROTECTED_REPAIR_EXACT_PATHS = frozenset(
    {
        ".gitignore",
        "AGENTS.md",
        "SEQUENTIAL_CLOSURE_STATE.md",
        "conftest.py",
        "pyproject.toml",
        "src/twelve_six/__init__.py",
        "pytest.ini",
        "setup.cfg",
        "tox.ini",
        "src/twelve_six/ai_qa_control.py",
        "src/twelve_six/capability_map.py",
        "src/twelve_six/ci_workflow_policy.py",
        "src/twelve_six/sil_qualification.py",
    }
)
_PROTECTED_REPAIR_PREFIXES = (
    ".github/",
    "configs/control/",
    "requirements/",
    "tests/",
    "tools/",
)


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


def _is_exact_type(value: object, expected: type[object]) -> bool:
    return type(value) is expected


def _require_sha256(name: str, value: object) -> str:
    if not _is_exact_type(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return value


def _require_git_sha(name: str, value: object) -> str:
    if not _is_exact_type(value, str) or _SHA40_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be an exact lowercase 40-hex Git SHA")
    return value


def _require_id(name: str, value: object) -> str:
    if not _is_exact_type(value, str) or _ID_RE.fullmatch(value) is None:
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
    if not _is_exact_type(value, dict):
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


_SEALED_FAILURE_SOURCES = tuple((item, str.__str__(item)) for item in FailureSource)
_SEALED_FAILURE_CLASSES = tuple((item, str.__str__(item)) for item in FailureClass)
_SEALED_GATE_KINDS = tuple((item, str.__str__(item)) for item in GateKind)
_SEALED_GATE_VERDICTS = tuple((item, str.__str__(item)) for item in GateVerdict)
_SEALED_PHYSICAL_SCOPES = tuple((item, str.__str__(item)) for item in PhysicalScope)
_SEALED_ENUM_POLICIES = (
    (FailureSource, _SEALED_FAILURE_SOURCES),
    (FailureClass, _SEALED_FAILURE_CLASSES),
    (GateKind, _SEALED_GATE_KINDS),
    (GateVerdict, _SEALED_GATE_VERDICTS),
    (PhysicalScope, _SEALED_PHYSICAL_SCOPES),
)


def _require_sealed_enum(
    value: object,
    enum_type: type[Enum],
    sealed_members: tuple[tuple[Enum, str], ...],
    *,
    type_error: str,
    wire_error: str,
    _sealed_policies: tuple[
        tuple[type[Enum], tuple[tuple[Enum, str], ...]], ...
    ] = _SEALED_ENUM_POLICIES,
) -> Enum:
    canonical_members = next(
        (
            members
            for policy_type, members in _sealed_policies
            if enum_type is policy_type
        ),
        None,
    )
    if canonical_members is None or sealed_members is not canonical_members:
        raise ValueError(wire_error)
    if type(value) is not enum_type:
        raise ValueError(type_error)
    for member, wire_value in canonical_members:
        if value is member:
            raw_value = str.__str__(member)
            stored_value = object.__getattribute__(member, "_value_")
            if (
                not _is_exact_type(stored_value, str)
                or stored_value != raw_value
                or raw_value != wire_value
            ):
                raise ValueError(wire_error)
            return member
    raise ValueError(type_error)


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
        if (
            not _is_exact_type(self.automated_gate_order, tuple)
            or any(
                not _is_exact_type(item, GateKind)
                for item in self.automated_gate_order
            )
        ):
            raise ValueError("automated gate order is non-canonical")
        if (
            not _is_exact_type(self.promotion_gate_order, tuple)
            or any(
                not _is_exact_type(item, GateKind)
                for item in self.promotion_gate_order
            )
        ):
            raise ValueError("promotion gate order is non-canonical")
        for item in (*self.automated_gate_order, *self.promotion_gate_order):
            _require_sealed_enum(
                item,
                GateKind,
                _SEALED_GATE_KINDS,
                type_error="AI QA gate order is non-canonical",
                wire_error="AI QA gate wire value is non-canonical",
            )
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
    if not _is_exact_type(automated, list) or not _is_exact_type(promotion, list):
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


def _require_canonical_pytest_argv(
    name: str,
    value: object,
) -> tuple[str, ...]:
    """Require the exact no-shell pytest argv emitted by parse_vector_command."""

    if (
        not _is_exact_type(value, tuple)
        or len(value) < 5
        or not all(_is_exact_type(item, str) for item in value)
    ):
        raise ValueError(f"{name} must be a canonical pytest argv tuple")
    try:
        expected = parse_vector_command(
            "pytest -q " + " ".join(shlex.quote(path) for path in value[4:])
        )
    except ValueError as exc:
        raise ValueError(f"{name} must be a canonical pytest argv tuple") from exc
    if value != expected:
        raise ValueError(
            f"{name} must use the current interpreter and canonical tests/*.py paths"
        )
    return value


def classify_failure(source: FailureSource, summary: str) -> FailureClass:
    _require_sealed_enum(
        source,
        FailureSource,
        _SEALED_FAILURE_SOURCES,
        type_error="source must be a FailureSource",
        wire_error="failure source wire value is non-canonical",
    )
    if not _is_exact_type(summary, str) or not summary.strip():
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
        _require_sealed_enum(
            self.source,
            FailureSource,
            _SEALED_FAILURE_SOURCES,
            type_error="source must be a FailureSource",
            wire_error="failure source wire value is non-canonical",
        )
        _require_sealed_enum(
            self.failure_class,
            FailureClass,
            _SEALED_FAILURE_CLASSES,
            type_error="failure_class must be a FailureClass",
            wire_error="failure class wire value is non-canonical",
        )
        _require_sealed_enum(
            self.physical_scope,
            PhysicalScope,
            _SEALED_PHYSICAL_SCOPES,
            type_error="physical_scope must be a PhysicalScope",
            wire_error="physical scope wire value is non-canonical",
        )
        _require_git_sha("failing_git_sha", self.failing_git_sha)
        _require_sha256(
            "source_evidence_identity_sha256",
            self.source_evidence_identity_sha256,
        )
        if not _is_exact_type(self.failure_summary, str) or not self.failure_summary:
            raise ValueError("failure_summary must be non-empty")
        _require_sha256("failure_summary_sha256", self.failure_summary_sha256)
        if _sha256_bytes(self.failure_summary.encode("utf-8")) != self.failure_summary_sha256:
            raise ValueError("failure_summary_sha256 does not match failure_summary")
        _require_canonical_pytest_argv("reproducer_argv", self.reproducer_argv)
        if self.physical_scope is PhysicalScope.REQUIRED:
            if self.physical_gate_id is None:
                raise ValueError("physical_gate_id is required for physical scope")
            _require_id("physical_gate_id", self.physical_gate_id)
        elif self.physical_gate_id is not None:
            raise ValueError("physical_gate_id must be null when physical scope is NONE")

    def to_dict(self) -> dict[str, Any]:
        FailurePacket.__post_init__(self)
        return {
            "schema_version": self.schema_version,
            "defect_id": self.defect_id,
            "source": str.__str__(self.source),
            "failure_class": str.__str__(self.failure_class),
            "failing_git_sha": self.failing_git_sha,
            "source_evidence_identity_sha256": self.source_evidence_identity_sha256,
            "failure_summary": self.failure_summary,
            "failure_summary_sha256": self.failure_summary_sha256,
            "reproducer_argv": list(self.reproducer_argv),
            "physical_scope": str.__str__(self.physical_scope),
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
        if (
            not _is_exact_type(self.schema_version, str)
            or self.schema_version != "12-6.aiqa-observation.v1"
        ):
            raise ValueError("unsupported external observation schema")
        _require_sealed_enum(
            self.source,
            FailureSource,
            _SEALED_FAILURE_SOURCES,
            type_error="external observations support CI or PHYSICAL only",
            wire_error="failure source wire value is non-canonical",
        )
        if self.source not in {FailureSource.CI, FailureSource.PHYSICAL}:
            raise ValueError("external observations support CI or PHYSICAL only")
        _require_git_sha("git_sha", self.git_sha)
        _require_sha256("evidence_identity_sha256", self.evidence_identity_sha256)
        if not _is_exact_type(self.failure_summary, str) or not self.failure_summary:
            raise ValueError("failure_summary must be non-empty")
        if not _is_exact_type(self.reproducer_command, str):
            raise ValueError("reproducer_command must be text")
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
    if not _is_exact_type(observation, ExternalObservation):
        raise ValueError("observation must be an ExternalObservation")
    if not _is_exact_type(policy, AIQAPolicy):
        raise ValueError("policy must be an AIQAPolicy")
    ExternalObservation.__post_init__(observation)
    AIQAPolicy.__post_init__(policy)
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
    expected_environment_receipt: dict[str, Any],
    expected_registry: CapabilityRegistry,
    expected_scenario: SILScenario,
    expected_git_sha: str | None = None,
    physical_scope: PhysicalScope = PhysicalScope.NONE,
    physical_gate_id: str | None = None,
) -> FailurePacket:
    if not _is_exact_type(policy, AIQAPolicy):
        raise ValueError("policy must be an AIQAPolicy")
    AIQAPolicy.__post_init__(policy)
    _require_sealed_enum(
        physical_scope,
        PhysicalScope,
        _SEALED_PHYSICAL_SCOPES,
        type_error="physical_scope must be a PhysicalScope",
        wire_error="physical scope wire value is non-canonical",
    )
    evidence = verify_sil_evidence(
        evidence_path,
        log_path,
        expected_package_bytes=expected_package_bytes,
        expected_environment_receipt=expected_environment_receipt,
        expected_registry=expected_registry,
        expected_scenario=expected_scenario,
        expected_git_sha=expected_git_sha,
        require_pass=False,
    )
    if evidence["verdict"] != "FAIL":
        raise ValueError("SIL evidence is not a failure")
    failed = [
        (index, item)
        for index, item in enumerate(evidence["executions"])
        if type(item) is dict and item.get("return_code") != 0
    ]
    if not failed:
        raise ValueError("SIL FAIL has no failing integration execution")
    failed_index, failed_execution = failed[0]
    argv = failed_execution.get("argv")
    if (
        not _is_exact_type(argv, list)
        or len(argv) < 5
        or argv[1:4] != ["-m", "pytest", "-q"]
    ):
        raise ValueError("SIL failure has no canonical pytest reproducer")
    test_paths = argv[4:]
    if not all(_is_exact_type(item, str) for item in test_paths):
        raise ValueError("SIL reproducer paths are malformed")
    reproducer = parse_vector_command(
        "pytest -q " + " ".join(shlex.quote(path) for path in test_paths)
    )

    # verify_sil_evidence() has already strictly parsed and cross-bound every JSONL
    # record to the evidence executions.  Select the same failed record by index so
    # a later passing vector cannot overwrite the failure classification context.
    log_records = [
        json.loads(line)
        for line in Path(log_path).read_text(
            encoding="utf-8", errors="strict"
        ).splitlines()
    ]
    if len(log_records) != len(evidence["executions"]):
        raise ValueError("SIL log execution count changed after verification")
    failed_log = log_records[failed_index]
    summary_source = "\n".join(
        part
        for part in (failed_log.get("stderr"), failed_log.get("stdout"))
        if _is_exact_type(part, str) and part
    )
    if not summary_source.strip():
        summary_source = (
            f"SIL vector {failed_execution.get('vector_id')} failed with "
            f"return code {failed_execution.get('return_code')}"
        )
    summary_bytes = summary_source.encode("utf-8")
    if len(summary_bytes) > policy.max_failure_summary_bytes:
        summary = summary_bytes[-policy.max_failure_summary_bytes :].decode(
            "utf-8", errors="ignore"
        )
    else:
        summary = summary_source
    if not summary.strip():
        raise ValueError("bounded SIL failure summary is empty")
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
        RepairCandidate.__post_init__(self)
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
    if not _is_exact_type(failure, FailurePacket):
        raise ValueError("failure must be a FailurePacket")
    if not _is_exact_type(policy, AIQAPolicy):
        raise ValueError("policy must be an AIQAPolicy")
    FailurePacket.__post_init__(failure)
    AIQAPolicy.__post_init__(policy)
    if base_git_sha != failure.failing_git_sha:
        raise ValueError("repair candidate base Git SHA must equal failing Git SHA")
    if not _is_exact_type(patch_bytes, bytes) or not patch_bytes:
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


def _aiqa_git_subprocess_env() -> dict[str, str]:
    """Return a bounded local-Git environment immune to ambient repository redirects."""

    env = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    env.update(
        {
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
        }
    )
    return env


def _git_command(
    repo_root: Path,
    *args: str,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    effective_env = _aiqa_git_subprocess_env() if env is None else env
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=check,
        capture_output=True,
        text=True,
        env=effective_env,
        stdin=subprocess.DEVNULL,
        shell=False,
    )


CandidateParentProbe = Callable[[Path, str], tuple[str, ...]]


def _validate_repair_index_entries(raw_diff: str) -> None:
    """Reject repair deltas that materialize non-regular Git entries."""

    if not _is_exact_type(raw_diff, str):
        raise ValueError("repair index diff must be text")
    for line in raw_diff.splitlines():
        if not line:
            continue
        try:
            metadata, _path = line.split("\t", 1)
            fields = metadata.split()
            old_mode, new_mode = fields[0][1:], fields[1]
            old_sha, new_sha = fields[2], fields[3]
            status = fields[4]
        except (IndexError, ValueError) as exc:
            raise ValueError("repair index emitted a non-canonical raw diff record") from exc
        if (
            len(fields) != 5
            or not old_mode.isdigit()
            or not new_mode.isdigit()
            or _SHA40_RE.fullmatch(old_sha) is None
            or _SHA40_RE.fullmatch(new_sha) is None
            or not status
        ):
            raise ValueError("repair index emitted a non-canonical raw diff record")
        if status.startswith("D"):
            if new_mode != "000000":
                raise ValueError("deleted repair entry has a non-canonical Git mode")
            continue
        if new_mode not in {"100644", "100755"}:
            raise ValueError("repair candidate may materialize only regular Git files")


def _validate_repair_paths(raw_names: str) -> None:
    """Reject repair deltas that can rewrite their own qualification trust roots."""

    if not _is_exact_type(raw_names, str):
        raise ValueError("repair path listing must be text")
    if raw_names and not raw_names.endswith("\0"):
        raise ValueError("repair path listing is missing its NUL delimiter")
    paths = raw_names[:-1].split("\0") if raw_names else []
    if not paths:
        raise ValueError("repair candidate path set must not be empty")

    for path in paths:
        candidate = PurePosixPath(path)
        if (
            not path
            or "\\" in path
            or candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.as_posix() != path
            or any(ord(char) < 32 or ord(char) == 127 for char in path)
        ):
            raise ValueError("repair candidate path is non-canonical")
        if path in _PROTECTED_REPAIR_EXACT_PATHS or any(
            path.startswith(prefix) for prefix in _PROTECTED_REPAIR_PREFIXES
        ):
            raise ValueError(
                "repair candidate may not modify qualification trust-root path: "
                f"{path}"
            )


def probe_candidate_parents(
    repo_root: Path,
    candidate_git_sha: str,
) -> tuple[str, ...]:
    candidate_git_sha = _require_git_sha("candidate_git_sha", candidate_git_sha)
    env = _aiqa_git_subprocess_env()
    result = _git_command(
        repo_root,
        "show",
        "-s",
        "--format=%P",
        candidate_git_sha,
        check=False,
        env=env,
    )
    if result.returncode != 0:
        raise ValueError("cannot resolve repair candidate parents")
    parents = tuple(part for part in result.stdout.strip().split() if part)
    for parent in parents:
        _require_git_sha("repair candidate parent Git SHA", parent)
    return parents


def materialize_local_repair_candidate(
    failure: FailurePacket,
    *,
    repo_root: str | Path,
    patch_bytes: bytes,
    proposer_actor_id: str,
    policy: AIQAPolicy,
) -> tuple[RepairCandidate, str]:
    """Create an isolated deterministic local Git repair lineage from the exact failing SHA."""

    if not _is_exact_type(failure, FailurePacket):
        raise ValueError("failure must be a FailurePacket")
    if not _is_exact_type(policy, AIQAPolicy):
        raise ValueError("policy must be an AIQAPolicy")
    FailurePacket.__post_init__(failure)
    AIQAPolicy.__post_init__(policy)
    _require_id("proposer_actor_id", proposer_actor_id)
    if not _is_exact_type(patch_bytes, bytes) or not patch_bytes:
        raise ValueError("isolated repair patch must be non-empty bytes")
    if len(patch_bytes) > policy.max_patch_bytes:
        raise ValueError("isolated repair patch exceeds AI QA policy bound")

    root = Path(repo_root).resolve()
    base_sha = failure.failing_git_sha
    exact_object_env = _aiqa_git_subprocess_env()
    base_probe = _git_command(
        root,
        "cat-file",
        "-e",
        f"{base_sha}^{{commit}}",
        check=False,
        env=exact_object_env,
    )
    if base_probe.returncode != 0:
        raise ValueError("failing Git SHA is not an available exact local commit")

    patch_sha = _sha256_bytes(patch_bytes)
    branch_name = f"aiqa/repair/{base_sha[:12]}-{patch_sha[:12]}"
    ref_name = f"refs/heads/{branch_name}"
    ref_check = _git_command(root, "check-ref-format", ref_name, check=False)
    if ref_check.returncode != 0:
        raise ValueError("derived repair branch name is not a valid Git ref")

    with tempfile.TemporaryDirectory(prefix="twelve-six-aiqa-") as temp_root:
        temp_path = Path(temp_root)
        patch_path = temp_path / "repair.patch"
        message_path = temp_path / "commit-message.txt"
        index_path = temp_path / "index"
        empty_hooks = temp_path / "empty-hooks"
        empty_hooks.mkdir()
        patch_path.write_bytes(patch_bytes)

        index_env = exact_object_env.copy()
        index_env["GIT_INDEX_FILE"] = str(index_path)
        read_tree = _git_command(root, "read-tree", base_sha, check=False, env=index_env)
        if read_tree.returncode != 0:
            raise ValueError("cannot initialize isolated repair index from failing SHA")

        apply_result = _git_command(
            root,
            "apply",
            "--cached",
            "--whitespace=nowarn",
            str(patch_path),
            check=False,
            env=index_env,
        )
        if apply_result.returncode != 0:
            raise ValueError("isolated repair patch does not apply cleanly to failing SHA")

        staged = _git_command(
            root,
            "diff",
            "--cached",
            "--quiet",
            base_sha,
            "--",
            check=False,
            env=index_env,
        )
        if staged.returncode == 0:
            raise ValueError("isolated repair patch produces no staged change")
        if staged.returncode != 1:
            raise ValueError("cannot verify isolated repair staged delta")

        raw_diff = _git_command(
            root,
            "diff",
            "--cached",
            "--raw",
            "--no-renames",
            "--abbrev=40",
            base_sha,
            "--",
            check=False,
            env=index_env,
        )
        if raw_diff.returncode != 0:
            raise ValueError("cannot inspect isolated repair Git entry modes")
        _validate_repair_index_entries(raw_diff.stdout)

        path_diff = _git_command(
            root,
            "diff",
            "--cached",
            "--name-only",
            "-z",
            "--no-renames",
            base_sha,
            "--",
            check=False,
            env=index_env,
        )
        if path_diff.returncode != 0:
            raise ValueError("cannot inspect isolated repair paths")
        _validate_repair_paths(path_diff.stdout)

        tree_result = _git_command(root, "write-tree", check=False, env=index_env)
        if tree_result.returncode != 0:
            raise ValueError("cannot materialize isolated repair tree")
        tree_sha = tree_result.stdout.strip()
        _require_git_sha("materialized repair tree SHA", tree_sha)

        message = (
            f"AI QA repair {failure.defect_id}\n\n"
            f"Failure-SHA: {base_sha}\n"
            f"Patch-SHA256: {patch_sha}\n"
        )
        message_path.write_text(message, encoding="utf-8")
        commit_env = exact_object_env.copy()
        commit_env.update(
            {
                "GIT_AUTHOR_NAME": "12-6 AI QA",
                "GIT_AUTHOR_EMAIL": "aiqa@localhost",
                "GIT_COMMITTER_NAME": "12-6 AI QA",
                "GIT_COMMITTER_EMAIL": "aiqa@localhost",
                "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
                "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000",
            }
        )
        commit_result = _git_command(
            root,
            "-c",
            "commit.gpgSign=false",
            "commit-tree",
            tree_sha,
            "-p",
            base_sha,
            "-F",
            str(message_path),
            env=commit_env,
            check=False,
        )
        if commit_result.returncode != 0:
            raise ValueError("cannot create isolated repair candidate commit")
        candidate_sha = commit_result.stdout.strip()
        _require_git_sha("materialized candidate Git SHA", candidate_sha)

        parent = _git_command(
            root,
            "show",
            "-s",
            "--format=%P",
            candidate_sha,
            check=False,
            env=exact_object_env,
        )
        if parent.returncode != 0 or parent.stdout.strip() != base_sha:
            raise ValueError("materialized repair candidate parent is not exact failing SHA")

        existing = _git_command(
            root,
            "rev-parse",
            "--verify",
            "--quiet",
            ref_name,
            check=False,
            env=exact_object_env,
        )
        if existing.returncode == 0:
            if existing.stdout.strip() != candidate_sha:
                raise ValueError("repair branch already exists with a different candidate")
        elif existing.returncode == 1:
            created = _git_command(
                root,
                "-c",
                f"core.hooksPath={empty_hooks}",
                "update-ref",
                ref_name,
                candidate_sha,
                "0" * 40,
                check=False,
                env=exact_object_env,
            )
            if created.returncode != 0:
                raise ValueError("cannot atomically create isolated repair branch")
        else:
            raise ValueError("cannot inspect isolated repair branch state")

    return (
        build_repair_candidate(
            failure,
            base_git_sha=base_sha,
            candidate_git_sha=candidate_sha,
            patch_bytes=patch_bytes,
            proposer_actor_id=proposer_actor_id,
            policy=policy,
        ),
        branch_name,
    )

@dataclass(frozen=True, slots=True)
class RegressionChain:
    schema_version: int
    defect_id: str
    candidate_identity_sha256: str
    base_git_sha: str
    candidate_git_sha: str
    component_argv: tuple[str, ...]
    adversarial_argv: tuple[str, ...]
    physical_scope: PhysicalScope
    physical_gate_id: str | None

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("unsupported RegressionChain schema_version")
        _require_sealed_enum(
            self.physical_scope,
            PhysicalScope,
            _SEALED_PHYSICAL_SCOPES,
            type_error="physical_scope must be a PhysicalScope",
            wire_error="physical scope wire value is non-canonical",
        )
        _require_id("defect_id", self.defect_id)
        _require_sha256("candidate_identity_sha256", self.candidate_identity_sha256)
        _require_git_sha("base_git_sha", self.base_git_sha)
        _require_git_sha("candidate_git_sha", self.candidate_git_sha)
        _require_canonical_pytest_argv("component_argv", self.component_argv)
        _require_canonical_pytest_argv("adversarial_argv", self.adversarial_argv)
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
    if not _is_exact_type(failure, FailurePacket):
        raise ValueError("failure must be a FailurePacket")
    if not _is_exact_type(candidate, RepairCandidate):
        raise ValueError("candidate must be a RepairCandidate")
    FailurePacket.__post_init__(failure)
    RepairCandidate.__post_init__(candidate)
    if not _is_exact_type(adversarial_command, str):
        raise ValueError("adversarial_command must be text")
    if candidate.defect_id != failure.defect_id:
        raise ValueError("repair candidate defect identity does not match failure packet")
    if candidate.failure_packet_identity_sha256 != failure.identity_sha256():
        raise ValueError("repair candidate is not bound to this failure packet")
    if candidate.base_git_sha != failure.failing_git_sha:
        raise ValueError("repair candidate base Git SHA does not match failing Git SHA")
    return RegressionChain(
        schema_version=1,
        defect_id=failure.defect_id,
        candidate_identity_sha256=candidate.identity_sha256(),
        base_git_sha=candidate.base_git_sha,
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
        _require_sealed_enum(
            self.gate,
            GateKind,
            _SEALED_GATE_KINDS,
            type_error="gate receipt gate must be a GateKind",
            wire_error="gate wire value is non-canonical",
        )
        _require_sealed_enum(
            self.verdict,
            GateVerdict,
            _SEALED_GATE_VERDICTS,
            type_error="gate receipt verdict must be a GateVerdict",
            wire_error="gate verdict wire value is non-canonical",
        )
        _require_git_sha("gate receipt git_sha", self.git_sha)
        _require_sha256("gate receipt evidence_identity_sha256", self.evidence_identity_sha256)
        _require_id("gate receipt actor_id", self.actor_id)
        if self.reason is not None and not _is_exact_type(self.reason, str):
            raise ValueError("gate receipt reason must be text")
        if self.verdict is GateVerdict.NOT_APPLICABLE:
            if self.gate is not GateKind.PHYSICAL:
                raise ValueError("NOT_APPLICABLE is allowed only for a physical gate")
            if not _is_exact_type(self.reason, str) or not self.reason.strip():
                raise ValueError("NOT_APPLICABLE physical gate needs an explicit reason")

    def to_dict(self) -> dict[str, Any]:
        GateReceipt.__post_init__(self)
        return {
            "gate": str.__str__(self.gate),
            "verdict": str.__str__(self.verdict),
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
    candidate_parent_probe: CandidateParentProbe = probe_candidate_parents,
) -> tuple[GateReceipt, GateReceipt]:
    if not _is_exact_type(chain, RegressionChain):
        raise ValueError("chain must be a RegressionChain")
    RegressionChain.__post_init__(chain)
    _require_id("actor_id", actor_id)
    if type(timeout_seconds) is not int or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive integer")
    root = Path(repo_root)
    state = git_probe(root)
    if not _is_exact_type(state, GitState):
        raise ValueError("git probe must return exact GitState")
    GitState.__post_init__(state)
    if state.sha != chain.candidate_git_sha:
        raise ValueError("regression chain exact candidate SHA mismatch")
    if not state.tracked_clean:
        raise ValueError("regression candidate checkout is dirty")
    parents = candidate_parent_probe(root, chain.candidate_git_sha)
    if not _is_exact_type(parents, tuple) or any(
        not _is_exact_type(parent, str) for parent in parents
    ):
        raise ValueError("candidate parent probe must return exact Git SHA tuple")
    if parents != (chain.base_git_sha,):
        raise ValueError(
            "repair candidate must be a direct child of the exact failing Git SHA"
        )

    receipts: list[GateReceipt] = []
    for gate, argv in (
        (GateKind.COMPONENT, chain.component_argv),
        (GateKind.ADVERSARIAL, chain.adversarial_argv),
    ):
        pre_gate_state = git_probe(root)
        if not _is_exact_type(pre_gate_state, GitState):
            raise ValueError("git probe must return exact GitState")
        GitState.__post_init__(pre_gate_state)
        if pre_gate_state.sha != chain.candidate_git_sha:
            raise ValueError(
                f"regression candidate SHA changed before {str.__str__(gate)} gate"
            )
        if not pre_gate_state.tracked_clean:
            raise ValueError(
                f"regression candidate checkout is dirty before {str.__str__(gate)} gate"
            )

        input_envelope = {
            "schema_version": "12-6.aiqa-regression-input.v1",
            "defect_id": chain.defect_id,
            "candidate_identity_sha256": chain.candidate_identity_sha256,
            "candidate_git_sha": chain.candidate_git_sha,
            "gate": str.__str__(gate),
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
        if not _is_exact_type(result, CommandExecution):
            raise ValueError("command runner must return exact CommandExecution")
        CommandExecution.__post_init__(result)

        post_gate_state = git_probe(root)
        if not _is_exact_type(post_gate_state, GitState):
            raise ValueError("git probe must return exact GitState")
        GitState.__post_init__(post_gate_state)
        if post_gate_state.sha != chain.candidate_git_sha:
            raise ValueError(
                f"regression candidate SHA changed during {str.__str__(gate)} gate"
            )
        if not post_gate_state.tracked_clean:
            raise ValueError(
                f"regression candidate checkout became dirty during {str.__str__(gate)} gate"
            )

        evidence = {
            "gate": str.__str__(gate),
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
        if (
            not _is_exact_type(self.decision, str)
            or self.decision not in {"READY_FOR_INDEPENDENT_PROMOTION", "BLOCK"}
        ):
            raise ValueError(
                "promotion decision must be READY_FOR_INDEPENDENT_PROMOTION or BLOCK"
            )
        _require_sha256("candidate_identity_sha256", self.candidate_identity_sha256)
        _require_id("certifier_actor_id", self.certifier_actor_id)
        if (
            not _is_exact_type(self.reasons, tuple)
            or any(not _is_exact_type(reason, str) or not reason for reason in self.reasons)
        ):
            raise ValueError("promotion decision reasons are non-canonical")
        if self.decision == "READY_FOR_INDEPENDENT_PROMOTION" and self.reasons:
            raise ValueError(
                "READY_FOR_INDEPENDENT_PROMOTION cannot contain blocker reasons"
            )
        if self.decision == "BLOCK" and not self.reasons:
            raise ValueError("BLOCK needs at least one reason")

    def to_dict(self) -> dict[str, Any]:
        PromotionDecision.__post_init__(self)
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
    if not _is_exact_type(failure, FailurePacket):
        raise ValueError("failure must be a FailurePacket")
    if not _is_exact_type(candidate, RepairCandidate):
        raise ValueError("candidate must be a RepairCandidate")
    if not _is_exact_type(policy, AIQAPolicy):
        raise ValueError("policy must be an AIQAPolicy")
    if (
        not _is_exact_type(receipts, tuple)
        or any(not _is_exact_type(receipt, GateReceipt) for receipt in receipts)
    ):
        raise ValueError("receipts must contain exact GateReceipt values")
    FailurePacket.__post_init__(failure)
    RepairCandidate.__post_init__(candidate)
    AIQAPolicy.__post_init__(policy)
    for receipt in receipts:
        GateReceipt.__post_init__(receipt)
    _require_id("certifier_actor_id", certifier_actor_id)
    if candidate.defect_id != failure.defect_id:
        raise ValueError("candidate/failure defect mismatch")
    if candidate.failure_packet_identity_sha256 != failure.identity_sha256():
        raise ValueError("candidate/failure packet binding mismatch")
    if candidate.base_git_sha != failure.failing_git_sha:
        raise ValueError("candidate base Git SHA does not match failing Git SHA")

    reasons: list[str] = []
    if (
        policy.require_independent_certifier
        and certifier_actor_id == candidate.proposer_actor_id
    ):
        reasons.append("certifier is the repair proposer")

    by_gate: dict[GateKind, GateReceipt] = {}
    for receipt in receipts:
        if receipt.gate in by_gate:
            raise ValueError(f"duplicate gate receipt: {str.__str__(receipt.gate)}")
        by_gate[receipt.gate] = receipt
        if policy.require_exact_candidate_sha and receipt.git_sha != candidate.candidate_git_sha:
            reasons.append(f"{str.__str__(receipt.gate)} receipt is bound to a different Git SHA")

    for gate in policy.promotion_gate_order:
        if gate not in by_gate:
            reasons.append(f"missing {str.__str__(gate)} gate receipt")

    for gate in (GateKind.COMPONENT, GateKind.ADVERSARIAL, GateKind.SIL):
        receipt = by_gate.get(gate)
        if receipt is not None and receipt.verdict is not GateVerdict.PASS:
            reasons.append(f"{str.__str__(gate)} gate is not PASS")

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
            reasons.append(f"{str.__str__(gate)} evidence is not independent of repair proposer")

    if reasons:
        return PromotionDecision(
            decision="BLOCK",
            candidate_identity_sha256=candidate.identity_sha256(),
            certifier_actor_id=certifier_actor_id,
            reasons=tuple(reasons),
        )
    # Actor IDs are provenance labels, not authenticated worker identities.  The local
    # control plane may prove evidence completeness, but it must not self-authorize the
    # final promotion that Section 4.2 reserves for independent external evidence.
    return PromotionDecision(
        decision="READY_FOR_INDEPENDENT_PROMOTION",
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
    if not _is_exact_type(argv, list) or not all(_is_exact_type(item, str) for item in argv):
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
    if not _is_exact_type(value, dict) or set(value) != {
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
    trusted_receipts: tuple[GateReceipt, ...],
) -> tuple[GateReceipt, ...]:
    """Accept durable receipts only when they exactly match live-verified gate evidence."""

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
    if not _is_exact_type(receipts, list) or not receipts:
        raise ValueError("gate receipt bundle must contain receipts")
    if not _is_exact_type(trusted_receipts, tuple):
        raise ValueError("trusted_receipts must be an immutable tuple")

    trusted_by_gate: dict[GateKind, GateReceipt] = {}
    for trusted in trusted_receipts:
        if not _is_exact_type(trusted, GateReceipt):
            raise ValueError("trusted_receipts must contain GateReceipt values")
        if trusted.gate in trusted_by_gate:
            raise ValueError(f"duplicate trusted gate receipt: {str.__str__(trusted.gate)}")
        trusted_by_gate[trusted.gate] = trusted

    parsed = tuple(_receipt_from_dict(item) for item in receipts)
    seen: set[GateKind] = set()
    for receipt in parsed:
        if receipt.gate in seen:
            raise ValueError(f"duplicate bundled gate receipt: {str.__str__(receipt.gate)}")
        seen.add(receipt.gate)
        trusted = trusted_by_gate.get(receipt.gate)
        if trusted is None:
            raise ValueError(
                f"{str.__str__(receipt.gate)} gate receipt has no live trusted verifier result"
            )
        if receipt != trusted:
            raise ValueError(
                f"{str.__str__(receipt.gate)} gate receipt does not match live trusted evidence"
            )
    return parsed


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


def _materialize_candidate_cli(args: argparse.Namespace) -> int:
    policy = load_ai_qa_policy(args.policy)
    failure = load_failure_packet(args.failure)
    candidate, branch_name = materialize_local_repair_candidate(
        failure,
        repo_root=args.repo_root,
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
                "base_git_sha": candidate.base_git_sha,
                "candidate_git_sha": candidate.candidate_git_sha,
                "branch_name": branch_name,
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
                "component": str.__str__(receipts[0].verdict),
                "adversarial": str.__str__(receipts[1].verdict),
            },
            sort_keys=True,
        )
    )
    return 0 if passed else 1


def verify_candidate_sil_evidence(
    evidence_path: str | Path,
    log_path: str | Path,
    *,
    repo_root: str | Path,
    candidate_git_sha: str,
    expected_environment_receipt: dict[str, Any],
    expected_registry: CapabilityRegistry,
    expected_scenario: SILScenario,
    git_probe: GitProbe = probe_git_state,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    require_exact_clean_git_state(
        root,
        candidate_git_sha,
        git_probe=git_probe,
    )
    evidence = verify_sil_evidence(
        evidence_path,
        log_path,
        expected_package_bytes=build_package_manifest_bytes(root),
        expected_environment_receipt=expected_environment_receipt,
        expected_registry=expected_registry,
        expected_scenario=expected_scenario,
        expected_git_sha=candidate_git_sha,
        require_pass=False,
    )
    require_exact_clean_git_state(
        root,
        candidate_git_sha,
        git_probe=git_probe,
    )
    return evidence


def _sil_receipt_cli(args: argparse.Namespace) -> int:
    candidate = load_repair_candidate(args.candidate)
    root = Path(args.repo_root).resolve()
    environment_receipt = load_sil_environment_receipt(args.environment_receipt)
    registry = load_capability_registry(args.capability_registry)
    scenario = load_sil_scenario(args.scenario)
    evidence = verify_candidate_sil_evidence(
        args.evidence,
        args.log,
        repo_root=root,
        candidate_git_sha=candidate.candidate_git_sha,
        expected_environment_receipt=environment_receipt,
        expected_registry=registry,
        expected_scenario=scenario,
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
        "physical_scope": str.__str__(failure.physical_scope),
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
    root = Path(args.repo_root).resolve()
    environment_receipt = load_sil_environment_receipt(args.environment_receipt)

    chain = build_regression_chain(
        failure,
        candidate,
        adversarial_command=args.adversarial_command,
    )
    component, adversarial = execute_automated_regressions(
        chain,
        repo_root=root,
        actor_id=args.regression_actor_id,
        timeout_seconds=args.timeout_seconds,
    )

    registry = load_capability_registry(args.capability_registry)
    scenario = load_sil_scenario(args.scenario)
    sil_evidence = verify_candidate_sil_evidence(
        args.sil_evidence,
        args.sil_log,
        repo_root=root,
        candidate_git_sha=candidate.candidate_git_sha,
        expected_environment_receipt=environment_receipt,
        expected_registry=registry,
        expected_scenario=scenario,
    )
    sil_receipt = GateReceipt(
        gate=GateKind.SIL,
        verdict=(
            GateVerdict.PASS
            if sil_evidence["verdict"] == "PASS"
            else GateVerdict.FAIL
        ),
        git_sha=candidate.candidate_git_sha,
        evidence_identity_sha256=sil_evidence["evidence_identity_sha256"],
        actor_id=args.sil_actor_id,
        reason=(
            None
            if sil_evidence["verdict"] == "PASS"
            else "SIL evidence verdict is FAIL"
        ),
    )

    trusted: list[GateReceipt] = [component, adversarial, sil_receipt]
    if failure.physical_scope is PhysicalScope.NONE:
        if not args.physical_actor_id or not args.physical_not_applicable_reason:
            raise ValueError(
                "software-only promotion requires physical actor and explicit "
                "NOT_APPLICABLE reason"
            )
        physical_payload = {
            "candidate_identity_sha256": candidate.identity_sha256(),
            "candidate_git_sha": candidate.candidate_git_sha,
            "failure_packet_identity_sha256": failure.identity_sha256(),
            "physical_scope": str.__str__(failure.physical_scope),
            "reason": args.physical_not_applicable_reason,
            "actor_id": args.physical_actor_id,
        }
        trusted.append(
            GateReceipt(
                gate=GateKind.PHYSICAL,
                verdict=GateVerdict.NOT_APPLICABLE,
                git_sha=candidate.candidate_git_sha,
                evidence_identity_sha256=_canonical_sha256(physical_payload),
                actor_id=args.physical_actor_id,
                reason=args.physical_not_applicable_reason,
            )
        )
    else:
        raise ValueError(
            "required physical PASS has no Section-4 trusted verifier; "
            "promotion stays blocked until an authoritative physical verifier is integrated"
        )

    trusted_receipts = tuple(trusted)
    durable_receipts: list[GateReceipt] = []
    for bundle in args.receipt_bundle:
        durable_receipts.extend(
            load_gate_receipt_bundle(
                bundle,
                expected_candidate_identity_sha256=candidate.identity_sha256(),
                trusted_receipts=trusted_receipts,
            )
        )

    expected_by_gate = {receipt.gate: receipt for receipt in trusted_receipts}
    durable_by_gate: dict[GateKind, GateReceipt] = {}
    for receipt in durable_receipts:
        if receipt.gate in durable_by_gate:
            raise ValueError(f"duplicate durable gate receipt: {str.__str__(receipt.gate)}")
        durable_by_gate[receipt.gate] = receipt
    if durable_by_gate != expected_by_gate:
        raise ValueError("durable receipt bundles do not cover the exact live-verified gate set")

    decision = evaluate_promotion(
        failure,
        candidate,
        tuple(durable_by_gate[gate] for gate in policy.promotion_gate_order),
        certifier_actor_id=args.certifier_actor_id,
        policy=policy,
    )
    payload = decision.to_dict()
    payload["decision_identity_sha256"] = decision.identity_sha256()
    _write_json(args.output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0 if decision.decision == "READY_FOR_INDEPENDENT_PROMOTION" else 1


def _sil_failure_cli(args: argparse.Namespace) -> int:
    policy = load_ai_qa_policy(args.policy)
    root = Path(args.repo_root).resolve()
    failing_state = probe_git_state(root)
    if not failing_state.tracked_clean:
        raise ValueError("SIL failure ingestion checkout is dirty")
    registry = load_capability_registry(args.capability_registry)
    scenario = load_sil_scenario(args.scenario)
    environment_receipt = load_sil_environment_receipt(args.environment_receipt)
    packet = failure_packet_from_sil(
        args.evidence,
        args.log,
        defect_id=args.defect_id,
        policy=policy,
        expected_package_bytes=build_package_manifest_bytes(root),
        expected_environment_receipt=environment_receipt,
        expected_registry=registry,
        expected_scenario=scenario,
        expected_git_sha=failing_state.sha,
        physical_scope=PhysicalScope(args.physical_scope),
        physical_gate_id=args.physical_gate_id,
    )
    payload = packet.to_dict()
    payload["failure_packet_identity_sha256"] = packet.identity_sha256()
    _write_json(args.output, payload)
    print(json.dumps({"defect_id": packet.defect_id, "failure_class": str.__str__(packet.failure_class)}))
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
    print(json.dumps({"defect_id": packet.defect_id, "failure_class": str.__str__(packet.failure_class)}))
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
    sil.add_argument("--environment-receipt", required=True)
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

    materialize = subparsers.add_parser("materialize-candidate")
    materialize.add_argument("--failure", required=True)
    materialize.add_argument("--policy", required=True)
    materialize.add_argument("--repo-root", required=True)
    materialize.add_argument("--patch-file", required=True)
    materialize.add_argument("--proposer-actor-id", required=True)
    materialize.add_argument("--output", required=True)
    materialize.set_defaults(func=_materialize_candidate_cli)

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
    sil_receipt.add_argument("--environment-receipt", required=True)
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
    assess.add_argument("--repo-root", required=True)
    assess.add_argument("--capability-registry", required=True)
    assess.add_argument("--scenario", required=True)
    assess.add_argument("--environment-receipt", required=True)
    assess.add_argument("--sil-evidence", required=True)
    assess.add_argument("--sil-log", required=True)
    assess.add_argument("--sil-actor-id", required=True)
    assess.add_argument("--adversarial-command", required=True)
    assess.add_argument("--regression-actor-id", required=True)
    assess.add_argument("--timeout-seconds", type=int, default=300)
    assess.add_argument("--physical-actor-id")
    assess.add_argument("--physical-not-applicable-reason")
    assess.add_argument("--receipt-bundle", action="append", required=True)
    assess.add_argument("--certifier-actor-id", required=True)
    assess.add_argument("--output", required=True)
    assess.set_defaults(func=_assess_cli)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())