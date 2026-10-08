"""Plan-1 versioned packaging of accepted Migration Contract Baseline v1.

This is a facade and inert exchange envelope, not a new model, registry,
checkpoint controller, evidence authority, or scheduler.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from twelve_six.artifact_identity import (
    CANONICAL_ARTIFACT_KINDS,
    ArtifactKind,
    ArtifactManifest,
    ArtifactRef,
    GenerationIdentityManifest,
    ParentBinding,
    bind_artifact,
    build_generation_identity_manifest,
    parse_generation_identity_manifest,
    verify_parent_bindings,
)
from twelve_six.system_architecture import (
    InterfaceContract,
    SystemArchitectureManifest,
    SystemPlane,
    TypedBoundary,
    canonical_runtime_shell_v1,
    canonical_system_architecture_v1,
)

BASELINE_SCHEMA_VERSION = 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,63}$")
_NAME = re.compile(r"^[a-z][a-z0-9_.]{0,127}$")
_STATES = frozenset({"CREATED", "VERIFIED", "REJECTED"})


class ContractCompatibilityError(ValueError):
    """An exchange contract is missing an exact versioned trust binding."""


def _digest(value: object, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ContractCompatibilityError(f"{name} must be lowercase SHA-256")
    return value


def _version(value: object, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ContractCompatibilityError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class ContractEvidenceRef:
    """Opaque evidence identity; never an assertion that a gate passed."""

    sha256: str
    source_commit_sha: str

    def __post_init__(self) -> None:
        _digest(self.sha256, "evidence")
        if type(self.source_commit_sha) is not str or not _GIT_SHA.fullmatch(
            self.source_commit_sha
        ):
            raise ContractCompatibilityError("source commit must be a full Git SHA")

    def to_dict(self) -> dict[str, str]:
        ContractEvidenceRef.__post_init__(self)
        return {"sha256": self.sha256, "source_commit_sha": self.source_commit_sha}


@dataclass(frozen=True, slots=True)
class ContractSignal:
    """Inert errors/lifecycle interchange; no action or state promotion."""

    schema_version: int
    event: str
    code: str
    artifact: ArtifactRef
    evidence: ContractEvidenceRef

    def __post_init__(self) -> None:
        if _version(self.schema_version, "schema_version") != BASELINE_SCHEMA_VERSION:
            raise ContractCompatibilityError("unsupported contract signal schema")
        if type(self.event) is not str or self.event not in _STATES:
            raise ContractCompatibilityError("unsupported lifecycle event")
        if type(self.code) is not str or _CODE.fullmatch(self.code) is None:
            raise ContractCompatibilityError("invalid error/status code")
        if type(self.artifact) is not ArtifactRef:
            raise ContractCompatibilityError("artifact must be a canonical ArtifactRef")
        self.artifact.to_dict()
        if type(self.evidence) is not ContractEvidenceRef:
            raise ContractCompatibilityError("evidence must be a ContractEvidenceRef")
        self.evidence.to_dict()

    def to_dict(self) -> dict[str, Any]:
        ContractSignal.__post_init__(self)
        return {
            "schema_version": self.schema_version,
            "event": self.event,
            "code": self.code,
            "artifact": self.artifact.to_dict(),
            "evidence": self.evidence.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: object) -> ContractSignal:
        if cls is not ContractSignal or type(value) is not dict or set(value) != {
            "schema_version", "event", "code", "artifact", "evidence"
        }:
            raise ContractCompatibilityError("contract signal fields mismatch")
        evidence = value["evidence"]
        if type(evidence) is not dict or set(evidence) != {
            "sha256", "source_commit_sha"
        }:
            raise ContractCompatibilityError("contract evidence fields mismatch")
        return cls(
            schema_version=value["schema_version"],
            event=value["event"],
            code=value["code"],
            artifact=ArtifactRef.from_dict(value["artifact"]),
            evidence=ContractEvidenceRef(**evidence),
        )


@dataclass(frozen=True, slots=True)
class ContractEvolutionReceipt:
    """Compatibility review input only; cannot authorize adoption or reopening."""

    contract_name: str
    source_version: int
    target_version: int
    source_semantics_sha256: str
    target_semantics_sha256: str
    migration_fixture_sha256: str | None = None
    affected_plans: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if type(self.contract_name) is not str or _NAME.fullmatch(
            self.contract_name
        ) is None:
            raise ContractCompatibilityError("invalid contract name")
        _version(self.source_version, "source_version")
        _version(self.target_version, "target_version")
        _digest(self.source_semantics_sha256, "source semantics")
        _digest(self.target_semantics_sha256, "target semantics")
        if self.target_version < self.source_version:
            raise ContractCompatibilityError("schema downgrade denied")
        if type(self.affected_plans) is not tuple or any(
            type(p) is not int or p < 1 or p > 10 for p in self.affected_plans
        ):
            raise ContractCompatibilityError("affected plans must be exact IDs")
        if len(set(self.affected_plans)) != len(self.affected_plans):
            raise ContractCompatibilityError("duplicate affected plan")
        if tuple(sorted(self.affected_plans)) != self.affected_plans:
            raise ContractCompatibilityError("affected plans must be sorted")
        if self.target_version == self.source_version:
            if self.source_semantics_sha256 != self.target_semantics_sha256:
                raise ContractCompatibilityError("semantic drift requires a new version")
            if self.migration_fixture_sha256 is not None or self.affected_plans:
                raise ContractCompatibilityError("unnecessary migration authority")
        else:
            _digest(self.migration_fixture_sha256, "migration fixture")
            if not self.affected_plans:
                raise ContractCompatibilityError("breaking version needs impact analysis")

    @property
    def needs_requalification(self) -> bool:
        ContractEvolutionReceipt.__post_init__(self)
        return self.target_version != self.source_version


__all__ = (
    "BASELINE_SCHEMA_VERSION",
    "CANONICAL_ARTIFACT_KINDS",
    "ArtifactKind",
    "ArtifactManifest",
    "ArtifactRef",
    "ContractCompatibilityError",
    "ContractEvidenceRef",
    "ContractEvolutionReceipt",
    "ContractSignal",
    "GenerationIdentityManifest",
    "InterfaceContract",
    "ParentBinding",
    "SystemArchitectureManifest",
    "SystemPlane",
    "TypedBoundary",
    "bind_artifact",
    "build_generation_identity_manifest",
    "canonical_runtime_shell_v1",
    "canonical_system_architecture_v1",
    "parse_generation_identity_manifest",
    "verify_parent_bindings",
)
