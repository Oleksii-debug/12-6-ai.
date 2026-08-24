"""Fail-closed release attestation for exact 12-6 AI stage candidates."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .dependency_lock import (
    SUPPORTED_PROFILES,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    validate_lock_index,
)
from .manifest import (
    AuditEvidence,
    AuditVerdict,
    CandidateStatus,
    CIEvidence,
    ComponentDisposition,
    ComponentRef,
    ReleaseArtifactEvidence,
    StageCandidateManifest,
    validate_repository_evidence,
)

SCHEMA_VERSION = "12-6.release-attestation.v1"
CANONICAL_REPOSITORY = "Oleksii-debug/12-6-ai."
_REQUIRED_CHECKPOINT_KINDS = frozenset({"checkpoint_manifest", "model_weights"})
_REQUIRED_SUPPLY_CHAIN_KINDS = frozenset({"sbom", "dependency_report"})
_GIT_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ReleaseAttestationError(ValueError):
    """Raised when release evidence is stale, incomplete, ambiguous, or tampered."""


def _require_git_sha(value: str, field_name: str) -> None:
    if _GIT_SHA_RE.fullmatch(value) is None:
        raise ReleaseAttestationError(
            f"{field_name} must be an exact lowercase 40- or 64-hex git SHA"
        )


def _require_sha256(value: str, field_name: str) -> None:
    if _SHA256_RE.fullmatch(value) is None:
        raise ReleaseAttestationError(f"{field_name} must be a lowercase 64-hex SHA-256")


def _require_evidence_ref(value: str, field_name: str) -> None:
    if not value.strip():
        raise ReleaseAttestationError(f"{field_name} must be non-empty")


def _parse_aware_timestamp(value: str, field_name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ReleaseAttestationError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ReleaseAttestationError(f"{field_name} must include a timezone offset")
    return parsed


def _safe_relative_path(value: str, field_name: str) -> None:
    candidate = Path(value)
    if not value.strip() or candidate.is_absolute() or ".." in candidate.parts:
        raise ReleaseAttestationError(f"{field_name} must be a safe relative path")


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    """A materialized artifact hash bound to the candidate that produced it."""

    kind: str
    path: str
    sha256: str
    producer_sha: str
    evidence_ref: str

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ReleaseAttestationError("artifact kind must be non-empty")
        _safe_relative_path(self.path, "artifact path")
        _require_sha256(self.sha256, "artifact sha256")
        _require_git_sha(self.producer_sha, "artifact producer_sha")
        _require_evidence_ref(self.evidence_ref, "artifact evidence_ref")


@dataclass(frozen=True, slots=True)
class DependencyLockBinding:
    """Physical and semantic identity for the complete dependency-lock index."""

    path: str
    file_sha256: str
    index_sha256: str
    producer_sha: str
    evidence_ref: str

    def __post_init__(self) -> None:
        _safe_relative_path(self.path, "dependency lock path")
        _require_sha256(self.file_sha256, "dependency lock file_sha256")
        _require_sha256(self.index_sha256, "dependency lock index_sha256")
        _require_git_sha(self.producer_sha, "dependency lock producer_sha")
        _require_evidence_ref(self.evidence_ref, "dependency lock evidence_ref")


@dataclass(frozen=True, slots=True)
class CandidateCIEvidence:
    """Completed combined CI for the exact candidate head."""

    run_id: int
    head_sha: str
    conclusion: str
    completed_at_utc: str
    evidence_ref: str

    def __post_init__(self) -> None:
        if self.run_id <= 0:
            raise ReleaseAttestationError("candidate CI run_id must be positive")
        _require_git_sha(self.head_sha, "candidate CI head_sha")
        if not self.conclusion.strip():
            raise ReleaseAttestationError("candidate CI conclusion must be non-empty")
        _parse_aware_timestamp(self.completed_at_utc, "candidate CI completed_at_utc")
        _require_evidence_ref(self.evidence_ref, "candidate CI evidence_ref")

    @property
    def passes(self) -> bool:
        return self.conclusion.lower() == "success"


@dataclass(frozen=True, slots=True)
class EnvironmentArtifactEvidence:
    """Retained locked-environment workflow artifact for one required profile."""

    profile_id: str
    run_id: int
    source_sha: str
    artifact_id: int
    archive_sha256: str
    evidence_ref: str

    def __post_init__(self) -> None:
        if self.profile_id not in SUPPORTED_PROFILES:
            raise ReleaseAttestationError(f"unsupported environment profile: {self.profile_id}")
        if self.run_id <= 0 or self.artifact_id <= 0:
            raise ReleaseAttestationError("environment run_id/artifact_id must be positive")
        _require_git_sha(self.source_sha, "environment source_sha")
        _require_sha256(self.archive_sha256, "environment archive_sha256")
        _require_evidence_ref(self.evidence_ref, "environment evidence_ref")


@dataclass(frozen=True, slots=True)
class ReleaseAttestation:
    """Exact evidence envelope for CANDIDATE/AUDITED_CANDIDATE/STABLE transitions."""

    repository: str
    stage: str
    status: CandidateStatus
    candidate_sha: str | None
    candidate_manifest: ArtifactBinding | None
    dependency_lock: DependencyLockBinding | None
    candidate_ci: CandidateCIEvidence | None
    environment_evidence: tuple[EnvironmentArtifactEvidence, ...]
    checkpoint_artifacts: tuple[ArtifactBinding, ...]
    supply_chain_artifacts: tuple[ArtifactBinding, ...]
    release_artifact: ArtifactBinding | None
    promotion_authority_ref: str | None
    attestation_sha256: str

    def __post_init__(self) -> None:
        if self.repository != CANONICAL_REPOSITORY:
            raise ReleaseAttestationError(
                f"repository must be exact physical identity {CANONICAL_REPOSITORY!r}"
            )
        if self.stage.upper() != "S0":
            raise ReleaseAttestationError("release attestation v1 is scoped to S0")
        if self.candidate_sha is not None:
            _require_git_sha(self.candidate_sha, "candidate_sha")
        _require_sha256(self.attestation_sha256, "attestation_sha256")

        self._require_unique_kinds(self.checkpoint_artifacts, "checkpoint")
        self._require_unique_kinds(self.supply_chain_artifacts, "supply-chain")
        profile_ids = [item.profile_id for item in self.environment_evidence]
        if len(profile_ids) != len(set(profile_ids)):
            raise ReleaseAttestationError("environment evidence contains duplicate profiles")
        artifact_ids = [item.artifact_id for item in self.environment_evidence]
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ReleaseAttestationError("environment evidence contains duplicate artifact IDs")

        gated = self.status in {
            CandidateStatus.CANDIDATE,
            CandidateStatus.AUDITED_CANDIDATE,
            CandidateStatus.STABLE,
        }
        if gated:
            self._validate_gated_shape()
        if self.status is CandidateStatus.STABLE:
            if self.release_artifact is None:
                raise ReleaseAttestationError("STABLE requires release artifact evidence")
            if self.promotion_authority_ref is None:
                raise ReleaseAttestationError("STABLE requires external promotion_authority_ref")
            _require_evidence_ref(self.promotion_authority_ref, "promotion_authority_ref")

    @staticmethod
    def _require_unique_kinds(items: tuple[ArtifactBinding, ...], label: str) -> None:
        kinds = [item.kind for item in items]
        if len(kinds) != len(set(kinds)):
            raise ReleaseAttestationError(f"{label} artifacts contain duplicate kind values")

    def _validate_gated_shape(self) -> None:
        if self.candidate_sha is None:
            raise ReleaseAttestationError("candidate transition requires exact candidate_sha")
        if self.candidate_manifest is None:
            raise ReleaseAttestationError("candidate transition requires candidate manifest hash")
        if self.dependency_lock is None:
            raise ReleaseAttestationError("candidate transition requires dependency lock evidence")
        if self.candidate_ci is None:
            raise ReleaseAttestationError(
                "candidate transition requires exact combined CI evidence"
            )
        if self.candidate_ci.head_sha != self.candidate_sha:
            raise ReleaseAttestationError("candidate CI is stale for candidate_sha")
        if not self.candidate_ci.passes:
            raise ReleaseAttestationError("candidate transition requires successful combined CI")

        for label, producer_sha in (
            ("candidate manifest", self.candidate_manifest.producer_sha),
            ("dependency lock", self.dependency_lock.producer_sha),
        ):
            if producer_sha != self.candidate_sha:
                raise ReleaseAttestationError(f"{label} is stale for candidate_sha")

        profiles = frozenset(item.profile_id for item in self.environment_evidence)
        if profiles != SUPPORTED_PROFILES:
            missing = sorted(SUPPORTED_PROFILES - profiles)
            extra = sorted(profiles - SUPPORTED_PROFILES)
            raise ReleaseAttestationError(
                f"candidate environment profile set mismatch; missing={missing}, extra={extra}"
            )
        for item in self.environment_evidence:
            if item.run_id != self.candidate_ci.run_id or item.source_sha != self.candidate_sha:
                raise ReleaseAttestationError(
                    f"environment evidence for {item.profile_id} is not bound to candidate CI/head"
                )

        checkpoint_kinds = frozenset(item.kind for item in self.checkpoint_artifacts)
        missing_checkpoint = _REQUIRED_CHECKPOINT_KINDS - checkpoint_kinds
        if missing_checkpoint:
            raise ReleaseAttestationError(
                "candidate transition missing checkpoint evidence: "
                + ", ".join(sorted(missing_checkpoint))
            )
        supply_kinds = frozenset(item.kind for item in self.supply_chain_artifacts)
        missing_supply = _REQUIRED_SUPPLY_CHAIN_KINDS - supply_kinds
        if missing_supply:
            raise ReleaseAttestationError(
                "candidate transition missing supply-chain evidence: "
                + ", ".join(sorted(missing_supply))
            )

        for item in (
            *self.checkpoint_artifacts,
            *self.supply_chain_artifacts,
            *((self.release_artifact,) if self.release_artifact is not None else ()),
        ):
            if item.producer_sha != self.candidate_sha:
                raise ReleaseAttestationError(
                    f"artifact {item.kind!r} is stale for candidate_sha"
                )


def _artifact_payload(value: ArtifactBinding | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "kind": value.kind,
        "path": value.path,
        "sha256": value.sha256,
        "producer_sha": value.producer_sha,
        "evidence_ref": value.evidence_ref,
    }


def _dependency_lock_payload(value: DependencyLockBinding | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "path": value.path,
        "file_sha256": value.file_sha256,
        "index_sha256": value.index_sha256,
        "producer_sha": value.producer_sha,
        "evidence_ref": value.evidence_ref,
    }


def _candidate_ci_payload(value: CandidateCIEvidence | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {
        "run_id": value.run_id,
        "head_sha": value.head_sha,
        "conclusion": value.conclusion,
        "completed_at_utc": value.completed_at_utc,
        "evidence_ref": value.evidence_ref,
    }


def _environment_payload(value: EnvironmentArtifactEvidence) -> dict[str, Any]:
    return {
        "profile_id": value.profile_id,
        "run_id": value.run_id,
        "source_sha": value.source_sha,
        "artifact_id": value.artifact_id,
        "archive_sha256": value.archive_sha256,
        "evidence_ref": value.evidence_ref,
    }


def attestation_payload(attestation: ReleaseAttestation) -> dict[str, Any]:
    """Return the canonical payload covered by attestation_sha256."""

    return {
        "schema_version": SCHEMA_VERSION,
        "repository": attestation.repository,
        "stage": attestation.stage,
        "status": attestation.status.value,
        "candidate_sha": attestation.candidate_sha,
        "candidate_manifest": _artifact_payload(attestation.candidate_manifest),
        "dependency_lock": _dependency_lock_payload(attestation.dependency_lock),
        "candidate_ci": _candidate_ci_payload(attestation.candidate_ci),
        "environment_evidence": [
            _environment_payload(item) for item in attestation.environment_evidence
        ],
        "checkpoint_artifacts": [
            _artifact_payload(item) for item in attestation.checkpoint_artifacts
        ],
        "supply_chain_artifacts": [
            _artifact_payload(item) for item in attestation.supply_chain_artifacts
        ],
        "release_artifact": _artifact_payload(attestation.release_artifact),
        "promotion_authority_ref": attestation.promotion_authority_ref,
    }


def compute_attestation_sha256(payload: dict[str, Any]) -> str:
    """Hash a raw attestation payload that does not contain attestation_sha256."""

    value = dict(payload)
    value.pop("attestation_sha256", None)
    return sha256_bytes(canonical_json_bytes(value))


def _artifact_from_raw(value: Any, field_name: str) -> ArtifactBinding | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be null or an object")
    return ArtifactBinding(
        kind=value["kind"],
        path=value["path"],
        sha256=value["sha256"],
        producer_sha=value["producer_sha"],
        evidence_ref=value["evidence_ref"],
    )


def _dependency_lock_from_raw(value: Any) -> DependencyLockBinding | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("dependency_lock must be null or an object")
    return DependencyLockBinding(
        path=value["path"],
        file_sha256=value["file_sha256"],
        index_sha256=value["index_sha256"],
        producer_sha=value["producer_sha"],
        evidence_ref=value["evidence_ref"],
    )


def _candidate_ci_from_raw(value: Any) -> CandidateCIEvidence | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("candidate_ci must be null or an object")
    return CandidateCIEvidence(
        run_id=value["run_id"],
        head_sha=value["head_sha"],
        conclusion=value["conclusion"],
        completed_at_utc=value["completed_at_utc"],
        evidence_ref=value["evidence_ref"],
    )


def _environment_from_raw(value: Any) -> EnvironmentArtifactEvidence:
    if not isinstance(value, dict):
        raise TypeError("environment_evidence items must be objects")
    return EnvironmentArtifactEvidence(
        profile_id=value["profile_id"],
        run_id=value["run_id"],
        source_sha=value["source_sha"],
        artifact_id=value["artifact_id"],
        archive_sha256=value["archive_sha256"],
        evidence_ref=value["evidence_ref"],
    )


def load_release_attestation(path: str | Path) -> ReleaseAttestation:
    """Load an attestation and reject any self-hash or schema tampering."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("release attestation must contain a JSON object")
    if raw.get("schema_version") != SCHEMA_VERSION:
        raise ReleaseAttestationError("unsupported release attestation schema")
    claimed = raw.get("attestation_sha256")
    if not isinstance(claimed, str):
        raise ReleaseAttestationError("attestation_sha256 is required")
    _require_sha256(claimed, "attestation_sha256")
    if compute_attestation_sha256(raw) != claimed:
        raise ReleaseAttestationError("release attestation self-hash mismatch")

    environment_raw = raw.get("environment_evidence", [])
    checkpoint_raw = raw.get("checkpoint_artifacts", [])
    supply_raw = raw.get("supply_chain_artifacts", [])
    if not isinstance(environment_raw, list):
        raise TypeError("environment_evidence must be an array")
    if not isinstance(checkpoint_raw, list):
        raise TypeError("checkpoint_artifacts must be an array")
    if not isinstance(supply_raw, list):
        raise TypeError("supply_chain_artifacts must be an array")

    return ReleaseAttestation(
        repository=raw["repository"],
        stage=raw["stage"],
        status=CandidateStatus(raw["status"]),
        candidate_sha=raw.get("candidate_sha"),
        candidate_manifest=_artifact_from_raw(raw.get("candidate_manifest"), "candidate_manifest"),
        dependency_lock=_dependency_lock_from_raw(raw.get("dependency_lock")),
        candidate_ci=_candidate_ci_from_raw(raw.get("candidate_ci")),
        environment_evidence=tuple(_environment_from_raw(item) for item in environment_raw),
        checkpoint_artifacts=tuple(
            _artifact_from_raw(item, "checkpoint_artifacts item") for item in checkpoint_raw
        ),
        supply_chain_artifacts=tuple(
            _artifact_from_raw(item, "supply_chain_artifacts item") for item in supply_raw
        ),
        release_artifact=_artifact_from_raw(raw.get("release_artifact"), "release_artifact"),
        promotion_authority_ref=raw.get("promotion_authority_ref"),
        attestation_sha256=claimed,
    )


def _audit_evidence(value: Any, field_name: str) -> AuditEvidence | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be null or an object")
    return AuditEvidence(
        auditor_id=value["auditor_id"],
        verdict=AuditVerdict(value["verdict"]),
        candidate_sha=value["candidate_sha"],
        cutoff_utc=value["cutoff_utc"],
        evidence_ref=value["evidence_ref"],
    )


def _component_ci(value: Any) -> CIEvidence | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("component ci_evidence must be null or an object")
    return CIEvidence(
        run_id=value["run_id"],
        head_sha=value["head_sha"],
        conclusion=value["conclusion"],
        evidence_ref=value["evidence_ref"],
    )


def _release_artifact_evidence(value: Any) -> ReleaseArtifactEvidence | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise TypeError("stage release_artifact must be null or an object")
    return ReleaseArtifactEvidence(
        path=value["path"],
        sha256=value["sha256"],
        evidence_ref=value["evidence_ref"],
    )


def load_stage_candidate_manifest(path: str | Path) -> StageCandidateManifest:
    """Load the existing D10 stage-candidate contract for cross-binding."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError("stage candidate manifest must contain a JSON object")
    if raw.get("format_version") != "stage-candidate-manifest-v1":
        raise ReleaseAttestationError("unsupported stage candidate manifest format")
    if not isinstance(raw.get("base_lineage"), bool):
        raise TypeError("stage candidate base_lineage must be a JSON boolean")
    components_raw = raw.get("components", [])
    if not isinstance(components_raw, list):
        raise TypeError("stage candidate components must be an array")

    components = tuple(
        ComponentRef(
            lane=item["lane"],
            source_sha=item["source_sha"],
            disposition=ComponentDisposition(item["disposition"]),
            component_kind=item["component_kind"],
            pr_number=item.get("pr_number"),
            ci_evidence=_component_ci(item.get("ci_evidence")),
            artifact_path=item.get("artifact_path"),
            artifact_sha256=item.get("artifact_sha256"),
            artifact_evidence_ref=item.get("artifact_evidence_ref"),
            contains_behavioral_weights=item.get("contains_behavioral_weights"),
            contains_foreign_pretrained_weights=item.get("contains_foreign_pretrained_weights"),
            notes=item.get("notes", ""),
        )
        for item in components_raw
    )
    required_lanes = raw.get(
        "required_lanes",
        ["D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08"],
    )
    if not isinstance(required_lanes, list):
        raise TypeError("stage candidate required_lanes must be an array")
    return StageCandidateManifest.compose(
        stage=raw["stage"],
        integration_anchor_sha=raw["integration_anchor_sha"],
        status=CandidateStatus(raw["status"]),
        base_lineage=raw["base_lineage"],
        components=components,
        candidate_sha=raw.get("candidate_sha"),
        audit_a=_audit_evidence(raw.get("audit_a"), "audit_a"),
        audit_b=_audit_evidence(raw.get("audit_b"), "audit_b"),
        release_artifact=_release_artifact_evidence(raw.get("release_artifact")),
        required_lanes=frozenset(required_lanes),
    )


def _materialized_path(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    if not candidate.is_relative_to(root) or not candidate.is_file():
        raise ReleaseAttestationError(f"materialized artifact is missing: {relative_path}")
    return candidate


def _verify_artifact(root: Path, artifact: ArtifactBinding) -> Path:
    path = _materialized_path(root, artifact.path)
    if sha256_file(path) != artifact.sha256:
        raise ReleaseAttestationError(f"artifact SHA-256 mismatch: {artifact.path}")
    return path


def _verify_dependency_lock(root: Path, binding: DependencyLockBinding) -> None:
    path = _materialized_path(root, binding.path)
    if sha256_file(path) != binding.file_sha256:
        raise ReleaseAttestationError("dependency lock physical SHA-256 mismatch")
    index = validate_lock_index(root=root, index_path=binding.path)
    if index.get("index_sha256") != binding.index_sha256:
        raise ReleaseAttestationError("dependency lock semantic index SHA-256 mismatch")


def _verify_attestation_self_hash(attestation: ReleaseAttestation) -> None:
    expected = sha256_bytes(canonical_json_bytes(attestation_payload(attestation)))
    if expected != attestation.attestation_sha256:
        raise ReleaseAttestationError("release attestation object self-hash mismatch")


def validate_audit_freshness(
    stage_manifest: StageCandidateManifest,
    candidate_ci: CandidateCIEvidence,
) -> None:
    ci_completed = _parse_aware_timestamp(
        candidate_ci.completed_at_utc,
        "candidate CI completed_at_utc",
    )
    for label, audit in (("AUDIT-A", stage_manifest.audit_a), ("AUDIT-B", stage_manifest.audit_b)):
        if audit is None:
            continue
        cutoff = _parse_aware_timestamp(audit.cutoff_utc, f"{label} cutoff_utc")
        if cutoff < ci_completed:
            raise ReleaseAttestationError(
                f"{label} evidence predates exact candidate combined CI completion"
            )


def validate_release_attestation(
    attestation: ReleaseAttestation,
    *,
    repo_root: str | Path = ".",
    artifact_root: str | Path | None = None,
) -> StageCandidateManifest | None:
    """Validate exact candidate, lock, checkpoint, supply-chain, audit, and release evidence.

    ``artifact_root`` supports CI/release workflows that materialize evidence outside the Git
    checkout. Git ancestry is always checked against ``repo_root``. No verdict or promotion is
    created by this function; it only rejects invalid evidence.
    """

    _verify_attestation_self_hash(attestation)
    repository_root = Path(repo_root).resolve()
    material_root = Path(artifact_root).resolve() if artifact_root is not None else repository_root

    candidate_manifest_path: Path | None = None
    if attestation.candidate_manifest is not None:
        candidate_manifest_path = _verify_artifact(material_root, attestation.candidate_manifest)
    if attestation.dependency_lock is not None:
        _verify_dependency_lock(material_root, attestation.dependency_lock)
    for artifact in (
        *attestation.checkpoint_artifacts,
        *attestation.supply_chain_artifacts,
        *((attestation.release_artifact,) if attestation.release_artifact is not None else ()),
    ):
        _verify_artifact(material_root, artifact)

    if attestation.status is CandidateStatus.EXPERIMENTAL:
        return (
            load_stage_candidate_manifest(candidate_manifest_path)
            if candidate_manifest_path is not None
            else None
        )

    if candidate_manifest_path is None or attestation.candidate_ci is None:
        raise ReleaseAttestationError("gated release evidence is incomplete")
    stage_manifest = load_stage_candidate_manifest(candidate_manifest_path)
    if stage_manifest.stage.upper() != attestation.stage.upper():
        raise ReleaseAttestationError("candidate manifest stage differs from release attestation")
    if stage_manifest.status is not attestation.status:
        raise ReleaseAttestationError("candidate manifest status differs from release attestation")
    if stage_manifest.candidate_sha != attestation.candidate_sha:
        raise ReleaseAttestationError("candidate manifest SHA differs from release attestation")

    validate_repository_evidence(stage_manifest, repository_root)
    validate_audit_freshness(stage_manifest, attestation.candidate_ci)

    if attestation.status in {CandidateStatus.AUDITED_CANDIDATE, CandidateStatus.STABLE}:
        if not stage_manifest.audits_pass():
            raise ReleaseAttestationError("audited transition requires independent passing audits")

    if attestation.status is CandidateStatus.STABLE:
        if stage_manifest.release_artifact is None or attestation.release_artifact is None:
            raise ReleaseAttestationError("STABLE release evidence is incomplete")
        if (
            stage_manifest.release_artifact.path != attestation.release_artifact.path
            or stage_manifest.release_artifact.sha256 != attestation.release_artifact.sha256
            or stage_manifest.release_artifact.evidence_ref
            != attestation.release_artifact.evidence_ref
        ):
            raise ReleaseAttestationError(
                "STABLE release artifact differs between candidate manifest and attestation"
            )

    return stage_manifest
