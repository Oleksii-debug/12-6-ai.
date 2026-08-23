"""Exact provenance contracts for selective stage composition."""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

_GIT_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
S0_REQUIRED_LANES = frozenset(
    {"D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08"}
)
_PASSING_AUDITS = frozenset({"PASS", "PASS_WITH_NOTES"})
_BEHAVIORAL_WEIGHT_KINDS = frozenset(
    {
        "behavioral_weights",
        "alignment_weights",
        "instruction_weights",
        "preference_weights",
        "rl_weights",
        "specialization_weights",
        "refusal_policy_weights",
        "assistant_personality_weights",
    }
)


class ComponentDisposition(StrEnum):
    ACCEPTED = "accepted"
    HELD = "held"
    REJECTED = "rejected"


class CandidateStatus(StrEnum):
    EXPERIMENTAL = "experimental"
    CANDIDATE = "candidate"
    AUDITED_CANDIDATE = "audited_candidate"
    STABLE = "stable"


class AuditVerdict(StrEnum):
    NOT_RUN = "NOT_RUN"
    PASS = "PASS"
    PASS_WITH_NOTES = "PASS_WITH_NOTES"
    CHANGES_REQUIRED = "CHANGES_REQUIRED"
    BLOCKED = "BLOCKED"


def _require_evidence_ref(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _require_aware_timestamp(value: str, field_name: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone offset")


@dataclass(frozen=True, slots=True)
class CIEvidence:
    """One exact workflow result bound to the component head that it tested."""

    run_id: int
    head_sha: str
    conclusion: str
    evidence_ref: str

    def __post_init__(self) -> None:
        if self.run_id <= 0:
            raise ValueError("CI run_id must be positive")
        if not _GIT_SHA_RE.fullmatch(self.head_sha):
            raise ValueError("CI head_sha must be an exact git SHA")
        if not self.conclusion.strip():
            raise ValueError("CI conclusion must be non-empty")
        _require_evidence_ref(self.evidence_ref, "CI evidence_ref")

    @property
    def passes(self) -> bool:
        return self.conclusion.lower() == "success"


@dataclass(frozen=True, slots=True)
class AuditEvidence:
    """Durable independent-audit verdict bound to one exact candidate SHA."""

    auditor_id: str
    verdict: AuditVerdict
    candidate_sha: str
    cutoff_utc: str
    evidence_ref: str

    def __post_init__(self) -> None:
        if not self.auditor_id.strip():
            raise ValueError("auditor_id must be non-empty")
        if not _GIT_SHA_RE.fullmatch(self.candidate_sha):
            raise ValueError("audit candidate_sha must be an exact git SHA")
        _require_aware_timestamp(self.cutoff_utc, "audit cutoff_utc")
        _require_evidence_ref(self.evidence_ref, "audit evidence_ref")

    @property
    def passes(self) -> bool:
        return self.verdict.value in _PASSING_AUDITS


@dataclass(frozen=True, slots=True)
class ReleaseArtifactEvidence:
    """Release payload evidence that can be re-hashed from a controlled checkout."""

    path: str
    sha256: str
    evidence_ref: str

    def __post_init__(self) -> None:
        candidate = Path(self.path)
        if not self.path.strip() or candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("release artifact path must be a safe relative path")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("release artifact sha256 must be a lowercase 64-hex digest")
        _require_evidence_ref(self.evidence_ref, "release artifact evidence_ref")


@dataclass(frozen=True, slots=True)
class ComponentRef:
    lane: str
    source_sha: str
    disposition: ComponentDisposition
    component_kind: str
    pr_number: int | None = None
    ci_evidence: CIEvidence | None = None
    artifact_path: str | None = None
    artifact_sha256: str | None = None
    artifact_evidence_ref: str | None = None
    contains_behavioral_weights: bool | None = None
    contains_foreign_pretrained_weights: bool | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.lane.strip():
            raise ValueError("lane must be non-empty")
        if not _GIT_SHA_RE.fullmatch(self.source_sha):
            raise ValueError("source_sha must be an exact 40- or 64-character lowercase git SHA")
        if not self.component_kind.strip():
            raise ValueError("component_kind must be non-empty")
        if self.pr_number is not None and self.pr_number <= 0:
            raise ValueError("pr_number must be positive when supplied")
        artifact_fields = (
            self.artifact_path,
            self.artifact_sha256,
            self.artifact_evidence_ref,
        )
        if any(value is not None for value in artifact_fields):
            if any(value is None for value in artifact_fields):
                raise ValueError(
                    "component artifact evidence requires path, sha256 and evidence_ref together"
                )
            candidate = Path(self.artifact_path or "")
            if candidate.is_absolute() or ".." in candidate.parts:
                raise ValueError("component artifact_path must be a safe relative path")
            if not _SHA256_RE.fullmatch(self.artifact_sha256 or ""):
                raise ValueError("artifact_sha256 must be a lowercase 64-hex digest")
            _require_evidence_ref(
                self.artifact_evidence_ref or "", "component artifact evidence_ref"
            )
        for field_name, value in (
            ("contains_behavioral_weights", self.contains_behavioral_weights),
            ("contains_foreign_pretrained_weights", self.contains_foreign_pretrained_weights),
        ):
            if value is not None and not isinstance(value, bool):
                raise TypeError(f"{field_name} must be bool or None")


@dataclass(frozen=True, slots=True)
class StageCandidateManifest:
    """Selective integration manifest with fail-closed promotion rules."""

    stage: str
    integration_anchor_sha: str
    status: CandidateStatus
    base_lineage: bool
    components: tuple[ComponentRef, ...]
    candidate_sha: str | None = None
    audit_a: AuditEvidence | None = None
    audit_b: AuditEvidence | None = None
    release_artifact: ReleaseArtifactEvidence | None = None
    required_lanes: frozenset[str] = S0_REQUIRED_LANES

    def __post_init__(self) -> None:
        if not self.stage.strip():
            raise ValueError("stage must be non-empty")
        if not _GIT_SHA_RE.fullmatch(self.integration_anchor_sha):
            raise ValueError("integration_anchor_sha must be an exact git SHA")
        if self.candidate_sha is not None and not _GIT_SHA_RE.fullmatch(self.candidate_sha):
            raise ValueError("candidate_sha must be an exact git SHA")

        if self.stage.upper() == "S0" and not S0_REQUIRED_LANES.issubset(self.required_lanes):
            missing_policy_lanes = sorted(S0_REQUIRED_LANES - self.required_lanes)
            raise ValueError(
                "S0 required_lanes cannot weaken canonical policy; missing: "
                + ", ".join(missing_policy_lanes)
            )

        lanes = [component.lane for component in self.components]
        if len(lanes) != len(set(lanes)):
            raise ValueError("manifest contains duplicate lane entries")

        if self.base_lineage:
            for component in self.components:
                if component.disposition is not ComponentDisposition.ACCEPTED:
                    continue
                if (
                    component.contains_behavioral_weights is None
                    or component.contains_foreign_pretrained_weights is None
                ):
                    raise ValueError(
                        "accepted Base components require explicit forbidden-weight classification"
                    )
                if component.contains_foreign_pretrained_weights:
                    raise ValueError("foreign pretrained weights cannot enter Base lineage")
                if (
                    component.contains_behavioral_weights
                    or component.component_kind in _BEHAVIORAL_WEIGHT_KINDS
                ):
                    raise ValueError(
                        "behavioral/alignment/specialization weights cannot enter Base lineage"
                    )

        gated_statuses = {
            CandidateStatus.CANDIDATE,
            CandidateStatus.AUDITED_CANDIDATE,
            CandidateStatus.STABLE,
        }
        if self.status in gated_statuses:
            if self.candidate_sha is None:
                raise ValueError("candidate status requires exact candidate_sha")
            missing = self.missing_required_lanes()
            if missing:
                raise ValueError(
                    f"candidate status missing required accepted lanes: {', '.join(missing)}"
                )
            for component in self.components:
                if component.disposition is not ComponentDisposition.ACCEPTED:
                    continue
                if component.ci_evidence is None:
                    raise ValueError(
                        f"accepted candidate component {component.lane} requires CI evidence"
                    )
                if component.ci_evidence.head_sha != component.source_sha:
                    raise ValueError(
                        f"CI evidence for {component.lane} is not bound to source_sha"
                    )
                if not component.ci_evidence.passes:
                    raise ValueError(
                        f"accepted candidate component {component.lane} requires successful CI"
                    )

        audits = tuple(audit for audit in (self.audit_a, self.audit_b) if audit is not None)
        if audits and self.candidate_sha is None:
            raise ValueError("audit evidence requires exact candidate_sha")
        for audit in audits:
            if audit.candidate_sha != self.candidate_sha:
                raise ValueError("audit evidence is not bound to this exact candidate_sha")

        if self.audit_a is not None and self.audit_a.auditor_id != "AUDIT-A":
            raise ValueError("audit_a evidence must identify AUDIT-A")
        if self.audit_b is not None and self.audit_b.auditor_id != "AUDIT-B":
            raise ValueError("audit_b evidence must identify AUDIT-B")

        if self.status in {CandidateStatus.AUDITED_CANDIDATE, CandidateStatus.STABLE}:
            if self.audit_a is None or self.audit_b is None:
                raise ValueError(
                    "AUDITED_CANDIDATE/STABLE require AUDIT-A and AUDIT-B evidence"
                )
            if not self.audit_a.passes or not self.audit_b.passes:
                raise ValueError(
                    "AUDITED_CANDIDATE/STABLE require passing independent "
                    "AUDIT-A and AUDIT-B verdicts"
                )
            if self.audit_a.evidence_ref == self.audit_b.evidence_ref:
                raise ValueError("AUDIT-A and AUDIT-B must have distinct evidence_ref values")

        if self.status is CandidateStatus.STABLE and self.release_artifact is None:
            raise ValueError("STABLE requires release artifact hash evidence")

    @classmethod
    def compose(
        cls,
        *,
        stage: str,
        integration_anchor_sha: str,
        status: CandidateStatus,
        base_lineage: bool,
        components: Iterable[ComponentRef],
        candidate_sha: str | None = None,
        audit_a: AuditEvidence | None = None,
        audit_b: AuditEvidence | None = None,
        release_artifact: ReleaseArtifactEvidence | None = None,
        required_lanes: frozenset[str] = S0_REQUIRED_LANES,
    ) -> StageCandidateManifest:
        return cls(
            stage=stage,
            integration_anchor_sha=integration_anchor_sha,
            status=status,
            base_lineage=base_lineage,
            components=tuple(components),
            candidate_sha=candidate_sha,
            audit_a=audit_a,
            audit_b=audit_b,
            release_artifact=release_artifact,
            required_lanes=required_lanes,
        )

    def accepted_lanes(self) -> frozenset[str]:
        return frozenset(
            component.lane
            for component in self.components
            if component.disposition is ComponentDisposition.ACCEPTED
        )

    def missing_required_lanes(self) -> tuple[str, ...]:
        return tuple(sorted(self.required_lanes - self.accepted_lanes()))

    def ready_for_candidate(self) -> bool:
        return not self.missing_required_lanes()

    def audits_pass(self) -> bool:
        return bool(
            self.audit_a is not None
            and self.audit_b is not None
            and self.audit_a.passes
            and self.audit_b.passes
            and self.audit_a.candidate_sha == self.candidate_sha
            and self.audit_b.candidate_sha == self.candidate_sha
            and self.audit_a.auditor_id == "AUDIT-A"
            and self.audit_b.auditor_id == "AUDIT-B"
            and self.audit_a.evidence_ref != self.audit_b.evidence_ref
        )


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_artifact(repo_root: Path, relative_path: str, expected_sha256: str) -> None:
    artifact = (repo_root / relative_path).resolve()
    if not artifact.is_relative_to(repo_root) or not artifact.is_file():
        raise ValueError(f"artifact does not exist inside repository: {relative_path}")
    if _sha256_file(artifact) != expected_sha256:
        raise ValueError(f"artifact sha256 mismatch: {relative_path}")


def validate_repository_evidence(
    manifest: StageCandidateManifest,
    repo_root: str | Path = ".",
) -> None:
    """Verify candidate ancestry and locally materialized artifact hashes.

    CI/audit evidence is captured in the manifest and independently inspectable
    through its evidence refs. Git ancestry and artifact bytes are re-verified
    here instead of trusting manifest strings.
    """

    if manifest.status is CandidateStatus.EXPERIMENTAL:
        return
    if manifest.candidate_sha is None:
        raise ValueError("repository evidence validation requires candidate_sha")

    root = Path(repo_root).resolve()
    head = _git(root, "rev-parse", "HEAD")
    if head.returncode != 0:
        raise ValueError("repo_root is not a readable Git checkout")
    if head.stdout.strip() != manifest.candidate_sha:
        raise ValueError("candidate_sha does not equal checked-out Git HEAD")

    for label, source_sha in (
        ("integration anchor", manifest.integration_anchor_sha),
        *(
            (f"component {component.lane}", component.source_sha)
            for component in manifest.components
            if component.disposition is ComponentDisposition.ACCEPTED
        ),
    ):
        result = _git(root, "merge-base", "--is-ancestor", source_sha, manifest.candidate_sha)
        if result.returncode != 0:
            raise ValueError(f"{label} SHA is not contained by candidate ancestry")

    for component in manifest.components:
        if (
            component.disposition is ComponentDisposition.ACCEPTED
            and component.artifact_path is not None
            and component.artifact_sha256 is not None
        ):
            _verify_artifact(root, component.artifact_path, component.artifact_sha256)

    if manifest.status is CandidateStatus.STABLE:
        if manifest.release_artifact is None:
            raise ValueError("STABLE requires release artifact hash evidence")
        _verify_artifact(
            root,
            manifest.release_artifact.path,
            manifest.release_artifact.sha256,
        )
