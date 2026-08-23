"""Exact provenance contracts for selective stage composition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re
from typing import Iterable

_GIT_SHA_RE = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
S0_REQUIRED_LANES = frozenset(
    {"D01", "D02", "D03", "D04", "D05", "D06", "D07", "D08"}
)
_PASSING_AUDITS = frozenset({"PASS", "PASS_WITH_NOTES"})


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


@dataclass(frozen=True, slots=True)
class ComponentRef:
    lane: str
    source_sha: str
    disposition: ComponentDisposition
    component_kind: str
    pr_number: int | None = None
    artifact_sha256: str | None = None
    contains_behavioral_weights: bool = False
    contains_foreign_pretrained_weights: bool = False
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
        if self.artifact_sha256 is not None and not _SHA256_RE.fullmatch(
            self.artifact_sha256
        ):
            raise ValueError("artifact_sha256 must be a lowercase 64-hex digest")


@dataclass(frozen=True, slots=True)
class StageCandidateManifest:
    """Selective integration manifest with fail-closed promotion rules."""

    stage: str
    integration_anchor_sha: str
    status: CandidateStatus
    base_lineage: bool
    components: tuple[ComponentRef, ...]
    candidate_sha: str | None = None
    audit_a: AuditVerdict = AuditVerdict.NOT_RUN
    audit_b: AuditVerdict = AuditVerdict.NOT_RUN
    required_lanes: frozenset[str] = S0_REQUIRED_LANES

    def __post_init__(self) -> None:
        if not self.stage.strip():
            raise ValueError("stage must be non-empty")
        if not _GIT_SHA_RE.fullmatch(self.integration_anchor_sha):
            raise ValueError("integration_anchor_sha must be an exact git SHA")
        if self.candidate_sha is not None and not _GIT_SHA_RE.fullmatch(self.candidate_sha):
            raise ValueError("candidate_sha must be an exact git SHA")

        lanes = [component.lane for component in self.components]
        if len(lanes) != len(set(lanes)):
            raise ValueError("manifest contains duplicate lane entries")

        if self.base_lineage:
            for component in self.components:
                if component.disposition is not ComponentDisposition.ACCEPTED:
                    continue
                if component.contains_foreign_pretrained_weights:
                    raise ValueError("foreign pretrained weights cannot enter Base lineage")
                behavioral_kind = component.component_kind in {
                    "behavioral_weights",
                    "alignment_weights",
                    "instruction_weights",
                    "preference_weights",
                    "rl_weights",
                }
                if component.lane == "D09" and (
                    component.contains_behavioral_weights or behavioral_kind
                ):
                    raise ValueError("D09 behavioral/alignment weights cannot enter Base lineage")

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

        if self.status in {CandidateStatus.AUDITED_CANDIDATE, CandidateStatus.STABLE}:
            audits_pass = (
                self.audit_a.value in _PASSING_AUDITS
                and self.audit_b.value in _PASSING_AUDITS
            )
            if not audits_pass:
                raise ValueError(
                    "AUDITED_CANDIDATE/STABLE require passing independent "
                    "AUDIT-A and AUDIT-B verdicts"
                )

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
        audit_a: AuditVerdict = AuditVerdict.NOT_RUN,
        audit_b: AuditVerdict = AuditVerdict.NOT_RUN,
        required_lanes: frozenset[str] = S0_REQUIRED_LANES,
    ) -> "StageCandidateManifest":
        return cls(
            stage=stage,
            integration_anchor_sha=integration_anchor_sha,
            status=status,
            base_lineage=base_lineage,
            components=tuple(components),
            candidate_sha=candidate_sha,
            audit_a=audit_a,
            audit_b=audit_b,
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
        return self.audit_a.value in _PASSING_AUDITS and self.audit_b.value in _PASSING_AUDITS
