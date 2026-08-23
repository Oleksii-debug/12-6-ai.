"""Exact provenance contracts for selective stage composition."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

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


@dataclass(frozen=True, slots=True)
class AuditEvidence:
    """Durable audit verdict bound to one exact candidate SHA."""

    auditor_id: str
    verdict: AuditVerdict
    candidate_sha: str
    evidence_ref: str

    def __post_init__(self) -> None:
        if not self.auditor_id.strip():
            raise ValueError("auditor_id must be non-empty")
        if not _GIT_SHA_RE.fullmatch(self.candidate_sha):
            raise ValueError("audit candidate_sha must be an exact git SHA")
        if not self.evidence_ref.strip():
            raise ValueError("audit evidence_ref must be non-empty")

    @property
    def passes(self) -> bool:
        return self.verdict.value in _PASSING_AUDITS


@dataclass(frozen=True, slots=True)
class ComponentRef:
    lane: str
    source_sha: str
    disposition: ComponentDisposition
    component_kind: str
    pr_number: int | None = None
    artifact_sha256: str | None = None
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
        if self.artifact_sha256 is not None and not _SHA256_RE.fullmatch(
            self.artifact_sha256
        ):
            raise ValueError("artifact_sha256 must be a lowercase 64-hex digest")
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
                    raise ValueError("behavioral/alignment/specialization weights cannot enter Base lineage")

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

        audits = tuple(audit for audit in (self.audit_a, self.audit_b) if audit is not None)
        if audits and self.candidate_sha is None:
            raise ValueError("audit evidence requires exact candidate_sha")
        for audit in audits:
            if audit.candidate_sha != self.candidate_sha:
                raise ValueError("audit evidence is not bound to this exact candidate_sha")

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
            if self.audit_a.auditor_id == self.audit_b.auditor_id:
                raise ValueError("AUDIT-A and AUDIT-B must have distinct auditor_id values")
            if self.audit_a.evidence_ref == self.audit_b.evidence_ref:
                raise ValueError("AUDIT-A and AUDIT-B must have distinct evidence_ref values")

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
            and self.audit_a.auditor_id != self.audit_b.auditor_id
            and self.audit_a.evidence_ref != self.audit_b.evidence_ref
        )
