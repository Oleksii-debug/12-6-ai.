import pytest

from twelve_six.integration import (
    AuditEvidence,
    AuditVerdict,
    CandidateStatus,
    ComponentDisposition,
    ComponentRef,
    StageCandidateManifest,
)

SHA = "f2e94c7212888cdb960bb66154d56d210e9b27ab"
OTHER_SHA = "a" * 40


def _component(
    lane: str,
    disposition: ComponentDisposition = ComponentDisposition.ACCEPTED,
    *,
    behavioral: bool = False,
    foreign_pretrained: bool = False,
    component_kind: str = "infrastructure",
) -> ComponentRef:
    return ComponentRef(
        lane=lane,
        source_sha=SHA,
        disposition=disposition,
        component_kind=component_kind,
        contains_behavioral_weights=behavioral,
        contains_foreign_pretrained_weights=foreign_pretrained,
    )


def _all_s0_components() -> list[ComponentRef]:
    return [_component(f"D0{index}") for index in range(1, 9)]


def _audit(
    auditor_id: str,
    verdict: AuditVerdict = AuditVerdict.PASS,
    *,
    candidate_sha: str = SHA,
    evidence_ref: str | None = None,
) -> AuditEvidence:
    return AuditEvidence(
        auditor_id=auditor_id,
        verdict=verdict,
        candidate_sha=candidate_sha,
        evidence_ref=evidence_ref or f"issue://{auditor_id}",
    )


def test_experimental_manifest_can_be_incomplete() -> None:
    manifest = StageCandidateManifest.compose(
        stage="S0",
        integration_anchor_sha=SHA,
        status=CandidateStatus.EXPERIMENTAL,
        base_lineage=True,
        components=[_component("D01")],
    )
    assert not manifest.ready_for_candidate()
    assert "D02" in manifest.missing_required_lanes()
    assert "D08" in manifest.missing_required_lanes()


def test_component_requires_exact_full_source_sha() -> None:
    with pytest.raises(ValueError, match="exact 40- or 64"):
        ComponentRef(
            lane="D01",
            source_sha=SHA[:7],
            disposition=ComponentDisposition.ACCEPTED,
            component_kind="model",
        )


def test_candidate_status_requires_exact_candidate_sha() -> None:
    with pytest.raises(ValueError, match="candidate_sha"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            status=CandidateStatus.CANDIDATE,
            base_lineage=True,
            components=_all_s0_components(),
        )


def test_candidate_status_requires_d01_through_d08() -> None:
    with pytest.raises(ValueError, match="D08"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            candidate_sha=SHA,
            status=CandidateStatus.CANDIDATE,
            base_lineage=True,
            components=_all_s0_components()[:-1],
        )


def test_s0_required_lanes_cannot_be_weakened() -> None:
    with pytest.raises(ValueError, match="cannot weaken canonical policy"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            status=CandidateStatus.EXPERIMENTAL,
            base_lineage=True,
            components=[],
            required_lanes=frozenset({"D01"}),
        )


def test_accepted_base_component_requires_explicit_weight_classification() -> None:
    component = ComponentRef(
        lane="D01",
        source_sha=SHA,
        disposition=ComponentDisposition.ACCEPTED,
        component_kind="model",
    )
    with pytest.raises(ValueError, match="explicit forbidden-weight classification"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            status=CandidateStatus.EXPERIMENTAL,
            base_lineage=True,
            components=[component],
        )


def test_base_manifest_rejects_accepted_d09_behavioral_weights() -> None:
    d09 = ComponentRef(
        lane="D09",
        source_sha=SHA,
        disposition=ComponentDisposition.ACCEPTED,
        component_kind="infrastructure",
        contains_behavioral_weights=True,
        contains_foreign_pretrained_weights=False,
    )
    with pytest.raises(ValueError, match="cannot enter Base lineage"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            status=CandidateStatus.EXPERIMENTAL,
            base_lineage=True,
            components=[d09],
        )


def test_base_manifest_rejects_behavioral_weight_kind_from_any_lane() -> None:
    component = _component("D07", component_kind="instruction_weights")
    with pytest.raises(ValueError, match="cannot enter Base lineage"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            status=CandidateStatus.EXPERIMENTAL,
            base_lineage=True,
            components=[component],
        )


def test_base_manifest_rejects_foreign_pretrained_weights() -> None:
    d01 = _component("D01", foreign_pretrained=True)
    with pytest.raises(ValueError, match="foreign pretrained"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            status=CandidateStatus.EXPERIMENTAL,
            base_lineage=True,
            components=[d01],
        )


def test_audit_evidence_must_match_exact_candidate_sha() -> None:
    with pytest.raises(ValueError, match="exact candidate_sha"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            candidate_sha=SHA,
            status=CandidateStatus.CANDIDATE,
            base_lineage=True,
            components=_all_s0_components(),
            audit_a=_audit("AUDIT-A", candidate_sha=OTHER_SHA),
        )


def test_audit_slots_require_named_independent_auditors() -> None:
    with pytest.raises(ValueError, match="audit_a evidence must identify AUDIT-A"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            candidate_sha=SHA,
            status=CandidateStatus.CANDIDATE,
            base_lineage=True,
            components=_all_s0_components(),
            audit_a=_audit("AUDIT-B"),
        )


def test_audited_candidate_requires_both_audits() -> None:
    with pytest.raises(ValueError, match="AUDIT-A and AUDIT-B evidence"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            candidate_sha=SHA,
            status=CandidateStatus.AUDITED_CANDIDATE,
            base_lineage=True,
            components=_all_s0_components(),
            audit_a=_audit("AUDIT-A"),
        )


def test_audited_candidate_rejects_nonpassing_audit() -> None:
    with pytest.raises(ValueError, match="passing independent"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            candidate_sha=SHA,
            status=CandidateStatus.AUDITED_CANDIDATE,
            base_lineage=True,
            components=_all_s0_components(),
            audit_a=_audit("AUDIT-A"),
            audit_b=_audit("AUDIT-B", AuditVerdict.CHANGES_REQUIRED),
        )


def test_stable_requires_distinct_audit_evidence_refs() -> None:
    with pytest.raises(ValueError, match="distinct evidence_ref"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            candidate_sha=SHA,
            status=CandidateStatus.STABLE,
            base_lineage=True,
            components=_all_s0_components(),
            audit_a=_audit("AUDIT-A", evidence_ref="issue://same"),
            audit_b=_audit("AUDIT-B", evidence_ref="issue://same"),
        )


def test_stable_accepts_exact_independent_passing_audits() -> None:
    manifest = StageCandidateManifest.compose(
        stage="S0",
        integration_anchor_sha=SHA,
        candidate_sha=SHA,
        status=CandidateStatus.STABLE,
        base_lineage=True,
        components=_all_s0_components(),
        audit_a=_audit("AUDIT-A"),
        audit_b=_audit("AUDIT-B", AuditVerdict.PASS_WITH_NOTES),
    )
    assert manifest.ready_for_candidate()
    assert manifest.audits_pass()


def test_duplicate_lane_entries_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate lane"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            status=CandidateStatus.EXPERIMENTAL,
            base_lineage=True,
            components=[_component("D01"), _component("D01")],
        )
