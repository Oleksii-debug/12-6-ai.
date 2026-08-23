import pytest

from twelve_six.integration import (
    AuditVerdict,
    CandidateStatus,
    ComponentDisposition,
    ComponentRef,
    StageCandidateManifest,
)

SHA = "f2e94c7212888cdb960bb66154d56d210e9b27ab"


def _component(
    lane: str,
    disposition: ComponentDisposition = ComponentDisposition.ACCEPTED,
    *,
    behavioral: bool = False,
    foreign_pretrained: bool = False,
) -> ComponentRef:
    return ComponentRef(
        lane=lane,
        source_sha=SHA,
        disposition=disposition,
        component_kind="infrastructure",
        contains_behavioral_weights=behavioral,
        contains_foreign_pretrained_weights=foreign_pretrained,
    )


def _all_s0_components() -> list[ComponentRef]:
    return [_component(f"D0{index}") for index in range(1, 9)]


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


def test_base_manifest_rejects_accepted_d09_behavioral_weights() -> None:
    d09 = ComponentRef(
        lane="D09",
        source_sha=SHA,
        disposition=ComponentDisposition.ACCEPTED,
        component_kind="infrastructure",
        contains_behavioral_weights=True,
    )
    with pytest.raises(ValueError, match="cannot enter Base lineage"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            status=CandidateStatus.EXPERIMENTAL,
            base_lineage=True,
            components=[d09],
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


def test_audited_candidate_requires_both_audits() -> None:
    with pytest.raises(ValueError, match="AUDIT-A and AUDIT-B"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            candidate_sha=SHA,
            status=CandidateStatus.AUDITED_CANDIDATE,
            base_lineage=True,
            components=_all_s0_components(),
            audit_a=AuditVerdict.PASS,
            audit_b=AuditVerdict.NOT_RUN,
        )


def test_stable_requires_both_audits_and_required_lanes() -> None:
    with pytest.raises(ValueError, match="AUDIT-A and AUDIT-B"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            candidate_sha=SHA,
            status=CandidateStatus.STABLE,
            base_lineage=True,
            components=_all_s0_components(),
            audit_a=AuditVerdict.PASS,
            audit_b=AuditVerdict.NOT_RUN,
        )

    manifest = StageCandidateManifest.compose(
        stage="S0",
        integration_anchor_sha=SHA,
        candidate_sha=SHA,
        status=CandidateStatus.STABLE,
        base_lineage=True,
        components=_all_s0_components(),
        audit_a=AuditVerdict.PASS,
        audit_b=AuditVerdict.PASS_WITH_NOTES,
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
