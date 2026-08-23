import pytest

from twelve_six.integration import (
    AuditVerdict,
    CandidateStatus,
    ComponentDisposition,
    ComponentRef,
    StageCandidateManifest,
)

SHA = "f2e94c7212888cdb960bb66154d56d210e9b27ab"


def _component(lane: str, disposition: ComponentDisposition = ComponentDisposition.ACCEPTED) -> ComponentRef:
    return ComponentRef(
        lane=lane,
        source_sha=SHA,
        disposition=disposition,
        component_kind="infrastructure",
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


def test_candidate_status_requires_exact_candidate_sha() -> None:
    with pytest.raises(ValueError, match="candidate_sha"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            status=CandidateStatus.CANDIDATE,
            base_lineage=True,
            components=[],
        )


def test_base_manifest_rejects_accepted_d09_behavioral_weights() -> None:
    d09 = ComponentRef(
        lane="D09",
        source_sha=SHA,
        disposition=ComponentDisposition.ACCEPTED,
        component_kind="behavioral_weights",
    )
    with pytest.raises(ValueError, match="cannot enter Base lineage"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            status=CandidateStatus.EXPERIMENTAL,
            base_lineage=True,
            components=[d09],
        )


def test_stable_requires_both_audits_and_required_lanes() -> None:
    components = [_component(f"D0{index}") for index in range(1, 8)]
    with pytest.raises(ValueError, match="AUDIT-A and AUDIT-B"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            candidate_sha=SHA,
            status=CandidateStatus.STABLE,
            base_lineage=True,
            components=components,
            audit_a=AuditVerdict.PASS,
            audit_b=AuditVerdict.NOT_RUN,
        )

    manifest = StageCandidateManifest.compose(
        stage="S0",
        integration_anchor_sha=SHA,
        candidate_sha=SHA,
        status=CandidateStatus.STABLE,
        base_lineage=True,
        components=components,
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
