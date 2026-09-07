from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from twelve_six.integration import (
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

SHA = "f2e94c7212888cdb960bb66154d56d210e9b27ab"
OTHER_SHA = "a" * 40
CUTOFF = "2026-08-23T13:40:00Z"


def _ci(
    head_sha: str = SHA,
    *,
    conclusion: str = "success",
    run_id: int = 1,
) -> CIEvidence:
    return CIEvidence(
        run_id=run_id,
        head_sha=head_sha,
        conclusion=conclusion,
        evidence_ref=f"https://github.test/actions/runs/{run_id}",
    )


def _component(
    lane: str,
    disposition: ComponentDisposition = ComponentDisposition.ACCEPTED,
    *,
    source_sha: str = SHA,
    ci_evidence: CIEvidence | None = None,
    behavioral: bool = False,
    foreign_pretrained: bool = False,
    component_kind: str = "infrastructure",
) -> ComponentRef:
    return ComponentRef(
        lane=lane,
        source_sha=source_sha,
        disposition=disposition,
        component_kind=component_kind,
        ci_evidence=ci_evidence if ci_evidence is not None else _ci(source_sha),
        contains_behavioral_weights=behavioral,
        contains_foreign_pretrained_weights=foreign_pretrained,
    )


def _all_s0_components(source_sha: str = SHA) -> list[ComponentRef]:
    return [_component(f"D0{index}", source_sha=source_sha) for index in range(1, 9)]


def _audit(
    auditor_id: str,
    verdict: AuditVerdict = AuditVerdict.PASS,
    *,
    candidate_sha: str = SHA,
    evidence_ref: str | None = None,
    cutoff_utc: str = CUTOFF,
) -> AuditEvidence:
    return AuditEvidence(
        auditor_id=auditor_id,
        verdict=verdict,
        candidate_sha=candidate_sha,
        cutoff_utc=cutoff_utc,
        evidence_ref=evidence_ref or f"issue://{auditor_id}",
    )


def _release(path: str = "release.bin", payload: bytes = b"release") -> ReleaseArtifactEvidence:
    return ReleaseArtifactEvidence(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        evidence_ref="issue://release",
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


def test_candidate_rejects_missing_failed_or_stale_ci() -> None:
    missing = _all_s0_components()
    missing[-1] = ComponentRef(
        lane="D08",
        source_sha=SHA,
        disposition=ComponentDisposition.ACCEPTED,
        component_kind="infrastructure",
        contains_behavioral_weights=False,
        contains_foreign_pretrained_weights=False,
    )
    with pytest.raises(ValueError, match="requires CI evidence"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            candidate_sha=SHA,
            status=CandidateStatus.CANDIDATE,
            base_lineage=True,
            components=missing,
        )

    failed = _all_s0_components()
    failed[-1] = _component("D08", ci_evidence=_ci(conclusion="failure"))
    with pytest.raises(ValueError, match="requires successful CI"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            candidate_sha=SHA,
            status=CandidateStatus.CANDIDATE,
            base_lineage=True,
            components=failed,
        )

    stale = _all_s0_components()
    stale[-1] = _component("D08", ci_evidence=_ci(OTHER_SHA))
    with pytest.raises(ValueError, match="not bound to source_sha"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            candidate_sha=SHA,
            status=CandidateStatus.CANDIDATE,
            base_lineage=True,
            components=stale,
        )


def test_base_manifest_rejects_behavioral_or_foreign_weights() -> None:
    with pytest.raises(ValueError, match="cannot enter Base lineage"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            status=CandidateStatus.EXPERIMENTAL,
            base_lineage=True,
            components=[_component("D09", behavioral=True)],
        )
    with pytest.raises(ValueError, match="foreign pretrained"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            status=CandidateStatus.EXPERIMENTAL,
            base_lineage=True,
            components=[_component("D01", foreign_pretrained=True)],
        )


def test_audit_evidence_is_candidate_and_cutoff_bound() -> None:
    with pytest.raises(ValueError, match="timezone offset"):
        _audit("AUDIT-A", cutoff_utc="2026-08-23T13:40:00")
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


def test_audited_candidate_requires_named_passing_independent_audits() -> None:
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


def test_stable_requires_release_artifact_evidence() -> None:
    with pytest.raises(ValueError, match="release artifact"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            candidate_sha=SHA,
            status=CandidateStatus.STABLE,
            base_lineage=True,
            components=_all_s0_components(),
            audit_a=_audit("AUDIT-A"),
            audit_b=_audit("AUDIT-B"),
        )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _two_commit_repo(repo: Path) -> tuple[str, str]:
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.email", "d10@example.invalid")
    _git(repo, "config", "user.name", "D10 Test")
    tracked = repo / "tracked.txt"
    tracked.write_text("one\n", encoding="utf-8")
    _git(repo, "add", "tracked.txt")
    _git(repo, "commit", "-m", "first")
    first = _git(repo, "rev-parse", "HEAD")
    tracked.write_text("two\n", encoding="utf-8")
    _git(repo, "commit", "-am", "second")
    return first, _git(repo, "rev-parse", "HEAD")


def _candidate(first: str, second: str) -> StageCandidateManifest:
    components = [
        _component(f"D0{index}", source_sha=first, ci_evidence=_ci(first, run_id=index))
        for index in range(1, 9)
    ]
    return StageCandidateManifest.compose(
        stage="S0",
        integration_anchor_sha=first,
        candidate_sha=second,
        status=CandidateStatus.CANDIDATE,
        base_lineage=True,
        components=components,
    )


def test_repository_evidence_rejects_nonancestor_component(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    first, second = _two_commit_repo(repo)
    validate_repository_evidence(_candidate(first, second), repo)

    bad_components = [
        _component(f"D0{index}", source_sha=OTHER_SHA, ci_evidence=_ci(OTHER_SHA, run_id=index))
        for index in range(1, 9)
    ]
    bad = StageCandidateManifest.compose(
        stage="S0",
        integration_anchor_sha=first,
        candidate_sha=second,
        status=CandidateStatus.CANDIDATE,
        base_lineage=True,
        components=bad_components,
    )
    with pytest.raises(ValueError, match="not contained by candidate ancestry"):
        validate_repository_evidence(bad, repo)


def test_stable_release_hash_is_recomputed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    first, second = _two_commit_repo(repo)
    payload = b"release"
    (repo / "release.bin").write_bytes(payload)
    components = [
        _component(f"D0{index}", source_sha=first, ci_evidence=_ci(first, run_id=index))
        for index in range(1, 9)
    ]
    stable = StageCandidateManifest.compose(
        stage="S0",
        integration_anchor_sha=first,
        candidate_sha=second,
        status=CandidateStatus.STABLE,
        base_lineage=True,
        components=components,
        audit_a=_audit("AUDIT-A", candidate_sha=second),
        audit_b=_audit("AUDIT-B", candidate_sha=second),
        release_artifact=_release(payload=payload),
    )
    validate_repository_evidence(stable, repo)
    (repo / "release.bin").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        validate_repository_evidence(stable, repo)


def test_duplicate_lane_entries_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate lane"):
        StageCandidateManifest.compose(
            stage="S0",
            integration_anchor_sha=SHA,
            status=CandidateStatus.EXPERIMENTAL,
            base_lineage=True,
            components=[_component("D01"), _component("D01")],
        )
