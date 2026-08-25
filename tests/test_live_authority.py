from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from twelve_six.integration import live_authority as live
from twelve_six.integration.dependency_lock import SUPPORTED_PROFILES, canonical_json_bytes
from twelve_six.integration.manifest import (
    AuditEvidence,
    AuditVerdict,
    CandidateStatus,
    CIEvidence,
    ComponentDisposition,
    ComponentRef,
    StageCandidateManifest,
)
from twelve_six.integration.release_attestation import (
    ArtifactBinding,
    CandidateCIEvidence,
    DependencyLockBinding,
    EnvironmentArtifactEvidence,
    ReleaseAttestation,
    ReleaseAttestationError,
)

CANDIDATE_SHA = "c" * 40
ANCHOR_SHA = "d" * 40
CI_COMPLETED = "2026-08-24T15:00:00+00:00"


def _run_ref(run_id: int) -> str:
    return f"{live.GITHUB_WEB_ROOT}/actions/runs/{run_id}"


def _artifact_ref(run_id: int, artifact_id: int) -> str:
    return f"{_run_ref(run_id)}/artifacts/{artifact_id}"


def _run_payload(
    run_id: int,
    head_sha: str,
    *,
    name: str = live.CANDIDATE_WORKFLOW_NAME,
    status: str = "completed",
    conclusion: str | None = "success",
    attempt: int = 1,
    updated_at: str = CI_COMPLETED,
    repository: str = live.CANONICAL_REPOSITORY,
) -> dict[str, Any]:
    return {
        "id": run_id,
        "name": name,
        "status": status,
        "conclusion": conclusion,
        "head_sha": head_sha,
        "html_url": _run_ref(run_id),
        "run_attempt": attempt,
        "updated_at": updated_at,
        "repository": {"full_name": repository},
    }


def _artifact_payload(
    artifact_id: int,
    *,
    run_id: int,
    head_sha: str,
    digest: str | None = None,
    expired: bool = False,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": artifact_id,
        "expired": expired,
        "workflow_run": {"id": run_id, "head_sha": head_sha},
    }
    if digest is not None:
        result["digest"] = f"sha256:{digest}"
    return result


def _components() -> tuple[ComponentRef, ...]:
    result: list[ComponentRef] = []
    for index in range(1, 9):
        lane = f"D0{index}"
        source_sha = f"{index:040x}"
        run_id = 100 + index
        result.append(
            ComponentRef(
                lane=lane,
                source_sha=source_sha,
                disposition=ComponentDisposition.ACCEPTED,
                component_kind="source",
                pr_number=index,
                ci_evidence=CIEvidence(
                    run_id=run_id,
                    head_sha=source_sha,
                    conclusion="success",
                    evidence_ref=_run_ref(run_id),
                ),
                contains_behavioral_weights=False,
                contains_foreign_pretrained_weights=False,
            )
        )
    return tuple(result)


def _stage_manifest() -> StageCandidateManifest:
    return StageCandidateManifest(
        stage="S0",
        integration_anchor_sha=ANCHOR_SHA,
        status=CandidateStatus.CANDIDATE,
        base_lineage=True,
        components=_components(),
        candidate_sha=CANDIDATE_SHA,
    )


def _write_supply(root: Path, *, review_required: bool = False) -> None:
    profiles: dict[str, Any] = {}
    for index, profile_id in enumerate(sorted(SUPPORTED_PROFILES), start=1):
        profiles[profile_id] = {
            "profile_manifest_sha256": f"{index:064x}",
            "component_count": 1,
            "components": [
                {
                    "name": "demo",
                    "version": "1.0",
                    "purl": "pkg:pypi/demo@1.0",
                    "groups": ["runtime"],
                    "artifact_sha256": [f"{index + 10:064x}"],
                }
            ],
        }
    lock_index = {"semantic_sha256": "3" * 64, "file_sha256": "2" * 64}
    sbom: dict[str, Any] = {
        "schema_version": live.DEPENDENCY_SBOM_SCHEMA,
        "project": "twelve-six-ai",
        "repository_full_name": live.CANONICAL_REPOSITORY,
        "source_sha": CANDIDATE_SHA,
        "lock_index": lock_index,
        "profiles": profiles,
    }
    sbom["sbom_sha256"] = hashlib.sha256(canonical_json_bytes(sbom)).hexdigest()
    report: dict[str, Any] = {
        "schema_version": live.DEPENDENCY_EVIDENCE_SCHEMA,
        "repository_full_name": live.CANONICAL_REPOSITORY,
        "source_sha": CANDIDATE_SHA,
        "generated_at": "2026-08-24T14:59:00+00:00",
        "lock_index": lock_index,
        "sbom_sha256": sbom["sbom_sha256"],
        "scan_sources": {"osv": {"status": "SUCCESS"}, "pypi": {"status": "SUCCESS"}},
        "status": (
            "EVIDENCE_COMPLETE_REVIEW_REQUIRED"
            if review_required
            else live.DEPENDENCY_CLEAN_STATUS
        ),
        "components": [
            {
                "key": "demo==1.0",
                "name": "demo",
                "version": "1.0",
                "purl": "pkg:pypi/demo@1.0",
                "profiles": sorted(SUPPORTED_PROFILES),
                "license": {
                    "status": "UNRESOLVED" if review_required else "DECLARED",
                    "metadata_sha256": "4" * 64,
                },
                "advisories": {
                    "status": "QUERIED",
                    "response_sha256": "5" * 64,
                    "vulnerabilities": [],
                },
            }
        ],
        "truth_boundary": {
            "audit_verdict": False,
            "license_approval": False,
            "vulnerability_risk_acceptance": False,
            "promotion_authority": False,
        },
    }
    report["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    evidence = root / "evidence"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "sbom.json").write_text(
        json.dumps(sbom, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (evidence / "dependency-report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _attestation(root: Path) -> ReleaseAttestation:
    _write_supply(root)
    return ReleaseAttestation(
        repository=live.CANONICAL_REPOSITORY,
        stage="S0",
        status=CandidateStatus.CANDIDATE,
        candidate_sha=CANDIDATE_SHA,
        candidate_manifest=ArtifactBinding(
            kind="candidate_manifest",
            path="candidate.json",
            sha256="1" * 64,
            producer_sha=CANDIDATE_SHA,
            evidence_ref=f"{live.GITHUB_WEB_ROOT}/pull/99",
        ),
        dependency_lock=DependencyLockBinding(
            path="requirements/locks/index.json",
            file_sha256="2" * 64,
            index_sha256="3" * 64,
            producer_sha=CANDIDATE_SHA,
            evidence_ref=f"{live.GITHUB_WEB_ROOT}/pull/58",
        ),
        candidate_ci=CandidateCIEvidence(
            run_id=200,
            head_sha=CANDIDATE_SHA,
            conclusion="success",
            completed_at_utc=CI_COMPLETED,
            evidence_ref=_run_ref(200),
        ),
        environment_evidence=tuple(
            EnvironmentArtifactEvidence(
                profile_id=profile_id,
                run_id=200,
                source_sha=CANDIDATE_SHA,
                artifact_id=201 + index,
                archive_sha256=f"{20 + index:064x}",
                evidence_ref=_run_ref(200),
            )
            for index, profile_id in enumerate(sorted(SUPPORTED_PROFILES))
        ),
        checkpoint_artifacts=(
            ArtifactBinding(
                kind="checkpoint_manifest",
                path="checkpoint/MANIFEST.json",
                sha256="6" * 64,
                producer_sha=CANDIDATE_SHA,
                evidence_ref=_artifact_ref(200, 301),
            ),
            ArtifactBinding(
                kind="model_weights",
                path="checkpoint/model.safetensors",
                sha256="7" * 64,
                producer_sha=CANDIDATE_SHA,
                evidence_ref=_artifact_ref(200, 301),
            ),
        ),
        supply_chain_artifacts=(
            ArtifactBinding(
                kind="sbom",
                path="evidence/sbom.json",
                sha256="8" * 64,
                producer_sha=CANDIDATE_SHA,
                evidence_ref=_artifact_ref(300, 401),
            ),
            ArtifactBinding(
                kind="dependency_report",
                path="evidence/dependency-report.json",
                sha256="9" * 64,
                producer_sha=CANDIDATE_SHA,
                evidence_ref=_artifact_ref(300, 402),
            ),
        ),
        release_artifact=None,
        promotion_authority_ref=None,
        attestation_sha256="a" * 64,
    )


def _responses(attestation: ReleaseAttestation) -> dict[str, Mapping[str, Any]]:
    responses: dict[str, Mapping[str, Any]] = {
        f"{live.GITHUB_API_ROOT}/actions/runs/200": _run_payload(200, CANDIDATE_SHA),
        f"{live.GITHUB_API_ROOT}/actions/runs/300": _run_payload(
            300,
            CANDIDATE_SHA,
            name=live.DEPENDENCY_WORKFLOW_NAME,
        ),
    }
    for component in _components():
        assert component.ci_evidence is not None
        responses[
            f"{live.GITHUB_API_ROOT}/actions/runs/{component.ci_evidence.run_id}"
        ] = _run_payload(component.ci_evidence.run_id, component.source_sha)
    candidate_artifacts = [
        _artifact_payload(
            item.artifact_id,
            run_id=200,
            head_sha=CANDIDATE_SHA,
            digest=item.archive_sha256,
        )
        for item in attestation.environment_evidence
    ]
    candidate_artifacts.append(
        _artifact_payload(301, run_id=200, head_sha=CANDIDATE_SHA)
    )
    responses[
        f"{live.GITHUB_API_ROOT}/actions/runs/200/artifacts?per_page=100&page=1"
    ] = {"total_count": len(candidate_artifacts), "artifacts": candidate_artifacts}
    supply_artifacts = [
        _artifact_payload(401, run_id=300, head_sha=CANDIDATE_SHA),
        _artifact_payload(402, run_id=300, head_sha=CANDIDATE_SHA),
    ]
    responses[
        f"{live.GITHUB_API_ROOT}/actions/runs/300/artifacts?per_page=100&page=1"
    ] = {"total_count": 2, "artifacts": supply_artifacts}
    return responses


def _getter(responses: Mapping[str, Mapping[str, Any]]):
    def get_json(url: str) -> Mapping[str, Any]:
        try:
            return responses[url]
        except KeyError as exc:
            raise AssertionError(f"unexpected live URL: {url}") from exc

    return get_json


def test_exact_completed_first_attempt_workflow_is_accepted() -> None:
    payload = _run_payload(42, CANDIDATE_SHA)
    result = live.verify_workflow_run(
        run_id=42,
        expected_head_sha=CANDIDATE_SHA,
        evidence_ref=_run_ref(42),
        get_json=lambda _: payload,
        expected_completed_at=CI_COMPLETED,
    )
    assert result["id"] == 42


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "in_progress", "not completed"),
        ("conclusion", "failure", "did not conclude success"),
        ("head_sha", "b" * 40, "head SHA is stale"),
        ("run_attempt", 2, "reruns are not bindable"),
        ("repository", {"full_name": "foreign/repo"}, "repository identity mismatch"),
        ("updated_at", "2026-08-24T15:00:01Z", "completion timestamp differs"),
    ],
)
def test_workflow_run_rejects_stale_failed_or_replayed_authority(
    field: str,
    value: Any,
    message: str,
) -> None:
    payload = _run_payload(42, CANDIDATE_SHA)
    payload[field] = value
    with pytest.raises(live.LiveAuthorityError, match=message):
        live.verify_workflow_run(
            run_id=42,
            expected_head_sha=CANDIDATE_SHA,
            evidence_ref=_run_ref(42),
            get_json=lambda _: payload,
            expected_completed_at=CI_COMPLETED,
        )


def test_workflow_run_rejects_fabricated_ref() -> None:
    with pytest.raises(live.LiveAuthorityError, match="canonical GitHub Actions run URL"):
        live.verify_workflow_run(
            run_id=42,
            expected_head_sha=CANDIDATE_SHA,
            evidence_ref="issue://D10/fake-ci",
            get_json=lambda _: _run_payload(42, CANDIDATE_SHA),
        )


def test_audit_comment_binds_candidate_verdict_cutoff_and_issue() -> None:
    cutoff = "2026-08-24T15:05:00+00:00"
    ref = f"{live.GITHUB_WEB_ROOT}/issues/13#issuecomment-7001"
    audit = AuditEvidence(
        auditor_id="AUDIT-A",
        verdict=AuditVerdict.PASS,
        candidate_sha=CANDIDATE_SHA,
        cutoff_utc=cutoff,
        evidence_ref=ref,
    )
    comment = {
        "html_url": ref,
        "created_at": "2026-08-24T15:06:00Z",
        "body": (
            f"Candidate SHA: `{CANDIDATE_SHA}`\n"
            f"Audit cutoff: `{cutoff}`\n"
            "**Verdict:** **PASS**\n"
        ),
    }
    live.verify_audit_evidence(
        audit,
        expected_issue=13,
        candidate_ci_completed_at=CI_COMPLETED,
        get_json=lambda _: comment,
    )
    bad = dict(comment)
    bad["body"] = f"Candidate SHA: {CANDIDATE_SHA}\nAudit cutoff: {cutoff}\nVerdict: BLOCKED\n"
    with pytest.raises(live.LiveAuthorityError, match="verdict differs"):
        live.verify_audit_evidence(
            audit,
            expected_issue=13,
            candidate_ci_completed_at=CI_COMPLETED,
            get_json=lambda _: bad,
        )
    crossed = AuditEvidence(
        auditor_id="AUDIT-A",
        verdict=AuditVerdict.PASS,
        candidate_sha=CANDIDATE_SHA,
        cutoff_utc=cutoff,
        evidence_ref=f"{live.GITHUB_WEB_ROOT}/issues/14#issuecomment-7001",
    )
    with pytest.raises(live.LiveAuthorityError, match="canonical auditor issue"):
        live.verify_audit_evidence(
            crossed,
            expected_issue=13,
            candidate_ci_completed_at=CI_COMPLETED,
            get_json=lambda _: {},
        )


def test_audit_comment_rejects_missing_candidate_and_stale_publication() -> None:
    cutoff = "2026-08-24T15:05:00+00:00"
    ref = f"{live.GITHUB_WEB_ROOT}/issues/13#issuecomment-7001"
    audit = AuditEvidence(
        auditor_id="AUDIT-A",
        verdict=AuditVerdict.PASS,
        candidate_sha=CANDIDATE_SHA,
        cutoff_utc=cutoff,
        evidence_ref=ref,
    )
    missing = {"html_url": ref, "created_at": "2026-08-24T15:06:00Z", "body": f"Audit cutoff: {cutoff}\nVerdict: PASS\n"}
    with pytest.raises(live.LiveAuthorityError, match="exact candidate SHA"):
        live.verify_audit_evidence(
            audit,
            expected_issue=13,
            candidate_ci_completed_at=CI_COMPLETED,
            get_json=lambda _: missing,
        )
    stale = {
        "html_url": ref,
        "created_at": "2026-08-24T14:59:00Z",
        "body": f"Candidate SHA: {CANDIDATE_SHA}\nAudit cutoff: {cutoff}\nVerdict: PASS\n",
    }
    with pytest.raises(live.LiveAuthorityError, match="predates its cutoff"):
        live.verify_audit_evidence(
            audit,
            expected_issue=13,
            candidate_ci_completed_at=CI_COMPLETED,
            get_json=lambda _: stale,
        )


def test_stable_authority_requires_exact_owner_marker_and_freshness() -> None:
    ref = f"{live.GITHUB_WEB_ROOT}/issues/1#issuecomment-8001"
    comment = {
        "html_url": ref,
        "created_at": "2026-08-24T15:10:00Z",
        "body": f"Candidate SHA: {CANDIDATE_SHA}\nPROMOTION_AUTHORIZED: STABLE\n",
    }
    live.verify_promotion_authority(
        ref,
        candidate_sha=CANDIDATE_SHA,
        not_before=datetime(2026, 8, 24, 15, 9, tzinfo=UTC),
        get_json=lambda _: comment,
    )
    for body in (
        "PROMOTION_AUTHORIZED: STABLE\n",
        f"Candidate SHA: {CANDIDATE_SHA}\n",
        f"Candidate SHA: {'b' * 40}\nPROMOTION_AUTHORIZED: STABLE\n",
    ):
        bad = dict(comment)
        bad["body"] = body
        with pytest.raises(live.LiveAuthorityError):
            live.verify_promotion_authority(
                ref,
                candidate_sha=CANDIDATE_SHA,
                not_before=datetime(2026, 8, 24, 15, 9, tzinfo=UTC),
                get_json=lambda _, value=bad: value,
            )


def test_dependency_security_requires_clean_complete_exact_candidate_evidence(
    tmp_path: Path,
) -> None:
    attestation = _attestation(tmp_path)
    live.verify_dependency_security_evidence(attestation, artifact_root=tmp_path)
    _write_supply(tmp_path, review_required=True)
    with pytest.raises(live.LiveAuthorityError, match="license evidence remains unresolved"):
        live.verify_dependency_security_evidence(attestation, artifact_root=tmp_path)


def test_dependency_security_rejects_missing_component_coverage(tmp_path: Path) -> None:
    attestation = _attestation(tmp_path)
    path = tmp_path / "evidence" / "dependency-report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    report.pop("evidence_sha256")
    report["components"] = []
    report["evidence_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(live.LiveAuthorityError, match="component evidence is missing"):
        live.verify_dependency_security_evidence(attestation, artifact_root=tmp_path)


def test_live_candidate_composes_offline_and_live_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attestation = _attestation(tmp_path)
    stage_manifest = _stage_manifest()
    monkeypatch.setattr(live, "validate_release_attestation", lambda *_args, **_kwargs: stage_manifest)
    responses = _responses(attestation)
    assert live.validate_live_promotion_authority(
        attestation,
        repo_root=tmp_path,
        artifact_root=tmp_path,
        get_json=_getter(responses),
    ) == stage_manifest


@pytest.mark.parametrize("target", ["component", "supply", "artifact"])
def test_live_candidate_rejects_stale_or_missing_external_authority(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    attestation = _attestation(tmp_path)
    stage_manifest = _stage_manifest()
    monkeypatch.setattr(live, "validate_release_attestation", lambda *_args, **_kwargs: stage_manifest)
    responses = _responses(attestation)
    if target == "component":
        component = stage_manifest.components[0]
        assert component.ci_evidence is not None
        url = f"{live.GITHUB_API_ROOT}/actions/runs/{component.ci_evidence.run_id}"
        changed = dict(responses[url])
        changed["head_sha"] = "b" * 40
        responses[url] = changed
    elif target == "supply":
        url = f"{live.GITHUB_API_ROOT}/actions/runs/300"
        changed = dict(responses[url])
        changed["conclusion"] = "failure"
        responses[url] = changed
    else:
        url = f"{live.GITHUB_API_ROOT}/actions/runs/200/artifacts?per_page=100&page=1"
        listing = deepcopy(responses[url])
        assert isinstance(listing["artifacts"], list)
        listing["artifacts"][0]["expired"] = True
        responses[url] = listing
    with pytest.raises(live.LiveAuthorityError):
        live.validate_live_promotion_authority(
            attestation,
            repo_root=tmp_path,
            artifact_root=tmp_path,
            get_json=_getter(responses),
        )


def test_offline_foreign_weight_and_ancestry_guards_run_before_live_calls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attestation = _attestation(tmp_path)

    def reject_offline(*_args, **_kwargs):
        raise ReleaseAttestationError("foreign pretrained or non-ancestor component rejected")

    monkeypatch.setattr(live, "validate_release_attestation", reject_offline)
    called = False

    def should_not_call(_: str) -> Mapping[str, Any]:
        nonlocal called
        called = True
        return {}

    with pytest.raises(ReleaseAttestationError, match="foreign pretrained or non-ancestor"):
        live.validate_live_promotion_authority(
            attestation,
            repo_root=tmp_path,
            artifact_root=tmp_path,
            get_json=should_not_call,
        )
    assert called is False
