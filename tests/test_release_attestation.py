from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from twelve_six.integration.dependency_lock import sha256_file
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
    CandidateCIEvidence,
    ReleaseAttestation,
    ReleaseAttestationError,
    compute_attestation_sha256,
    load_release_attestation,
    validate_audit_freshness,
    validate_release_attestation,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_SHA = "3d5d2332577d1ccb2b6ecbb5197b1d95a4baba6f"
OTHER_SHA = "a" * 40
CI_RUN = 32742220948
CANDIDATE_TEMPLATE_SHA256 = "6391442df431f7b0d8ac413184a7de92da62c08824b45f2515d9c4e9eb685c39"
LOCK_FILE_SHA256 = "61fa31fbb5da7a4289cccce5abfcebde943664f5318b0ce3d69ae9bb3db852ac"
LOCK_INDEX_SHA256 = "5de40d40012123ccf654b3e29d9cd47df814978e4155ca9dde232b61e9cd6341"


def _environment_evidence() -> list[dict[str, object]]:
    return [
        {
            "profile_id": "linux-aarch64",
            "run_id": CI_RUN,
            "source_sha": BASE_SHA,
            "artifact_id": 9525665931,
            "archive_sha256": "35a575485108c734b531005e2aef3fa6fb3037b232fd751a0cf03504231d72d3",
            "evidence_ref": (
                "https://github.com/Oleksii-debug/12-6-ai./actions/runs/32742220948"
            ),
        },
        {
            "profile_id": "linux-x86_64",
            "run_id": CI_RUN,
            "source_sha": BASE_SHA,
            "artifact_id": 9525668681,
            "archive_sha256": "60f1e475ed0d851f859c8d98baeacda2756809818e3d3a1b3d3c865d1a2a12d3",
            "evidence_ref": (
                "https://github.com/Oleksii-debug/12-6-ai./actions/runs/32742220948"
            ),
        },
    ]


def _binding(kind: str, *, path: str, sha256: str, producer_sha: str = BASE_SHA) -> dict[str, str]:
    return {
        "kind": kind,
        "path": path,
        "sha256": sha256,
        "producer_sha": producer_sha,
        "evidence_ref": f"issue://D10/{kind}",
    }


def _gated_payload(status: str = "candidate") -> dict[str, object]:
    return {
        "schema_version": "12-6.release-attestation.v1",
        "repository": "Oleksii-debug/12-6-ai.",
        "stage": "S0",
        "status": status,
        "candidate_sha": BASE_SHA,
        "candidate_manifest": _binding(
            "candidate_manifest",
            path="configs/releases/s0_candidate.template.json",
            sha256=CANDIDATE_TEMPLATE_SHA256,
        ),
        "dependency_lock": {
            "path": "requirements/locks/index.json",
            "file_sha256": LOCK_FILE_SHA256,
            "index_sha256": LOCK_INDEX_SHA256,
            "producer_sha": BASE_SHA,
            "evidence_ref": "issue://D10/dependency-lock",
        },
        "candidate_ci": {
            "run_id": CI_RUN,
            "head_sha": BASE_SHA,
            "conclusion": "success",
            "completed_at_utc": "2026-08-24T15:03:00Z",
            "evidence_ref": (
                "https://github.com/Oleksii-debug/12-6-ai./actions/runs/32742220948"
            ),
        },
        "environment_evidence": _environment_evidence(),
        "checkpoint_artifacts": [
            _binding("checkpoint_manifest", path="checkpoint/MANIFEST.json", sha256="1" * 64),
            _binding("model_weights", path="checkpoint/model.safetensors", sha256="2" * 64),
        ],
        "supply_chain_artifacts": [
            _binding("sbom", path="evidence/sbom.json", sha256="3" * 64),
            _binding("dependency_report", path="evidence/dependencies.json", sha256="4" * 64),
        ],
        "release_artifact": None,
        "promotion_authority_ref": None,
    }


def _write_and_load(
    tmp_path: Path, payload: dict[str, object]
) -> ReleaseAttestation:
    document = copy.deepcopy(payload)
    document["attestation_sha256"] = compute_attestation_sha256(document)
    path = tmp_path / "attestation.json"
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return load_release_attestation(path)


def _repo_binding(kind: str, path: str) -> dict[str, str]:
    return _binding(kind, path=path, sha256=sha256_file(REPO_ROOT / path))


def test_prepared_attestation_validates_exact_green_substrate() -> None:
    path = REPO_ROOT / "configs/releases/s0_release_attestation.prepared.json"
    attestation = load_release_attestation(path)
    manifest = validate_release_attestation(attestation, repo_root=REPO_ROOT)
    assert attestation.status is CandidateStatus.EXPERIMENTAL
    assert manifest is not None
    assert manifest.status is CandidateStatus.EXPERIMENTAL


def test_attestation_self_hash_tamper_is_rejected(tmp_path: Path) -> None:
    raw = json.loads(
        (REPO_ROOT / "configs/releases/s0_release_attestation.prepared.json").read_text(encoding="utf-8")
    )
    raw["repository"] = "Oleksii-debug/12-6-ai"
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ReleaseAttestationError, match="self-hash mismatch"):
        load_release_attestation(path)


@pytest.mark.parametrize("mode", ["missing", "failed", "stale"])
def test_candidate_combined_ci_must_be_exact_and_successful(tmp_path: Path, mode: str) -> None:
    payload = _gated_payload()
    if mode == "missing":
        payload["candidate_ci"] = None
        expected = "combined CI evidence"
    else:
        candidate_ci = payload["candidate_ci"]
        assert isinstance(candidate_ci, dict)
        if mode == "failed":
            candidate_ci["conclusion"] = "failure"
            expected = "successful combined CI"
        else:
            candidate_ci["head_sha"] = OTHER_SHA
            expected = "stale for candidate_sha"
    with pytest.raises(ReleaseAttestationError, match=expected):
        _write_and_load(tmp_path, payload)


@pytest.mark.parametrize("mode", ["missing_profile", "stale_head", "wrong_run"])
def test_candidate_environment_evidence_is_exact_head_bound(tmp_path: Path, mode: str) -> None:
    payload = _gated_payload()
    environment = payload["environment_evidence"]
    assert isinstance(environment, list)
    if mode == "missing_profile":
        environment.pop()
        expected = "profile set mismatch"
    elif mode == "stale_head":
        environment[0]["source_sha"] = OTHER_SHA
        expected = "not bound to candidate CI/head"
    else:
        environment[0]["run_id"] = CI_RUN + 1
        expected = "not bound to candidate CI/head"
    with pytest.raises(ReleaseAttestationError, match=expected):
        _write_and_load(tmp_path, payload)


@pytest.mark.parametrize("surface", ["checkpoint", "supply_chain"])
def test_candidate_requires_checkpoint_and_supply_chain_artifacts(
    tmp_path: Path,
    surface: str,
) -> None:
    payload = _gated_payload()
    if surface == "checkpoint":
        payload["checkpoint_artifacts"] = payload["checkpoint_artifacts"][:1]
        expected = "missing checkpoint evidence"
    else:
        payload["supply_chain_artifacts"] = payload["supply_chain_artifacts"][:1]
        expected = "missing supply-chain evidence"
    with pytest.raises(ReleaseAttestationError, match=expected):
        _write_and_load(tmp_path, payload)


def test_candidate_rejects_stale_artifact_producer(tmp_path: Path) -> None:
    payload = _gated_payload()
    checkpoints = payload["checkpoint_artifacts"]
    assert isinstance(checkpoints, list)
    checkpoints[0]["producer_sha"] = OTHER_SHA
    with pytest.raises(ReleaseAttestationError, match="stale for candidate_sha"):
        _write_and_load(tmp_path, payload)


def test_candidate_manifest_physical_hash_tamper_is_rejected(tmp_path: Path) -> None:
    raw = json.loads(
        (REPO_ROOT / "configs/releases/s0_release_attestation.prepared.json").read_text(encoding="utf-8")
    )
    raw["candidate_manifest"]["sha256"] = "0" * 64
    raw["attestation_sha256"] = compute_attestation_sha256(raw)
    path = tmp_path / "manifest-hash-tamper.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    attestation = load_release_attestation(path)
    with pytest.raises(ReleaseAttestationError, match="artifact SHA-256 mismatch"):
        validate_release_attestation(attestation, repo_root=REPO_ROOT)


@pytest.mark.parametrize("field", ["file_sha256", "index_sha256"])
def test_dependency_lock_tamper_is_rejected(tmp_path: Path, field: str) -> None:
    raw = json.loads(
        (REPO_ROOT / "configs/releases/s0_release_attestation.prepared.json").read_text(encoding="utf-8")
    )
    raw["dependency_lock"][field] = "0" * 64
    raw["attestation_sha256"] = compute_attestation_sha256(raw)
    path = tmp_path / f"lock-{field}.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    attestation = load_release_attestation(path)
    expected = "physical SHA-256 mismatch" if field == "file_sha256" else "semantic index"
    with pytest.raises(ReleaseAttestationError, match=expected):
        validate_release_attestation(attestation, repo_root=REPO_ROOT)


def test_release_attestation_cannot_override_candidate_manifest_status(tmp_path: Path) -> None:
    payload = _gated_payload()
    payload["checkpoint_artifacts"] = [
        _repo_binding("checkpoint_manifest", "README.md"),
        _repo_binding("model_weights", "docs/DEPENDENCY_LOCKS.md"),
    ]
    payload["supply_chain_artifacts"] = [
        _repo_binding("sbom", "requirements/locks/index.json"),
        _repo_binding("dependency_report", "pyproject.toml"),
    ]
    attestation = _write_and_load(tmp_path, payload)
    with pytest.raises(ReleaseAttestationError, match="status differs"):
        validate_release_attestation(attestation, repo_root=REPO_ROOT)


def test_stale_audit_predating_candidate_ci_is_rejected() -> None:
    components = tuple(
        ComponentRef(
            lane=f"D0{index}",
            source_sha=BASE_SHA,
            disposition=ComponentDisposition.ACCEPTED,
            component_kind="infrastructure",
            ci_evidence=CIEvidence(
                run_id=index,
                head_sha=BASE_SHA,
                conclusion="success",
                evidence_ref=f"issue://D0{index}",
            ),
            contains_behavioral_weights=False,
            contains_foreign_pretrained_weights=False,
        )
        for index in range(1, 9)
    )
    stage_manifest = StageCandidateManifest.compose(
        stage="S0",
        integration_anchor_sha=BASE_SHA,
        status=CandidateStatus.AUDITED_CANDIDATE,
        base_lineage=True,
        components=components,
        candidate_sha=BASE_SHA,
        audit_a=AuditEvidence(
            auditor_id="AUDIT-A",
            verdict=AuditVerdict.PASS,
            candidate_sha=BASE_SHA,
            cutoff_utc="2026-08-24T15:02:00Z",
            evidence_ref="issue://13/audit-a",
        ),
        audit_b=AuditEvidence(
            auditor_id="AUDIT-B",
            verdict=AuditVerdict.PASS_WITH_NOTES,
            candidate_sha=BASE_SHA,
            cutoff_utc="2026-08-24T15:02:30Z",
            evidence_ref="issue://14/audit-b",
        ),
    )
    candidate_ci = CandidateCIEvidence(
        run_id=CI_RUN,
        head_sha=BASE_SHA,
        conclusion="success",
        completed_at_utc="2026-08-24T15:03:00Z",
        evidence_ref="issue://candidate-ci",
    )
    with pytest.raises(ReleaseAttestationError, match="predates exact candidate"):
        validate_audit_freshness(stage_manifest, candidate_ci)


def test_stable_requires_external_promotion_authority(tmp_path: Path) -> None:
    payload = _gated_payload(status="stable")
    payload["release_artifact"] = _binding(
        "release_bundle",
        path="release/s0.tar.zst",
        sha256="5" * 64,
    )
    with pytest.raises(ReleaseAttestationError, match="external promotion_authority_ref"):
        _write_and_load(tmp_path, payload)


def test_exact_physical_repository_identity_is_required(tmp_path: Path) -> None:
    payload = _gated_payload()
    payload["repository"] = "Oleksii-debug/12-6-ai"
    with pytest.raises(ReleaseAttestationError, match="physical identity"):
        _write_and_load(tmp_path, payload)
