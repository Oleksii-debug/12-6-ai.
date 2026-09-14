import json
from pathlib import Path

import pytest

from twelve_six.export_runtime_parity import (
    EVIDENCE_SCHEMA,
    ExportRuntimeEvidenceError,
    assess_parity_evidence,
    compare_numeric_outputs,
    derive_adoption_state,
    load_target_manifest,
    manifest_identity,
    semantic_sha256,
    validate_target_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "configs/research/export_runtime_targets_v1.json"


def _manifest():
    return load_target_manifest(MANIFEST_PATH)


def _evidence(target_id="onnxruntime"):
    manifest = _manifest()
    target = next(item for item in manifest["targets"] if item["id"] == target_id)
    return {
        "schema": EVIDENCE_SCHEMA,
        "target_id": target_id,
        "candidate_upstream_commit_sha": target["upstream_commit_sha"],
        "candidate_release_tag": target["release_tag"],
        "backend_execution": True,
        "project_git_sha": "1" * 40,
        "model_identity": "2" * 64,
        "input_identity": "3" * 64,
        "execution_environment": {
            "python": "3.11.16",
            "platform": "linux-x86_64",
            "backend_version": target["release_tag"],
            "backend_device": "CPU",
        },
        "reference_backend": "twelve_six_first_party",
        "reference_output": [[1.0, -2.0], [3.0, 4.0]],
        "candidate_output": [[1.0 + 1e-7, -2.0], [3.0, 4.0 - 1e-7]],
        "atol": 1e-6,
        "rtol": 1e-6,
    }


def test_checked_in_manifest_is_valid_and_candidate_only():
    manifest = _manifest()
    assert validate_target_manifest(manifest) == manifest["manifest_identity"]
    assert {target["candidate_state"] for target in manifest["targets"]} == {"CANDIDATE"}
    assert all(target["parity_evidence_identity"] is None for target in manifest["targets"])
    assert all(target["adoption_authority"] is None for target in manifest["targets"])


def test_manifest_identity_is_deterministic_and_drift_sensitive():
    manifest = _manifest()
    assert manifest_identity(manifest) == manifest_identity(json.loads(json.dumps(manifest)))
    changed = json.loads(json.dumps(manifest))
    changed["targets"][0]["release_tag"] = "drifted"
    assert manifest_identity(changed) != manifest["manifest_identity"]


def test_numeric_parity_passes_inside_tolerance():
    result = compare_numeric_outputs([1.0, 2.0], [1.000001, 1.999999], atol=2e-6, rtol=0)
    assert result.passed
    assert result.mismatch_count == 0
    assert result.shape == (2,)


def test_numeric_parity_fails_outside_tolerance():
    result = compare_numeric_outputs([1.0, 2.0], [1.1, 2.0], atol=1e-6, rtol=1e-6)
    assert not result.passed
    assert result.mismatch_count == 1


def test_numeric_parity_rejects_shape_ragged_and_nonfinite_evidence():
    with pytest.raises(ExportRuntimeEvidenceError, match="shape mismatch"):
        compare_numeric_outputs([[1.0, 2.0]], [1.0, 2.0], atol=0, rtol=0)
    with pytest.raises(ExportRuntimeEvidenceError, match="ragged"):
        compare_numeric_outputs([[1.0], [2.0, 3.0]], [[1.0], [2.0, 3.0]], atol=0, rtol=0)
    with pytest.raises(ExportRuntimeEvidenceError, match="non-finite"):
        compare_numeric_outputs([1.0], [float("nan")], atol=0, rtol=0)


def test_parity_evidence_produces_deterministic_nonadopted_report():
    evidence = _evidence()
    first = assess_parity_evidence(_manifest(), evidence)
    second = assess_parity_evidence(_manifest(), json.loads(json.dumps(evidence)))
    assert first == second
    assert first["derived_state"] == "PARITY_PROVEN"
    assert first["adopted"] is False
    assert derive_adoption_state(first, None) == "PARITY_PROVEN"
    assert len(first["evidence_identity"]) == 64


def test_parity_evidence_identity_changes_on_material_output_drift():
    evidence = _evidence()
    first = assess_parity_evidence(_manifest(), evidence)
    evidence["candidate_output"][0][0] += 2e-7
    second = assess_parity_evidence(_manifest(), evidence)
    assert first["evidence_identity"] != second["evidence_identity"]
    assert first["candidate_output_identity"] != second["candidate_output_identity"]


def test_missing_backend_execution_cannot_prove_parity():
    evidence = _evidence()
    evidence["backend_execution"] = False
    with pytest.raises(ExportRuntimeEvidenceError, match="cannot prove parity"):
        assess_parity_evidence(_manifest(), evidence)


def test_wrong_backend_pin_or_version_fails_closed():
    evidence = _evidence()
    evidence["candidate_upstream_commit_sha"] = "f" * 40
    with pytest.raises(ExportRuntimeEvidenceError, match="upstream commit"):
        assess_parity_evidence(_manifest(), evidence)

    evidence = _evidence()
    evidence["execution_environment"]["backend_version"] = "other"
    with pytest.raises(ExportRuntimeEvidenceError, match="backend version"):
        assess_parity_evidence(_manifest(), evidence)


def test_failed_parity_cannot_be_adopted():
    evidence = _evidence()
    evidence["candidate_output"][0][0] = 9.0
    report = assess_parity_evidence(_manifest(), evidence)
    assert report["derived_state"] == "CANDIDATE"
    with pytest.raises(ExportRuntimeEvidenceError, match="without parity proof"):
        derive_adoption_state(report, "issue-999")


def test_adoption_requires_separate_authority_even_after_parity():
    report = assess_parity_evidence(_manifest(), _evidence())
    assert derive_adoption_state(report, None) == "PARITY_PROVEN"
    assert derive_adoption_state(report, "issue-999-approved") == "ADOPTED"


def test_fabricated_checked_in_adoption_or_parity_state_is_rejected():
    manifest = _manifest()
    target = manifest["targets"][0]
    target["candidate_state"] = "ADOPTED"
    target["parity_evidence_identity"] = None
    target["adoption_authority"] = "self-asserted"
    manifest["manifest_identity"] = manifest_identity(manifest)
    with pytest.raises(ExportRuntimeEvidenceError, match="parity_evidence_identity"):
        validate_target_manifest(manifest)


def test_semantic_hash_rejects_nan_serialization():
    with pytest.raises(ValueError):
        semantic_sha256({"bad": float("nan")})
