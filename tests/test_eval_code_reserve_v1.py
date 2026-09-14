from __future__ import annotations

import copy
import importlib.util
import json
import os
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools/validate_eval_code_reserve_v1.py"
MATERIALIZER = ROOT / "tools/materialize_eval_code_reserve_v1.py"
MANIFEST = ROOT / "configs/evaluation/eval_code_reserve_v1.json"
EVIDENCE = ROOT / "evidence/eval647/code_selection_source_materialization_v1.json"

validator_spec = importlib.util.spec_from_file_location("eval647_validator", VALIDATOR)
assert validator_spec is not None and validator_spec.loader is not None
validator = importlib.util.module_from_spec(validator_spec)
validator_spec.loader.exec_module(validator)

materializer_spec = importlib.util.spec_from_file_location("eval647_materializer", MATERIALIZER)
assert materializer_spec is not None and materializer_spec.loader is not None
materializer = importlib.util.module_from_spec(materializer_spec)
materializer_spec.loader.exec_module(materializer)


def _manifest() -> dict:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def _evidence() -> dict:
    return json.loads(EVIDENCE.read_text(encoding="utf-8"))


def test_contract_is_source_sealed_but_not_authorized() -> None:
    result = validator.validate_document(_manifest())
    assert result["reserved_objects"] == 2
    assert result["independent_families"] == 2
    assert result["selection_validation_records_authorized"] == 0
    assert result["status"] == "EXACT_RAW_OBJECTS_SEALED_PENDING_PROJECT_OVERLAP_AUDIT"


def test_committed_source_evidence_binds_sealed_contract() -> None:
    validator.validate_materialization_evidence(_manifest(), _evidence())


def test_training_promotion_fails_closed() -> None:
    mutated = copy.deepcopy(_manifest())
    mutated["objects"][0]["training_allowed"] = True
    with pytest.raises(ValueError):
        validator.validate_document(mutated)


def test_final_test_access_fails_closed() -> None:
    mutated = copy.deepcopy(_manifest())
    mutated["reservation"]["final_test_payload_access_allowed"] = True
    with pytest.raises(ValueError):
        validator.validate_document(mutated)


def test_blob_identity_mutation_fails_closed() -> None:
    mutated = copy.deepcopy(_manifest())
    mutated["objects"][1]["git_blob_sha1"] = "0" * 40
    with pytest.raises(ValueError):
        validator.validate_document(mutated)


def test_raw_sha_mutation_fails_closed() -> None:
    mutated = copy.deepcopy(_manifest())
    mutated["objects"][0]["raw_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validator.validate_document(mutated)


def test_source_evidence_tamper_fails_closed() -> None:
    mutated = copy.deepcopy(_evidence())
    mutated["objects"][0]["raw_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validator.validate_materialization_evidence(_manifest(), mutated)


def test_license_evidence_tamper_fails_closed() -> None:
    mutated = copy.deepcopy(_evidence())
    mutated["objects"][0]["license_raw_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        validator.validate_materialization_evidence(_manifest(), mutated)


def test_object_set_identity_tamper_fails_closed() -> None:
    mutated = copy.deepcopy(_evidence())
    mutated["object_set_identity_sha256"] = "0" * 64
    body = copy.deepcopy(mutated)
    body.pop("evidence_identity_sha256")
    mutated["evidence_identity_sha256"] = validator.hashlib.sha256(
        validator._canonical_bytes(body)
    ).hexdigest()
    original_identity = validator.EXPECTED_EVIDENCE_IDENTITY
    validator.EXPECTED_EVIDENCE_IDENTITY = mutated["evidence_identity_sha256"]
    try:
        with pytest.raises(ValueError, match="object-set identity drift"):
            validator.validate_materialization_evidence(_manifest(), mutated)
    finally:
        validator.EXPECTED_EVIDENCE_IDENTITY = original_identity


def test_wrong_source_bytes_fail_before_credit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(materializer, "_fetch", lambda *_: b"wrong")
    with pytest.raises(RuntimeError, match="raw byte-size drift"):
        materializer.materialize(_manifest())


@pytest.mark.skipif(os.environ.get("GITHUB_ACTIONS") != "true", reason="needs GitHub network")
def test_live_pinned_source_materialization_is_deterministic() -> None:
    first = materializer.materialize(_manifest())
    second = materializer.materialize(_manifest())
    assert first == second
    assert first["objects"] == _evidence()["objects"]
    assert first["object_set_identity_sha256"] == _evidence()["object_set_identity_sha256"]
    assert first["reserved_object_count"] == 2
    assert first["independent_family_count"] == 2
    assert first["raw_payload_persisted_in_repository"] is False
    assert first["selection_validation_records_authorized"] == 0
    discovered = {
        item["repository"]: {
            "raw_bytes": item["raw_bytes"],
            "raw_sha256": item["raw_sha256"],
            "git_blob_sha1": item["git_blob_sha1"],
            "license_raw_sha256": item["license_raw_sha256"],
            "license_git_blob_sha1": item["license_git_blob_sha1"],
        }
        for item in first["objects"]
    }
    warnings.warn(
        "EVAL647_LIVE_SOURCE_DISCOVERY=" + json.dumps(discovered, sort_keys=True),
        stacklevel=1,
    )
