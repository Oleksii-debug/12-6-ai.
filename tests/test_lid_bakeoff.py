from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from twelve_six.lid_bakeoff import (
    LIDBakeoffError,
    git_blob_sha1,
    preflight_report,
    score_evidence,
    validate_contract,
)


ROOT = Path(__file__).parents[1]
CONTRACT_PATH = ROOT / "configs/research/lid_bakeoff_v1.json"
FIXTURE_PATH = ROOT / "tests/fixtures/lid_bakeoff_v1.jsonl"


def _production_contract() -> dict:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _synthetic_registry() -> bytes:
    contract = _production_contract()
    components = [deepcopy(candidate) for candidate in contract["comparison_candidates"]]
    components.append(
        {
            "id": "NLLB_LID218E",
            "kind": "language_id",
            "upstream": "https://github.com/facebookresearch/fairseq/tree/nllb",
            "license": "CC-BY-NC-4.0 model family context",
            "decision": "DO_NOT_USE_AS_HIDDEN_UNRESTRICTED_DEPENDENCY",
        }
    )
    return json.dumps(
        {
            "schema_version": 2,
            "registry_id": "OPEN-SOURCE-REUSE-REGISTRY-V2",
            "components": components,
        },
        sort_keys=True,
    ).encode()


def _test_contract(registry_bytes: bytes) -> dict:
    contract = _production_contract()
    contract["authority"]["open_source_registry_blob_sha1"] = git_blob_sha1(registry_bytes)
    return contract


def _perfect_evidence(contract: dict, registry_bytes: bytes, fixture_bytes: bytes) -> dict:
    fixture = [json.loads(line) for line in fixture_bytes.decode().splitlines() if line]
    executions = []
    for candidate in contract["comparison_candidates"]:
        executions.append(
            {
                "candidate_id": candidate["id"],
                "executed": True,
                "runtime_identity": {
                    "upstream_ref": "test-ref",
                    "artifact_identity": "sha256:test-artifact",
                    "adapter_identity": "test-adapter-v1",
                    "command_identity": "sha256:test-command",
                },
                "license_review": {
                    "status": "REVIEWED_FOR_BAKEOFF",
                    "reference": "test://license-review",
                },
                "automatic_adoption_requested": False,
                "predictions": [
                    {
                        "case_id": record["case_id"],
                        "predicted_label": record["expected_label"],
                        "raw_label": record["expected_label"],
                        "confidence": 1.0,
                    }
                    for record in fixture
                ],
            }
        )
    return {
        "schema_version": 1,
        "contract_id": contract["contract_id"],
        "fixture_sha256": contract["fixture"]["sha256"],
        "registry_blob_sha1": git_blob_sha1(registry_bytes),
        "evidence_kind": "TEST_FIXTURE_ONLY",
        "executions": executions,
    }


def test_checked_in_contract_binds_live_registry_and_truth_boundary() -> None:
    contract = _production_contract()
    assert contract["authority"]["open_source_registry_blob_sha1"] == (
        "d80a60357c56eacac135f948b8a72556bb849e5a"
    )
    assert [item["id"] for item in contract["comparison_candidates"]] == [
        "FASTTEXT_LID176",
        "OPENLID_V3",
        "GLOTLID",
        "LINGUA",
    ]
    assert contract["excluded_candidates"][0]["id"] == "NLLB_LID218E"
    assert contract["excluded_candidates"][0]["unrestricted_adoption_allowed"] is False
    assert contract["automatic_adoption_allowed"] is False
    assert all(value is False for value in contract["truth_boundaries"].values())


def test_live_repository_registry_binding() -> None:
    registry_path = ROOT / "configs/research/open_source_reuse_registry_v2.json"
    if not registry_path.exists():
        pytest.skip("isolated local fixture does not contain the live repository registry")
    validate_contract(
        _production_contract(),
        registry_path.read_bytes(),
        FIXTURE_PATH.read_bytes(),
    )


def test_preflight_validates_fixture_and_registry_contract() -> None:
    registry = _synthetic_registry()
    contract = _test_contract(registry)
    fixture = FIXTURE_PATH.read_bytes()
    report = preflight_report(contract, registry, fixture)
    assert report["status"] == "PREPARED_NOT_EXECUTED"
    assert report["case_count"] == 20
    assert report["external_lid_model_executed"] is False


def test_perfect_test_evidence_scores_without_adopting() -> None:
    registry = _synthetic_registry()
    contract = _test_contract(registry)
    fixture = FIXTURE_PATH.read_bytes()
    evidence = _perfect_evidence(contract, registry, fixture)
    report = score_evidence(
        contract,
        registry,
        fixture,
        evidence,
        allow_test_evidence=True,
    )
    assert report["comparison_status"] == "COMPARABLE_EVIDENCE_READY"
    assert report["automatic_adoption_allowed"] is False
    assert report["selected_candidate"] is None
    assert report["scientific_verdict"] == "NOT_ADOPTED_REQUIRES_D03_REVIEW"
    assert all(score["accuracy"] == 1.0 for score in report["scores"])


def test_missing_prediction_fails_closed() -> None:
    registry = _synthetic_registry()
    contract = _test_contract(registry)
    fixture = FIXTURE_PATH.read_bytes()
    evidence = _perfect_evidence(contract, registry, fixture)
    evidence["executions"][0]["predictions"].pop()
    with pytest.raises(LIDBakeoffError, match="missing predictions"):
        score_evidence(contract, registry, fixture, evidence, allow_test_evidence=True)


def test_duplicate_prediction_fails_closed() -> None:
    registry = _synthetic_registry()
    contract = _test_contract(registry)
    fixture = FIXTURE_PATH.read_bytes()
    evidence = _perfect_evidence(contract, registry, fixture)
    predictions = evidence["executions"][1]["predictions"]
    predictions.append(deepcopy(predictions[0]))
    with pytest.raises(LIDBakeoffError, match="duplicate prediction"):
        score_evidence(contract, registry, fixture, evidence, allow_test_evidence=True)


def test_nonfinite_confidence_fails_closed() -> None:
    registry = _synthetic_registry()
    contract = _test_contract(registry)
    fixture = FIXTURE_PATH.read_bytes()
    evidence = _perfect_evidence(contract, registry, fixture)
    evidence["executions"][2]["predictions"][0]["confidence"] = float("nan")
    with pytest.raises(LIDBakeoffError, match="invalid confidence"):
        score_evidence(contract, registry, fixture, evidence, allow_test_evidence=True)


def test_runtime_identity_and_license_review_are_mandatory() -> None:
    registry = _synthetic_registry()
    contract = _test_contract(registry)
    fixture = FIXTURE_PATH.read_bytes()
    evidence = _perfect_evidence(contract, registry, fixture)
    del evidence["executions"][0]["runtime_identity"]["artifact_identity"]
    with pytest.raises(LIDBakeoffError, match="artifact_identity missing"):
        score_evidence(contract, registry, fixture, evidence, allow_test_evidence=True)

    evidence = _perfect_evidence(contract, registry, fixture)
    evidence["executions"][0]["license_review"]["status"] = "PENDING"
    with pytest.raises(LIDBakeoffError, match="license review incomplete"):
        score_evidence(contract, registry, fixture, evidence, allow_test_evidence=True)


def test_test_evidence_cannot_masquerade_as_external_execution() -> None:
    registry = _synthetic_registry()
    contract = _test_contract(registry)
    fixture = FIXTURE_PATH.read_bytes()
    evidence = _perfect_evidence(contract, registry, fixture)
    with pytest.raises(LIDBakeoffError, match="EXTERNAL_CANDIDATE_EXECUTION"):
        score_evidence(contract, registry, fixture, evidence)


def test_registry_drift_and_nllb_policy_drift_fail_closed() -> None:
    registry = _synthetic_registry()
    contract = _test_contract(registry)
    fixture = FIXTURE_PATH.read_bytes()
    validate_contract(contract, registry, fixture)

    drifted = registry + b"\n"
    with pytest.raises(LIDBakeoffError, match="registry Git blob identity drift"):
        validate_contract(contract, drifted, fixture)

    registry_obj = json.loads(registry)
    for component in registry_obj["components"]:
        if component["id"] == "NLLB_LID218E":
            component["decision"] = "P0_LID_BAKEOFF"
    altered = json.dumps(registry_obj, sort_keys=True).encode()
    altered_contract = _test_contract(altered)
    with pytest.raises(LIDBakeoffError, match="NLLB unrestricted-dependency exclusion drift"):
        validate_contract(altered_contract, altered, fixture)


def test_fixture_training_or_category_drift_fails_closed() -> None:
    registry = _synthetic_registry()
    contract = _test_contract(registry)
    fixture_records = [
        json.loads(line) for line in FIXTURE_PATH.read_text(encoding="utf-8").splitlines() if line
    ]
    fixture_records[0]["training_allowed"] = True
    rendered = "\n".join(json.dumps(item, ensure_ascii=False) for item in fixture_records) + "\n"
    modified = rendered.encode()
    contract["fixture"]["sha256"] = __import__("hashlib").sha256(modified).hexdigest()
    with pytest.raises(LIDBakeoffError, match="cannot allow training"):
        validate_contract(contract, registry, modified)
