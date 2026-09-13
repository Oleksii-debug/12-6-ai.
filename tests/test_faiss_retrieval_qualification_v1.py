from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from twelve_six.faiss_retrieval_qualification import (
    QualificationError,
    brute_force_search,
    build_evidence,
    probe_faiss,
    validate_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "configs/research/faiss_retrieval_qualification_v1.json"


def load_contract() -> dict:
    return json.loads(CONTRACT.read_text(encoding="utf-8"))


def assert_rejected(mutator) -> None:
    contract = load_contract()
    mutator(contract)
    with pytest.raises(QualificationError):
        validate_contract(contract)


def test_valid_contract_and_evidence_are_deterministic() -> None:
    contract = load_contract()
    validate_contract(contract)
    first = build_evidence(contract)
    second = build_evidence(copy.deepcopy(contract))
    assert first == second
    assert first["promotion_state"] == "CANDIDATE"
    assert first["parity_proven"] is False
    assert first["canonical_base_changed"] is False
    assert first["training_authorized"] is False


def test_reference_search_matches_preregistered_results() -> None:
    contract = load_contract()
    for query in contract["queries"]:
        rows = brute_force_search(contract, query["vector"], query["top_k"])
        assert [row["id"] for row in rows] == query["expected_record_ids"]


def test_deterministic_tie_break_is_record_id() -> None:
    contract = load_contract()
    contract["records"] = [
        {"id": "b", "vector": [1.0, 0.0, 0.0]},
        {"id": "a", "vector": [1.0, 0.0, 0.0]},
    ]
    contract["queries"] = [
        {"id": "tie", "vector": [1.0, 0.0, 0.0], "top_k": 2, "expected_record_ids": ["a", "b"]}
    ]
    rows = brute_force_search(contract, [1.0, 0.0, 0.0], 2)
    assert [row["id"] for row in rows] == ["a", "b"]


def test_foreign_embedding_model_fails_closed() -> None:
    assert_rejected(lambda c: c["vector_source"].update(foreign_pretrained_model=True))


def test_hidden_embedding_model_field_fails_closed() -> None:
    assert_rejected(lambda c: c["vector_source"].update(embedding_model="foreign/model"))


def test_duplicate_record_ids_fail_closed() -> None:
    def mutate(contract):
        contract["records"][1]["id"] = contract["records"][0]["id"]

    assert_rejected(mutate)


def test_nonfinite_vector_fails_closed() -> None:
    def mutate(contract):
        contract["records"][0]["vector"][0] = math.nan

    assert_rejected(mutate)


def test_dimension_and_metric_drift_fail_closed() -> None:
    assert_rejected(lambda c: c["index"].update(dimension=4))
    assert_rejected(lambda c: c["index"].update(metric="COSINE"))


def test_registry_and_source_authority_drift_fail_closed() -> None:
    assert_rejected(lambda c: c["authority"].update(base_git_sha="0" * 40))
    assert_rejected(lambda c: c["authority"].update(registry_git_blob_sha="1" * 40))
    assert_rejected(lambda c: c["upstream"].update(release_commit="2" * 40))


def test_distribution_identity_drift_fails_closed() -> None:
    assert_rejected(lambda c: c["package"].update(version="1.14.3"))
    assert_rejected(lambda c: c["package"].update(qualified_linux_x86_64_wheel_sha256="3" * 64))


def test_untrusted_loading_fails_closed() -> None:
    assert_rejected(lambda c: c["persistence"].update(allow_untrusted_load=True))


def test_executed_backend_requires_hash_exact_version_and_parity() -> None:
    contract = load_contract()
    contract["backend_execution"] = {
        "status": "EXECUTED_PASS",
        "import_version": "1.15.0",
        "reference_parity": True,
    }
    with pytest.raises(QualificationError):
        validate_contract(contract)
    contract["persistence"]["index_sha256"] = "a" * 64
    validate_contract(contract)
    contract["backend_execution"]["import_version"] = "1.14.1"
    with pytest.raises(QualificationError):
        validate_contract(contract)


def test_tracking_only_self_promotion_fails_closed() -> None:
    assert_rejected(lambda c: c["policy"].update(requested_promotion_state="ADOPTED"))
    assert_rejected(lambda c: c["policy"].update(requested_promotion_state="PARITY_PROVEN"))


def test_local_probe_never_fabricates_parity_when_dependency_absent_or_drifted() -> None:
    result = probe_faiss(load_contract())
    assert result["status"] in {"NOT_EXECUTED_DEPENDENCY_ABSENT", "EXECUTED_FAIL", "EXECUTED_PASS"}
    if result["status"] != "EXECUTED_PASS":
        assert result.get("reference_parity") is not True


def test_material_fixture_identity_drift_changes_evidence_identity() -> None:
    original = load_contract()
    changed = copy.deepcopy(original)
    changed["vector_source"]["identity"] = "SWARM-751-ORIGINAL-DENSE-FIXTURE-V1B"
    assert build_evidence(original)["fixture_sha256"] != build_evidence(changed)["fixture_sha256"]
