from __future__ import annotations

import json

import pytest

import tools.run_d03_languk_g05_diagnostic_v1 as diagnostic


def test_numeric_summary_is_deterministic_and_bounded() -> None:
    assert diagnostic.numeric_summary([4, 1, 3, 2]) == {
        "count": 4,
        "min": 1,
        "median": 2.5,
        "max": 4,
    }
    assert diagnostic.numeric_summary([0.3333333333, 0.1]) == {
        "count": 2,
        "min": 0.1,
        "median": 0.216667,
        "max": 0.333333,
    }
    with pytest.raises(
        diagnostic.LangUkG05DiagnosticError,
        match="empty numeric",
    ):
        diagnostic.numeric_summary([])


def test_aggregate_decisions_emits_no_ids_or_text() -> None:
    repetitive = "слово " * 120
    diverse = " ".join(
        f"термін{index} абзац{index} рішення{index}"
        for index in range(80)
    )
    decisions = [
        diagnostic.assess_document("secret-record-a", repetitive, "uk"),
        diagnostic.assess_document("secret-record-b", diverse, "uk"),
    ]

    first = diagnostic.aggregate_decisions(decisions)
    second = diagnostic.aggregate_decisions(decisions)
    assert first == second
    assert first["documents"] == 2
    assert first["accepted_documents"] + first["rejected_documents"] == 2
    assert first["reason_counts"]["low_token_diversity"] >= 1
    assert first["threshold_crossings"]["below_distinct_token_ratio"] >= 1

    durable = json.dumps(first, ensure_ascii=False, sort_keys=True)
    assert "secret-record-a" not in durable
    assert "secret-record-b" not in durable
    assert repetitive not in durable
    assert diverse not in durable


def test_aggregate_reason_sets_are_counts_not_exemplars() -> None:
    first_id = "SECRET_RECORD_ALPHA_987654321"
    second_id = "SECRET_RECORD_BETA_123456789"
    decisions = [
        diagnostic.assess_document(first_id, "повтор " * 100, "uk"),
        diagnostic.assess_document(second_id, "повтор " * 100, "uk"),
    ]
    result = diagnostic.aggregate_decisions(decisions)
    assert sum(result["reason_set_counts"].values()) == 2
    assert all(isinstance(value, int) for value in result["reason_set_counts"].values())
    durable = json.dumps(result, ensure_ascii=False, sort_keys=True)
    assert first_id not in durable
    assert second_id not in durable


def test_authority_constants_are_fail_closed() -> None:
    assert diagnostic.OWNER_HEAD == "7cda697ac4fc8330754b28f0a02e119e395dbd25"
    assert diagnostic.EXPECTED_RECORDS == 256
    assert diagnostic.EXPECTED_SELECTED_MANIFEST == (
        "bbcbd9c2235234007737128c37f375ba0802f32a6dcc45bac02aff30fe4c3fc3"
    )
    assert diagnostic.EXPECTED_QUALITY_EXECUTION == (
        "537b302bfe32314fbdcfe284094c98c817fddd0a9910bb475b1c2dcdbba1a732"
    )
    assert diagnostic.EXPECTED_THRESHOLD_POLICY == (
        "97b9fe1452b22c6275a27f85524f670253a7f4012377361c4cb007004aeccd1d"
    )
    assert diagnostic.EXPECTED_GRANULARITY_POLICY == (
        "e8685c2c6b265b9b289ded7a5245888d8d16ae4d6e881f6229f3bc777601f857"
    )
    assert diagnostic._TRUTH["training_authorized_bytes"] == 0
    assert diagnostic._TRUTH["tokenizer_fit_authorized"] is False
    assert diagnostic._TRUTH["final_test_outcomes_read"] is False
    assert diagnostic._TRUTH["paid_compute_used"] is False


def test_self_hash_round_trip() -> None:
    core = {
        "schema_version": diagnostic.SCHEMA,
        "diagnostic": {"reason_counts": {"low_token_diversity": 256}},
        "truth_boundary": dict(diagnostic._TRUTH),
    }
    document = diagnostic.self_hashed(core, "diagnostic_identity_sha256")
    claimed = document.pop("diagnostic_identity_sha256")
    assert diagnostic.sha256(diagnostic.canonical(document)) == claimed
