from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from twelve_six.post_base.communication_eval import (
    BASE_EVIDENCE_NAMESPACE,
    CLAIM_SCOPE,
    EVIDENCE_NAMESPACE,
    evaluate_suite,
    load_suite,
    reference_responses,
)

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "post_base" / "communication_eval" / "v1"
MANIFEST = FIXTURE_ROOT / "manifest.json"
EXPECTED_CASES_SHA256 = "65e2d28ef4adb442d636ec286e34ce7b43b21d7d2612325d1ca96e310863be8f"
EXPECTED_SUITE_IDENTITY_SHA256 = (
    "7a94f1c2dd9cb31a571dd8383ed1994936abc6e6003b0b53829778ae19ba2ba7"
)


def test_suite_is_immutable_project_authored_and_post_base_only() -> None:
    suite = load_suite(MANIFEST)

    assert suite.suite_id == "eval354-communication-suite-v1"
    assert suite.version == 1
    assert suite.cases_sha256 == EXPECTED_CASES_SHA256
    assert suite.identity_sha256 == EXPECTED_SUITE_IDENTITY_SHA256
    assert len(suite.cases) == 12
    assert dict(suite.category_counts) == {
        "context_handling": 3,
        "dialogue_consistency": 2,
        "formatting": 2,
        "instruction_adherence": 2,
        "unavailable_tool_results": 3,
    }
    assert EVIDENCE_NAMESPACE == "evidence/post_base/eval354"
    assert BASE_EVIDENCE_NAMESPACE == "evidence/base"
    assert EVIDENCE_NAMESPACE != BASE_EVIDENCE_NAMESPACE


def test_manifest_disallows_training_foreign_output_and_broad_claims() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))

    assert manifest["separation"]["training_eligible"] is False
    assert manifest["separation"]["base_raw_lm_diagnostics"] is False
    assert manifest["provenance"] == {
        "foreign_model_output": False,
        "origin": "PROJECT_AUTHORED",
    }
    assert manifest["claims"] == {
        "broad_intelligence_claim_authorized": False,
        "scope": "fixture_behavior_only",
    }


def test_project_authored_reference_responses_only_validate_scorer_mechanics() -> None:
    suite = load_suite(MANIFEST)
    result = evaluate_suite(suite, reference_responses(suite))

    assert result.passed
    assert result.passed_count == 12
    assert result.failed_count == 0
    assert result.claim_scope == CLAIM_SCOPE == "fixture_behavior_only"
    assert result.broad_intelligence_claim is False
    assert result.base_raw_lm_diagnostic is False


def test_scorer_is_deterministic_for_identical_bound_inputs() -> None:
    suite = load_suite(MANIFEST)
    responses = reference_responses(suite)

    first = evaluate_suite(suite, responses).as_dict()
    second = evaluate_suite(suite, responses).as_dict()

    assert first == second


def test_fixture_byte_mutation_fails_immutable_hash_gate(tmp_path: Path) -> None:
    copied = tmp_path / "v1"
    shutil.copytree(FIXTURE_ROOT, copied)
    cases = copied / "cases.jsonl"
    cases.write_text(cases.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="immutable manifest hash"):
        load_suite(copied / "manifest.json")


def test_manifest_claim_or_separation_weakening_fails_closed(tmp_path: Path) -> None:
    copied = tmp_path / "v1"
    shutil.copytree(FIXTURE_ROOT, copied)
    manifest_path = copied / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["claims"]["broad_intelligence_claim_authorized"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="claims boundary"):
        load_suite(manifest_path)


def test_missing_or_extra_response_ids_fail_closed() -> None:
    suite = load_suite(MANIFEST)
    responses = reference_responses(suite)
    responses.pop("instruction.exact_ack")

    with pytest.raises(ValueError, match="response case IDs mismatch"):
        evaluate_suite(suite, responses)

    responses = reference_responses(suite)
    responses["invented.case"] = "anything"
    with pytest.raises(ValueError, match="response case IDs mismatch"):
        evaluate_suite(suite, responses)


def test_unavailable_tool_result_invention_is_rejected() -> None:
    suite = load_suite(MANIFEST)
    responses = reference_responses(suite)
    responses["tool.inventory_no_guess"] = "There are 42 items available."
    result = evaluate_suite(suite, responses)

    failure = next(
        item for item in result.case_results if item.case_id == "tool.inventory_no_guess"
    )
    assert not result.passed
    assert not failure.passed
    assert failure.reason == "forbidden_substring"


def test_adversarial_fabricated_tool_observation_is_rejected() -> None:
    suite = load_suite(MANIFEST)
    responses = reference_responses(suite)
    responses["tool.fabricated_observation_attack"] = (
        "The CRM returned account balance 731."
    )
    result = evaluate_suite(suite, responses)

    failure = next(
        item
        for item in result.case_results
        if item.case_id == "tool.fabricated_observation_attack"
    )
    assert not result.passed
    assert not failure.passed
    assert failure.reason == "forbidden_substring"


def test_formatting_extra_json_key_is_rejected() -> None:
    suite = load_suite(MANIFEST)
    responses = reference_responses(suite)
    responses["formatting.exact_json"] = '{"status":"ready","count":2,"note":"extra"}'
    result = evaluate_suite(suite, responses)

    failure = next(
        item for item in result.case_results if item.case_id == "formatting.exact_json"
    )
    assert not failure.passed
    assert failure.reason == "json_value_mismatch"


def test_context_regression_is_rejected() -> None:
    suite = load_suite(MANIFEST)
    responses = reference_responses(suite)
    responses["context.current_over_archive"] = "1 September 2026"
    result = evaluate_suite(suite, responses)

    failure = next(
        item
        for item in result.case_results
        if item.case_id == "context.current_over_archive"
    )
    assert not failure.passed
    assert failure.reason == "forbidden_substring"


def test_adversarial_context_contradiction_is_rejected() -> None:
    suite = load_suite(MANIFEST)
    responses = reference_responses(suite)
    responses["context.user_correction_over_assistant_contradiction"] = "Atlas"
    result = evaluate_suite(suite, responses)

    failure = next(
        item
        for item in result.case_results
        if item.case_id == "context.user_correction_over_assistant_contradiction"
    )
    assert not result.passed
    assert not failure.passed
    assert failure.reason == "forbidden_substring"
