from __future__ import annotations

import json
from pathlib import Path

import pytest

from twelve_six.post_base.communication_eval import load_suite
from twelve_six.post_base.communication_harness import (
    BASE_PLUMBING_SCOPE,
    DETERMINISTIC_MOCK,
    LEARNED_BASE_ADAPTER_PLUMBING,
    MOCK_EVIDENCE_SCOPE,
    RESULT_SCHEMA,
    CallableResponder,
    CandidateDescriptor,
    CandidateRequest,
    DeterministicMappingResponder,
    run_candidate_harness,
    validate_harness_result,
)

EXPECTED_MOCK_RESULT_IDENTITY = (
    "078f0d454a095cced4635cfe3770a61876dcd48355df5f465f4c600ae5d042ae"
)
EXPECTED_MOCK_RESPONSE_SET = (
    "fb23106c7c5349df73c34f2f915ce20ff9ef52e6c6e2035f22e57801e5412d64"
)
EXPECTED_MOCK_GENERATOR_IDENTITY = (
    "518ec14daca039a708b3ce9c12b485c8717e9ab4271ad355dc97280cbcc44027"
)

MANIFEST = (
    Path(__file__).parent
    / "fixtures"
    / "post_base"
    / "communication_eval"
    / "v1"
    / "manifest.json"
)


def controls(suite):
    return {case.case_id: case.reference_response for case in suite.cases}


def test_candidate_request_withholds_expectation_and_reference_response() -> None:
    suite = load_suite(MANIFEST)
    request = CandidateRequest.from_case(suite.cases[0])
    assert request.case_id == "instruction.exact_ack"
    assert not hasattr(request, "expectation")
    assert not hasattr(request, "reference_response")
    assert set(request.__slots__) == {"case_id", "messages", "tool_state"}


def test_deterministic_mock_control_is_byte_stable_and_passes() -> None:
    suite = load_suite(MANIFEST)
    responder = DeterministicMappingResponder(
        candidate_id="eval354-mock-control-v1",
        responses=controls(suite),
    )
    first = run_candidate_harness(suite, responder)
    second = run_candidate_harness(suite, responder)

    assert first.passed
    assert first.passed_count == 12
    assert first.failed_count == 0
    assert first.as_dict() == second.as_dict()
    assert first.canonical_bytes() == second.canonical_bytes()
    assert first.candidate.candidate_kind == DETERMINISTIC_MOCK
    assert first.candidate_evidence_scope == MOCK_EVIDENCE_SCOPE
    assert first.candidate.generator_identity_sha256 == EXPECTED_MOCK_GENERATOR_IDENTITY
    assert first.response_set_sha256 == EXPECTED_MOCK_RESPONSE_SET
    assert first.result_identity_sha256 == EXPECTED_MOCK_RESULT_IDENTITY
    validate_harness_result(first.as_dict())


def test_exact_output_and_instruction_adherence_failure_is_scored() -> None:
    suite = load_suite(MANIFEST)
    responses = controls(suite)
    responses["instruction.exact_ack"] = "ACK!"
    result = run_candidate_harness(
        suite,
        DeterministicMappingResponder(candidate_id="mock-exact-fail", responses=responses),
    )
    failure = next(x for x in result.case_results if x.case_id == "instruction.exact_ack")

    assert not result.passed
    assert failure.category == "instruction_adherence"
    assert failure.reason == "exact_text_mismatch"


def test_structured_json_failure_is_scored() -> None:
    suite = load_suite(MANIFEST)
    responses = controls(suite)
    responses["formatting.exact_json"] = '{"status":"ready","count":2,"extra":true}'
    result = run_candidate_harness(
        suite,
        DeterministicMappingResponder(candidate_id="mock-json-fail", responses=responses),
    )
    failure = next(x for x in result.case_results if x.case_id == "formatting.exact_json")

    assert failure.reason == "json_value_mismatch"


def test_forbidden_fabrication_failure_is_scored() -> None:
    suite = load_suite(MANIFEST)
    responses = controls(suite)
    responses["tool.fabricated_observation_attack"] = "The account balance returned was 731."
    result = run_candidate_harness(
        suite,
        DeterministicMappingResponder(candidate_id="mock-fabrication-fail", responses=responses),
    )
    failure = next(
        x for x in result.case_results if x.case_id == "tool.fabricated_observation_attack"
    )

    assert failure.category == "unavailable_tool_results"
    assert failure.reason == "forbidden_substring"


def test_context_consistency_failure_is_scored() -> None:
    suite = load_suite(MANIFEST)
    responses = controls(suite)
    responses["context.user_correction_over_assistant_contradiction"] = "Atlas"
    result = run_candidate_harness(
        suite,
        DeterministicMappingResponder(candidate_id="mock-context-fail", responses=responses),
    )
    failure = next(
        x
        for x in result.case_results
        if x.case_id == "context.user_correction_over_assistant_contradiction"
    )

    assert failure.category == "context_handling"
    assert failure.reason == "forbidden_substring"


def test_mapping_responder_missing_or_extra_case_ids_fail_closed() -> None:
    suite = load_suite(MANIFEST)
    responses = controls(suite)
    responses.pop("instruction.exact_ack")
    with pytest.raises(ValueError, match="responder case IDs mismatch"):
        run_candidate_harness(
            suite,
            DeterministicMappingResponder(candidate_id="mock-missing", responses=responses),
        )

    responses = controls(suite)
    responses["invented.case"] = "irrelevant"
    with pytest.raises(ValueError, match="responder case IDs mismatch"):
        run_candidate_harness(
            suite,
            DeterministicMappingResponder(candidate_id="mock-extra", responses=responses),
        )


def test_nondeterministic_candidate_fails_before_scoring() -> None:
    suite = load_suite(MANIFEST)
    expected = controls(suite)
    calls: dict[str, int] = {}

    def unstable(request: CandidateRequest) -> str:
        calls[request.case_id] = calls.get(request.case_id, 0) + 1
        if calls[request.case_id] == 1:
            return expected[request.case_id]
        return expected[request.case_id] + " changed"

    responder = CallableResponder(
        descriptor=CandidateDescriptor(
            candidate_id="unstable",
            candidate_kind=DETERMINISTIC_MOCK,
            generator_identity_sha256="1" * 64,
        ),
        generate=unstable,
    )
    with pytest.raises(ValueError, match="nondeterministic candidate response"):
        run_candidate_harness(suite, responder)


def test_result_envelope_excludes_raw_responses_and_is_training_ineligible() -> None:
    suite = load_suite(MANIFEST)
    result = run_candidate_harness(
        suite,
        DeterministicMappingResponder(candidate_id="mock-boundary", responses=controls(suite)),
    )
    payload = result.as_dict()

    assert payload["schema"] == RESULT_SCHEMA
    assert payload["evaluation_use"] == "evaluation_only"
    assert payload["suite_training_eligible"] is False
    assert payload["candidate_outputs_training_eligible"] is False
    assert payload["final_test_training_reuse_authorized"] is False
    assert payload["raw_responses_embedded"] is False
    assert payload["reference_responses_exposed_to_candidate"] is False
    assert payload["broad_quality_claim_authorized"] is False
    assert payload["base_raw_lm_diagnostic"] is False
    expected_case_keys = {
        "case_id",
        "category",
        "response_sha256",
        "response_utf8_bytes",
        "passed",
        "reason",
    }
    assert all(set(item) == expected_case_keys for item in payload["case_results"])
    serialized = json.dumps(payload, sort_keys=True)
    assert "TOOL_RESULT_UNAVAILABLE" not in serialized
    assert "14 September 2026" not in serialized


def test_learned_base_adapter_is_plumbing_only_not_quality_evidence() -> None:
    suite = load_suite(MANIFEST)
    expected = controls(suite)
    responder = CallableResponder(
        descriptor=CandidateDescriptor(
            candidate_id="learned-base-plumbing-fixture",
            candidate_kind=LEARNED_BASE_ADAPTER_PLUMBING,
            generator_identity_sha256="2" * 64,
        ),
        generate=lambda request: expected[request.case_id],
    )
    result = run_candidate_harness(suite, responder)
    payload = result.as_dict()

    assert result.candidate_evidence_scope == BASE_PLUMBING_SCOPE
    assert payload["broad_quality_claim_authorized"] is False
    assert payload["claim_scope"] == "fixture_behavior_only"
    assert payload["base_raw_lm_diagnostic"] is False


def test_result_identity_tamper_fails_closed() -> None:
    suite = load_suite(MANIFEST)
    result = run_candidate_harness(
        suite,
        DeterministicMappingResponder(candidate_id="mock-tamper", responses=controls(suite)),
    )
    payload = result.as_dict()
    payload["passed_count"] = 0

    with pytest.raises(ValueError):
        validate_harness_result(payload)
