from __future__ import annotations

import copy
import hashlib
import json

import pytest

from twelve_six.learned20m_launch_input import (
    LaunchInputAuthorityError,
    build_launch_input_authority,
    verify_launch_input_authority,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _identity(value: dict, field: str) -> str:
    body = copy.deepcopy(value)
    body.pop(field, None)
    encoded = (
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _rehash(value: dict, field: str) -> None:
    value[field] = _identity(value, field)


def _fixture() -> tuple[dict, dict, dict, dict]:
    stages = {
        "normalization": _sha("normalization"),
        "evaluation_reservations": _sha("evaluation"),
        "dedup": _sha("dedup"),
        "split": _sha("split"),
        "packing": _sha("packing-stage"),
    }
    tokenizer_identity = _sha("tokenizer")
    packing_identity = _sha("packing")
    runtime_identity = _sha("runtime")
    materialization = _sha("materialization")
    proof = {
        "schema_version": "12-6.postpack-two-clean-proof.v2",
        "input_packet_identity_sha256": _sha("input"),
        "terminal_corpus_authority_identity_sha256": _sha("corpus"),
        "stage_bindings": copy.deepcopy(stages),
        "tokenizer_identity_sha256": tokenizer_identity,
        "packing_identity_sha256": packing_identity,
        "runtime_identity_sha256": runtime_identity,
        "fresh_process_count": 2,
        "byte_identical": True,
        "build_a_sha256": _sha("same-output"),
        "build_b_sha256": _sha("same-output"),
        "materialization_identity_sha256": materialization,
        "claim_boundary": {
            "contains_source_text": False,
            "authorizes_training": False,
            "authorizes_paid_compute": False,
            "creates_positive_unique_loss_authority": False,
        },
    }
    proof["proof_identity_sha256"] = _identity(proof, "proof_identity_sha256")

    ledger = {
        "schema_version": "12-6.unique-loss-position-ledger.v2",
        "position_policy": "logical-causal-token-target-postpack-v2",
        "materialization_identity_sha256": materialization,
        "stage_bindings": copy.deepcopy(stages),
        "tokenizer": {
            "name": "byte-v1",
            "identity_sha256": tokenizer_identity,
            "source_bytes_are_loss_positions": False,
        },
        "packing_identity_sha256": packing_identity,
        "complete_one_pass": True,
        "eligible_causal_targets_before_packing": 7,
        "one_pass_unique_nonignored_causal_loss_positions": 7,
        "eligible_targets_not_packed": 0,
        "by_language": {"en": 7},
        "by_modality": {"text": 7},
        "by_family": {"fixture": 7},
        "segments": [
            {
                "segment_identity_sha256": _sha("segment"),
                "document_id": "doc-1",
                "target_start": 1,
                "target_end": 8,
                "pack_id": "pack-1",
                "pack_target_start": 1,
                "pack_target_end": 8,
                "normalized_payload_sha256": _sha("payload"),
                "language": "en",
                "modality": "text",
                "family_id": "fixture",
                "loss_position_count": 7,
            }
        ],
        "padding_loss_positions": 0,
        "cross_document_loss_positions": 0,
        "source_bytes_relabelled_as_loss_positions": False,
    }
    ledger["ledger_identity_sha256"] = _identity(ledger, "ledger_identity_sha256")

    carrier = {
        "repository": "Oleksii-debug/12-6-ai.",
        "git_sha": "a" * 40,
        "modelspec_sha256": _sha("modelspec"),
        "initialization_identity_sha256": _sha("init"),
        "canonical_base": "random_init",
        "foreign_pretrained_weights_used": False,
        "terminal": True,
        "workflow_run_id": 12345,
        "workflow_status": "completed",
        "workflow_conclusion": "success",
        "workflow_head_sha": "a" * 40,
        "evidence_sha256": _sha("carrier-evidence"),
    }
    expected = {
        "expected_two_clean_proof_identity_sha256": proof["proof_identity_sha256"],
        "expected_two_clean_input_packet_identity_sha256": proof[
            "input_packet_identity_sha256"
        ],
        "expected_unique_loss_ledger_identity_sha256": ledger[
            "ledger_identity_sha256"
        ],
        "expected_terminal_corpus_authority_identity_sha256": _sha("corpus"),
        "expected_stage_bindings": copy.deepcopy(stages),
        "expected_tokenizer_identity_sha256": tokenizer_identity,
        "expected_packing_identity_sha256": packing_identity,
        "expected_runtime_identity_sha256": runtime_identity,
        "expected_carrier_git_sha": "a" * 40,
        "expected_modelspec_sha256": _sha("modelspec"),
        "expected_initialization_identity_sha256": _sha("init"),
        "expected_carrier_workflow_run_id": 12345,
        "expected_carrier_evidence_sha256": _sha("carrier-evidence"),
        "requested_unique_loss_positions": 7,
    }
    return proof, ledger, carrier, expected


def _build() -> dict:
    proof, ledger, carrier, expected = _fixture()
    return build_launch_input_authority(proof, ledger, carrier, **expected)


def test_binds_v2_chain_deterministically_without_authorization() -> None:
    first = _build()
    second = _build()
    assert first == second
    assert first["binding_status"] == "READY_FOR_READINESS_BINDING"
    assert first["data_spine"]["one_pass_unique_nonignored_causal_loss_positions"] == 7
    assert first["claim_boundary"]["authorized_optimized_target_exposure"] == 0
    assert first["claim_boundary"]["authorizes_training"] is False
    assert first["claim_boundary"]["authorizes_compute"] is False
    assert first["carrier"]["workflow_head_sha"] == first["carrier"]["git_sha"]
    assert "doc-1" not in json.dumps(first)
    assert "segments" not in json.dumps(first)
    verify_launch_input_authority(first)


def test_launch_authority_tamper_fails_self_hash() -> None:
    authority = _build()
    authority["data_spine"]["requested_unique_loss_positions"] = 6
    with pytest.raises(LaunchInputAuthorityError, match="self-identity mismatch"):
        verify_launch_input_authority(authority)


def test_rehashed_extra_final_test_field_fails_closed_world() -> None:
    authority = _build()
    authority["final_test_result"] = "forbidden"
    _rehash(authority, "authority_identity_sha256")
    with pytest.raises(LaunchInputAuthorityError, match="unexpected or missing fields"):
        verify_launch_input_authority(authority)


def test_rehashed_extra_candidate_text_fails_closed_world() -> None:
    authority = _build()
    authority["data_spine"]["candidate_text"] = "forbidden"
    _rehash(authority, "authority_identity_sha256")
    with pytest.raises(LaunchInputAuthorityError, match="data spine"):
        verify_launch_input_authority(authority)


def test_v1_two_clean_proof_is_rejected() -> None:
    proof, ledger, carrier, expected = _fixture()
    proof["schema_version"] = "12-6.postpack-two-clean-proof.v1"
    _rehash(proof, "proof_identity_sha256")
    expected["expected_two_clean_proof_identity_sha256"] = proof["proof_identity_sha256"]
    with pytest.raises(LaunchInputAuthorityError, match="canonical two-clean proof rejected"):
        build_launch_input_authority(proof, ledger, carrier, **expected)


def test_v2_two_clean_proof_rejects_rehashed_extra_field() -> None:
    proof, ledger, carrier, expected = _fixture()
    proof["candidate_text"] = "forbidden"
    _rehash(proof, "proof_identity_sha256")
    expected["expected_two_clean_proof_identity_sha256"] = proof["proof_identity_sha256"]
    with pytest.raises(LaunchInputAuthorityError, match="unexpected or missing fields"):
        build_launch_input_authority(proof, ledger, carrier, **expected)


@pytest.mark.parametrize(
    ("expected_field", "replacement", "message"),
    [
        (
            "expected_two_clean_input_packet_identity_sha256",
            _sha("other-input"),
            "input_packet_identity_sha256 mismatch",
        ),
        (
            "expected_runtime_identity_sha256",
            _sha("other-runtime"),
            "runtime_identity_sha256 mismatch",
        ),
        (
            "expected_tokenizer_identity_sha256",
            _sha("other-tokenizer"),
            "tokenizer_identity_sha256 mismatch",
        ),
        (
            "expected_packing_identity_sha256",
            _sha("other-packing"),
            "packing_identity_sha256 mismatch",
        ),
    ],
)
def test_v2_proof_identities_are_independently_expected(
    expected_field: str,
    replacement: str,
    message: str,
) -> None:
    proof, ledger, carrier, expected = _fixture()
    expected[expected_field] = replacement
    with pytest.raises(LaunchInputAuthorityError, match=message):
        build_launch_input_authority(proof, ledger, carrier, **expected)


def test_two_clean_build_hash_mismatch_fails() -> None:
    proof, ledger, carrier, expected = _fixture()
    proof["build_b_sha256"] = _sha("different")
    _rehash(proof, "proof_identity_sha256")
    expected["expected_two_clean_proof_identity_sha256"] = proof["proof_identity_sha256"]
    with pytest.raises(LaunchInputAuthorityError, match="build hashes differ"):
        build_launch_input_authority(proof, ledger, carrier, **expected)


def test_ledger_materialization_substitution_fails() -> None:
    proof, ledger, carrier, expected = _fixture()
    ledger["materialization_identity_sha256"] = _sha("other-materialization")
    _rehash(ledger, "ledger_identity_sha256")
    expected["expected_unique_loss_ledger_identity_sha256"] = ledger[
        "ledger_identity_sha256"
    ]
    with pytest.raises(LaunchInputAuthorityError, match="materialization substitution"):
        build_launch_input_authority(proof, ledger, carrier, **expected)


@pytest.mark.parametrize("value", [0, -1, True, 1.5, "7"])
def test_nonpositive_or_nonintegral_unique_capacity_fails(value: object) -> None:
    proof, ledger, carrier, expected = _fixture()
    ledger["one_pass_unique_nonignored_causal_loss_positions"] = value
    _rehash(ledger, "ledger_identity_sha256")
    expected["expected_unique_loss_ledger_identity_sha256"] = ledger[
        "ledger_identity_sha256"
    ]
    with pytest.raises(LaunchInputAuthorityError, match="positive integer"):
        build_launch_input_authority(proof, ledger, carrier, **expected)


def test_requested_unique_exposure_cannot_exceed_supply() -> None:
    proof, ledger, carrier, expected = _fixture()
    expected["requested_unique_loss_positions"] = 8
    with pytest.raises(LaunchInputAuthorityError, match="exceeds one-pass unique capacity"):
        build_launch_input_authority(proof, ledger, carrier, **expected)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("padding_loss_positions", 1, "padding"),
        ("cross_document_loss_positions", 1, "cross-document"),
    ],
)
def test_nonunique_positions_cannot_manufacture_capacity(
    field: str, value: int, message: str
) -> None:
    proof, ledger, carrier, expected = _fixture()
    ledger[field] = value
    _rehash(ledger, "ledger_identity_sha256")
    expected["expected_unique_loss_ledger_identity_sha256"] = ledger[
        "ledger_identity_sha256"
    ]
    with pytest.raises(LaunchInputAuthorityError, match=message):
        build_launch_input_authority(proof, ledger, carrier, **expected)


def test_replayed_logical_range_cannot_manufacture_capacity() -> None:
    proof, ledger, carrier, expected = _fixture()
    duplicate = copy.deepcopy(ledger["segments"][0])
    duplicate["segment_identity_sha256"] = _sha("second-segment-id")
    ledger["segments"].append(duplicate)
    ledger["one_pass_unique_nonignored_causal_loss_positions"] = 14
    _rehash(ledger, "ledger_identity_sha256")
    expected["expected_unique_loss_ledger_identity_sha256"] = ledger[
        "ledger_identity_sha256"
    ]
    with pytest.raises(LaunchInputAuthorityError, match="duplicate logical range"):
        build_launch_input_authority(proof, ledger, carrier, **expected)


@pytest.mark.parametrize("conclusion", ["failure", "cancelled", "queued", None])
def test_carrier_requires_terminal_success(conclusion: object) -> None:
    proof, ledger, carrier, expected = _fixture()
    carrier["workflow_conclusion"] = conclusion
    with pytest.raises(LaunchInputAuthorityError, match="terminal success"):
        build_launch_input_authority(proof, ledger, carrier, **expected)


@pytest.mark.parametrize("status", ["queued", "in_progress", None])
def test_carrier_requires_terminal_completed_status(status: object) -> None:
    proof, ledger, carrier, expected = _fixture()
    carrier["workflow_status"] = status
    with pytest.raises(LaunchInputAuthorityError, match="terminal completed"):
        build_launch_input_authority(proof, ledger, carrier, **expected)


def test_carrier_workflow_head_must_equal_exact_carrier_sha() -> None:
    proof, ledger, carrier, expected = _fixture()
    carrier["workflow_head_sha"] = "b" * 40
    with pytest.raises(LaunchInputAuthorityError, match="workflow head"):
        build_launch_input_authority(proof, ledger, carrier, **expected)


def test_carrier_workflow_run_is_independently_expected() -> None:
    proof, ledger, carrier, expected = _fixture()
    expected["expected_carrier_workflow_run_id"] = 54321
    with pytest.raises(LaunchInputAuthorityError, match="workflow run substitution"):
        build_launch_input_authority(proof, ledger, carrier, **expected)


def test_carrier_evidence_is_independently_expected() -> None:
    proof, ledger, carrier, expected = _fixture()
    expected["expected_carrier_evidence_sha256"] = _sha("other-evidence")
    with pytest.raises(LaunchInputAuthorityError, match="evidence substitution"):
        build_launch_input_authority(proof, ledger, carrier, **expected)


def test_nonterminal_carrier_fails() -> None:
    proof, ledger, carrier, expected = _fixture()
    carrier["terminal"] = False
    with pytest.raises(LaunchInputAuthorityError, match="nonterminal"):
        build_launch_input_authority(proof, ledger, carrier, **expected)


def test_foreign_pretrained_carrier_fails_closed() -> None:
    proof, ledger, carrier, expected = _fixture()
    carrier["foreign_pretrained_weights_used"] = True
    with pytest.raises(LaunchInputAuthorityError, match="foreign pretrained"):
        build_launch_input_authority(proof, ledger, carrier, **expected)
