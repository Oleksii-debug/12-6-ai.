from __future__ import annotations

import copy
import hashlib
import json

import pytest

from twelve_six.learned20m_launch_input import (
    LaunchInputAuthorityError,
    verify_launch_input_authority,
)


def _hash(value: dict) -> str:
    body = copy.deepcopy(value)
    body.pop("authority_identity_sha256", None)
    encoded = (
        json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _authority() -> dict:
    stages = {
        "normalization": "1" * 64,
        "evaluation_reservations": "2" * 64,
        "dedup": "3" * 64,
        "split": "4" * 64,
        "packing": "5" * 64,
    }
    value = {
        "schema_version": "12-6.learned20m-launch-input-authority.v1",
        "binding_status": "READY_FOR_READINESS_BINDING",
        "data_spine": {
            "terminal_corpus_authority_identity_sha256": "6" * 64,
            "stage_bindings": stages,
            "two_clean_proof_identity_sha256": "7" * 64,
            "two_clean_input_packet_identity_sha256": "8" * 64,
            "two_clean_runtime_identity_sha256": "9" * 64,
            "materialization_identity_sha256": "a" * 64,
            "unique_loss_ledger_identity_sha256": "b" * 64,
            "tokenizer_identity_sha256": "c" * 64,
            "packing_identity_sha256": "d" * 64,
            "one_pass_unique_nonignored_causal_loss_positions": 7,
            "requested_unique_loss_positions": 7,
        },
        "carrier": {
            "repository": "Oleksii-debug/12-6-ai.",
            "git_sha": "e" * 40,
            "modelspec_sha256": "f" * 64,
            "initialization_identity_sha256": "0" * 64,
            "canonical_base": "random_init",
            "foreign_pretrained_weights_used": False,
            "terminal": True,
            "workflow_run_id": 12345,
            "workflow_status": "completed",
            "workflow_conclusion": "success",
            "workflow_head_sha": "e" * 40,
            "evidence_sha256": "1" * 64,
        },
        "claim_boundary": {
            "contains_source_text": False,
            "final_test_payload_consumed": False,
            "authorizes_training": False,
            "authorizes_compute": False,
            "authorized_optimized_target_exposure": 0,
            "replay_padding_or_replacement_can_increase_unique_capacity": False,
        },
    }
    value["authority_identity_sha256"] = _hash(value)
    return value


def _rehash(value: dict) -> None:
    value["authority_identity_sha256"] = _hash(value)


def _verify(value: dict, *, expected_identity_sha256: str | None = None) -> None:
    if expected_identity_sha256 is None:
        expected_identity_sha256 = value["authority_identity_sha256"]
    verify_launch_input_authority(
        value,
        expected_authority_identity_sha256=expected_identity_sha256,
    )


def test_valid_closed_world_authority_verifies() -> None:
    _verify(_authority())


def test_rehashed_coherent_substitution_fails_external_identity() -> None:
    value = _authority()
    expected_identity = value["authority_identity_sha256"]
    value["data_spine"]["terminal_corpus_authority_identity_sha256"] = "a" * 64
    value["data_spine"]["tokenizer_identity_sha256"] = "b" * 64
    value["carrier"]["git_sha"] = "c" * 40
    value["carrier"]["workflow_head_sha"] = "c" * 40
    value["carrier"]["modelspec_sha256"] = "d" * 64
    value["carrier"]["initialization_identity_sha256"] = "e" * 64
    value["carrier"]["evidence_sha256"] = "f" * 64
    _rehash(value)
    with pytest.raises(LaunchInputAuthorityError, match="independently expected"):
        _verify(value, expected_identity_sha256=expected_identity)


def test_rehashed_oversubscription_fails_semantically() -> None:
    value = _authority()
    value["data_spine"]["requested_unique_loss_positions"] = 8
    _rehash(value)
    with pytest.raises(LaunchInputAuthorityError, match="exceeds one-pass"):
        _verify(value)


def test_rehashed_nonterminal_workflow_status_fails_semantically() -> None:
    value = _authority()
    value["carrier"]["workflow_status"] = "queued"
    _rehash(value)
    with pytest.raises(LaunchInputAuthorityError, match="not completed"):
        _verify(value)


def test_rehashed_workflow_head_substitution_fails_semantically() -> None:
    value = _authority()
    value["carrier"]["workflow_head_sha"] = "a" * 40
    _rehash(value)
    with pytest.raises(LaunchInputAuthorityError, match="does not match"):
        _verify(value)


def test_rehashed_foreign_pretrained_flag_fails_semantically() -> None:
    value = _authority()
    value["carrier"]["foreign_pretrained_weights_used"] = True
    _rehash(value)
    with pytest.raises(LaunchInputAuthorityError, match="foreign pretrained"):
        _verify(value)


def test_rehashed_nonterminal_carrier_fails_semantically() -> None:
    value = _authority()
    value["carrier"]["terminal"] = False
    _rehash(value)
    with pytest.raises(LaunchInputAuthorityError, match="terminality"):
        _verify(value)
