from __future__ import annotations

import copy
import json

import pytest

from twelve_six.inference import fresh_process as fp
from twelve_six.inference.contracts import GenerationConfig

_PROCESS_PID = 4242
_PARENT_PID = 31337
_CHALLENGE = "a" * 64
_CHECKPOINT_ID = "b" * 64
_PROMPT_SUITE_ID = "test-suite:v1"
_PROMPT_PAYLOAD_ID = fp.prompt_payload_identity((65,))
_GENERATION_CONFIG_ID = fp.generation_config_identity(
    GenerationConfig(max_new_tokens=1, sample=False, seed=11),
    cache_mode="stateless",
)


def _synthetic_child_response() -> dict[str, object]:
    return {
        "schema_version": fp._CHILD_RESPONSE_SCHEMA,
        "parent_pid": _PARENT_PID,
        "child_pid": _PROCESS_PID,
        "challenge": _CHALLENGE,
        "prompt_suite_identity": _PROMPT_SUITE_ID,
        "prompt_payload_sha256": _PROMPT_PAYLOAD_ID,
        "generation_config_identity": _GENERATION_CONFIG_ID,
        "generated_token_ids": [66],
        "stop_reason": "max_new_tokens",
        "diagnostics": {
            "checkpoint_id": _CHECKPOINT_ID,
            "git_sha": "1" * 40,
            "model_spec_sha256": "2" * 64,
            "tokenizer_config_sha256": "3" * 64,
            "tokenizer_vocab_sha256": "4" * 64,
            "dataset_manifest_sha256": "5" * 64,
            "run_manifest_sha256": "6" * 64,
            "step": 1,
            "tokens_seen": 16,
            "source_kind": "checkpoint",
        },
    }


def _parse_response(response: dict[str, object]) -> dict[str, object]:
    return fp._parse_child_response(
        json.dumps(response, sort_keys=True, separators=(",", ":")),
        process_pid=_PROCESS_PID,
        parent_pid=_PARENT_PID,
        challenge=_CHALLENGE,
        expected_checkpoint_identity=_CHECKPOINT_ID,
        expected_prompt_suite_identity=_PROMPT_SUITE_ID,
        expected_prompt_payload_sha256=_PROMPT_PAYLOAD_ID,
        expected_generation_config_identity=_GENERATION_CONFIG_ID,
    )


def _valid_receipt() -> dict[str, object]:
    return _parse_response(_synthetic_child_response())


def _expectations(receipt: dict[str, object]) -> dict[str, str]:
    return {
        "expected_receipt_identity": str(receipt["receipt_identity"]),
        "expected_checkpoint_identity": str(receipt["checkpoint_identity"]),
        "expected_prompt_suite_identity": str(receipt["prompt_suite_identity"]),
        "expected_prompt_payload_sha256": str(receipt["prompt_payload_sha256"]),
        "expected_generation_config_identity": str(receipt["generation_config_identity"]),
    }


def _reseal(receipt: dict[str, object], *, provenance_changed: bool = False) -> None:
    if provenance_changed:
        receipt["checkpoint_provenance_identity"] = fp._checkpoint_provenance_identity(receipt)
    receipt["process_run_identity"] = fp._process_run_identity(receipt)
    receipt["receipt_identity"] = fp._receipt_identity(receipt)


def test_synthetic_child_response_round_trip_is_canonical() -> None:
    receipt = _valid_receipt()
    assert fp.validate_fresh_process_receipt(receipt, **_expectations(receipt)) == []


@pytest.mark.parametrize(
    ("field", "bad_value", "expected_blocker"),
    [
        ("checkpoint_identity", "B" * 64, "d07.fresh_process.checkpoint_identity_invalid"),
        ("git_sha", "1" * 39, "d07.fresh_process.git_sha_invalid"),
        ("model_spec_sha256", "z" * 64, "d07.fresh_process.model_spec_sha256_invalid"),
        (
            "tokenizer_config_sha256",
            "3" * 63,
            "d07.fresh_process.tokenizer_config_sha256_invalid",
        ),
        (
            "tokenizer_vocab_sha256",
            17,
            "d07.fresh_process.tokenizer_vocab_sha256_invalid",
        ),
        (
            "dataset_manifest_sha256",
            None,
            "d07.fresh_process.dataset_manifest_sha256_invalid",
        ),
        (
            "run_manifest_sha256",
            "6" * 65,
            "d07.fresh_process.run_manifest_sha256_invalid",
        ),
        (
            "prompt_payload_sha256",
            "sha256:" + "0" * 63,
            "d07.fresh_process.prompt_payload_sha256_invalid",
        ),
        (
            "generation_config_identity",
            "sha256:" + "G" * 64,
            "d07.fresh_process.generation_config_identity_invalid",
        ),
        (
            "output_fingerprint",
            "sha256:" + "0" * 63,
            "d07.fresh_process.output_fingerprint_invalid",
        ),
        (
            "challenge_sha256",
            True,
            "d07.fresh_process.challenge_sha256_invalid",
        ),
        (
            "checkpoint_provenance_identity",
            "sha256:" + "0" * 63,
            "d07.fresh_process.checkpoint_provenance_identity_invalid",
        ),
        (
            "process_run_identity",
            "sha256:" + "0" * 63,
            "d07.fresh_process.process_run_identity_invalid",
        ),
        (
            "receipt_identity",
            "sha256:" + "0" * 63,
            "d07.fresh_process.receipt_identity_invalid",
        ),
    ],
)
def test_receipt_rejects_malformed_hash_and_identity_fields(
    field: str,
    bad_value: object,
    expected_blocker: str,
) -> None:
    receipt = _valid_receipt()
    expectations = _expectations(receipt)
    forged = copy.deepcopy(receipt)
    forged[field] = bad_value

    blockers = fp.validate_fresh_process_receipt(forged, **expectations)

    assert expected_blocker in blockers


@pytest.mark.parametrize(
    ("field", "bad_value", "expected_blocker"),
    [
        ("optimizer_step", True, "d07.fresh_process.optimizer_step_invalid"),
        ("tokens_seen", False, "d07.fresh_process.tokens_seen_invalid"),
        ("child_pid", True, "d07.fresh_process.child_pid_invalid"),
        ("child_pid", 42.0, "d07.fresh_process.child_pid_invalid"),
        ("parent_pid", False, "d07.fresh_process.parent_pid_invalid"),
        ("parent_pid", 31337.0, "d07.fresh_process.parent_pid_invalid"),
    ],
)
def test_receipt_rejects_numeric_bool_and_float_aliases(
    field: str,
    bad_value: object,
    expected_blocker: str,
) -> None:
    receipt = _valid_receipt()
    expectations = _expectations(receipt)
    forged = copy.deepcopy(receipt)
    forged[field] = bad_value
    _reseal(forged, provenance_changed=field in {"optimizer_step", "tokens_seen"})

    blockers = fp.validate_fresh_process_receipt(forged, **expectations)

    assert expected_blocker in blockers


def test_coherent_checkpoint_substitution_cannot_reuse_stale_external_expectations() -> None:
    receipt = _valid_receipt()
    expectations = _expectations(receipt)
    forged = copy.deepcopy(receipt)
    forged["checkpoint_identity"] = "c" * 64
    _reseal(forged, provenance_changed=True)

    blockers = fp.validate_fresh_process_receipt(forged, **expectations)

    assert "d07.fresh_process.checkpoint_identity_mismatch" in blockers
    assert "d07.fresh_process.receipt_identity_expected_mismatch" in blockers


def test_coherent_prompt_payload_substitution_cannot_reuse_stale_external_expectations() -> None:
    receipt = _valid_receipt()
    expectations = _expectations(receipt)
    forged = copy.deepcopy(receipt)
    forged["prompt_payload_sha256"] = fp.prompt_payload_identity((66,))
    _reseal(forged)

    blockers = fp.validate_fresh_process_receipt(forged, **expectations)

    assert "d07.fresh_process.prompt_payload_sha256_mismatch" in blockers
    assert "d07.fresh_process.receipt_identity_expected_mismatch" in blockers


def test_coherent_generation_config_substitution_cannot_reuse_stale_external_expectations() -> None:
    receipt = _valid_receipt()
    expectations = _expectations(receipt)
    forged = copy.deepcopy(receipt)
    forged["generation_config_identity"] = fp.generation_config_identity(
        GenerationConfig(max_new_tokens=2, sample=False, seed=11),
        cache_mode="stateless",
    )
    _reseal(forged)

    blockers = fp.validate_fresh_process_receipt(forged, **expectations)

    assert "d07.fresh_process.generation_config_identity_mismatch" in blockers
    assert "d07.fresh_process.receipt_identity_expected_mismatch" in blockers


@pytest.mark.parametrize(
    ("field", "bad_value", "error_match"),
    [
        ("challenge", "f" * 64, "challenge mismatch"),
        ("child_pid", _PROCESS_PID + 1, "OS process separation not proven"),
        ("parent_pid", _PARENT_PID + 1, "parent PID mismatch"),
        ("child_pid", True, "OS process separation not proven"),
        ("parent_pid", True, "parent PID mismatch"),
    ],
)
def test_parent_parser_rejects_child_challenge_and_pid_substitution(
    field: str,
    bad_value: object,
    error_match: str,
) -> None:
    response = _synthetic_child_response()
    response[field] = bad_value

    with pytest.raises(fp.FreshProcessEvidenceError, match=error_match):
        _parse_response(response)
