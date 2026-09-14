from __future__ import annotations

import copy

import pytest
import tools.prepare_data232_ephemeral_handoff_v1 as runner


HEX_A = "a" * 40
HEX_B = "b" * 40
HEX_1 = "1" * 64
HEX_2 = "2" * 64
HEX_3 = "3" * 64
HEX_4 = "4" * 64
HEX_5 = "5" * 64
HEX_6 = "6" * 64


def _receipt() -> dict[str, object]:
    return runner.build_receipt(
        carrier_git_sha=HEX_A,
        upstream_git_sha=HEX_B,
        inventory_file_sha256=HEX_1,
        payload_file_sha256=HEX_2,
        training_records_sha256=HEX_3,
        training_handoff_sha256=HEX_4,
        training_records_file_bytes=17,
        inventory={"retained_payload_bytes": 11},
        handoff={
            "postdedup_inventory_identity_sha256": HEX_5,
            "input_survivor_authority_sha256": HEX_6,
            "handoff_identity_sha256": HEX_1,
            "retained_source_count": 1,
        },
    )


def _reseal(receipt: dict[str, object]) -> None:
    body = copy.deepcopy(receipt)
    body.pop("receipt_identity_sha256", None)
    receipt["receipt_identity_sha256"] = runner._sha256_bytes(
        runner._canonical_bytes(body)
    )


def test_receipt_makes_scoped_external_llm_cleanliness_nonclaim() -> None:
    receipt = _receipt()
    runner.verify_receipt(receipt)
    assert "external_llm_or_api_used_for_data_or_intelligence" not in receipt
    assert receipt["current_corpus_external_llm_free_claimed_by_this_carrier"] is False


def test_self_reseal_cannot_turn_scoped_nonclaim_into_clean_claim() -> None:
    receipt = _receipt()
    receipt["current_corpus_external_llm_free_claimed_by_this_carrier"] = True
    _reseal(receipt)
    with pytest.raises(ValueError, match="truth boundary widened"):
        runner.verify_receipt(receipt)


@pytest.mark.parametrize(
    ("group", "mode"),
    [
        ("input_files_sha256", "missing"),
        ("input_files_sha256", "renamed"),
        ("input_files_sha256", "extra"),
        ("output_files_sha256", "missing"),
        ("output_files_sha256", "renamed"),
        ("output_files_sha256", "extra"),
    ],
)
def test_self_resealed_nested_hash_key_drift_is_rejected(group: str, mode: str) -> None:
    receipt = _receipt()
    hashes = receipt[group]
    assert isinstance(hashes, dict)
    first_key = min(hashes)
    value = hashes[first_key]
    if mode == "missing":
        del hashes[first_key]
    elif mode == "renamed":
        del hashes[first_key]
        hashes[f"renamed_{first_key}"] = value
    else:
        hashes["unexpected_extra"] = "f" * 64
    _reseal(receipt)
    with pytest.raises(ValueError, match=f"{group} key set drift"):
        runner.verify_receipt(receipt)


def test_self_resealed_legacy_global_external_llm_false_is_rejected() -> None:
    receipt = _receipt()
    receipt["external_llm_or_api_used_for_data_or_intelligence"] = False
    _reseal(receipt)
    with pytest.raises(ValueError, match="execution receipt key set drift"):
        runner.verify_receipt(receipt)


@pytest.mark.parametrize(
    "key",
    [
        "authorized_optimized_target_exposure",
        "optimizer_updates_executed_on_real_targets",
    ],
)
def test_self_resealed_zero_truth_counter_rejects_float_zero(key: str) -> None:
    receipt = _receipt()
    receipt[key] = 0.0
    _reseal(receipt)
    with pytest.raises(ValueError, match="truth boundary widened"):
        runner.verify_receipt(receipt)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("retained_source_count", True),
        ("retained_source_count", "1"),
        ("retained_source_count", 0),
        ("retained_source_count", -1),
        ("retained_source_count", 1.0),
        ("retained_payload_bytes", True),
        ("retained_payload_bytes", "1"),
        ("retained_payload_bytes", 0),
        ("retained_payload_bytes", -1),
        ("retained_payload_bytes", 1.0),
        ("training_records_file_bytes", True),
        ("training_records_file_bytes", "1"),
        ("training_records_file_bytes", 0),
        ("training_records_file_bytes", -1),
        ("training_records_file_bytes", 1.0),
    ],
)
def test_self_resealed_positive_evidence_fields_require_strict_int(
    key: str,
    value: object,
) -> None:
    receipt = _receipt()
    receipt[key] = value
    _reseal(receipt)
    with pytest.raises(ValueError, match="must be a positive integer"):
        runner.verify_receipt(receipt)
