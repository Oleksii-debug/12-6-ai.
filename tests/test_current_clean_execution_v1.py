from __future__ import annotations

from copy import deepcopy

import pytest

from twelve_six.data import current_clean_execution_v1 as runner


def _training_records() -> list[dict[str, str]]:
    return [
        {
            "record_id": "keep-uk",
            "source_id": "source-1",
            "source_family": "family-1",
            "modality": "ua",
            "text": "Корисний український текст.",
        },
        {
            "record_id": "drop-en",
            "source_id": "source-2",
            "source_family": "family-2",
            "modality": "en",
            "text": "reserved overlap",
        },
    ]


def _report(records: list[dict[str, str]]) -> dict[str, object]:
    excluded = runner._sha256(records[1]["record_id"].encode("utf-8"))
    return {
        "report_sha256": "a" * 64,
        "excluded_records": [{"record_id_sha256": excluded}],
    }


def test_dependency_blobs_match_current_canonical_sources() -> None:
    assert runner.verify_dependency_blobs() == dict(
        sorted(runner.EXPECTED_DEPENDENCY_BLOBS.items())
    )


def test_post_decontamination_maps_exclusions_and_normalizes_ua() -> None:
    records = _training_records()
    survivors, metadata, excluded_count = runner._post_decontamination_records(
        records,
        _report(records),
    )
    assert survivors == [
        {
            "id": "keep-uk",
            "text": "Корисний український текст.",
            "mode": "uk",
        }
    ]
    assert metadata == {
        "keep-uk": {
            "source_id": "source-1",
            "family": "family-1",
            "mode": "uk",
        }
    }
    assert excluded_count == 1


def test_unknown_exclusion_hash_fails_closed() -> None:
    with pytest.raises(
        runner.CurrentCleanExecutionError,
        match="exclusions do not map exactly",
    ):
        runner._post_decontamination_records(
            _training_records(),
            {"excluded_records": [{"record_id_sha256": "0" * 64}]},
        )


def test_unsupported_post_decontamination_mode_fails_closed() -> None:
    records = [
        {
            "record_id": "r1",
            "source_id": "s1",
            "source_family": "f1",
            "modality": "text",
            "text": "ambiguous language",
        }
    ]
    with pytest.raises(
        runner.CurrentCleanExecutionError,
        match="unsupported post-decontamination modality",
    ):
        runner._post_decontamination_records(
            records,
            {"excluded_records": []},
        )


def test_clean_release_binding_rejects_caller_selected_small_corpus() -> None:
    with pytest.raises(
        runner.CurrentCleanExecutionError,
        match="record count is not independently expected",
    ):
        runner._verify_clean_release_binding(_training_records(), {})


def _synthetic_receipt() -> dict[str, object]:
    rejection_counts = {key: 0 for key in runner._REJECTION_KEYS}
    receipt: dict[str, object] = {
        "schema_version": runner.COMPOSITION_SCHEMA,
        "status": "CLEAN_SURVIVOR_MATERIALIZED_ZERO_CREDIT",
        "clean_training_records_sha256": runner.PRODUCTION_TRAINING_RECORDS_SHA256,
        "clean_training_handoff_sha256": runner.PRODUCTION_TRAINING_HANDOFF_SHA256,
        "data232_report_sha256": "a" * 64,
        "decontamination_execution_identity_sha256": "b" * 64,
        "eval647_execution_receipt_identity_sha256": "c" * 64,
        "quality_execution_identity_sha256": "d" * 64,
        "privacy_execution_identity_sha256": "e" * 64,
        "post_decontamination_input_rows_sha256": "f" * 64,
        "post_quality_input_rows_sha256": "0" * 64,
        "survivor_jsonl_sha256": "1" * 64,
        "survivor_record_inventory_digest_sha256": "2" * 64,
        "survivor_payload_inventory_digest_sha256": "3" * 64,
        "input_training_records": runner.PRODUCTION_INPUT_RECORD_COUNT,
        "post_decontamination_records": runner.PRODUCTION_INPUT_RECORD_COUNT - 1,
        "post_quality_records": runner.PRODUCTION_INPUT_RECORD_COUNT - 2,
        "survivor_records": runner.PRODUCTION_INPUT_RECORD_COUNT - 3,
        "survivor_payload_bytes": 1000,
        "survivor_source_objects": 10,
        "rejection_counts": rejection_counts,
        "privacy_detector_counts": {"EMAIL": 0},
        "dependency_git_blobs": dict(runner.EXPECTED_DEPENDENCY_BLOBS),
        "durable_evidence_hash_only": True,
        "current_corpus_launch_authority_promoted": False,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
    }
    receipt["receipt_identity_sha256"] = runner._sha256(runner._cjson(receipt))
    return receipt


def _verify_receipt(
    receipt: dict[str, object],
    *,
    expected: dict[str, object] | None = None,
) -> str:
    pins = receipt if expected is None else expected
    return runner.verify_current_clean_composition_receipt(
        receipt,
        expected_receipt_identity_sha256=receipt["receipt_identity_sha256"],
        expected_data232_report_sha256=pins["data232_report_sha256"],
        expected_decontamination_execution_identity_sha256=(
            pins["decontamination_execution_identity_sha256"]
        ),
        expected_eval647_execution_receipt_identity_sha256=(
            pins["eval647_execution_receipt_identity_sha256"]
        ),
        expected_quality_execution_identity_sha256=(
            pins["quality_execution_identity_sha256"]
        ),
        expected_privacy_execution_identity_sha256=(
            pins["privacy_execution_identity_sha256"]
        ),
        expected_post_decontamination_input_rows_sha256=(
            pins["post_decontamination_input_rows_sha256"]
        ),
        expected_post_quality_input_rows_sha256=(
            pins["post_quality_input_rows_sha256"]
        ),
        expected_survivor_jsonl_sha256=pins["survivor_jsonl_sha256"],
        expected_survivor_record_inventory_digest_sha256=(
            pins["survivor_record_inventory_digest_sha256"]
        ),
        expected_survivor_payload_inventory_digest_sha256=(
            pins["survivor_payload_inventory_digest_sha256"]
        ),
    )


def test_receipt_verifier_accepts_exact_v2_independent_roots() -> None:
    receipt = _synthetic_receipt()
    assert _verify_receipt(receipt) == receipt["receipt_identity_sha256"]


def test_coherent_reseal_cannot_substitute_nested_quality_root() -> None:
    original = _synthetic_receipt()
    receipt = deepcopy(original)
    receipt["quality_execution_identity_sha256"] = "0" * 64
    body = dict(receipt)
    body.pop("receipt_identity_sha256")
    receipt["receipt_identity_sha256"] = runner._sha256(runner._cjson(body))

    with pytest.raises(
        runner.CurrentCleanExecutionError,
        match="nested execution root drift",
    ):
        _verify_receipt(receipt, expected=original)


def test_boolean_survivor_count_fails_closed() -> None:
    receipt = _synthetic_receipt()
    receipt["survivor_records"] = True
    body = dict(receipt)
    body.pop("receipt_identity_sha256")
    receipt["receipt_identity_sha256"] = runner._sha256(runner._cjson(body))
    with pytest.raises(
        runner.CurrentCleanExecutionError,
        match="invalid positive count",
    ):
        _verify_receipt(receipt)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("authorized_optimized_target_exposure", 1),
        ("optimizer_updates_executed_on_real_targets", True),
        ("tokenizer_fit_authorized", True),
        ("training_executed", True),
        ("learned_weights_created", True),
        ("final_test_outcomes_read", True),
        ("paid_compute_used", True),
        ("foreign_pretrained_weights", True),
    ],
)
def test_receipt_authority_widening_fails_closed(field: str, value: object) -> None:
    receipt = _synthetic_receipt()
    receipt[field] = value
    body = dict(receipt)
    body.pop("receipt_identity_sha256")
    receipt["receipt_identity_sha256"] = runner._sha256(runner._cjson(body))
    with pytest.raises(runner.CurrentCleanExecutionError):
        _verify_receipt(receipt)
