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


def test_exclusions_are_mapped_by_hash_and_ua_normalizes_to_uk() -> None:
    records = _training_records()
    survivors = runner._normalize_post_decontamination_records(
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


def test_unknown_exclusion_hash_fails_closed() -> None:
    report = {
        "excluded_records": [{"record_id_sha256": "0" * 64}],
    }
    with pytest.raises(
        runner.CurrentCleanExecutionError,
        match="exclusions do not map exactly",
    ):
        runner._normalize_post_decontamination_records(
            _training_records(),
            report,
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
        runner._normalize_post_decontamination_records(
            records,
            {"excluded_records": []},
        )


def test_execute_composes_incumbent_authorities_and_stays_zero_credit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _training_records()
    report = _report(records)
    decontam = {"execution_identity_sha256": "b" * 64}
    eval_receipt = {"receipt_identity_sha256": "c" * 64}
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        runner,
        "verify_dependency_blobs",
        lambda: dict(sorted(runner.EXPECTED_DEPENDENCY_BLOBS.items())),
    )

    def fake_decontam(*args, **kwargs):
        calls["decontam_training"] = args[0]
        calls["decontam_kwargs"] = kwargs
        return report, decontam, eval_receipt

    monkeypatch.setattr(
        runner,
        "execute_eval647_reserved_decontamination",
        fake_decontam,
    )
    monkeypatch.setattr(
        runner,
        "verify_eval647_reserved_decontamination_receipt",
        lambda *args, **kwargs: None,
    )

    def fake_quality(rows, *, input_manifest_sha256, expected_input_rows_sha256):
        calls["quality_rows"] = rows
        calls["quality_manifest"] = input_manifest_sha256
        calls["quality_input"] = expected_input_rows_sha256
        return {"execution_identity_sha256": "d" * 64}

    def fake_privacy(rows, *, expected_input_rows_sha256):
        calls["privacy_rows"] = rows
        calls["privacy_input"] = expected_input_rows_sha256
        return {"execution_identity_sha256": "e" * 64}

    monkeypatch.setattr(runner, "build_quality_execution_authority", fake_quality)
    monkeypatch.setattr(
        runner,
        "verify_quality_execution_authority",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(runner, "build_privacy_execution_authority", fake_privacy)
    monkeypatch.setattr(
        runner,
        "verify_privacy_execution_authority",
        lambda *args, **kwargs: None,
    )

    receipt, observed_report, quality, privacy = (
        runner.execute_current_clean_composition(
            records,
            [],
            training_handoff_evidence={},
            base_reserved_binding={},
            eval647_manifest={},
            eval647_materialization_evidence={},
            expected_base_reserved_binding_identity_sha256="1" * 64,
            expected_composed_reserved_binding_identity_sha256="2" * 64,
            expected_eval647_materialization_evidence_identity_sha256="3" * 64,
            expected_eval647_object_set_identity_sha256="4" * 64,
            expected_inventory_identity_sha256="5" * 64,
            expected_survivor_authority_sha256="6" * 64,
            expected_training_handoff_identity_sha256="7" * 64,
            expected_selection_validation_identity_sha256="8" * 64,
            expected_final_test_identity_sha256="9" * 64,
        )
    )

    survivors = calls["quality_rows"]
    assert survivors == calls["privacy_rows"]
    assert survivors == [
        {
            "id": "keep-uk",
            "text": "Корисний український текст.",
            "mode": "uk",
        }
    ]
    assert calls["quality_manifest"] == "b" * 64
    assert calls["quality_input"] == calls["privacy_input"]
    assert observed_report is report
    assert quality["execution_identity_sha256"] == "d" * 64
    assert privacy["execution_identity_sha256"] == "e" * 64
    assert receipt["input_training_records"] == 2
    assert receipt["excluded_training_records"] == 1
    assert receipt["post_decontamination_records"] == 1
    assert receipt["authorized_optimized_target_exposure"] == 0
    assert receipt["optimizer_updates_executed_on_real_targets"] == 0
    assert receipt["tokenizer_fit_authorized"] is False
    assert receipt["training_executed"] is False
    assert receipt["learned_weights_created"] is False
    assert receipt["final_test_outcomes_read"] is False
    assert receipt["paid_compute_used"] is False
    assert receipt["foreign_pretrained_weights"] is False


def _verify_receipt(receipt: dict[str, object]) -> str:
    return runner.verify_current_clean_composition_receipt(
        receipt,
        expected_receipt_identity_sha256=receipt["receipt_identity_sha256"],
        expected_data232_report_sha256=receipt["data232_report_sha256"],
        expected_decontamination_execution_identity_sha256=(
            receipt["decontamination_execution_identity_sha256"]
        ),
        expected_eval647_execution_receipt_identity_sha256=(
            receipt["eval647_execution_receipt_identity_sha256"]
        ),
        expected_quality_execution_identity_sha256=(
            receipt["quality_execution_identity_sha256"]
        ),
        expected_privacy_execution_identity_sha256=(
            receipt["privacy_execution_identity_sha256"]
        ),
        expected_post_decontamination_input_rows_sha256=(
            receipt["post_decontamination_input_rows_sha256"]
        ),
    )


def _synthetic_receipt() -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema_version": runner.COMPOSITION_SCHEMA,
        "status": "CURRENT_CLEAN_DECONTAM_G05_G06_EXECUTED_ZERO_CREDIT",
        "data232_report_sha256": "a" * 64,
        "decontamination_execution_identity_sha256": "b" * 64,
        "eval647_execution_receipt_identity_sha256": "c" * 64,
        "quality_execution_identity_sha256": "d" * 64,
        "privacy_execution_identity_sha256": "e" * 64,
        "post_decontamination_input_rows_sha256": "f" * 64,
        "input_training_records": 2,
        "excluded_training_records": 1,
        "post_decontamination_records": 1,
        "post_decontamination_utf8_bytes": 10,
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


def test_receipt_verifier_accepts_exact_independently_bound_roots() -> None:
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
        runner.verify_current_clean_composition_receipt(
            receipt,
            expected_receipt_identity_sha256=receipt["receipt_identity_sha256"],
            expected_data232_report_sha256=original["data232_report_sha256"],
            expected_decontamination_execution_identity_sha256=(
                original["decontamination_execution_identity_sha256"]
            ),
            expected_eval647_execution_receipt_identity_sha256=(
                original["eval647_execution_receipt_identity_sha256"]
            ),
            expected_quality_execution_identity_sha256=(
                original["quality_execution_identity_sha256"]
            ),
            expected_privacy_execution_identity_sha256=(
                original["privacy_execution_identity_sha256"]
            ),
            expected_post_decontamination_input_rows_sha256=(
                original["post_decontamination_input_rows_sha256"]
            ),
        )


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
