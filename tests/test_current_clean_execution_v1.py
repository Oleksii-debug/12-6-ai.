from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from twelve_six.data import current_clean_execution_v1 as runner


def _payload_sha(text: str) -> str:
    return runner._sha256(text.encode("utf-8"))


def test_dependency_blobs_match_current_canonical_sources() -> None:
    assert runner.verify_dependency_blobs() == dict(
        sorted(runner.EXPECTED_DEPENDENCY_BLOBS.items())
    )


def test_clean_release_binding_rejects_caller_selected_synthetic_input() -> None:
    with pytest.raises(
        runner.CurrentCleanExecutionError,
        match="record count is not independently expected",
    ):
        runner._verify_clean_release_binding(
            [
                {
                    "record_id": "r1",
                    "source_id": "s1",
                    "source_family": "f1",
                    "modality": "uk",
                    "text": "text",
                }
            ],
            {},
        )


def test_post_decontamination_maps_hash_exclusions_and_normalizes_ua() -> None:
    records = [
        {
            "record_id": "drop",
            "source_id": "s1",
            "source_family": "f1",
            "modality": "en",
            "text": "drop me",
        },
        {
            "record_id": "keep",
            "source_id": "s2",
            "source_family": "f2",
            "modality": "ua",
            "text": "Корисний текст",
        },
    ]
    report = {
        "excluded_records": [
            {"record_id_sha256": runner._sha256(b"drop")}
        ]
    }
    survivors, metadata, excluded = runner._post_decontamination_records(
        records,
        report,
    )
    assert excluded == 1
    assert survivors == [
        {"id": "keep", "text": "Корисний текст", "mode": "uk"}
    ]
    assert metadata["keep"] == {
        "source_id": "s2",
        "family": "f2",
        "mode": "uk",
    }


def test_unknown_decontamination_exclusion_fails_closed() -> None:
    records = [
        {
            "record_id": "keep",
            "source_id": "s1",
            "source_family": "f1",
            "modality": "en",
            "text": "text",
        }
    ]
    with pytest.raises(
        runner.CurrentCleanExecutionError,
        match="exclusions do not map exactly",
    ):
        runner._post_decontamination_records(
            records,
            {"excluded_records": [{"record_id_sha256": "0" * 64}]},
        )


def test_quality_retain_all_with_multiple_windows_materializes_once() -> None:
    text = "abcdefghij"
    inputs = [{"id": "r1", "text": text, "mode": "en"}]
    quality = {
        "records": [
            {
                "record_id": "r1",
                "mode": "en",
                "payload_sha256": _payload_sha(text),
                "utf8_bytes": 10,
                "status": "RETAIN_ALL",
                "retained_utf8_bytes": 10,
                "rejected_utf8_bytes": 0,
                "units": [
                    {
                        "unit_id": "r1#quality-window-0000",
                        "start_char": 0,
                        "end_char": 4,
                        "payload_sha256": _payload_sha("abcd"),
                        "utf8_bytes": 4,
                        "accepted": True,
                    },
                    {
                        "unit_id": "r1#quality-window-0001",
                        "start_char": 4,
                        "end_char": 10,
                        "payload_sha256": _payload_sha("efghij"),
                        "utf8_bytes": 6,
                        "accepted": True,
                    },
                ],
            }
        ]
    }
    output, stats = runner._materialize_quality_survivors(
        inputs,
        {"r1": {"source_id": "s1", "family": "f1", "mode": "en"}},
        quality,
    )
    assert output == [
        {
            "record_id": "r1",
            "source_id": "s1",
            "family": "f1",
            "modality": "en",
            "normalized_payload": text,
        }
    ]
    assert stats["g05_rejected_units"] == 0
    assert stats["g05_rejected_utf8_bytes"] == 0


def test_quality_partial_fails_closed_without_canonical_physical_authority() -> None:
    text = "abcdefghij"
    inputs = [{"id": "r1", "text": text, "mode": "en"}]
    quality = {
        "records": [
            {
                "record_id": "r1",
                "mode": "en",
                "payload_sha256": _payload_sha(text),
                "utf8_bytes": 10,
                "status": "RETAIN_PARTIAL",
                "retained_utf8_bytes": 6,
                "rejected_utf8_bytes": 4,
                "units": [
                    {
                        "unit_id": "r1#quality-window-0000",
                        "start_char": 0,
                        "end_char": 4,
                        "payload_sha256": _payload_sha("abcd"),
                        "utf8_bytes": 4,
                        "accepted": False,
                    },
                    {
                        "unit_id": "r1#quality-window-0001",
                        "start_char": 4,
                        "end_char": 10,
                        "payload_sha256": _payload_sha("efghij"),
                        "utf8_bytes": 6,
                        "accepted": True,
                    },
                ],
            }
        ]
    }
    with pytest.raises(
        runner.CurrentCleanExecutionError,
        match="lacks canonical physical materialization authority",
    ):
        runner._materialize_quality_survivors(
            inputs,
            {"r1": {"source_id": "s1", "family": "f1", "mode": "en"}},
            quality,
        )

def test_quality_reject_document_is_physically_absent_from_survivors() -> None:
    inputs = [
        {"id": "drop", "text": "discard me", "mode": "en"},
        {"id": "keep", "text": "retain me", "mode": "en"},
    ]

    def row(record_id: str, text: str, status: str, accepted: bool) -> dict[str, object]:
        payload = text.encode("utf-8")
        return {
            "record_id": record_id,
            "mode": "en",
            "payload_sha256": _payload_sha(text),
            "utf8_bytes": len(payload),
            "status": status,
            "retained_utf8_bytes": len(payload) if accepted else 0,
            "rejected_utf8_bytes": 0 if accepted else len(payload),
            "units": [
                {
                    "unit_id": record_id,
                    "start_char": 0,
                    "end_char": len(text),
                    "payload_sha256": _payload_sha(text),
                    "utf8_bytes": len(payload),
                    "accepted": accepted,
                }
            ],
        }

    quality = {
        "records": [
            row("drop", "discard me", "REJECT_DOCUMENT", False),
            row("keep", "retain me", "RETAIN_ALL", True),
        ]
    }
    metadata = {
        "drop": {"source_id": "s-drop", "family": "f", "mode": "en"},
        "keep": {"source_id": "s-keep", "family": "f", "mode": "en"},
    }
    output, stats = runner._materialize_quality_survivors(inputs, metadata, quality)
    assert [record["record_id"] for record in output] == ["keep"]
    assert all(record["normalized_payload"] != "discard me" for record in output)
    assert stats["g05_reject_documents"] == 1
    assert stats["g05_rejected_utf8_bytes"] == len("discard me".encode("utf-8"))


def test_privacy_quarantine_and_exclude_are_physically_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {
            "record_id": "allow",
            "source_id": "s1",
            "family": "f",
            "modality": "en",
            "normalized_payload": "public",
        },
        {
            "record_id": "exclude",
            "source_id": "s2",
            "family": "f",
            "modality": "en",
            "normalized_payload": "private-exclude",
        },
        {
            "record_id": "quarantine",
            "source_id": "s3",
            "family": "f",
            "modality": "en",
            "normalized_payload": "private-quarantine",
        },
    ]

    def privacy_row(record: dict[str, str], action: str) -> dict[str, object]:
        payload = record["normalized_payload"].encode("utf-8")
        return {
            "record_id": record["record_id"],
            "mode": record["modality"],
            "payload_sha256": _payload_sha(record["normalized_payload"]),
            "utf8_bytes": len(payload),
            "action": action,
        }

    privacy = {
        "privacy_binding": {"placeholder": True},
        "records": [
            privacy_row(records[0], "ALLOW"),
            privacy_row(records[1], "EXCLUDE"),
            privacy_row(records[2], "QUARANTINE"),
        ],
    }
    monkeypatch.setattr(
        runner,
        "_privacy_runtime",
        lambda path, privacy_binding: (lambda text: [], lambda raw: None, "f" * 40),
    )
    output, stats = runner._materialize_privacy_survivors(records, privacy)
    assert [record["record_id"] for record in output] == ["allow"]
    serialized = runner.canonical_record_bytes(output)
    assert b"private-exclude" not in serialized
    assert b"private-quarantine" not in serialized
    assert stats["g06_exclude_records"] == 1
    assert stats["g06_quarantine_records"] == 1


def test_privacy_redaction_uses_canonical_materializer_and_rescans_allow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = [
        {
            "record_id": "r1",
            "source_id": "s1",
            "family": "f1",
            "modality": "en",
            "normalized_payload": "email a@b.com end",
        }
    ]
    privacy = {
        "privacy_binding": {"placeholder": True},
        "records": [
            {
                "record_id": "r1",
                "mode": "en",
                "payload_sha256": _payload_sha("email a@b.com end"),
                "utf8_bytes": len("email a@b.com end".encode()),
                "action": "REDACT",
            }
        ],
    }

    def detect(text: str):
        start = text.index("a@b.com")
        return [SimpleNamespace(action="REDACT", start=start, end=start + 7)]

    class Result:
        action = "ALLOW"

    monkeypatch.setattr(
        runner,
        "_privacy_runtime",
        lambda path, privacy_binding: (detect, lambda raw: Result(), "f" * 40),
    )
    output, stats = runner._materialize_privacy_survivors(records, privacy)
    assert output[0]["normalized_payload"] == "email <redacted> end"
    assert stats["g06_redacted_records"] == 1
    assert stats["g06_dropped_utf8_bytes"] == 0


def test_execute_composes_serial_quality_then_privacy_and_survivor_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = [
        {
            "record_id": "r1",
            "source_id": "s1",
            "source_family": "f1",
            "modality": "en",
            "text": "clean source text",
        }
    ]
    monkeypatch.setattr(
        runner,
        "verify_dependency_blobs",
        lambda: dict(sorted(runner.EXPECTED_DEPENDENCY_BLOBS.items())),
    )
    monkeypatch.setattr(runner, "_verify_clean_release_binding", lambda *args: None)
    report = {"report_sha256": "a" * 64, "excluded_records": []}
    decontam = {"execution_identity_sha256": "b" * 64}
    eval_receipt = {"receipt_identity_sha256": "c" * 64}
    monkeypatch.setattr(
        runner,
        "execute_eval647_reserved_decontamination",
        lambda *args, **kwargs: (report, decontam, eval_receipt),
    )
    monkeypatch.setattr(
        runner,
        "verify_eval647_reserved_decontamination_receipt",
        lambda *args, **kwargs: None,
    )

    observed: dict[str, object] = {}

    def fake_quality(rows, *, input_manifest_sha256, expected_input_rows_sha256):
        observed["quality_rows"] = rows
        observed["quality_input_root"] = expected_input_rows_sha256
        return {"execution_identity_sha256": "d" * 64}

    monkeypatch.setattr(runner, "build_quality_execution_authority", fake_quality)
    monkeypatch.setattr(
        runner,
        "verify_quality_execution_authority",
        lambda *args, **kwargs: None,
    )
    quality_survivors = [
        {
            "record_id": "r1",
            "source_id": "s1",
            "family": "f1",
            "modality": "en",
            "normalized_payload": "quality survivor",
        }
    ]
    monkeypatch.setattr(
        runner,
        "_materialize_quality_survivors",
        lambda *args: (
            quality_survivors,
            {
                "g05_reject_documents": 0,
                "g05_partial_documents": 0,
                "g05_rejected_units": 0,
                "g05_rejected_utf8_bytes": 0,
            },
        ),
    )

    def fake_privacy(rows, *, expected_input_rows_sha256):
        observed["privacy_rows"] = rows
        observed["privacy_input_root"] = expected_input_rows_sha256
        return {
            "execution_identity_sha256": "e" * 64,
            "detector_counts": {},
        }

    monkeypatch.setattr(runner, "build_privacy_execution_authority", fake_privacy)
    monkeypatch.setattr(
        runner,
        "verify_privacy_execution_authority",
        lambda *args, **kwargs: None,
    )
    final = [
        {
            "record_id": "r1",
            "source_id": "s1",
            "family": "f1",
            "modality": "en",
            "normalized_payload": "final survivor",
        }
    ]
    monkeypatch.setattr(
        runner,
        "_materialize_privacy_survivors",
        lambda *args: (
            final,
            {
                "g06_redacted_records": 0,
                "g06_quarantine_records": 0,
                "g06_exclude_records": 0,
                "g06_dropped_utf8_bytes": 0,
            },
        ),
    )

    (
        receipt,
        observed_report,
        observed_decontam,
        observed_eval,
        quality,
        privacy,
        final_survivors,
        inventory,
    ) = runner.execute_current_clean_composition(
        training,
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

    assert observed["quality_rows"][0]["text"] == "clean source text"
    assert observed["privacy_rows"] == [
        {"id": "r1", "text": "quality survivor", "mode": "en"}
    ]
    assert observed["quality_input_root"] != observed["privacy_input_root"]
    assert observed_report is report
    assert observed_decontam is decontam
    assert observed_eval is eval_receipt
    assert quality["execution_identity_sha256"] == "d" * 64
    assert privacy["execution_identity_sha256"] == "e" * 64
    assert final_survivors == final
    assert inventory["record_count"] == 1
    assert receipt["status"] == "CLEAN_SURVIVOR_MATERIALIZED_PENDING_INDEPENDENT_QUALIFICATION"
    assert receipt["survivor_records"] == 1
    assert receipt["terminal_post_g05_g06_authority"] is False
    assert receipt["independent_qualification_required"] is True
    assert receipt["authorized_optimized_target_exposure"] == 0
    assert receipt["tokenizer_fit_authorized"] is False
    assert receipt["training_executed"] is False
    assert receipt["final_test_outcomes_read"] is False


def _synthetic_receipt() -> dict[str, object]:
    receipt: dict[str, object] = {
        "schema_version": runner.COMPOSITION_SCHEMA,
        "status": "CLEAN_SURVIVOR_MATERIALIZED_PENDING_INDEPENDENT_QUALIFICATION",
        "clean_training_records_sha256": runner.PRODUCTION_TRAINING_RECORDS_SHA256,
        "clean_training_handoff_sha256": runner.PRODUCTION_TRAINING_HANDOFF_SHA256,
        "data232_report_sha256": "a" * 64,
        "decontamination_execution_identity_sha256": "b" * 64,
        "eval647_execution_receipt_identity_sha256": "c" * 64,
        "quality_execution_identity_sha256": "d" * 64,
        "privacy_execution_identity_sha256": "e" * 64,
        "post_decontamination_input_rows_sha256": "f" * 64,
        "post_quality_input_rows_sha256": "1" * 64,
        "survivor_jsonl_sha256": "2" * 64,
        "survivor_record_inventory_digest_sha256": "3" * 64,
        "survivor_payload_inventory_digest_sha256": "4" * 64,
        "input_training_records": runner.PRODUCTION_INPUT_RECORD_COUNT,
        "post_decontamination_records": 256,
        "post_quality_records": 255,
        "survivor_records": 254,
        "survivor_payload_bytes": 1000,
        "survivor_source_objects": 200,
        "rejection_counts": {key: 0 for key in runner._REJECTION_KEYS},
        "privacy_detector_counts": {},
        "dependency_git_blobs": dict(runner.EXPECTED_DEPENDENCY_BLOBS),
        "durable_evidence_hash_only": True,
        "terminal_post_g05_g06_authority": False,
        "independent_qualification_required": True,
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
    expected_quality: str | None = None,
) -> str:
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
            expected_quality or receipt["quality_execution_identity_sha256"]
        ),
        expected_privacy_execution_identity_sha256=(
            receipt["privacy_execution_identity_sha256"]
        ),
        expected_post_decontamination_input_rows_sha256=(
            receipt["post_decontamination_input_rows_sha256"]
        ),
        expected_post_quality_input_rows_sha256=(
            receipt["post_quality_input_rows_sha256"]
        ),
        expected_survivor_jsonl_sha256=receipt["survivor_jsonl_sha256"],
        expected_survivor_record_inventory_digest_sha256=(
            receipt["survivor_record_inventory_digest_sha256"]
        ),
        expected_survivor_payload_inventory_digest_sha256=(
            receipt["survivor_payload_inventory_digest_sha256"]
        ),
    )


def test_receipt_verifier_accepts_exact_independent_roots() -> None:
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
        _verify_receipt(
            receipt,
            expected_quality=original["quality_execution_identity_sha256"],
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
