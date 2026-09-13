from __future__ import annotations

import copy
import hashlib
import json

import pytest

from twelve_six.data.current_reserved_decontamination_v1 import (
    CurrentDecontaminationExecutionError,
    build_reserved_payload_binding,
    execute_reserved_decontamination,
    verify_execution_evidence,
)

INVENTORY = "1" * 64
SURVIVOR = "2" * 64
SELECTION = "3" * 64
FINAL = "4" * 64


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _row(
    record_id: str,
    text: str,
    *,
    source: str,
    family: str,
    modality: str = "en",
) -> dict[str, str]:
    return {
        "record_id": record_id,
        "source_id": source,
        "source_family": family,
        "modality": modality,
        "text": text,
    }


def _projection(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    result = []
    for row in rows:
        raw = row["text"].encode("utf-8")
        result.append(
            {
                "record_id": row["record_id"],
                "source_id": row["source_id"],
                "source_family": row["source_family"],
                "modality": row["modality"].lower(),
                "text_sha256": _sha(raw),
                "text_utf8_bytes": len(raw),
            }
        )
    return sorted(result, key=lambda item: str(item["record_id"]))


def _training_handoff(rows: list[dict[str, str]]) -> dict[str, object]:
    projection = _projection(rows)
    core: dict[str, object] = {
        "schema_version": "12-6.postdedup-decontam-handoff.v1",
        "postdedup_inventory_identity_sha256": INVENTORY,
        "input_survivor_authority_sha256": SURVIVOR,
        "retained_source_count": len(rows),
        "matcher_input_projection": projection,
        "matcher_input_projection_sha256": _sha(_canonical_bytes(projection)),
        "raw_text_persisted_in_evidence": False,
        "final_test_payload_accessed": False,
        "final_test_outcomes_accessed": False,
        "authorized_training_exposure": 0,
    }
    core["handoff_identity_sha256"] = _sha(_canonical_bytes(core))
    return core


def _member(row: dict[str, str]) -> dict[str, object]:
    raw = row["text"].encode("utf-8")
    return {
        "record_id": row["record_id"],
        "source_id": row["source_id"],
        "source_family": row["source_family"],
        "modality": row["modality"],
        "content_sha256": _sha(raw),
        "utf8_bytes": len(raw),
        "training_prohibited": True,
        "outcomes_included": False,
    }


def _reserved_binding(
    selection_rows: list[dict[str, str]],
    final_rows: list[dict[str, str]],
) -> dict[str, object]:
    return build_reserved_payload_binding(
        [
            {
                "authority_id": "eval303-selection",
                "identity_sha256": SELECTION,
                "role": "selection_validation",
                "source_sha": "a" * 40,
                "source_membership_identity_sha256": "5" * 64,
                "members": [_member(row) for row in selection_rows],
            },
            {
                "authority_id": "eval233-final",
                "identity_sha256": FINAL,
                "role": "final_test",
                "source_sha": "b" * 40,
                "source_membership_identity_sha256": "6" * 64,
                "members": [_member(row) for row in final_rows],
            },
        ]
    )


def _execute(
    training: list[dict[str, str]],
    selection_rows: list[dict[str, str]],
    final_rows: list[dict[str, str]],
    *,
    handoff: dict[str, object] | None = None,
    binding: dict[str, object] | None = None,
    expected_handoff_identity: str | None = None,
):
    actual_handoff = handoff or _training_handoff(training)
    actual_binding = binding or _reserved_binding(selection_rows, final_rows)
    evaluation = [*selection_rows, *final_rows]
    return execute_reserved_decontamination(
        training,
        evaluation,
        training_handoff_evidence=actual_handoff,
        reserved_payload_binding=actual_binding,
        expected_inventory_identity_sha256=INVENTORY,
        expected_survivor_authority_sha256=SURVIVOR,
        expected_training_handoff_identity_sha256=(
            expected_handoff_identity
            or str(actual_handoff["handoff_identity_sha256"])
        ),
        expected_reserved_binding_identity_sha256=str(
            actual_binding["binding_identity_sha256"]
        ),
        expected_selection_validation_identity_sha256=SELECTION,
        expected_final_test_identity_sha256=FINAL,
        quarantine_cross_source_families=False,
    )


def _reserved_rows() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    selection = [
        _row(
            "selection-1",
            "selection payload",
            source="selection-source",
            family="selection-family",
        )
    ]
    final = [
        _row(
            "final-1",
            "final payload",
            source="final-source",
            family="final-family",
        )
    ]
    return selection, final


def test_clean_scan_binds_both_reserved_roles_and_truthfully_records_payload_access():
    training = [
        _row(
            "train-1",
            "independent training material about weather systems",
            source="train-source",
            family="train-family",
        )
    ]
    selection = [
        _row(
            "selection-1",
            "reserved validation discussion about authentication",
            source="selection-source",
            family="selection-family",
        )
    ]
    final = [
        _row(
            "final-1",
            "sealed final test discussion about astronomy",
            source="final-source",
            family="final-family",
        )
    ]
    handoff = _training_handoff(training)
    report, evidence = _execute(training, selection, final, handoff=handoff)
    assert report["status"] == "PASS_CLEAN"
    assert report["training_corpus_identity"] == INVENTORY
    assert report["selection_validation_identity"] == SELECTION
    assert report["final_test_identity"] == FINAL
    assert report["final_test_outcomes_read"] is False
    assert evidence["training_handoff_identity_sha256"] == handoff[
        "handoff_identity_sha256"
    ]
    assert evidence["final_test_payload_accessed_for_decontamination"] is True
    assert evidence["final_test_outcomes_read"] is False
    assert evidence["authorized_training_exposure"] == 0
    verify_execution_evidence(evidence, report)


def test_final_test_exact_overlap_is_excluded_without_persisting_text():
    leaked = "sealed final evaluation sentence must never enter training"
    training = [
        _row(
            "train-leak",
            leaked,
            source="train-source",
            family="train-family",
        )
    ]
    selection = [
        _row(
            "selection-1",
            "unrelated selection validation payload",
            source="selection-source",
            family="selection-family",
        )
    ]
    final = [
        _row(
            "final-1",
            leaked,
            source="final-source",
            family="final-family",
        )
    ]
    report, evidence = _execute(training, selection, final)
    assert report["status"] == "PASS_WITH_EXCLUSIONS"
    assert report["counts"]["excluded_training_records"] == 1
    durable = json.dumps({"report": report, "evidence": evidence})
    assert leaked not in durable
    assert "train-leak" not in durable
    assert "final-1" not in durable


def test_training_row_mutation_is_rejected_by_handoff_projection():
    original = [
        _row(
            "train-1",
            "original training payload",
            source="train-source",
            family="train-family",
        )
    ]
    handoff = _training_handoff(original)
    mutated = copy.deepcopy(original)
    mutated[0]["text"] = "mutated training payload"
    selection, final = _reserved_rows()
    with pytest.raises(
        CurrentDecontaminationExecutionError,
        match="matcher projection",
    ):
        _execute(mutated, selection, final, handoff=handoff)


def test_self_consistent_inventory_substitution_is_rejected_by_external_identity():
    training = [
        _row(
            "train-1",
            "training payload",
            source="train-source",
            family="train-family",
        )
    ]
    handoff = _training_handoff(training)
    handoff["postdedup_inventory_identity_sha256"] = "9" * 64
    handoff.pop("handoff_identity_sha256")
    handoff["handoff_identity_sha256"] = _sha(_canonical_bytes(handoff))
    selection, final = _reserved_rows()
    with pytest.raises(
        CurrentDecontaminationExecutionError,
        match="inventory identity is not independently expected",
    ):
        _execute(training, selection, final, handoff=handoff)


def test_rehashed_training_projection_substitution_is_rejected_by_external_handoff_identity():
    original = [
        _row(
            "train-1",
            "original authority-bound training payload",
            source="train-source",
            family="train-family",
        )
    ]
    original_handoff = _training_handoff(original)
    substituted = copy.deepcopy(original)
    substituted[0]["text"] = "self-consistent substituted training payload"
    substituted_handoff = _training_handoff(substituted)
    assert substituted_handoff["postdedup_inventory_identity_sha256"] == INVENTORY
    assert substituted_handoff["input_survivor_authority_sha256"] == SURVIVOR
    assert (
        substituted_handoff["handoff_identity_sha256"]
        != original_handoff["handoff_identity_sha256"]
    )
    selection, final = _reserved_rows()
    with pytest.raises(
        CurrentDecontaminationExecutionError,
        match="handoff identity is not independently expected",
    ):
        _execute(
            substituted,
            selection,
            final,
            handoff=substituted_handoff,
            expected_handoff_identity=str(original_handoff["handoff_identity_sha256"]),
        )


def test_training_handoff_cannot_claim_prior_final_test_payload_access():
    training = [
        _row(
            "train-1",
            "training payload",
            source="train-source",
            family="train-family",
        )
    ]
    handoff = _training_handoff(training)
    handoff["final_test_payload_accessed"] = True
    handoff.pop("handoff_identity_sha256")
    handoff["handoff_identity_sha256"] = _sha(_canonical_bytes(handoff))
    selection, final = _reserved_rows()
    with pytest.raises(
        CurrentDecontaminationExecutionError,
        match="accessed final-test payload before decontamination",
    ):
        _execute(training, selection, final, handoff=handoff)


def test_reserved_payload_text_substitution_is_rejected_by_membership_hash():
    training = [
        _row(
            "train-1",
            "training payload",
            source="train-source",
            family="train-family",
        )
    ]
    selection, final = _reserved_rows()
    binding = _reserved_binding(selection, final)
    mutated_selection = copy.deepcopy(selection)
    mutated_selection[0]["text"] = "tampered selection payload"
    with pytest.raises(
        CurrentDecontaminationExecutionError,
        match="content SHA-256 drift",
    ):
        _execute(training, mutated_selection, final, binding=binding)


def test_self_consistent_reserved_binding_substitution_needs_external_rebinding():
    training = [
        _row(
            "train-1",
            "training payload",
            source="train-source",
            family="train-family",
        )
    ]
    selection, final = _reserved_rows()
    original = _reserved_binding(selection, final)
    substituted = copy.deepcopy(original)
    substituted["reserved_sets"][0]["source_membership_identity_sha256"] = "8" * 64
    substituted.pop("binding_identity_sha256")
    substituted["binding_identity_sha256"] = _sha(_canonical_bytes(substituted))
    handoff = _training_handoff(training)
    with pytest.raises(
        CurrentDecontaminationExecutionError,
        match="independently expected",
    ):
        execute_reserved_decontamination(
            training,
            [*selection, *final],
            training_handoff_evidence=handoff,
            reserved_payload_binding=substituted,
            expected_inventory_identity_sha256=INVENTORY,
            expected_survivor_authority_sha256=SURVIVOR,
            expected_training_handoff_identity_sha256=str(
                handoff["handoff_identity_sha256"]
            ),
            expected_reserved_binding_identity_sha256=str(
                original["binding_identity_sha256"]
            ),
            expected_selection_validation_identity_sha256=SELECTION,
            expected_final_test_identity_sha256=FINAL,
        )


def test_binding_requires_nonempty_selection_and_final_test_sets():
    selection = [
        _row(
            "selection-1",
            "selection payload",
            source="selection-source",
            family="selection-family",
        )
    ]
    with pytest.raises(
        CurrentDecontaminationExecutionError,
        match="final-test payload sets are required",
    ):
        build_reserved_payload_binding(
            [
                {
                    "authority_id": "eval303-selection",
                    "identity_sha256": SELECTION,
                    "role": "selection_validation",
                    "source_sha": "a" * 40,
                    "source_membership_identity_sha256": "5" * 64,
                    "members": [_member(row) for row in selection],
                }
            ]
        )


def test_binding_rejects_outcome_bearing_reserved_payload_metadata():
    final = _row(
        "final-1",
        "final payload",
        source="final-source",
        family="final-family",
    )
    member = _member(final)
    member["outcomes_included"] = True
    with pytest.raises(
        CurrentDecontaminationExecutionError,
        match="outcome-bearing",
    ):
        build_reserved_payload_binding(
            [
                {
                    "authority_id": "selection",
                    "identity_sha256": SELECTION,
                    "role": "selection_validation",
                    "source_sha": "a" * 40,
                    "source_membership_identity_sha256": "5" * 64,
                    "members": [
                        _member(
                            _row(
                                "selection-1",
                                "selection payload",
                                source="selection-source",
                                family="selection-family",
                            )
                        )
                    ],
                },
                {
                    "authority_id": "final",
                    "identity_sha256": FINAL,
                    "role": "final_test",
                    "source_sha": "b" * 40,
                    "source_membership_identity_sha256": "6" * 64,
                    "members": [member],
                },
            ]
        )


def test_execution_evidence_tamper_is_rejected():
    training = [
        _row(
            "train-1",
            "independent training payload",
            source="train-source",
            family="train-family",
        )
    ]
    selection, final = _reserved_rows()
    report, evidence = _execute(training, selection, final)
    tampered = copy.deepcopy(evidence)
    tampered["authorized_training_exposure"] = 1
    with pytest.raises(
        CurrentDecontaminationExecutionError,
        match="evidence hash drift",
    ):
        verify_execution_evidence(tampered, report)
