"""Consume the sealed EVAL-647 reserve in the incumbent DATA-232 executor.

This is a deliberately thin D06 adapter.  It does not implement a second
contamination matcher and it does not authorize EVAL-647 for model selection.
Instead it composes the exact sealed EVAL-647 set as ``auxiliary_reserved``,
requires independently expected identities for that composition, and then
executes the existing DATA-232 reserved-decontamination path.  The returned
receipt is hash-only and explicitly does not promote the scanned corpus to a
launch-authoritative corpus.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from twelve_six.data.current_reserved_decontamination_v1 import (
    execute_reserved_decontamination,
    verify_execution_evidence,
)
from twelve_six.data.eval647_future_training_exclusion_v1 import (
    AUXILIARY_ROLE,
    compose_eval647_future_training_exclusion,
    require_eval647_future_training_exclusion,
)

EXECUTION_RECEIPT_SCHEMA = "12-6.eval647-reserved-decontamination-execution.v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Eval647ReservedDecontaminationError(RuntimeError):
    """Raised when EVAL-647 cannot be proven present in a DATA-232 execution."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Eval647ReservedDecontaminationError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be lowercase SHA-256",
    )
    return str(value)


def execute_eval647_reserved_decontamination(
    training_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
    *,
    training_handoff_evidence: Mapping[str, Any],
    base_reserved_binding: Mapping[str, Any],
    manifest: Mapping[str, Any],
    materialization_evidence: Mapping[str, Any],
    expected_base_reserved_binding_identity_sha256: str,
    expected_composed_reserved_binding_identity_sha256: str,
    expected_eval647_materialization_evidence_identity_sha256: str,
    expected_eval647_object_set_identity_sha256: str,
    expected_inventory_identity_sha256: str,
    expected_survivor_authority_sha256: str,
    expected_training_handoff_identity_sha256: str,
    expected_selection_validation_identity_sha256: str,
    expected_final_test_identity_sha256: str,
    quarantine_cross_source_families: bool = True,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Run DATA-232 only after independently pinning the EVAL-647 exclusion.

    ``evaluation_records`` must contain the complete payload universe described
    by the composed binding: incumbent selection-validation records, final-test
    records and both sealed EVAL-647 code objects.  Consequently the incumbent
    executor itself verifies the raw EVAL-647 bytes before matching them against
    training records.

    The two EVAL-647 identities and both base/composed binding identities are
    supplied independently by the caller.  Rehashing a substituted binding or a
    substituted EVAL-647 materialization therefore cannot silently re-authorize
    it at this execution surface.
    """
    expected_base = _require_sha256(
        expected_base_reserved_binding_identity_sha256,
        "expected_base_reserved_binding_identity_sha256",
    )
    actual_base = _require_sha256(
        base_reserved_binding.get("binding_identity_sha256"),
        "base_reserved_binding.binding_identity_sha256",
    )
    _require(
        actual_base == expected_base,
        "base reserved binding is not independently expected",
    )

    composed, composition_receipt = compose_eval647_future_training_exclusion(
        base_reserved_binding,
        manifest,
        materialization_evidence,
    )
    auxiliary = require_eval647_future_training_exclusion(
        composed,
        manifest,
        materialization_evidence,
    )

    expected_eval647_evidence = _require_sha256(
        expected_eval647_materialization_evidence_identity_sha256,
        "expected_eval647_materialization_evidence_identity_sha256",
    )
    expected_eval647_membership = _require_sha256(
        expected_eval647_object_set_identity_sha256,
        "expected_eval647_object_set_identity_sha256",
    )
    _require(
        auxiliary.get("identity_sha256") == expected_eval647_evidence,
        "EVAL-647 materialization evidence is not independently expected",
    )
    _require(
        auxiliary.get("source_membership_identity_sha256")
        == expected_eval647_membership,
        "EVAL-647 object-set identity is not independently expected",
    )

    expected_composed = _require_sha256(
        expected_composed_reserved_binding_identity_sha256,
        "expected_composed_reserved_binding_identity_sha256",
    )
    actual_composed = _require_sha256(
        composed.get("binding_identity_sha256"),
        "composed_reserved_binding.binding_identity_sha256",
    )
    _require(
        actual_composed == expected_composed,
        "composed EVAL-647 reserved binding is not independently expected",
    )
    _require(
        composition_receipt.get("base_reserved_binding_identity_sha256")
        == expected_base,
        "EVAL-647 composition receipt base identity drift",
    )
    _require(
        composition_receipt.get("composed_reserved_binding_identity_sha256")
        == expected_composed,
        "EVAL-647 composition receipt binding identity drift",
    )
    _require(
        composition_receipt.get("eval647_materialization_evidence_identity_sha256")
        == expected_eval647_evidence,
        "EVAL-647 composition receipt evidence identity drift",
    )
    _require(
        composition_receipt.get("eval647_object_set_identity_sha256")
        == expected_eval647_membership,
        "EVAL-647 composition receipt membership identity drift",
    )

    report, execution_evidence = execute_reserved_decontamination(
        training_records,
        evaluation_records,
        training_handoff_evidence=training_handoff_evidence,
        reserved_payload_binding=composed,
        expected_inventory_identity_sha256=expected_inventory_identity_sha256,
        expected_survivor_authority_sha256=expected_survivor_authority_sha256,
        expected_training_handoff_identity_sha256=(
            expected_training_handoff_identity_sha256
        ),
        expected_reserved_binding_identity_sha256=expected_composed,
        expected_selection_validation_identity_sha256=(
            expected_selection_validation_identity_sha256
        ),
        expected_final_test_identity_sha256=expected_final_test_identity_sha256,
        quarantine_cross_source_families=quarantine_cross_source_families,
    )
    verify_execution_evidence(execution_evidence, report)

    report_identity = _require_sha256(
        report.get("report_sha256"),
        "decontamination_report.report_sha256",
    )
    execution_identity = _require_sha256(
        execution_evidence.get("execution_identity_sha256"),
        "decontamination_execution.execution_identity_sha256",
    )
    _require(
        execution_evidence.get("reserved_payload_binding_identity_sha256")
        == expected_composed,
        "decontamination execution did not consume the expected composed binding",
    )

    receipt: dict[str, Any] = {
        "schema_version": EXECUTION_RECEIPT_SCHEMA,
        "status": "EVAL647_FUTURE_TRAINING_EXCLUSION_CONSUMED_BY_DATA232_EXECUTION",
        "data232_report_sha256": report_identity,
        "data232_execution_identity_sha256": execution_identity,
        "base_reserved_binding_identity_sha256": expected_base,
        "composed_reserved_binding_identity_sha256": expected_composed,
        "eval647_materialization_evidence_identity_sha256": expected_eval647_evidence,
        "eval647_object_set_identity_sha256": expected_eval647_membership,
        "eval647_role": AUXILIARY_ROLE,
        "eval647_reserved_record_count": len(auxiliary["members"]),
        "future_training_exclusion_consumed_by_decontamination_execution": True,
        "current_corpus_launch_authority_promoted": False,
        "selection_validation_records_authorized": 0,
        "authorized_optimized_target_exposure": 0,
        "optimizer_updates_executed_on_real_targets": 0,
        "tokenizer_fit_authorized": False,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "raw_payload_persisted": False,
    }
    receipt["receipt_identity_sha256"] = _sha256(_canonical_bytes(receipt))
    return report, execution_evidence, receipt


def verify_eval647_reserved_decontamination_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_composed_reserved_binding_identity_sha256: str,
    expected_eval647_materialization_evidence_identity_sha256: str,
    expected_eval647_object_set_identity_sha256: str,
) -> None:
    """Verify the durable hash-only execution receipt against external pins."""
    _require(
        receipt.get("schema_version") == EXECUTION_RECEIPT_SCHEMA,
        "EVAL-647 execution receipt schema drift",
    )
    claimed = _require_sha256(
        receipt.get("receipt_identity_sha256"),
        "receipt_identity_sha256",
    )
    body = dict(receipt)
    body.pop("receipt_identity_sha256", None)
    _require(
        _sha256(_canonical_bytes(body)) == claimed,
        "EVAL-647 execution receipt hash drift",
    )
    _require(
        receipt.get("composed_reserved_binding_identity_sha256")
        == _require_sha256(
            expected_composed_reserved_binding_identity_sha256,
            "expected_composed_reserved_binding_identity_sha256",
        ),
        "EVAL-647 execution receipt composed-binding drift",
    )
    _require(
        receipt.get("eval647_materialization_evidence_identity_sha256")
        == _require_sha256(
            expected_eval647_materialization_evidence_identity_sha256,
            "expected_eval647_materialization_evidence_identity_sha256",
        ),
        "EVAL-647 execution receipt materialization drift",
    )
    _require(
        receipt.get("eval647_object_set_identity_sha256")
        == _require_sha256(
            expected_eval647_object_set_identity_sha256,
            "expected_eval647_object_set_identity_sha256",
        ),
        "EVAL-647 execution receipt object-set drift",
    )
    _require(
        receipt.get("future_training_exclusion_consumed_by_decontamination_execution")
        is True,
        "EVAL-647 future-training exclusion was not consumed",
    )
    _require(
        receipt.get("current_corpus_launch_authority_promoted") is False,
        "EVAL-647 receipt illegally promoted corpus launch authority",
    )
    for key in (
        "selection_validation_records_authorized",
        "authorized_optimized_target_exposure",
        "optimizer_updates_executed_on_real_targets",
    ):
        _require(receipt.get(key) == 0, f"EVAL-647 receipt widened authority: {key}")
    for key in (
        "tokenizer_fit_authorized",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "raw_payload_persisted",
    ):
        _require(receipt.get(key) is False, f"EVAL-647 receipt weakened boundary: {key}")
    for key in (
        "data232_report_sha256",
        "data232_execution_identity_sha256",
        "base_reserved_binding_identity_sha256",
    ):
        _require_sha256(receipt.get(key), key)
