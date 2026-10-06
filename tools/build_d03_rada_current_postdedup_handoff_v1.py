"""Bind exact current-Rada global-dedup survivors to DATA-232 inputs.

This execution-only adapter does not choose survivors and does not grant corpus,
tokenizer-fit, training, evaluation-outcome, paid-compute, or scale authority.
It consumes the incumbent full survivor selection plus the current-Rada slice
authority, verifies exact identities, and exposes comparison payload text only
ephemerally after byte/hash coverage is exact.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

REPORT_SCHEMA = "12-6.next100-065-cross-source-dedup-report.v3"
CURRENT_AUTHORITY_SCHEMA = "12-6.d03-rada-current-global-dedup-survivors.v1"
INVENTORY_SCHEMA = "12-6.d03-rada-current-postdedup-retained-inventory.v1"
HANDOFF_SCHEMA = "12-6.postdedup-decontam-handoff.v1"
SELECTION_POLICY = "D03_RADA_CURRENT_REPLACEMENT_GLOBAL_DEDUP_FULL_SELECTION_V1"


class CurrentRadaPostDedupHandoffError(RuntimeError):
    """Fail-closed current-Rada retained-inventory/handoff error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CurrentRadaPostDedupHandoffError(message)


def _canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be lowercase 64-hex SHA-256",
    )
    return value


def _exact_nonnegative_int(value: Any, label: str) -> int:
    _require(
        type(value) is int and value >= 0,
        f"{label} must be an exact non-negative integer",
    )
    return value


def _exact_positive_int(value: Any, label: str) -> int:
    result = _exact_nonnegative_int(value, label)
    _require(result > 0, f"{label} must be positive")
    return result


def _self_hash(
    value: Mapping[str, Any],
    *,
    field: str,
    newline: bool,
) -> str:
    observed = _require_sha256(value.get(field), field)
    core = deepcopy(dict(value))
    core.pop(field, None)
    expected = _sha256_bytes(_canonical_bytes(core, newline=newline))
    _require(observed == expected, f"{field} self-hash mismatch")
    return observed


def _source_rows(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_rows = report.get("sources")
    _require(
        isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes)),
        "matcher report sources must be a sequence",
    )
    rows: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_rows):
        _require(isinstance(raw, Mapping), f"sources[{index}] must be an object")
        source_id = raw.get("source_id")
        _require(
            type(source_id) is str and source_id,
            f"sources[{index}].source_id must be non-empty text",
        )
        _require(source_id not in rows, "matcher report source_id is not unique")
        rows[source_id] = dict(raw)
    _require(
        len(rows) == _exact_positive_int(report.get("source_count"), "source_count"),
        "matcher report source_count drift",
    )
    return rows


def _validate_truth_boundary(
    report: Mapping[str, Any],
    current_authority: Mapping[str, Any],
) -> None:
    for key in ("raw_text_emitted", "model_training_executed", "source_admission_authority"):
        _require(report.get(key) is False, f"matcher report boundary weakened: {key}")
    for key in (
        "tokenizer_fit_authorized",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "scale_promotion_authorized",
    ):
        _require(
            current_authority.get(key) is False,
            f"current-Rada survivor boundary weakened: {key}",
        )
    for key in (
        "canonical_capacity_credited",
        "training_authorized_bytes",
        "authorized_unique_loss_positions",
        "authorized_optimized_target_exposure",
    ):
        _require(
            _exact_nonnegative_int(current_authority.get(key), key) == 0,
            f"current-Rada survivor boundary widened: {key}",
        )


def build_retained_inventory(
    report: Mapping[str, Any],
    current_authority: Mapping[str, Any],
    full_selection: Mapping[str, Any],
    *,
    expected_report_sha256: str,
    expected_current_authority_sha256: str,
    expected_full_selection_sha256: str,
) -> dict[str, Any]:
    """Build a text-free inventory from externally identified incumbent survivors."""

    _require(isinstance(report, Mapping), "matcher report must be an object")
    _require(report.get("schema_version") == REPORT_SCHEMA, "matcher report schema drift")
    report_sha = _self_hash(report, field="report_sha256", newline=True)
    _require(
        report_sha == _require_sha256(expected_report_sha256, "expected_report_sha256"),
        "matcher report does not match independently expected identity",
    )

    _require(
        isinstance(current_authority, Mapping),
        "current-Rada survivor authority must be an object",
    )
    _require(
        current_authority.get("schema_version") == CURRENT_AUTHORITY_SCHEMA,
        "current-Rada survivor authority schema drift",
    )
    current_sha = _self_hash(
        current_authority,
        field="survivor_authority_sha256",
        newline=False,
    )
    _require(
        current_sha
        == _require_sha256(
            expected_current_authority_sha256,
            "expected_current_authority_sha256",
        ),
        "current-Rada survivor authority does not match independently expected identity",
    )
    _require(
        current_authority.get("matcher_report_sha256") == report_sha,
        "current-Rada survivor authority matcher binding drift",
    )

    _require(isinstance(full_selection, Mapping), "full survivor selection must be an object")
    selection_sha = _self_hash(
        full_selection,
        field="survivor_authority_sha256",
        newline=False,
    )
    _require(
        selection_sha
        == _require_sha256(
            expected_full_selection_sha256,
            "expected_full_selection_sha256",
        ),
        "full survivor selection does not match independently expected identity",
    )
    _require(
        current_authority.get("selection_projection_sha256") == selection_sha,
        "current-Rada authority full-selection binding drift",
    )
    _require(
        current_authority.get("selection_projection_schema")
        == full_selection.get("schema_version"),
        "current-Rada authority full-selection schema drift",
    )
    _validate_truth_boundary(report, current_authority)

    rows_by_id = _source_rows(report)
    survivor_ids = full_selection.get("survivor_source_ids")
    _require(
        isinstance(survivor_ids, Sequence)
        and not isinstance(survivor_ids, (str, bytes))
        and all(type(value) is str and value for value in survivor_ids),
        "full survivor selection source IDs invalid",
    )
    survivor_ids = list(survivor_ids)
    _require(
        len(survivor_ids) == len(set(survivor_ids)),
        "full survivor selection source IDs are not unique",
    )
    _require(
        set(survivor_ids) <= set(rows_by_id),
        "full survivor selection references an unknown matcher source",
    )

    post_count = _exact_positive_int(
        full_selection.get("post_dedup_survivor_source_object_count"),
        "post_dedup_survivor_source_object_count",
    )
    _require(len(survivor_ids) == post_count, "full survivor count drift")
    post_bytes = _exact_positive_int(
        full_selection.get("post_dedup_declared_capacity_bytes"),
        "post_dedup_declared_capacity_bytes",
    )
    terminal = report.get("terminal_candidates")
    _require(isinstance(terminal, Mapping), "matcher report terminal_candidates missing")
    _require(
        terminal.get("conservative_unique_capacity_bytes_after") == post_bytes,
        "full survivor capacity does not match matcher terminal capacity",
    )
    for key in (
        "pre_dedup_source_object_count",
        "pre_dedup_declared_capacity_bytes",
        "post_dedup_survivor_source_object_count",
        "post_dedup_declared_capacity_bytes",
        "duplicate_discount_bytes",
        "duplicate_cluster_count",
    ):
        _require(
            current_authority.get(key) == full_selection.get(key),
            f"current-Rada/full-selection drift: {key}",
        )

    retained: list[dict[str, Any]] = []
    retained_capacity = 0
    for source_id in sorted(survivor_ids):
        raw = rows_by_id[source_id]
        source_family = raw.get("source_family")
        modality = raw.get("modality")
        comparison_policy = raw.get("comparison_policy")
        for label, value in (
            ("source_family", source_family),
            ("modality", modality),
            ("comparison_policy", comparison_policy),
        ):
            _require(
                type(value) is str and value,
                f"{source_id}:{label} must be non-empty text",
            )
        declared = _exact_positive_int(
            raw.get("declared_capacity_bytes"),
            f"{source_id}:declared_capacity_bytes",
        )
        comparison_bytes = _exact_nonnegative_int(
            raw.get("comparison_payload_bytes"),
            f"{source_id}:comparison_payload_bytes",
        )
        comparison_sha = _require_sha256(
            raw.get("comparison_payload_sha256"),
            f"{source_id}:comparison_payload_sha256",
        )
        retained_capacity += declared
        retained.append(
            {
                "source_id": source_id,
                "source_family": source_family,
                "modality": modality,
                "declared_capacity_bytes": declared,
                "comparison_policy": comparison_policy,
                "comparison_payload_bytes": comparison_bytes,
                "comparison_payload_sha256": comparison_sha,
                "verified_raw_sha256": _require_sha256(
                    raw.get("verified_raw_sha256"),
                    f"{source_id}:verified_raw_sha256",
                ),
                "stable_origin_id_sha256": _require_sha256(
                    raw.get("stable_origin_id_sha256"),
                    f"{source_id}:stable_origin_id_sha256",
                ),
                "stable_object_id_sha256": _require_sha256(
                    raw.get("stable_object_id_sha256"),
                    f"{source_id}:stable_object_id_sha256",
                ),
            }
        )
    _require(retained_capacity == post_bytes, "retained declared-capacity sum drift")

    current_count = _exact_positive_int(
        current_authority.get("current_rada_survivor_source_object_count"),
        "current_rada_survivor_source_object_count",
    )
    current_bytes = _exact_positive_int(
        current_authority.get("current_rada_survivor_declared_capacity_bytes"),
        "current_rada_survivor_declared_capacity_bytes",
    )
    core = {
        "schema_version": INVENTORY_SCHEMA,
        "selection_policy": SELECTION_POLICY,
        "input_matcher_report_sha256": report_sha,
        "input_full_selection_sha256": selection_sha,
        "input_current_rada_survivor_authority_sha256": current_sha,
        "retained_source_count": len(retained),
        "retained_declared_capacity_bytes": retained_capacity,
        "current_rada_survivor_source_object_count": current_count,
        "current_rada_survivor_declared_capacity_bytes": current_bytes,
        "retained_sources": retained,
        "raw_text_emitted": False,
        "reserved_evaluation_decontamination_complete": False,
        "authorized_training_exposure": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "final_test_payload_read": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "scale_promotion_authorized": False,
    }
    return {
        **core,
        "inventory_identity_sha256": _sha256_bytes(_canonical_bytes(core)),
    }


def _validate_inventory(
    inventory: Mapping[str, Any],
    *,
    expected_inventory_identity_sha256: str,
    expected_full_selection_sha256: str,
    expected_current_authority_sha256: str,
) -> list[dict[str, Any]]:
    _require(isinstance(inventory, Mapping), "retained inventory must be an object")
    _require(inventory.get("schema_version") == INVENTORY_SCHEMA, "inventory schema drift")
    _require(
        inventory.get("selection_policy") == SELECTION_POLICY,
        "inventory selection policy drift",
    )
    observed = _self_hash(
        inventory,
        field="inventory_identity_sha256",
        newline=False,
    )
    _require(
        observed
        == _require_sha256(
            expected_inventory_identity_sha256,
            "expected_inventory_identity_sha256",
        ),
        "inventory does not match independently expected identity",
    )
    expected_selection = _require_sha256(
        expected_full_selection_sha256,
        "expected_full_selection_sha256",
    )
    expected_current = _require_sha256(
        expected_current_authority_sha256,
        "expected_current_authority_sha256",
    )
    _require(
        inventory.get("input_full_selection_sha256") == expected_selection,
        "inventory full-selection identity drift",
    )
    _require(
        inventory.get("input_current_rada_survivor_authority_sha256")
        == expected_current,
        "inventory current-Rada authority identity drift",
    )
    for key in (
        "raw_text_emitted",
        "reserved_evaluation_decontamination_complete",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "final_test_payload_read",
        "final_test_outcomes_read",
        "paid_compute_used",
        "scale_promotion_authorized",
    ):
        _require(inventory.get(key) is False, f"inventory boundary weakened: {key}")
    _require(
        _exact_nonnegative_int(
            inventory.get("authorized_training_exposure"),
            "authorized_training_exposure",
        )
        == 0,
        "inventory grants training exposure",
    )

    raw_rows = inventory.get("retained_sources")
    _require(
        isinstance(raw_rows, Sequence) and not isinstance(raw_rows, (str, bytes)),
        "retained_sources must be a sequence",
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rows):
        _require(isinstance(raw, Mapping), f"retained_sources[{index}] must be an object")
        source_id = raw.get("source_id")
        source_family = raw.get("source_family")
        modality = raw.get("modality")
        comparison_policy = raw.get("comparison_policy")
        for label, value in (
            ("source_id", source_id),
            ("source_family", source_family),
            ("modality", modality),
            ("comparison_policy", comparison_policy),
        ):
            _require(
                type(value) is str and value,
                f"retained_sources[{index}].{label} must be non-empty text",
            )
        _require(source_id not in seen, "retained source_id is not unique")
        seen.add(source_id)
        rows.append(
            {
                "source_id": source_id,
                "source_family": source_family,
                "modality": modality,
                "comparison_policy": comparison_policy,
                "comparison_payload_bytes": _exact_nonnegative_int(
                    raw.get("comparison_payload_bytes"),
                    f"{source_id}:comparison_payload_bytes",
                ),
                "comparison_payload_sha256": _require_sha256(
                    raw.get("comparison_payload_sha256"),
                    f"{source_id}:comparison_payload_sha256",
                ),
            }
        )
    _require(
        len(rows)
        == _exact_positive_int(inventory.get("retained_source_count"), "retained_source_count"),
        "retained source count drift",
    )
    return sorted(rows, key=lambda row: row["source_id"])


def prepare_ephemeral_data232_rows(
    inventory: Mapping[str, Any],
    comparison_payloads: Mapping[str, bytes],
    *,
    expected_inventory_identity_sha256: str,
    expected_full_selection_sha256: str,
    expected_current_authority_sha256: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Verify exact retained comparison bytes and create a text-free handoff."""

    _require(
        isinstance(comparison_payloads, Mapping),
        "comparison_payloads must be an object",
    )
    retained = _validate_inventory(
        inventory,
        expected_inventory_identity_sha256=expected_inventory_identity_sha256,
        expected_full_selection_sha256=expected_full_selection_sha256,
        expected_current_authority_sha256=expected_current_authority_sha256,
    )
    expected_ids = {row["source_id"] for row in retained}
    _require(
        set(comparison_payloads) == expected_ids,
        "comparison payload coverage must equal retained inventory",
    )

    rows: list[dict[str, str]] = []
    projection: list[dict[str, Any]] = []
    for authority in retained:
        source_id = authority["source_id"]
        payload = comparison_payloads[source_id]
        _require(type(payload) is bytes, f"comparison payload must be bytes: {source_id}")
        _require(
            len(payload) == authority["comparison_payload_bytes"],
            f"comparison payload byte-count drift: {source_id}",
        )
        observed_sha = _sha256_bytes(payload)
        _require(
            observed_sha == authority["comparison_payload_sha256"],
            f"comparison payload identity drift: {source_id}",
        )
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise CurrentRadaPostDedupHandoffError(
                f"comparison payload is not strict UTF-8: {source_id}"
            ) from exc
        rows.append(
            {
                "record_id": source_id,
                "source_id": source_id,
                "source_family": authority["source_family"],
                "modality": authority["modality"],
                "text": text,
            }
        )
        projection.append(
            {
                "source_id": source_id,
                "source_family": authority["source_family"],
                "modality": authority["modality"],
                "comparison_policy": authority["comparison_policy"],
                "comparison_payload_bytes": len(payload),
                "comparison_payload_sha256": observed_sha,
            }
        )
    rows.sort(key=lambda row: row["record_id"])
    projection.sort(key=lambda row: row["source_id"])
    matcher_projection = [
        {
            "record_id": row["record_id"],
            "source_id": row["source_id"],
            "source_family": row["source_family"],
            "modality": row["modality"],
            "text_sha256": _sha256_bytes(row["text"].encode("utf-8")),
            "text_utf8_bytes": len(row["text"].encode("utf-8")),
        }
        for row in rows
    ]
    core = {
        "schema_version": HANDOFF_SCHEMA,
        "postdedup_inventory_identity_sha256": _require_sha256(
            expected_inventory_identity_sha256,
            "expected_inventory_identity_sha256",
        ),
        "input_survivor_authority_sha256": _require_sha256(
            expected_full_selection_sha256,
            "expected_full_selection_sha256",
        ),
        "retained_source_count": len(rows),
        "comparison_payload_projection": projection,
        "matcher_input_projection": matcher_projection,
        "matcher_input_projection_sha256": _sha256_bytes(
            _canonical_bytes(matcher_projection)
        ),
        "raw_text_persisted_in_evidence": False,
        "final_test_payload_accessed": False,
        "final_test_outcomes_accessed": False,
        "authorized_training_exposure": 0,
    }
    handoff = {
        **core,
        "handoff_identity_sha256": _sha256_bytes(_canonical_bytes(core)),
    }
    return rows, handoff
