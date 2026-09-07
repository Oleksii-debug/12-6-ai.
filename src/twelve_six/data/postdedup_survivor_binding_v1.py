"""Bind post-dedup inventory to the terminal NEXT100-065F survivor authority.

NEXT100-065F owns source-object survivor selection. This module does not derive a
second survivor set. It verifies an externally identified terminal survivor authority,
checks that the V8-backed inventory agrees with that authority exactly on every shared
source-object field, and emits a text-free binding for downstream DATA-232 handoff.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from twelve_six.data import postdedup_inventory_v1 as postdedup

SURVIVOR_SCHEMA = "12-6.next100-065f-post-dedup-survivors.v1"
SURVIVOR_SELECTION_RULE = (
    "largest_declared_capacity_then_lexicographically_smallest_source_id"
)
BINDING_SCHEMA = "12-6.postdedup-survivor-inventory-binding.v1"

_SHARED_SOURCE_FIELDS = (
    "source_id",
    "source_family",
    "modality",
    "declared_capacity_bytes",
    "verified_raw_sha256",
    "normalized_sha256",
    "stable_origin_id_sha256",
    "stable_object_id_sha256",
)


class PostDedupSurvivorBindingError(RuntimeError):
    """Fail-closed terminal-survivor binding error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PostDedupSurvivorBindingError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _survivor_canonical_bytes(value: Any) -> bytes:
    """Reproduce NEXT100-065F survivor-authority serialization exactly."""
    return json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be lowercase 64-hex SHA-256",
    )
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{label} must be a non-negative integer",
    )
    return int(value)


def _validate_truth_boundary(authority: Mapping[str, Any]) -> None:
    boundary = authority.get("truth_boundary")
    _require(isinstance(boundary, Mapping), "survivor truth boundary missing")
    _require(
        boundary.get("source_object_authority_only") is True,
        "survivor authority scope drift",
    )
    _require(
        boundary.get("training_record_inventory_materialized") is False,
        "survivor authority falsely claims training-record materialization",
    )
    _require(
        boundary.get("evaluation_decontamination_passed") is False,
        "survivor authority falsely claims evaluation decontamination",
    )
    _require(
        boundary.get("authorized_training_exposure") == 0,
        "survivor authority already grants training exposure",
    )
    for key in (
        "tokenizer_fit_authorized",
        "model_training_executed",
        "final_test_payload_read",
        "paid_compute_used",
    ):
        _require(
            boundary.get(key) is False,
            f"survivor authority boundary weakened: {key}",
        )


def _source_rows_by_id(v8_report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    dedup = v8_report.get("dedup_v3")
    _require(isinstance(dedup, Mapping), "V8 report missing nested V3 report")
    raw_sources = dedup.get("sources")
    _require(
        isinstance(raw_sources, Sequence)
        and not isinstance(raw_sources, (str, bytes)),
        "nested V3 sources must be a sequence",
    )
    result: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_sources):
        _require(isinstance(raw, Mapping), f"nested V3 sources[{index}] must be an object")
        source_id = raw.get("source_id")
        _require(
            isinstance(source_id, str) and bool(source_id),
            f"nested V3 sources[{index}].source_id must be non-empty text",
        )
        _require(source_id not in result, "nested V3 source_id is not unique")
        result[source_id] = raw
    return result


def validate_survivor_authority(
    v8_report: Mapping[str, Any],
    survivor_authority: Mapping[str, Any],
    *,
    expected_v8_report_sha256: str,
    expected_survivor_authority_sha256: str,
) -> list[dict[str, Any]]:
    """Validate one externally identified terminal survivor authority.

    The authority identity is supplied independently. The survivor set is never
    rederived here; only its exact hash/bindings/metadata are verified.
    """
    _require(isinstance(v8_report, Mapping), "V8 report must be an object")
    _require(isinstance(survivor_authority, Mapping), "survivor authority must be an object")
    _require(
        survivor_authority.get("schema_version") == SURVIVOR_SCHEMA,
        "survivor authority schema drift",
    )
    _require(
        survivor_authority.get("selection_rule") == SURVIVOR_SELECTION_RULE,
        "survivor selection-rule drift",
    )

    expected_v8 = _require_sha256(
        expected_v8_report_sha256,
        "expected_v8_report_sha256",
    )
    observed_v8 = _require_sha256(
        v8_report.get("report_sha256"),
        "V8 report_sha256",
    )
    _require(observed_v8 == expected_v8, "V8 report does not match expected terminal identity")

    expected_survivor = _require_sha256(
        expected_survivor_authority_sha256,
        "expected_survivor_authority_sha256",
    )
    observed_survivor = _require_sha256(
        survivor_authority.get("survivor_authority_sha256"),
        "survivor_authority_sha256",
    )
    _require(
        observed_survivor == expected_survivor,
        "survivor authority does not match expected terminal identity",
    )
    survivor_core = deepcopy(dict(survivor_authority))
    survivor_core.pop("survivor_authority_sha256", None)
    _require(
        _sha256_bytes(_survivor_canonical_bytes(survivor_core)) == observed_survivor,
        "survivor authority self-hash mismatch",
    )
    _require(
        survivor_authority.get("v8_report_sha256") == expected_v8,
        "survivor authority is not bound to expected V8 report",
    )

    dedup = v8_report.get("dedup_v3")
    _require(isinstance(dedup, Mapping), "V8 report missing nested V3 report")
    nested_sha = _require_sha256(
        dedup.get("report_sha256"),
        "nested V3 report_sha256",
    )
    _require(
        survivor_authority.get("nested_v3_report_sha256") == nested_sha,
        "survivor authority nested V3 identity drift",
    )
    _validate_truth_boundary(survivor_authority)

    source_by_id = _source_rows_by_id(v8_report)
    raw_survivors = survivor_authority.get("survivors")
    _require(
        isinstance(raw_survivors, Sequence)
        and not isinstance(raw_survivors, (str, bytes)),
        "survivor rows must be a sequence",
    )
    survivors: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_survivors):
        _require(isinstance(raw, Mapping), f"survivors[{index}] must be an object")
        source_id = raw.get("source_id")
        _require(
            isinstance(source_id, str) and bool(source_id),
            f"survivors[{index}].source_id must be non-empty text",
        )
        _require(source_id not in seen, "survivor source_id is not unique")
        _require(source_id in source_by_id, "survivor authority references unknown V8 source")
        seen.add(source_id)
        v8_source = source_by_id[source_id]
        normalized: dict[str, Any] = {}
        for field in _SHARED_SOURCE_FIELDS:
            _require(field in raw, f"survivor row missing {field}: {source_id}")
            _require(
                raw.get(field) == v8_source.get(field),
                f"survivor/V8 source metadata drift: {source_id}:{field}",
            )
            normalized[field] = raw.get(field)
        survivors.append(normalized)

    survivors.sort(key=lambda row: str(row["source_id"]))
    observed_count = _nonnegative_int(
        survivor_authority.get("post_dedup_survivor_source_object_count"),
        "post_dedup_survivor_source_object_count",
    )
    _require(observed_count == len(survivors), "survivor source-count summary drift")
    observed_capacity = _nonnegative_int(
        survivor_authority.get("post_dedup_declared_capacity_bytes"),
        "post_dedup_declared_capacity_bytes",
    )
    _require(
        observed_capacity
        == sum(int(row["declared_capacity_bytes"]) for row in survivors),
        "survivor capacity summary drift",
    )

    vector = v8_report.get("source_vector")
    _require(isinstance(vector, Mapping), "V8 report missing source vector")
    _require(
        observed_capacity
        == vector.get("conservative_unique_capacity_bytes_after_global_dedup"),
        "survivor capacity does not match V8 post-dedup capacity",
    )
    _require(
        survivor_authority.get("pre_dedup_source_object_count")
        == vector.get("source_object_count"),
        "survivor pre-dedup source-count drift",
    )
    _require(
        survivor_authority.get("pre_dedup_declared_capacity_bytes")
        == vector.get("source_capacity_bytes_before_global_dedup"),
        "survivor pre-dedup capacity drift",
    )
    _require(
        survivor_authority.get("duplicate_discount_bytes")
        == vector.get("duplicate_discount_bytes"),
        "survivor duplicate-discount drift",
    )
    _require(
        survivor_authority.get("duplicate_cluster_count")
        == vector.get("duplicate_cluster_count"),
        "survivor duplicate-cluster-count drift",
    )
    return survivors


def build_survivor_inventory_binding(
    v8_report: Mapping[str, Any],
    survivor_authority: Mapping[str, Any],
    inventory: Mapping[str, Any],
    *,
    expected_v8_report_sha256: str,
    expected_survivor_authority_sha256: str,
) -> dict[str, Any]:
    """Bind an exact V8-backed inventory to the sole terminal survivor authority."""
    survivors = validate_survivor_authority(
        v8_report,
        survivor_authority,
        expected_v8_report_sha256=expected_v8_report_sha256,
        expected_survivor_authority_sha256=expected_survivor_authority_sha256,
    )
    postdedup.verify_postdedup_inventory(
        v8_report,
        inventory,
        expected_v8_report_sha256=expected_v8_report_sha256,
    )
    _require(
        inventory.get("schema_version") == postdedup.OUTPUT_SCHEMA,
        "post-dedup inventory schema drift",
    )
    inventory_identity = _require_sha256(
        inventory.get("inventory_identity_sha256"),
        "inventory_identity_sha256",
    )
    _require(
        inventory.get("authorized_training_exposure") == 0,
        "post-dedup inventory already grants training exposure",
    )
    _require(
        inventory.get("reserved_evaluation_decontamination_complete") is False,
        "post-dedup inventory falsely claims decontamination complete",
    )

    raw_retained = inventory.get("retained_sources")
    _require(
        isinstance(raw_retained, Sequence)
        and not isinstance(raw_retained, (str, bytes)),
        "retained_sources must be a sequence",
    )
    retained_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(raw_retained):
        _require(isinstance(raw, Mapping), f"retained_sources[{index}] must be an object")
        source_id = raw.get("source_id")
        _require(
            isinstance(source_id, str) and bool(source_id),
            f"retained_sources[{index}].source_id must be non-empty text",
        )
        _require(source_id not in retained_by_id, "retained source_id is not unique")
        retained_by_id[source_id] = raw

    survivor_ids = {str(row["source_id"]) for row in survivors}
    _require(
        set(retained_by_id) == survivor_ids,
        "inventory retained set diverges from terminal survivor authority",
    )
    for survivor in survivors:
        source_id = str(survivor["source_id"])
        retained = retained_by_id[source_id]
        for field in _SHARED_SOURCE_FIELDS:
            _require(
                retained.get(field) == survivor.get(field),
                f"inventory/survivor metadata drift: {source_id}:{field}",
            )

    retained_count = _nonnegative_int(
        inventory.get("retained_source_count"),
        "retained_source_count",
    )
    retained_capacity = _nonnegative_int(
        inventory.get("retained_unique_capacity_bytes"),
        "retained_unique_capacity_bytes",
    )
    _require(retained_count == len(survivors), "inventory retained-source count drift")
    _require(
        retained_capacity
        == survivor_authority.get("post_dedup_declared_capacity_bytes"),
        "inventory retained capacity diverges from terminal survivor authority",
    )

    core: dict[str, Any] = {
        "schema_version": BINDING_SCHEMA,
        "input_v8_report_sha256": _require_sha256(
            expected_v8_report_sha256,
            "expected_v8_report_sha256",
        ),
        "survivor_authority_sha256": _require_sha256(
            expected_survivor_authority_sha256,
            "expected_survivor_authority_sha256",
        ),
        "postdedup_inventory_identity_sha256": inventory_identity,
        "selection_authority": "NEXT100_065F_TERMINAL_SURVIVOR_AUTHORITY",
        "comparison_metadata_authority": "TERMINAL_V8_NESTED_V3_SOURCE_ROWS",
        "retained_source_count": retained_count,
        "retained_unique_capacity_bytes": retained_capacity,
        "raw_text_emitted": False,
        "reserved_evaluation_decontamination_complete": False,
        "authorized_training_exposure": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "final_test_payload_read": False,
        "paid_compute_used": False,
    }
    core["binding_identity_sha256"] = _sha256_bytes(_canonical_bytes(core))
    return core


def verify_survivor_inventory_binding(
    v8_report: Mapping[str, Any],
    survivor_authority: Mapping[str, Any],
    inventory: Mapping[str, Any],
    binding: Mapping[str, Any],
    *,
    expected_v8_report_sha256: str,
    expected_survivor_authority_sha256: str,
) -> None:
    """Verify byte-exact survivor/inventory binding evidence."""
    rebuilt = build_survivor_inventory_binding(
        v8_report,
        survivor_authority,
        inventory,
        expected_v8_report_sha256=expected_v8_report_sha256,
        expected_survivor_authority_sha256=expected_survivor_authority_sha256,
    )
    _require(
        _canonical_bytes(rebuilt) == _canonical_bytes(dict(binding)),
        "survivor/inventory binding does not match deterministic rebuild",
    )
