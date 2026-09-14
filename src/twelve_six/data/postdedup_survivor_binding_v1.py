"""Text-free evidence binding terminal survivors to the post-dedup inventory.

Physical survivor selection is owned by NEXT100-065F and enforced directly inside
``postdedup_inventory_v1``. This module adds a compact durable proof that the exact V8,
terminal survivor authority and resulting inventory identities were consumed together.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from twelve_six.data import postdedup_inventory_v1 as postdedup

SURVIVOR_SCHEMA = postdedup.SURVIVOR_SCHEMA
SURVIVOR_SELECTION_RULE = postdedup.SURVIVOR_SELECTION_RULE
_SHARED_SOURCE_FIELDS = postdedup.SURVIVOR_SHARED_SOURCE_FIELDS
BINDING_SCHEMA = "12-6.postdedup-survivor-inventory-binding.v1"


class PostDedupSurvivorBindingError(RuntimeError):
    """Fail-closed survivor/inventory binding error."""


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
    return postdedup._survivor_canonical_bytes(value)


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{label} must be lowercase 64-hex SHA-256",
    )
    return value


def build_survivor_inventory_binding(
    v8_report: Mapping[str, Any],
    survivor_authority: Mapping[str, Any],
    inventory: Mapping[str, Any],
    *,
    expected_v8_report_sha256: str,
    expected_survivor_authority_sha256: str,
) -> dict[str, Any]:
    """Build compact evidence only after full terminal-survivor verification passes."""
    try:
        postdedup.verify_postdedup_inventory(
            v8_report,
            survivor_authority,
            inventory,
            expected_v8_report_sha256=expected_v8_report_sha256,
            expected_survivor_authority_sha256=expected_survivor_authority_sha256,
        )
    except postdedup.PostDedupInventoryError as exc:
        raise PostDedupSurvivorBindingError(str(exc)) from exc

    expected_v8 = _require_sha256(
        expected_v8_report_sha256,
        "expected_v8_report_sha256",
    )
    expected_survivor = _require_sha256(
        expected_survivor_authority_sha256,
        "expected_survivor_authority_sha256",
    )
    inventory_identity = _require_sha256(
        inventory.get("inventory_identity_sha256"),
        "inventory_identity_sha256",
    )
    _require(
        inventory.get("input_survivor_authority_sha256") == expected_survivor,
        "inventory is not identity-bound to terminal survivor authority",
    )
    _require(
        inventory.get("input_v8_report_sha256") == expected_v8,
        "inventory is not identity-bound to terminal V8 report",
    )
    _require(
        inventory.get("authorized_training_exposure") == 0,
        "post-dedup inventory already grants training exposure",
    )
    _require(
        inventory.get("reserved_evaluation_decontamination_complete") is False,
        "post-dedup inventory falsely claims decontamination complete",
    )

    retained_count = inventory.get("retained_source_count")
    retained_capacity = inventory.get("retained_unique_capacity_bytes")
    _require(
        isinstance(retained_count, int)
        and not isinstance(retained_count, bool)
        and retained_count >= 0,
        "retained_source_count must be a non-negative integer",
    )
    _require(
        isinstance(retained_capacity, int)
        and not isinstance(retained_capacity, bool)
        and retained_capacity >= 0,
        "retained_unique_capacity_bytes must be a non-negative integer",
    )

    core: dict[str, Any] = {
        "schema_version": BINDING_SCHEMA,
        "input_v8_report_sha256": expected_v8,
        "survivor_authority_sha256": expected_survivor,
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
