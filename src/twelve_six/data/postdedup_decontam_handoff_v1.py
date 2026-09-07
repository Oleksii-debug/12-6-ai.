"""Ephemeral retained-payload handoff from survivor-bound inventory to DATA-232.

Durable evidence remains text-free. Callers reconstruct authority-approved comparison
payloads in an ephemeral workspace, and this module proves exact coverage/hash/size
before exposing in-memory DATA-232 matcher rows.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

from twelve_six.data.postdedup_inventory_v1 import (
    OUTPUT_SCHEMA,
    SELECTION_POLICY,
    SURVIVOR_SELECTION_RULE,
)

HANDOFF_SCHEMA = "12-6.postdedup-decontam-handoff.v1"


class PostDedupDecontamHandoffError(RuntimeError):
    """Fail-closed retained-payload handoff error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PostDedupDecontamHandoffError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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


def _validated_retained_rows(
    inventory: Mapping[str, Any],
    *,
    expected_inventory_identity_sha256: str,
) -> list[dict[str, Any]]:
    _require(isinstance(inventory, Mapping), "post-dedup inventory must be an object")
    _require(inventory.get("schema_version") == OUTPUT_SCHEMA, "post-dedup inventory schema drift")

    observed_identity = _require_sha256(
        inventory.get("inventory_identity_sha256"),
        "inventory_identity_sha256",
    )
    expected_identity = _require_sha256(
        expected_inventory_identity_sha256,
        "expected_inventory_identity_sha256",
    )
    _require(
        observed_identity == expected_identity,
        "post-dedup inventory does not match expected terminal inventory identity",
    )
    core = deepcopy(dict(inventory))
    core.pop("inventory_identity_sha256", None)
    _require(
        _sha256_bytes(_canonical_bytes(core)) == observed_identity,
        "post-dedup inventory self-hash mismatch",
    )
    _require(
        inventory.get("selection_policy") == SELECTION_POLICY,
        "post-dedup inventory survivor-selection authority drift",
    )
    _require(
        inventory.get("upstream_survivor_selection_rule") == SURVIVOR_SELECTION_RULE,
        "post-dedup inventory upstream survivor rule drift",
    )
    _require_sha256(
        inventory.get("input_survivor_authority_sha256"),
        "input_survivor_authority_sha256",
    )
    _require(inventory.get("raw_text_emitted") is False, "post-dedup inventory raw-text boundary drift")
    _require(
        inventory.get("reserved_evaluation_decontamination_complete") is False,
        "post-dedup inventory falsely claims decontamination complete",
    )
    _require(
        inventory.get("authorized_training_exposure") == 0,
        "post-dedup inventory already grants training exposure",
    )
    for key in (
        "tokenizer_fit_authorized",
        "model_training_executed",
        "final_test_payload_read",
        "paid_compute_used",
    ):
        _require(inventory.get(key) is False, f"post-dedup inventory boundary weakened: {key}")

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
                isinstance(value, str) and bool(value),
                f"retained_sources[{index}].{label} must be non-empty text",
            )
        _require(str(source_id) not in seen, "retained source_id is not unique")
        seen.add(str(source_id))
        rows.append(
            {
                "source_id": str(source_id),
                "source_family": str(source_family),
                "modality": str(modality),
                "comparison_policy": str(comparison_policy),
                "comparison_payload_bytes": _nonnegative_int(
                    raw.get("comparison_payload_bytes"),
                    f"retained_sources[{index}].comparison_payload_bytes",
                ),
                "comparison_payload_sha256": _require_sha256(
                    raw.get("comparison_payload_sha256"),
                    f"retained_sources[{index}].comparison_payload_sha256",
                ),
            }
        )
    _require(
        len(rows) == inventory.get("retained_source_count"),
        "retained_source_count does not match retained_sources",
    )
    return sorted(rows, key=lambda row: row["source_id"])


def prepare_ephemeral_data232_rows(
    inventory: Mapping[str, Any],
    comparison_payloads: Mapping[str, bytes],
    *,
    expected_inventory_identity_sha256: str,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Verify exact retained payloads and return ephemeral DATA-232 matcher rows.

    The inventory must match an independently supplied terminal identity; its own
    self-hash is necessary but is not treated as external authority. The returned
    rows contain text and must remain in an ephemeral execution context. The
    companion evidence object is text-free and may be persisted by a successor.
    """
    _require(isinstance(comparison_payloads, Mapping), "comparison_payloads must be an object")
    retained = _validated_retained_rows(
        inventory,
        expected_inventory_identity_sha256=expected_inventory_identity_sha256,
    )
    survivor_authority_sha = _require_sha256(
        inventory.get("input_survivor_authority_sha256"),
        "input_survivor_authority_sha256",
    )
    expected_ids = {row["source_id"] for row in retained}
    _require(
        set(comparison_payloads) == expected_ids,
        "ephemeral payload coverage must equal the exact retained source inventory",
    )

    matcher_rows: list[dict[str, str]] = []
    projection: list[dict[str, Any]] = []
    for authority in retained:
        source_id = authority["source_id"]
        payload = comparison_payloads[source_id]
        _require(isinstance(payload, bytes), f"retained payload must be bytes: {source_id}")
        _require(
            len(payload) == authority["comparison_payload_bytes"],
            f"retained comparison payload byte-count drift: {source_id}",
        )
        observed_sha = _sha256_bytes(payload)
        _require(
            observed_sha == authority["comparison_payload_sha256"],
            f"retained comparison payload identity drift: {source_id}",
        )
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise PostDedupDecontamHandoffError(
                f"retained comparison payload is not strict UTF-8: {source_id}"
            ) from exc
        matcher_rows.append(
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

    matcher_rows.sort(key=lambda row: row["record_id"])
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
        for row in matcher_rows
    ]
    evidence_core: dict[str, Any] = {
        "schema_version": HANDOFF_SCHEMA,
        "postdedup_inventory_identity_sha256": _require_sha256(
            expected_inventory_identity_sha256,
            "expected_inventory_identity_sha256",
        ),
        "input_survivor_authority_sha256": survivor_authority_sha,
        "retained_source_count": len(retained),
        "comparison_payload_projection": projection,
        "matcher_input_projection": matcher_projection,
        "matcher_input_projection_sha256": _sha256_bytes(_canonical_bytes(matcher_projection)),
        "raw_text_persisted_in_evidence": False,
        "final_test_payload_accessed": False,
        "final_test_outcomes_accessed": False,
        "authorized_training_exposure": 0,
    }
    evidence_core["handoff_identity_sha256"] = _sha256_bytes(
        _canonical_bytes(evidence_core)
    )
    return matcher_rows, evidence_core
