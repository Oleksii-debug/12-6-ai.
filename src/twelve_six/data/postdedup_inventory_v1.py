"""Deterministic retained-source inventory over terminal NEXT100-065F V8 evidence.

This module does not perform matching and does not grant corpus/training authority. It
turns the incumbent global-dedup report into one deterministic, text-free retained set
so decontamination/split/packing successors cannot choose different representatives
from the same capacity-collapsing duplicate component.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

V8_SCHEMA = "12-6.next100-065f-global-dedup-report.v8"
V3_SCHEMA = "12-6.next100-065-cross-source-dedup-report.v3"
OUTPUT_SCHEMA = "12-6.postdedup-retained-source-inventory.v1"
SELECTION_POLICY = "MAX_DECLARED_CAPACITY_THEN_SOURCE_ID_ASC_V1"


class PostDedupInventoryError(RuntimeError):
    """Fail-closed post-dedup inventory error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PostDedupInventoryError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _v3_canonical_bytes(value: Any) -> bytes:
    """Reproduce the incumbent DATA-298/V3 report serialization exactly."""
    rendered = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{rendered}\n".encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_obj(value: Any) -> str:
    return _sha256_bytes(_canonical_bytes(value))


def _v3_report_sha256(value: Mapping[str, Any]) -> str:
    core = dict(value)
    core.pop("report_sha256", None)
    return _sha256_bytes(_v3_canonical_bytes(core))


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


def _nonempty_text(value: Any, label: str) -> str:
    _require(isinstance(value, str) and bool(value), f"{label} must be non-empty text")
    return value


def _v8_self_hash_matches(value: Mapping[str, Any], label: str) -> str:
    observed = _require_sha256(value.get("report_sha256"), f"{label}.report_sha256")
    core = dict(value)
    core.pop("report_sha256", None)
    _require(_sha256_obj(core) == observed, f"{label} self-hash mismatch")
    return observed


def _v3_self_hash_matches(value: Mapping[str, Any], label: str) -> str:
    observed = _require_sha256(value.get("report_sha256"), f"{label}.report_sha256")
    _require(_v3_report_sha256(value) == observed, f"{label} self-hash mismatch")
    return observed


def _validate_v8(
    report: Mapping[str, Any],
    *,
    expected_v8_report_sha256: str,
) -> Mapping[str, Any]:
    _require(isinstance(report, Mapping), "V8 report must be an object")
    _require(report.get("schema_version") == V8_SCHEMA, "unsupported V8 report schema")
    observed_v8 = _v8_self_hash_matches(report, "V8 report")
    expected_v8 = _require_sha256(
        expected_v8_report_sha256,
        "expected_v8_report_sha256",
    )
    _require(observed_v8 == expected_v8, "V8 report does not match expected terminal handoff")
    _require(report.get("execution_profile") == "LOCAL_FREE", "V8 execution profile drift")
    _require(report.get("raw_text_emitted") is False, "V8 durable report leaked raw text")

    boundary = report.get("claim_boundary")
    _require(isinstance(boundary, Mapping), "V8 claim boundary missing")
    _require(
        boundary.get("authorized_training_exposure") == 0,
        "V8 report already grants training exposure",
    )
    for key in (
        "tokenizer_fit_authorized",
        "model_training_executed",
        "final_test_payload_read",
        "paid_compute_used",
    ):
        _require(boundary.get(key) is False, f"V8 claim boundary weakened: {key}")

    dedup = report.get("dedup_v3")
    _require(isinstance(dedup, Mapping), "V8 report missing nested V3 dedup report")
    _require(dedup.get("schema_version") == V3_SCHEMA, "nested V3 report schema drift")
    _v3_self_hash_matches(dedup, "nested V3 report")
    _require(dedup.get("raw_text_emitted") is False, "nested V3 report leaked raw text")
    _require(dedup.get("model_training_executed") is False, "nested V3 training boundary drift")

    vector = report.get("source_vector")
    _require(isinstance(vector, Mapping), "V8 source vector missing")
    terminal = dedup.get("terminal_candidates")
    _require(isinstance(terminal, Mapping), "nested V3 terminal candidate summary missing")
    _require(
        vector.get("source_object_count") == dedup.get("source_count"),
        "V8/V3 source-count mismatch",
    )
    _require(
        vector.get("source_capacity_bytes_before_global_dedup")
        == terminal.get("declared_capacity_bytes_before"),
        "V8/V3 pre-dedup capacity mismatch",
    )
    _require(
        vector.get("conservative_unique_capacity_bytes_after_global_dedup")
        == terminal.get("conservative_unique_capacity_bytes_after"),
        "V8/V3 post-dedup capacity mismatch",
    )
    _require(
        vector.get("duplicate_discount_bytes") == terminal.get("duplicate_discount_bytes"),
        "V8/V3 duplicate-discount mismatch",
    )
    return dedup


def _sources_by_id(dedup: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw_sources = dedup.get("sources")
    _require(
        isinstance(raw_sources, Sequence) and not isinstance(raw_sources, (str, bytes)),
        "nested V3 sources must be a sequence",
    )
    result: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(raw_sources):
        _require(isinstance(raw, Mapping), f"sources[{index}] must be an object")
        source_id = _nonempty_text(raw.get("source_id"), f"sources[{index}].source_id")
        _require(source_id not in result, "nested V3 source_id is not unique")
        result[source_id] = {
            "source_id": source_id,
            "source_family": _nonempty_text(
                raw.get("source_family"),
                f"sources[{index}].source_family",
            ),
            "modality": _nonempty_text(raw.get("modality"), f"sources[{index}].modality"),
            "declared_capacity_bytes": _nonnegative_int(
                raw.get("declared_capacity_bytes"),
                f"sources[{index}].declared_capacity_bytes",
            ),
            "stable_origin_id_sha256": _require_sha256(
                raw.get("stable_origin_id_sha256"),
                f"sources[{index}].stable_origin_id_sha256",
            ),
            "stable_object_id_sha256": _require_sha256(
                raw.get("stable_object_id_sha256"),
                f"sources[{index}].stable_object_id_sha256",
            ),
            "verified_raw_bytes": _nonnegative_int(
                raw.get("verified_raw_bytes"),
                f"sources[{index}].verified_raw_bytes",
            ),
            "verified_raw_sha256": _require_sha256(
                raw.get("verified_raw_sha256"),
                f"sources[{index}].verified_raw_sha256",
            ),
            "comparison_policy": _nonempty_text(
                raw.get("comparison_policy"),
                f"sources[{index}].comparison_policy",
            ),
            "comparison_payload_bytes": _nonnegative_int(
                raw.get("comparison_payload_bytes"),
                f"sources[{index}].comparison_payload_bytes",
            ),
            "comparison_payload_sha256": _require_sha256(
                raw.get("comparison_payload_sha256"),
                f"sources[{index}].comparison_payload_sha256",
            ),
            "normalized_sha256": _require_sha256(
                raw.get("normalized_sha256"),
                f"sources[{index}].normalized_sha256",
            ),
        }
    _require(
        len(result) == dedup.get("source_count"),
        "nested V3 source_count does not match source inventory",
    )
    return result


def _capacity_components(
    dedup: Mapping[str, Any],
    source_by_id: Mapping[str, Mapping[str, Any]],
) -> list[list[str]]:
    adjacency: dict[str, set[str]] = {source_id: set() for source_id in source_by_id}
    raw_matches = dedup.get("matches")
    _require(
        isinstance(raw_matches, Sequence) and not isinstance(raw_matches, (str, bytes)),
        "nested V3 matches must be a sequence",
    )
    for index, raw in enumerate(raw_matches):
        _require(isinstance(raw, Mapping), f"matches[{index}] must be an object")
        left = raw.get("left_source_id")
        right = raw.get("right_source_id")
        _require(
            left in source_by_id and right in source_by_id and left != right,
            f"matches[{index}] endpoints must be distinct known sources",
        )
        capacity_collapsing = raw.get("capacity_collapsing")
        _require(
            isinstance(capacity_collapsing, bool),
            f"matches[{index}].capacity_collapsing must be boolean",
        )
        if capacity_collapsing:
            adjacency[str(left)].add(str(right))
            adjacency[str(right)].add(str(left))

    components: list[list[str]] = []
    unseen = set(source_by_id)
    while unseen:
        root = min(unseen)
        stack = [root]
        members: set[str] = set()
        while stack:
            current = stack.pop()
            if current in members:
                continue
            members.add(current)
            stack.extend(sorted(adjacency[current] - members, reverse=True))
        unseen -= members
        components.append(sorted(members))
    components.sort(key=lambda members: members[0])
    return components


def _validate_component_summary(
    dedup: Mapping[str, Any],
    components: Sequence[Sequence[str]],
    source_by_id: Mapping[str, Mapping[str, Any]],
) -> int:
    terminal = dedup["terminal_candidates"]
    expected_clusters = sorted(
        sorted(component) for component in components if len(component) > 1
    )
    observed_clusters = terminal.get("duplicate_clusters")
    _require(
        isinstance(observed_clusters, Sequence)
        and not isinstance(observed_clusters, (str, bytes)),
        "nested V3 duplicate_clusters must be a sequence",
    )
    normalized_observed: list[list[str]] = []
    for index, cluster in enumerate(observed_clusters):
        _require(
            isinstance(cluster, Sequence) and not isinstance(cluster, (str, bytes)),
            f"duplicate_clusters[{index}] must be a sequence",
        )
        members = sorted(str(source_id) for source_id in cluster)
        _require(
            all(source_id in source_by_id for source_id in members),
            "duplicate cluster references unknown source",
        )
        normalized_observed.append(members)
    normalized_observed.sort()
    _require(
        normalized_observed == expected_clusters,
        "nested V3 duplicate clusters do not match capacity-collapsing components",
    )

    expected_after = sum(
        max(source_by_id[source_id]["declared_capacity_bytes"] for source_id in component)
        for component in components
    )
    observed_after = _nonnegative_int(
        terminal.get("conservative_unique_capacity_bytes_after"),
        "terminal_candidates.conservative_unique_capacity_bytes_after",
    )
    _require(
        observed_after == expected_after,
        "nested V3 capacity arithmetic does not match connected components",
    )
    return expected_after


def _retained_summary(
    retained: Sequence[Mapping[str, Any]],
    *,
    grouping_key: str,
) -> dict[str, dict[str, int]]:
    groups: dict[str, dict[str, int]] = {}
    for row in retained:
        group = str(row[grouping_key])
        summary = groups.setdefault(
            group,
            {"retained_source_count": 0, "retained_capacity_bytes": 0},
        )
        summary["retained_source_count"] += 1
        summary["retained_capacity_bytes"] += int(row["declared_capacity_bytes"])
    return dict(sorted(groups.items()))


def materialize_postdedup_inventory(
    v8_report: Mapping[str, Any],
    *,
    expected_v8_report_sha256: str,
) -> dict[str, Any]:
    """Materialize one deterministic retained source per capacity component."""
    dedup = _validate_v8(
        v8_report,
        expected_v8_report_sha256=expected_v8_report_sha256,
    )
    source_by_id = _sources_by_id(dedup)
    components = _capacity_components(dedup, source_by_id)
    expected_unique_capacity = _validate_component_summary(
        dedup,
        components,
        source_by_id,
    )

    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for component_index, component in enumerate(components):
        representative_id = min(
            component,
            key=lambda source_id: (
                -source_by_id[source_id]["declared_capacity_bytes"],
                source_id,
            ),
        )
        component_identity = _sha256_obj(
            {
                "selection_policy": SELECTION_POLICY,
                "members": list(component),
            }
        )
        representative = source_by_id[representative_id]
        retained.append(
            {
                **dict(representative),
                "component_index": component_index,
                "component_size": len(component),
                "component_identity_sha256": component_identity,
            }
        )
        for source_id in component:
            if source_id == representative_id:
                continue
            excluded.append(
                {
                    "source_id": source_id,
                    "retained_source_id": representative_id,
                    "component_index": component_index,
                    "component_identity_sha256": component_identity,
                    "reason": "GLOBAL_CAPACITY_COLLAPSING_COMPONENT",
                }
            )

    retained.sort(key=lambda row: row["source_id"])
    excluded.sort(key=lambda row: row["source_id"])
    retained_capacity = sum(row["declared_capacity_bytes"] for row in retained)
    _require(
        retained_capacity == expected_unique_capacity,
        "retained representative capacity does not match V3 unique-capacity authority",
    )

    nested_sha = _require_sha256(dedup.get("report_sha256"), "nested V3 report_sha256")
    core: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "selection_policy": SELECTION_POLICY,
        "input_v8_report_sha256": expected_v8_report_sha256,
        "input_v3_dedup_report_sha256": nested_sha,
        "input_source_count": len(source_by_id),
        "capacity_component_count": len(components),
        "duplicate_component_count": sum(len(component) > 1 for component in components),
        "retained_source_count": len(retained),
        "excluded_duplicate_source_count": len(excluded),
        "retained_unique_capacity_bytes": retained_capacity,
        "retained_by_modality": _retained_summary(retained, grouping_key="modality"),
        "retained_by_source_family": _retained_summary(
            retained,
            grouping_key="source_family",
        ),
        "retained_sources": retained,
        "excluded_duplicate_sources": excluded,
        "raw_text_emitted": False,
        "reserved_evaluation_decontamination_complete": False,
        "cluster_safe_split_complete": False,
        "deterministic_packing_complete": False,
        "postpack_unique_loss_ledger_complete": False,
        "tokenizer_fit_authorized": False,
        "authorized_training_exposure": 0,
        "model_training_executed": False,
        "final_test_payload_read": False,
        "paid_compute_used": False,
    }
    core["inventory_identity_sha256"] = _sha256_obj(core)
    return core


def verify_postdedup_inventory(
    v8_report: Mapping[str, Any],
    inventory: Mapping[str, Any],
    *,
    expected_v8_report_sha256: str,
) -> None:
    rebuilt = materialize_postdedup_inventory(
        v8_report,
        expected_v8_report_sha256=expected_v8_report_sha256,
    )
    _require(
        _canonical_bytes(rebuilt) == _canonical_bytes(dict(inventory)),
        "post-dedup inventory does not match deterministic rebuild",
    )
