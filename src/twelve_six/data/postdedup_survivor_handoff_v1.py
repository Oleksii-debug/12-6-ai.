"""Bind terminal V8 survivor authority to downstream decontamination/split identities.

This module never selects dedup survivors. The canonical survivor set is produced by
NEXT100-065F. This layer only verifies that exact authority against its terminal V8
report, enriches each retained source with comparison-payload identities required by
DATA-232 and with hash-only V3 independence-cluster bindings required by a later
cluster-safe split, and preserves a zero-training truth boundary.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

V8_SCHEMA = "12-6.next100-065f-global-dedup-report.v8"
V3_SCHEMA = "12-6.next100-065-cross-source-dedup-report.v3"
SURVIVOR_SCHEMA = "12-6.next100-065f-post-dedup-survivors.v1"
OUTPUT_SCHEMA = "12-6.postdedup-survivor-handoff.v1"
ORIGIN_CLUSTER_SCHEMA = "12-6.postdedup-origin-cluster.v1"


class PostDedupSurvivorHandoffError(RuntimeError):
    """Fail-closed terminal survivor-handoff error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PostDedupSurvivorHandoffError(message)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_bytes(value: Any, *, ensure_ascii: bool = False) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=ensure_ascii,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _v8_report_sha256(value: Mapping[str, Any]) -> str:
    core = dict(value)
    core.pop("report_sha256", None)
    return _sha256_bytes(_canonical_bytes(core))


def _v3_report_sha256(value: Mapping[str, Any]) -> str:
    core = dict(value)
    core.pop("report_sha256", None)
    return _sha256_bytes(_canonical_bytes(core, ensure_ascii=True) + b"\n")


def _survivor_authority_sha256(value: Mapping[str, Any]) -> str:
    core = dict(value)
    core.pop("survivor_authority_sha256", None)
    return _sha256_bytes(_canonical_bytes(core, ensure_ascii=True))


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


def _validate_v8(
    report: Mapping[str, Any],
    *,
    expected_v8_report_sha256: str,
) -> Mapping[str, Any]:
    _require(isinstance(report, Mapping), "V8 report must be an object")
    _require(report.get("schema_version") == V8_SCHEMA, "unsupported V8 report schema")
    observed = _require_sha256(report.get("report_sha256"), "V8 report_sha256")
    _require(_v8_report_sha256(report) == observed, "V8 report self-hash mismatch")
    _require(
        observed == _require_sha256(expected_v8_report_sha256, "expected_v8_report_sha256"),
        "V8 report does not match terminal binding",
    )
    _require(report.get("execution_profile") == "LOCAL_FREE", "V8 execution profile drift")
    _require(report.get("raw_text_emitted") is False, "V8 durable report leaked raw text")
    boundary = report.get("claim_boundary")
    _require(isinstance(boundary, Mapping), "V8 claim boundary missing")
    _require(boundary.get("authorized_training_exposure") == 0, "V8 grants training exposure")
    for key in (
        "tokenizer_fit_authorized",
        "model_training_executed",
        "final_test_payload_read",
        "paid_compute_used",
    ):
        _require(boundary.get(key) is False, f"V8 boundary weakened: {key}")

    nested = report.get("dedup_v3")
    _require(isinstance(nested, Mapping), "V8 report missing nested V3 report")
    _require(nested.get("schema_version") == V3_SCHEMA, "nested V3 schema drift")
    nested_sha = _require_sha256(nested.get("report_sha256"), "nested V3 report_sha256")
    _require(_v3_report_sha256(nested) == nested_sha, "nested V3 self-hash mismatch")
    _require(nested.get("raw_text_emitted") is False, "nested V3 leaked raw text")
    _require(nested.get("model_training_executed") is False, "nested V3 training drift")
    return nested


def _validate_survivor_authority(
    survivor: Mapping[str, Any],
    *,
    v8_report_sha256: str,
    nested_v3_report_sha256: str,
    expected_survivor_authority_sha256: str,
) -> list[Mapping[str, Any]]:
    _require(isinstance(survivor, Mapping), "survivor authority must be an object")
    _require(survivor.get("schema_version") == SURVIVOR_SCHEMA, "survivor schema drift")
    observed = _require_sha256(
        survivor.get("survivor_authority_sha256"),
        "survivor_authority_sha256",
    )
    _require(
        _survivor_authority_sha256(survivor) == observed,
        "survivor authority self-hash mismatch",
    )
    _require(
        observed
        == _require_sha256(
            expected_survivor_authority_sha256,
            "expected_survivor_authority_sha256",
        ),
        "survivor authority does not match terminal binding",
    )
    _require(survivor.get("v8_report_sha256") == v8_report_sha256, "survivor/V8 drift")
    _require(
        survivor.get("nested_v3_report_sha256") == nested_v3_report_sha256,
        "survivor/nested-V3 drift",
    )
    boundary = survivor.get("truth_boundary")
    _require(isinstance(boundary, Mapping), "survivor truth boundary missing")
    _require(boundary.get("source_object_authority_only") is True, "survivor authority drift")
    _require(
        boundary.get("training_record_inventory_materialized") is False,
        "survivor falsely claims training-record materialization",
    )
    _require(
        boundary.get("evaluation_decontamination_passed") is False,
        "survivor falsely claims decontamination",
    )
    _require(boundary.get("authorized_training_exposure") == 0, "survivor grants exposure")
    for key in (
        "tokenizer_fit_authorized",
        "model_training_executed",
        "final_test_payload_read",
        "paid_compute_used",
    ):
        _require(boundary.get(key) is False, f"survivor boundary weakened: {key}")
    rows = survivor.get("survivors")
    _require(
        isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) and bool(rows),
        "survivor rows must be a non-empty sequence",
    )
    return list(rows)


def _source_rows_by_id(nested: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = nested.get("sources")
    _require(
        isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) and bool(rows),
        "nested V3 sources must be a non-empty sequence",
    )
    by_id: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(rows):
        _require(isinstance(raw, Mapping), f"sources[{index}] must be an object")
        source_id = _nonempty_text(raw.get("source_id"), f"sources[{index}].source_id")
        _require(source_id not in by_id, "nested V3 source_id is not unique")
        row = {
            "source_id": source_id,
            "source_family": _nonempty_text(raw.get("source_family"), f"{source_id}.source_family"),
            "modality": _nonempty_text(raw.get("modality"), f"{source_id}.modality"),
            "declared_capacity_bytes": _nonnegative_int(
                raw.get("declared_capacity_bytes"), f"{source_id}.declared_capacity_bytes"
            ),
            "stable_origin_id_sha256": _require_sha256(
                raw.get("stable_origin_id_sha256"), f"{source_id}.stable_origin_id_sha256"
            ),
            "stable_object_id_sha256": _require_sha256(
                raw.get("stable_object_id_sha256"), f"{source_id}.stable_object_id_sha256"
            ),
            "verified_raw_bytes": _nonnegative_int(
                raw.get("verified_raw_bytes"), f"{source_id}.verified_raw_bytes"
            ),
            "verified_raw_sha256": _require_sha256(
                raw.get("verified_raw_sha256"), f"{source_id}.verified_raw_sha256"
            ),
            "comparison_policy": _nonempty_text(
                raw.get("comparison_policy"), f"{source_id}.comparison_policy"
            ),
            "comparison_payload_bytes": _nonnegative_int(
                raw.get("comparison_payload_bytes"), f"{source_id}.comparison_payload_bytes"
            ),
            "comparison_payload_sha256": _require_sha256(
                raw.get("comparison_payload_sha256"), f"{source_id}.comparison_payload_sha256"
            ),
            "normalized_sha256": _require_sha256(
                raw.get("normalized_sha256"), f"{source_id}.normalized_sha256"
            ),
        }
        by_id[source_id] = row
    _require(len(by_id) == nested.get("source_count"), "nested V3 source_count drift")
    return by_id


def _origin_cluster_bindings(
    nested: Mapping[str, Any],
    source_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, tuple[int, str]], int]:
    terminal = nested.get("terminal_candidates")
    _require(isinstance(terminal, Mapping), "nested V3 terminal candidates missing")
    raw_clusters = terminal.get("origin_clusters")
    _require(
        isinstance(raw_clusters, Sequence) and not isinstance(raw_clusters, (str, bytes)),
        "nested V3 origin_clusters missing",
    )
    expected_count = _nonnegative_int(
        terminal.get("effective_independent_origin_count"),
        "effective_independent_origin_count",
    )
    _require(len(raw_clusters) == expected_count, "origin-cluster count drift")

    normalized_clusters: list[list[str]] = []
    for cluster in raw_clusters:
        _require(
            isinstance(cluster, Sequence) and not isinstance(cluster, (str, bytes)) and bool(cluster),
            "origin cluster must be non-empty",
        )
        members = sorted(_nonempty_text(member, "origin-cluster member") for member in cluster)
        _require(len(members) == len(set(members)), "origin cluster contains duplicate members")
        normalized_clusters.append(sorted(_sha256_bytes(member.encode("utf-8")) for member in members))
    normalized_clusters.sort()

    observed: set[str] = set()
    bindings: dict[str, tuple[int, str]] = {}
    for index, hashes in enumerate(normalized_clusters):
        cluster_identity = _sha256_bytes(
            _canonical_bytes(
                {
                    "schema_version": ORIGIN_CLUSTER_SCHEMA,
                    "stable_origin_id_sha256_members": hashes,
                }
            )
        )
        for origin_hash in hashes:
            _require(origin_hash not in observed, "stable origin appears in multiple clusters")
            observed.add(origin_hash)
            bindings[origin_hash] = (index, cluster_identity)
    expected = {row["stable_origin_id_sha256"] for row in source_by_id.values()}
    _require(observed == expected, "origin clusters do not cover exact V3 source origins")
    return bindings, expected_count


def materialize_postdedup_survivor_handoff(
    v8_report: Mapping[str, Any],
    survivor_authority: Mapping[str, Any],
    *,
    expected_v8_report_sha256: str,
    expected_survivor_authority_sha256: str,
) -> dict[str, Any]:
    """Enrich an already-canonical V8 survivor authority; never reselect survivors."""
    nested = _validate_v8(v8_report, expected_v8_report_sha256=expected_v8_report_sha256)
    survivor_rows = _validate_survivor_authority(
        survivor_authority,
        v8_report_sha256=str(v8_report["report_sha256"]),
        nested_v3_report_sha256=str(nested["report_sha256"]),
        expected_survivor_authority_sha256=expected_survivor_authority_sha256,
    )
    source_by_id = _source_rows_by_id(nested)
    origin_bindings, origin_cluster_count = _origin_cluster_bindings(nested, source_by_id)

    retained: list[dict[str, Any]] = []
    seen: set[str] = set()
    subset_keys = (
        "source_family",
        "modality",
        "declared_capacity_bytes",
        "verified_raw_sha256",
        "normalized_sha256",
        "stable_origin_id_sha256",
        "stable_object_id_sha256",
    )
    for index, raw in enumerate(survivor_rows):
        _require(isinstance(raw, Mapping), f"survivors[{index}] must be an object")
        source_id = _nonempty_text(raw.get("source_id"), f"survivors[{index}].source_id")
        _require(source_id in source_by_id and source_id not in seen, "survivor source_id drift")
        seen.add(source_id)
        source = source_by_id[source_id]
        for key in subset_keys:
            _require(raw.get(key) == source.get(key), f"survivor/V3 drift: {source_id}:{key}")
        cluster_index, cluster_identity = origin_bindings[source["stable_origin_id_sha256"]]
        retained.append(
            {
                **source,
                "independence_cluster_index": cluster_index,
                "independence_cluster_identity_sha256": cluster_identity,
            }
        )
    retained.sort(key=lambda row: row["source_id"])

    expected_count = _nonnegative_int(
        survivor_authority.get("post_dedup_survivor_source_object_count"),
        "post_dedup_survivor_source_object_count",
    )
    _require(len(retained) == expected_count, "retained survivor count drift")
    retained_capacity = sum(row["declared_capacity_bytes"] for row in retained)
    expected_capacity = _nonnegative_int(
        survivor_authority.get("post_dedup_declared_capacity_bytes"),
        "post_dedup_declared_capacity_bytes",
    )
    _require(retained_capacity == expected_capacity, "retained survivor capacity drift")
    vector = v8_report.get("source_vector")
    _require(isinstance(vector, Mapping), "V8 source vector missing")
    _require(
        retained_capacity == vector.get("conservative_unique_capacity_bytes_after_global_dedup"),
        "handoff/V8 unique-capacity drift",
    )

    core: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "input_v8_report_sha256": str(v8_report["report_sha256"]),
        "input_v3_report_sha256": str(nested["report_sha256"]),
        "input_survivor_authority_sha256": str(survivor_authority["survivor_authority_sha256"]),
        "retained_source_count": len(retained),
        "retained_declared_capacity_bytes": retained_capacity,
        "independence_cluster_count": origin_cluster_count,
        "retained_by_modality": _retained_summary(retained, grouping_key="modality"),
        "retained_by_source_family": _retained_summary(retained, grouping_key="source_family"),
        "retained_sources": retained,
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
    core["handoff_identity_sha256"] = _sha256_bytes(_canonical_bytes(core))
    return core


def verify_postdedup_survivor_handoff(
    v8_report: Mapping[str, Any],
    survivor_authority: Mapping[str, Any],
    handoff: Mapping[str, Any],
    *,
    expected_v8_report_sha256: str,
    expected_survivor_authority_sha256: str,
) -> None:
    rebuilt = materialize_postdedup_survivor_handoff(
        v8_report,
        survivor_authority,
        expected_v8_report_sha256=expected_v8_report_sha256,
        expected_survivor_authority_sha256=expected_survivor_authority_sha256,
    )
    _require(
        _canonical_bytes(rebuilt) == _canonical_bytes(dict(handoff)),
        "post-dedup survivor handoff does not match deterministic rebuild",
    )
