"""Terminal-survivor-bound retained inventory for NEXT100-065F V8 evidence.

NEXT100-065F owns physical source-object survivor selection. This module never chooses
an alternative representative. It consumes an externally identified terminal survivor
authority, uses the nested V3 graph only as a fail-closed scientific cross-check, and
attaches the V8 comparison-payload metadata required by downstream DATA-232.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

V8_SCHEMA = "12-6.next100-065f-global-dedup-report.v8"
V3_SCHEMA = "12-6.next100-065-cross-source-dedup-report.v3"
SURVIVOR_SCHEMA = "12-6.next100-065f-post-dedup-survivors.v1"
SURVIVOR_SELECTION_RULE = (
    "largest_declared_capacity_then_lexicographically_smallest_source_id"
)
OUTPUT_SCHEMA = "12-6.postdedup-retained-source-inventory.v1"
ORIGIN_CLUSTER_SCHEMA = "12-6.postdedup-origin-cluster.v1"
SELECTION_POLICY = "NEXT100_065F_TERMINAL_SURVIVOR_AUTHORITY_V1"
CAPACITY_COLLAPSE_MATCH_TYPES = frozenset(
    {
        "origin_alias",
        "raw_exact",
        "normalized_exact",
        "near_match",
        "document_fragment",
        "code_fork_copy",
        "lineage_mirror",
        "lineage_same_origin_alias",
        "lineage_repository_transfer_alias",
        "lineage_fork",
        "lineage_vendor",
        "lineage_generated_derivative",
    }
)
SURVIVOR_SHARED_SOURCE_FIELDS = (
    "source_id",
    "source_family",
    "modality",
    "declared_capacity_bytes",
    "verified_raw_sha256",
    "normalized_sha256",
    "stable_origin_id_sha256",
    "stable_object_id_sha256",
)


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


def _survivor_canonical_bytes(value: Any) -> bytes:
    """Reproduce NEXT100-065F survivor authority serialization exactly."""
    return json.dumps(
        value,
        ensure_ascii=True,
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
    return f"{rendered}\n".encode()


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
    _require(
        vector.get("duplicate_cluster_count") == terminal.get("duplicate_cluster_count"),
        "V8/V3 duplicate-cluster-count mismatch",
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
        match_type = _nonempty_text(raw.get("match_type"), f"matches[{index}].match_type")
        capacity_collapsing = raw.get("capacity_collapsing")
        _require(
            isinstance(capacity_collapsing, bool),
            f"matches[{index}].capacity_collapsing must be boolean",
        )
        if capacity_collapsing:
            _require(
                match_type in CAPACITY_COLLAPSE_MATCH_TYPES,
                f"unsupported capacity-collapsing match type: {match_type}",
            )
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


def _origin_cluster_bindings(
    dedup: Mapping[str, Any],
    source_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[int, str]]:
    """Bind retained sources to V3 independence clusters without exposing origin text."""
    terminal = dedup.get("terminal_candidates")
    _require(isinstance(terminal, Mapping), "nested V3 terminal candidate summary missing")
    raw_clusters = terminal.get("origin_clusters")
    _require(
        isinstance(raw_clusters, Sequence) and not isinstance(raw_clusters, (str, bytes)),
        "nested V3 origin_clusters must be a sequence",
    )
    expected_count = _nonnegative_int(
        terminal.get("effective_independent_origin_count"),
        "terminal_candidates.effective_independent_origin_count",
    )
    _require(
        len(raw_clusters) == expected_count,
        "nested V3 origin cluster count does not match independence summary",
    )

    expected_origin_hashes = {
        row["stable_origin_id_sha256"] for row in source_by_id.values()
    }
    observed_origin_hashes: set[str] = set()
    bindings: dict[str, tuple[int, str]] = {}
    normalized_clusters: list[tuple[list[str], list[str]]] = []
    for cluster in raw_clusters:
        _require(
            isinstance(cluster, Sequence) and not isinstance(cluster, (str, bytes)) and cluster,
            "nested V3 origin cluster must be a non-empty sequence",
        )
        raw_members = sorted(_nonempty_text(member, "origin cluster member") for member in cluster)
        member_hashes = sorted(_sha256_bytes(member.encode("utf-8")) for member in raw_members)
        _require(
            len(member_hashes) == len(set(member_hashes)),
            "nested V3 origin cluster contains duplicate members",
        )
        normalized_clusters.append((raw_members, member_hashes))

    normalized_clusters.sort(key=lambda item: item[1])
    for cluster_index, (_, member_hashes) in enumerate(normalized_clusters):
        cluster_identity = _sha256_obj(
            {
                "schema_version": ORIGIN_CLUSTER_SCHEMA,
                "stable_origin_id_sha256_members": member_hashes,
            }
        )
        for origin_hash in member_hashes:
            _require(
                origin_hash not in observed_origin_hashes,
                "stable origin appears in multiple nested V3 origin clusters",
            )
            observed_origin_hashes.add(origin_hash)
            bindings[origin_hash] = (cluster_index, cluster_identity)

    _require(
        observed_origin_hashes == expected_origin_hashes,
        "nested V3 origin clusters do not cover the exact source origins",
    )
    return bindings


def _validate_survivor_truth_boundary(authority: Mapping[str, Any]) -> None:
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


def _validate_survivor_authority(
    v8_report: Mapping[str, Any],
    dedup: Mapping[str, Any],
    survivor_authority: Mapping[str, Any],
    source_by_id: Mapping[str, Mapping[str, Any]],
    components: Sequence[Sequence[str]],
    *,
    expected_v8_report_sha256: str,
    expected_survivor_authority_sha256: str,
    expected_unique_capacity: int,
) -> set[str]:
    _require(isinstance(survivor_authority, Mapping), "survivor authority must be an object")
    _require(
        survivor_authority.get("schema_version") == SURVIVOR_SCHEMA,
        "survivor authority schema drift",
    )
    _require(
        survivor_authority.get("selection_rule") == SURVIVOR_SELECTION_RULE,
        "survivor selection-rule drift",
    )
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

    expected_v8 = _require_sha256(
        expected_v8_report_sha256,
        "expected_v8_report_sha256",
    )
    _require(
        survivor_authority.get("v8_report_sha256") == expected_v8,
        "survivor authority is not bound to expected V8 report",
    )
    nested_sha = _require_sha256(dedup.get("report_sha256"), "nested V3 report_sha256")
    _require(
        survivor_authority.get("nested_v3_report_sha256") == nested_sha,
        "survivor authority nested V3 identity drift",
    )
    _validate_survivor_truth_boundary(survivor_authority)

    raw_survivors = survivor_authority.get("survivors")
    _require(
        isinstance(raw_survivors, Sequence)
        and not isinstance(raw_survivors, (str, bytes)),
        "survivor rows must be a sequence",
    )
    survivor_ids: set[str] = set()
    for index, raw in enumerate(raw_survivors):
        _require(isinstance(raw, Mapping), f"survivors[{index}] must be an object")
        source_id = _nonempty_text(raw.get("source_id"), f"survivors[{index}].source_id")
        _require(source_id not in survivor_ids, "survivor source_id is not unique")
        _require(source_id in source_by_id, "survivor authority references unknown V8 source")
        survivor_ids.add(source_id)
        v8_source = source_by_id[source_id]
        for field in SURVIVOR_SHARED_SOURCE_FIELDS:
            _require(field in raw, f"survivor row missing {field}: {source_id}")
            _require(
                raw.get(field) == v8_source.get(field),
                f"survivor/V8 source metadata drift: {source_id}:{field}",
            )

    _require(
        len(survivor_ids) == len(components),
        "survivor count does not match capacity-component count",
    )
    for component in components:
        selected = survivor_ids.intersection(component)
        _require(
            len(selected) == 1,
            "terminal survivor authority must select exactly one source per capacity component",
        )

    selected_capacity = sum(source_by_id[source_id]["declared_capacity_bytes"] for source_id in survivor_ids)
    _require(
        selected_capacity == expected_unique_capacity,
        "terminal survivor set does not reproduce V3 conservative capacity",
    )

    vector = v8_report.get("source_vector")
    _require(isinstance(vector, Mapping), "V8 source vector missing")
    _require(
        survivor_authority.get("pre_dedup_source_object_count") == len(source_by_id),
        "survivor pre-dedup source-count drift",
    )
    _require(
        survivor_authority.get("post_dedup_survivor_source_object_count")
        == len(survivor_ids),
        "survivor post-dedup source-count drift",
    )
    _require(
        survivor_authority.get("pre_dedup_declared_capacity_bytes")
        == vector.get("source_capacity_bytes_before_global_dedup"),
        "survivor pre-dedup capacity drift",
    )
    _require(
        survivor_authority.get("post_dedup_declared_capacity_bytes")
        == expected_unique_capacity,
        "survivor post-dedup capacity drift",
    )
    _require(
        survivor_authority.get("duplicate_discount_bytes")
        == vector.get("duplicate_discount_bytes"),
        "survivor duplicate-discount drift",
    )
    _require(
        survivor_authority.get("duplicate_cluster_count")
        == sum(len(component) > 1 for component in components),
        "survivor duplicate-cluster-count drift",
    )

    raw_clusters = survivor_authority.get("duplicate_clusters")
    _require(
        isinstance(raw_clusters, Sequence) and not isinstance(raw_clusters, (str, bytes)),
        "survivor duplicate_clusters must be a sequence",
    )
    observed_clusters: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_clusters):
        _require(isinstance(raw, Mapping), f"survivor duplicate_clusters[{index}] must be an object")
        raw_members = raw.get("member_source_ids")
        _require(
            isinstance(raw_members, Sequence)
            and not isinstance(raw_members, (str, bytes)),
            f"survivor duplicate_clusters[{index}].member_source_ids must be a sequence",
        )
        members = sorted(str(source_id) for source_id in raw_members)
        selected_source_id = _nonempty_text(
            raw.get("selected_source_id"),
            f"survivor duplicate_clusters[{index}].selected_source_id",
        )
        selected_capacity_value = _nonnegative_int(
            raw.get("selected_declared_capacity_bytes"),
            f"survivor duplicate_clusters[{index}].selected_declared_capacity_bytes",
        )
        observed_clusters.append(
            {
                "member_source_ids": members,
                "selected_source_id": selected_source_id,
                "selected_declared_capacity_bytes": selected_capacity_value,
            }
        )
    observed_clusters.sort(key=lambda row: tuple(row["member_source_ids"]))

    expected_clusters: list[dict[str, Any]] = []
    for component in components:
        if len(component) == 1:
            continue
        selected = sorted(survivor_ids.intersection(component))
        _require(len(selected) == 1, "duplicate component must have one terminal survivor")
        selected_source_id = selected[0]
        expected_clusters.append(
            {
                "member_source_ids": sorted(component),
                "selected_source_id": selected_source_id,
                "selected_declared_capacity_bytes": source_by_id[selected_source_id][
                    "declared_capacity_bytes"
                ],
            }
        )
    expected_clusters.sort(key=lambda row: tuple(row["member_source_ids"]))
    _require(
        observed_clusters == expected_clusters,
        "survivor duplicate-cluster selection does not match terminal V8 components",
    )
    return survivor_ids


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
    survivor_authority: Mapping[str, Any],
    *,
    expected_v8_report_sha256: str,
    expected_survivor_authority_sha256: str,
) -> dict[str, Any]:
    """Materialize exact physical survivors selected only by terminal NEXT100-065F."""
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
    origin_bindings = _origin_cluster_bindings(dedup, source_by_id)
    survivor_ids = _validate_survivor_authority(
        v8_report,
        dedup,
        survivor_authority,
        source_by_id,
        components,
        expected_v8_report_sha256=expected_v8_report_sha256,
        expected_survivor_authority_sha256=expected_survivor_authority_sha256,
        expected_unique_capacity=expected_unique_capacity,
    )

    retained: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    for component_index, component in enumerate(components):
        selected = sorted(survivor_ids.intersection(component))
        _require(
            len(selected) == 1,
            "terminal survivor authority must select exactly one source per component",
        )
        representative_id = selected[0]
        component_identity = _sha256_obj(
            {
                "selection_policy": SELECTION_POLICY,
                "survivor_authority_sha256": expected_survivor_authority_sha256,
                "members": list(component),
                "selected_source_id": representative_id,
            }
        )
        representative = source_by_id[representative_id]
        origin_cluster_index, origin_cluster_identity = origin_bindings[
            representative["stable_origin_id_sha256"]
        ]
        retained.append(
            {
                **dict(representative),
                "component_index": component_index,
                "component_size": len(component),
                "component_identity_sha256": component_identity,
                "independence_cluster_index": origin_cluster_index,
                "independence_cluster_identity_sha256": origin_cluster_identity,
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
        "retained terminal-survivor capacity does not match V3 unique-capacity authority",
    )

    nested_sha = _require_sha256(dedup.get("report_sha256"), "nested V3 report_sha256")
    survivor_sha = _require_sha256(
        expected_survivor_authority_sha256,
        "expected_survivor_authority_sha256",
    )
    core: dict[str, Any] = {
        "schema_version": OUTPUT_SCHEMA,
        "selection_policy": SELECTION_POLICY,
        "upstream_survivor_selection_rule": SURVIVOR_SELECTION_RULE,
        "input_v8_report_sha256": expected_v8_report_sha256,
        "input_v3_dedup_report_sha256": nested_sha,
        "input_survivor_authority_sha256": survivor_sha,
        "input_source_count": len(source_by_id),
        "capacity_component_count": len(components),
        "duplicate_component_count": sum(len(component) > 1 for component in components),
        "independence_cluster_count": len(set(origin_bindings.values())),
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
    survivor_authority: Mapping[str, Any],
    inventory: Mapping[str, Any],
    *,
    expected_v8_report_sha256: str,
    expected_survivor_authority_sha256: str,
) -> None:
    rebuilt = materialize_postdedup_inventory(
        v8_report,
        survivor_authority,
        expected_v8_report_sha256=expected_v8_report_sha256,
        expected_survivor_authority_sha256=expected_survivor_authority_sha256,
    )
    _require(
        _canonical_bytes(rebuilt) == _canonical_bytes(dict(inventory)),
        "post-dedup inventory does not match terminal-survivor-bound rebuild",
    )
