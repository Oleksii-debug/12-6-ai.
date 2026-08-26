"""NEXT100-065 lineage-aware cross-source deduplication red-team."""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from twelve_six.data import cross_source_capacity_audit as v1

SCHEMA = "12-6.next100-065-cross-source-dedup-report.v3"
INVENTORY_SCHEMA = "12-6.next100-065-cross-source-dedup.v3"
ALGORITHM = "data232-byte-near-copy-plus-lineage-connected-components-v3"
TERMINAL_STATUSES = {"REGISTRY_TERMINAL", "DEDICATED_TERMINAL"}
RELATION_MATCH_TYPES = {
    "mirror": "lineage_mirror",
    "same_origin_alias": "lineage_same_origin_alias",
    "repository_transfer_alias": "lineage_repository_transfer_alias",
    "fork": "lineage_fork",
    "vendor": "lineage_vendor",
    "generated_derivative": "lineage_generated_derivative",
    "sibling_same_origin": "lineage_sibling_same_origin",
}
LINEAGE_COLLAPSE_MATCH_TYPES = {
    "lineage_mirror",
    "lineage_same_origin_alias",
    "lineage_repository_transfer_alias",
    "lineage_fork",
    "lineage_vendor",
    "lineage_generated_derivative",
}
CAPACITY_COLLAPSE_MATCH_TYPES = set(v1.COLLAPSE_MATCH_TYPES) | LINEAGE_COLLAPSE_MATCH_TYPES


class CrossSourceV3Error(RuntimeError):
    """Fail-closed V3 input or report error."""


def _validate_inventory(inventory: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if inventory.get("schema_version") != INVENTORY_SCHEMA:
        raise CrossSourceV3Error("unsupported V3 inventory schema")
    if inventory.get("local_free_only") is not True:
        raise CrossSourceV3Error("inventory must be LOCAL_FREE only")
    if inventory.get("model_training_executed") is not False:
        raise CrossSourceV3Error("inventory must bind model_training_executed=false")
    rows = inventory.get("sources")
    if not isinstance(rows, list) or not rows:
        raise CrossSourceV3Error("terminal source inventory must be nonempty")
    seen: set[str] = set()
    normalized_rows: list[dict[str, Any]] = []
    for raw in rows:
        if not isinstance(raw, Mapping):
            raise CrossSourceV3Error("source row must be an object")
        row = dict(raw)
        source_id = row.get("source_id")
        if not isinstance(source_id, str) or not source_id or source_id in seen:
            raise CrossSourceV3Error("source_id must be nonempty and unique")
        seen.add(source_id)
        if row.get("evidence_status") not in TERMINAL_STATUSES:
            raise CrossSourceV3Error(f"nonterminal source present: {source_id}")
        for key in ("stable_origin_id", "stable_object_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise CrossSourceV3Error(f"{source_id}: missing {key}")
        normalized_rows.append(row)

    edges_raw = inventory.get("lineage_edges", [])
    if not isinstance(edges_raw, list):
        raise CrossSourceV3Error("lineage_edges must be a list")
    edges: list[dict[str, Any]] = []
    for raw in edges_raw:
        if not isinstance(raw, Mapping):
            raise CrossSourceV3Error("lineage edge must be an object")
        edge = dict(raw)
        left = edge.get("left_source_id")
        right = edge.get("right_source_id")
        relation = edge.get("relation")
        if left not in seen or right not in seen or left == right:
            raise CrossSourceV3Error("lineage edge endpoints must be distinct known sources")
        if relation not in RELATION_MATCH_TYPES:
            raise CrossSourceV3Error(f"unsupported lineage relation: {relation}")
        if not isinstance(edge.get("capacity_collapsing"), bool):
            raise CrossSourceV3Error("lineage edge requires capacity_collapsing bool")
        if not isinstance(edge.get("independence_collapsing"), bool):
            raise CrossSourceV3Error("lineage edge requires independence_collapsing bool")
        if relation == "sibling_same_origin" and edge["capacity_collapsing"]:
            raise CrossSourceV3Error("sibling_same_origin cannot collapse capacity without copy evidence")
        if not isinstance(edge.get("evidence"), str) or not edge["evidence"]:
            raise CrossSourceV3Error("lineage edge requires evidence")
        edges.append(edge)
    return normalized_rows, edges


def _as_v1_inventory(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": "12-6.data298-cross-source-inventory.v1",
        "local_free_only": True,
        "sources": [dict(row) for row in rows],
    }


def _lineage_matches(
    fingerprints: Sequence[dict[str, Any]],
    edges: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id = {item["row"]["source_id"]: item for item in fingerprints}
    matches: list[dict[str, Any]] = []

    stable_objects: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in fingerprints:
        stable_objects[item["row"]["stable_object_id"]].append(item)
    for group in stable_objects.values():
        if len(group) < 2:
            continue
        for index, left in enumerate(group):
            for right in group[index + 1 :]:
                a = left["row"]
                b = right["row"]
                matches.append(
                    {
                        "left_source_id": a["source_id"],
                        "right_source_id": b["source_id"],
                        "left_source_family": a["source_family"],
                        "right_source_family": b["source_family"],
                        "match_type": "lineage_same_origin_alias",
                        "score": 1.0,
                        "cross_source_family": a["source_family"] != b["source_family"],
                        "capacity_collapsing": True,
                        "independence_collapsing": True,
                        "evidence_class": "stable_object_identity",
                    }
                )

    for edge in edges:
        left = by_id[edge["left_source_id"]]["row"]
        right = by_id[edge["right_source_id"]]["row"]
        matches.append(
            {
                "left_source_id": left["source_id"],
                "right_source_id": right["source_id"],
                "left_source_family": left["source_family"],
                "right_source_family": right["source_family"],
                "match_type": RELATION_MATCH_TYPES[edge["relation"]],
                "score": 1.0,
                "cross_source_family": left["source_family"] != right["source_family"],
                "capacity_collapsing": edge["capacity_collapsing"],
                "independence_collapsing": edge["independence_collapsing"],
                "evidence_class": "declared_lineage_authority",
            }
        )
    return matches


def _summary_for_ids(
    fingerprints: Sequence[dict[str, Any]],
    matches: Sequence[dict[str, Any]],
    ids: set[str],
) -> dict[str, Any]:
    selected = [item for item in fingerprints if item["row"]["source_id"] in ids]
    by_id = {item["row"]["source_id"]: item for item in selected}
    collapse_pairs = [
        (match["left_source_id"], match["right_source_id"])
        for match in matches
        if match.get("capacity_collapsing") is True
        and match["match_type"] in CAPACITY_COLLAPSE_MATCH_TYPES
        and match["left_source_id"] in ids
        and match["right_source_id"] in ids
    ]
    components = v1._components(sorted(ids), collapse_pairs)
    duplicate_clusters = [component for component in components if len(component) > 1]
    before = sum(item["row"]["declared_capacity_bytes"] for item in selected)
    after = sum(
        max(by_id[source_id]["row"]["declared_capacity_bytes"] for source_id in component)
        for component in components
    )

    origin_ids = sorted({item["row"]["stable_origin_id"] for item in selected})
    origin_pairs: set[tuple[str, str]] = set()
    for match in matches:
        if not match.get("independence_collapsing"):
            continue
        left_id = match["left_source_id"]
        right_id = match["right_source_id"]
        if left_id not in ids or right_id not in ids:
            continue
        left_origin = by_id[left_id]["row"]["stable_origin_id"]
        right_origin = by_id[right_id]["row"]["stable_origin_id"]
        if left_origin != right_origin:
            origin_pairs.add(tuple(sorted((left_origin, right_origin))))
    origin_components = v1._components(origin_ids, sorted(origin_pairs))

    return {
        "source_count": len(selected),
        "declared_source_family_count": len({item["row"]["source_family"] for item in selected}),
        "stable_origin_count": len(origin_ids),
        "effective_independent_origin_count": len(origin_components),
        "raw_bytes_before": sum(item["raw_bytes"] for item in selected),
        "declared_capacity_bytes_before": before,
        "conservative_unique_capacity_bytes_after": after,
        "duplicate_discount_bytes": before - after,
        "duplicate_discount_fraction": round((before - after) / before, 12) if before else 0.0,
        "duplicate_cluster_count": len(duplicate_clusters),
        "duplicate_clusters": duplicate_clusters,
        "origin_clusters": origin_components,
    }


def audit_payloads(inventory: Mapping[str, Any], payloads: Mapping[str, bytes]) -> dict[str, Any]:
    rows, edges = _validate_inventory(inventory)
    expected_ids = {row["source_id"] for row in rows}
    if set(payloads) != expected_ids:
        raise CrossSourceV3Error("payload coverage must equal the exact terminal source inventory")

    v1_inventory = _as_v1_inventory(rows)
    validated_rows = v1._validate_inventory(v1_inventory)
    fingerprints = [v1._fingerprint(row, payloads[row["source_id"]]) for row in validated_rows]
    matches = [
        match
        for index, left in enumerate(fingerprints)
        for right in fingerprints[index + 1 :]
        for match in v1._pair_matches(left, right)
    ]
    matches.extend(_lineage_matches(fingerprints, edges))
    matches.sort(key=lambda item: (item["left_source_id"], item["right_source_id"], item["match_type"]))

    all_ids = {item["row"]["source_id"] for item in fingerprints}
    terminal = _summary_for_ids(fingerprints, matches, all_ids)
    modalities = sorted({item["row"]["modality"] for item in fingerprints})
    by_modality = {
        modality: _summary_for_ids(
            fingerprints,
            matches,
            {
                item["row"]["source_id"]
                for item in fingerprints
                if item["row"]["modality"] == modality
            },
        )
        for modality in modalities
    }
    match_counts = Counter(match["match_type"] for match in matches)
    collapsing_counts = Counter(
        match["match_type"] for match in matches if match.get("capacity_collapsing") is True
    )
    sources = [
        {
            "source_id": item["row"]["source_id"],
            "source_family": item["row"]["source_family"],
            "stable_origin_id_sha256": v1._sha256(item["row"]["stable_origin_id"].encode()),
            "stable_object_id_sha256": v1._sha256(item["row"]["stable_object_id"].encode()),
            "modality": item["row"]["modality"],
            "evidence_status": item["row"]["evidence_status"],
            "declared_capacity_bytes": item["row"]["declared_capacity_bytes"],
            "verified_raw_bytes": item["raw_bytes"],
            "verified_raw_sha256": item["raw_sha256"],
            "normalized_sha256": item["normalized_sha256"],
        }
        for item in fingerprints
    ]
    sources.sort(key=lambda item: item["source_id"])

    core = {
        "schema_version": SCHEMA,
        "algorithm": ALGORITHM,
        "local_free_only": True,
        "model_training_executed": False,
        "source_admission_authority": False,
        "source_count": len(fingerprints),
        "matching_authority": "DATA-232 / DATA-298 exact, normalized, near, fragment and code-skeleton semantics plus NEXT100-065 explicit lineage graph",
        "thresholds": dict(v1.DEFAULT_THRESHOLDS),
        "sources": sources,
        "matches": matches,
        "match_counts": dict(sorted(match_counts.items())),
        "capacity_collapsing_match_counts": dict(sorted(collapsing_counts.items())),
        "terminal_candidates": {**terminal, "by_modality": by_modality},
        "capacity_policy": {
            "metric": "conservative_unique_capacity_bytes_after",
            "connected_component_rule": "count at most the largest declared-capacity member of each connected capacity-collapsing duplicate cluster",
            "url_rule": "URL distinctness is never independence evidence",
            "stable_object_rule": "identical stable_object_id collapses capacity even when acquisition URLs or wrapper bytes differ",
            "origin_rule": "stable_origin_id and explicit lineage edges determine independence accounting",
            "sibling_rule": "same-origin sibling files collapse independence but retain capacity unless byte/copy/derivative evidence creates a capacity-collapsing edge",
        },
        "raw_text_emitted": False,
    }
    return {**core, "report_sha256": v1._sha256(v1._canonical_bytes(core))}


def audit_live(inventory: Mapping[str, Any]) -> dict[str, Any]:
    rows, _ = _validate_inventory(inventory)
    payloads = {row["source_id"]: v1.fetch_exact_source(row["acquisition_url"]) for row in rows}
    return audit_payloads(inventory, payloads)


def verify_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != SCHEMA:
        raise CrossSourceV3Error("unsupported report schema")
    expected = report.get("report_sha256")
    core = dict(report)
    core.pop("report_sha256", None)
    if expected != v1._sha256(v1._canonical_bytes(core)):
        raise CrossSourceV3Error("report self-hash mismatch")
    if report.get("local_free_only") is not True:
        raise CrossSourceV3Error("LOCAL_FREE invariant failed")
    if report.get("model_training_executed") is not False:
        raise CrossSourceV3Error("training invariant failed")
    if report.get("raw_text_emitted") is not False:
        raise CrossSourceV3Error("raw text emission is forbidden")
    scope = report.get("terminal_candidates", {})
    if scope.get("conservative_unique_capacity_bytes_after", 0) > scope.get("declared_capacity_bytes_before", -1):
        raise CrossSourceV3Error("deduplication inflated capacity")


def write_report(report: Mapping[str, Any], path: str | Path) -> None:
    v1.write_report(report, path)
