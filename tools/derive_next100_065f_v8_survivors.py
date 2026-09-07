#!/usr/bin/env python3
"""Derive a deterministic hash-only post-dedup source survivor authority for V8.

The incumbent V3 matcher accounts each capacity-collapsing connected component at
at most its largest declared-capacity member.  This tool does not change that
scientific rule.  It makes the implied representative set explicit so the next
record-materialization stage has an unambiguous source-object input authority.

This is still source-object authority, not a training-record inventory, tokenizer
input, packed loss ledger, or training authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA = "12-6.next100-065f-post-dedup-survivors.v1"
SELECTION_RULE = "largest_declared_capacity_then_lexicographically_smallest_source_id"


class SurvivorAuthorityError(RuntimeError):
    """Raised when the retained V8 report cannot yield one exact survivor set."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SurvivorAuthorityError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _source_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    nested = report.get("dedup_v3")
    _require(isinstance(nested, Mapping), "V8 report missing nested V3 report")
    rows = nested.get("sources")
    _require(isinstance(rows, list) and rows, "nested V3 report has no source rows")
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in rows:
        _require(isinstance(raw, Mapping), "nested source row must be an object")
        source_id = raw.get("source_id")
        capacity = raw.get("declared_capacity_bytes")
        _require(isinstance(source_id, str) and source_id and source_id not in seen, "invalid/duplicate source_id")
        _require(isinstance(capacity, int) and capacity > 0, f"invalid declared capacity: {source_id}")
        for key in (
            "source_family",
            "modality",
            "verified_raw_sha256",
            "normalized_sha256",
            "stable_origin_id_sha256",
            "stable_object_id_sha256",
        ):
            _require(isinstance(raw.get(key), str) and raw.get(key), f"missing {key}: {source_id}")
        seen.add(source_id)
        normalized.append(dict(raw))
    return normalized


def derive_survivor_authority(report: Mapping[str, Any]) -> dict[str, Any]:
    outer_hash = report.get("report_sha256")
    _require(isinstance(outer_hash, str) and len(outer_hash) == 64, "missing V8 report identity")
    vector = report.get("source_vector")
    _require(isinstance(vector, Mapping), "V8 report missing source vector")
    nested = report.get("dedup_v3")
    _require(isinstance(nested, Mapping), "V8 report missing nested V3 report")
    nested_hash = nested.get("report_sha256")
    _require(isinstance(nested_hash, str) and len(nested_hash) == 64, "missing nested V3 report identity")

    rows = _source_rows(report)
    by_id = {row["source_id"]: row for row in rows}
    expected_source_count = vector.get("source_object_count")
    _require(expected_source_count == len(rows), "outer/nested source-count mismatch")

    terminal = nested.get("terminal_candidates")
    _require(isinstance(terminal, Mapping), "nested V3 report missing terminal candidates")
    clusters = terminal.get("duplicate_clusters")
    _require(isinstance(clusters, list), "nested V3 duplicate clusters missing")

    cluster_members: set[str] = set()
    representatives: dict[str, str] = {}
    dropped: set[str] = set()
    normalized_clusters: list[dict[str, Any]] = []
    for raw_cluster in clusters:
        _require(isinstance(raw_cluster, Sequence) and not isinstance(raw_cluster, (str, bytes)), "invalid duplicate cluster")
        cluster = sorted(str(value) for value in raw_cluster)
        _require(len(cluster) >= 2 and len(cluster) == len(set(cluster)), "duplicate cluster must contain >=2 unique ids")
        _require(all(source_id in by_id for source_id in cluster), "duplicate cluster references unknown source")
        _require(not (cluster_members & set(cluster)), "duplicate clusters overlap")
        cluster_members.update(cluster)

        maximum = max(int(by_id[source_id]["declared_capacity_bytes"]) for source_id in cluster)
        tied = sorted(
            source_id
            for source_id in cluster
            if int(by_id[source_id]["declared_capacity_bytes"]) == maximum
        )
        survivor = tied[0]
        representatives["\x1f".join(cluster)] = survivor
        dropped.update(source_id for source_id in cluster if source_id != survivor)
        normalized_clusters.append(
            {
                "member_source_ids": cluster,
                "selected_source_id": survivor,
                "selected_declared_capacity_bytes": maximum,
            }
        )

    survivors = sorted(source_id for source_id in by_id if source_id not in dropped)
    survivor_rows = [
        {
            "source_id": source_id,
            "source_family": by_id[source_id]["source_family"],
            "modality": by_id[source_id]["modality"],
            "declared_capacity_bytes": by_id[source_id]["declared_capacity_bytes"],
            "verified_raw_sha256": by_id[source_id]["verified_raw_sha256"],
            "normalized_sha256": by_id[source_id]["normalized_sha256"],
            "stable_origin_id_sha256": by_id[source_id]["stable_origin_id_sha256"],
            "stable_object_id_sha256": by_id[source_id]["stable_object_id_sha256"],
        }
        for source_id in survivors
    ]
    survivor_capacity = sum(int(row["declared_capacity_bytes"]) for row in survivor_rows)
    expected_post = vector.get("conservative_unique_capacity_bytes_after_global_dedup")
    _require(survivor_capacity == expected_post, "survivor capacity does not reproduce V3 conservative capacity")
    expected_discount = vector.get("duplicate_discount_bytes")
    pre = vector.get("source_capacity_bytes_before_global_dedup")
    _require(isinstance(pre, int) and pre - survivor_capacity == expected_discount, "survivor discount does not reproduce V8 summary")
    _require(len(normalized_clusters) == vector.get("duplicate_cluster_count"), "survivor cluster count does not reproduce V8 summary")

    modality_counts = {
        modality: {
            "source_object_count": sum(1 for row in survivor_rows if row["modality"] == modality),
            "declared_capacity_bytes": sum(
                int(row["declared_capacity_bytes"]) for row in survivor_rows if row["modality"] == modality
            ),
        }
        for modality in ("uk", "en", "code")
    }

    core: dict[str, Any] = {
        "schema_version": SCHEMA,
        "selection_rule": SELECTION_RULE,
        "v8_report_sha256": outer_hash,
        "nested_v3_report_sha256": nested_hash,
        "pre_dedup_source_object_count": len(rows),
        "post_dedup_survivor_source_object_count": len(survivor_rows),
        "pre_dedup_declared_capacity_bytes": pre,
        "post_dedup_declared_capacity_bytes": survivor_capacity,
        "duplicate_discount_bytes": expected_discount,
        "duplicate_cluster_count": len(normalized_clusters),
        "duplicate_clusters": normalized_clusters,
        "survivors": survivor_rows,
        "by_modality": modality_counts,
        "truth_boundary": {
            "source_object_authority_only": True,
            "training_record_inventory_materialized": False,
            "evaluation_decontamination_passed": False,
            "tokenizer_fit_authorized": False,
            "authorized_training_exposure": 0,
            "model_training_executed": False,
            "final_test_payload_read": False,
            "paid_compute_used": False,
        },
    }
    core["survivor_authority_sha256"] = _sha256(_canonical_bytes(core))
    return core


def verify_survivor_authority(report: Mapping[str, Any], authority: Mapping[str, Any]) -> None:
    expected = derive_survivor_authority(report)
    _require(dict(authority) == expected, "survivor authority is not the exact deterministic derivation")
    identity = authority.get("survivor_authority_sha256")
    body = dict(authority)
    body.pop("survivor_authority_sha256", None)
    _require(identity == _sha256(_canonical_bytes(body)), "survivor authority self-hash mismatch")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("derive", "verify"))
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = _read_json(args.report)
    if args.command == "derive":
        authority = derive_survivor_authority(report)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(authority, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        print(f"survivor_authority_sha256={authority['survivor_authority_sha256']}")
        print(f"post_dedup_survivor_source_object_count={authority['post_dedup_survivor_source_object_count']}")
        return 0
    authority = _read_json(args.output)
    verify_survivor_authority(report, authority)
    print("PASS_V8_SURVIVOR_AUTHORITY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
