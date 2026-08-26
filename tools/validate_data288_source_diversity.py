#!/usr/bin/env python3
"""Deterministic DATA-288 source-family diversity audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

SCHEMA = "12-6.data288-source-diversity-audit.v2"


class AuditError(RuntimeError):
    pass


def canonical_bytes(value: Any) -> bytes:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return encoded.encode("utf-8")


def stable_identity(value: Any) -> str:
    payload = b"DATA288-SOURCE-DIVERSITY-AUDIT-V2\0" + canonical_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def nearest_rank(sorted_values: list[int], probability: float) -> int:
    if not sorted_values:
        raise AuditError("empty distribution")
    rank = math.ceil(probability * len(sorted_values))
    return sorted_values[max(1, rank) - 1]


def distribution(values: list[int]) -> dict[str, Any]:
    ordered = sorted(values)
    n = len(ordered)
    return {
        "count": n,
        "ordered_bytes": ordered,
        "min": ordered[0],
        "p25_nearest_rank": nearest_rank(ordered, 0.25),
        "median": (
            ordered[n // 2]
            if n % 2
            else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
        ),
        "mean": sum(ordered) / n,
        "p75_nearest_rank": nearest_rank(ordered, 0.75),
        "max": ordered[-1],
    }


def _check_inputs(data: dict[str, Any]) -> None:
    if data.get("schema_version") != "12-6.data288-source-diversity-inputs.v2":
        raise AuditError("unexpected input schema")
    if data.get("worker_id") != "DATA-288-SOURCE-DIVERSITY-AUDIT-V2":
        raise AuditError("unexpected worker id")
    if data.get("local_free_only") is not True:
        raise AuditError("LOCAL_FREE must be true")

    authorities = data["terminal_authorities"]
    if authorities["DATA-229"]["status"] != "TERMINAL_SUCCESS":
        raise AuditError("DATA-229 is not terminal success")
    if authorities["DATA-227"]["status"] != "TERMINAL_SUCCESS":
        raise AuditError("DATA-227 is not terminal success")
    if authorities["DATA-228"]["status"] != "EXCLUDED_TERMINAL_FAILURE":
        raise AuditError("DATA-228 must remain excluded")
    if authorities["DATA-228"]["dedicated_workflow_conclusion"] != "failure":
        raise AuditError("DATA-228 failure evidence changed")

    seen_ids: set[str] = set()
    raw_hashes: set[str] = set()
    norm_hashes: set[str] = set()
    for source in data["sources"]:
        if source["source_id"] in seen_ids:
            raise AuditError("duplicate source_id")
        seen_ids.add(source["source_id"])
        if source["training_rights"] != "ALLOWED":
            raise AuditError(f"training rights not allowed: {source['source_id']}")
        if source["raw_sha256"] in raw_hashes:
            raise AuditError("raw exact duplicate source object")
        if source["normalized_sha256"] in norm_hashes:
            raise AuditError("normalized exact duplicate source object")
        raw_hashes.add(source["raw_sha256"])
        norm_hashes.add(source["normalized_sha256"])
        if source["raw_bytes"] <= 0 or source["normalized_unique_bytes"] <= 0:
            raise AuditError("source byte count must be positive")

    overlap = data["independent_overlap_audit"]
    if overlap["all_raw_sha256_distinct"] is not True:
        raise AuditError("raw duplicate evidence present")
    if overlap["all_normalized_sha256_distinct"] is not True:
        raise AuditError("normalized duplicate evidence present")
    if overlap["cross_family_duplicate_or_mirror_edges"] != 0:
        raise AuditError("cross-family duplicate/mirror edge present")


def build_report(data: dict[str, Any]) -> dict[str, Any]:
    _check_inputs(data)
    family_bytes: dict[str, int] = defaultdict(int)
    family_raw: dict[str, int] = defaultdict(int)
    family_stratum: dict[str, str] = {}
    files_by_family: dict[str, list[str]] = defaultdict(list)
    stratum_bytes: dict[str, int] = defaultdict(int)
    stratum_raw: dict[str, int] = defaultdict(int)
    stratum_families: dict[str, set[str]] = defaultdict(set)

    for source in data["sources"]:
        family = source["family_id"]
        stratum = source["stratum"]
        prior = family_stratum.setdefault(family, stratum)
        if prior != stratum:
            raise AuditError(f"family crosses strata without explicit lineage handling: {family}")
        family_bytes[family] += int(source["normalized_unique_bytes"])
        family_raw[family] += int(source["raw_bytes"])
        files_by_family[family].append(source["source_id"])
        stratum_bytes[stratum] += int(source["normalized_unique_bytes"])
        stratum_raw[stratum] += int(source["raw_bytes"])
        stratum_families[stratum].add(family)

    normalized_total = sum(family_bytes.values())
    raw_total = sum(family_raw.values())
    family_prob = {key: value / normalized_total for key, value in family_bytes.items()}
    raw_family_prob = {key: value / raw_total for key, value in family_raw.items()}
    entropy_nats = -sum(p * math.log(p) for p in family_prob.values())
    entropy_bits = entropy_nats / math.log(2)
    effective = math.exp(entropy_nats)

    family_rows = []
    for family in sorted(family_bytes):
        family_rows.append(
            {
                "family_id": family,
                "stratum": family_stratum[family],
                "file_count": len(files_by_family[family]),
                "source_ids": sorted(files_by_family[family]),
                "raw_bytes": family_raw[family],
                "normalized_unique_bytes": family_bytes[family],
                "normalized_share": family_prob[family],
                "raw_share_diagnostic": raw_family_prob[family],
            }
        )

    within_stratum_top: dict[str, float] = {}
    for stratum in sorted(stratum_bytes):
        within_stratum_top[stratum] = max(
            family_bytes[f] / stratum_bytes[stratum] for f in stratum_families[stratum]
        )

    gates = data["next_corpus_diversity_gates"]
    gate_results = {
        "independent_family_count": {
            "value": len(family_bytes),
            "threshold": gates["minimum_independent_families_total"],
            "pass": len(family_bytes) >= gates["minimum_independent_families_total"],
        },
        "family_count_by_stratum": {
            "value": {k: len(stratum_families.get(k, set())) for k in ("uk", "en", "code")},
            "threshold": gates["minimum_independent_families_by_stratum"],
        },
        "top_family_share": {
            "value": max(family_prob.values()),
            "threshold": gates["maximum_top_family_share_of_normalized_unique_bytes"],
            "pass": (
                max(family_prob.values())
                <= gates["maximum_top_family_share_of_normalized_unique_bytes"]
            ),
        },
        "within_stratum_top_family_share": {
            "value": within_stratum_top,
            "threshold": gates["maximum_top_family_share_within_each_stratum"],
            "pass": all(
                value <= gates["maximum_top_family_share_within_each_stratum"]
                for value in within_stratum_top.values()
            ),
        },
        "shannon_effective_family_count": {
            "value": effective,
            "threshold": gates["minimum_shannon_effective_family_count"],
            "pass": effective >= gates["minimum_shannon_effective_family_count"],
        },
        "cross_family_duplicate_or_mirror_edges": {
            "value": data["independent_overlap_audit"][
                "cross_family_duplicate_or_mirror_edges"
            ],
            "threshold": gates["maximum_cross_family_duplicate_or_mirror_edges"],
            "pass": (
                data["independent_overlap_audit"]["cross_family_duplicate_or_mirror_edges"]
                <= gates["maximum_cross_family_duplicate_or_mirror_edges"]
            ),
        },
    }
    count_by_stratum = gate_results["family_count_by_stratum"]
    count_by_stratum["pass"] = all(
        count_by_stratum["value"][key] >= count_by_stratum["threshold"][key]
        for key in ("uk", "en", "code")
    )

    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "worker_id": data["worker_id"],
        "authority_cutoff_utc": data["cutoff_utc"],
        "local_free_only": True,
        "training_executed": False,
        "terminal_inputs": data["terminal_authorities"],
        "excluded_nonadmitted_candidates": ["DATA-228"],
        "object_count": len(data["sources"]),
        "independent_family_count": len(family_bytes),
        "independent_family_count_by_stratum": {
            key: len(stratum_families.get(key, set())) for key in ("uk", "en", "code")
        },
        "raw_source_bytes": raw_total,
        "normalized_unique_bytes": normalized_total,
        "normalized_unique_bytes_by_stratum": {
            key: stratum_bytes.get(key, 0) for key in ("uk", "en", "code")
        },
        "raw_bytes_by_stratum_diagnostic": {
            key: stratum_raw.get(key, 0) for key in ("uk", "en", "code")
        },
        "family_rows": family_rows,
        "top_family": max(family_prob, key=lambda key: (family_prob[key], key)),
        "top_family_share_normalized_unique_bytes": max(family_prob.values()),
        "top_family_share_raw_bytes_diagnostic": max(raw_family_prob.values()),
        "shannon_entropy_nats": entropy_nats,
        "shannon_entropy_bits": entropy_bits,
        "shannon_effective_family_count": effective,
        "normalized_entropy_fraction_of_observed_max": entropy_nats / math.log(len(family_bytes)),
        "file_length_distribution_raw_bytes": distribution(
            [int(x["raw_bytes"]) for x in data["sources"]]
        ),
        "file_length_distribution_normalized_unique_bytes": distribution(
            [int(x["normalized_unique_bytes"]) for x in data["sources"]]
        ),
        "family_length_distribution_normalized_unique_bytes": distribution(
            list(family_bytes.values())
        ),
        "overlap_and_lineage_audit": data["independent_overlap_audit"],
        "family_resolution_policy": data["family_resolution_policy"],
        "next_corpus_diversity_gates": gates,
        "gate_results_on_current_terminal_inventory": gate_results,
    }
    report["next_corpus_diversity_status"] = (
        "PASS"
        if all(item["pass"] for item in gate_results.values())
        else "BLOCKED_SOURCE_DIVERSITY"
    )
    report["audit_identity_sha256"] = stable_identity(report)
    return report


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AuditError("JSON root must be object")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path("configs/data/data288_source_diversity_inputs_v2.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("evidence/data288/source_diversity_audit_v2.json"),
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    report = build_report(load(args.inputs))
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.verify:
        if not args.output.exists():
            raise AuditError(f"missing report: {args.output}")
        if args.output.read_text(encoding="utf-8") != encoded:
            raise AuditError(
                "committed DATA-288 report is not byte-identical to deterministic rebuild"
            )
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(json.dumps({
        "audit_identity_sha256": report["audit_identity_sha256"],
        "independent_family_count": report["independent_family_count"],
        "normalized_unique_bytes": report["normalized_unique_bytes"],
        "status": report["next_corpus_diversity_status"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
