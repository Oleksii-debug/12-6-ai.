#!/usr/bin/env python3
"""DATA-312 external source diversity qualification.

Reads the exact DATA-300 frozen candidate contract, groups source objects by the
declared independent family field, and evaluates the DATA-295 hard diversity
gates bound by that contract. No network access and no model compute.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

WORKER_ID = "DATA-312-EXTERNAL-DIVERSITY-QUALIFICATION"
SCHEMA_VERSION = "12-6.data312-external-diversity-qualification.v1"
EXPECTED_CONTRACT_IDENTITY = "07d7beaaff4616e839450de6af3d407855c832bf75a24a959d1a12de5d9364e5"
CANDIDATE_HEAD_SHA = "8ea7f830e50a23754d189dd4134f4afad76a7ee9"
CANDIDATE_CONTRACT_BLOB_SHA1 = "39d4fa07ea17e66e042a3ccb1a55b8e5e1c5d7bf"
POLICY_HEAD_SHA = "9498e78da5a9fa239e39f5c145d6fd986ebed7b4"
POLICY_CONFIG_BLOB_SHA1 = "a3195511fd85805ab57b481f19e86ddc8e951ce0"
FAMILY_IDENTITY_RULE = (
    "Independent upstream publisher/project lineage; mirrors, forks, vendored copies, "
    "generated derivatives, and duplicated documents do not create new families."
)


def _entropy(byte_counts: list[int]) -> dict[str, float]:
    total = sum(byte_counts)
    if total <= 0:
        return {
            "shannon_nats": 0.0,
            "shannon_bits": 0.0,
            "effective_family_count": 0.0,
            "normalized_entropy": 0.0,
        }
    probabilities = [value / total for value in byte_counts if value > 0]
    h_nats = -sum(p * math.log(p) for p in probabilities)
    h_bits = h_nats / math.log(2.0)
    family_count = len(probabilities)
    normalized = h_nats / math.log(family_count) if family_count > 1 else 0.0
    return {
        "shannon_nats": round(h_nats, 12),
        "shannon_bits": round(h_bits, 12),
        "effective_family_count": round(math.exp(h_nats), 12),
        "normalized_entropy": round(normalized, 12),
    }


def _share(value: int, total: int) -> float:
    return round(value / total, 12) if total else 0.0


def _pct(value: int, total: int) -> float:
    return round(100.0 * value / total, 6) if total else 0.0


def qualify(contract: dict[str, Any]) -> dict[str, Any]:
    if contract.get("contract_identity_sha256") != EXPECTED_CONTRACT_IDENTITY:
        raise ValueError("unexpected DATA-300 contract identity")
    if contract.get("corpus_state") == "TERMINAL":
        raise ValueError("unexpected terminal corpus state; DATA-312 is bound to the frozen candidate")

    inventory = contract["exact_training_candidate_inventory"]
    sources = inventory["sources"]
    balance = contract["terminal_component_lock"]["balance"]

    if balance["head_sha"] != POLICY_HEAD_SHA:
        raise ValueError("DATA-295 head binding mismatch")
    if balance["config_git_blob_sha1"] != POLICY_CONFIG_BLOB_SHA1:
        raise ValueError("DATA-295 config blob binding mismatch")

    min_families = int(balance["minimum_independent_families_per_stratum"])
    max_total_share = float(balance["maximum_family_share_total"])
    max_stratum_share = float(balance["maximum_family_share_within_stratum"])
    expected_stratum_fraction = dict(balance["stratum_fraction"])

    family_bytes: dict[str, int] = defaultdict(int)
    family_source_counts: dict[str, int] = defaultdict(int)
    family_strata: dict[str, set[str]] = defaultdict(set)
    family_modalities: dict[str, set[str]] = defaultdict(set)
    stratum_bytes: dict[str, int] = defaultdict(int)
    stratum_sources: dict[str, int] = defaultdict(int)
    stratum_families: dict[str, set[str]] = defaultdict(set)
    modality_bytes: dict[str, int] = defaultdict(int)
    modality_sources: dict[str, int] = defaultdict(int)
    modality_families: dict[str, set[str]] = defaultdict(set)

    for source in sources:
        family = source.get("family")
        stratum = source.get("language")
        modality = source.get("modality")
        byte_count = int(source.get("normalized_bytes", 0))
        if not family or not stratum or not modality or byte_count < 0:
            raise ValueError(f"invalid source record: {source.get('source_id', '<unknown>')}")
        family_bytes[family] += byte_count
        family_source_counts[family] += 1
        family_strata[family].add(stratum)
        family_modalities[family].add(modality)
        stratum_bytes[stratum] += byte_count
        stratum_sources[stratum] += 1
        stratum_families[stratum].add(family)
        modality_bytes[modality] += byte_count
        modality_sources[modality] += 1
        modality_families[modality].add(family)

    total_bytes = sum(family_bytes.values())
    if total_bytes != int(inventory["admitted_source_bytes"]):
        raise ValueError("candidate source-byte total does not match contract")
    if len(sources) != int(inventory["source_count"]):
        raise ValueError("candidate source count does not match contract")
    if len(family_bytes) != int(inventory["independent_family_count"]):
        raise ValueError("independent family count does not match contract")

    declared_stratum_bytes = {k: int(v) for k, v in inventory["by_stratum_bytes"].items()}
    if dict(sorted(stratum_bytes.items())) != dict(sorted(declared_stratum_bytes.items())):
        raise ValueError("stratum byte totals do not match contract")

    declared_stratum_families = {k: int(v) for k, v in inventory["by_stratum_families"].items()}
    measured_stratum_families = {k: len(v) for k, v in stratum_families.items()}
    if dict(sorted(measured_stratum_families.items())) != dict(sorted(declared_stratum_families.items())):
        raise ValueError("stratum family counts do not match contract")

    families = []
    for family, byte_count in sorted(family_bytes.items(), key=lambda kv: (-kv[1], kv[0])):
        families.append(
            {
                "family": family,
                "source_object_count": family_source_counts[family],
                "strata": sorted(family_strata[family]),
                "modalities": sorted(family_modalities[family]),
                "inventory_unique_bytes": byte_count,
                "share_of_total": _share(byte_count, total_bytes),
                "share_of_total_percent": _pct(byte_count, total_bytes),
            }
        )

    strata = {}
    stratum_top_shares = {}
    for stratum in sorted(stratum_bytes):
        family_values = {
            family: family_bytes[family]
            for family in sorted(stratum_families[stratum])
        }
        top_family, top_bytes = max(family_values.items(), key=lambda kv: (kv[1], kv[0]))
        top_share = top_bytes / stratum_bytes[stratum]
        stratum_top_shares[stratum] = top_share
        strata[stratum] = {
            "source_object_count": stratum_sources[stratum],
            "independent_family_count": len(family_values),
            "inventory_unique_bytes": stratum_bytes[stratum],
            "share_of_total": _share(stratum_bytes[stratum], total_bytes),
            "share_of_total_percent": _pct(stratum_bytes[stratum], total_bytes),
            "top_family": top_family,
            "top_family_share_within_stratum": round(top_share, 12),
            "top_family_share_within_stratum_percent": round(100.0 * top_share, 6),
            "entropy": _entropy(list(family_values.values())),
        }

    modalities = {}
    for modality in sorted(modality_bytes):
        modalities[modality] = {
            "source_object_count": modality_sources[modality],
            "independent_family_count": len(modality_families[modality]),
            "inventory_unique_bytes": modality_bytes[modality],
            "share_of_total": _share(modality_bytes[modality], total_bytes),
            "share_of_total_percent": _pct(modality_bytes[modality], total_bytes),
        }

    top_family = families[0]
    total_share_violations = [
        family["family"]
        for family in families
        if family["share_of_total"] > max_total_share
    ]
    min_family_failures = [
        stratum
        for stratum in sorted(expected_stratum_fraction)
        if len(stratum_families.get(stratum, set())) < min_families
    ]
    within_stratum_violations = [
        stratum
        for stratum in sorted(stratum_top_shares)
        if stratum_top_shares[stratum] > max_stratum_share
    ]

    gate_results = {
        "minimum_independent_families_per_stratum": {
            "threshold": min_families,
            "pass": not min_family_failures,
            "failing_strata": min_family_failures,
        },
        "maximum_family_share_of_total": {
            "threshold": max_total_share,
            "threshold_percent": round(100.0 * max_total_share, 6),
            "pass": not total_share_violations,
            "violating_families": total_share_violations,
        },
        "maximum_family_share_within_stratum": {
            "threshold": max_stratum_share,
            "threshold_percent": round(100.0 * max_stratum_share, 6),
            "pass": not within_stratum_violations,
            "violating_strata": within_stratum_violations,
        },
    }
    hard_pass = all(item["pass"] for item in gate_results.values())

    return {
        "schema_version": SCHEMA_VERSION,
        "worker_id": WORKER_ID,
        "execution_profile": "LOCAL_FREE",
        "candidate_authority": {
            "branch": "data300/corpus-v03-frozen-build-contract-20260826",
            "head_sha": CANDIDATE_HEAD_SHA,
            "contract_path": "configs/data/data300_corpus_v03_frozen_build_contract_v2.json",
            "contract_git_blob_sha1": CANDIDATE_CONTRACT_BLOB_SHA1,
            "contract_identity_sha256": EXPECTED_CONTRACT_IDENTITY,
            "corpus_state": contract["corpus_state"],
        },
        "preregistered_diversity_authority": {
            "worker_id": "DATA-295-BALANCE-POLICY-20M-V1",
            "head_sha": POLICY_HEAD_SHA,
            "config_git_blob_sha1": POLICY_CONFIG_BLOB_SHA1,
            "dedicated_workflow_run": balance["dedicated_workflow_run"],
            "dedicated_workflow_conclusion": balance["dedicated_workflow_conclusion"],
            "selected_policy": "continuity_45_35_20",
            "stratum_fraction": expected_stratum_fraction,
            "family_identity_rule": FAMILY_IDENTITY_RULE,
            "hard_thresholds": {
                "minimum_independent_families_per_stratum": min_families,
                "maximum_family_share_of_total": max_total_share,
                "maximum_family_share_within_stratum": max_stratum_share,
            },
        },
        "truth_boundary": {
            "measurement_unit": "authority-declared unique training-eligible source bytes at the frozen candidate inventory",
            "post_wave3_terminal_deduplicated_corpus_unique_bytes_claimed": False,
            "entropy_threshold_preregistered": False,
            "entropy_and_effective_family_count_are_descriptive_only": True,
            "source_objects_are_not_independent_families": True,
            "same_declared_family_is_collapsed_across_multiple_files": True,
        },
        "measurements": {
            "source_object_count": len(sources),
            "independent_family_count": len(families),
            "inventory_unique_bytes": total_bytes,
            "by_language_or_stratum": strata,
            "by_modality": modalities,
            "families": families,
            "top_family": {
                "family": top_family["family"],
                "inventory_unique_bytes": top_family["inventory_unique_bytes"],
                "share_of_total": top_family["share_of_total"],
                "share_of_total_percent": top_family["share_of_total_percent"],
            },
            "overall_family_entropy": _entropy(list(family_bytes.values())),
        },
        "threshold_evaluation": gate_results,
        "verdict": "PASS_DIVERSITY" if hard_pass else "FAIL_DIVERSITY",
        "repair_boundary": {
            "replay_or_document_duplication_can_fix": False,
            "current_frozen_contract_can_accept_silent_source_additions": False,
            "required_direction": (
                "Acquire independently authorized UK and EN family lineages, and rebalance/subsample "
                "dominant families (including code) under a successor DATA-300 contract before corpus freeze."
                if not hard_pass
                else "No diversity repair required."
            ),
        },
    }


def _canonical_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--contract",
        default="configs/data/data300_corpus_v03_frozen_build_contract_v2.json",
        help="Exact DATA-300 frozen candidate contract.",
    )
    parser.add_argument("--output", help="Write canonical qualification JSON.")
    parser.add_argument(
        "--check-report",
        help="Compare generated qualification byte-for-byte with a checked-in report.",
    )
    args = parser.parse_args()

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    payload = qualify(contract)
    text = _canonical_text(payload)

    if args.check_report:
        expected = Path(args.check_report).read_text(encoding="utf-8")
        if expected != text:
            raise SystemExit("DATA-312 report mismatch")
        print(f"{payload['verdict']}: report is deterministic and current")
        return 0

    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
