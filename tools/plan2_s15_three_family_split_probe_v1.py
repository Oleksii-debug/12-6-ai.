"""S15: real three-family whole-source split mechanics, NONAUTHORITATIVE.

All 15 physical source members are passed to the incumbent S10 splitter with a
synthetic S9-shaped candidate wrapper. The wrapper is not a physical S9 release,
training permission, or custody of real final-test outcomes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import plan2_cluster_split_v1 as split
from tools import plan2_corpus_mixture_v1 as mixture
from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_d03_physical_sources_v1 as d03
from tools import plan2_s15_three_family_physical_v1 as combined
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-s15-three-family-source-split-probe.v1"
OUTPUT = "three-family-source-split-probe.json"


class ThreeFamilySplitDenied(ValueError):
    """Missing source, damaged boundaries or illegal corpus promotion."""


def need(value: bool, reason: str) -> None:
    if not value:
        raise ThreeFamilySplitDenied(reason)


def physical_rows(root: Path, proof: dict[str, Any]) -> list[dict[str, str]]:
    source_meta = proof["source_members"]
    ua_config = json.loads(books.read_checked(root, d03.CONFIG))
    ua_objects = {item["physical_path"]: item for item in ua_config["objects"]}
    en_sources = {item["source_id"]: item
                  for item in books.inspect(root)["books"]}
    records = []
    for member in source_meta:
        if member["modality"] == "en":
            en_metadata = en_sources.get(member["source_id"])
            need(type(en_metadata) is dict, "unknown physical book source")
            payload = books.read_checked(root, en_metadata["snapshot_path"])
        else:
            path = member["document_family"]
            obj = ua_objects.get(path)
            need(type(obj) is dict, "unknown historical D03 source document")
            raw = books.read_checked(root, path)
            payload = d03.normalize_php(raw) if obj["family"] == "php" else d03.normalize_rust(raw)
        need(books.sha(payload) == member["normalized_sha256"]
             and len(payload) == member["normalized_bytes"],
             "physical selected source bytes were modified")
        records.append({
            "record_id": member["record_id"],
            "source_id": member["source_id"],
            "text": payload.decode("utf-8", "strict"),
        })
    need(len(records) == 15, "incomplete three-family physical rows")
    return records


def inspect(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    proof = combined.inspect(root)
    need(proof["source_family_count"] == 3
         and proof["document_identity_count"] == 15
         and proof["global_exact_duplicates"] == 0
         and proof["same_modality_cross_family_near_pairs"] == 0
         and proof["g06_rejected_record_ids"] == []
         and proof["data232_excluded_record_count"] == 0
         and proof["data232_quarantined_source_family_count"] == 0
         and proof["training_corpus_authorized"] is False
         and proof["physical_s3_s9_multi_family_admitted"] is False,
         "S1-S9 physical candidates not fully privacy/fixture-clean")
    rows = physical_rows(root, proof)
    ids = sorted(row["record_id"] for row in rows)
    fixture = {
        "schema_version": mixture.SCHEMA,
        "selected_record_ids": ids, "selected_record_count": len(ids),
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "real_final_test_material_accessed": False,
        "fixture_only": True,
        "three_family_source_sha256": proof["manifest_sha256"],
    }
    fixture["dataset_candidate_sha256"] = split._sha(split._canonical(fixture))
    policy = split._policy(books.read_checked(root, split.POLICY_PATH))
    assigned = split.build_cluster_split(fixture, rows, policy, fixture=True)
    need(assigned["purpose"] == "LOCAL_FREE_SYNTHETIC_COMPONENT"
         and assigned["training_corpus_authorized"] is False
         and assigned["tokenizer_fit_authorized"] is False
         and assigned["evaluation_release_authorized"] is False
         and assigned["cluster_leakage_count"] == 0,
         "S10 source-cluster mechanics accidentally promoted data")
    by_source: dict[str, set[str]] = {}
    for row in rows:
        by_source.setdefault(row["source_id"], set()).add(
            assigned["record_assignments"][row["record_id"]]
        )
    need(len(by_source) == 5
         and all(len(x) == 1 for x in by_source.values()),
         "physical source-cluster leakage or fabricated family credit")
    assignments = {key: next(iter(value))
                   for key, value in sorted(by_source.items())}
    source_to_family: dict[str, str] = {}
    for item in proof["source_members"]:
        sid = item["source_id"]
        family = item["source_family"]
        need(sid not in source_to_family or source_to_family[sid] == family,
             "same source ID falsely credited to two different families")
        source_to_family[sid] = family
    need(set(source_to_family) == set(assignments),
         "family accounting and source assignments disconnected")
    family_roles = {role: sorted({
        source_to_family[source] for source, value in assignments.items()
        if value == role
    }) for role in ("train", "validation", "test")}
    need(set(assignments.values()) == {"train", "validation", "test"},
         "empty three-way physical source split")
    core = {
        "schema_version": SCHEMA,
        "decision": "THREE_REAL_FAMILY_SOURCE_SPLIT_MECHANICS_NOT_ADMISSION",
        "upstream_three_family_sha256": proof["manifest_sha256"],
        "s9_synthetic_wrapper_sha256": fixture["dataset_candidate_sha256"],
        "s10_split_manifest_sha256": assigned["split_manifest_sha256"],
        "canonical_source_family_count": 3,
        "physical_document_count": 15,
        "whole_source_cluster_count": 5,
        "physical_source_cluster_assignments": assignments,
        "source_family_roles": family_roles,
        "train_source_family_count": len(family_roles["train"]),
        "validation_source_family_count": len(family_roles["validation"]),
        "test_source_family_count": len(family_roles["test"]),
        "production_train_mixture_admitted": False,
        "train_record_count": len(assigned["train_record_ids"]),
        "validation_record_count": len(assigned["validation_record_ids"]),
        "test_record_count": len(assigned["test_record_ids"]),
        "cluster_leakage_count": 0,
        "real_final_test_outcomes_read": False,
        "physical_s9_admitted": False,
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "production_release_authorized": False,
        "terminal_done": False,
    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}


def stage(root: Path, destination: Path) -> dict[str, Any]:
    need(not any(path.is_symlink() for path in (destination, *destination.parents)),
         "symlink publication")
    result = inspect(root)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / OUTPUT
    raw = books.canonical(result)
    if target.exists() or target.is_symlink():
        need(target.is_file() and not target.is_symlink()
             and _read_destination(target) == raw,
             "immutable physical family split changed")
    else:
        _atomic_write(destination, target, raw)
    need(_read_destination(target) == raw, "physical family split readback drift")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    value = stage(args.root, args.out_dir)
    print(json.dumps({
        "decision": value["decision"],
        "manifest_sha256": value["manifest_sha256"],
        "terminal_done": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
