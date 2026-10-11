"""S15: audit 3 actually pinned source families together, NOT S3-S9 release.

The physical 3-book Gutenberg family + 10 PHP Manual UK XML pages + 2 Rust
Book UK pages are re-read from committed Git bytes. Reuse G06, DATA232 and
the S7 matcher. This DOES NOT authorize tokenizer fit, eval or training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import plan2_near_dedup_v1 as near
from tools import plan2_public_domain_books_v1 as books
from tools import plan2_reserved_eval_firewall_v1 as firewall
from tools import plan2_s15_d03_physical_sources_v1 as d03
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination
from twelve_six.data import decontamination_authority_v2 as data232
from twelve_six.data import privacy_execution_authority as g06

SCHEMA = "12-6.plan2-s15-three-family-physical-candidate.v1"
OUTPUT = "three-family-physical-candidate.json"


class ThreeFamilyDenied(ValueError):
    """Unverifiable physical family boundaries or privacy/data lineage."""


def need(value: bool, reason: str) -> None:
    if not value:
        raise ThreeFamilyDenied(reason)


def inspect(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    eng = books.inspect(root)
    ua = d03.inspect(root)
    need(eng["physical_source_families"] == 1
         and eng["physical_document_families"] == 3
         and eng["g06_rejected_record_ids"] == []
         and eng["production_release_authorized"] is False
         and ua["physical_family_count"] == 2
         and ua["physical_record_count"] == 12
         and ua["training_corpus_authorized"] is False
         and ua["production_release_authorized"] is False,
         "family source evidence or privacy not qualified")
    members: list[dict[str, Any]] = []
    bodies: dict[str, str] = {}
    for item in eng["books"]:
        raw = books.read_checked(root, item["snapshot_path"])
        need(books.sha(raw) == item["snapshot_sha256"]
             and len(raw) == item["snapshot_bytes"], "English source drift")
        rid = item["source_id"] + ":r00000000"
        members.append({
            "record_id": rid, "source_id": item["source_id"],
            "source_family": books.GUTENBERG_FAMILY,
            "document_family": item["document_family"],
            "modality": "en", "normalized_sha256": books.sha(raw),
            "normalized_bytes": len(raw),
        })
        bodies[rid] = raw.decode("utf-8", "strict")
    config = json.loads(books.read_checked(root, d03.CONFIG))
    by_path = {row["source_path"]: row for row in ua["records"]}
    need(len(by_path) == len(config["objects"]) == 12,
         "D03 source members not physically complete")
    for obj in config["objects"]:
        source_path = obj["physical_path"]
        metadata = by_path[source_path]
        raw = books.read_checked(root, source_path)
        normalized = (d03.normalize_php(raw) if obj["family"] == "php"
                      else d03.normalize_rust(raw))
        need(books.sha(normalized) == metadata["normalized_sha256"]
             and len(normalized) == metadata["normalized_bytes"],
             "physical Ukrainian normalization lineage changed")
        rid = metadata["record_id"]
        members.append({
            "record_id": rid, "source_id": metadata["source_id"],
            "source_family": metadata["source_family"],
            "document_family": metadata["source_path"],
            "modality": "uk", "normalized_sha256": metadata["normalized_sha256"],
            "normalized_bytes": len(normalized),
        })
        bodies[rid] = normalized.decode("utf-8", "strict")
    families = {x["source_family"] for x in members}
    need(len(members) == len(bodies) == 15
         and len({x["record_id"] for x in members}) == 15
         and len({x["document_family"] for x in members}) == 15
         and len({x["normalized_sha256"] for x in members}) == 15
         and families == {
             books.GUTENBERG_FAMILY,
             d03.FAMILIES["php"], d03.FAMILIES["rust"],
         }, "cross-family physical document or exact dedup identity failed")
    # Reuse the S7 approved matcher for cross-family *same-modality* mirrors.
    # Cross-language comparisons require independent multilingual semantic review.
    duplicate_pairs: list[dict[str, str]] = []
    for i, left in enumerate(members):
        for right in members[i + 1:]:
            if left["source_family"] == right["source_family"]:
                continue
            if left["modality"] != right["modality"]:
                continue
            kind = near.match_kind(bodies[left["record_id"]],
                                   bodies[right["record_id"]])
            if kind:
                duplicate_pairs.append({
                    "left_sha256": books.sha(left["record_id"].encode()),
                    "right_sha256": books.sha(right["record_id"].encode()),
                    "kind": kind,
                })
    need(not duplicate_pairs, "cross-family physical source near mirror")
    inventory = [{
        "record_id": row["record_id"], "source_id": row["source_id"],
        "family": row["source_family"], "modality": row["modality"],
        "payload_sha256": row["normalized_sha256"],
        "payload_bytes": row["normalized_bytes"],
    } for row in members]
    rows = [{"id": row["record_id"], "text": bodies[row["record_id"]],
             "mode": row["modality"]} for row in members]
    input_sha = g06.input_rows_sha256_from_text_free_inventory(inventory)
    privacy = g06.build_privacy_execution_authority(
        rows, expected_input_rows_sha256=input_sha)
    g06.verify_privacy_execution_authority(
        privacy, rows, expected_input_rows_sha256=input_sha,
        expected_execution_identity_sha256=privacy["execution_identity_sha256"])
    rejected = sorted(x["record_id"] for x in privacy["records"]
                      if x["action"] != "ALLOW")
    reserved = books.read_checked(root, firewall.RESERVE_PATH)
    _fixture, eval_rows, authorities = firewall._reserve(reserved)
    train = [{
        "record_id": x["record_id"], "source_id": x["source_id"],
        "source_family": x["source_family"],
        "lineage_family": x["document_family"],
        "modality": x["modality"], "text": bodies[x["record_id"]],
    } for x in members if x["record_id"] not in rejected]
    need(bool(train), "G06 denied entire multi-family physical candidate")
    selection_sha = books.sha(books.canonical([
        row for row in eval_rows if row["record_id"].startswith("reserved.selection.")
    ]))
    final_sha = books.sha(books.canonical([
        row for row in eval_rows if row["record_id"].startswith("reserved.final.")
    ]))
    report = data232.build_report(
        train, eval_rows,
        training_corpus_identity=books.sha(books.canonical(members)),
        selection_validation_identity=selection_sha,
        final_test_identity=final_sha, authorities=authorities,
        quarantine_cross_source_families=True)
    data232.verify_report(report)
    cleaned = (not rejected
               and report["counts"]["excluded_training_records"] == 0
               and report["counts"]["quarantined_source_families"] == 0)
    core = {
        "schema_version": SCHEMA,
        "decision": "THREE_REAL_SOURCE_FAMILIES_S15_CANDIDATE_ONLY",
        "source_family_count": 3, "document_identity_count": 15,
        "total_physical_normalized_bytes": sum(x["normalized_bytes"] for x in members),
        "families": sorted(families),
        "gutenberg_source_sha256": eng["manifest_sha256"],
        "d03_source_sha256": ua["manifest_sha256"],
        "source_members_sha256": books.sha(books.canonical(members)),
        "source_members": sorted(members, key=lambda x: x["record_id"]),
        "global_exact_duplicates": 0,
        "same_modality_cross_family_near_pairs": len(duplicate_pairs),
        "cross_language_semantic_dedup_established": False,
        "g06_input_sha256": input_sha,
        "g06_execution_identity_sha256": privacy["execution_identity_sha256"],
        "g06_rejected_record_ids": rejected,
        "data232_report_sha256": report["report_sha256"],
        "data232_excluded_record_count": report["counts"]["excluded_training_records"],
        "data232_quarantined_source_family_count":
            report["counts"]["quarantined_source_families"],
        "physical_candidate_fixture_eval_clean": cleaned,
        "real_evaluation_custody_established": False,
        "physical_s3_s9_multi_family_admitted": False,
        "physical_production_split_established": False,
        "production_tokenizer_fit_authorized": False,
        "training_corpus_authorized": False,
        "production_release_authorized": False,
        "terminal_done": False,
    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}


def stage(root: Path, destination: Path) -> dict[str, Any]:
    need(not any(p.is_symlink() for p in (destination, *destination.parents)),
         "symlink physical source destination")
    proof = inspect(root)
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / OUTPUT
    raw = books.canonical(proof)
    if output.exists() or output.is_symlink():
        need(output.is_file() and not output.is_symlink()
             and _read_destination(output) == raw,
             "immutable multi-family source receipt changed")
    else:
        _atomic_write(destination, output, raw)
    need(_read_destination(output) == raw, "source receipt readback changed")
    return proof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    report = stage(args.root, args.out_dir)
    print(json.dumps({
        "decision": report["decision"],
        "manifest_sha256": report["manifest_sha256"],
        "terminal_done": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
