"""S15 physical heldout DATA-232 audit over real pinned public-domain documents.

Not a terminal dataset, tokenizer fit, external evaluator or Plan9 authorization.
Uses the incumbent public-domain/G06/DATA232 and S10 document-cluster probes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import plan2_near_dedup_v1 as near
from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_physical_book_split_probe_v1 as split_probe
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination
from twelve_six.data import decontamination_authority_v2 as data232

SCHEMA = "12-6.plan2-s15-physical-heldout-decontamination.v1"
OUTPUT = "physical-heldout-decontamination.json"
ROLES = {"validation": "selection_validation", "test": "final_test"}


class PhysicalHeldoutDenied(ValueError):
    """Unqualified holdout, source drift, or escalation of candidate authority."""


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise PhysicalHeldoutDenied(reason)


def canonical(value: Any) -> bytes:
    return books.canonical(value)


def fingerprint(value: Any) -> str:
    return books.sha(canonical(value))


def interdocument_mirror_evidence(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Use existing S7 matcher; never conflate two spans of one document."""
    result: list[dict[str, str]] = []
    for i, left in enumerate(rows):
        for right in rows[i + 1:]:
            if left["source_id"] == right["source_id"]:
                continue
            relation = near.match_kind(left["text"], right["text"])
            if relation is not None:
                result.append({
                    "left_record_id_sha256": books.sha(left["record_id"].encode()),
                    "right_record_id_sha256": books.sha(right["record_id"].encode()),
                    "relation": relation,
                })
    return sorted(result, key=lambda x: (
        x["left_record_id_sha256"], x["right_record_id_sha256"], x["relation"]
    ))


def inspect(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    source = books.inspect(root)
    split = split_probe.inspect(root)
    need(source["training_corpus_authorized"] is False
         and source["production_release_authorized"] is False
         and split["production_release_authorized"] is False
         and split["physical_s9_admitted"] is False
         and source["g06_rejected_record_ids"] == []
         and source["data232_excluded_record_count"] == 0
         and source["data232_quarantined_source_family_count"] == 0,
         "uncleared privacy, upstream reserve or release authority")
    by_source = split["assigned_document_splits"]
    need(type(by_source) is dict and len(by_source) == 3
         and set(by_source.values()) == {"train", "validation", "test"},
         "missing three-way physical document roles")
    need(set(by_source) == {item["source_id"] for item in source["books"]},
         "split/source identity mismatch")
    doc_meta = {item["source_id"]: item for item in source["books"]}
    rows = split_probe._rows(root, source)
    interdocument_mirrors = interdocument_mirror_evidence(rows)
    training: list[dict[str, str]] = []
    evaluation: list[dict[str, str]] = []
    role_documents: dict[str, list[dict[str, str]]] = {
        "selection_validation": [], "final_test": [],
    }
    for row in rows:
        sid = row["source_id"]
        doc = doc_meta[sid]
        lineage = doc["document_family"]
        data = {
            **row,
            "source_family": books.GUTENBERG_FAMILY,
            "lineage_family": lineage,
            "modality": "en",
        }
        role = by_source[sid]
        need(role in {"train", "validation", "test"},
             "unexpected source document split")
        if role == "train":
            training.append(data)
        else:
            evaluation.append(data)
            role_documents[ROLES[role]].append({
                "source_id": sid, "record_id": row["record_id"],
                "snapshot_sha256": doc["snapshot_sha256"],
                "source_revision": doc["source_revision"],
            })
    need(len(training) == 2 and len(evaluation) == 4
         and all(len(rows) == 2 for rows in role_documents.values()),
         "missing physical train or heldout members")
    grants = []
    for role, members in sorted(role_documents.items()):
        sid = members[0]["source_id"]
        need(all(x["source_id"] == sid for x in members),
             "heldout role mixes physical source documents")
        authority = {
            "role": role, "source_id": sid,
            "document_family": doc_meta[sid]["document_family"],
            "source_snapshot_sha256": doc_meta[sid]["snapshot_sha256"],
            "records": sorted(members, key=lambda x: x["record_id"]),
        }
        grants.append({
            "authority_id": sid,
            "identity_sha256": fingerprint(authority),
            "role": role,
            "source_sha": doc_meta[sid]["source_revision"],
        })
    authorities = {
        "schema": "12-6.data232-reserved-authorities.v1",
        "authorities": sorted(grants, key=lambda x: x["authority_id"]),
    }
    data232.validate_authority_metadata(authorities)
    validation_id = next(x["identity_sha256"] for x in grants
                         if x["role"] == "selection_validation")
    final_id = next(x["identity_sha256"] for x in grants
                    if x["role"] == "final_test")
    need(validation_id != final_id and all(
        x["record_id"] not in {y["record_id"] for y in evaluation}
        for x in training
    ), "cross-partition exposure")
    report = data232.build_report(
        training, evaluation,
        training_corpus_identity=fingerprint({
            "source_manifest_sha256": source["manifest_sha256"],
            "split_sha256": split["s10_fixture_split_sha256"],
            "train_records": sorted(x["record_id"] for x in training),
        }),
        selection_validation_identity=validation_id,
        final_test_identity=final_id, authorities=authorities,
        quarantine_cross_source_families=True,
    )
    data232.verify_report(report)
    need(report["final_test_outcomes_read"] is False
         and report["counts"]["training_records"] == len(training)
         and report["counts"]["evaluation_records"] == len(evaluation)
         and report["hash_only_evidence"] is True,
         "DATA232 actual heldout audit is invalid")
    core = {
        "schema_version": SCHEMA,
        "decision": "REAL_HELDOUT_SCANNED_COMPONENT_ONLY_NOT_RELEASE",
        "physical_source_manifest_sha256": source["manifest_sha256"],
        "physical_split_probe_sha256": split["manifest_sha256"],
        "canonical_source_family_count": source["physical_source_families"],
        "document_family_count": source["physical_document_families"],
        "training_document_count": 1,
        "validation_document_count": 1,
        "final_test_document_count": 1,
        "training_span_count": len(training),
        "heldout_span_count": len(evaluation),
        "reserved_role_identities": {
            "selection_validation": validation_id,
            "final_test": final_id,
        },
        "data232_report_sha256": report["report_sha256"],
        "data232_status": report["status"],
        "excluded_training_record_count":
            report["counts"]["excluded_training_records"],
        "quarantined_source_family_count":
            report["counts"]["quarantined_source_families"],
        "interdocument_mirror_count": len(interdocument_mirrors),
        "interdocument_mirror_evidence": interdocument_mirrors,
        "physical_heldout_decontamination_clean": (
            report["counts"]["excluded_training_records"] == 0
            and report["counts"]["quarantined_source_families"] == 0
            and not interdocument_mirrors
        ),
        "final_test_outcomes_read": False,
        "physical_s9_admitted": False,
        "tokenizer_fit_authorized": False,
        "training_corpus_authorized": False,
        "production_release_authorized": False,
        "terminal_done": False,
    }
    return {**core, "manifest_sha256": fingerprint(core)}


def stage(root: Path, destination: Path) -> dict[str, Any]:
    need(not any(p.is_symlink() for p in (destination, *destination.parents)),
         "symlink output path")
    proof = inspect(root)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / OUTPUT
    raw = canonical(proof)
    if target.exists() or target.is_symlink():
        need(not target.is_symlink() and target.is_file()
             and _read_destination(target) == raw,
             "immutable heldout receipt drift")
    else:
        _atomic_write(destination, target, raw)
    need(_read_destination(target) == raw,
         "physical heldout receipt readback mismatch")
    return proof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    proof = stage(args.root, args.out_dir)
    print(json.dumps({
        "decision": proof["decision"],
        "manifest_sha256": proof["manifest_sha256"],
        "terminal_done": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
