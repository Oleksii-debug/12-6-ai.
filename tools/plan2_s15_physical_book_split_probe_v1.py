"""Physical-book S10 split *mechanics* probe, not a corpus admission.

Consumes pinned S15 source/rights/PII/decontamination candidate evidence and
the already accepted S10 cluster splitter. It does not create an S9 receipt or
authorize source promotion: its synthetic S9-shaped wrapper is explicitly a
non-authoritative mechanics fixture.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import plan2_cluster_split_v1 as split
from tools import plan2_corpus_mixture_v1 as mixture
from tools import plan2_public_domain_books_v1 as books
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-s15-physical-book-split-probe.v1"
OUTPUT = "book-split-probe.json"


class BookSplitProbeDenied(ValueError):
    """Missing physical authority, changed split or unsafe publication."""


def need(ok: bool, message: str) -> None:
    if not ok:
        raise BookSplitProbeDenied(message)


def _rows(root: Path, proof: dict[str, Any]) -> list[dict[str, str]]:
    """Two byte-preserving, non-overlapping line spans per verified book."""
    result: list[dict[str, str]] = []
    for book in proof["books"]:
        source_id = book["source_id"]
        raw = books.read_checked(root, book["snapshot_path"])
        need(books.sha(raw) == book["snapshot_sha256"]
             and len(raw) == book["snapshot_bytes"],
             "book bytes differ from rights-bound source audit")
        text = raw.decode("utf-8", "strict")
        lines = text.splitlines(keepends=True)
        midpoint = len(lines) // 2
        need(midpoint > 0 and midpoint < len(lines),
             "book has no independently addressable spans")
        spans = ("".join(lines[:midpoint]), "".join(lines[midpoint:]))
        need("".join(spans).encode("utf-8") == raw
             and all(span.strip() for span in spans),
             "physical source reconstruction drift")
        for index, body in enumerate(spans):
            result.append({
                "record_id": f"{source_id}:r{index:08d}",
                "source_id": source_id,
                "text": body,
            })
    need(len(result) == 6 and len({r["source_id"] for r in result}) == 3,
         "physical document-cluster count drift")
    return result


def inspect(root: Path) -> dict[str, Any]:
    proof = books.inspect(root)
    need(proof["physical_source_families"] == 1
         and proof["physical_document_families"] == 3
         and proof["training_corpus_authorized"] is False
         and proof["production_release_authorized"] is False
         and proof["data232_fixture_eval_only"] is True,
         "source identity or non-release rights boundary changed")
    # The probe consumes whole original bodies. Do not reintroduce records
    # excluded by G06 privacy or the DATA-232 decontamination firewall.
    need(proof.get("g06_rejected_record_ids") == []
         and proof.get("data232_excluded_record_count") == 0
         and proof.get("data232_quarantined_source_family_count") == 0,
         "privacy or decontamination excluded source; split would reuse rejected text")
    rows = _rows(root, proof)
    ids = sorted(r["record_id"] for r in rows)
    # Deliberately synthetic receipt: not produced by current physical S9,
    # never a permission or training corpus authority.
    s9_fixture = {
        "schema_version": mixture.SCHEMA,
        "selected_record_ids": ids,
        "selected_record_count": len(ids),
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "real_final_test_material_accessed": False,
        "fixture_only": True,
        "physical_books_candidate_audit_sha256": proof["manifest_sha256"],
    }
    s9_fixture["dataset_candidate_sha256"] = split._sha(
        split._canonical(s9_fixture))
    policy = split._policy(books.read_checked(root, split.POLICY_PATH))
    candidate = split.build_cluster_split(
        s9_fixture, rows, policy, fixture=True)
    assignments = candidate["record_assignments"]
    need(type(assignments) is dict and set(assignments) == set(ids),
         "partition membership identity drift")
    for name in ("train", "validation", "test"):
        actual = {rid for rid, assigned in assignments.items() if assigned == name}
        need(actual == set(candidate[f"{name}_record_ids"]),
             "partition assignments contradict S10 receipt")
    by_book = {}
    for row in rows:
        by_book.setdefault(row["source_id"], set()).add(
            assignments[row["record_id"]])
    need(len(by_book) == 3 and all(len(assigned) == 1
                                  for assigned in by_book.values()),
         "whole-document train/validation/test leakage")
    need(all(candidate[f"{name}_record_ids"]
             for name in ("train", "validation", "test")),
         "physical book split lacks required partition")
    need(candidate["training_corpus_authorized"] is False
         and candidate["tokenizer_fit_authorized"] is False
         and candidate["evaluation_release_authorized"] is False
         and candidate["cluster_leakage_count"] == 0,
         "split mechanics unexpectedly grant release authority")
    core = {
        "schema_version": SCHEMA,
        "decision": "S10_MECHANICS_ONLY_NOT_PHYSICAL_S9_ADMISSION",
        "physical_books_manifest_sha256": proof["manifest_sha256"],
        "verified_physical_bytes": proof["physical_source_bytes"],
        "physical_document_clusters": 3,
        "canonical_source_families": 1,
        "physical_book_span_count": len(rows),
        "s10_fixture_split_sha256": candidate["split_manifest_sha256"],
        "s10_fixture_upstream_sha256": s9_fixture["dataset_candidate_sha256"],
        "assigned_document_splits": {
            key: next(iter(value)) for key, value in sorted(by_book.items())
        },
        "physical_s9_admitted": False,
        "training_corpus_authorized": False,
        "production_release_authorized": False,
        "tokenizer_fit_authorized": False,
        "real_final_test_accessed": False,
    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}


def stage(root: Path, destination: Path) -> dict[str, Any]:
    need(not any(p.is_symlink() for p in (destination, *destination.parents)),
         "symlink destination denied")
    proof = inspect(root)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / OUTPUT
    blob = books.canonical(proof)
    if target.exists() or target.is_symlink():
        need(not target.is_symlink() and _read_destination(target) == blob,
             "immutable physical split probe changed")
    else:
        _atomic_write(destination, target, blob)
    need(_read_destination(target) == blob, "split probe readback drift")
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
        "production_release_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
