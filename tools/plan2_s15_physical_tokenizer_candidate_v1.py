"""Plan 2 S15: fit existing frozen BPE on real physical training book only.

This is a corpus-bound local FREE candidate. The frozen S12 algorithm is reused,
but the incumbent S12 fixture-only policy is NOT a production permission grant.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_physical_book_split_probe_v1 as split_probe
from tools import plan2_s15_physical_heldout_decontam_v1 as heldout
from tools import plan2_tokenizer_fit_freeze_v1 as freeze
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-s15-physical-tokenizer-candidate.v1"
OUTPUT = "physical-tokenizer-candidate.json"
MAX_BYTES = 32768


class PhysicalFitDenied(ValueError):
    """Unsafe, contaminated, non-reproducible or unauthorised physical fit."""


def need(ok: bool, why: str) -> None:
    if not ok:
        raise PhysicalFitDenied(why)


def segments(source_id: str, text: str) -> list[dict[str, str]]:
    """Exact deterministic UTF-8/line-preserving chunks for the canonical fitter."""
    need(type(source_id) is str and bool(source_id)
         and type(text) is str and bool(text), "invalid fit document")
    lines = text.splitlines(keepends=True)
    chunks: list[str] = []
    group: list[str] = []
    size = 0
    for line in lines:
        raw = line.encode("utf-8", "strict")
        need(0 < len(raw) <= MAX_BYTES, "oversized physical document line")
        if group and size + len(raw) > MAX_BYTES:
            chunks.append("".join(group))
            group = []
            size = 0
        group.append(line)
        size += len(raw)
    if group:
        chunks.append("".join(group))
    need(bool(chunks) and "".join(chunks) == text,
         "physical fit segmentation lost source bytes")
    return [
        {"record_id": f"{source_id}:r{50_000_000 + index:08d}", "text": payload}
        for index, payload in enumerate(chunks)
    ]


def inspect(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    admission = heldout.inspect(root)
    need(admission["physical_heldout_decontamination_clean"] is True
         and admission["final_test_outcomes_read"] is False
         and admission["terminal_done"] is False
         and admission["physical_s9_admitted"] is False,
         "physical real-text heldout fails DATA232 boundary")
    source = books.inspect(root)
    assignment = split_probe.inspect(root)
    need(source["g06_rejected_record_ids"] == []
         and source["data232_excluded_record_count"] == 0
         and assignment["physical_s9_admitted"] is False
         and source["training_corpus_authorized"] is False,
         "source privacy or candidate authority drift")
    roles = assignment["assigned_document_splits"]
    train = [x for x in source["books"] if roles.get(x["source_id"]) == "train"]
    need(len(train) == 1 and len(roles) == 3
         and set(roles.values()) == {"train", "validation", "test"},
         "train split is missing or leaked")
    doc = train[0]
    payload = books.read_checked(root, doc["snapshot_path"])
    need(books.sha(payload) == doc["snapshot_sha256"]
         and len(payload) == doc["snapshot_bytes"],
         "real training snapshot changed")
    rows = segments(doc["source_id"], payload.decode("utf-8", "strict"))
    policy = freeze._policy(root)
    baseline = freeze.freeze_fixture(root)
    merges, records_sha = freeze._fit(rows, policy["max_fixture_merges"])
    tokenizer = freeze.FrozenBPE(merges)
    need(all(tokenizer.decode(tokenizer.encode(r["text"])) == r["text"]
             for r in rows), "real-text physical tokenizer is noninvertible")
    core = {
        "schema_version": SCHEMA,
        "decision": "PHYSICAL_REAL_TEXT_FITTED_CANDIDATE_NOT_RELEASE",
        "physical_source_manifest_sha256": source["manifest_sha256"],
        "actual_heldout_audit_sha256": admission["manifest_sha256"],
        "physical_split_probe_sha256": assignment["manifest_sha256"],
        "physical_split_manifest_sha256": assignment["s10_fixture_split_sha256"],
        "train_document_source_id": doc["source_id"],
        "train_document_sha256": doc["snapshot_sha256"],
        "train_document_bytes": doc["snapshot_bytes"],
        "train_record_count": len(rows),
        "train_records_sha256": records_sha,
        "s11_architecture_manifest_sha256":
            baseline["s11_architecture_manifest_sha256"],
        "fit_policy_git_blob": freeze.POLICY_BLOB,
        "fit_merges": merges,
        "tokenizer_identity": tokenizer.identity.to_dict(),
        "frozen": True,
        "byte_roundtrip_verified": True,
        "sampled_only": False,
        "physical_training_corpus_selected_from_real_holdout": True,
        "physical_s9_admitted": False,
        "tokenizer_fit_authorized": False,
        "training_corpus_authorized": False,
        "production_release_authorized": False,
        "terminal_done": False,
    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}


def stage(root: Path, destination: Path) -> dict[str, Any]:
    need(not any(p.is_symlink() for p in (destination, *destination.parents)),
         "symlink publication path")
    report = inspect(root)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / OUTPUT
    raw = books.canonical(report)
    if target.is_symlink() or target.exists():
        need(target.is_file() and not target.is_symlink()
             and _read_destination(target) == raw,
             "immutable fitted candidate changed")
    else:
        _atomic_write(destination, target, raw)
    need(_read_destination(target) == raw, "fitted candidate readback drift")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = stage(args.root, args.out_dir)
    print(json.dumps({
        "decision": result["decision"],
        "manifest_sha256": result["manifest_sha256"],
        "production_release_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
