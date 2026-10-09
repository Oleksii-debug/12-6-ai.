"""S15 fit incumbent byte BPE to exact S10 three-family TRAIN bytes only.

Read all physical train-only members from the S15 partition builder; never
include selection validation, final test, or reserved EVAL233 texts. Incumbent
S12 merge cap (96) is not a production 32K tokenizer or training permission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_physical_tokenizer_candidate_v1 as segmenter
from tools import plan2_s15_three_family_train_materialization_v1 as training
from tools import plan2_tokenizer_fit_freeze_v1 as freeze
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-s15-physical-three-family-train-bpe.v1"
OUTPUT = "three-family-train-bpe-candidate.json"


class TrainBpeDenied(ValueError):
    """No S10-bound training corpus or unsafe model identity/receipt."""


def need(ok: bool, why: str) -> None:
    if not ok:
        raise TrainBpeDenied(why)


def inspect(root: Path) -> dict[str, Any]:
    root = root.resolve(strict=True)
    partition, physical = training.build(root)
    need(partition["source_families_total"] == 3
         and partition["train_document_count"] == len(physical)
         and partition["heldout_document_count"] > 0
         and partition["heldout_plaintext_materialized"] is False
         and partition["physical_s9_admitted"] is False
         and partition["tokenizer_fit_authorized"] is False
         and partition["production_release_authorized"] is False
         and partition["training_corpus_authorized"] is False,
         "not a valid heldout-isolated training source candidate")
    policy = freeze._policy(root)
    need(policy["target_vocab_size"] == 32768
         and policy["purpose"] == "LOCAL_FREE_SYNTHETIC_NO_RELEASE"
         and 1 <= policy["max_fixture_merges"] <= 128,
         "incumbent non-release BPE policy changed")
    rows: list[dict[str, str]] = []
    source_digests: list[dict[str, Any]] = []
    for member in sorted(partition["train_members"], key=lambda m: m["path"]):
        path = member["path"]
        need(path.startswith("train/") and path in physical,
             "train member not part of exact S10 partition")
        raw = physical[path]
        need(books.sha(raw) == member["normalized_sha256"]
             and len(raw) == member["normalized_bytes"],
             "training member differs from S10 physical bytes")
        doc_id = member["record_id_sha256"]
        segments = segmenter.segments(doc_id, raw.decode("utf-8", "strict"))
        need(len(segments) > 0
             and "".join(s["text"] for s in segments).encode("utf-8") == raw,
             "train input segmentation lost bytes")
        rows.extend(segments)
        source_digests.append({
            "record_id_sha256": doc_id,
            "member_sha256": books.sha(raw),
            "segment_count": len(segments),
        })
    need(rows and len(rows) <= 1000
         and len({x["record_id"] for x in rows}) == len(rows)
         and len(source_digests) == len(partition["train_members"]),
         "invalid or duplicated physical training segment identities")
    merges, source_record_sha = freeze._fit(rows, policy["max_fixture_merges"])
    model = freeze.FrozenBPE(merges)
    need(model.vocab_size <= 388
         and all(model.decode(model.encode(r["text"])) == r["text"]
                 for r in rows),
         "three-family physical BPE not fully reversible")
    core = {
        "schema_version": SCHEMA,
        "decision": "THREE_FAMILY_TRAIN_ONLY_BPE_CANDIDATE_NOT_PRODUCTION",
        "source_train_partition_sha256": partition["manifest_sha256"],
        "source_s10_split_manifest_sha256": partition["s10_split_manifest_sha256"],
        "source_train_document_count": partition["train_document_count"],
        "source_train_utf8_bytes": partition["physical_train_bytes"],
        "source_record_sha256": source_record_sha,
        "training_segment_count": len(rows),
        "train_member_digests": source_digests,
        "fitted_merges": merges,
        "tokenizer_identity": model.identity.to_dict(),
        "target_vocab_size": policy["target_vocab_size"],
        "candidate_vocab_size": model.vocab_size,
        "all_train_segments_byte_roundtrip": True,
        "heldout_payloads_fitted": False,
        "real_evaluation_outcomes_read": False,
        "production_target_vocab_frozen": False,
        "physical_s9_admitted": False,
        "tokenizer_fit_authorized": False,
        "training_corpus_authorized": False,
        "production_release_authorized": False,
        "terminal_done": False,
    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}


def stage(root: Path, destination: Path) -> dict[str, Any]:
    need(not any(p.is_symlink() for p in (destination, *destination.parents)),
         "symlink tokenizer publication")
    report = inspect(root)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / OUTPUT
    raw = books.canonical(report)
    if target.exists() or target.is_symlink():
        need(target.is_file() and not target.is_symlink()
             and _read_destination(target) == raw,
             "immutable three-family BPE changed")
    else:
        _atomic_write(destination, target, raw)
    need(_read_destination(target) == raw,
         "physical tokenizer readback mismatch")
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
        "terminal_done": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
