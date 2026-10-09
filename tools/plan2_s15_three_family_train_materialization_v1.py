"""S15 physical three-family training partition; HOLDOUTS HASH-ONLY.

Materializes only the train role from the incumbent S10 whole-source split
mechanics. Both validation and final-test bodies are NEVER published here.
This is a LOCAL_FREE source candidate; S3-S9 admission and Plan9 authority
remain absent. No tokenizer fit or model optimization is permitted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_three_family_physical_v1 as combined
from tools import plan2_s15_three_family_split_probe_v1 as split_probe
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-s15-three-family-physical-train-candidate.v1"
OUTPUT = "three-family-train-partition.json"


class TrainPartitionDenied(ValueError):
    """Missing upstream controls, unsafe output or leaked heldout source."""


def need(ok: bool, why: str) -> None:
    if not ok:
        raise TrainPartitionDenied(why)


def build(root: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    root = root.resolve(strict=True)
    proof = combined.inspect(root)
    assigned = split_probe.inspect(root)
    need(proof["source_family_count"] == 3
         and proof["document_identity_count"] == 15
         and proof["global_exact_duplicates"] == 0
         and proof["g06_rejected_record_ids"] == []
         and proof["data232_excluded_record_count"] == 0
         and proof["data232_quarantined_source_family_count"] == 0
         and proof["physical_s3_s9_multi_family_admitted"] is False
         and proof["production_release_authorized"] is False
         and assigned["canonical_source_family_count"] == 3
         and assigned["physical_document_count"] == 15
         and assigned["upstream_three_family_sha256"] == proof["manifest_sha256"]
         and assigned["physical_s9_admitted"] is False
         and assigned["training_corpus_authorized"] is False
         and assigned["production_release_authorized"] is False,
         "source, privacy or whole-source split is not an admitted candidate")
    roles = assigned["physical_source_cluster_assignments"]
    need(type(roles) is dict and len(roles) == 5
         and set(roles.values()) == {"train", "validation", "test"},
         "incomplete physical source role assignment")
    rows = split_probe.physical_rows(root, proof)
    metadata = {item["record_id"]: item for item in proof["source_members"]}
    need(len(rows) == len(metadata) == 15
         and {x["record_id"] for x in rows} == set(metadata),
         "physical source inventory and S10 partitions differ")
    train_payloads: dict[str, bytes] = {}
    training: list[dict[str, Any]] = []
    heldout: list[dict[str, Any]] = []
    train_families: set[str] = set()
    accounted = 0
    for row in sorted(rows, key=lambda x: x["record_id"]):
        record_id = row["record_id"]
        origin = metadata[record_id]
        need(row["source_id"] == origin["source_id"]
             and row["source_id"] in roles,
             "S10 source identity does not match sealed source")
        role = roles[row["source_id"]]
        raw = row["text"].encode("utf-8", "strict")
        need(books.sha(raw) == origin["normalized_sha256"]
             and len(raw) == origin["normalized_bytes"],
             "source text changed after G06/DATA232 gates")
        accounted += len(raw)
        member = {
            "record_id_sha256": books.sha(record_id.encode("utf-8")),
            "source_family": origin["source_family"],
            "normalized_sha256": origin["normalized_sha256"],
            "normalized_bytes": len(raw),
        }
        if role == "train":
            target = "train/" + member["record_id_sha256"] + ".utf8"
            need(target not in train_payloads, "train record target collision")
            train_payloads[target] = raw
            training.append({**member, "path": target})
            train_families.add(origin["source_family"])
        else:
            # No heldout plaintext or raw record IDs in the published
            # partition. Do not merge validation/final identities.
            heldout.append({**member, "role": role})
    need(accounted == proof["total_physical_normalized_bytes"]
         and len(training) + len(heldout) == 15
         and len(training) == assigned["train_record_count"]
         and len([x for x in heldout if x["role"] == "validation"])
             == assigned["validation_record_count"]
         and len([x for x in heldout if x["role"] == "test"])
             == assigned["test_record_count"]
         and len(train_payloads) == len(training) > 0
         and len(heldout) > 0,
         "train/heldout source membership incomplete or leaking")
    core = {
        "schema_version": SCHEMA,
        "decision": "REAL_THREE_FAMILY_TRAIN_ONLY_PHYSICAL_CANDIDATE_NOT_RELEASE",
        "upstream_three_family_sha256": proof["manifest_sha256"],
        "s10_split_manifest_sha256": assigned["s10_split_manifest_sha256"],
        "s10_split_probe_sha256": assigned["manifest_sha256"],
        "source_families_total": 3,
        "source_clusters_total": 5,
        "train_source_family_count": len(train_families),
        "train_document_count": len(training),
        "heldout_document_count": len(heldout),
        "physical_train_bytes": sum(len(x) for x in train_payloads.values()),
        "total_accounted_source_bytes": accounted,
        "train_members": training,
        "heldout_members_hash_only": heldout,
        "heldout_plaintext_materialized": False,
        "final_test_outcomes_read": False,
        "physical_s9_admitted": False,
        "physical_production_split_authorized": False,
        "tokenizer_fit_authorized": False,
        "training_corpus_authorized": False,
        "production_release_authorized": False,
        "terminal_done": False,
    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}, train_payloads


def stage(root: Path, destination: Path) -> dict[str, Any]:
    need(not any(p.is_symlink() for p in (destination, *destination.parents)),
         "symlink train publication path")
    report, payloads = build(root)
    destination.mkdir(parents=True, exist_ok=True)
    expected = set(payloads) | {OUTPUT}
    for member in destination.rglob("*"):
        need(not member.is_symlink() and (member.is_file() or member.is_dir()),
             "unsafe source partition publication member")
        if member.is_file():
            need(member.relative_to(destination).as_posix() in expected,
                 "unexpected/stale physical partition member")
    for name, raw in sorted(payloads.items()):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        need(not target.parent.is_symlink() and not target.is_symlink(),
             "symlinked training text member")
        if target.exists():
            need(target.is_file() and _read_destination(target) == raw,
                 "immutable train source modified")
        else:
            _atomic_write(destination, target, raw)
        need(_read_destination(target) == raw, "train text readback drift")
    output = destination / OUTPUT
    canonical = books.canonical(report)
    if output.exists() or output.is_symlink():
        need(output.is_file() and not output.is_symlink()
             and _read_destination(output) == canonical,
             "immutable train partition changed")
    else:
        _atomic_write(destination, output, canonical)
    need(_read_destination(output) == canonical, "partition readback drift")
    return report


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
