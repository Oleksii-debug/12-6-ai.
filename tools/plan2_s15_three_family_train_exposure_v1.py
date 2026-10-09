"""S15 3-family S13 -> incumbent S14 -> independent replay, candidate only.

Reuse actual physical train-only S13 packing, canonical incumbent S14 exposure
IDs and ordered-chain logic and existing independent S13/S14 reader.
Never reconstruct or publish validation/final-test payloads in this lane.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_physical_exposure_candidate_v1 as exposure
from tools import plan2_s15_physical_packing_candidate_v1 as physical
from tools import plan2_s15_physical_readback_v1 as readback
from tools import plan2_s15_three_family_train_packing_v1 as train_packing
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-s15-three-family-s13-s14-verified-candidate.v1"
OUTPUT = "three-family-physical-s13-s14-candidate.json"


class TrainExposureDenied(ValueError):
    """Physical training shards/exposures are not complete or reproducible."""


def need(condition: bool, message: str) -> None:
    if not condition:
        raise TrainExposureDenied(message)


def stage(root: Path, destination: Path, *,
          fitted: dict[str, Any] | None = None) -> dict[str, Any]:
    need(not any(x.is_symlink() for x in (destination, *destination.parents)),
         "symlink physical training publication")
    packed, shard_bytes = (train_packing.build(root) if fitted is None
                           else train_packing.build(root, fitted=fitted))
    need(packed["source_train_partition_sha256"]
         and packed["source_manifest_sha256"]
         and packed["train_document_count"] > 0
         and packed["heldout_document_count"] > 0
         and packed["heldout_plaintext_materialized"] is False
         and packed["production_release_authorized"] is False
         and packed["training_corpus_authorized"] is False,
         "three-family S10 train-only source was promoted")
    ledger, exposure_bytes = exposure.build_from_packed(packed, shard_bytes)
    need(ledger["packing_manifest_sha256"] == packed["manifest_sha256"]
         and ledger["cluster_split_sha256"] == packed["cluster_split_sha256"]
         and ledger["tokenizer_manifest_sha256"] ==
             packed["tokenizer_manifest_sha256"]
         and ledger["target_count"] == packed["target_count"]
         and ledger["block_count"] == packed["block_count"]
         and ledger["production_release_authorized"] is False
         and ledger["optimizer_effect_authorized"] is False,
         "S14 physical ordered source exposure drifted")
    destination.mkdir(parents=True, exist_ok=True)
    expected = {"shards", "exposures", OUTPUT}
    for member in destination.iterdir():
        need(not member.is_symlink() and member.name in expected,
             "unexpected or linked combined physical publication")
    shards_dir = destination / "shards"
    events_dir = destination / "exposures"
    physical.stage_manifest(packed, shard_bytes, shards_dir)
    exposure.stage_manifest(ledger, exposure_bytes, events_dir)
    proof = readback.verify(
        shards_dir, events_dir,
        expected_packed_sha256=packed["manifest_sha256"],
        expected_exposure_sha256=ledger["manifest_sha256"],
    )
    need(proof["readback_target_count"] == packed["target_count"]
         and proof["readback_block_count"] == packed["block_count"]
         and proof["last_chain_sha256"] == ledger["chain_head_sha256"]
         and proof["production_release_authorized"] is False
         and proof["terminal_done"] is False,
         "independent three-family target/exposure causal replay mismatch")
    core = {
        "schema_version": SCHEMA,
        "decision": "THREE_REAL_FAMILIES_S13_S14_PACKED_AND_REPLAYED_NOT_RELEASE",
        "source_cohort_manifest_sha256": packed["source_manifest_sha256"],
        "source_train_partition_sha256": packed["source_train_partition_sha256"],
        "source_s10_split_sha256": packed["cluster_split_sha256"],
        "frozen_s12_candidate_sha256": packed["tokenizer_manifest_sha256"],
        "physical_s13_shard_sha256": packed["manifest_sha256"],
        "physical_s14_exposure_sha256": ledger["manifest_sha256"],
        "independent_s13_s14_readback_sha256": proof["readback_sha256"],
        "train_document_count": packed["train_document_count"],
        "heldout_document_count": packed["heldout_document_count"],
        "physical_train_byte_count": packed["train_document_bytes"],
        "block_count": packed["block_count"],
        "target_count": packed["target_count"],
        "ordered_exposure_chain_sha256": ledger["chain_head_sha256"],
        "heldout_payloads_exposed": False,
        "real_final_test_outcomes_read": False,
        "plan9_optimizer_permission": False,
        "physical_s9_admitted": False,
        "training_corpus_authorized": False,
        "production_release_authorized": False,
        "terminal_done": False,
    }
    manifest = {**core, "manifest_sha256": books.sha(books.canonical(core))}
    target = destination / OUTPUT
    raw = books.canonical(manifest)
    if target.exists() or target.is_symlink():
        need(target.is_file() and not target.is_symlink()
             and _read_destination(target) == raw,
             "immutable three-family S13/S14 publication drifted")
    else:
        _atomic_write(destination, target, raw)
    need(_read_destination(target) == raw, "three-family readback changed")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    value = stage(args.root, args.out_dir)
    print(json.dumps({
        "decision": value["decision"],
        "manifest_sha256": value["manifest_sha256"],
        "target_count": value["target_count"],
        "terminal_done": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
