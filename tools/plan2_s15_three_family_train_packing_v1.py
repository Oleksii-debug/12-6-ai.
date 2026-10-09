"""S15 real 3-family train-only S13 shards, reuse incumbent S12/S13 machinery.

The input is the exact physical S10 whole-source train partition only; no
validation or EVAL233 final-test payload is opened for packing. This module
reuses frozen 32K ByteBPE, the incumbent S13 pack and immutable physical publisher.
All products are NONRELEASE until original Plan2 S3-S9 and Plan9 admission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import plan2_deterministic_packing_v1 as packing
from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_physical_packing_candidate_v1 as physical
from tools import plan2_s15_physical_tokenizer_candidate_v1 as segmenter
from tools import plan2_s15_real_32k_bpe_v1 as fit
from tools import plan2_s15_three_family_train_materialization_v1 as training

SCHEMA = physical.SCHEMA
OUTPUT = physical.OUTPUT
BLOCKS_PER_SHARD = physical.BLOCKS_PER_SHARD


class ThreeFamilyPackingDenied(ValueError):
    """Physical source, BPE, masks, and S10 lineage must agree."""


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise ThreeFamilyPackingDenied(reason)


def build(root: Path, *, fitted: dict[str, Any] | None = None
          ) -> tuple[dict[str, Any], dict[str, bytes]]:
    root = root.resolve(strict=True)
    partition, members = training.build(root)
    fitted = fit.inspect(root) if fitted is None else fitted
    need(type(fitted) is dict
         and fitted.get("schema_version") == fit.SCHEMA
         and fitted.get("manifest_sha256") == books.sha(books.canonical({
             k: v for k, v in fitted.items() if k != "manifest_sha256"
         })),
         "precomputed 32K physical fit authority or hash was tampered")
    need(partition.get("train_source_family_count") == 3
         and partition["train_document_count"] == 13,
         "physical multi-family training records were not fully admitted as candidates")
    need(fitted["source_train_partition_sha256"] == partition["manifest_sha256"]
         and fitted["source_s10_split_sha256"] ==
             partition["s10_split_manifest_sha256"]
         and fitted["source_train_document_count"] == len(members)
         == partition["train_document_count"]
         and fitted["source_train_utf8_bytes"] == partition["physical_train_bytes"]
         and fitted["heldout_payloads_fitted"] is False
         and fitted["actual_vocab_size"] == 32768
         and fitted["production_train_source_admitted"] is False
         and fitted["production_release_authorized"] is False
         and partition["heldout_plaintext_materialized"] is False
         and partition["physical_s9_admitted"] is False
         and partition["training_corpus_authorized"] is False,
         "physical S10 train-only tokens/rights or holdout boundary changed")
    tokenizer = fit.ByteBPE32k(fitted["fitted_merges"])
    need(tokenizer.identity.to_dict() == fitted["tokenizer_identity"]
         and tokenizer.vocab_size == fitted["actual_vocab_size"],
         "S12 frozen training tokenizer identity mismatch")
    items = sorted(partition["train_members"], key=lambda row: row["path"])
    need(len(items) > 0 and len({x["path"] for x in items}) == len(items),
         "physical S10 train source membership invalid")
    blocks: list[dict[str, Any]] = []
    segment_map: list[dict[str, Any]] = []
    ids: list[str] = []
    train_document_map: list[dict[str, Any]] = []
    source_byte_offset = 0
    for item in items:
        path = item["path"]
        need(type(path) is str and path.startswith("train/")
             and path in members and type(item.get("source_id")) is str,
             "training input missing true source provenance")
        raw = members[path]
        need(len(raw) == item["normalized_bytes"]
             and books.sha(raw) == item["normalized_sha256"],
             "S10 real training bytes drifted before packing")
        opaque_source_id = item["record_id_sha256"]
        document_segments = segmenter.segments(
            opaque_source_id, raw.decode("utf-8", "strict"),
        )
        document_start = source_byte_offset
        for row in document_segments:
            rid = row["record_id"]
            source = row["text"].encode("utf-8", "strict")
            need(bool(source), "physical training segment empty")
            ids.append(rid)
            segment_map.append({
                "record_id": rid,
                "source_id": opaque_source_id,
                "origin_source_id": item["source_id"],
                "origin_member_sha256": item["normalized_sha256"],
                "source_byte_start": source_byte_offset,
                "source_byte_end": source_byte_offset + len(source),
                "source_sha256": books.sha(source),
            })
            source_byte_offset += len(source)
            blocks.extend(packing.pack(tokenizer, [{
                "record_id": rid, "source_id": opaque_source_id,
                "text": row["text"],
            }]))
        need(source_byte_offset - document_start == len(raw),
             "a real training document was truncated")
        train_document_map.append({
            "source_id": item["source_id"],
            "opaque_source_id": opaque_source_id,
            "normalized_sha256": item["normalized_sha256"],
            "byte_start": document_start,
            "byte_end": source_byte_offset,
            "segment_count": len(document_segments),
        })
    need(source_byte_offset == partition["physical_train_bytes"]
         and len(ids) == len(set(ids))
         and len(segment_map) == fitted["source_train_record_count"]
         and len(train_document_map) == fitted["source_train_document_count"]
         and bool(blocks), "S10 training bytes, segment count or IDs changed")
    shard_bytes: dict[str, bytes] = {}
    receipts: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    token_count = 0
    for offset in range(0, len(blocks), BLOCKS_PER_SHARD):
        shard_no = offset // BLOCKS_PER_SHARD
        name = f"shards/shard-{shard_no:06d}.json"
        group = blocks[offset:offset + BLOCKS_PER_SHARD]
        raw = packing.canonical({
            "schema_version": "12-6.plan2-token-shard.v1",
            "shard_index": shard_no, "block_size": packing.BLOCK,
            "blocks": group,
        })
        shard_bytes[name] = raw
        targets = sum(b["target_count"] for b in group)
        token_count += targets
        receipts.append({
            "path": name, "sha256": packing.digest(raw),
            "block_count": len(group), "target_count": targets,
            "byte_count": len(raw),
        })
        for ordinal, block in enumerate(group):
            mapping.append({
                "record_id": block["record_id"],
                "source_id": block["source_id"],
                "source_sha256": block["source_sha256"],
                "token_offset": block["token_offset"],
                "target_count": block["target_count"],
                "shard_path": name, "block_ordinal": ordinal,
            })
    need(token_count > 0 and len(mapping) == len(blocks)
         and len(shard_bytes) == len(receipts)
         and sum(x["block_count"] for x in receipts) == len(blocks)
         and sum(x["target_count"] for x in receipts) == token_count,
         "physical token targets or shard blocks were lost")
    core = {
        "schema_version": SCHEMA,
        "decision": "REAL_PHYSICAL_PACKING_CANDIDATE_NOT_RELEASE",
        "migration_contract": "migration-contract-baseline-v1",
        "source_manifest_sha256": partition["upstream_three_family_sha256"],
        "source_train_partition_sha256": partition["manifest_sha256"],
        "train_document_bytes": partition["physical_train_bytes"],
        "train_document_count": partition["train_document_count"],
        "train_document_map": train_document_map,
        "heldout_document_count": partition["heldout_document_count"],
        "heldout_plaintext_materialized": False,
        "tokenizer_manifest_sha256": fitted["manifest_sha256"],
        "tokenizer_identity": fitted["tokenizer_identity"],
        "cluster_split_sha256": fitted["source_s10_split_sha256"],
        "physical_split_probe_sha256": partition["s10_split_probe_sha256"],
        "packing_policy": "physical_no_cross_record_no_truncation_v1",
        "block_size": packing.BLOCK,
        "blocks_per_shard": BLOCKS_PER_SHARD,
        "train_record_ids": ids,
        "source_segment_byte_map": segment_map,
        "block_count": len(blocks), "target_count": token_count,
        "shards": receipts,
        "source_token_shard_mapping": mapping,
        "physical_s9_admitted": False,
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "production_release_authorized": False,
        "terminal_done": False,
        "paid_compute_used": False,
    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}, shard_bytes


def stage(root: Path, destination: Path) -> dict[str, Any]:
    """Reuse existing immutable physical S13 writer for all three families."""
    return physical.stage_manifest(*build(root), destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = stage(args.root, args.out_dir)
    print(json.dumps({
        "decision": receipt["decision"],
        "manifest_sha256": receipt["manifest_sha256"],
        "target_count": receipt["target_count"],
        "production_release_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
