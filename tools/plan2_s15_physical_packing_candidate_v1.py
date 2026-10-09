"""S15: physical tokenizer -> deterministic bounded shards, candidate only.

Reuses S12 FrozenBPE, S13 pack/byte encodings, original public-domain source
and exact three-book heldout audit. Not a production release or Plan9 grant.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import plan2_deterministic_packing_v1 as packing
from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_physical_tokenizer_candidate_v1 as realfit
from tools import plan2_tokenizer_fit_freeze_v1 as frozen
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-s15-physical-packed-shards-candidate.v1"
OUTPUT = "physical-shard-candidate.json"
BLOCKS_PER_SHARD = 64


class PhysicalPackingDenied(ValueError):
    """Untrusted physical tokenizer, shard identity or publication."""


def need(ok: bool, why: str) -> None:
    if not ok:
        raise PhysicalPackingDenied(why)


def build(root: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    root = root.resolve(strict=True)
    fitted = realfit.inspect(root)
    need(fitted["byte_roundtrip_verified"] is True
         and fitted["terminal_done"] is False
         and fitted["production_release_authorized"] is False
         and fitted["physical_s9_admitted"] is False,
         "unqualified tokenizer authority")
    cohort = books.inspect(root)
    docs = [x for x in cohort["books"]
            if x["source_id"] == fitted["train_document_source_id"]]
    need(len(docs) == 1, "source document missing from verified corpus")
    doc = docs[0]
    payload = books.read_checked(root, doc["snapshot_path"])
    need(books.sha(payload) == fitted["train_document_sha256"]
         and len(payload) == fitted["train_document_bytes"],
         "physically fitted source bytes changed")
    rows = realfit.segments(doc["source_id"], payload.decode("utf-8", "strict"))
    token_model = frozen.FrozenBPE(fitted["fit_merges"])
    need(token_model.identity.to_dict() == fitted["tokenizer_identity"]
         and len(rows) == fitted["train_record_count"],
         "physical tokenizer source and model identity mismatch")
    ids = [x["record_id"] for x in rows]
    need(len(ids) == len(set(ids))
         and set(ids).isdisjoint({
             doc["source_id"] + ":r00000000",
             doc["source_id"] + ":r00000001",
         }), "S10 physical span IDs were reused by the S15 tokenizer")
    # The existing S13 packer enforces BOS/EOS, exact roundtrip,
    # next-token labels, masks and record boundaries for every physical row.
    blocks: list[dict[str, Any]] = []
    for row in rows:
        blocks.extend(packing.pack(token_model, [{
            "record_id": row["record_id"], "source_id": doc["source_id"],
            "text": row["text"],
        }]))
    need(bool(blocks), "no real physical training targets")
    shard_bytes: dict[str, bytes] = {}
    shard_receipts: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    target_count = 0
    for offset in range(0, len(blocks), BLOCKS_PER_SHARD):
        index = offset // BLOCKS_PER_SHARD
        name = f"shards/shard-{index:06d}.json"
        group = blocks[offset:offset + BLOCKS_PER_SHARD]
        raw = packing.canonical({
            "schema_version": "12-6.plan2-token-shard.v1",
            "shard_index": index,
            "block_size": packing.BLOCK,
            "blocks": group,
        })
        digest = packing.digest(raw)
        shard_bytes[name] = raw
        targets = sum(block["target_count"] for block in group)
        target_count += targets
        shard_receipts.append({
            "path": name, "sha256": digest,
            "block_count": len(group), "target_count": targets,
            "byte_count": len(raw),
        })
        for block_index, block in enumerate(group):
            mapping.append({
                "record_id": block["record_id"],
                "source_id": block["source_id"],
                "source_sha256": block["source_sha256"],
                "token_offset": block["token_offset"],
                "target_count": block["target_count"],
                "shard_path": name,
                "block_ordinal": block_index,
            })
    need(target_count > 0 and sum(x["block_count"] for x in shard_receipts)
         == len(blocks) == len(mapping), "lost or repeated physical block")
    core = {
        "schema_version": SCHEMA,
        "decision": "REAL_PHYSICAL_PACKING_CANDIDATE_NOT_RELEASE",
        "migration_contract": "migration-contract-baseline-v1",
        "source_manifest_sha256": cohort["manifest_sha256"],
        "train_document_sha256": fitted["train_document_sha256"],
        "tokenizer_manifest_sha256": fitted["manifest_sha256"],
        "tokenizer_identity": fitted["tokenizer_identity"],
        "heldout_audit_sha256": fitted["actual_heldout_audit_sha256"],
        "cluster_split_sha256": fitted["physical_split_probe_sha256"],
        "packing_policy": "physical_no_cross_record_no_truncation_v1",
        "block_size": packing.BLOCK,
        "blocks_per_shard": BLOCKS_PER_SHARD,
        "train_record_ids": ids,
        "block_count": len(blocks),
        "target_count": target_count,
        "shards": shard_receipts,
        "source_token_shard_mapping": mapping,
        "physical_s9_admitted": False,
        "training_corpus_authorized": False,
        "production_release_authorized": False,
        "terminal_done": False,
        "paid_compute_used": False,
    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}, shard_bytes


def stage(root: Path, destination: Path) -> dict[str, Any]:
    need(not any(p.is_symlink() for p in (destination, *destination.parents)),
         "symlink publication path")
    proof, payloads = build(root)
    destination.mkdir(parents=True, exist_ok=True)
    for name, raw in sorted(payloads.items()):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        need(not target.is_symlink() and not target.parent.is_symlink(),
             "unsafe shard publication path")
        if target.exists():
            need(target.is_file() and _read_destination(target) == raw,
                 "immutable physical shard changed")
        else:
            _atomic_write(destination, target, raw)
        need(_read_destination(target) == raw, "shard hash readback drift")
    receipt = destination / OUTPUT
    raw = books.canonical(proof)
    if receipt.exists() or receipt.is_symlink():
        need(receipt.is_file() and not receipt.is_symlink()
             and _read_destination(receipt) == raw,
             "immutable physical shard manifest changed")
    else:
        _atomic_write(destination, receipt, raw)
    need(_read_destination(receipt) == raw, "physical shard manifest readback drift")
    return proof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    result = stage(args.root, args.out_dir)
    print(json.dumps({
        "decision": result["decision"],
        "manifest_sha256": result["manifest_sha256"],
        "target_count": result["target_count"],
        "production_release_authorized": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
