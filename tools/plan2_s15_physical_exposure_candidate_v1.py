"""S15 physical ordered exposure ledger with S14 target IDs, candidate only.

This is a bound replay of actual real-text physical shards with the incumbent
S14 hash algorithm. It does not authorize an optimizer, training, or Plan9.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import plan2_deterministic_packing_v1 as packing
from tools import plan2_exposure_ledger_v1 as exposure
from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_physical_packing_candidate_v1 as physical
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-s15-physical-ordered-exposure-candidate.v1"
OUTPUT = "physical-exposure-candidate.json"
ZERO = "0" * 64


class PhysicalExposureDenied(ValueError):
    """Corrupt shards, repeated targets, missing source or unsafe replay."""


def need(ok: bool, why: str) -> None:
    if not ok:
        raise PhysicalExposureDenied(why)


def build(root: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    packed, physical_bytes = physical.build(root)
    need(packed["target_count"] > 0
         and packed["production_release_authorized"] is False
         and packed["physical_s9_admitted"] is False,
         "physical packer incorrectly promoted unadmitted corpus")
    mapping = packed["source_token_shard_mapping"]
    need(len(mapping) == packed["block_count"], "physical block ledger missing")
    offsets: dict[str, int] = {}
    seen_targets: set[str] = set()
    seen_exposures: set[str] = set()
    shard_receipts: list[dict[str, Any]] = []
    outputs: dict[str, bytes] = {}
    previous = ZERO
    index = 0
    mapped_index = 0
    for meta in packed["shards"]:
        source_name = meta["path"]
        need(source_name in physical_bytes
             and packing.digest(physical_bytes[source_name]) == meta["sha256"],
             "source shard SHA mismatch")
        shard = json.loads(physical_bytes[source_name].decode("utf-8"))
        blocks = shard["blocks"]
        need(shard["block_size"] == packed["block_size"]
             and len(blocks) == meta["block_count"],
             "physical block count or policy mismatch")
        entries: list[dict[str, Any]] = []
        for ordinal, block in enumerate(blocks):
            need(mapped_index < len(mapping), "unexpected physical block")
            link = mapping[mapped_index]
            mapped_index += 1
            need(link["shard_path"] == source_name
                 and link["block_ordinal"] == ordinal
                 and all(link[key] == block[key] for key in (
                     "record_id", "source_id", "source_sha256", "token_offset",
                     "target_count",
                 )), "physical source-to-block provenance drift")
            rid = block["record_id"]
            count = block["target_count"]
            offset = block["token_offset"]
            need(type(count) is int and 0 < count <= packing.BLOCK
                 and offset == offsets.get(rid, 0),
                 "physical loss offset skipped or replayed")
            need(len(block["target_ids"]) == packing.BLOCK
                 and block["loss_mask"] == [1] * count
                 + [0] * (packing.BLOCK - count)
                 and block["attention_mask"] == block["loss_mask"],
                 "masked physical next-token target invariant broken")
            offsets[rid] = offset + count
            for position in range(count):
                token = block["target_ids"][position]
                need(type(token) is int and token >= 0,
                     "invalid physical token ID")
                basis = {
                    "contract": exposure.CONTRACT,
                    "split_sha256": packed["cluster_split_sha256"],
                    "tokenizer_sha256": packed["tokenizer_manifest_sha256"],
                    "record_id": rid,
                    "source_sha256": block["source_sha256"],
                    "token_offset": offset + position,
                    "target_token_id": token,
                }
                target_id = exposure.hashed(basis)
                exposure_id = exposure.hashed({
                    "packing_sha256": packed["manifest_sha256"],
                    "target_id": target_id,
                    "global_index": index,
                })
                need(target_id not in seen_targets
                     and exposure_id not in seen_exposures,
                     "duplicate physical target or exposure identity")
                seen_targets.add(target_id)
                seen_exposures.add(exposure_id)
                row = {
                    "index": index, "record_id": rid,
                    "token_offset": offset + position,
                    "shard_path": source_name,
                    "block_ordinal": ordinal,
                    "position_in_block": position,
                    "target_token_id": token,
                    "target_id": target_id,
                    "exposure_id": exposure_id,
                }
                previous = exposure.hashed({
                    "previous": previous, "exposure": row,
                })
                entries.append({**row, "chain_sha256": previous})
                index += 1
        outname = source_name.replace("shards/", "exposures/", 1)
        need(outname != source_name, "physical exposure path mismatch")
        raw = books.canonical({
            "schema_version": "12-6.plan2-s15-physical-exposure-shard.v1",
            "source_shard_sha256": meta["sha256"],
            "source_shard_path": source_name,
            "ordered_exposures": entries,
        })
        outputs[outname] = raw
        shard_receipts.append({
            "source_shard_path": source_name,
            "path": outname,
            "sha256": books.sha(raw),
            "exposure_count": len(entries),
            "chain_tail_sha256": previous,
            "byte_count": len(raw),
        })
    need(mapped_index == packed["block_count"]
         and index == packed["target_count"] == len(seen_targets)
         == len(seen_exposures) and bool(shard_receipts),
         "physical targets missing or duplicated")
    core = {
        "schema_version": SCHEMA,
        "decision": "PHYSICAL_ORDERED_TARGETS_CANDIDATE_NOT_RELEASE",
        "migration_contract": exposure.CONTRACT,
        "packing_manifest_sha256": packed["manifest_sha256"],
        "cluster_split_sha256": packed["cluster_split_sha256"],
        "tokenizer_manifest_sha256": packed["tokenizer_manifest_sha256"],
        "target_count": index,
        "block_count": packed["block_count"],
        "source_shard_count": len(packed["shards"]),
        "exposure_shards": shard_receipts,
        "chain_head_sha256": previous,
        "training_corpus_authorized": False,
        "optimizer_effect_authorized": False,
        "production_release_authorized": False,
        "terminal_done": False,
        "paid_compute_used": False,
    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}, outputs


def stage(root: Path, destination: Path) -> dict[str, Any]:
    need(not any(p.is_symlink() for p in (destination, *destination.parents)),
         "symlink exposure publication")
    manifest, payloads = build(root)
    destination.mkdir(parents=True, exist_ok=True)
    expected = set(payloads) | {OUTPUT}
    for path in destination.rglob("*"):
        need(not path.is_symlink() and (path.is_file() or path.is_dir()),
             "unsafe physical exposure member")
        if path.is_file():
            need(path.relative_to(destination).as_posix() in expected,
                 "unexpected physical exposure publication member")
    for name, raw in sorted(payloads.items()):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        need(not target.is_symlink() and not target.parent.is_symlink(),
             "unsafe exposure path")
        if target.exists():
            need(target.is_file() and _read_destination(target) == raw,
                 "immutable physical exposures changed")
        else:
            _atomic_write(destination, target, raw)
        need(_read_destination(target) == raw, "physical exposure readback drift")
    target = destination / OUTPUT
    raw = books.canonical(manifest)
    if target.exists() or target.is_symlink():
        need(target.is_file() and not target.is_symlink()
             and _read_destination(target) == raw,
             "immutable exposure manifest changed")
    else:
        _atomic_write(destination, target, raw)
    need(_read_destination(target) == raw, "physical exposure manifest readback drift")
    return manifest


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
