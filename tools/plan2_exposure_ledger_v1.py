"""Plan 2 S14: ordered LOCAL_FREE target/exposure ledger, not optimizer authority."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import plan2_deterministic_packing_v1 as packing
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-ordered-exposure-ledger.v1"
CONTRACT = "migration-contract-baseline-v1"
FILE = "exposure-ledger.json"
ZERO = "0" * 64


class ExposureDenied(ValueError):
    """Invalid identity, unsafe artifact or non-contiguous replay."""


def need(condition: bool, reason: str) -> None:
    if not condition:
        raise ExposureDenied(reason)


def canonical(value: Any) -> bytes:
    return packing.canonical(value)


def hashed(value: Any) -> str:
    return packing.digest(canonical(value))


def _safe_path(path: Path) -> None:
    need(not any(member.is_symlink() for member in (path, *path.parents)),
         "symlink destination")


def build(root: Path, shards_dir: Path) -> dict:
    """Rebuild the complete ordered stream from independently verified S13 shards."""
    packed, _ = packing.build(root)
    blocks = packing.read_blocks(root, shards_dir)
    mapping = packed["source_token_shard_mapping"]
    need(len(blocks) == len(mapping) == packed["block_count"], "block-map drift")
    need(packed["training_corpus_authorized"] is False
         and packed["production_release_authorized"] is False,
         "unexpected corpus authority")
    entries: list[dict] = []
    prev_chain = ZERO
    offsets: dict[str, int] = {}
    target_keys: set[str] = set()
    exposure_keys: set[str] = set()
    for block_index, (read, link) in enumerate(zip(blocks, mapping, strict=True)):
        block = read["block"]
        need(read["block_index"] == block_index
             and read["next_block"] == block_index + 1
             and all(block[key] == link[key] for key in
                     ("record_id", "source_id", "source_sha256", "token_offset",
                      "target_count")), "source/block binding drift")
        count = block["target_count"]
        rid = block["record_id"]
        need(type(count) is int and 0 < count <= packing.BLOCK
             and type(block["token_offset"]) is int
             and block["token_offset"] == offsets.get(rid, 0),
             "skipped or repeated token offset")
        need(len(block["target_ids"]) == packing.BLOCK
             and block["loss_mask"] == [1] * count + [0] * (packing.BLOCK - count)
             and block["attention_mask"] == block["loss_mask"],
             "invalid next-token boundary")
        offsets[rid] = block["token_offset"] + count
        for position in range(count):
            index = len(entries)
            token = block["target_ids"][position]
            need(type(token) is int and token >= 0, "invalid token ID")
            target_basis = {
                "contract": CONTRACT,
                "split_sha256": packed["split_manifest_sha256"],
                "tokenizer_sha256": packed["tokenizer_manifest_sha256"],
                "record_id": rid, "source_sha256": block["source_sha256"],
                "token_offset": block["token_offset"] + position,
                "target_token_id": token,
            }
            target_id = hashed(target_basis)
            exposure_id = hashed({"packing_sha256": packed["manifest_sha256"],
                                  "target_id": target_id,
                                  "global_index": index})
            need(target_id not in target_keys and exposure_id not in exposure_keys,
                 "duplicate target/exposure")
            target_keys.add(target_id)
            exposure_keys.add(exposure_id)
            body = {
                "index": index, "record_id": rid,
                "token_offset": target_basis["token_offset"],
                "block_index": block_index,
                "shard_path": link["shard_path"],
                "position_in_block": position, "target_token_id": token,
                "target_id": target_id, "exposure_id": exposure_id,
            }
            prev_chain = hashed({"previous": prev_chain, "exposure": body})
            entries.append({**body, "chain_sha256": prev_chain})
    need(len(entries) == packed["target_count"] and len(entries) > 0,
         "target count mismatch")
    core = {
        "schema_version": SCHEMA, "migration_contract": CONTRACT,
        "purpose": "LOCAL_FREE_SYNTHETIC_COMPONENT",
        "packing_manifest_sha256": packed["manifest_sha256"],
        "split_manifest_sha256": packed["split_manifest_sha256"],
        "tokenizer_manifest_sha256": packed["tokenizer_manifest_sha256"],
        "block_count": packed["block_count"],
        "target_count": len(entries), "chain_head_sha256": prev_chain,
        "ordered_exposures": entries,
        "training_corpus_authorized": False,
        "optimizer_effect_authorized": False,
        "production_release_authorized": False,
        "paid_compute_used": False,
    }
    return {**core, "manifest_sha256": hashed(core)}


def stage(root: Path, shards_dir: Path, out: Path) -> dict:
    """No-clobber publish; existing tampered artifacts never become authority."""
    _safe_path(out)
    manifest = build(root, shards_dir)
    out.mkdir(parents=True, exist_ok=True)
    for member in out.iterdir():
        if member.name.startswith(".plan2-partial-"):
            need(member.is_file() and not member.is_symlink(),
                 "unsafe interrupted publication")
            member.unlink()
        else:
            need(member.name == FILE and member.is_file()
                 and not member.is_symlink(), "unexpected publication member")
    target = out / FILE
    raw = canonical(manifest)
    if target.exists() or target.is_symlink():
        need(not target.is_symlink() and _read_destination(target) == raw,
             "immutable ledger corruption")
    else:
        _atomic_write(out, target, raw)
    need(_read_destination(target) == raw, "ledger readback drift")
    return manifest


def resume_receipt(manifest: dict, next_target: int) -> dict:
    need(type(next_target) is int and 0 <= next_target <= manifest["target_count"],
         "invalid resume index")
    last = (ZERO if next_target == 0 else
            manifest["ordered_exposures"][next_target - 1]["chain_sha256"])
    body = {"schema_version": "12-6.plan2-exposure-cursor.v1",
            "manifest_sha256": manifest["manifest_sha256"],
            "next_target": next_target, "chain_before_next_sha256": last}
    return {**body, "receipt_sha256": hashed(body)}


def read_exposures(root: Path, shards_dir: Path, out: Path, *,
                   start_target: int = 0, expected_resume: dict | None = None
                   ) -> list[dict]:
    """Return verified suffix; consumer must atomically persist cursor with optimizer state."""
    need(type(start_target) is int and start_target >= 0, "invalid cursor")
    _safe_path(out)
    expected = build(root, shards_dir)
    try:
        need(out.is_dir() and {p.name for p in out.iterdir()} == {FILE},
             "missing/extra published members")
        data_path = out / FILE
        raw = _read_destination(data_path)
        actual = json.loads(raw.decode("utf-8", "strict"))
        need(canonical(actual) == raw and actual == expected,
             "ledger substitution")
        need(start_target <= actual["target_count"], "cursor past end")
        if expected_resume is not None:
            need(type(expected_resume) is dict
                 and expected_resume == resume_receipt(expected, start_target),
                 "stale/skipped/forged resume receipt")
    except (OSError, ValueError, UnicodeError, TypeError, KeyError) as exc:
        raise ExposureDenied("invalid ledger reader authority") from exc
    return [
        {"next_target": item["index"] + 1, **item}
        for item in expected["ordered_exposures"][start_target:]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--shards-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    m = stage(args.root, args.shards_dir, args.out_dir)
    print(json.dumps({"manifest_sha256": m["manifest_sha256"],
                      "chain_head_sha256": m["chain_head_sha256"],
                      "target_count": m["target_count"],
                      "training_corpus_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()
