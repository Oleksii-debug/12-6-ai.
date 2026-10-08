"""Plan 2 S13: deterministic fixture tokenization, packing and shards; NO training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from tools import plan2_cluster_split_v1 as split
from tools import plan2_tokenizer_fit_freeze_v1 as fit
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-packing-fixture.v1"
BLOCK = 32
PER_SHARD = 4


class PackingDenied(ValueError):
    """Untrusted parent, malformed record or immutable publication violation."""


def need(ok: bool, why: str) -> None:
    if not ok:
        raise PackingDenied(why)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return fit.canonical(value)


def inputs(root: Path) -> tuple[dict, dict, Any, list[dict]]:
    frozen = fit.freeze_fixture(root)
    tokenizer = fit.verify_frozen(frozen)
    s9, records = split._synthetic_rows()
    policy = split._policy((root / split.POLICY_PATH).read_bytes())
    selected = split.build_cluster_split(s9, records, policy, fixture=True)
    need(selected["split_manifest_sha256"] == frozen["s10_split_manifest_sha256"]
         and selected["purpose"] == "LOCAL_FREE_SYNTHETIC_COMPONENT"
         and selected["cluster_leakage_count"] == 0
         and selected["training_corpus_authorized"] is False
         and selected["tokenizer_fit_authorized"] is False
         and frozen["training_corpus_authorized"] is False
         and frozen["production_release_authorized"] is False,
         "unqualified S10/S12 fixture")
    train = set(selected["train_record_ids"])
    need(train.isdisjoint(selected["validation_record_ids"])
         and train.isdisjoint(selected["test_record_ids"]), "holdout exposure")
    rows = sorted((row for row in records if row["record_id"] in train),
                  key=lambda row: row["record_id"])
    need(len(rows) == len(train) == frozen["train_record_count"],
         "missing or duplicate train record")
    return frozen, selected, tokenizer, rows


def pack(tokenizer: Any, rows: list[dict]) -> list[dict]:
    need(type(rows) is list and 0 < len(rows) <= 1000, "invalid row count")
    blocks: list[dict] = []
    seen: set[str] = set()
    for row in sorted(rows, key=lambda item: item.get("record_id", "")):
        need(type(row) is dict and set(row) == {"record_id", "source_id", "text"},
             "malformed record")
        rid, source, text = row["record_id"], row["source_id"], row["text"]
        need(type(rid) is str and type(source) is str and source
             and rid.startswith(source + ":r") and rid not in seen
             and type(text) is str and text, "invalid/duplicate identity")
        seen.add(rid)
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        need(len(ids) >= 3 and ids[0] == tokenizer.bos_id
             and ids[-1] == tokenizer.eos_id
             and tokenizer.decode(ids) == text, "tokenizer mismatch")
        features, targets = ids[:-1], ids[1:]
        for offset in range(0, len(targets), BLOCK):
            count = min(BLOCK, len(targets) - offset)
            blocks.append({
                "record_id": rid,
                "source_id": source,
                "source_sha256": digest(text.encode("utf-8", "strict")),
                "token_offset": offset,
                "target_count": count,
                "input_ids": features[offset:offset+count]
                + [tokenizer.pad_id] * (BLOCK - count),
                "target_ids": targets[offset:offset+count]
                + [tokenizer.pad_id] * (BLOCK - count),
                "attention_mask": [1] * count + [0] * (BLOCK - count),
                "loss_mask": [1] * count + [0] * (BLOCK - count),
            })
    need(0 < len(blocks) <= 4096, "invalid block count")
    return blocks


def build(root: Path) -> tuple[dict, dict[str, bytes]]:
    frozen, selected, tokenizer, rows = inputs(root)
    blocks = pack(tokenizer, rows)
    shards: dict[str, bytes] = {}
    refs: list[dict] = []
    mapping: list[dict] = []
    for offset in range(0, len(blocks), PER_SHARD):
        index = offset // PER_SHARD
        name = f"shards/shard-{index:06d}.json"
        group = blocks[offset:offset + PER_SHARD]
        raw = canonical({
            "schema_version": "12-6.plan2-token-shard.v1",
            "shard_index": index, "block_size": BLOCK, "blocks": group,
        })
        shards[name] = raw
        refs.append({"path": name, "sha256": digest(raw),
                     "block_count": len(group),
                     "target_count": sum(b["target_count"] for b in group)})
        for ordinal, block in enumerate(group):
            mapping.append({
                "record_id": block["record_id"], "source_id": block["source_id"],
                "source_sha256": block["source_sha256"],
                "token_offset": block["token_offset"],
                "target_count": block["target_count"],
                "shard_path": name, "block_ordinal": ordinal,
            })
    core = {
        "schema_version": SCHEMA,
        "purpose": "LOCAL_FREE_SYNTHETIC_COMPONENT",
        "packing_policy": "no-cross-record_no-truncation_v1",
        "block_size": BLOCK, "blocks_per_shard": PER_SHARD,
        "tokenizer_manifest_sha256": frozen["manifest_sha256"],
        "tokenizer_identity": frozen["tokenizer_identity"],
        "split_manifest_sha256": selected["split_manifest_sha256"],
        "upstream_s9_sha256": selected["upstream_s9_dataset_candidate_sha256"],
        "train_record_ids": selected["train_record_ids"],
        "block_count": len(blocks),
        "target_count": sum(b["target_count"] for b in blocks),
        "shards": refs, "source_token_shard_mapping": mapping,
        "training_corpus_authorized": False,
        "production_release_authorized": False,
        "paid_compute_used": False,
    }
    return {**core, "manifest_sha256": digest(canonical(core))}, shards


def verify(root: Path, manifest: dict, shards: dict[str, bytes]) -> None:
    need(type(manifest) is dict and manifest.get("schema_version") == SCHEMA,
         "unknown manifest")
    core = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    need(manifest.get("manifest_sha256") == digest(canonical(core)),
         "manifest digest drift")
    expected, bytes_by_name = build(root)
    need(manifest == expected and shards == bytes_by_name,
         "unauthorized manifest/shard substitution")
    need(manifest["training_corpus_authorized"] is False
         and manifest["production_release_authorized"] is False
         and manifest["paid_compute_used"] is False, "unauthorized release")


def stage(root: Path, destination: Path) -> dict:
    need(not any(p.is_symlink() for p in (destination, *destination.parents)),
         "symlink output")
    manifest, shards = build(root)
    verify(root, manifest, shards)
    destination.mkdir(parents=True, exist_ok=True)
    folder = destination / "shards"
    for member in destination.iterdir():
        if member.name.startswith(".plan2-partial-"):
            need(member.is_file() and not member.is_symlink(),
                 "unsafe interrupted staging member")
            member.unlink()
        else:
            need(member.name in {"shards", "packing-manifest.json"},
                 "unexpected destination member")
    need(not folder.is_symlink(), "symlink shard folder")
    if folder.exists():
        need(folder.is_dir(), "non-directory shard path")
        for member in folder.iterdir():
            if member.name.startswith(".plan2-partial-"):
                need(member.is_file() and not member.is_symlink(),
                     "unsafe interrupted shard")
                member.unlink()
        present = {f"shards/{p.name}" for p in folder.iterdir()}
        manifest_exists = (destination / "packing-manifest.json").exists()
        need(present.issubset(set(shards))
             and (not manifest_exists or present == set(shards)),
             "unexpected/missing shard")
    else:
        folder.mkdir()
    for name, raw in shards.items():
        target = destination / name
        if target.exists() or target.is_symlink():
            need(not target.is_symlink() and _read_destination(target) == raw,
                 "corrupt immutable shard")
        else:
            _atomic_write(folder, target, raw)
        need(_read_destination(target) == raw, "shard readback drift")
    target = destination / "packing-manifest.json"
    raw = canonical(manifest)
    if target.exists() or target.is_symlink():
        need(not target.is_symlink() and _read_destination(target) == raw,
             "corrupt immutable manifest")
    else:
        _atomic_write(destination, target, raw)
    need(_read_destination(target) == raw, "manifest readback drift")
    return manifest



def read_blocks(root: Path, destination: Path, *, start_block: int = 0) -> list[dict]:
    """Fail-closed, optimizer-independent deterministic restart from a block cursor.

    Validate the complete immutable publication before returning any block.
    A saved `next_block` is the only resume cursor; the caller owns optimizer
    state and must commit that cursor atomically with its own state.
    """
    need(type(start_block) is int and start_block >= 0,
         "invalid restart block cursor")
    need(not any(path.is_symlink() for path in (destination, *destination.parents)),
         "symlink output")
    expected, expected_shards = build(root)
    manifest_path = destination / "packing-manifest.json"
    folder = destination / "shards"
    need(folder.is_dir() and not folder.is_symlink(), "missing/unsafe shard folder")
    try:
        raw = _read_destination(manifest_path)
        manifest = json.loads(raw.decode("utf-8", "strict"))
        need(type(manifest) is dict and canonical(manifest) == raw,
             "noncanonical manifest")
        need(manifest == expected, "manifest authority drift")
        need({p.name for p in destination.iterdir()}
             == {"packing-manifest.json", "shards"},
             "unexpected publication member")
        need(start_block <= manifest["block_count"], "cursor past end")
        need({item.name for item in folder.iterdir()}
             == {Path(path).name for path in expected_shards},
             "missing/unexpected shard")
        verified: list[dict] = []
        for ref in manifest["shards"]:
            path = destination / ref["path"]
            actual = _read_destination(path)
            need(actual == expected_shards[ref["path"]]
                 and digest(actual) == ref["sha256"], "shard bytes/hash drift")
            data = json.loads(actual.decode("utf-8", "strict"))
            need(canonical(data) == actual, "noncanonical shard")
            verified.extend(data["blocks"])
    except (OSError, ValueError, UnicodeError, TypeError, KeyError) as exc:
        raise PackingDenied("invalid shard reader authority") from exc
    need(len(verified) == manifest["block_count"], "block count drift")
    return [
        {"block_index": i, "next_block": i + 1, "block": block}
        for i, block in enumerate(verified)
        if i >= start_block
    ]

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    result = stage(args.root, args.out_dir)
    print(json.dumps({"manifest_sha256": result["manifest_sha256"],
                      "shard_count": len(result["shards"]),
                      "target_count": result["target_count"],
                      "training_corpus_authorized": False}, sort_keys=True))


if __name__ == "__main__":
    main()
