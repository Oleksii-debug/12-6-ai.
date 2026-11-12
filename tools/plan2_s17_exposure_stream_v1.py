"""Publish the incumbent S14 target and exposure stream for physical S17 shards."""
from __future__ import annotations

import json
import re
import stat
from collections import Counter
from pathlib import Path
from typing import Any

from tools import plan2_exposure_ledger_v1 as s14
from tools import plan2_public_domain_books_v1 as books
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-s17-ordered-exposure-ledger.v1"
SHARD_SCHEMA = "12-6.plan2-s15-physical-exposure-shard.v1"
MANIFEST = "exposure-ledger.json"
_HEX = re.compile(r"[0-9a-f]{64}\Z")
_BLOCK_FIELDS = {
    "record_id", "source_id", "source_sha256", "token_offset", "target_count",
    "input_ids", "target_ids", "attention_mask", "loss_mask",
}


def _sha256(value: object) -> bool:
    return type(value) is str and _HEX.fullmatch(value) is not None


def _linked(info: Any) -> bool:
    return bool(getattr(info, "st_file_attributes", 0)
                & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _check_directory(path: Path) -> None:
    for member in (path, *path.parents):
        try:
            info = member.lstat()
        except FileNotFoundError:
            continue
        s14.need(stat.S_ISDIR(info.st_mode) and not _linked(info), "unsafe exposure directory")


def _read(path: Path) -> bytes:
    info = path.lstat()
    s14.need(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and not _linked(info),
             "unsafe exposure member")
    return _read_destination(path)


def _publish(path: Path, raw: bytes) -> None:
    _check_directory(path.parent)
    if not path.exists() and not path.is_symlink():
        _atomic_write(path.parent, path, raw)
    s14.need(_read(path) == raw, "immutable exposure readback mismatch")


def _validate_block(
    block: dict[str, Any], segment: dict[str, Any], offset: int, previous: int | None,
    block_size: int, vocab_size: int, specials: dict[str, int],
) -> tuple[int, list[int]]:
    s14.need(type(block) is dict and set(block) == _BLOCK_FIELDS
             and all(block[k] == segment[k] for k in ("record_id", "source_id", "source_sha256")),
             "shard block is not bound to the expected source segment")
    count = block["target_count"]
    s14.need(type(count) is int and count == min(block_size, segment["target_count"] - offset)
             and count > 0 and type(block["token_offset"]) is int and block["token_offset"] == offset,
             "shard block skips, repeats or truncates source targets")
    inputs, targets = block["input_ids"], block["target_ids"]
    masks = [1] * count + [0] * (block_size - count)
    s14.need(all(type(vector) is list and len(vector) == block_size
                 and all(type(value) is int for value in vector)
                 for vector in (inputs, targets, block["attention_mask"], block["loss_mask"]))
             and block["attention_mask"] == block["loss_mask"] == masks,
             "shard block masks or token dimensions changed")
    pad, bos, eos = (specials[name] for name in ("pad", "bos", "eos"))
    stop = offset + count == segment["target_count"]
    active = targets[:count]
    forbidden = set(specials.values())
    s14.need(inputs[count:] == targets[count:] == [pad] * (block_size - count)
             and inputs[0] == (bos if offset == 0 else previous)
             and inputs[1:count] == active[:-1]
             and all(0 <= value < vocab_size and value not in forbidden
                     for value in (active[:-1] if stop else active))
             and (not stop or active[-1] == eos),
             "shard block changes causal token shift, padding, or record boundary")
    return count, active


def stage_exposure(
    destination: Path, packing: dict[str, Any], tokenizer_identity: dict[str, Any],
) -> dict[str, Any]:
    """Validate sealed S13 shards and publish byte-exact S14 exposure rows."""
    destination = destination.absolute()
    _check_directory(destination)
    core = {key: value for key, value in packing.items() if key != "manifest_sha256"}
    packing_sha = packing["manifest_sha256"]
    split_sha = packing["cluster_split_sha256"]
    tokenizer_sha = packing["tokenizer_manifest_sha256"]
    s14.need(_sha256(packing_sha) and books.sha(books.canonical(core)) == packing_sha
             and _sha256(split_sha) and _sha256(tokenizer_sha),
             "S13 packing, split or tokenizer lineage changed")
    vocab_size = tokenizer_identity["vocab_size"]
    specials = tokenizer_identity["special_tokens"]
    s14.need(type(vocab_size) is int and vocab_size >= 260 and type(specials) is dict
             and specials == {"pad": 256, "bos": 257, "eos": 258, "unk": 259},
             "unexpected frozen tokenizer vocabulary or special tokens")

    block_size, shards = packing["block_size"], packing["shards"]
    segments = packing["source_segment_byte_map"]
    s14.need(type(block_size) is int and block_size > 0
             and type(shards) is list and bool(shards)
             and type(segments) is list and bool(segments),
             "missing physical shard or source-segment membership")
    segment_ids: set[str] = set()
    origin_by_document: dict[str, tuple[str, str]] = {}
    source_targets: Counter[str] = Counter()
    next_byte = 0
    for segment in segments:
        rid, source, digest = (segment[key] for key in ("record_id", "source_id", "source_sha256"))
        origin, origin_sha = segment["origin_source_id"], segment["origin_member_sha256"]
        end = segment["source_byte_end"]
        count = segment["target_count"]
        s14.need(type(rid) is str and bool(rid) and rid not in segment_ids
                 and _sha256(source) and rid.startswith(source + ":r") and _sha256(digest)
                 and type(origin) is str and bool(origin) and _sha256(origin_sha)
                 and type(segment["source_byte_start"]) is int and segment["source_byte_start"] == next_byte
                 and type(end) is int and end > next_byte and type(count) is int
                 and 2 <= count <= end - next_byte + 1,
                 "invalid or repeated physical source segment")
        s14.need(origin_by_document.setdefault(source, (origin, origin_sha)) == (origin, origin_sha),
                 "source document provenance changed across segments")
        segment_ids.add(rid)
        source_targets[origin] += count
        next_byte = end
    s14.need(sum(source_targets.values()) == packing["target_count"]
             and dict(source_targets) == packing["actual_loss_targets_by_source"]
             and len(origin_by_document) == packing["document_count"]
             and len(segment_ids) == packing["segment_count"]
             and next_byte == packing["source_utf8_bytes"],
             "physical membership target count differs from S13")

    source_folder = destination / "shards"
    output_folder = destination / "exposures"
    _check_directory(source_folder)
    _check_directory(output_folder)
    allowed = {f"shard-{index:06d}.json" for index in range(len(shards))}
    s14.need({member.name for member in source_folder.iterdir()} == allowed,
             "physical source shard inventory drift")
    output_folder.mkdir(exist_ok=True)
    s14.need({member.name for member in output_folder.iterdir()} <= allowed,
             "unexpected exposure publication member")
    chain = s14.ZERO
    target_index = block_index = segment_index = offset = 0
    previous_token: int | None = None
    references: list[dict[str, Any]] = []
    for shard_index, ref in enumerate(shards):
        name = f"shard-{shard_index:06d}.json"
        path = f"shards/{name}"
        s14.need(ref["path"] == path, "noncanonical source shard order")
        raw = _read(source_folder / name)
        s14.need(books.sha(raw) == ref["sha256"] and len(raw) == ref["byte_count"],
                 "physical source shard bytes changed")
        shard = json.loads(raw)
        blocks = shard["blocks"]
        s14.need(type(shard) is dict and set(shard) == {"schema_version", "shard_index", "block_size", "blocks"}
                 and shard["schema_version"] == "12-6.plan2-token-shard.v1"
                 and type(shard["shard_index"]) is int and shard["shard_index"] == shard_index
                 and type(shard["block_size"]) is int and shard["block_size"] == block_size
                 and type(blocks) is list and 0 < len(blocks) == ref["block_count"]
                 and raw == books.canonical(shard), "invalid physical S13 shard")
        rows = []
        for block in blocks:
            s14.need(segment_index < len(segments), "unexpected physical source block")
            segment = segments[segment_index]
            count, active = _validate_block(block, segment, offset, previous_token,
                                            block_size, vocab_size, specials)
            rid, source_sha = segment["record_id"], segment["source_sha256"]
            for position, token in enumerate(active):
                token_offset = offset + position
                target_id = s14.hashed({
                    "contract": s14.CONTRACT, "split_sha256": split_sha,
                    "tokenizer_sha256": tokenizer_sha, "record_id": rid,
                    "source_sha256": source_sha, "token_offset": token_offset,
                    "target_token_id": token,
                })
                exposure_id = s14.hashed({
                    "packing_sha256": packing_sha, "target_id": target_id,
                    "global_index": target_index,
                })
                row = {
                    "index": target_index, "record_id": rid, "token_offset": token_offset,
                    "block_index": block_index, "shard_path": path, "position_in_block": position,
                    "target_token_id": token, "target_id": target_id, "exposure_id": exposure_id,
                }
                chain = s14.hashed({"previous": chain, "exposure": row})
                rows.append({**row, "chain_sha256": chain})
                target_index += 1
            # Unique ordered source segments and offsets establish unique
            # target preimages without retaining one key per training token.
            offset += count
            previous_token = active[-1]
            if offset == segment["target_count"]:
                segment_index += 1
                offset, previous_token = 0, None
            block_index += 1
        s14.need(len(rows) == ref["target_count"], "physical shard target count changed")
        event_raw = books.canonical({
            "schema_version": SHARD_SCHEMA, "source_shard_sha256": ref["sha256"],
            "source_shard_path": path, "ordered_exposures": rows,
        })
        output_path = output_folder / name
        _publish(output_path, event_raw)
        references.append({
            "path": f"exposures/{name}", "source_shard_path": path,
            "sha256": books.sha(event_raw), "byte_count": len(event_raw),
            "exposure_count": len(rows), "chain_tail_sha256": chain,
        })
    s14.need(segment_index == len(segments) and offset == 0
             and target_index == packing["target_count"]
             and block_index == packing["block_count"]
             and {member.name for member in output_folder.iterdir()} == allowed,
             "physical S14 target stream is incomplete")
    result = {
        "schema_version": SCHEMA, "migration_contract": s14.CONTRACT,
        "packing_manifest_sha256": packing_sha, "cluster_split_sha256": split_sha,
        "tokenizer_manifest_sha256": tokenizer_sha,
        "target_count": target_index, "block_count": block_index,
        "source_shard_count": len(shards), "exposure_shards": references,
        "chain_head_sha256": chain, "optimizer_effect_authorized": False,
        "production_release_authorized": False, "paid_compute_used": False,
    }
    manifest = {**result, "manifest_sha256": s14.hashed(result)}
    _publish(destination / MANIFEST, books.canonical(manifest))
    return manifest
