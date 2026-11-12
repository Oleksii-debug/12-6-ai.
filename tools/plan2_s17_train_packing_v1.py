"""Pack admitted Plan 2 training rows with the existing frozen tokenizer."""
from __future__ import annotations

import re
import stat
from collections import Counter
from pathlib import Path
from typing import Any

from tools import plan2_deterministic_packing_v1 as packing
from tools import plan2_s15_physical_tokenizer_candidate_v1 as segmenter
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination
from tools.plan2_s15_physical_packing_candidate_v1 import BLOCKS_PER_SHARD
from tools.plan2_s15_real_32k_bpe_v1 import ByteBPE32k

SCHEMA = "12-6.plan2-s17-train-packing.v1"
SHARD_SCHEMA = "12-6.plan2-token-shard.v1"
_SHARD_NAME = re.compile(r"shard-[0-9]{6}\.json")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _plain_path(path: Path, *, directory: bool) -> bool:
    try:
        status = path.lstat()
    except FileNotFoundError:
        return False
    expected = stat.S_ISDIR(status.st_mode) if directory else stat.S_ISREG(status.st_mode)
    reparse = getattr(status, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
    packing.need(expected and not reparse, "unsafe training shard publication path")
    return True


def _write_shard(directory: Path, index: int, blocks: list[dict[str, Any]]) -> dict[str, Any]:
    name = f"shard-{index:06d}.json"
    packing.need(_SHARD_NAME.fullmatch(name) is not None, "training shard index exceeds its filename width")
    raw = packing.canonical({
        "schema_version": SHARD_SCHEMA,
        "shard_index": index,
        "block_size": packing.BLOCK,
        "blocks": blocks,
    })
    target = directory / name
    if not _plain_path(target, directory=False):
        _atomic_write(directory, target, raw)
    packing.need(_read_destination(target) == raw, "immutable training shard differs from the verified rebuild")
    return {
        "path": f"shards/{name}",
        "sha256": packing.digest(raw),
        "byte_count": len(raw),
        "block_count": len(blocks),
        "target_count": sum(block["target_count"] for block in blocks),
    }


def stage_train_shards(
    destination: Path, tokenizer: ByteBPE32k, rows: list[dict],
) -> dict[str, Any]:
    """Stream bounded S13 shards from verified S16 training rows.

    Return the packing data for the caller's lineage manifest. Each shard
    can be reused after interruption only when its bytes match a fresh pack.
    """
    ids = [row["record_id_sha256"] for row in rows]
    packing.need(bool(ids) and all(type(value) is str and _SHA256.fullmatch(value) for value in ids)
                 and len(ids) == len(set(ids)), "invalid or repeated admitted training record identity")

    directory = destination / "shards"
    for path in reversed((directory, *directory.parents)):
        _plain_path(path, directory=True)
    directory.mkdir(parents=True, exist_ok=True)
    existing = set()
    for path in directory.iterdir():
        packing.need(_SHARD_NAME.fullmatch(path.name) is not None and _plain_path(path, directory=False),
                     "unexpected training shard member")
        existing.add(f"shards/{path.name}")

    shard_receipts: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    source_targets: Counter[str] = Counter()
    family_targets: Counter[str] = Counter()
    stratum_targets: Counter[str] = Counter()
    segment_map: list[dict[str, Any]] = []
    document_bytes = source_offset = 0

    def flush() -> None:
        receipt = _write_shard(directory, len(shard_receipts), pending)
        shard_receipts.append(receipt)
        existing.discard(receipt["path"])
        pending.clear()

    for row in sorted(rows, key=lambda item: item["record_id_sha256"]):
        source = row["record_id_sha256"]
        size = len(row["text"].encode("utf-8", "strict"))
        packing.need(size == row["utf8_bytes"], "admitted training record size drift")
        document_bytes += size
        count = 0
        for segment in segmenter.segments(source, row["text"]):
            blocks = packing.pack(tokenizer, [{**segment, "source_id": source}])
            segment_bytes = len(segment["text"].encode("utf-8"))
            segment_targets = sum(block["target_count"] for block in blocks)
            segment_map.append({
                "record_id": segment["record_id"],
                "source_id": source,
                "source_sha256": blocks[0]["source_sha256"],
                "origin_source_id": row["source_id"],
                "origin_member_sha256": row["content_sha256"],
                "source_byte_start": source_offset,
                "source_byte_end": source_offset + segment_bytes,
                "target_count": segment_targets,
            })
            source_offset += segment_bytes
            count += segment_targets
            for block in blocks:
                pending.append(block)
                if len(pending) == BLOCKS_PER_SHARD:
                    flush()
        source_targets[row["source_id"]] += count
        family_targets[row["source_family"]] += count
        stratum_targets[row["stratum"]] += count
    if pending:
        flush()

    packing.need(not existing, "unexpected training shard from a different corpus or tokenizer")
    target_count = sum(shard["target_count"] for shard in shard_receipts)
    packing.need(source_offset == document_bytes and target_count > 0
                 and target_count == sum(source_targets.values()) == sum(family_targets.values())
                 == sum(stratum_targets.values()), "training shard byte or target totals drift")
    return {
        "schema_version": SCHEMA,
        "split": "train",
        "packing_policy": "physical_no_cross_record_no_truncation_v1",
        "block_size": packing.BLOCK,
        "blocks_per_shard": BLOCKS_PER_SHARD,
        "block_count": sum(shard["block_count"] for shard in shard_receipts),
        "target_count": target_count,
        "document_count": len(rows),
        "segment_count": len(segment_map),
        "source_utf8_bytes": document_bytes,
        "source_segment_byte_map": segment_map,
        "actual_loss_targets_by_source": dict(sorted(source_targets.items())),
        "actual_loss_targets_by_source_family": dict(sorted(family_targets.items())),
        "actual_loss_targets_by_stratum": dict(sorted(stratum_targets.items())),
        "shards": shard_receipts,
    }
