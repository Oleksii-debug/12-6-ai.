"""Read and resume the exact local Plan 9 cyber pilot prefix without a model."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from tools.plan2_s17_exposure_stream_v1 import _validate_block
from tools.plan2_s17_frozen_tokenizer_v1 import load_frozen

BUNDLE_RECEIPT_SHA256 = "68fac420223b639efbf5d8dc203367322e23bd01b24d1d82bc61943ddc67df41"
PACKING_MANIFEST_SHA256 = "4c47e88d6dad443db5ffeccdbafb50be943bb11d7e9c4271933c4a10b6e3796e"
RIGHTS_RECEIPT_SHA256 = "55065f081aa55d6cde11fa27959d6244b390253de59e455cdbe017666d932b79"
TOKENIZER_IDENTITY_SHA256 = "f6beab1910dfd9173b73783ae97a599b4c2bb7122ef95b824d1d20c5b6c2d07f"
PREFIX_SHA256 = "fbe0ecd0e271112eec0c4e6003813ff2dee4e9611527d9483e1c6f072efeed9d"
PREFIX_TARGETS = 2696
CURSOR_SCHEMA = "12-6.plan9-cyber-pilot-stream-cursor.v1"
STREAM_SCHEMA = "12-6.plan9-cyber-pilot-stream.v1"
MICROBATCH = 2
ACCUMULATION = 12
UPDATES = 4
BLOCK_WIDTH = 32
PREFIX_BLOCKS = MICROBATCH * ACCUMULATION * UPDATES
FIELDS = ("input_ids", "target_ids", "loss_mask")
_HEX = re.compile(r"[0-9a-f]{64}\Z")


def _canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _pinned_json(path: Path, expected: str, maximum: int) -> dict:
    reparse = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    for parent in path.parents:
        status = parent.lstat()
        _require(stat.S_ISDIR(status.st_mode) and not getattr(status, "st_file_attributes", 0) & reparse,
                 "linked or invalid pilot artifact directory")
    status = path.lstat()
    _require(stat.S_ISREG(status.st_mode) and status.st_nlink == 1
             and not getattr(status, "st_file_attributes", 0) & reparse and 0 < status.st_size <= maximum,
             "linked, invalid or oversized pilot artifact")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    _require(len(raw) == status.st_size and _sha(raw) == expected, "pilot artifact digest differs")
    value = json.loads(raw)
    _require(type(value) is dict and raw == _canonical(value), "pilot artifact encoding differs")
    return value


@dataclass(frozen=True)
class PilotStream:
    """A verified bounded prefix with portable, recipe-bound optimizer cursors."""

    recipe_sha256: str
    stream_identity_sha256: str
    _blocks: tuple[tuple[tuple[int, ...], ...], ...]
    _prefixes: tuple[tuple[int, str], ...]

    @property
    def block_count(self) -> int:
        return len(self._blocks)

    @property
    def batch_count(self) -> int:
        return self.block_count // MICROBATCH

    @property
    def prefix_total_targets(self) -> int:
        return self._prefixes[-1][0]

    @property
    def prefix_sha256(self) -> str:
        return self._prefixes[-1][1]

    def cursor(self, next_microbatch_index: int) -> dict:
        _require(type(next_microbatch_index) is int and 0 <= next_microbatch_index <= self.batch_count
                 and next_microbatch_index % ACCUMULATION == 0,
                 "pilot cursor must be at an optimizer boundary inside the window")
        blocks = next_microbatch_index * MICROBATCH
        targets, digest = self._prefixes[blocks]
        core = {"schema": CURSOR_SCHEMA, "stream_identity_sha256": self.stream_identity_sha256,
                "bundle_receipt_sha256": BUNDLE_RECEIPT_SHA256, "packing_manifest_sha256": PACKING_MANIFEST_SHA256,
                "tokenizer_identity_sha256": TOKENIZER_IDENTITY_SHA256, "recipe_sha256": self.recipe_sha256,
                "next_microbatch_index": next_microbatch_index,
                "optimizer_step": next_microbatch_index // ACCUMULATION, "consumed_blocks": blocks,
                "consumed_loss_positions": targets, "consumed_prefix_sha256": digest,
                "complete": next_microbatch_index == self.batch_count}
        return {**core, "cursor_sha256": _sha(_canonical(core))}

    def verify_cursor(self, cursor: dict, expected_cursor_sha256: str) -> None:
        _require(type(cursor) is dict and type(expected_cursor_sha256) is str
                 and _HEX.fullmatch(expected_cursor_sha256) is not None,
                 "expected an exact pilot cursor and trusted digest")
        verified = self.cursor(cursor.get("next_microbatch_index"))
        _require(verified["cursor_sha256"] == expected_cursor_sha256 and _canonical(verified) == _canonical(cursor),
                 "pilot cursor differs from the sealed physical prefix")

    def iter_batches(self, cursor: dict | None = None,
                     expected_cursor_sha256: str | None = None) -> Iterator[dict[str, list[list[int]]]]:
        """Yield fresh aligned integer rows, with no tensor, model, or optimizer allocation."""
        if cursor is None:
            _require(expected_cursor_sha256 is None, "a cursor digest requires a cursor")
            start = 0
        else:
            self.verify_cursor(cursor, expected_cursor_sha256)
            start = cursor["consumed_blocks"]
        for index in range(start, self.block_count, MICROBATCH):
            blocks = self._blocks[index:index + MICROBATCH]
            yield {name: [list(block[field]) for block in blocks] for field, name in enumerate(FIELDS)}


def open_pilot_stream(root: Path, bundle: Path, *, recipe_sha256: str, optimizer_updates: int = UPDATES,
                      microbatch_size: int = MICROBATCH,
                      gradient_accumulation_steps: int = ACCUMULATION) -> PilotStream:
    """Verify exact candidate metadata, frozen tokenizer, and the only allowed pilot window."""
    geometry = (optimizer_updates, microbatch_size, gradient_accumulation_steps)
    _require(all(type(value) is int for value in geometry) and geometry == (UPDATES, MICROBATCH, ACCUMULATION),
             "pilot window differs from 4 updates, microbatch 2, and accumulation 12")
    _require(type(recipe_sha256) is str and _HEX.fullmatch(recipe_sha256) is not None,
             "pilot stream requires a recipe SHA-256")
    bundle = bundle.absolute()
    receipt = _pinned_json(bundle / "receipt.json", BUNDLE_RECEIPT_SHA256, 16_384)
    rights = _pinned_json(bundle / "rights/receipt.json", RIGHTS_RECEIPT_SHA256, 16_384)
    _require(receipt["native_packing_manifest_sha256"] == PACKING_MANIFEST_SHA256
             and rights["bundle_receipt_sha256"] == BUNDLE_RECEIPT_SHA256
             and receipt["expanded_corpus_admitted"] is False and receipt["campaign_authorized"] is False,
             "pilot data or rights identity differs")
    packing = _pinned_json(bundle / "packing-manifest.json", PACKING_MANIFEST_SHA256, 40 * 1024 * 1024)
    _require(packing["split"] == "train" and packing["block_size"] == BLOCK_WIDTH and packing["blocks_per_shard"] == 64,
             "pilot requires the sealed 32-wide train packing")
    _, tokenizer = load_frozen(root)
    identity = tokenizer["tokenizer_identity"]
    _require(tokenizer == receipt["tokenizer"] and _sha(_canonical(identity)) == TOKENIZER_IDENTITY_SHA256,
             "pilot tokenizer identity differs")
    segments = iter(packing["source_segment_byte_map"])
    segment = next(segments)
    offset, previous, total = 0, None, 0
    digest = hashlib.sha256()
    prefixes = [(0, digest.hexdigest())]
    blocks = []
    for shard_index, ref in enumerate(packing["shards"][:2]):
        name = f"shards/shard-{shard_index:06d}.json"
        _require(ref["path"] == name, "pilot shard is out of order")
        shard = _pinned_json(bundle / name, ref["sha256"], 100_000)
        _require(shard["shard_index"] == shard_index and shard["block_size"] == BLOCK_WIDTH
                 and shard["schema_version"] == "12-6.plan2-token-shard.v1"
                 and len(shard["blocks"]) == ref["block_count"] == 64,
                 "pilot shard geometry differs")
        for block in shard["blocks"][:PREFIX_BLOCKS - len(blocks)]:
            count, active = _validate_block(block, segment, offset, previous, BLOCK_WIDTH,
                                            identity["vocab_size"], identity["special_tokens"])
            offset += count
            previous = active[-1]
            if offset == segment["target_count"]:
                segment = next(segments)
                offset, previous = 0, None
            total += count
            digest.update(_canonical(block))
            prefixes.append((total, digest.hexdigest()))
            blocks.append(tuple(tuple(block[field]) for field in FIELDS))
    _require(len(blocks) == PREFIX_BLOCKS and total == PREFIX_TARGETS and digest.hexdigest() == PREFIX_SHA256,
             "pilot prefix differs from its expected physical identity")
    stream = {"schema": STREAM_SCHEMA, "bundle_receipt_sha256": BUNDLE_RECEIPT_SHA256,
              "packing_manifest_sha256": PACKING_MANIFEST_SHA256, "tokenizer_identity_sha256": TOKENIZER_IDENTITY_SHA256,
              "recipe_sha256": recipe_sha256, "prefix_sha256": PREFIX_SHA256, "prefix_total_targets": total,
              "block_width": BLOCK_WIDTH, "microbatch_size": MICROBATCH, "optimizer_updates": UPDATES,
              "gradient_accumulation_steps": ACCUMULATION}
    return PilotStream(recipe_sha256, _sha(_canonical(stream)), tuple(blocks), tuple(prefixes))
