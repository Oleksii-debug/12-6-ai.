"""S15: independent readback of already-staged real S13/S14 candidate artifacts.

No rebuilding from inputs, no final test access, no production authorization.
Checks the exact published physical shard bytes against the entire ordered
target/exposure chain, including inter-shard resume boundaries.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from tools import plan2_deterministic_packing_v1 as packing
from tools import plan2_exposure_ledger_v1 as exposure
from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_physical_packing_candidate_v1 as physical_packing
from tools import plan2_s15_physical_exposure_candidate_v1 as physical_exposure

SCHEMA = "12-6.plan2-s15-physical-readback-v1"
ZERO = "0" * 64


class PhysicalReadbackDenied(ValueError):
    """Published physical token shards and exposures were not verified."""


def need(ok: bool, reason: str) -> None:
    if not ok:
        raise PhysicalReadbackDenied(reason)


def read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    need(path.is_file() and not path.is_symlink(), "missing or linked artifact")
    data = path.read_bytes()
    try:
        doc = json.loads(data.decode("utf-8", "strict"))
    except (UnicodeError, ValueError) as exc:
        raise PhysicalReadbackDenied("invalid published JSON") from exc
    need(type(doc) is dict and data == books.canonical(doc),
         "noncanonical published JSON")
    return doc, data


def check_manifest(directory: Path, name: str, *, expected: str) -> dict[str, Any]:
    need(directory.is_dir() and not directory.is_symlink(),
         "published artifact tree missing or linked")
    manifest, data = read_json(directory / name)
    digest = manifest.get("manifest_sha256")
    core = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    need(digest == expected and digest == books.sha(books.canonical(core))
         and books.sha(data) == books.sha(books.canonical(manifest))
         and manifest.get("training_corpus_authorized") is False
         and manifest.get("production_release_authorized") is False
         and manifest.get("terminal_done") is False,
         "physical manifest corrupted or release flag escalated")
    for cap in ("physical_s9_admitted", "tokenizer_fit_authorized",
                "optimizer_effect_authorized", "evaluation_release_authorized",
                "evaluation_authorized", "generated_auto_reentry_authorized",
                "paid_compute_used"):
        if cap in manifest:
            need(manifest[cap] is False,
                 "published candidate escalated authority: " + cap)
    return manifest


def verify(packed_dir: Path, exposure_dir: Path, *,
           expected_packed_sha256: str,
           expected_exposure_sha256: str) -> dict[str, Any]:
    packed = check_manifest(
        packed_dir, physical_packing.OUTPUT,
        expected=expected_packed_sha256,
    )
    ledger = check_manifest(
        exposure_dir, physical_exposure.OUTPUT,
        expected=expected_exposure_sha256,
    )
    need(packed.get("schema_version") == physical_packing.SCHEMA
         and packed.get("decision") ==
             "REAL_PHYSICAL_PACKING_CANDIDATE_NOT_RELEASE"
         and ledger.get("schema_version") == physical_exposure.SCHEMA
         and ledger.get("decision") ==
             "PHYSICAL_ORDERED_TARGETS_CANDIDATE_NOT_RELEASE",
         "wrong physical candidate schema or decision")
    need(packed["manifest_sha256"] == ledger["packing_manifest_sha256"]
         and packed["cluster_split_sha256"] == ledger["cluster_split_sha256"]
         and packed["tokenizer_manifest_sha256"] == ledger["tokenizer_manifest_sha256"]
         and packed["target_count"] == ledger["target_count"]
         and packed["block_count"] == ledger["block_count"]
         and len(packed["shards"]) == len(ledger["exposure_shards"]),
         "S13-to-S14 physical parent identity changed")
    expected_packed_files = {
        physical_packing.OUTPUT, *(
            x["path"] for x in packed["shards"]
        )
    }
    expected_exposure_files = {
        physical_exposure.OUTPUT, *(
            x["path"] for x in ledger["exposure_shards"]
        )
    }
    for folder, expected_members in (
        (packed_dir, expected_packed_files),
        (exposure_dir, expected_exposure_files),
    ):
        observed: set[str] = set()
        for member in folder.rglob("*"):
            need(not member.is_symlink() and (member.is_file() or member.is_dir()),
                 "unsafe physical publication member")
            if member.is_file():
                relative = member.relative_to(folder).as_posix()
                need(relative not in observed, "duplicate publication member")
                observed.add(relative)
        need(observed == expected_members, "missing or unexpected physical artifact")
    member_ids = packed["train_record_ids"]
    segment_map = packed["source_segment_byte_map"]
    need(type(member_ids) is list and bool(member_ids)
         and len(member_ids) == len(set(member_ids))
         and type(segment_map) is list
         and len(segment_map) == len(member_ids)
         and [row["record_id"] for row in segment_map] == member_ids,
         "missing or inconsistent physical train segment identities")
    next_byte = 0
    segment_digests: dict[str, str] = {}
    for segment in segment_map:
        need(segment["source_byte_start"] == next_byte
             and type(segment["source_byte_end"]) is int
             and segment["source_byte_end"] > next_byte
             and type(segment["source_sha256"]) is str
             and re.fullmatch(r"[0-9a-f]{64}", segment["source_sha256"])
                 is not None
             and segment["record_id"].startswith(segment["source_id"] + ":r"),
             "physical book-to-token byte lineage changed")
        next_byte = segment["source_byte_end"]
        segment_digests[segment["record_id"]] = segment["source_sha256"]
    need(next_byte == packed["train_document_bytes"],
         "training source coverage has gap or truncation")
    offsets: dict[str, int] = {}
    targets: set[str] = set()
    exposures: set[str] = set()
    count = 0
    chain = ZERO
    blocks_read = 0
    mapping = packed["source_token_shard_mapping"]
    shards = sorted(packed["shards"], key=lambda x: x["path"])
    ledger_shards = sorted(
        ledger["exposure_shards"], key=lambda x: x["source_shard_path"]
    )
    need([x["path"] for x in shards]
         == [x["source_shard_path"] for x in ledger_shards],
         "physical exposure shard parent order mismatch")
    for source, evidence in zip(shards, ledger_shards, strict=True):
        p, raw = read_json(packed_dir / source["path"])
        e, event_bytes = read_json(exposure_dir / evidence["path"])
        need(books.sha(raw) == source["sha256"]
             and books.sha(event_bytes) == evidence["sha256"]
             and len(raw) == source["byte_count"]
             and len(event_bytes) == evidence["byte_count"]
             and p.get("schema_version") == "12-6.plan2-token-shard.v1"
             and p["block_size"] == packed["block_size"]
             and e.get("schema_version")
                 == "12-6.plan2-s15-physical-exposure-shard.v1"
             and e.get("source_shard_sha256") == source["sha256"]
             and e.get("source_shard_path") == source["path"]
             and evidence["source_shard_path"] == source["path"]
             and evidence["chain_tail_sha256"] is not None,
             "physical shard SHA, byte count or schema mismatch")
        blocks = p["blocks"]
        events = e["ordered_exposures"]
        need(type(blocks) is list and type(events) is list
             and len(blocks) == source["block_count"]
             and len(events) == source["target_count"]
             == evidence["exposure_count"],
             "physical shard target count mismatch")
        pointer = 0
        for ordinal, block in enumerate(blocks):
            need(blocks_read < len(mapping), "extra physical block")
            link = mapping[blocks_read]
            need(link["shard_path"] == source["path"]
                 and link["block_ordinal"] == ordinal
                 and all(link[k] == block[k] for k in (
                     "record_id", "source_id", "source_sha256",
                     "token_offset", "target_count",
                 )), "published source/block lineage mismatch")
            rid = block["record_id"]
            need(rid in segment_digests
                 and block["source_sha256"] == segment_digests[rid],
                 "unapproved physical training record in shard")
            n = block["target_count"]
            start = block["token_offset"]
            need(type(n) is int and 0 < n <= packing.BLOCK
                 and start == offsets.get(rid, 0)
                 and len(block["target_ids"]) == packing.BLOCK
                 and len(block["loss_mask"]) == packing.BLOCK
                 and block["loss_mask"] == block["attention_mask"]
                 and block["loss_mask"] == [1] * n +
                 [0] * (packing.BLOCK - n),
                 "loss token offsets, boundaries or masks invalid")
            offsets[rid] = start + n
            for position in range(n):
                need(pointer < len(events), "missing physical exposure")
                row = events[pointer]
                pointer += 1
                token = block["target_ids"][position]
                basis = {
                    "contract": exposure.CONTRACT,
                    "split_sha256": packed["cluster_split_sha256"],
                    "tokenizer_sha256": packed["tokenizer_manifest_sha256"],
                    "record_id": rid, "source_sha256": block["source_sha256"],
                    "token_offset": start + position,
                    "target_token_id": token,
                }
                target_id = exposure.hashed(basis)
                exp_id = exposure.hashed({
                    "packing_sha256": packed["manifest_sha256"],
                    "target_id": target_id,
                    "global_index": count,
                })
                need(row["index"] == count and row["record_id"] == rid
                     and row["token_offset"] == start + position
                     and row["block_index"] == blocks_read
                     and row["shard_path"] == source["path"]
                     and row["position_in_block"] == position
                     and row["target_token_id"] == token
                     and row["target_id"] == target_id
                     and row["exposure_id"] == exp_id,
                     "physical S14 target/exposure replay mismatch")
                need(target_id not in targets and exp_id not in exposures,
                     "duplicate physical target or exposure")
                targets.add(target_id)
                exposures.add(exp_id)
                body = {k: v for k, v in row.items() if k != "chain_sha256"}
                chain = exposure.hashed({"previous": chain, "exposure": body})
                need(row["chain_sha256"] == chain,
                     "physical exposure chain broke")
                count += 1
            blocks_read += 1
        need(pointer == len(events)
             and chain == evidence["chain_tail_sha256"],
             "unaccounted physical exposure or shard-end chain drift")
    need(set(offsets) == set(member_ids),
         "missing or repeated physical training source segment")
    need(count == packed["target_count"] == ledger["target_count"]
         and blocks_read == packed["block_count"] == len(mapping)
         and chain == ledger["chain_head_sha256"],
         "physical complete stream or resume chain boundary mismatch")
    core = {
        "schema_version": SCHEMA,
        "decision": "REAL_PHYSICAL_S13_S14_READBACK_VERIFIED_NOT_RELEASE",
        "packing_manifest_sha256": packed["manifest_sha256"],
        "exposure_manifest_sha256": ledger["manifest_sha256"],
        "readback_target_count": count,
        "readback_block_count": blocks_read,
        "shard_count": len(shards),
        "last_chain_sha256": chain,
        "terminal_done": False,
        "production_release_authorized": False,
    }
    return {**core, "readback_sha256": books.sha(books.canonical(core))}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical-shards", type=Path, required=True)
    parser.add_argument("--physical-exposures", type=Path, required=True)
    parser.add_argument("--packing-sha256", required=True)
    parser.add_argument("--exposure-sha256", required=True)
    args = parser.parse_args()
    report = verify(
        args.physical_shards, args.physical_exposures,
        expected_packed_sha256=args.packing_sha256,
        expected_exposure_sha256=args.exposure_sha256,
    )
    print(json.dumps({
        "decision": report["decision"],
        "readback_sha256": report["readback_sha256"],
        "terminal_done": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
