"""S15 physical D03 normalized candidate materialization with immutable readback.

Reuses pinned source blobs, historical normalizers and incumbent G06/DATA232.
No production tokenizer/training/final-test/Plan9 authority is created.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools import plan2_public_domain_books_v1 as books
from tools import plan2_s15_d03_physical_sources_v1 as d03
from tools.plan2_physical_materialization_v1 import _atomic_write, _read_destination

SCHEMA = "12-6.plan2-s15-d03-normalized-physical-candidate.v1"
OUTPUT = "d03-normalized-physical-candidate.json"


class D03MaterializationDenied(ValueError):
    """Missing/unsafe normalized physical source or unauthorized admission."""


def need(value: bool, reason: str) -> None:
    if not value:
        raise D03MaterializationDenied(reason)


def build(root: Path) -> tuple[dict[str, Any], dict[str, bytes]]:
    root = root.resolve(strict=True)
    proof = d03.inspect(root)
    need(proof["physical_family_count"] == 2
         and proof["physical_record_count"] == 12
         and proof["training_corpus_authorized"] is False
         and proof["production_release_authorized"] is False,
         "physical D03 source authority changed")
    config = json.loads(books.read_checked(root, d03.CONFIG))
    members = {item["source_path"]: item for item in proof["records"]}
    need(len(config["objects"]) == len(members) == 12,
         "physical normalized source coverage incomplete")
    files: dict[str, bytes] = {}
    receipts: list[dict[str, Any]] = []
    for row in config["objects"]:
        path = row["physical_path"]
        meta = members.get(path)
        need(type(meta) is dict, "unverified D03 normalized source member")
        raw = books.read_checked(root, path)
        payload = (d03.normalize_php(raw) if row["family"] == "php"
                   else d03.normalize_rust(raw))
        need(books.sha(payload) == meta["normalized_sha256"]
             and len(payload) == meta["normalized_bytes"],
             "D03 normalized payload drifted after G06/DATA232 qualification")
        marker = d03.PREFIX + "raw/" + row["family"] + "/"
        need(path.startswith(marker), "D03 path escaped pinned source scope")
        target = "normalized/" + row["family"] + "/" + path[len(marker):] + ".txt"
        need(target not in files, "normalized output path collision")
        files[target] = payload
        receipts.append({
            "path": target,
            "source_path": path,
            "record_id": meta["record_id"],
            "source_family": meta["source_family"],
            "normalized_sha256": books.sha(payload),
            "normalized_bytes": len(payload),
        })
    need(len(files) == 12 and sum(len(x) for x in files.values()) == 48675,
         "physical D03 data coverage differs from signed source evidence")
    core = {
        "schema_version": SCHEMA,
        "decision": "D03_RECONSTRUCTED_PHYSICAL_NORMALIZED_CANDIDATE_NOT_RELEASE",
        "d03_source_manifest_sha256": proof["manifest_sha256"],
        "normalized_member_count": len(receipts),
        "normalized_total_bytes": sum(len(x) for x in files.values()),
        "normalized_members": sorted(receipts, key=lambda x: x["path"]),
        "g06_input_root_sha256": proof["g06_input_root_sha256"],
        "data232_report_sha256": proof["data232_report_sha256"],
        "physical_s3_s9_admitted": False,
        "training_corpus_authorized": False,
        "tokenizer_fit_authorized": False,
        "production_release_authorized": False,
        "terminal_done": False,
    }
    return {**core, "manifest_sha256": books.sha(books.canonical(core))}, files


def stage(root: Path, destination: Path) -> dict[str, Any]:
    need(not any(x.is_symlink() for x in (destination, *destination.parents)),
         "symlink normalized destination")
    report, files = build(root)
    destination.mkdir(parents=True, exist_ok=True)
    expected = set(files) | {OUTPUT}
    for member in destination.rglob("*"):
        need(not member.is_symlink() and (member.is_file() or member.is_dir()),
             "unsafe D03 normalized publication member")
        if member.is_file():
            need(member.relative_to(destination).as_posix() in expected,
                 "unexpected existing D03 normalized member")
    for name, raw in sorted(files.items()):
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        need(not target.is_symlink() and not target.parent.is_symlink(),
             "symlink normalized payload")
        if target.exists():
            need(target.is_file() and _read_destination(target) == raw,
                 "immutable D03 normalized source changed")
        else:
            _atomic_write(destination, target, raw)
        need(_read_destination(target) == raw,
             "physical normalized source readback mismatch")
    target = destination / OUTPUT
    raw = books.canonical(report)
    if target.exists() or target.is_symlink():
        need(target.is_file() and not target.is_symlink()
             and _read_destination(target) == raw,
             "immutable normalized corpus manifest changed")
    else:
        _atomic_write(destination, target, raw)
    need(_read_destination(target) == raw,
         "normalized corpus manifest readback mismatch")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    receipt = stage(args.root, args.out_dir)
    print(json.dumps({
        "decision": receipt["decision"],
        "manifest_sha256": receipt["manifest_sha256"],
        "terminal_done": False,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
