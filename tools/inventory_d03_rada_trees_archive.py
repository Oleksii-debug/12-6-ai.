#!/usr/bin/env python3
"""Fail-closed Rada_Trees 7z archive inventory materializer.

This layer proves transport identity and member inventory only. It never grants
training capacity, corpus admission, tokenizer-fit authority, or model-training
permission.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/data/d03_rada_trees_archive_inventory_v1.json"
HEX64 = re.compile(r"[0-9a-f]{64}")
DRIVE = re.compile(r"^[A-Za-z]:")


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_member_path(raw: str) -> str:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError("member path must be a non-empty NUL-free string")
    value = unicodedata.normalize("NFKC", raw.replace("\\", "/"))
    if value.startswith("/") or DRIVE.match(value):
        raise ValueError(f"absolute/archive-drive path forbidden: {raw!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe member path: {raw!r}")
    return PurePosixPath(*parts).as_posix()


def _parse_size(value: str, path: str) -> int:
    try:
        size = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid Size for {path!r}") from exc
    if size < 0:
        raise ValueError(f"negative Size for {path!r}")
    return size


def parse_7z_slt_listing(text: str) -> list[dict[str, Any]]:
    """Parse file entries from `7z l -slt -ba` output."""
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in text.splitlines() + [""]:
        if not line.strip():
            if current:
                records.append(current)
                current = {}
            continue
        if " = " in line:
            key, value = line.split(" = ", 1)
            current[key.strip()] = value

    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        raw_path = record.get("Path")
        if raw_path is None:
            continue
        is_folder = record.get("Folder") == "+" or record.get("Attributes", "").startswith("D")
        if is_folder:
            continue
        if "Size" not in record:
            raise ValueError(f"archive member missing Size: {raw_path!r}")
        path = canonical_member_path(raw_path)
        if path in seen:
            raise ValueError(f"duplicate normalized member path: {path!r}")
        seen.add(path)
        result.append({"path": path, "size_bytes": _parse_size(record["Size"], path)})
    result.sort(key=lambda item: item["path"])
    if not result:
        raise ValueError("archive listing contains no regular files")
    return result


def validate_bounds(entries: Iterable[dict[str, Any]], max_member: int, max_total: int) -> int:
    total = 0
    for entry in entries:
        size = int(entry["size_bytes"])
        if size > max_member:
            raise ValueError(f"member exceeds max_single_member_bytes: {entry['path']}")
        total += size
        if total > max_total:
            raise ValueError("archive exceeds max_total_uncompressed_bytes")
    return total


def inventory_extracted_tree(root: Path, expected: list[dict[str, Any]], max_member: int, max_total: int) -> list[dict[str, Any]]:
    root = root.resolve()
    expected_map = {item["path"]: int(item["size_bytes"]) for item in expected}
    actual: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0

    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        canonical = canonical_member_path(relative)
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"symlink forbidden in archive extraction: {canonical}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"special file forbidden in archive extraction: {canonical}")
        if canonical in seen:
            raise ValueError(f"duplicate normalized extracted path: {canonical}")
        seen.add(canonical)
        size = path.stat().st_size
        if size > max_member:
            raise ValueError(f"member exceeds max_single_member_bytes: {canonical}")
        total += size
        if total > max_total:
            raise ValueError("extracted tree exceeds max_total_uncompressed_bytes")
        actual.append({"path": canonical, "size_bytes": size, "sha256": sha256_file(path)})

    actual_map = {item["path"]: item["size_bytes"] for item in actual}
    if actual_map != expected_map:
        missing = sorted(set(expected_map) - set(actual_map))
        extra = sorted(set(actual_map) - set(expected_map))
        mismatched = sorted(path for path in set(expected_map) & set(actual_map) if expected_map[path] != actual_map[path])
        raise ValueError(f"listing/extraction mismatch missing={missing} extra={extra} size_mismatch={mismatched}")
    if not actual:
        raise ValueError("extracted tree contains no regular files")
    return actual


def find_extractor(preferred: str | None, accepted: list[str]) -> str:
    candidates = [preferred] if preferred else accepted
    for candidate in candidates:
        if candidate and shutil.which(candidate):
            return candidate
    raise RuntimeError("7z extractor unavailable; install one of: " + ", ".join(accepted))


def extractor_version(executable: str) -> str:
    proc = subprocess.run([executable], check=False, capture_output=True, text=True, timeout=30)
    text = (proc.stdout or proc.stderr).strip().splitlines()
    if not text:
        return "UNKNOWN"
    return text[0].strip()[:200]


def list_archive(executable: str, archive: Path) -> list[dict[str, Any]]:
    proc = subprocess.run(
        [executable, "l", "-slt", "-ba", "--", str(archive)],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"7z listing failed rc={proc.returncode}: {proc.stderr[-1000:]}")
    return parse_7z_slt_listing(proc.stdout)


def extract_archive(executable: str, archive: Path, destination: Path) -> None:
    proc = subprocess.run(
        [executable, "x", "-y", "-bd", f"-o{destination}", "--", str(archive)],
        check=False,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"7z extraction failed rc={proc.returncode}: {proc.stderr[-1000:]}")


def load_config(path: Path = CONFIG) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert value["schema_version"] == "12-6.d03-rada-trees-archive-inventory.v1"
    assert value["execution_profile"] == "LOCAL_FREE"
    assert value["parent"]["probe_pr"] == 638
    assert value["parent"]["probe_head_sha"] == "92c1fd05d4399b0f0c4a35f0689160383f963c9c"
    assert value["parent"]["dataset"] == "uacorpus/Rada_Trees"
    assert value["parent"]["dataset_head_sha"] == "1b994a5804dcda122721e8d33a03fd172cf8d867"
    assert value["primary_archive"]["path"] == "Rada_Trees.7z"
    assert value["primary_archive"]["exact_content_sha256"] is None
    policy = value["inventory_policy"]
    assert policy["hash_algorithm"] == "sha256"
    assert policy["reject_symlinks"] is True
    assert policy["reject_special_files"] is True
    assert policy["reject_duplicate_normalized_paths"] is True
    assert policy["require_preextract_member_listing"] is True
    assert policy["require_listing_vs_extraction_exact_match"] is True
    boundary = value["claim_boundary"]
    assert boundary["training_authorized_bytes"] == 0
    assert boundary["normalized_capacity_credited"] == 0
    assert boundary["training_exposure_authorized"] is False
    assert boundary["model_training_executed"] is False
    assert boundary["optimizer_updates"] == 0
    assert boundary["paid_compute_used"] is False
    return value


def build_report(
    config: dict[str, Any],
    archive: Path,
    expected_sha256: str,
    expected_size: int,
    upstream_object_identity: str,
    extractor: str,
) -> dict[str, Any]:
    if not HEX64.fullmatch(expected_sha256):
        raise ValueError("expected archive SHA-256 must be 64 lowercase hex chars")
    if expected_size <= 0:
        raise ValueError("expected archive size must be positive")
    if not upstream_object_identity.strip():
        raise ValueError("upstream object identity is required")
    archive = archive.resolve()
    if not archive.is_file() or archive.is_symlink():
        raise ValueError("archive must be a regular non-symlink file")
    actual_size = archive.stat().st_size
    actual_sha256 = sha256_file(archive)
    if actual_size != expected_size:
        raise ValueError(f"archive size mismatch expected={expected_size} actual={actual_size}")
    if actual_sha256 != expected_sha256:
        raise ValueError("archive SHA-256 mismatch")

    policy = config["inventory_policy"]
    listing = list_archive(extractor, archive)
    total = validate_bounds(listing, policy["max_single_member_bytes"], policy["max_total_uncompressed_bytes"])
    with tempfile.TemporaryDirectory(prefix="rada-trees-inventory-") as tmp:
        temp_root = Path(tmp)
        extract_archive(extractor, archive, temp_root)
        members = inventory_extracted_tree(
            temp_root,
            listing,
            policy["max_single_member_bytes"],
            policy["max_total_uncompressed_bytes"],
        )

    inventory_payload = {
        "dataset_head_sha": config["parent"]["dataset_head_sha"],
        "archive_path": config["primary_archive"]["path"],
        "upstream_object_identity": upstream_object_identity,
        "archive_sha256": actual_sha256,
        "archive_size_bytes": actual_size,
        "members": members,
    }
    inventory_identity = sha256_bytes(canonical_json(inventory_payload))
    report: dict[str, Any] = {
        "schema_version": "12-6.d03-rada-trees-archive-inventory-report.v1",
        "state": "EXACT_ARCHIVE_AND_MEMBER_INVENTORY_MATERIALIZED_CLASSIFICATION_NOT_RUN",
        "parent_probe_head_sha": config["parent"]["probe_head_sha"],
        "dataset_head_sha": config["parent"]["dataset_head_sha"],
        "archive": {
            "path": config["primary_archive"]["path"],
            "upstream_object_identity": upstream_object_identity,
            "sha256": actual_sha256,
            "size_bytes": actual_size,
        },
        "extractor": {"name": Path(extractor).name, "version_observed": extractor_version(extractor)},
        "member_count": len(members),
        "uncompressed_bytes_observed": total,
        "members": members,
        "inventory_identity_sha256": inventory_identity,
        "member_payload_classification": "NOT_RUN_SUCCESSOR_REQUIRED",
        "normalized_capacity_credited": 0,
        "training_authorized_bytes": 0,
        "training_exposure_authorized": False,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "optimizer_updates": 0,
        "paid_compute_used": False,
        "next_gate": "CLASSIFY_PLAIN_TEXT_AND_BIND_MEMBER_LEVEL_PROVENANCE",
    }
    stable_for_report_id = dict(report)
    stable_for_report_id["extractor"] = {"name": report["extractor"]["name"]}
    report["report_identity_sha256"] = sha256_bytes(canonical_json(stable_for_report_id))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--expected-size", type=int, required=True)
    parser.add_argument("--upstream-object-identity", required=True)
    parser.add_argument("--extractor")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    extractor = find_extractor(args.extractor, config["inventory_policy"]["accepted_extractors"])
    report = build_report(
        config,
        args.archive,
        args.expected_sha256,
        args.expected_size,
        args.upstream_object_identity,
        extractor,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(report)
    tmp = args.output.with_suffix(args.output.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, args.output)
    print("D03_RADA_TREES_ARCHIVE_INVENTORY=MATERIALIZED_ZERO_CREDIT")
    print("INVENTORY_IDENTITY_SHA256=" + report["inventory_identity_sha256"])
    print("TRAINING_AUTHORIZED_BYTES=0")
    print("NEXT=CLASSIFY_PLAIN_TEXT_AND_BIND_MEMBER_LEVEL_PROVENANCE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
