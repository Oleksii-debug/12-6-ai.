#!/usr/bin/env python3
"""Fail-closed local materializer for the pinned Rada_Trees primary 7z archive.

The tool never grants training capacity. It verifies an operator-supplied archive
SHA-256, obtains a deterministic 7-Zip technical listing, rejects unsafe archive
members, and can optionally extract into an isolated temporary directory to
hash every regular file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

DATASET = "uacorpus/Rada_Trees"
DATASET_HEAD = "1b994a5804dcda122721e8d33a03fd172cf8d867"
PRIMARY_ARCHIVE = "Rada_Trees.7z"
MAX_COMPRESSED_BYTES = 1_000_000_000
MAX_MEMBER_BYTES = 50_000_000
MAX_TOTAL_UNCOMPRESSED_BYTES = 10_000_000_000


class IntakeError(RuntimeError):
    """Fail-closed acquisition error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_member_path(raw: str) -> str:
    if not raw or "\x00" in raw:
        raise IntakeError("empty/NUL archive member path")
    if "\\" in raw:
        raise IntakeError(f"backslash archive path rejected: {raw!r}")
    path = PurePosixPath(raw)
    if path.is_absolute():
        raise IntakeError(f"absolute archive path rejected: {raw!r}")
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise IntakeError(f"unsafe archive path rejected: {raw!r}")
    return path.as_posix()


def _parse_int(value: str, field: str, path: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise IntakeError(f"invalid {field} for {path!r}: {value!r}") from exc
    if result < 0:
        raise IntakeError(f"negative {field} for {path!r}")
    return result


def parse_7z_slt(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip().startswith("----------")) + 1
    except StopIteration as exc:
        raise IntakeError("7z listing missing member separator") from exc
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in lines[start:]:
        if not line.strip():
            if current:
                records.append(current)
                current = {}
            continue
        if " = " in line:
            key, value = line.split(" = ", 1)
            current[key.strip()] = value
    if current:
        records.append(current)

    members: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in records:
        if "Path" not in record:
            continue
        member_path = normalize_member_path(record["Path"])
        if member_path in seen:
            raise IntakeError(f"duplicate normalized archive path: {member_path}")
        seen.add(member_path)
        attributes = record.get("Attributes", "")
        is_directory = member_path.endswith("/") or attributes.startswith("D")
        if any(key in record for key in ("Symbolic Link", "Hard Link")):
            raise IntakeError(f"archive link member rejected: {member_path}")
        size = _parse_int(record.get("Size", "0"), "Size", member_path)
        if not is_directory and size > MAX_MEMBER_BYTES:
            raise IntakeError(f"member exceeds {MAX_MEMBER_BYTES} byte bound: {member_path} ({size})")
        members.append({"path": member_path.rstrip("/") if is_directory else member_path, "size": size, "is_directory": is_directory, "sha256": None})

    if not members:
        raise IntakeError("7z archive contains no listed members")
    total = sum(m["size"] for m in members if not m["is_directory"])
    if total <= 0:
        raise IntakeError("7z archive contains no non-empty file payload")
    if total > MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise IntakeError(f"archive exceeds {MAX_TOTAL_UNCOMPRESSED_BYTES} byte uncompressed bound")
    return members


def run_7z_listing(archive: Path, executable: str = "7z") -> list[dict[str, Any]]:
    resolved = shutil.which(executable)
    if resolved is None:
        raise IntakeError(f"required 7z executable not found: {executable}")
    result = subprocess.run([resolved, "l", "-slt", "--", str(archive)], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="strict", timeout=180)
    if result.returncode != 0:
        raise IntakeError(f"7z listing failed with code {result.returncode}: {result.stderr[-500:]}")
    return parse_7z_slt(result.stdout)


def extract_and_hash(archive: Path, members: list[dict[str, Any]], executable: str = "7z") -> list[dict[str, Any]]:
    resolved = shutil.which(executable)
    if resolved is None:
        raise IntakeError(f"required 7z executable not found: {executable}")
    expected = {m["path"] for m in members if not m["is_directory"]}
    with tempfile.TemporaryDirectory(prefix="rada-trees-intake-") as tmp:
        root = Path(tmp)
        result = subprocess.run([resolved, "x", "-y", "-bd", f"-o{root}", "--", str(archive)], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace", timeout=1800)
        if result.returncode != 0:
            raise IntakeError(f"7z extraction failed with code {result.returncode}: {result.stderr[-500:]}")
        actual: dict[str, Path] = {}
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            base = Path(dirpath)
            for name in dirnames:
                if (base / name).is_symlink():
                    raise IntakeError(f"extracted directory symlink rejected: {base / name}")
            for name in filenames:
                candidate = base / name
                if candidate.is_symlink():
                    raise IntakeError(f"extracted file symlink rejected: {candidate}")
                actual[normalize_member_path(candidate.relative_to(root).as_posix())] = candidate
        if set(actual) != expected:
            raise IntakeError(f"extracted member set mismatch; missing={sorted(expected-set(actual))[:5]} extra={sorted(set(actual)-expected)[:5]}")
        enriched: list[dict[str, Any]] = []
        for item0 in sorted(members, key=lambda item: item["path"]):
            item = dict(item0)
            if not item["is_directory"]:
                disk = actual[item["path"]]
                if disk.stat().st_size != item["size"]:
                    raise IntakeError(f"size mismatch after extraction for {item['path']}")
                item["sha256"] = sha256_file(disk)
            enriched.append(item)
        return enriched


def build_report(archive: Path, expected_sha256: str, *, executable: str = "7z", hash_members: bool = False) -> dict[str, Any]:
    if archive.name != PRIMARY_ARCHIVE or not archive.is_file():
        raise IntakeError("expected an existing file named exactly Rada_Trees.7z")
    compressed_bytes = archive.stat().st_size
    if compressed_bytes <= 0 or compressed_bytes > MAX_COMPRESSED_BYTES:
        raise IntakeError(f"compressed archive size outside bound: {compressed_bytes}")
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise IntakeError("expected SHA-256 must be 64 lowercase hexadecimal characters")
    actual_sha256 = sha256_file(archive)
    if actual_sha256 != expected_sha256:
        raise IntakeError("archive SHA-256 mismatch")
    members = run_7z_listing(archive, executable)
    if hash_members:
        members = extract_and_hash(archive, members, executable)
    files = [m for m in members if not m["is_directory"]]
    complete = all(m["sha256"] is not None for m in files)
    report: dict[str, Any] = {
        "schema_version": "12-6.d03-rada-trees-archive-intake.v1",
        "dataset": DATASET,
        "dataset_head": DATASET_HEAD,
        "archive": {"filename": PRIMARY_ARCHIVE, "compressed_bytes": compressed_bytes, "sha256": actual_sha256, "sha256_verified": True},
        "inventory": {"member_count": len(members), "file_count": len(files), "total_uncompressed_file_bytes": sum(m["size"] for m in files), "member_hashes_complete": complete, "members": sorted(members, key=lambda item: item["path"])},
        "claim_boundary": {"archive_identity_verified": True, "safe_member_inventory_verified": True, "member_content_hashes_verified": complete, "plain_text_classification_complete": False, "rights_member_scope_terminal": False, "language_quality_privacy_complete": False, "global_lineage_dedup_complete": False, "evaluation_decontamination_complete": False, "training_authorized_bytes": 0, "tokenizer_fit_authorized": False, "model_training_authorized": False, "safe_result": "ARCHIVE_AND_MEMBER_HASHES_VERIFIED_NEXT_CLASSIFY" if complete else "ARCHIVE_IDENTITY_AND_SAFE_LISTING_VERIFIED_MEMBER_HASHES_PENDING"},
    }
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    report["report_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seven-zip", default="7z")
    parser.add_argument("--hash-members", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_report(args.archive, args.expected_sha256, executable=args.seven_zip, hash_members=args.hash_members)
    except IntakeError as exc:
        print(f"BLOCKED: {exc}")
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report["claim_boundary"]["safe_result"])
    print(report["report_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
