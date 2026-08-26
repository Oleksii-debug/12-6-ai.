#!/usr/bin/env python3
"""Fail-closed local materializer for the pinned Rada_Trees primary 7z archive.

The tool consumes the exact Hugging Face object-identity snapshot produced by
``probe_d03_rada_trees_hf_objects.py``. It then verifies an independently
supplied content SHA-256, obtains a 7-Zip technical listing, rejects unsafe
archive members, and can optionally extract/hash every regular file.

This tool never grants training capacity.
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
from typing import Any

DATASET = "uacorpus/Rada_Trees"
DATASET_HEAD = "1b994a5804dcda122721e8d33a03fd172cf8d867"
PRIMARY_ARCHIVE = "Rada_Trees.7z"
OBJECT_SNAPSHOT_SCHEMA = "12-6.d03-rada-trees-hf-object-identity.v1"
MAX_COMPRESSED_BYTES = 1_000_000_000
MAX_MEMBER_BYTES = 50_000_000
MAX_TOTAL_UNCOMPRESSED_BYTES = 10_000_000_000
_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_DRIVE = re.compile(r"^[A-Za-z]:")


class IntakeError(RuntimeError):
    """Fail-closed acquisition error."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_object_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Validate the exact metadata-only parent handoff and return primary row."""
    identity = snapshot.get("snapshot_identity_sha256")
    if not isinstance(identity, str) or _HEX64.fullmatch(identity) is None:
        raise IntakeError("object snapshot is missing a valid self identity")

    body = dict(snapshot)
    del body["snapshot_identity_sha256"]
    if _canonical_sha256(body) != identity:
        raise IntakeError("object snapshot self identity mismatch")

    if snapshot.get("schema_version") != OBJECT_SNAPSHOT_SCHEMA:
        raise IntakeError("unexpected object snapshot schema")
    source = snapshot.get("source")
    if not isinstance(source, dict):
        raise IntakeError("object snapshot source is missing")
    if source.get("repo_id") != DATASET or source.get("revision") != DATASET_HEAD:
        raise IntakeError("object snapshot source/revision drift")

    verification = snapshot.get("verification")
    required_verification = {
        "tree_revision_is_immutable_40hex": True,
        "git_blob_oids_bound": True,
        "xet_hashes_bound": True,
        "resolve_header_xet_hashes_match_tree": True,
    }
    if not isinstance(verification, dict):
        raise IntakeError("object snapshot verification is missing")
    for key, expected in required_verification.items():
        if verification.get(key) is not expected:
            raise IntakeError(f"object snapshot verification not terminal: {key}")

    boundary = snapshot.get("claim_boundary")
    if not isinstance(boundary, dict):
        raise IntakeError("object snapshot claim boundary is missing")
    if boundary.get("training_authorized_bytes") != 0:
        raise IntakeError("metadata snapshot cannot authorize training bytes")
    if boundary.get("archives_downloaded") is not False:
        raise IntakeError("metadata snapshot cannot claim archive download")
    if boundary.get("archive_content_sha256_verified") is not False:
        raise IntakeError("metadata snapshot cannot claim content SHA-256")

    files = snapshot.get("files")
    if not isinstance(files, list):
        raise IntakeError("object snapshot files are missing")
    primary_rows = [
        row for row in files if isinstance(row, dict) and row.get("path") == PRIMARY_ARCHIVE
    ]
    if len(primary_rows) != 1:
        raise IntakeError("object snapshot must contain exactly one primary archive row")
    primary = primary_rows[0]

    size = primary.get("size_bytes")
    git_oid = primary.get("git_blob_oid")
    xet_hash = primary.get("xet_hash")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise IntakeError("primary archive metadata has no exact positive size")
    if size > MAX_COMPRESSED_BYTES:
        raise IntakeError("primary archive metadata exceeds compressed-byte bound")
    if not isinstance(git_oid, str) or _HEX40.fullmatch(git_oid) is None:
        raise IntakeError("primary archive Git object identity is malformed")
    if not isinstance(xet_hash, str) or _HEX64.fullmatch(xet_hash) is None:
        raise IntakeError("primary archive Xet object identity is malformed")
    return primary


def load_object_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise IntakeError(f"cannot read object snapshot: {path}") from exc
    if not isinstance(value, dict):
        raise IntakeError("object snapshot root must be a JSON object")
    return value, validate_object_snapshot(value)


def normalize_member_path(raw: str) -> str:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise IntakeError("empty/NUL archive member path")
    if "\\" in raw:
        raise IntakeError(f"backslash archive path rejected: {raw!r}")

    value = unicodedata.normalize("NFKC", raw)
    if value.startswith("/") or _DRIVE.match(value):
        raise IntakeError(f"absolute/archive-drive path rejected: {raw!r}")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise IntakeError(f"unsafe archive path rejected: {raw!r}")
    return PurePosixPath(*parts).as_posix()


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
        start = next(
            i for i, line in enumerate(lines) if line.strip().startswith("----------")
        ) + 1
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
            raw_key, value = line.split(" = ", 1)
            key = raw_key.strip()
            if key in current:
                raise IntakeError(f"duplicate 7z listing field: {key}")
            current[key] = value
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
        is_directory = attributes.startswith("D")
        if any(key in record for key in ("Symbolic Link", "Hard Link")):
            raise IntakeError(f"archive link member rejected: {member_path}")
        if not is_directory and "Size" not in record:
            raise IntakeError(f"archive file member missing Size: {member_path}")

        size = _parse_int(record.get("Size", "0"), "Size", member_path)
        if not is_directory and size > MAX_MEMBER_BYTES:
            raise IntakeError(
                f"member exceeds {MAX_MEMBER_BYTES} byte bound: {member_path} ({size})"
            )
        members.append(
            {
                "path": member_path,
                "size": size,
                "is_directory": is_directory,
                "sha256": None,
            }
        )

    if not members:
        raise IntakeError("7z archive contains no listed members")
    total = sum(member["size"] for member in members if not member["is_directory"])
    if total <= 0:
        raise IntakeError("7z archive contains no non-empty file payload")
    if total > MAX_TOTAL_UNCOMPRESSED_BYTES:
        raise IntakeError(
            f"archive exceeds {MAX_TOTAL_UNCOMPRESSED_BYTES} byte uncompressed bound"
        )
    return members


def run_7z_listing(archive: Path, executable: str = "7z") -> list[dict[str, Any]]:
    resolved = shutil.which(executable)
    if resolved is None:
        raise IntakeError(f"required 7z executable not found: {executable}")
    result = subprocess.run(
        [resolved, "l", "-slt", "--", str(archive)],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
        timeout=180,
    )
    if result.returncode != 0:
        raise IntakeError(
            f"7z listing failed with code {result.returncode}: {result.stderr[-500:]}"
        )
    return parse_7z_slt(result.stdout)


def _collect_extracted_files(root: Path) -> dict[str, Path]:
    actual: dict[str, Path] = {}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        base = Path(dirpath)
        for name in dirnames:
            candidate = base / name
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise IntakeError(f"extracted directory symlink rejected: {candidate}")
            if not stat.S_ISDIR(mode):
                raise IntakeError(f"extracted non-directory entry rejected: {candidate}")
        for name in filenames:
            candidate = base / name
            mode = candidate.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise IntakeError(f"extracted file symlink rejected: {candidate}")
            if not stat.S_ISREG(mode):
                raise IntakeError(f"extracted special file rejected: {candidate}")
            rel = normalize_member_path(candidate.relative_to(root).as_posix())
            if rel in actual:
                raise IntakeError(f"duplicate normalized extracted path: {rel}")
            actual[rel] = candidate
    return actual


def extract_and_hash(
    archive: Path,
    members: list[dict[str, Any]],
    executable: str = "7z",
) -> list[dict[str, Any]]:
    resolved = shutil.which(executable)
    if resolved is None:
        raise IntakeError(f"required 7z executable not found: {executable}")
    expected = {
        member["path"]: member["size"]
        for member in members
        if not member["is_directory"]
    }

    with tempfile.TemporaryDirectory(prefix="rada-trees-intake-") as tmp:
        root = Path(tmp)
        result = subprocess.run(
            [resolved, "x", "-y", "-bd", f"-o{root}", "--", str(archive)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=1800,
        )
        if result.returncode != 0:
            raise IntakeError(
                f"7z extraction failed with code {result.returncode}: {result.stderr[-500:]}"
            )

        actual = _collect_extracted_files(root)
        actual_sizes = {path: file.stat().st_size for path, file in actual.items()}
        if actual_sizes != expected:
            missing = sorted(set(expected) - set(actual_sizes))
            extra = sorted(set(actual_sizes) - set(expected))
            mismatched = sorted(
                path
                for path in set(expected) & set(actual_sizes)
                if expected[path] != actual_sizes[path]
            )
            raise IntakeError(
                "extracted member mismatch; "
                f"missing={missing[:5]} extra={extra[:5]} size_mismatch={mismatched[:5]}"
            )

        enriched: list[dict[str, Any]] = []
        for source in sorted(members, key=lambda item: item["path"]):
            item = dict(source)
            if not item["is_directory"]:
                item["sha256"] = sha256_file(actual[item["path"]])
            enriched.append(item)
        return enriched


def build_report(
    archive: Path,
    expected_sha256: str,
    object_snapshot: dict[str, Any],
    *,
    executable: str = "7z",
    hash_members: bool = False,
) -> dict[str, Any]:
    primary_object = validate_object_snapshot(object_snapshot)
    if archive.name != PRIMARY_ARCHIVE or not archive.is_file() or archive.is_symlink():
        raise IntakeError("expected a regular non-symlink file named exactly Rada_Trees.7z")

    compressed_bytes = archive.stat().st_size
    if compressed_bytes != primary_object["size_bytes"]:
        raise IntakeError(
            "downloaded archive byte size does not match immutable object snapshot"
        )
    if compressed_bytes <= 0 or compressed_bytes > MAX_COMPRESSED_BYTES:
        raise IntakeError(f"compressed archive size outside bound: {compressed_bytes}")
    if _HEX64.fullmatch(expected_sha256) is None:
        raise IntakeError("expected SHA-256 must be 64 lowercase hexadecimal characters")

    actual_sha256 = sha256_file(archive)
    if actual_sha256 != expected_sha256:
        raise IntakeError("archive SHA-256 mismatch")

    members = run_7z_listing(archive, executable)
    if hash_members:
        members = extract_and_hash(archive, members, executable)
    files = [member for member in members if not member["is_directory"]]
    hashes_complete = all(member["sha256"] is not None for member in files)

    report: dict[str, Any] = {
        "schema_version": "12-6.d03-rada-trees-archive-intake.v2",
        "dataset": DATASET,
        "dataset_head": DATASET_HEAD,
        "object_snapshot_identity_sha256": object_snapshot["snapshot_identity_sha256"],
        "archive": {
            "filename": PRIMARY_ARCHIVE,
            "compressed_bytes": compressed_bytes,
            "git_blob_oid": primary_object["git_blob_oid"],
            "xet_hash": primary_object["xet_hash"],
            "content_sha256": actual_sha256,
            "content_sha256_verified": True,
            "object_snapshot_size_match": True,
        },
        "inventory": {
            "member_count": len(members),
            "file_count": len(files),
            "total_uncompressed_file_bytes": sum(member["size"] for member in files),
            "member_hashes_complete": hashes_complete,
            "members": sorted(members, key=lambda item: item["path"]),
        },
        "claim_boundary": {
            "hf_object_identity_verified": True,
            "archive_content_identity_verified": True,
            "safe_member_inventory_verified": True,
            "member_content_hashes_verified": hashes_complete,
            "plain_text_classification_complete": False,
            "period_provenance_stratification_complete": False,
            "rights_member_scope_terminal": False,
            "language_quality_privacy_complete": False,
            "global_lineage_dedup_complete": False,
            "evaluation_decontamination_complete": False,
            "training_authorized_bytes": 0,
            "tokenizer_fit_authorized": False,
            "model_training_authorized": False,
            "safe_result": (
                "ARCHIVE_AND_MEMBER_HASHES_VERIFIED_NEXT_CLASSIFY_AND_STRATIFY"
                if hashes_complete
                else "ARCHIVE_IDENTITY_AND_SAFE_LISTING_VERIFIED_MEMBER_HASHES_PENDING"
            ),
        },
    }
    report["report_sha256"] = _canonical_sha256(report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--object-snapshot", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--seven-zip", default="7z")
    parser.add_argument("--hash-members", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    try:
        object_snapshot, _ = load_object_snapshot(args.object_snapshot)
        report = build_report(
            args.archive,
            args.expected_sha256,
            object_snapshot,
            executable=args.seven_zip,
            hash_members=args.hash_members,
        )
    except IntakeError as exc:
        print(f"BLOCKED: {exc}")
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(report["claim_boundary"]["safe_result"])
    print("REPORT_SHA256=" + report["report_sha256"])
    print("TRAINING_AUTHORIZED_BYTES=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
