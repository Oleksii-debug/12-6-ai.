#!/usr/bin/env python3
"""Fail-closed LOCAL_FREE audit for the pinned Rada_Trees primary archive.

The tool never grants training capacity.  It verifies the exact archive SHA-256,
obtains a deterministic 7z member listing, rejects unsafe/ambiguous paths and
oversized members, then streams every regular member through 7z to compute an
exact payload SHA-256 without extracting an untrusted directory tree.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED_ARCHIVE_SHA256 = "5e53939cd255276c58190569aebfaa6c90fb085fb10063e3e5f661747749719d"
EXPECTED_XET_HASH = "a31d24710d417246fb7e48028baaf6b9efb9a199983d78264f7997c69e42a801"
EXPECTED_REVISION = "1b994a5804dcda122721e8d33a03fd172cf8d867"
EXPECTED_ARCHIVE_NAME = "Rada_Trees.7z"
MAX_MEMBER_BYTES = 50_000_000
MAX_TOTAL_UNCOMPRESSED_BYTES = 10_000_000_000


class AuditError(RuntimeError):
    """Raised when the archive cannot be admitted to the next audit stage."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_member_path(raw: str) -> str:
    if not raw or "\x00" in raw:
        raise AuditError("empty/NUL archive member path")
    replaced = raw.replace("\\", "/")
    if replaced.startswith("/"):
        raise AuditError(f"absolute archive member path: {raw!r}")
    if len(replaced) >= 2 and replaced[1] == ":" and replaced[0].isalpha():
        raise AuditError(f"drive-qualified archive member path: {raw!r}")
    normalized = posixpath.normpath(replaced)
    if normalized in {"", ".", ".."} or normalized.startswith("../"):
        raise AuditError(f"traversal archive member path: {raw!r}")
    parts = PurePosixPath(normalized).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise AuditError(f"ambiguous archive member path: {raw!r}")
    return normalized


def parse_7z_slt(text: str, archive_name: str = EXPECTED_ARCHIVE_NAME) -> list[dict[str, Any]]:
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            if current:
                blocks.append(current)
                current = {}
            continue
        if " = " in line:
            key, value = line.split(" = ", 1)
            current[key.strip()] = value.strip()
    if current:
        blocks.append(current)

    members: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0
    for block in blocks:
        raw_path = block.get("Path")
        if raw_path is None or raw_path == archive_name:
            continue
        folder = block.get("Folder", "-") == "+"
        if folder:
            continue
        path = normalize_member_path(raw_path)
        if path in seen:
            raise AuditError(f"duplicate normalized member path: {path}")
        seen.add(path)
        try:
            size = int(block.get("Size", "-1"))
        except ValueError as exc:
            raise AuditError(f"invalid member size for {path}") from exc
        if size < 0 or size > MAX_MEMBER_BYTES:
            raise AuditError(f"member size outside policy for {path}: {size}")
        total += size
        if total > MAX_TOTAL_UNCOMPRESSED_BYTES:
            raise AuditError("archive total uncompressed bytes exceed policy")
        members.append({"path": path, "size_bytes": size})
    if not members:
        raise AuditError("archive contains no regular members")
    return sorted(members, key=lambda item: item["path"])


def list_members(archive: Path, seven_zip: str) -> list[dict[str, Any]]:
    result = subprocess.run(
        [seven_zip, "l", "-slt", "--", str(archive)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="strict",
    )
    return parse_7z_slt(result.stdout, archive.name)


def stream_member_sha256(archive: Path, member: str, seven_zip: str) -> tuple[str, int]:
    process = subprocess.Popen(
        [seven_zip, "x", "-so", "--", str(archive), member],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    digest = hashlib.sha256()
    count = 0
    for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
        count += len(block)
        if count > MAX_MEMBER_BYTES:
            process.kill()
            raise AuditError(f"streamed member exceeds policy: {member}")
        digest.update(block)
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    returncode = process.wait()
    if returncode != 0:
        raise AuditError(f"7z failed for {member}: {stderr[-500:]}")
    return digest.hexdigest(), count


def audit_archive(archive: Path, seven_zip: str) -> dict[str, Any]:
    if archive.name != EXPECTED_ARCHIVE_NAME:
        raise AuditError(f"expected {EXPECTED_ARCHIVE_NAME}, got {archive.name}")
    observed_sha = sha256_file(archive)
    if observed_sha != EXPECTED_ARCHIVE_SHA256:
        raise AuditError("primary archive SHA-256 mismatch")

    members = list_members(archive, seven_zip)
    audited: list[dict[str, Any]] = []
    for item in members:
        digest, streamed = stream_member_sha256(archive, item["path"], seven_zip)
        if streamed != item["size_bytes"]:
            raise AuditError(f"listed/streamed byte mismatch for {item['path']}")
        audited.append({**item, "sha256": digest})

    canonical = json.dumps(audited, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return {
        "schema_version": "12-6.d03-rada-trees-primary-archive-audit.v1",
        "execution_profile": "LOCAL_FREE",
        "source": {
            "dataset": "uacorpus/Rada_Trees",
            "revision": EXPECTED_REVISION,
            "archive": EXPECTED_ARCHIVE_NAME,
            "archive_sha256": observed_sha,
            "xet_hash": EXPECTED_XET_HASH,
        },
        "member_count": len(audited),
        "total_uncompressed_bytes": sum(item["size_bytes"] for item in audited),
        "member_inventory_sha256": hashlib.sha256(canonical).hexdigest(),
        "members": audited,
        "claim_boundary": {
            "archive_content_sha256_verified": True,
            "archive_members_inventoried": True,
            "language_quality_privacy_pass": False,
            "lineage_dedup_pass": False,
            "evaluation_decontamination_pass": False,
            "family_independence_terminal": False,
            "training_authorized_bytes": 0,
            "tokenizer_fit_authorized": False,
            "training_exposure_authorized": False,
            "model_training_executed": False,
            "paid_compute_used": False,
            "safe_result": "ARCHIVE_BYTES_AND_MEMBERS_BOUND_DOWNSTREAM_ADMISSION_REQUIRED",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    parser.add_argument("--seven-zip", default=os.environ.get("SEVEN_ZIP", "7z"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit_archive(args.archive, args.seven_zip)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("D03_RADA_TREES_PRIMARY_ARCHIVE_AUDIT=PASS")
    print("MEMBER_INVENTORY_SHA256=" + report["member_inventory_sha256"])
    print("TRAINING_AUTHORIZED_BYTES=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
