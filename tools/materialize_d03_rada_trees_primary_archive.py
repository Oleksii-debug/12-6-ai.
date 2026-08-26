#!/usr/bin/env python3
"""Materialize and inventory the exact Rada_Trees primary archive.

This successor consumes the immutable Hugging Face/Xet metadata snapshot from
``probe_d03_rada_trees_hf_objects.py``. It can download only the primary
``Rada_Trees.7z`` object at the exact dataset commit, seal its full-content
SHA-256, and build a deterministic metadata-only 7z member inventory.

It never extracts member payloads and cannot grant corpus/training capacity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import probe_d03_rada_trees_hf_objects as object_probe
import validate_d03_rada_trees_acquisition_probe as parent_probe

REPO_ID = object_probe.REPO_ID
REVISION = object_probe.REVISION
PRIMARY_PATH = "Rada_Trees.7z"
REPORT_SCHEMA = "12-6.d03-rada-trees-primary-archive-materialization.v1"
DEFAULT_PIN = ROOT / "evidence/d03-rada-trees/hf-object-identity-v1.json"
DEFAULT_REPORT = ROOT / "evidence/d03-rada-trees/primary-archive-materialization-v1.json"
USER_AGENT = "12-6-ai-rada-trees-primary-materializer/1.0"
CHUNK_BYTES = 1024 * 1024
MAX_MEMBERS = 200_000

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class MaterializationError(RuntimeError):
    """Raised when exact acquisition or archive inventory cannot be proven."""


def _canonical_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(CHUNK_BYTES)
            if not chunk:
                break
            total += len(chunk)
            digest.update(chunk)
    return total, digest.hexdigest()


def _resolve_url(path: str) -> str:
    repo = "/".join(urllib.parse.quote(part, safe="") for part in REPO_ID.split("/"))
    revision = urllib.parse.quote(REVISION, safe="")
    encoded = "/".join(urllib.parse.quote(part, safe="") for part in path.split("/"))
    return f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{encoded}"


def validate_pin_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    try:
        object_probe.validate_snapshot(snapshot)
    except Exception as exc:
        raise MaterializationError(f"invalid parent object-pin snapshot: {exc}") from exc

    source = snapshot.get("source")
    if not isinstance(source, dict):
        raise MaterializationError("object-pin snapshot source is missing")
    if source.get("repo_id") != REPO_ID or source.get("revision") != REVISION:
        raise MaterializationError("object-pin snapshot source identity drift")

    files = snapshot.get("files")
    if not isinstance(files, list):
        raise MaterializationError("object-pin snapshot files are missing")
    matches = [row for row in files if isinstance(row, dict) and row.get("path") == PRIMARY_PATH]
    if len(matches) != 1:
        raise MaterializationError("object-pin snapshot must contain exactly one primary archive")
    row = dict(matches[0])

    size = row.get("size_bytes")
    oid = row.get("git_blob_oid")
    xet_hash = row.get("xet_hash")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise MaterializationError("primary archive exact byte size is invalid")
    if not isinstance(oid, str) or _HEX40.fullmatch(oid) is None:
        raise MaterializationError("primary archive Git blob OID is invalid")
    if not isinstance(xet_hash, str) or _HEX64.fullmatch(xet_hash) is None:
        raise MaterializationError("primary archive Xet identity is invalid")

    boundary = snapshot.get("claim_boundary")
    if not isinstance(boundary, dict) or boundary.get("training_authorized_bytes") != 0:
        raise MaterializationError("parent object-pin snapshot cannot authorize training bytes")
    return row


def download_exact_primary(
    snapshot: dict[str, Any], destination: Path, timeout: float
) -> dict[str, Any]:
    row = validate_pin_snapshot(snapshot)
    expected_size = row["size_bytes"]
    destination = destination.resolve()
    if destination.exists():
        raise MaterializationError(f"refusing to overwrite existing archive: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    url = _resolve_url(PRIMARY_PATH)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": USER_AGENT, "Accept": "application/octet-stream"},
        method="GET",
    )
    staging_fd, staging_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".partial", dir=destination.parent
    )
    os.close(staging_fd)
    staging = Path(staging_name)
    digest = hashlib.sha256()
    total = 0
    final_url = url
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            final_url = response.geturl()
            parsed = urllib.parse.urlparse(final_url)
            if parsed.scheme != "https":
                raise MaterializationError(f"archive redirected to non-HTTPS URL: {final_url}")
            declared = response.headers.get("Content-Length")
            if declared is not None:
                try:
                    declared_size = int(declared)
                except ValueError as exc:
                    raise MaterializationError("malformed Content-Length") from exc
                if declared_size != expected_size:
                    raise MaterializationError(
                        f"Content-Length drift: expected {expected_size}, got {declared_size}"
                    )

            with staging.open("wb") as handle:
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > expected_size:
                        raise MaterializationError("download exceeded exact pinned byte size")
                    digest.update(chunk)
                    handle.write(chunk)
        if total != expected_size:
            raise MaterializationError(
                f"download byte count drift: expected {expected_size}, got {total}"
            )
        os.replace(staging, destination)
    except Exception:
        staging.unlink(missing_ok=True)
        raise

    return {
        "resolve_url": url,
        "final_https_scheme_verified": True,
        "downloaded_bytes": total,
        "content_sha256": digest.hexdigest(),
        "xet_hash": row["xet_hash"],
        "git_blob_oid": row["git_blob_oid"],
    }


def _safe_member_path(value: str) -> str:
    if not value or "\x00" in value or "\\" in value or ":" in value:
        raise MaterializationError(f"unsafe archive member path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise MaterializationError(f"unsafe archive member path: {value!r}")
    return value


def _parse_slt_records(text: str) -> list[dict[str, str]]:
    marker = "----------"
    if marker not in text:
        raise MaterializationError("7z technical listing is missing the archive/member separator")
    member_text = text.split(marker, 1)[1]
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw_line in member_text.splitlines():
        line = raw_line.rstrip("\r")
        if not line.strip():
            if current:
                records.append(current)
                current = {}
            continue
        if " = " not in line:
            continue
        key, value = line.split(" = ", 1)
        current[key] = value
    if current:
        records.append(current)
    return records


def _is_directory(record: Mapping[str, str]) -> bool:
    attrs = record.get("Attributes", "")
    folder = record.get("Folder")
    return folder == "+" or attrs.startswith("D")


def _reject_link_like(record: Mapping[str, str], path: str) -> None:
    for key in ("Symbolic Link", "Hard Link"):
        if record.get(key):
            raise MaterializationError(f"{path}: link-like archive member is prohibited")
    attrs = record.get("Attributes", "")
    for token in attrs.split():
        if len(token) >= 1 and token[0] == "l":
            raise MaterializationError(f"{path}: symlink attributes are prohibited")


def build_member_inventory(
    listing_text: str,
    *,
    max_single_member_bytes: int,
    max_total_uncompressed_bytes: int,
) -> dict[str, Any]:
    if max_single_member_bytes <= 0 or max_total_uncompressed_bytes <= 0:
        raise MaterializationError("archive safety limits must be positive")

    records = _parse_slt_records(listing_text)
    members: list[dict[str, Any]] = []
    seen: set[str] = set()
    total = 0
    for record in records:
        path_value = record.get("Path")
        if not path_value:
            raise MaterializationError("7z member record is missing Path")
        path = _safe_member_path(path_value)
        _reject_link_like(record, path)
        if _is_directory(record):
            continue

        encrypted = record.get("Encrypted")
        if encrypted not in {None, "-"}:
            raise MaterializationError(f"{path}: encrypted archive members are prohibited")

        raw_size = record.get("Size")
        if raw_size is None:
            raise MaterializationError(f"{path}: member size is missing")
        try:
            size = int(raw_size)
        except ValueError as exc:
            raise MaterializationError(f"{path}: malformed member size") from exc
        if size < 0 or size > max_single_member_bytes:
            raise MaterializationError(f"{path}: member size violates safety limit")

        folded = path.casefold()
        if folded in seen:
            raise MaterializationError(f"duplicate/case-colliding archive member path: {path}")
        seen.add(folded)
        total += size
        if total > max_total_uncompressed_bytes:
            raise MaterializationError("archive exceeds total uncompressed-byte safety limit")
        if len(members) >= MAX_MEMBERS:
            raise MaterializationError("archive exceeds member-count safety limit")

        members.append(
            {
                "path": path,
                "size_bytes": size,
                "crc": record.get("CRC"),
            }
        )

    if not members:
        raise MaterializationError("archive contains no regular file members")
    members.sort(key=lambda row: row["path"].encode("utf-8"))
    base = {
        "member_count": len(members),
        "total_uncompressed_bytes": total,
        "members": members,
    }
    return {
        **base,
        "inventory_identity_sha256": hashlib.sha256(_canonical_bytes(base)).hexdigest(),
    }


def find_7z() -> str:
    for candidate in ("7zz", "7z"):
        found = shutil.which(candidate)
        if found:
            return found
    raise MaterializationError("7z/7zz executable is required for archive inventory")


def list_archive_technical(archive: Path, executable: str | None = None) -> tuple[str, str]:
    binary = executable or find_7z()
    version_result = subprocess.run(
        [binary, "i"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if version_result.returncode != 0:
        raise MaterializationError("failed to query 7z runtime identity")
    version_line = next(
        (line.strip() for line in version_result.stdout.splitlines() if line.strip()),
        "UNKNOWN",
    )

    result = subprocess.run(
        [binary, "l", "-slt", "-sccUTF-8", "--", str(archive)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="strict",
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip()
        raise MaterializationError(f"7z archive listing failed: {detail}")
    return result.stdout, version_line


def build_report(
    snapshot: dict[str, Any],
    *,
    archive_path: Path,
    archive_bytes: int,
    archive_sha256: str,
    inventory: dict[str, Any],
    seven_zip_runtime: str,
) -> dict[str, Any]:
    row = validate_pin_snapshot(snapshot)
    if archive_bytes != row["size_bytes"]:
        raise MaterializationError("local archive byte count does not match immutable pin")
    if _HEX64.fullmatch(archive_sha256) is None:
        raise MaterializationError("local archive SHA-256 is malformed")

    parent = parent_probe.load_and_validate()
    acquisition = parent["acquisition_policy"]
    if inventory["total_uncompressed_bytes"] > acquisition["max_total_uncompressed_bytes"]:
        raise MaterializationError("inventory exceeds parent total-uncompressed safety bound")
    if any(
        member["size_bytes"] > acquisition["max_single_member_bytes"]
        for member in inventory["members"]
    ):
        raise MaterializationError("inventory exceeds parent per-member safety bound")

    payload: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "execution_profile": "LOCAL_FREE_ACQUISITION_AND_METADATA_INVENTORY",
        "parent_object_pin_identity_sha256": snapshot["snapshot_identity_sha256"],
        "source": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "path": PRIMARY_PATH,
            "git_blob_oid": row["git_blob_oid"],
            "xet_hash": row["xet_hash"],
            "pinned_bytes": row["size_bytes"],
        },
        "archive": {
            "local_name": archive_path.name,
            "content_bytes": archive_bytes,
            "content_sha256": archive_sha256,
            "seven_zip_runtime": seven_zip_runtime,
        },
        "inventory": inventory,
        "claim_boundary": {
            "archive_downloaded_or_exact_local_copy_verified": True,
            "archive_content_sha256_sealed": True,
            "archive_members_inventoried": True,
            "member_content_sha256_verified": False,
            "archive_members_extracted": False,
            "plain_text_members_classified": False,
            "member_level_provenance_bound": False,
            "quality_privacy_run": False,
            "global_dedup_run": False,
            "evaluation_decontamination_run": False,
            "normalized_capacity_claimed": False,
            "training_authorized_bytes": 0,
            "training_exposure_authorized": False,
            "tokenizer_fit_authorized": False,
            "model_training_executed": False,
            "optimizer_updates": 0,
            "paid_compute_used": False,
            "safe_result": "PRIMARY_ARCHIVE_SEALED_AND_INVENTORIED_MEMBER_HASHING_REQUIRED",
        },
    }
    payload["report_identity_sha256"] = hashlib.sha256(_canonical_bytes(payload)).hexdigest()
    return payload


def validate_report(report: dict[str, Any]) -> None:
    identity = report.get("report_identity_sha256")
    if not isinstance(identity, str) or _HEX64.fullmatch(identity) is None:
        raise MaterializationError("report identity is missing")
    copy = dict(report)
    del copy["report_identity_sha256"]
    if hashlib.sha256(_canonical_bytes(copy)).hexdigest() != identity:
        raise MaterializationError("report identity mismatch")
    if report.get("schema_version") != REPORT_SCHEMA:
        raise MaterializationError("report schema drift")
    boundary = report.get("claim_boundary")
    if not isinstance(boundary, dict):
        raise MaterializationError("claim boundary missing")
    if boundary.get("training_authorized_bytes") != 0:
        raise MaterializationError("archive inventory cannot authorize training bytes")
    if boundary.get("member_content_sha256_verified") is not False:
        raise MaterializationError(
            "metadata-only inventory cannot claim member SHA-256 verification"
        )
    if boundary.get("archive_members_extracted") is not False:
        raise MaterializationError("metadata-only inventory cannot claim extraction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pin-snapshot", type=Path, default=DEFAULT_PIN)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Explicitly allow the ~536 MB exact-revision archive download if --archive is absent.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    snapshot = json.loads(args.pin_snapshot.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict):
        raise MaterializationError("pin snapshot must be a JSON object")
    row = validate_pin_snapshot(snapshot)

    archive = args.archive
    if args.download:
        download_exact_primary(snapshot, archive, args.timeout)
    elif not archive.exists():
        raise MaterializationError("archive is absent; pass --download explicitly to acquire it")

    archive_bytes, archive_sha256 = _sha256_file(archive)
    if archive_bytes != row["size_bytes"]:
        raise MaterializationError(
            f"archive byte count drift: expected {row['size_bytes']}, got {archive_bytes}"
        )

    parent = parent_probe.load_and_validate()
    listing, seven_zip_runtime = list_archive_technical(archive)
    inventory = build_member_inventory(
        listing,
        max_single_member_bytes=parent["acquisition_policy"]["max_single_member_bytes"],
        max_total_uncompressed_bytes=parent["acquisition_policy"]["max_total_uncompressed_bytes"],
    )
    report = build_report(
        snapshot,
        archive_path=archive,
        archive_bytes=archive_bytes,
        archive_sha256=archive_sha256,
        inventory=inventory,
        seven_zip_runtime=seven_zip_runtime,
    )
    validate_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("D03_RADA_TREES_PRIMARY_ARCHIVE_MATERIALIZATION=PASS")
    print("ARCHIVE_CONTENT_SHA256=" + report["archive"]["content_sha256"])
    print("INVENTORY_IDENTITY_SHA256=" + report["inventory"]["inventory_identity_sha256"])
    print("TRAINING_AUTHORIZED_BYTES=0")
    print("NEXT=HASH_MEMBER_CONTENTS_AND_CLASSIFY_ORIGINAL_PLAIN_TEXT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
