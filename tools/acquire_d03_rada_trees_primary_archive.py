#!/usr/bin/env python3
"""Bounded Rada_Trees archive acquisition and safe 7z member inventory.

Consumes the exact Hugging Face object snapshot produced by
``probe_d03_rada_trees_hf_objects.py``. Downloads only the primary archive,
streams a full-content SHA-256, and inventories archive metadata without
extracting member bodies. It cannot grant training capacity.
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
import unicodedata
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import probe_d03_rada_trees_hf_objects as object_probe
import validate_d03_rada_trees_acquisition_probe as acquisition_probe

PRIMARY_ARCHIVE = "Rada_Trees.7z"
DEFAULT_OBJECT_SNAPSHOT = ROOT / "evidence/d03-rada-trees/hf-object-identity-v1.json"
DEFAULT_ARCHIVE = ROOT / "data/external/rada-trees/Rada_Trees.7z"
DEFAULT_REPORT = ROOT / "evidence/d03-rada-trees/archive-inventory-v1.json"
USER_AGENT = "12-6-ai-rada-trees-acquire/1.0"

_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class AcquisitionError(RuntimeError):
    """Fail-closed acquisition or inventory error."""


def _primary_from_snapshot(snapshot: Mapping[str, Any]) -> Mapping[str, Any]:
    object_probe.validate_snapshot(dict(snapshot))
    files = snapshot.get("files")
    if not isinstance(files, list):
        raise AcquisitionError("object snapshot files must be a list")
    matches = [
        item
        for item in files
        if isinstance(item, Mapping) and item.get("path") == PRIMARY_ARCHIVE
    ]
    if len(matches) != 1:
        raise AcquisitionError("object snapshot must contain exactly one primary archive")
    primary = matches[0]
    size = primary.get("size_bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise AcquisitionError("primary archive exact size is missing")
    xet_hash = primary.get("xet_hash")
    if not isinstance(xet_hash, str) or _HEX64.fullmatch(xet_hash) is None:
        raise AcquisitionError("primary archive Xet identity is missing")
    if snapshot["claim_boundary"]["training_authorized_bytes"] != 0:
        raise AcquisitionError("object snapshot unexpectedly authorizes training")
    return primary


def _download_url() -> str:
    return object_probe._resolve_url(PRIMARY_ARCHIVE)


def _download_stream(
    *,
    expected_size: int,
    expected_xet_hash: str,
    destination: Path,
    timeout: float,
    chunk_size: int,
) -> tuple[int, str]:
    if chunk_size < 64 * 1024 or chunk_size > 16 * 1024 * 1024:
        raise AcquisitionError("chunk size must be between 64 KiB and 16 MiB")
    if destination.exists():
        raise AcquisitionError(f"destination already exists: {destination}")

    # Re-prove the immutable object header immediately before following the
    # large-file redirect. The incumbent probe does not download the body.
    live_xet_hash = object_probe.fetch_resolve_xet_hash(PRIMARY_ARCHIVE, timeout=timeout)
    if live_xet_hash != expected_xet_hash:
        raise AcquisitionError("live resolve X-Xet-Hash drifted from the pinned object snapshot")

    request = urllib.request.Request(
        _download_url(),
        headers={"User-Agent": USER_AGENT, "Accept": "application/octet-stream"},
        method="GET",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0

    fd, tmp_name = tempfile.mkstemp(
        prefix=destination.name + ".",
        suffix=".part",
        dir=destination.parent,
    )
    os.close(fd)
    temp_path = Path(tmp_name)
    try:
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.URLError as exc:
            raise AcquisitionError(f"archive download failed: {exc.reason}") from exc
        with response, temp_path.open("wb") as output:
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared = int(content_length)
                except ValueError as exc:
                    raise AcquisitionError("download Content-Length is malformed") from exc
                if declared != expected_size:
                    raise AcquisitionError(
                        f"download Content-Length mismatch: expected {expected_size}, got {declared}"
                    )
            while True:
                block = response.read(chunk_size)
                if not block:
                    break
                total += len(block)
                if total > expected_size:
                    raise AcquisitionError("download exceeded pinned exact archive size")
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())

        if total != expected_size:
            raise AcquisitionError(f"download size mismatch: expected {expected_size}, got {total}")
        os.replace(temp_path, destination)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise

    return total, digest.hexdigest()


def _safe_member_path(raw_path: str) -> str:
    if not isinstance(raw_path, str) or not raw_path:
        raise AcquisitionError("archive member path is empty")
    if "\x00" in raw_path:
        raise AcquisitionError("archive member path contains NUL")
    normalized = unicodedata.normalize("NFC", raw_path.replace("\\", "/"))
    if normalized.startswith("/") or _DRIVE_PREFIX.match(normalized):
        raise AcquisitionError(f"absolute archive member path rejected: {raw_path!r}")
    while normalized.endswith("/"):
        normalized = normalized[:-1]
    parts = normalized.split("/")
    if not normalized or any(part in {"", ".", ".."} for part in parts):
        raise AcquisitionError(f"unsafe archive member path rejected: {raw_path!r}")
    if len(normalized.encode("utf-8")) > 4096:
        raise AcquisitionError("archive member path exceeds 4096 UTF-8 bytes")
    for char in normalized:
        if unicodedata.category(char).startswith("C"):
            raise AcquisitionError(f"archive member path contains control character: {raw_path!r}")
    return normalized


def parse_7z_slt(text: str) -> list[dict[str, str]]:
    """Parse ``7z l -slt -ba`` records without trusting archive paths."""
    records: list[dict[str, str]] = []
    current: dict[str, str] = {}

    def flush() -> None:
        nonlocal current
        if current:
            if "Path" not in current:
                raise AcquisitionError("7z technical record is missing Path")
            records.append(current)
            current = {}

    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if not line:
            flush()
            continue
        if " = " not in line:
            raise AcquisitionError(f"unexpected 7z technical output line: {line!r}")
        key, value = line.split(" = ", 1)
        if not key or key in current:
            raise AcquisitionError(f"duplicate or empty 7z technical field: {key!r}")
        current[key] = value
    flush()

    if not records:
        raise AcquisitionError("7z inventory returned no member records")
    return records


def _is_link_record(record: Mapping[str, str]) -> bool:
    link_keys = ("Symbolic Link", "Hard Link", "Link")
    return any(record.get(key, "").strip() not in {"", "-"} for key in link_keys)


def _member_classification(path: str, *, is_directory: bool) -> str:
    if is_directory:
        return "DIRECTORY"
    suffix = Path(path).suffix.lower()
    if suffix in {".txt", ".text"}:
        return "PLAIN_TEXT_CANDIDATE_EXTENSION_ONLY"
    if suffix in {".conllu", ".conll", ".xml", ".vert", ".json", ".jsonl", ".tsv", ".csv"}:
        return "ANNOTATION_OR_STRUCTURED_DERIVATIVE_HOLD"
    return "UNCLASSIFIED_HOLD"


def normalize_inventory(
    records: Sequence[Mapping[str, str]],
    *,
    max_single_member_bytes: int,
    max_total_uncompressed_bytes: int,
) -> list[dict[str, Any]]:
    if max_single_member_bytes <= 0 or max_total_uncompressed_bytes <= 0:
        raise AcquisitionError("archive limits must be positive")

    members: list[dict[str, Any]] = []
    exact_seen: set[str] = set()
    casefold_seen: set[str] = set()
    total = 0

    for record in records:
        path = _safe_member_path(record.get("Path", ""))
        folded = path.casefold()
        if path in exact_seen:
            raise AcquisitionError(f"duplicate archive member path: {path}")
        if folded in casefold_seen:
            raise AcquisitionError(f"case-fold archive member collision: {path}")
        exact_seen.add(path)
        casefold_seen.add(folded)

        if _is_link_record(record):
            raise AcquisitionError(f"archive link member rejected: {path}")
        if record.get("Encrypted", "-").strip() not in {"", "-", "0"}:
            raise AcquisitionError(f"encrypted archive member rejected: {path}")

        folder_marker = record.get("Folder", "").strip()
        attrs = record.get("Attributes", "")
        is_directory = folder_marker == "+" or attrs.startswith("D")

        raw_size = record.get("Size", "0" if is_directory else None)
        if raw_size is None:
            raise AcquisitionError(f"archive file member has no Size: {path}")
        try:
            size = int(raw_size)
        except ValueError as exc:
            raise AcquisitionError(f"archive member size is malformed: {path}") from exc
        if size < 0:
            raise AcquisitionError(f"archive member size is negative: {path}")
        if is_directory and size != 0:
            raise AcquisitionError(f"archive directory has nonzero size: {path}")
        if not is_directory and size > max_single_member_bytes:
            raise AcquisitionError(
                f"archive member exceeds max_single_member_bytes ({max_single_member_bytes}): {path}"
            )

        total += size
        if total > max_total_uncompressed_bytes:
            raise AcquisitionError(
                f"archive exceeds max_total_uncompressed_bytes ({max_total_uncompressed_bytes})"
            )

        members.append(
            {
                "path": path,
                "kind": "directory" if is_directory else "file",
                "size_bytes": size,
                "classification": _member_classification(path, is_directory=is_directory),
                "crc": record.get("CRC") or None,
            }
        )

    members.sort(key=lambda item: item["path"].encode("utf-8"))
    return members


def _find_7z() -> str:
    for candidate in ("7zz", "7z"):
        found = shutil.which(candidate)
        if found:
            return found
    raise AcquisitionError("7z/7zz executable is required for safe metadata inventory")


def inventory_archive(
    archive: Path,
    *,
    max_single_member_bytes: int,
    max_total_uncompressed_bytes: int,
    timeout: float,
) -> tuple[list[dict[str, Any]], str]:
    executable = _find_7z()
    try:
        completed = subprocess.run(
            [executable, "l", "-slt", "-ba", str(archive)],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as exc:
        raise AcquisitionError(f"7z inventory execution failed: {exc}") from exc
    if completed.returncode != 0:
        raise AcquisitionError("7z inventory failed: " + completed.stderr.strip()[:1000])
    members = normalize_inventory(
        parse_7z_slt(completed.stdout),
        max_single_member_bytes=max_single_member_bytes,
        max_total_uncompressed_bytes=max_total_uncompressed_bytes,
    )
    return members, Path(executable).name


def _report_identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_report(
    *,
    snapshot: Mapping[str, Any],
    archive_size: int,
    archive_sha256: str,
    members: Sequence[Mapping[str, Any]],
    inventory_tool: str,
) -> dict[str, Any]:
    primary = _primary_from_snapshot(snapshot)
    if archive_size != primary["size_bytes"]:
        raise AcquisitionError("downloaded archive size does not match pinned object snapshot")
    if not isinstance(archive_sha256, str) or _HEX64.fullmatch(archive_sha256) is None:
        raise AcquisitionError("archive content SHA-256 is malformed")
    if not members:
        raise AcquisitionError("archive member inventory is empty")

    counts = Counter(str(item["classification"]) for item in members)
    file_count = sum(item["kind"] == "file" for item in members)
    dir_count = sum(item["kind"] == "directory" for item in members)
    total_uncompressed = sum(int(item["size_bytes"]) for item in members)

    payload: dict[str, Any] = {
        "schema_version": "12-6.d03-rada-trees-archive-inventory.v1",
        "execution_profile": "LOCAL_FREE_ACQUISITION_NO_TRAINING",
        "source": {
            "repo_id": snapshot["source"]["repo_id"],
            "revision": snapshot["source"]["revision"],
            "object_snapshot_identity_sha256": snapshot["snapshot_identity_sha256"],
            "primary_archive_path": PRIMARY_ARCHIVE,
            "primary_xet_hash": primary["xet_hash"],
            "primary_git_blob_oid": primary["git_blob_oid"],
        },
        "archive": {
            "downloaded_size_bytes": archive_size,
            "content_sha256": archive_sha256,
        },
        "inventory": {
            "tool": inventory_tool,
            "format": "7z-list-slt-no-extraction",
            "member_count": len(members),
            "file_count": file_count,
            "directory_count": dir_count,
            "total_uncompressed_bytes": total_uncompressed,
            "classification_counts": dict(sorted(counts.items())),
            "members": list(members),
        },
        "claim_boundary": {
            "archive_downloaded": True,
            "archive_content_sha256_verified": True,
            "archive_members_inventoried": True,
            "member_content_sha256_verified": False,
            "plain_text_members_admitted": False,
            "family_independence_terminal": False,
            "quality_privacy_pass": False,
            "lineage_dedup_pass": False,
            "evaluation_decontamination_pass": False,
            "normalized_capacity_claimed": False,
            "training_authorized_bytes": 0,
            "training_exposure_authorized": False,
            "tokenizer_fit_authorized": False,
            "model_training_executed": False,
            "optimizer_updates": 0,
            "paid_compute_used": False,
            "safe_result": "ARCHIVE_BYTES_AND_SAFE_MEMBER_METADATA_PINNED_CONTENT_REVIEW_REQUIRED",
        },
    }
    payload["report_identity_sha256"] = _report_identity(payload)
    validate_report(payload)
    return payload


def validate_report(report: Mapping[str, Any]) -> None:
    if report.get("schema_version") != "12-6.d03-rada-trees-archive-inventory.v1":
        raise AcquisitionError("unexpected report schema")
    identity = report.get("report_identity_sha256")
    if not isinstance(identity, str) or _HEX64.fullmatch(identity) is None:
        raise AcquisitionError("report identity is missing")
    copy = dict(report)
    del copy["report_identity_sha256"]
    if _report_identity(copy) != identity:
        raise AcquisitionError("report self-hash mismatch")

    boundary = report["claim_boundary"]
    for key in ("archive_downloaded", "archive_content_sha256_verified", "archive_members_inventoried"):
        if boundary.get(key) is not True:
            raise AcquisitionError(f"acquisition evidence must prove {key}")
    if boundary["member_content_sha256_verified"] is not False:
        raise AcquisitionError("inventory-only stage cannot claim member content SHA-256")
    if boundary["training_authorized_bytes"] != 0 or boundary["optimizer_updates"] != 0:
        raise AcquisitionError("archive inventory cannot authorize model training")
    for key in (
        "plain_text_members_admitted",
        "family_independence_terminal",
        "quality_privacy_pass",
        "lineage_dedup_pass",
        "evaluation_decontamination_pass",
        "normalized_capacity_claimed",
        "training_exposure_authorized",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "paid_compute_used",
    ):
        if boundary[key] is not False:
            raise AcquisitionError(f"premature downstream claim rejected: {key}")


def acquire(
    *,
    object_snapshot_path: Path,
    archive_path: Path,
    report_path: Path,
    timeout: float,
    inventory_timeout: float,
    chunk_size: int,
) -> None:
    snapshot = json.loads(object_snapshot_path.read_text(encoding="utf-8"))
    primary = _primary_from_snapshot(snapshot)
    parent = acquisition_probe.load_and_validate()
    policy = parent["acquisition_policy"]

    archive_size, archive_sha256 = _download_stream(
        expected_size=int(primary["size_bytes"]),
        expected_xet_hash=str(primary["xet_hash"]),
        destination=archive_path,
        timeout=timeout,
        chunk_size=chunk_size,
    )
    members, inventory_tool = inventory_archive(
        archive_path,
        max_single_member_bytes=int(policy["max_single_member_bytes"]),
        max_total_uncompressed_bytes=int(policy["max_total_uncompressed_bytes"]),
        timeout=inventory_timeout,
    )
    report = build_report(
        snapshot=snapshot,
        archive_size=archive_size,
        archive_sha256=archive_sha256,
        members=members,
        inventory_tool=inventory_tool,
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    acquire_parser = sub.add_parser("acquire")
    acquire_parser.add_argument("--object-snapshot", type=Path, default=DEFAULT_OBJECT_SNAPSHOT)
    acquire_parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    acquire_parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    acquire_parser.add_argument("--timeout", type=float, default=120.0)
    acquire_parser.add_argument("--inventory-timeout", type=float, default=120.0)
    acquire_parser.add_argument("--chunk-size", type=int, default=1024 * 1024)

    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--report", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "acquire":
        acquire(
            object_snapshot_path=args.object_snapshot,
            archive_path=args.archive,
            report_path=args.report,
            timeout=args.timeout,
            inventory_timeout=args.inventory_timeout,
            chunk_size=args.chunk_size,
        )
    else:
        report = json.loads(args.report.read_text(encoding="utf-8"))
        validate_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
