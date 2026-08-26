#!/usr/bin/env python3
"""Verify and inventory the pinned Rada_Trees primary archive without granting data credit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import probe_d03_rada_trees_hf_objects as hf_probe

ARCHIVE_PATH = "Rada_Trees.7z"
REPO_ID = hf_probe.REPO_ID
REVISION = hf_probe.REVISION
DEFAULT_OUTPUT = ROOT / "evidence/d03-rada-trees/archive-member-inventory-v1.json"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")
WINDOWS_DEVICE = re.compile(
    r"^(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?$", re.IGNORECASE
)


class ArchiveIntakeError(RuntimeError):
    """Fail-closed archive verification error."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    if chunk_size <= 0:
        raise ArchiveIntakeError("chunk_size must be positive")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_hf_snapshot(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        snapshot = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ArchiveIntakeError(f"cannot read HF object snapshot: {exc}") from exc
    if not isinstance(snapshot, dict):
        raise ArchiveIntakeError("HF object snapshot must be a JSON object")
    try:
        hf_probe.validate_snapshot(snapshot)
    except (hf_probe.ProbeError, KeyError, TypeError, ValueError) as exc:
        raise ArchiveIntakeError(f"invalid HF object snapshot: {exc}") from exc
    if snapshot["source"]["repo_id"] != REPO_ID:
        raise ArchiveIntakeError("HF snapshot repository drift")
    if snapshot["source"]["revision"] != REVISION:
        raise ArchiveIntakeError("HF snapshot revision drift")
    matches = [item for item in snapshot["files"] if item.get("path") == ARCHIVE_PATH]
    if len(matches) != 1:
        raise ArchiveIntakeError("HF snapshot must contain exactly one primary archive")
    return snapshot, matches[0]


def validate_member_path(raw_path: str) -> str:
    if not raw_path or "\x00" in raw_path:
        raise ArchiveIntakeError("archive member path is empty or contains NUL")
    if raw_path != unicodedata.normalize("NFC", raw_path):
        raise ArchiveIntakeError(f"archive member path is not NFC: {raw_path!r}")
    if any(ord(char) < 32 or ord(char) == 127 for char in raw_path):
        raise ArchiveIntakeError(f"archive member path contains control characters: {raw_path!r}")
    if "\\" in raw_path:
        raise ArchiveIntakeError(f"archive member path uses backslash: {raw_path!r}")
    if raw_path.startswith("/") or DRIVE_PREFIX.match(raw_path):
        raise ArchiveIntakeError(f"archive member path is absolute: {raw_path!r}")
    pure = PurePosixPath(raw_path)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in raw_path.split("/")):
        raise ArchiveIntakeError(f"archive member path is non-canonical: {raw_path!r}")
    if pure.as_posix() != raw_path:
        raise ArchiveIntakeError(f"archive member path normalization drift: {raw_path!r}")
    if len(raw_path.encode("utf-8")) > 4096:
        raise ArchiveIntakeError("archive member path exceeds 4096 UTF-8 bytes")
    for part in pure.parts:
        encoded = part.encode("utf-8")
        if len(encoded) > 255:
            raise ArchiveIntakeError(f"archive member component exceeds 255 bytes: {part!r}")
        if part.endswith((" ", ".")):
            raise ArchiveIntakeError(f"archive member has unsafe trailing character: {part!r}")
        if ":" in part or WINDOWS_DEVICE.fullmatch(part):
            raise ArchiveIntakeError(f"archive member is not portable/safe: {part!r}")
    return raw_path


def _parse_slt_blocks(text: str) -> list[dict[str, str]]:
    separator_seen = False
    blocks: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.rstrip("\r")
        if not separator_seen:
            if line.strip() and set(line.strip()) == {"-"} and len(line.strip()) >= 5:
                separator_seen = True
            continue
        if not line.strip():
            if current:
                blocks.append(current)
                current = {}
            continue
        if " = " not in line:
            raise ArchiveIntakeError(f"malformed 7z -slt line: {line!r}")
        key, value = line.split(" = ", 1)
        if not key or key in current:
            raise ArchiveIntakeError(f"duplicate/malformed 7z field: {key!r}")
        current[key] = value
    if current:
        blocks.append(current)
    if not separator_seen:
        raise ArchiveIntakeError("7z listing member separator not found")
    return blocks


def parse_7z_slt(
    text: str,
    *,
    max_single_member_bytes: int,
    max_total_uncompressed_bytes: int,
) -> list[dict[str, Any]]:
    if max_single_member_bytes <= 0 or max_total_uncompressed_bytes <= 0:
        raise ArchiveIntakeError("archive expansion limits must be positive")
    blocks = _parse_slt_blocks(text)
    members: list[dict[str, Any]] = []
    seen_exact: set[str] = set()
    seen_casefold: set[str] = set()
    total = 0
    for block in blocks:
        path_value = block.get("Path")
        if path_value is None:
            raise ArchiveIntakeError("7z member block missing Path")
        path = validate_member_path(path_value)
        folder = block.get("Folder", "-")
        if folder not in {"-", "+"}:
            raise ArchiveIntakeError(f"{path}: malformed Folder field")
        if folder == "+":
            continue
        for link_key in ("Symbolic Link", "Hard Link"):
            if block.get(link_key, "") not in {"", "-"}:
                raise ArchiveIntakeError(f"{path}: {link_key.lower()} member rejected")
        attributes = block.get("Attributes", "")
        if "L" in attributes.upper():
            raise ArchiveIntakeError(f"{path}: link-like archive attributes rejected")
        encrypted = block.get("Encrypted", "-")
        if encrypted not in {"-", "0"}:
            raise ArchiveIntakeError(f"{path}: encrypted member rejected")
        raw_size = block.get("Size")
        if raw_size is None or not re.fullmatch(r"[0-9]+", raw_size):
            raise ArchiveIntakeError(f"{path}: missing or malformed Size")
        size = int(raw_size)
        if size > max_single_member_bytes:
            raise ArchiveIntakeError(f"{path}: member exceeds size limit")
        if path in seen_exact:
            raise ArchiveIntakeError(f"duplicate archive member path: {path}")
        folded = path.casefold()
        if folded in seen_casefold:
            raise ArchiveIntakeError(f"case-insensitive archive member collision: {path}")
        seen_exact.add(path)
        seen_casefold.add(folded)
        total += size
        if total > max_total_uncompressed_bytes:
            raise ArchiveIntakeError("archive exceeds total uncompressed-size limit")
        members.append({"path": path, "size_bytes": size, "sha256": None})
    if not members:
        raise ArchiveIntakeError("archive contains no regular-file members")
    if not any(item["size_bytes"] > 0 for item in members):
        raise ArchiveIntakeError("archive contains no non-empty payload")
    members.sort(key=lambda item: item["path"])
    return members


def run_7z_listing(
    archive: Path,
    *,
    seven_zip: str = "7z",
    timeout_seconds: int = 300,
) -> str:
    executable = shutil.which(seven_zip) if os.path.sep not in seven_zip else seven_zip
    if executable is None:
        raise ArchiveIntakeError(f"7z executable not found: {seven_zip}")
    try:
        result = subprocess.run(
            [executable, "l", "-slt", "--", str(archive)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            timeout=timeout_seconds,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired, UnicodeError) as exc:
        raise ArchiveIntakeError(f"7z listing failed: {exc}") from exc
    if result.returncode != 0:
        stderr = result.stderr.strip()
        raise ArchiveIntakeError(f"7z listing returned {result.returncode}: {stderr[:500]}")
    return result.stdout


def hash_extracted_members(
    archive: Path,
    members: list[dict[str, Any]],
    *,
    seven_zip: str = "7z",
    timeout_seconds: int = 1800,
) -> list[dict[str, Any]]:
    executable = shutil.which(seven_zip) if os.path.sep not in seven_zip else seven_zip
    if executable is None:
        raise ArchiveIntakeError(f"7z executable not found: {seven_zip}")
    expected = {item["path"]: item["size_bytes"] for item in members}
    with tempfile.TemporaryDirectory(prefix="12-6-rada-trees-") as temp_name:
        root = Path(temp_name)
        try:
            result = subprocess.run(
                [executable, "x", "-y", "-bd", f"-o{root}", "--", str(archive)],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ArchiveIntakeError(f"7z extraction failed: {exc}") from exc
        if result.returncode != 0:
            raise ArchiveIntakeError(
                f"7z extraction returned {result.returncode}: {result.stderr.strip()[:500]}"
            )

        actual: dict[str, Path] = {}
        for candidate in root.rglob("*"):
            relative = candidate.relative_to(root).as_posix()
            st = candidate.lstat()
            if stat.S_ISLNK(st.st_mode):
                raise ArchiveIntakeError(f"extracted symbolic link rejected: {relative}")
            if candidate.is_dir():
                continue
            if not stat.S_ISREG(st.st_mode):
                raise ArchiveIntakeError(f"extracted non-regular member rejected: {relative}")
            if st.st_nlink != 1:
                raise ArchiveIntakeError(f"extracted hard-linked member rejected: {relative}")
            safe_relative = validate_member_path(relative)
            if safe_relative in actual:
                raise ArchiveIntakeError(f"duplicate extracted member: {safe_relative}")
            actual[safe_relative] = candidate

        if set(actual) != set(expected):
            missing = sorted(set(expected) - set(actual))
            extra = sorted(set(actual) - set(expected))
            raise ArchiveIntakeError(
                f"extracted member set mismatch; missing={missing[:5]!r} extra={extra[:5]!r}"
            )

        output: list[dict[str, Any]] = []
        for path in sorted(expected):
            candidate = actual[path]
            size = candidate.stat().st_size
            if size != expected[path]:
                raise ArchiveIntakeError(
                    f"{path}: extracted size {size} != listed size {expected[path]}"
                )
            output.append(
                {"path": path, "size_bytes": size, "sha256": sha256_file(candidate)}
            )
        return output


def build_report(
    *,
    hf_snapshot: dict[str, Any],
    hf_archive: dict[str, Any],
    archive_sha256: str,
    archive_size_bytes: int,
    members: list[dict[str, Any]],
    hashes_verified: bool,
) -> dict[str, Any]:
    if not HEX64.fullmatch(archive_sha256):
        raise ArchiveIntakeError("archive SHA-256 must be lowercase 64-hex")
    if archive_size_bytes != hf_archive["size_bytes"]:
        raise ArchiveIntakeError("local archive size does not match pinned HF object size")
    if not members:
        raise ArchiveIntakeError("cannot build report without members")
    if hashes_verified:
        if any(not isinstance(item.get("sha256"), str) or not HEX64.fullmatch(item["sha256"])
               for item in members):
            raise ArchiveIntakeError("hashes_verified requires every member SHA-256")
    elif any(item.get("sha256") is not None for item in members):
        raise ArchiveIntakeError("unverified report cannot carry member SHA-256 values")

    inventory_core = [
        {"path": item["path"], "size_bytes": item["size_bytes"], "sha256": item["sha256"]}
        for item in members
    ]
    inventory_identity = hashlib.sha256(canonical_json_bytes(inventory_core)).hexdigest()
    total_bytes = sum(item["size_bytes"] for item in members)
    report: dict[str, Any] = {
        "schema_version": "12-6.d03-rada-trees-archive-inventory.v1",
        "execution_profile": "LOCAL_FREE",
        "source": {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "hf_snapshot_identity_sha256": hf_snapshot["snapshot_identity_sha256"],
            "archive_path": ARCHIVE_PATH,
            "hf_git_blob_oid": hf_archive["git_blob_oid"],
            "hf_xet_hash": hf_archive["xet_hash"],
            "archive_size_bytes": archive_size_bytes,
            "archive_sha256": archive_sha256,
        },
        "inventory": {
            "regular_file_count": len(members),
            "total_uncompressed_bytes": total_bytes,
            "member_hashes_verified": hashes_verified,
            "inventory_identity_sha256": inventory_identity,
            "members": inventory_core,
        },
        "verification": {
            "hf_object_snapshot_validated": True,
            "local_archive_size_matches_hf": True,
            "archive_content_sha256_verified": True,
            "member_paths_and_sizes_fail_closed": True,
            "member_sha256_verified": hashes_verified,
        },
        "claim_boundary": {
            "archive_download_performed_by_this_tool": False,
            "plain_text_members_classified": False,
            "member_rights_provenance_terminal": False,
            "quality_privacy_terminal": False,
            "global_lineage_dedup_terminal": False,
            "evaluation_decontamination_terminal": False,
            "family_cap_recomputed": False,
            "training_authorized_bytes": 0,
            "training_exposure_authorized": False,
            "tokenizer_fit_authorized": False,
            "model_training_authorized": False,
            "optimizer_updates": 0,
            "paid_compute_used": False,
            "safe_result": (
                "ARCHIVE_IDENTITY_AND_MEMBER_INVENTORY_VERIFIED_DOWNSTREAM_CLASSIFICATION_REQUIRED"
            ),
        },
    }
    report["report_identity_sha256"] = hashlib.sha256(canonical_json_bytes(report)).hexdigest()
    return report


def validate_report(report: dict[str, Any]) -> None:
    identity = report.get("report_identity_sha256")
    if not isinstance(identity, str) or not HEX64.fullmatch(identity):
        raise ArchiveIntakeError("report identity missing or malformed")
    core = dict(report)
    del core["report_identity_sha256"]
    if hashlib.sha256(canonical_json_bytes(core)).hexdigest() != identity:
        raise ArchiveIntakeError("report identity mismatch")
    if report["source"]["repo_id"] != REPO_ID or report["source"]["revision"] != REVISION:
        raise ArchiveIntakeError("report source authority drift")
    if report["source"]["archive_path"] != ARCHIVE_PATH:
        raise ArchiveIntakeError("report archive path drift")
    members = report["inventory"]["members"]
    boundary = report["claim_boundary"]
    expected_false = (
        "archive_download_performed_by_this_tool",
        "plain_text_members_classified",
        "member_rights_provenance_terminal",
        "quality_privacy_terminal",
        "global_lineage_dedup_terminal",
        "evaluation_decontamination_terminal",
        "family_cap_recomputed",
        "training_exposure_authorized",
        "tokenizer_fit_authorized",
        "model_training_authorized",
        "paid_compute_used",
    )
    if any(boundary[key] is not False for key in expected_false):
        raise ArchiveIntakeError("report claim boundary was broadened")
    if boundary["training_authorized_bytes"] != 0 or boundary["optimizer_updates"] != 0:
        raise ArchiveIntakeError("archive intake cannot authorize training or optimizer updates")
    if report["verification"]["archive_content_sha256_verified"] is not True:
        raise ArchiveIntakeError("report must prove archive content SHA-256")
    hashes_verified = report["inventory"]["member_hashes_verified"]
    if not isinstance(hashes_verified, bool):
        raise ArchiveIntakeError("member_hashes_verified must be boolean")
    for item in members:
        member_hash = item.get("sha256")
        if hashes_verified:
            if not isinstance(member_hash, str) or not HEX64.fullmatch(member_hash):
                raise ArchiveIntakeError("verified inventory requires every member SHA-256")
        elif member_hash is not None:
            raise ArchiveIntakeError("unverified inventory cannot carry member SHA-256")
    if report["inventory"]["regular_file_count"] != len(members):
        raise ArchiveIntakeError("report member count mismatch")
    total = sum(item["size_bytes"] for item in members)
    if report["inventory"]["total_uncompressed_bytes"] != total:
        raise ArchiveIntakeError("report uncompressed-byte total mismatch")
    expected_inventory = hashlib.sha256(canonical_json_bytes(members)).hexdigest()
    if report["inventory"]["inventory_identity_sha256"] != expected_inventory:
        raise ArchiveIntakeError("inventory identity mismatch")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--hf-snapshot", required=True, type=Path)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seven-zip", default="7z")
    parser.add_argument("--hash-members", action="store_true")
    parser.add_argument("--list-timeout-seconds", type=int, default=300)
    parser.add_argument("--extract-timeout-seconds", type=int, default=1800)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.archive.name != ARCHIVE_PATH:
        raise ArchiveIntakeError(f"expected local archive filename {ARCHIVE_PATH!r}")
    if not args.archive.is_file():
        raise ArchiveIntakeError("local archive does not exist or is not a regular file")
    expected_sha = args.expected_sha256.strip()
    if not HEX64.fullmatch(expected_sha):
        raise ArchiveIntakeError("--expected-sha256 must be lowercase 64-hex")

    hf_snapshot, hf_archive = load_hf_snapshot(args.hf_snapshot)
    archive_size = args.archive.stat().st_size
    if archive_size != hf_archive["size_bytes"]:
        raise ArchiveIntakeError(
            f"local archive size {archive_size} != pinned size {hf_archive['size_bytes']}"
        )
    actual_sha = sha256_file(args.archive)
    if actual_sha != expected_sha:
        raise ArchiveIntakeError("local archive SHA-256 does not match expected SHA-256")

    parent_config = hf_probe.parent_probe.load_and_validate()
    policy = parent_config["acquisition_policy"]
    listing = run_7z_listing(
        args.archive, seven_zip=args.seven_zip, timeout_seconds=args.list_timeout_seconds
    )
    members = parse_7z_slt(
        listing,
        max_single_member_bytes=policy["max_single_member_bytes"],
        max_total_uncompressed_bytes=policy["max_total_uncompressed_bytes"],
    )
    if args.hash_members:
        members = hash_extracted_members(
            args.archive,
            members,
            seven_zip=args.seven_zip,
            timeout_seconds=args.extract_timeout_seconds,
        )

    report = build_report(
        hf_snapshot=hf_snapshot,
        hf_archive=hf_archive,
        archive_sha256=actual_sha,
        archive_size_bytes=archive_size,
        members=members,
        hashes_verified=args.hash_members,
    )
    validate_report(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(serialized, encoding="utf-8")
    print("D03_RADA_TREES_ARCHIVE_INTAKE=PASS")
    print("REPORT_IDENTITY_SHA256=" + report["report_identity_sha256"])
    print("MEMBER_COUNT=" + str(report["inventory"]["regular_file_count"]))
    print("MEMBER_HASHES_VERIFIED=" + str(args.hash_members).lower())
    print("TRAINING_AUTHORIZED_BYTES=0")
    print("NEXT=CLASSIFY_PLAIN_TEXT_MEMBERS_AND_BIND_MEMBER_PROVENANCE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
