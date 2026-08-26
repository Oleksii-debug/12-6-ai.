#!/usr/bin/env python3
"""Fail-closed inventory probe for the Verkhovna Rada bulk laws-texts archive.

The probe deliberately stops before normalization, quality/privacy filtering,
cross-source deduplication, evaluation decontamination, split, tokenization,
packing, or training authorization. It turns a mutable upstream ZIP into an
exact content-addressed acquisition observation.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import stat
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, BinaryIO

SCHEMA = "12-6.d03-rada-bulk-source-probe-report.v1"
CONFIG_SCHEMA = "12-6.d03-rada-bulk-source-probe.v1"
DEFAULT_CONFIG = Path("configs/data/d03_rada_bulk_source_probe_v1.json")
CHUNK_BYTES = 1024 * 1024


class ProbeError(RuntimeError):
    """Raised when the upstream archive or probe input fails closed."""


def _sha256_stream(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    total = 0
    while True:
        chunk = stream.read(CHUNK_BYTES)
        if not chunk:
            break
        digest.update(chunk)
        total += len(chunk)
    return digest.hexdigest(), total


def _download(url: str, *, max_bytes: int) -> tuple[bytes, dict[str, str]]:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "12-6-ai-D03-RadaBulkProbe/1.0",
            "Accept": "application/zip,application/octet-stream,*/*;q=0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        content_length = response.headers.get("Content-Length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError as exc:
                raise ProbeError("invalid Content-Length from upstream") from exc
            if declared > max_bytes:
                raise ProbeError(
                    "archive Content-Length "
                    f"{declared} exceeds max_archive_bytes {max_bytes}"
                )
        data = response.read(max_bytes + 1)
        headers = {
            "etag": response.headers.get("ETag", ""),
            "last_modified": response.headers.get("Last-Modified", ""),
            "content_type": response.headers.get("Content-Type", ""),
        }
    if len(data) > max_bytes:
        raise ProbeError(f"downloaded archive exceeds max_archive_bytes {max_bytes}")
    return data, headers


def _zipinfo_is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def _safe_archive_name(name: str) -> bool:
    normalized = name.replace("\\", "/")
    if normalized.startswith("/"):
        return False
    parts = [part for part in normalized.split("/") if part not in ("", ".")]
    return ".." not in parts


def _load_config(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError(f"cannot load config: {path}") from exc
    if raw.get("schema_version") != CONFIG_SCHEMA:
        raise ProbeError("unsupported probe config schema")
    if raw.get("local_free_only") is not True:
        raise ProbeError("config must be LOCAL_FREE only")
    if raw.get("model_training_executed") is not False:
        raise ProbeError("config must bind model_training_executed=false")
    if raw.get("training_authorized_bytes") != 0:
        raise ProbeError("probe must authorize exactly zero training bytes")
    return raw


def inventory_archive(
    archive: bytes,
    config: dict[str, Any],
    *,
    expected_md5: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    """Hash and inventory a ZIP while retaining a zero-training truth boundary."""
    policy = config["probe_policy"]
    if len(archive) > int(policy["max_archive_bytes"]):
        raise ProbeError("archive exceeds configured size limit")
    if not archive.startswith(b"PK"):
        raise ProbeError("upstream payload is not a ZIP archive")

    archive_md5 = hashlib.md5(archive, usedforsecurity=False).hexdigest()
    archive_sha256 = hashlib.sha256(archive).hexdigest()
    if expected_bytes is not None and len(archive) != expected_bytes:
        raise ProbeError(
            f"archive byte identity drift: expected {expected_bytes}, got {len(archive)}"
        )
    if expected_md5 is not None and archive_md5.lower() != expected_md5.lower():
        raise ProbeError(
            f"archive MD5 identity drift: expected {expected_md5}, got {archive_md5}"
        )

    entry_re = re.compile(str(policy["canonical_entry_regex"]))
    max_entry_bytes = int(policy["max_entry_bytes"])
    max_total_uncompressed = int(policy["max_total_uncompressed_bytes"])
    min_canonical_entries = int(policy["min_canonical_entries"])

    rows: list[dict[str, Any]] = []
    seen_basenames: set[str] = set()
    total_uncompressed = 0
    ignored_files = 0

    try:
        archive_file = zipfile.ZipFile(io.BytesIO(archive))
    except zipfile.BadZipFile as exc:
        raise ProbeError("invalid ZIP archive") from exc

    with archive_file as zf:
        for info in zf.infolist():
            if not _safe_archive_name(info.filename):
                raise ProbeError(f"unsafe archive path: {info.filename!r}")
            if _zipinfo_is_symlink(info):
                raise ProbeError(f"symlink entry rejected: {info.filename!r}")
            if info.is_dir():
                continue
            if info.file_size > max_entry_bytes:
                raise ProbeError(
                    f"entry exceeds max_entry_bytes: {info.filename!r} {info.file_size}"
                )
            total_uncompressed += info.file_size
            if total_uncompressed > max_total_uncompressed:
                raise ProbeError("archive exceeds max_total_uncompressed_bytes")

            basename = Path(info.filename.replace("\\", "/")).name
            if not entry_re.fullmatch(basename):
                ignored_files += 1
                continue
            if basename in seen_basenames:
                raise ProbeError(f"duplicate canonical basename: {basename}")
            seen_basenames.add(basename)

            try:
                with zf.open(info, "r") as stream:
                    content_sha256, content_bytes = _sha256_stream(stream)
            except zipfile.BadZipFile as exc:
                raise ProbeError(f"ZIP integrity failure: {basename}") from exc
            if content_bytes != info.file_size:
                raise ProbeError(f"entry size drift while reading {basename}")
            rows.append(
                {
                    "path": info.filename.replace("\\", "/"),
                    "basename": basename,
                    "raw_bytes": content_bytes,
                    "raw_sha256": content_sha256,
                    "crc32": f"{info.CRC:08x}",
                }
            )

    if len(rows) < min_canonical_entries:
        raise ProbeError(
            f"canonical entry count {len(rows)} below minimum {min_canonical_entries}"
        )

    rows.sort(key=lambda row: row["basename"])
    identity = hashlib.sha256()
    for row in rows:
        identity.update(row["basename"].encode("utf-8"))
        identity.update(b"\0")
        identity.update(str(row["raw_bytes"]).encode("ascii"))
        identity.update(b"\0")
        identity.update(row["raw_sha256"].encode("ascii"))
        identity.update(b"\n")

    canonical_raw_bytes = sum(int(row["raw_bytes"]) for row in rows)
    return {
        "schema_version": SCHEMA,
        "worker_id": config["worker_id"],
        "source_family": config["source"]["family_id"],
        "source_dataset_id": config["source"]["dataset_id"],
        "archive": {
            "url": config["source"]["archive_url"],
            "bytes": len(archive),
            "md5": archive_md5,
            "sha256": archive_sha256,
        },
        "inventory": {
            "canonical_entry_count": len(rows),
            "canonical_raw_bytes": canonical_raw_bytes,
            "ignored_file_count": ignored_files,
            "total_zip_uncompressed_bytes": total_uncompressed,
            "entry_identity_sha256": identity.hexdigest(),
            "entries": rows,
        },
        "gates": {
            "exact_archive_identity": "PASS",
            "safe_zip_inventory": "PASS",
            "canonical_normalization": "NOT_RUN",
            "quality": "NOT_RUN",
            "privacy": "NOT_RUN",
            "global_cross_source_dedup": "NOT_RUN",
            "evaluation_decontamination": "NOT_RUN",
            "balance_diversity": "NOT_RUN",
            "corpus_materialization": "NOT_RUN",
            "unique_loss_ledger": "NOT_RUN",
        },
        "training_authorized_bytes": 0,
        "corpus_admitted": False,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "paid_compute_used": False,
        "safe_result": "EXACT_BULK_ARCHIVE_INVENTORIED_DOWNSTREAM_GATES_REQUIRED",
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--accept-current-upstream",
        action="store_true",
        help=(
            "Do not enforce the discovery-time MD5/byte observation. Use only "
            "for a new observation; review and pin the resulting SHA-256."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _load_config(args.config)
    policy = config["probe_policy"]

    response_headers: dict[str, str] = {}
    if args.archive is not None:
        try:
            archive = args.archive.read_bytes()
        except OSError as exc:
            raise ProbeError(f"cannot read archive: {args.archive}") from exc
    else:
        archive, response_headers = _download(
            config["source"]["archive_url"],
            max_bytes=int(policy["max_archive_bytes"]),
        )

    observed = config["discovery_observation"]
    expected_md5 = None if args.accept_current_upstream else str(observed["archive_md5"])
    expected_bytes = None if args.accept_current_upstream else int(observed["archive_bytes"])
    report = inventory_archive(
        archive,
        config,
        expected_md5=expected_md5,
        expected_bytes=expected_bytes,
    )
    report["http_response"] = response_headers
    report["discovery_observation_revalidated"] = not args.accept_current_upstream

    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
