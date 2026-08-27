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

EXPECTED_WORKER_ID = "D03-RADA-BULK-SOURCE-PROBE-20260826"
EXPECTED_PARENT = {
    "name": "DATA-287-EXTERNAL-SNAPSHOT-REGISTRY-V2",
    "head_sha": "b0523ccbc4b957615aac849d476cfa851be87578",
    "registry_identity_sha256": (
        "917e9bc31b2fa040d25e807ae3c01aa2cce32420752a891caacfb6c830e6632c"
    ),
    "existing_source_id": "ua.rada.open-data.laws-texts.d23314",
    "existing_family_id": "ua.rada.open-data.laws-texts",
    "existing_family_identity_sha256": (
        "b8f1d2f99a3db71d894a3233e9417d6283d11768c41b1634bc8b096ab77aba4e"
    ),
    "existing_normalized_bytes": 88565,
    "existing_rights_model_training": "ALLOWED",
}
EXPECTED_SOURCE = {
    "dataset_id": "laws-texts",
    "title": "Тексти первинних законів бази даних \"Законодавство України\"",
    "publisher": "Апарат Верховної Ради України",
    "page_url": "https://data.rada.gov.ua/open/data/en/laws-texts",
    "archive_url": "https://data.rada.gov.ua/ogd/zak/perv/text/texts.zip",
    "family_id": "ua.rada.open-data.laws-texts",
    "language": "uk",
    "modality": "text",
    "upstream_mutability": "FREQUENTLY_UPDATED_REQUIRES_EXACT_ARCHIVE_REPIN",
}
EXPECTED_DISCOVERY = {
    "observed_on_utc": "2026-08-26",
    "portal_publication_date": "2026-08-07T16:58:14+03:00",
    "portal_file_count": 5926,
    "archive_bytes": 46682933,
    "archive_md5": "06f239fd182e580ce22ab00dce867e31",
    "status": "DISCOVERY_ONLY_REVALIDATE_AT_PROBE",
}
EXPECTED_RIGHTS = {
    "portal_terms": "OPEN_DATA_REUSE_INCLUDING_COMMERCIAL_WITH_SOURCE_ATTRIBUTION",
    "portal_default_license": "CC-BY-4.0_UNLESS_OTHERWISE_SPECIFIED",
    "existing_project_family_training_authority": (
        "DATA-287 existing bounded Rada snapshot is ALLOWED for model training"
    ),
    "bulk_extension_status": "NOT_ADMITTED_BY_THIS_PROBE",
    "evaluation": "NOT_SEPARATELY_ADMITTED",
    "final_test": "PROHIBITED",
    "redistribution": "REQUIRES_SOURCE_ATTRIBUTION_AND_SUCCESSOR_RIGHTS_RECHECK",
}
EXPECTED_POLICY = {
    "canonical_entry_regex": r"d[0-9]+\.htm",
    "min_canonical_entries": 5000,
    "max_archive_bytes": 100000000,
    "max_entry_bytes": 20000000,
    "max_total_uncompressed_bytes": 1000000000,
    "hash_every_canonical_entry": True,
    "reject_path_traversal": True,
    "reject_symlinks": True,
    "require_unique_canonical_basenames": True,
}
EXPECTED_DOWNSTREAM = [
    "EXACT_ARCHIVE_SHA256_PIN_AND_SOURCE_MANIFEST",
    "CANONICAL_HTML_EXTRACTION_AND_NORMALIZATION",
    "QUALITY_FILTER",
    "PRIVACY_PII_FILTER",
    "GLOBAL_CROSS_SOURCE_EXACT_NEAR_DEDUP",
    "EVALUATION_DECONTAMINATION",
    "BALANCE_DIVERSITY_AND_FAMILY_CAP_RETEST",
    "DETERMINISTIC_SPLIT_SHARD_PACK",
    "UNIQUE_CAUSAL_LOSS_LEDGER",
    "TOKENIZER_FIT_AUTHORIZATION",
    "LEARNED_20M_COMPUTE_AUTHORIZATION",
]
EXPECTED_CLAIM = {
    "bulk_source_admitted": False,
    "normalized_capacity_claimed": False,
    "training_exposure_authorized": False,
    "research_corpus_v1_released": False,
    "learned_20m_claimed": False,
    "safe_result": "PROBE_ONLY",
}


class ProbeError(RuntimeError):
    """Raised when the upstream archive or probe input fails closed."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _config_identity(config: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json_bytes(config)).hexdigest()


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
    if url != EXPECTED_SOURCE["archive_url"]:
        raise ProbeError("archive URL is not the pinned Rada bulk endpoint")
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "12-6-ai-D03-RadaBulkProbe/1.0",
            "Accept": "application/zip,application/octet-stream,*/*;q=0.1",
        },
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        final_url = response.geturl()
        if final_url != EXPECTED_SOURCE["archive_url"]:
            raise ProbeError(f"unexpected archive redirect target: {final_url!r}")
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


def _require_exact_mapping(
    raw: dict[str, Any],
    field: str,
    expected: dict[str, Any],
) -> None:
    value = raw.get(field)
    if not isinstance(value, dict):
        raise ProbeError(f"{field} must be an object")
    for key, expected_value in expected.items():
        if value.get(key) != expected_value:
            raise ProbeError(f"{field}.{key} drifted from the pinned v1 authority")


def _validate_config(raw: dict[str, Any]) -> None:
    if raw.get("schema_version") != CONFIG_SCHEMA:
        raise ProbeError("unsupported probe config schema")
    if raw.get("worker_id") != EXPECTED_WORKER_ID:
        raise ProbeError("worker_id drifted from the pinned v1 authority")
    if raw.get("local_free_only") is not True:
        raise ProbeError("config must be LOCAL_FREE only")
    for field in ("model_training_executed", "tokenizer_fit_executed", "paid_compute_used"):
        if raw.get(field) is not False:
            raise ProbeError(f"config must bind {field}=false")
    if raw.get("training_authorized_bytes") != 0:
        raise ProbeError("probe must authorize exactly zero training bytes")

    _require_exact_mapping(raw, "parent_authority", EXPECTED_PARENT)
    _require_exact_mapping(raw, "source", EXPECTED_SOURCE)
    _require_exact_mapping(raw, "discovery_observation", EXPECTED_DISCOVERY)
    _require_exact_mapping(raw, "rights_boundary", EXPECTED_RIGHTS)
    _require_exact_mapping(raw, "probe_policy", EXPECTED_POLICY)

    if raw.get("downstream_required") != EXPECTED_DOWNSTREAM:
        raise ProbeError("downstream_required drifted from the pinned v1 authority")
    _require_exact_mapping(raw, "claim_boundary", EXPECTED_CLAIM)


def _load_config(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProbeError(f"cannot load config: {path}") from exc
    if not isinstance(raw, dict):
        raise ProbeError("probe config root must be an object")
    _validate_config(raw)
    return raw


def inventory_archive(
    archive: bytes,
    config: dict[str, Any],
    *,
    expected_md5: str | None = None,
    expected_bytes: int | None = None,
) -> dict[str, Any]:
    """Hash and inventory a ZIP while retaining a zero-training truth boundary."""
    if (expected_md5 is None) != (expected_bytes is None):
        raise ProbeError("expected_md5 and expected_bytes must be supplied together")
    strict_revalidation = expected_md5 is not None and expected_bytes is not None

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
    identity_gate = (
        "PASS_PINNED_DISCOVERY_REVALIDATED" if strict_revalidation else "OBSERVED_UNPINNED"
    )
    safe_result = (
        "PINNED_BULK_ARCHIVE_INVENTORIED_DOWNSTREAM_GATES_REQUIRED"
        if strict_revalidation
        else "CURRENT_UPSTREAM_OBSERVED_SUCCESSOR_PIN_REQUIRED"
    )
    return {
        "schema_version": SCHEMA,
        "worker_id": config["worker_id"],
        "config_identity_sha256": _config_identity(config),
        "parent_authority": {
            "head_sha": config["parent_authority"]["head_sha"],
            "registry_identity_sha256": config["parent_authority"][
                "registry_identity_sha256"
            ],
        },
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
            "exact_archive_identity": identity_gate,
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
        "safe_result": safe_result,
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
    strict_revalidation = not args.accept_current_upstream
    expected_md5 = str(observed["archive_md5"]) if strict_revalidation else None
    expected_bytes = int(observed["archive_bytes"]) if strict_revalidation else None
    report = inventory_archive(
        archive,
        config,
        expected_md5=expected_md5,
        expected_bytes=expected_bytes,
    )
    report["http_response"] = response_headers
    report["discovery_observation_revalidated"] = strict_revalidation

    encoded = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()