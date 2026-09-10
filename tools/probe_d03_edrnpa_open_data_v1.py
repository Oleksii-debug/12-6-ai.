#!/usr/bin/env python3
"""Probe and materialize a zero-credit EDRNPA Ukrainian legal-text candidate."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import stat
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

SCHEMA = "12-6.d03-edrnpa-open-data-candidate.v1"
DATASET_ID = "c98e830c-e39e-4da6-a13c-f9ba32a79bec"
RESOURCE_ID = "5616dd04-949a-489c-8efc-54004293b238"
RESOURCE_UPDATED = "2026-09-08T15:02:00+03:00"
RESOURCE_MD5 = "0ea96e1582e5584ced79be1027f0ae55"
DATASET_PAGE = f"https://data.gov.ua/dataset/{DATASET_ID}"
RESOURCE_PAGE = (
    f"https://data.gov.ua/dataset/{DATASET_ID}/resource/{RESOURCE_ID}"
)
DOWNLOAD_URL = (
    "https://data.gov.ua/dataset/b22d184c-4826-4e1d-8577-972effc5700c/"
    f"resource/{RESOURCE_ID}/download/25-edrnpa.zip"
)
NAIS_METADATA_PAGE = (
    "https://nais.gov.ua/m/ediniy-derjavniy-reestr-normativno-pravovih-aktiv-196"
)
FAMILY_ID = "ua.minjust.edrnpa.open-data"
MAX_ARCHIVE_BYTES = 640 * 1024 * 1024
MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_TOTAL_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024
MAX_MEMBERS = 300_000
MAX_COMPRESSION_RATIO = 250.0
MAX_SELECTED_BYTES = 4_000_000
MIN_NORMALIZED_BYTES = 500
MIN_UKRAINIAN_ALPHA_RATIO = 0.55

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<!\d)(?:\+?38)?0\d{2}[\s().-]*\d{3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)")
UK_LETTER = re.compile(r"[А-Яа-яІіЇїЄєҐґ]")
ALPHA = re.compile(r"[A-Za-zА-Яа-яІіЇїЄєҐґ]")
XML_DANGER = (b"<!DOCTYPE", b"<!ENTITY")
BANNED_FILENAME_PARTS = {"schema", "xsd", "readme", "license", "passport"}


class ProbeError(RuntimeError):
    """Fail-closed EDRNPA probe error."""


def cjson(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _safe_member_name(name: str) -> str:
    normalized = name.replace("\\", "/")
    if normalized.startswith("/") or "\x00" in normalized:
        raise ProbeError("unsafe zip member path")
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        raise ProbeError("unsafe zip member path")
    return "/".join(parts)


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_ISLNK(mode)


def _eligible_xml_path(path: str) -> bool:
    if not path.lower().endswith(".xml"):
        return False
    stem = Path(path).stem.casefold()
    return not any(part in stem for part in BANNED_FILENAME_PARTS)


def _normalize_xml_text(raw: bytes, path: str) -> str:
    upper = raw[:200_000].upper()
    if any(marker in upper for marker in XML_DANGER):
        raise ProbeError(f"{path}: DTD/entity declarations forbidden")
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ProbeError(f"{path}: malformed XML") from exc
    chunks: list[str] = []
    for node in root.iter():
        if node.text and node.text.strip():
            chunks.append(node.text)
        if node.tail and node.tail.strip():
            chunks.append(node.tail)
    text = unicodedata.normalize("NFKC", "\n".join(chunks))
    lines = [" ".join(line.split()) for line in text.splitlines()]
    normalized = "\n".join(line for line in lines if line).strip()
    return normalized


def _quality(text: str) -> dict[str, Any]:
    alpha = ALPHA.findall(text)
    uk = UK_LETTER.findall(text)
    ratio = len(uk) / len(alpha) if alpha else 0.0
    return {
        "alphabetic_chars": len(alpha),
        "ukrainian_chars": len(uk),
        "ukrainian_alpha_ratio": round(ratio, 8),
        "has_email": bool(EMAIL.search(text)),
        "has_phone": bool(PHONE.search(text)),
        "replacement_characters": text.count("\ufffd"),
    }


def materialize_archive_bytes(
    archive_bytes: bytes,
    *,
    expected_md5: str = RESOURCE_MD5,
    byte_cap: int = MAX_SELECTED_BYTES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise ProbeError("archive exceeds compressed-byte safety limit")
    observed_md5 = md5(archive_bytes)
    if observed_md5 != expected_md5:
        raise ProbeError("EDRNPA resource MD5 mismatch")
    if byte_cap < 1 or byte_cap > MAX_SELECTED_BYTES:
        raise ProbeError("byte_cap outside allowed range")
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except zipfile.BadZipFile as exc:
        raise ProbeError("invalid ZIP archive") from exc
    candidates: list[dict[str, Any]] = []
    quarantined_sensitive = 0
    ignored_nontext = 0
    total_uncompressed = 0
    with archive:
        infos = archive.infolist()
        if len(infos) > MAX_MEMBERS:
            raise ProbeError("ZIP member-count safety limit exceeded")
        for info in infos:
            path = _safe_member_name(info.filename)
            if info.flag_bits & 0x1:
                raise ProbeError(f"{path}: encrypted ZIP member forbidden")
            if _is_symlink(info):
                raise ProbeError(f"{path}: symlink ZIP member forbidden")
            if info.is_dir():
                continue
            if info.file_size < 0 or info.file_size > MAX_MEMBER_BYTES:
                raise ProbeError(f"{path}: member size safety limit exceeded")
            total_uncompressed += info.file_size
            if total_uncompressed > MAX_TOTAL_UNCOMPRESSED_BYTES:
                raise ProbeError("ZIP total-uncompressed safety limit exceeded")
            if info.compress_size > 0:
                ratio = info.file_size / info.compress_size
                if ratio > MAX_COMPRESSION_RATIO:
                    raise ProbeError(f"{path}: suspicious ZIP compression ratio")
            if not _eligible_xml_path(path):
                ignored_nontext += 1
                continue
            raw = archive.read(info)
            if len(raw) != info.file_size:
                raise ProbeError(f"{path}: member size mismatch")
            text = _normalize_xml_text(raw, path)
            normalized_bytes = text.encode("utf-8")
            if len(normalized_bytes) < MIN_NORMALIZED_BYTES:
                continue
            quality = _quality(text)
            if quality["replacement_characters"] != 0:
                raise ProbeError(f"{path}: Unicode replacement character")
            if quality["ukrainian_alpha_ratio"] < MIN_UKRAINIAN_ALPHA_RATIO:
                continue
            if quality["has_email"] or quality["has_phone"]:
                quarantined_sensitive += 1
                continue
            candidates.append(
                {
                    "record_id": f"edrnpa:{sha256(raw)[:24]}",
                    "source_path": path,
                    "family": FAMILY_ID,
                    "modality": "text",
                    "raw_xml_sha256": sha256(raw),
                    "normalized_sha256": sha256(normalized_bytes),
                    "normalized_bytes": len(normalized_bytes),
                    "quality": quality,
                    "text": text,
                    "training_eligible": False,
                    "evaluation_eligible": False,
                }
            )
    if not candidates:
        raise ProbeError("no eligible Ukrainian legal-text candidates")
    raw_ids = [row["raw_xml_sha256"] for row in candidates]
    norm_ids = [row["normalized_sha256"] for row in candidates]
    if len(raw_ids) != len(set(raw_ids)):
        raise ProbeError("exact duplicate raw XML candidate")
    if len(norm_ids) != len(set(norm_ids)):
        raise ProbeError("exact duplicate normalized candidate")
    candidates.sort(key=lambda row: (row["normalized_sha256"], row["source_path"]))
    selected: list[dict[str, Any]] = []
    selected_bytes = 0
    skipped_for_cap = 0
    for row in candidates:
        size = int(row["normalized_bytes"])
        if selected_bytes + size > byte_cap:
            skipped_for_cap += 1
            continue
        selected.append(row)
        selected_bytes += size
    if not selected:
        raise ProbeError("byte cap retained no EDRNPA candidates")
    projection = [
        {key: value for key, value in row.items() if key != "text"}
        for row in selected
    ]
    report: dict[str, Any] = {
        "schema_version": SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "source": {
            "dataset_id": DATASET_ID,
            "resource_id": RESOURCE_ID,
            "resource_updated": RESOURCE_UPDATED,
            "resource_md5": observed_md5,
            "dataset_page": DATASET_PAGE,
            "resource_page": RESOURCE_PAGE,
            "download_url": DOWNLOAD_URL,
            "nais_metadata_page": NAIS_METADATA_PAGE,
            "publisher": "Міністерство юстиції України",
            "language": "uk",
            "format": "XML-in-ZIP",
            "family_id": FAMILY_ID,
        },
        "rights": {
            "portal_license": "CC-BY-4.0",
            "open_data_free_reuse": True,
            "attribution_required": True,
            "training_rights_admitted_by_this_probe": False,
        },
        "selection": {
            "policy": "EDRNPA_UK_XML_TEXT_ZERO_CREDIT_V1",
            "byte_cap": byte_cap,
            "candidate_objects_after_quality": len(candidates),
            "selected_objects": len(selected),
            "selected_bytes": selected_bytes,
            "quarantined_email_or_phone": quarantined_sensitive,
            "ignored_nontext_or_schema_members": ignored_nontext,
            "skipped_for_cap": skipped_for_cap,
            "inventory_identity_sha256": sha256(cjson(projection)),
        },
        "truth_boundary": {
            "status": "CANDIDATE_MATERIALIZED_ZERO_CREDIT",
            "canonical_corpus_admitted": False,
            "canonical_capacity_credit_bytes": 0,
            "family_count_credit_added": 0,
            "training_authorized_bytes": 0,
            "unique_causal_loss_positions": 0,
            "tokenizer_fit_executed": False,
            "optimizer_updates": 0,
            "model_training_executed": False,
            "final_test_accessed": False,
            "paid_compute_used": False,
        },
        "mandatory_successors": [
            "object_level_rights_and_provenance_retest",
            "global_cross_source_dedup",
            "fresh_reserved_evaluation_decontamination",
            "post_composition_quality_and_privacy",
            "balance_and_family_caps",
            "cluster_safe_split",
            "deterministic_packing_and_two_clean_builds",
            "positive_exact_unique_causal_loss_ledger",
        ],
    }
    report["report_identity_sha256"] = sha256(cjson(report))
    return selected, report


def download_archive(url: str = DOWNLOAD_URL, *, timeout: float = 90.0) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "12-6-ai-edrnpa-open-data/1.0 (+LOCAL_FREE research)",
            "Accept": "application/zip,application/octet-stream",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > MAX_ARCHIVE_BYTES:
            raise ProbeError("remote archive exceeds compressed-byte limit")
        data = response.read(MAX_ARCHIVE_BYTES + 1)
    if len(data) > MAX_ARCHIVE_BYTES:
        raise ProbeError("remote archive exceeds compressed-byte limit")
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--byte-cap", type=int, default=MAX_SELECTED_BYTES)
    args = parser.parse_args()
    archive_bytes = args.archive.read_bytes() if args.archive else download_archive()
    rows, report = materialize_archive_bytes(archive_bytes, byte_cap=args.byte_cap)
    args.records.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.records.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    args.report.write_bytes(cjson(report))
    print(
        f"{report['truth_boundary']['status']} "
        f"objects={report['selection']['selected_objects']} "
        f"bytes={report['selection']['selected_bytes']} "
        f"report={report['report_identity_sha256']}"
    )


if __name__ == "__main__":
    main()
