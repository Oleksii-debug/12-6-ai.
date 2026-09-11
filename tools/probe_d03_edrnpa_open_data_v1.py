#!/usr/bin/env python3
"""Materialize a bounded zero-credit EDRNPA Ukrainian legal-text candidate."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import stat
import tempfile
import unicodedata
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, BinaryIO

SCHEMA = "12-6.d03-edrnpa-open-data-candidate.v2"
DATASET_ID = "c98e830c-e39e-4da6-a13c-f9ba32a79bec"
RESOURCE_ID = "5616dd04-949a-489c-8efc-54004293b238"
RESOURCE_UPDATED = "2026-09-08T15:02:00+03:00"
RESOURCE_MD5 = "0ea96e1582e5584ced79be1027f0ae55"
DATASET_PAGE = f"https://data.gov.ua/dataset/{DATASET_ID}"
RESOURCE_PAGE = f"https://data.gov.ua/dataset/{DATASET_ID}/resource/{RESOURCE_ID}"
DOWNLOAD_URL = (
    "https://data.gov.ua/dataset/b22d184c-4826-4e1d-8577-972effc5700c/"
    f"resource/{RESOURCE_ID}/download/25-edrnpa.zip"
)
NAIS_METADATA_PAGE = (
    "https://nais.gov.ua/m/ediniy-derjavniy-reestr-normativno-pravovih-aktiv-196"
)
FAMILY_ID = "ua.minjust.edrnpa.open-data"
NESTED_TEXT_ZIP = "25.2-edrnpa_text.zip"

MAX_ARCHIVE_BYTES = 640 * 1024 * 1024
MAX_OUTER_MEMBERS = 128
MAX_NESTED_ZIP_BYTES = 640 * 1024 * 1024
MAX_NESTED_MEMBERS = 8
MAX_NESTED_XML_BYTES = 4 * 1024 * 1024 * 1024
MAX_COMPRESSION_RATIO = 32.0
MAX_SELECTED_BYTES = 4_000_000
MIN_NORMALIZED_BYTES = 500
MIN_UKRAINIAN_ALPHA_RATIO = 0.55
MAX_DOCUMENT_CHARS = 8_000_000
MAX_PARAGRAPH_CHARS = 1_000_000

EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE = re.compile(r"(?<!\d)(?:\+?38)?0\d{2}[\s().-]*\d{3}[\s.-]*\d{2}[\s.-]*\d{2}(?!\d)")
UK_LETTER = re.compile(r"[А-Яа-яІіЇїЄєҐґ]")
ALPHA = re.compile(r"[A-Za-zА-Яа-яІіЇїЄєҐґ]")
XML_DANGER = (b"<!DOCTYPE", b"<!ENTITY")


class ProbeError(RuntimeError):
    """Fail-closed EDRNPA materializer error."""


def cjson(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def md5(data: bytes) -> str:
    return hashlib.md5(data, usedforsecurity=False).hexdigest()


def _hash_file(path: Path) -> tuple[int, str, str]:
    md5_hash = hashlib.md5(usedforsecurity=False)
    sha_hash = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            if size > MAX_ARCHIVE_BYTES:
                raise ProbeError("archive exceeds compressed-byte safety limit")
            md5_hash.update(chunk)
            sha_hash.update(chunk)
    return size, md5_hash.hexdigest(), sha_hash.hexdigest()


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


def _validate_zip_member(info: zipfile.ZipInfo, *, max_size: int) -> str:
    path = _safe_member_name(info.filename)
    if info.flag_bits & 0x1:
        raise ProbeError(f"{path}: encrypted ZIP member forbidden")
    if _is_symlink(info):
        raise ProbeError(f"{path}: symlink ZIP member forbidden")
    if info.file_size < 0 or info.file_size > max_size:
        raise ProbeError(f"{path}: member size safety limit exceeded")
    if info.compress_size < 0:
        raise ProbeError(f"{path}: invalid compressed size")
    if info.file_size and info.compress_size == 0:
        raise ProbeError(f"{path}: invalid zero compressed size")
    if info.compress_size > 0 and info.file_size / info.compress_size > MAX_COMPRESSION_RATIO:
        raise ProbeError(f"{path}: suspicious ZIP compression ratio")
    return path


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _normalize_text(chunks: list[str]) -> str:
    text = unicodedata.normalize("NFKC", "\n".join(chunks))
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line).strip()


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


class _GuardedXMLReader:
    """Stream wrapper that enforces exact size and rejects DTD/entity markers."""

    def __init__(self, source: BinaryIO, *, expected_size: int) -> None:
        self._source = source
        self._expected_size = expected_size
        self._seen = 0
        self._tail = b""
        self._max_marker = max(len(marker) for marker in XML_DANGER)

    @property
    def bytes_seen(self) -> int:
        return self._seen

    def read(self, size: int = -1) -> bytes:
        chunk = self._source.read(size)
        if chunk:
            self._seen += len(chunk)
            if self._seen > self._expected_size:
                raise ProbeError("nested XML exceeded declared member size")
            probe = (self._tail + chunk).upper()
            if any(marker in probe for marker in XML_DANGER):
                raise ProbeError("nested XML DTD/entity declarations forbidden")
            self._tail = probe[-(self._max_marker - 1) :]
            return chunk
        if self._seen != self._expected_size:
            raise ProbeError("nested XML byte count mismatch")
        return b""


def download_archive_to(
    path: Path,
    url: str = DOWNLOAD_URL,
    *,
    timeout: float = 240.0,
) -> dict[str, Any]:
    """Download the exact pinned source to disk with bounded streaming hashes."""
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "12-6-ai-edrnpa-open-data/2.0 (+LOCAL_FREE research)",
            "Accept": "application/zip,application/octet-stream",
        },
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    md5_hash = hashlib.md5(usedforsecurity=False)
    sha_hash = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(request, timeout=timeout) as response, path.open("wb") as output:
        length = response.headers.get("Content-Length")
        if length is not None and int(length) > MAX_ARCHIVE_BYTES:
            raise ProbeError("remote archive exceeds compressed-byte limit")
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_ARCHIVE_BYTES:
                raise ProbeError("remote archive exceeds compressed-byte limit")
            output.write(chunk)
            md5_hash.update(chunk)
            sha_hash.update(chunk)
    observed_md5 = md5_hash.hexdigest()
    if observed_md5 != RESOURCE_MD5:
        path.unlink(missing_ok=True)
        raise ProbeError("EDRNPA resource MD5 mismatch")
    return {
        "bytes": total,
        "md5": observed_md5,
        "sha256": sha_hash.hexdigest(),
    }


def download_archive(url: str = DOWNLOAD_URL, *, timeout: float = 240.0) -> bytes:
    """Compatibility helper for small tests/tools; real execution should use disk streaming."""
    with tempfile.TemporaryDirectory(prefix="edrnpa-download-") as tmp:
        path = Path(tmp) / "source.zip"
        download_archive_to(path, url=url, timeout=timeout)
        return path.read_bytes()


def _extract_nested_zip(source_path: Path, target_path: Path) -> tuple[str, int]:
    try:
        outer = zipfile.ZipFile(source_path)
    except zipfile.BadZipFile as exc:
        raise ProbeError("invalid outer ZIP archive") from exc
    with outer:
        infos = outer.infolist()
        if len(infos) > MAX_OUTER_MEMBERS:
            raise ProbeError("outer ZIP member-count safety limit exceeded")
        target_info: zipfile.ZipInfo | None = None
        for info in infos:
            path = _validate_zip_member(info, max_size=MAX_NESTED_ZIP_BYTES)
            if info.is_dir():
                continue
            if path == NESTED_TEXT_ZIP:
                if target_info is not None:
                    raise ProbeError("duplicate nested text ZIP member")
                target_info = info
        if target_info is None:
            raise ProbeError("pinned nested text ZIP member missing")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        copied = 0
        with outer.open(target_info) as src, target_path.open("wb") as dst:
            while True:
                chunk = src.read(1024 * 1024)
                if not chunk:
                    break
                copied += len(chunk)
                if copied > MAX_NESTED_ZIP_BYTES:
                    raise ProbeError("nested text ZIP exceeds byte safety limit")
                dst.write(chunk)
        if copied != target_info.file_size:
            raise ProbeError("nested text ZIP byte count mismatch")
    return NESTED_TEXT_ZIP, copied


def _nested_xml_info(nested_path: Path) -> tuple[zipfile.ZipFile, zipfile.ZipInfo, str]:
    try:
        nested = zipfile.ZipFile(nested_path)
    except zipfile.BadZipFile as exc:
        raise ProbeError("invalid nested text ZIP") from exc
    infos = nested.infolist()
    if len(infos) > MAX_NESTED_MEMBERS:
        nested.close()
        raise ProbeError("nested ZIP member-count safety limit exceeded")
    eligible: list[tuple[zipfile.ZipInfo, str]] = []
    try:
        for info in infos:
            path = _validate_zip_member(info, max_size=MAX_NESTED_XML_BYTES)
            if info.is_dir():
                continue
            if path.casefold().endswith(".xml"):
                eligible.append((info, path))
        if len(eligible) != 1:
            raise ProbeError("nested text ZIP must contain exactly one XML payload")
    except Exception:
        nested.close()
        raise
    info, path = eligible[0]
    return nested, info, path


def _create_candidate_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=OFF")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=FILE")
    conn.execute(
        """
        CREATE TABLE candidates (
            normalized_sha256 TEXT PRIMARY KEY,
            source_ordinal INTEGER NOT NULL,
            source_object_identity_sha256 TEXT NOT NULL,
            normalized_bytes INTEGER NOT NULL,
            quality_json TEXT NOT NULL,
            text TEXT NOT NULL
        )
        """
    )
    return conn


def _candidate_identity(
    *,
    source_sha256: str,
    nested_xml_path: str,
    source_ordinal: int,
    normalized_sha256: str,
) -> str:
    return sha256(
        cjson(
            {
                "source_sha256": source_sha256,
                "nested_xml_path": nested_xml_path,
                "document_ordinal": source_ordinal,
                "normalized_sha256": normalized_sha256,
            }
        )
    )


def _stream_documents_to_db(
    nested: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    xml_path: str,
    conn: sqlite3.Connection,
    *,
    source_sha256: str,
    byte_cap: int,
) -> dict[str, int]:
    counters = {
        "source_documents": 0,
        "documents_without_text": 0,
        "documents_too_short": 0,
        "documents_too_large": 0,
        "documents_non_ukrainian": 0,
        "quarantined_email_or_phone": 0,
        "duplicate_normalized_objects": 0,
        "candidate_objects_after_quality": 0,
    }
    stack: list[str] = []
    root_tag: str | None = None
    database_depth = 0
    current_chunks: list[str] | None = None
    current_chars = 0
    current_has_par = False
    current_ordinal = 0

    with nested.open(info) as raw_stream:
        guarded = _GuardedXMLReader(raw_stream, expected_size=info.file_size)
        try:
            iterator = ET.iterparse(guarded, events=("start", "end"))
            for event, elem in iterator:
                tag = _local_tag(elem.tag)
                if event == "start":
                    stack.append(tag)
                    if root_tag is None:
                        root_tag = tag
                        if root_tag != "rna":
                            raise ProbeError("unexpected EDRNPA XML root")
                    if tag == "database":
                        database_depth += 1
                    elif tag == "document":
                        if current_chunks is not None:
                            raise ProbeError("nested document elements forbidden")
                        if database_depth < 1:
                            raise ProbeError("document outside database container")
                        current_chunks = []
                        current_chars = 0
                        current_has_par = False
                        current_ordinal += 1
                    continue

                if not stack or stack[-1] != tag:
                    raise ProbeError("malformed XML element stack")
                if current_chunks is not None and tag == "par" and "text" in stack:
                    chunk = "".join(elem.itertext())
                    if len(chunk) > MAX_PARAGRAPH_CHARS:
                        current_chars = MAX_DOCUMENT_CHARS + 1
                    elif chunk.strip():
                        current_chunks.append(chunk)
                        current_chars += len(chunk)
                    current_has_par = True
                    elem.clear()
                elif current_chunks is not None and tag == "text" and not current_has_par:
                    chunk = "".join(elem.itertext())
                    if len(chunk) > MAX_DOCUMENT_CHARS:
                        current_chars = MAX_DOCUMENT_CHARS + 1
                    elif chunk.strip():
                        current_chunks.append(chunk)
                        current_chars += len(chunk)
                    elem.clear()
                elif tag == "document":
                    counters["source_documents"] += 1
                    if current_chunks is None:
                        raise ProbeError("document finalization state missing")
                    if current_chars > MAX_DOCUMENT_CHARS:
                        counters["documents_too_large"] += 1
                    else:
                        text = _normalize_text(current_chunks)
                        normalized_bytes = text.encode("utf-8")
                        if not text:
                            counters["documents_without_text"] += 1
                        elif len(normalized_bytes) < MIN_NORMALIZED_BYTES:
                            counters["documents_too_short"] += 1
                        elif len(normalized_bytes) > byte_cap:
                            counters["documents_too_large"] += 1
                        else:
                            quality = _quality(text)
                            if quality["replacement_characters"] != 0:
                                raise ProbeError("Unicode replacement character in candidate")
                            if quality["ukrainian_alpha_ratio"] < MIN_UKRAINIAN_ALPHA_RATIO:
                                counters["documents_non_ukrainian"] += 1
                            elif quality["has_email"] or quality["has_phone"]:
                                counters["quarantined_email_or_phone"] += 1
                            else:
                                normalized_sha = sha256(normalized_bytes)
                                identity = _candidate_identity(
                                    source_sha256=source_sha256,
                                    nested_xml_path=xml_path,
                                    source_ordinal=current_ordinal,
                                    normalized_sha256=normalized_sha,
                                )
                                cursor = conn.execute(
                                    """
                                    INSERT OR IGNORE INTO candidates (
                                        normalized_sha256,
                                        source_ordinal,
                                        source_object_identity_sha256,
                                        normalized_bytes,
                                        quality_json,
                                        text
                                    ) VALUES (?, ?, ?, ?, ?, ?)
                                    """,
                                    (
                                        normalized_sha,
                                        current_ordinal,
                                        identity,
                                        len(normalized_bytes),
                                        json.dumps(quality, sort_keys=True, separators=(",", ":")),
                                        text,
                                    ),
                                )
                                if cursor.rowcount == 0:
                                    counters["duplicate_normalized_objects"] += 1
                                else:
                                    counters["candidate_objects_after_quality"] += 1
                    current_chunks = None
                    current_chars = 0
                    current_has_par = False
                    elem.clear()
                elif tag == "database":
                    database_depth -= 1
                    if database_depth < 0:
                        raise ProbeError("database structure underflow")
                    elem.clear()
                stack.pop()
        except ET.ParseError as exc:
            raise ProbeError("malformed nested EDRNPA XML") from exc
        if guarded.bytes_seen != info.file_size:
            raise ProbeError("nested XML was not fully consumed")
    if stack:
        raise ProbeError("unterminated XML element stack")
    if root_tag != "rna" or database_depth != 0:
        raise ProbeError("unexpected EDRNPA XML structure")
    if counters["source_documents"] == 0:
        raise ProbeError("no document objects in EDRNPA XML")
    if counters["candidate_objects_after_quality"] == 0:
        raise ProbeError("no eligible Ukrainian legal-text candidates")
    conn.commit()
    return counters


def materialize_archive_path(
    archive_path: Path,
    *,
    expected_md5: str = RESOURCE_MD5,
    byte_cap: int = MAX_SELECTED_BYTES,
    work_dir: Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Materialize the exact nested EDRNPA source without loading multi-GB XML in memory."""
    if byte_cap < 1 or byte_cap > MAX_SELECTED_BYTES:
        raise ProbeError("byte_cap outside allowed range")
    archive_size, observed_md5, source_sha256 = _hash_file(archive_path)
    if observed_md5 != expected_md5:
        raise ProbeError("EDRNPA resource MD5 mismatch")

    temp_ctx: tempfile.TemporaryDirectory[str] | None = None
    if work_dir is None:
        temp_ctx = tempfile.TemporaryDirectory(prefix="edrnpa-materialize-")
        root = Path(temp_ctx.name)
    else:
        root = work_dir
        root.mkdir(parents=True, exist_ok=True)

    try:
        nested_path = root / "nested-text.zip"
        nested_name, nested_size = _extract_nested_zip(archive_path, nested_path)
        nested, xml_info, xml_path = _nested_xml_info(nested_path)
        db_path = root / "candidates.sqlite3"
        conn = _create_candidate_db(db_path)
        try:
            with nested:
                counters = _stream_documents_to_db(
                    nested,
                    xml_info,
                    xml_path,
                    conn,
                    source_sha256=source_sha256,
                    byte_cap=byte_cap,
                )
            selected: list[dict[str, Any]] = []
            selected_bytes = 0
            skipped_for_cap = 0
            cursor = conn.execute(
                """
                SELECT normalized_sha256, source_ordinal, source_object_identity_sha256,
                       normalized_bytes, quality_json, text
                FROM candidates
                ORDER BY normalized_sha256, source_ordinal
                """
            )
            for norm_sha, ordinal, identity, size, quality_json, text in cursor:
                size = int(size)
                if selected_bytes + size > byte_cap:
                    skipped_for_cap += 1
                    continue
                selected.append(
                    {
                        "record_id": f"edrnpa:{identity[:24]}",
                        "source_path": f"{nested_name}!/{xml_path}#document:{ordinal}",
                        "source_object_identity_sha256": identity,
                        "family": FAMILY_ID,
                        "modality": "text",
                        "normalized_sha256": norm_sha,
                        "normalized_bytes": size,
                        "quality": json.loads(quality_json),
                        "text": text,
                        "training_eligible": False,
                        "evaluation_eligible": False,
                    }
                )
                selected_bytes += size
            if not selected:
                raise ProbeError("byte cap retained no EDRNPA candidates")
        finally:
            conn.close()

        projection = [{k: v for k, v in row.items() if k != "text"} for row in selected]
        report: dict[str, Any] = {
            "schema_version": SCHEMA,
            "execution_profile": "LOCAL_FREE",
            "source": {
                "dataset_id": DATASET_ID,
                "resource_id": RESOURCE_ID,
                "resource_updated": RESOURCE_UPDATED,
                "resource_md5": observed_md5,
                "source_sha256": source_sha256,
                "source_bytes": archive_size,
                "dataset_page": DATASET_PAGE,
                "resource_page": RESOURCE_PAGE,
                "download_url": DOWNLOAD_URL,
                "nais_metadata_page": NAIS_METADATA_PAGE,
                "publisher": "Міністерство юстиції України",
                "language": "uk",
                "format": "XML-in-ZIP-in-ZIP",
                "nested_zip_member": nested_name,
                "nested_zip_bytes": nested_size,
                "nested_xml_member": xml_path,
                "nested_xml_bytes": xml_info.file_size,
                "family_id": FAMILY_ID,
            },
            "rights": {
                "portal_license": "CC-BY-4.0",
                "open_data_free_reuse": True,
                "attribution_required": True,
                "object_level_rights_and_provenance_retest_required": True,
                "training_rights_admitted_by_this_probe": False,
            },
            "selection": {
                "policy": "EDRNPA_UK_DOCUMENT_TEXT_ZERO_CREDIT_V2",
                "byte_cap": byte_cap,
                **counters,
                "selected_objects": len(selected),
                "selected_bytes": selected_bytes,
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
                "authorized_optimized_target_exposure": 0,
                "tokenizer_fit_executed": False,
                "optimizer_updates": 0,
                "model_training_executed": False,
                "final_test_accessed": False,
                "paid_compute_used": False,
                "foreign_pretrained_weights_used": False,
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
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


def materialize_archive_bytes(
    archive_bytes: bytes,
    *,
    expected_md5: str = RESOURCE_MD5,
    byte_cap: int = MAX_SELECTED_BYTES,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(archive_bytes) > MAX_ARCHIVE_BYTES:
        raise ProbeError("archive exceeds compressed-byte safety limit")
    with tempfile.TemporaryDirectory(prefix="edrnpa-bytes-") as tmp:
        archive_path = Path(tmp) / "source.zip"
        archive_path.write_bytes(archive_bytes)
        return materialize_archive_path(
            archive_path,
            expected_md5=expected_md5,
            byte_cap=byte_cap,
            work_dir=Path(tmp) / "work",
        )


def _write_outputs(
    rows: list[dict[str, Any]],
    report: dict[str, Any],
    *,
    records_path: Path,
    report_path: Path,
) -> None:
    records_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with records_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    report_path.write_bytes(cjson(report))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--byte-cap", type=int, default=MAX_SELECTED_BYTES)
    parser.add_argument("--timeout", type=float, default=240.0)
    args = parser.parse_args()

    if args.archive is not None:
        rows, report = materialize_archive_path(args.archive, byte_cap=args.byte_cap)
    else:
        with tempfile.TemporaryDirectory(prefix="edrnpa-real-") as tmp:
            source = Path(tmp) / "source.zip"
            download_archive_to(source, timeout=args.timeout)
            rows, report = materialize_archive_path(
                source,
                byte_cap=args.byte_cap,
                work_dir=Path(tmp) / "work",
            )
    _write_outputs(rows, report, records_path=args.records, report_path=args.report)
    print(
        f"{report['truth_boundary']['status']} "
        f"objects={report['selection']['selected_objects']} "
        f"bytes={report['selection']['selected_bytes']} "
        f"report={report['report_identity_sha256']}"
    )


if __name__ == "__main__":
    main()
