"""Authenticate audited NBU text materialization for incumbent D03 global dedup.

This adapter does not discover NBU documents, fetch PDFs, run pdftotext, implement
matching, or grant corpus/training authority. It validates the exact independently
audited NBU materialized JSONL and body-free evidence, then projects each record
one-for-one into the incumbent matcher source-object contract.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

SOURCE_ID: Final = "ua.nbu.official-resolutions"
SOURCE_FAMILY: Final = "ua.nbu.official-resolutions"
MATCHER_MODALITY: Final = "uk"
MATERIALIZATION_HEAD: Final = "13be2b000804b410aed7ba759a663408429f60f6"
MATERIALIZATION_RUN: Final = 36_531_989_330
MATERIALIZATION_JOB: Final = 109_287_497_128
MATERIALIZATION_ARTIFACT: Final = 11_017_211_425
MATERIALIZATION_AUDIT: Final = 2370
MATERIALIZATION_AUDIT_VERDICT: Final = "PASS_FOR_NBU_PHYSICAL_EVIDENCE_EXACT_HEAD"
CANDIDATE_RECORDS: Final = 40
CANDIDATE_TEXT_BYTES: Final = 794_091
CANDIDATE_SHA256: Final = (
    "bf48a0b540a1c069a6afa14f5ea24077aeea466a0f376c94c902ea175e7037d2"
)
EVIDENCE_IDENTITY_SHA256: Final = (
    "700756dd7afebfbcc663cf503e015ec221e2e9e64717762feda4f8b05610c5a0"
)
EXTRACTOR_BACKEND: Final = "poppler-pdftotext"
EXTRACTOR_VERSION: Final = "25.06.0"
EXTRACTOR_ARGUMENTS: Final = ["-enc", "UTF-8", "-nopgbrk"]
RECEIPT_SCHEMA: Final = "12-6.d03-nbu-dedup-intake-receipt.v1"
INCUMBENT_INVENTORY_SCHEMA: Final = "12-6.next100-065-cross-source-dedup.v3"

_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_RECORD_KEYS = {
    "schema",
    "record_id",
    "source_id",
    "family_id",
    "document_url",
    "pdf_url",
    "pdf_sha256",
    "text_sha256",
    "text_bytes",
    "extractor_backend",
    "extractor_version",
    "extractor_arguments",
    "text",
}
_MANIFEST_KEYS = _RECORD_KEYS - {"text"}


class NbuDedupIntakeError(RuntimeError):
    """Fail-closed audited NBU intake mismatch."""


@dataclass(frozen=True)
class NbuProjection:
    receipt: dict[str, Any]
    sources: tuple[dict[str, Any], ...] | None
    payloads: dict[str, bytes] | None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise NbuDedupIntakeError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _finite_float(raw: str) -> float:
    value = float(raw)
    _require(math.isfinite(value), "non-finite JSON number")
    return value


def _reject_constant(raw: str) -> None:
    raise NbuDedupIntakeError(f"non-finite JSON constant: {raw}")


def _loads(raw: str, *, context: str) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_float=_finite_float,
            parse_constant=_reject_constant,
        )
    except NbuDedupIntakeError:
        raise
    except (json.JSONDecodeError, ValueError, OverflowError) as exc:
        raise NbuDedupIntakeError(f"invalid {context} JSON") from exc


def _read_evidence(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise NbuDedupIntakeError(f"cannot read materialization evidence: {exc}") from exc
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise NbuDedupIntakeError("materialization evidence is not UTF-8") from exc
    value = _loads(text, context="materialization evidence")
    _require(type(value) is dict, "materialization evidence root must be exact object")
    return value


def _evidence_identity(value: Mapping[str, Any]) -> str:
    core = dict(value)
    core.pop("evidence_identity_sha256", None)
    return _sha256(_canonical(core) + b"\n")


def _validate_evidence(value: Mapping[str, Any], candidate_raw: bytes) -> list[dict[str, Any]]:
    _require(
        value.get("schema")
        == "12-6.d03-ua-nbu-pdftotext-materialization-evidence.v1",
        "materialization evidence schema drift",
    )
    _require(value.get("status") == "TEXT_MATERIALIZED_ZERO_CREDIT", "status drift")
    _require(value.get("source_id") == SOURCE_ID, "evidence source drift")
    _require(value.get("family_id") == SOURCE_FAMILY, "evidence family drift")
    _require(value.get("extractor_backend") == EXTRACTOR_BACKEND, "extractor backend drift")
    _require(value.get("extractor_version") == EXTRACTOR_VERSION, "extractor version drift")
    _require(value.get("extractor_arguments") == EXTRACTOR_ARGUMENTS, "extractor args drift")
    _require(value.get("materialized_records") == CANDIDATE_RECORDS, "record count drift")
    _require(value.get("observed_text_bytes") == CANDIDATE_TEXT_BYTES, "text byte drift")
    _require(value.get("text_artifact_bytes") == len(candidate_raw), "artifact byte drift")
    _require(value.get("text_artifact_sha256") == CANDIDATE_SHA256, "artifact hash drift")
    _require(_sha256(candidate_raw) == CANDIDATE_SHA256, "candidate hash drift")
    _require(
        value.get("evidence_identity_sha256") == EVIDENCE_IDENTITY_SHA256,
        "evidence identity authority drift",
    )
    _require(
        _evidence_identity(value) == EVIDENCE_IDENTITY_SHA256,
        "evidence self-identity drift",
    )
    truth = {
        "raw_text_emitted_in_evidence": False,
        "text_artifact_is_training_authority": False,
        "canonical_capacity_credit_bytes": 0,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "tokenizer_fit_authorized": False,
        "model_training_executed": False,
        "optimizer_updates": 0,
        "final_test_accessed": False,
        "paid_compute_used": False,
    }
    for key, expected in truth.items():
        _require(type(value.get(key)) is type(expected), f"truth type drift: {key}")
        _require(value.get(key) == expected, f"truth boundary drift: {key}")
    records = value.get("records")
    _require(type(records) is list and len(records) == CANDIDATE_RECORDS, "manifest drift")
    return records


def _validate_row(
    row: Any,
    manifest: Any,
    *,
    line_number: int,
) -> tuple[dict[str, Any], str, bytes, dict[str, Any]]:
    _require(type(row) is dict and set(row) == _RECORD_KEYS, f"row {line_number} schema drift")
    _require(
        type(manifest) is dict and set(manifest) == _MANIFEST_KEYS,
        f"manifest {line_number} schema drift",
    )
    for key in _MANIFEST_KEYS:
        _require(
            type(row[key]) is type(manifest[key]) and row[key] == manifest[key],
            f"manifest/candidate drift at row {line_number}: {key}",
        )
    _require(
        row["schema"] == "12-6.d03-ua-nbu-pdftotext-record.v1",
        f"row {line_number} schema version drift",
    )
    _require(row["source_id"] == SOURCE_ID, f"row {line_number} source drift")
    _require(row["family_id"] == SOURCE_FAMILY, f"row {line_number} family drift")
    _require(
        row["extractor_backend"] == EXTRACTOR_BACKEND
        and row["extractor_version"] == EXTRACTOR_VERSION
        and row["extractor_arguments"] == EXTRACTOR_ARGUMENTS,
        f"row {line_number} extractor drift",
    )
    record_id = row["record_id"]
    digest = row["text_sha256"]
    byte_count = row["text_bytes"]
    text = row["text"]
    _require(type(record_id) is str and _HEX64.fullmatch(record_id), "record id malformed")
    _require(type(digest) is str and _HEX64.fullmatch(digest), "text SHA malformed")
    provenance = {
        "document_url": row["document_url"],
        "pdf_url": row["pdf_url"],
        "pdf_sha256": row["pdf_sha256"],
        "text_sha256": digest,
    }
    _require(
        record_id == _sha256(_canonical(provenance) + b"\n"),
        "record provenance identity drift",
    )
    _require(type(byte_count) is int and byte_count > 0, "text byte count invalid")
    _require(type(text) is str and bool(text.strip()), "record text invalid")
    payload = text.encode("utf-8")
    _require(len(payload) == byte_count, "record text byte count drift")
    _require(_sha256(payload) == digest, "record text SHA drift")
    pdf_sha = row["pdf_sha256"]
    _require(type(pdf_sha) is str and _HEX64.fullmatch(pdf_sha), "PDF SHA malformed")
    for key in ("document_url", "pdf_url"):
        _require(
            type(row[key]) is str and row[key].startswith("https://bank.gov.ua/"),
            f"row {line_number} {key} origin drift",
        )

    matcher_source_id = f"nbu-admitted:{record_id}"
    matcher_row = {
        "source_id": matcher_source_id,
        "source_family": SOURCE_FAMILY,
        "stable_origin_id": f"{row['document_url']}#pdf-sha256:{pdf_sha}",
        "stable_object_id": f"sha256:{digest}",
        "modality": MATCHER_MODALITY,
        "evidence_status": "DEDICATED_TERMINAL",
        "authority_ref": (
            f"AUDIT{MATERIALIZATION_AUDIT}:{MATERIALIZATION_HEAD}:"
            f"run:{MATERIALIZATION_RUN}:job:{MATERIALIZATION_JOB}:"
            f"artifact:{MATERIALIZATION_ARTIFACT}"
        ),
        "declared_capacity_bytes": byte_count,
        "expected_raw_bytes": byte_count,
        "expected_raw_sha256": digest,
        "acquisition_url": row["document_url"],
        "origin_key": f"nbu:{record_id}",
    }
    inventory_row = {
        "record_id": record_id,
        "text_sha256": digest,
        "text_utf8_bytes": byte_count,
    }
    return matcher_row, matcher_source_id, payload, inventory_row


def validate_and_project_nbu(
    candidate_jsonl: Path,
    materialization_evidence_json: Path,
    *,
    retain_payloads: bool = True,
) -> NbuProjection:
    _require(type(retain_payloads) is bool, "retain_payloads must be exact bool")
    try:
        raw = candidate_jsonl.read_bytes()
    except OSError as exc:
        raise NbuDedupIntakeError(f"cannot read NBU candidate: {exc}") from exc
    _require(_sha256(raw) == CANDIDATE_SHA256, "candidate hash drift")
    evidence = _read_evidence(materialization_evidence_json)
    manifest = _validate_evidence(evidence, raw)

    lines = raw.splitlines()
    _require(len(lines) == CANDIDATE_RECORDS, "candidate line count drift")
    sources: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    inventory: list[dict[str, Any]] = []
    seen_records: set[str] = set()
    payload_bytes = 0
    for index, line in enumerate(lines, 1):
        _require(bool(line.strip()), f"blank candidate line {index}")
        try:
            text = line.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise NbuDedupIntakeError(f"candidate line {index} is not UTF-8") from exc
        row = _loads(text, context=f"candidate row {index}")
        matcher_row, matcher_id, payload, inventory_row = _validate_row(
            row,
            manifest[index - 1],
            line_number=index,
        )
        record_id = inventory_row["record_id"]
        _require(record_id not in seen_records, "duplicate NBU record id")
        _require(matcher_id not in payloads, "duplicate matcher source id")
        seen_records.add(record_id)
        sources.append(matcher_row)
        payloads[matcher_id] = payload
        inventory.append(inventory_row)
        payload_bytes += len(payload)

    _require(payload_bytes == CANDIDATE_TEXT_BYTES, "candidate text byte total drift")
    # Bind the receipt to the exact text-free matcher projection, not merely the
    # reduced record/hash/byte inventory. This mirrors the incumbent Franko intake
    # contract and makes source-family/origin/object/authority drift receipt-visible.
    projection_identity = _sha256(_canonical(sources))
    receipt_core: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "source_family": SOURCE_FAMILY,
        "authority_chain": {
            "materialization_head": MATERIALIZATION_HEAD,
            "workflow_run_id": MATERIALIZATION_RUN,
            "workflow_job_id": MATERIALIZATION_JOB,
            "artifact_id": MATERIALIZATION_ARTIFACT,
            "independent_audit_issue": MATERIALIZATION_AUDIT,
            "independent_audit_verdict": MATERIALIZATION_AUDIT_VERDICT,
            "materialization_evidence_identity_sha256": EVIDENCE_IDENTITY_SHA256,
        },
        "candidate": {
            "record_count": CANDIDATE_RECORDS,
            "text_utf8_bytes": CANDIDATE_TEXT_BYTES,
            "jsonl_sha256": CANDIDATE_SHA256,
        },
        "projection": {
            "matcher_inventory_schema": INCUMBENT_INVENTORY_SCHEMA,
            "source_object_count": len(sources),
            "payload_utf8_bytes": payload_bytes,
            "matcher_source_inventory_identity_sha256": projection_identity,
            "raw_text_emitted_in_receipt": False,
            "pre_dedup_or_aggregation_performed": False,
        },
        "execution_gate": {
            "canonical_global_dedup_executed": False,
            "matcher_invoked_by_this_adapter": False,
            "status": "READY_FOR_INCUMBENT_MATCHER_AFTER_DEPENDENCY_AUTHORITY",
        },
        "truth_boundary": {
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates_executed_on_real_targets": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
            "whole_corpus_external_llm_cleanliness_claimed": False,
        },
    }
    receipt = {
        **receipt_core,
        "receipt_identity_sha256": _sha256(_canonical(receipt_core)),
    }
    return NbuProjection(
        receipt=receipt,
        sources=tuple(sources) if retain_payloads else None,
        payloads=payloads if retain_payloads else None,
    )
