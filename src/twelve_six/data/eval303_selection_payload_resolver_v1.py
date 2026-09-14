"""Resolve EVAL-303 selection payloads from immutable EVAL-290/291 artifacts.

EVAL-303 itself is intentionally hash/provenance-only.  This resolver consumes the
exact terminal UA/EN component artifact ZIPs plus the exact EVAL-303 hash-only
membership JSONL, verifies all three authorities, and returns selection text only in
memory for DATA-232 contamination matching.  Durable evidence remains text-free.
"""
from __future__ import annotations

import hashlib
import json
import re
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

EVAL303_SELECTION_ID = "7b97a9ab04469236dc5bc17fc80155cb43430b01c443bb6209fac090557258fd"
EVAL303_MEMBERSHIP_SHA256 = "e4bb39dd7aa6a20c7ed34e093f563b5f4896ac16828151c6b375a83cd8a068c6"
EVAL303_TERMINAL_SOURCE_SHA = "5e5a1de3b594cee5612e63d3d4c2a70499740ac7"

EVAL290_HEAD = "029514654829cebc149cff6fc1fea2a8ba4fa566"
EVAL290_ARTIFACT_ID = 9606656857
EVAL290_ARTIFACT_SHA256 = "0c5f9f8d938284a1358bfb77284814b1f0569b2d6d13938f415ba31db64a6c3b"
EVAL290_SET_ID = "c32320a706a283049e35eb537eb20a1e7f5865b86c24397c8b73d1e3d2014164"
EVAL290_DATA_PATH = "tmp/eval290-a/eval290_ua_selection_validation_v1.jsonl"
EVAL290_DATA_SHA256 = "5400e37b8e8ddfdb9516a8b6eaa6fc01a8c5ec41811385d9f6907e6ef6d9614f"
EVAL290_MANIFEST_PATH = "tmp/eval290-a/eval290_ua_selection_validation_manifest_v1.json"
EVAL290_MANIFEST_SHA256 = "d0f8cf208724c80bda7b92aed54e10bea1492fa5bb3511a7e0fd828fc881e14a"

EVAL291_HEAD = "fb268061300127b62cc2a262664b30c614559dac"
EVAL291_ARTIFACT_ID = 9606132134
EVAL291_ARTIFACT_SHA256 = "3168a1c7884b10ba7a959c859c1539f2ad4957a852ad3f9aa3517df3c5110f94"
EVAL291_AUTHORITY_ID = "727f229c091f86748a4eee9ea5aec72bb65347b68d6b687fabbf33166b0eca1e"
EVAL291_DATA_PATH = "data/evaluation/eval291/selection-validation/en.jsonl"
EVAL291_DATA_SHA256 = "df20e3d3ec75208399039a283487c3b8958c80ec3f119cee278fcc09948c6bfb"
EVAL291_AUTHORITY_PATH = "evidence/eval291/en-selection-validation-v1-authority.json"
EVAL291_AUTHORITY_SHA256 = "493be56b8189325d122868ca778733676cc2143d24d121fb63f6b1b62a977177"

EXPECTED_RECORDS = 10
EXPECTED_UA_RECORDS = 8
EXPECTED_EN_RECORDS = 2


class SelectionPayloadResolverError(RuntimeError):
    """Fail-closed EVAL-303 payload-resolution error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SelectionPayloadResolverError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_exact_zip_member(
    archive: zipfile.ZipFile,
    path: str,
    expected_sha256: str,
) -> bytes:
    try:
        value = archive.read(path)
    except KeyError as exc:
        raise SelectionPayloadResolverError(f"artifact member missing: {path}") from exc
    _require(_sha256(value) == expected_sha256, f"artifact member SHA-256 drift: {path}")
    return value


def _parse_json_object(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectionPayloadResolverError(f"invalid JSON: {label}") from exc
    _require(isinstance(value, dict), f"expected JSON object: {label}")
    return value


def _parse_jsonl(raw: bytes, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(raw.splitlines(), start=1):
        if not line:
            continue
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SelectionPayloadResolverError(f"invalid JSONL row {index}: {label}") from exc
        _require(isinstance(value, dict), f"JSONL row {index} is not an object: {label}")
        rows.append(value)
    return rows


def _load_membership(path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw = path.read_bytes()
    _require(
        _sha256(raw) == EVAL303_MEMBERSHIP_SHA256,
        "EVAL-303 membership SHA-256 drift",
    )
    rows = _parse_jsonl(raw, "EVAL-303 membership")
    _require(len(rows) == EXPECTED_RECORDS, "EVAL-303 membership record-count drift")
    seen: set[str] = set()
    for row in rows:
        record_id = row.get("record_id")
        _require(isinstance(record_id, str) and bool(record_id), "EVAL-303 membership record_id invalid")
        _require(record_id not in seen, "EVAL-303 membership record_id duplicated")
        seen.add(record_id)
        _require(row.get("purpose") == "selection-validation", "EVAL-303 membership purpose drift")
        _require(row.get("selection_eligible") is True, "EVAL-303 selection eligibility drift")
        _require(row.get("training_eligible") is False, "EVAL-303 training boundary drift")
        _require(row.get("tokenizer_fit_eligible") is False, "EVAL-303 tokenizer-fit boundary drift")
        _require(row.get("final_test_eligible") is False, "EVAL-303 final-test boundary drift")
        content = row.get("content_sha256")
        _require(
            isinstance(content, str) and re.fullmatch(r"[0-9a-f]{64}", content) is not None,
            "EVAL-303 membership content SHA-256 invalid",
        )
        _require("text" not in row, "EVAL-303 membership unexpectedly contains payload text")
    return rows, raw


def _load_eval290(artifact: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_zip = artifact.read_bytes()
    _require(_sha256(raw_zip) == EVAL290_ARTIFACT_SHA256, "EVAL-290 artifact ZIP identity drift")
    with zipfile.ZipFile(artifact) as archive:
        data_raw = _read_exact_zip_member(archive, EVAL290_DATA_PATH, EVAL290_DATA_SHA256)
        manifest_raw = _read_exact_zip_member(
            archive,
            EVAL290_MANIFEST_PATH,
            EVAL290_MANIFEST_SHA256,
        )
    manifest = _parse_json_object(manifest_raw, "EVAL-290 manifest")
    _require(manifest.get("set_identity_sha256") == EVAL290_SET_ID, "EVAL-290 set identity drift")
    _require(manifest.get("documents") == EXPECTED_UA_RECORDS, "EVAL-290 document count drift")
    eligibility = manifest.get("eligibility", {})
    _require(eligibility.get("training") is False, "EVAL-290 training boundary drift")
    _require(eligibility.get("tokenizer_fit") is False, "EVAL-290 tokenizer boundary drift")
    _require(eligibility.get("final_test") is False, "EVAL-290 final-test boundary drift")
    _require(eligibility.get("model_selection") is True, "EVAL-290 selection boundary drift")
    rows = _parse_jsonl(data_raw, "EVAL-290 selection data")
    _require(len(rows) == EXPECTED_UA_RECORDS, "EVAL-290 row count drift")
    for row in rows:
        _require(row.get("purpose") == "selection-validation", "EVAL-290 purpose drift")
        _require(row.get("selection_eligible") is True, "EVAL-290 selection eligibility drift")
        _require(row.get("training_eligible") is False, "EVAL-290 training eligibility drift")
        _require(row.get("tokenizer_fit_eligible") is False, "EVAL-290 tokenizer eligibility drift")
        _require(row.get("final_test_eligible") is False, "EVAL-290 final-test eligibility drift")
        _require(row.get("future_training_prohibited") is True, "EVAL-290 reservation drift")
        text = row.get("text")
        _require(isinstance(text, str) and bool(text), "EVAL-290 selection text missing")
        text_raw = text.encode("utf-8")
        _require(_sha256(text_raw) == row.get("content_sha256"), "EVAL-290 content hash drift")
        _require(len(text_raw) == row.get("utf8_bytes"), "EVAL-290 content byte-count drift")
    return rows, manifest


def _load_eval291(artifact: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_zip = artifact.read_bytes()
    _require(_sha256(raw_zip) == EVAL291_ARTIFACT_SHA256, "EVAL-291 artifact ZIP identity drift")
    with zipfile.ZipFile(artifact) as archive:
        data_raw = _read_exact_zip_member(archive, EVAL291_DATA_PATH, EVAL291_DATA_SHA256)
        authority_raw = _read_exact_zip_member(
            archive,
            EVAL291_AUTHORITY_PATH,
            EVAL291_AUTHORITY_SHA256,
        )
    authority = _parse_json_object(authority_raw, "EVAL-291 authority")
    _require(
        authority.get("authority_identity_sha256") == EVAL291_AUTHORITY_ID,
        "EVAL-291 authority identity drift",
    )
    _require(authority.get("documents") == EXPECTED_EN_RECORDS, "EVAL-291 document count drift")
    _require(authority.get("purpose") == "selection_validation", "EVAL-291 purpose drift")
    firewalls = authority.get("firewalls", {})
    _require(firewalls.get("selection_bytes_are_training_eligible") is False, "EVAL-291 training firewall drift")
    _require(firewalls.get("selection_bytes_are_tokenizer_fit_eligible") is False, "EVAL-291 tokenizer firewall drift")
    _require(firewalls.get("selection_bytes_are_final_test_eligible") is False, "EVAL-291 final-test firewall drift")
    final_boundary = authority.get("final_test_boundary", {})
    _require(final_boundary.get("outcomes_read_for_construction") is False, "EVAL-291 final-test outcome boundary drift")
    _require(final_boundary.get("payload_read_for_construction") is False, "EVAL-291 final-test construction boundary drift")
    sources = authority.get("sources")
    _require(isinstance(sources, Sequence) and not isinstance(sources, (str, bytes)), "EVAL-291 sources invalid")
    by_document = {row.get("document_id"): row for row in sources if isinstance(row, Mapping)}
    rows = _parse_jsonl(data_raw, "EVAL-291 selection data")
    _require(len(rows) == EXPECTED_EN_RECORDS, "EVAL-291 row count drift")
    for row in rows:
        document_id = row.get("document_id")
        source = by_document.get(document_id)
        _require(isinstance(document_id, str) and isinstance(source, Mapping), "EVAL-291 document authority missing")
        _require(row.get("purpose") == "selection_validation", "EVAL-291 payload purpose drift")
        text = row.get("text")
        _require(isinstance(text, str) and bool(text), "EVAL-291 selection text missing")
        text_raw = text.encode("utf-8")
        _require(_sha256(text_raw) == row.get("content_sha256"), "EVAL-291 content hash drift")
        _require(_sha256(text_raw) == source.get("raw_sha256"), "EVAL-291 source hash drift")
        _require(len(text_raw) == source.get("raw_bytes"), "EVAL-291 source byte-count drift")
        reservation = source.get("project_reservation", {})
        _require(reservation.get("selection_validation") is True, "EVAL-291 reservation drift")
        _require(reservation.get("training") is False, "EVAL-291 training reservation drift")
        _require(reservation.get("tokenizer_fit") is False, "EVAL-291 tokenizer reservation drift")
        _require(reservation.get("final_test") is False, "EVAL-291 final-test reservation drift")
    return rows, authority


def resolve_eval303_selection_payloads(
    eval290_artifact_zip: Path,
    eval291_artifact_zip: Path,
    eval303_membership_jsonl: Path,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    """Resolve ten terminal selection records for decontamination-only use."""
    membership, _ = _load_membership(eval303_membership_jsonl)
    ua_rows, _ = _load_eval290(eval290_artifact_zip)
    en_rows, _ = _load_eval291(eval291_artifact_zip)

    payload_by_record: dict[str, dict[str, str]] = {}
    for row in ua_rows:
        record_id = str(row["record_id"])
        payload_by_record[record_id] = {
            "record_id": record_id,
            "source_id": str(row["source_id"]),
            "source_family": str(row["source_family"]),
            "modality": "ua",
            "text": str(row["text"]),
        }
    for row in en_rows:
        record_id = str(row["document_id"])
        source_family = str(row["source_family"])
        source_path = str(row["source_path"])
        source_id = f"{source_family}:{source_path}"
        payload_by_record[record_id] = {
            "record_id": record_id,
            "source_id": source_id,
            "source_family": source_family,
            "modality": "text",
            "text": str(row["text"]),
        }
    _require(len(payload_by_record) == EXPECTED_RECORDS, "resolved selection record count drift")

    membership_by_record = {str(row["record_id"]): row for row in membership}
    _require(set(payload_by_record) == set(membership_by_record), "selection artifact/membership record coverage drift")
    matcher_rows: list[dict[str, str]] = []
    members: list[dict[str, Any]] = []
    for record_id in sorted(payload_by_record):
        payload = payload_by_record[record_id]
        authority = membership_by_record[record_id]
        text_raw = payload["text"].encode("utf-8")
        _require(payload["source_id"] == authority.get("source_id"), f"selection source_id drift: {record_id}")
        _require(payload["source_family"] == authority.get("source_family"), f"selection source_family drift: {record_id}")
        _require(payload["modality"] == authority.get("modality"), f"selection modality drift: {record_id}")
        _require(_sha256(text_raw) == authority.get("content_sha256"), f"selection content hash drift: {record_id}")
        _require(len(text_raw) == authority.get("utf8_bytes"), f"selection content byte-count drift: {record_id}")
        matcher_rows.append(payload)
        members.append(
            {
                "record_id": record_id,
                "source_id": payload["source_id"],
                "source_family": payload["source_family"],
                "modality": payload["modality"],
                "content_sha256": _sha256(text_raw),
                "utf8_bytes": len(text_raw),
                "training_prohibited": True,
                "outcomes_included": False,
            }
        )

    reserved_set: dict[str, Any] = {
        "authority_id": "eval303-selection-validation",
        "identity_sha256": EVAL303_SELECTION_ID,
        "role": "selection_validation",
        "source_sha": EVAL303_TERMINAL_SOURCE_SHA,
        "source_membership_identity_sha256": EVAL303_MEMBERSHIP_SHA256,
        "members": members,
    }
    projection = [
        {
            "record_id_sha256": _sha256(row["record_id"].encode("utf-8")),
            "source_id_sha256": _sha256(row["source_id"].encode("utf-8")),
            "source_family_sha256": _sha256(row["source_family"].encode("utf-8")),
            "modality": row["modality"],
            "content_sha256": member["content_sha256"],
            "utf8_bytes": member["utf8_bytes"],
        }
        for row, member in zip(matcher_rows, members, strict=True)
    ]
    evidence_core: dict[str, Any] = {
        "schema_version": "12-6.eval303-selection-payload-resolver.v1",
        "eval303_selection_identity_sha256": EVAL303_SELECTION_ID,
        "eval303_membership_sha256": EVAL303_MEMBERSHIP_SHA256,
        "eval303_terminal_source_sha": EVAL303_TERMINAL_SOURCE_SHA,
        "eval290_head_sha": EVAL290_HEAD,
        "eval290_artifact_id": EVAL290_ARTIFACT_ID,
        "eval290_artifact_sha256": EVAL290_ARTIFACT_SHA256,
        "eval290_set_identity_sha256": EVAL290_SET_ID,
        "eval291_head_sha": EVAL291_HEAD,
        "eval291_artifact_id": EVAL291_ARTIFACT_ID,
        "eval291_artifact_sha256": EVAL291_ARTIFACT_SHA256,
        "eval291_authority_identity_sha256": EVAL291_AUTHORITY_ID,
        "documents": len(matcher_rows),
        "modality_documents": {"ua": EXPECTED_UA_RECORDS, "en": EXPECTED_EN_RECORDS, "code": 0},
        "member_projection": projection,
        "member_projection_sha256": _sha256(_canonical_bytes(projection)),
        "raw_text_persisted_in_evidence": False,
        "selection_payload_accessed_for_decontamination": True,
        "final_test_payload_accessed": False,
        "final_test_outcomes_read": False,
        "authorized_training_exposure": 0,
        "training_executed": False,
    }
    evidence_core["resolver_identity_sha256"] = _sha256(_canonical_bytes(evidence_core))
    return matcher_rows, reserved_set, evidence_core
