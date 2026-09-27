#!/usr/bin/env python3
"""Prepare a zero-credit DATA-232 handoff from the clean post-G05/G06 artifact.

The downstream handoff schema is intentionally unchanged.  This carrier replaces
the obsolete PR940 retained-inventory authority with the independently qualified
post-G05/G06 V2 materialization roots from PR #2153 / audit #2174.

Payload text remains ephemeral.  Durable output is a hash-only execution receipt
plus the exact downstream handoff.  This tool grants no tokenizer, optimizer,
training, final-test, or paid-compute authority.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

HANDOFF_SCHEMA = "12-6.postdedup-decontam-handoff.v1"
RECEIPT_SCHEMA = "12-6.data232-clean-retained-handoff-run-receipt.v1"
MATERIALIZATION_SCHEMA = "12-6.d03-post-g05-g06-materialization.v2"
INVENTORY_SCHEMA = "12-6.data526-record-inventory.v1"

TRAINING_RECORDS_NAME = "training_records.jsonl"
TRAINING_HANDOFF_NAME = "training_handoff.json"
RECEIPT_NAME = "execution_receipt.json"
CARRIER_MODULE = "tools/prepare_data232_ephemeral_handoff_v1.py"

_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")

# Independently qualified clean physical roots.  They are code-bound launch
# authority, not caller-provided values that can be coherently re-sealed.
_RELEASE_AUTHORITY: dict[str, Any] = {
    "materialization_schema_version": MATERIALIZATION_SCHEMA,
    "retained_source_count": 257,
    "distinct_physical_source_count": 244,
    "retained_payload_bytes": 5601716,
    "records_jsonl_sha256": "bbeb43b3b8e3e4b0e2631c16895d700896f3733d6cd6fe3833fe678594bc86d8",
    "record_inventory_digest_sha256": "dbdf741884ec1f147827647908b17a846584b145454e3f82fdb63120422c6059",
    "payload_inventory_digest_sha256": "2384480c89c19b14d188aa57130ee2967463512bb54241f90b6f17a525653a1e",
    "materialization_identity_sha256": "7061d74db13bf45a9a7a1266ebe50feab8e7d22c32fba7a81dd91c2be4135ade",
    "composition_preflight_identity_sha256": "1b3adfffab2a9d65af78e88b805ef221714a0cc94fca4055105665a6a155ce95",
    "physical_pr_number": 2153,
    "physical_head_git_sha": "6af889c7c3d15d38f463791c9e2ba56c8e936e16",
    "physical_run_id": 36026689718,
    "artifact_id": 10820342689,
    "artifact_zip_sha256": "99069ce2183abbbc374749cca5c538efa259df0c64658a9b88cc96b25c0fbba0",
    "independent_audit_issue_number": 2174,
    "independent_audit_status": "PASS_PHYSICAL_CLEAN_POST_G05G06_EXACT_RUN",
}

_INVENTORY_KEYS = {
    "schema_version",
    "record_count",
    "total_payload_bytes",
    "record_inventory_digest_sha256",
    "payload_inventory_digest_sha256",
    "records",
}
_INVENTORY_RECORD_KEYS = {
    "record_id",
    "source_id",
    "family",
    "modality",
    "payload_sha256",
    "payload_bytes",
}
_PAYLOAD_RECORD_KEYS = {
    "record_id",
    "source_id",
    "family",
    "modality",
    "normalized_payload",
}
_EVIDENCE_KEYS = {
    "schema_version",
    "status",
    "execution_profile",
    "execution_head_sha",
    "materializer_implementation_git_blob_sha1",
    "input",
    "transform",
    "result",
    "consumed_blockers",
    "remaining_materialization_blockers",
    "truth_boundary",
    "materializer_v2_implementation_git_blob_sha1",
    "base_v1_materialization_identity_sha256",
    "provenance_guard",
    "repeat_materialization_byte_identical",
    "materialization_identity_sha256",
}
_EVIDENCE_INPUT_KEYS = {
    "composition_preflight_identity_sha256",
    "g05_execution_identity_sha256",
    "g06_envelope_identity_sha256",
    "g06_execution_identity_sha256",
    "g06_terminal_qualification_identity_sha256",
    "input_rows_sha256",
    "privacy_binding",
    "privacy_implementation_git_blob_sha1",
    "record_payload_jsonl_sha256",
    "record_inventory_digest_sha256",
    "payload_inventory_digest_sha256",
    "record_count",
    "source_object_count",
    "total_payload_bytes",
    "external_llm_provenance_quarantine_identity_sha256",
}
_RESULT_KEYS = {
    "record_payload_jsonl_sha256",
    "record_inventory_digest_sha256",
    "payload_inventory_digest_sha256",
    "record_count",
    "source_object_count",
    "total_payload_bytes",
}
_PROVENANCE_KEYS = {
    "known_external_llm_contamination_absent",
    "quarantine_identity_sha256",
    "whole_corpus_external_llm_cleanliness_claimed",
    "successor_authority_rebuild_required_for_invalidated_v5_v6_v8",
}
_TRUTH_KEYS = {
    "current_corpus_eligible",
    "training_authorized_bytes",
    "authorized_unique_loss_positions",
    "authorized_optimized_target_exposure",
    "tokenizer_fit_authorized",
    "optimizer_updates_executed_on_real_targets",
    "training_executed",
    "learned_weights_created",
    "final_test_outcomes_read",
    "paid_compute_used",
    "foreign_pretrained_weights",
}
_TRUTH = {
    "current_corpus_eligible": False,
    "training_authorized_bytes": 0,
    "authorized_unique_loss_positions": 0,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates_executed_on_real_targets": 0,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights": False,
}
_RECEIPT_KEYS = {
    "schema_version",
    "carrier_implementation_git_sha",
    "clean_release_authority_sha256",
    "physical_release_authority",
    "input_files_sha256",
    "output_files_sha256",
    "materialization_identity_sha256",
    "composition_preflight_identity_sha256",
    "record_inventory_digest_sha256",
    "payload_inventory_digest_sha256",
    "records_jsonl_sha256",
    "retained_source_count",
    "distinct_physical_source_count",
    "retained_payload_bytes",
    "training_records_file_bytes",
    "payload_match_proven",
    "durable_receipt_hash_only",
    "raw_text_persisted_in_receipt",
    "record_ids_persisted_in_receipt",
    "authorized_optimized_target_exposure",
    "tokenizer_fit_authorized",
    "optimizer_updates_executed_on_real_targets",
    "training_executed",
    "learned_weights_created",
    "final_test_outcomes_read",
    "paid_compute_used",
    "foreign_pretrained_weights",
    "current_corpus_external_llm_free_claimed_by_this_carrier",
    "receipt_identity_sha256",
}


def _canonical(value: object, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _hex64(value: object, label: str) -> str:
    if type(value) is not str or _HEX64.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase 64-hex SHA-256")
    return value


def _git_sha(value: object, label: str) -> str:
    if type(value) is not str or _HEX40.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase 40-hex Git SHA")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be an exact positive integer")
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _strict_loads(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON in {label}") from exc
    if type(value) is not dict:
        raise TypeError(f"{label} must contain a JSON object")
    return value


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    return _strict_loads(raw, str(path)), _sha256(raw)


def _release_authority_identity() -> str:
    return _sha256(_canonical(_RELEASE_AUTHORITY, newline=True))


def _verify_inventory(inventory: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if set(inventory) != _INVENTORY_KEYS:
        raise ValueError("record inventory key set drift")
    if inventory.get("schema_version") != INVENTORY_SCHEMA:
        raise ValueError("record inventory schema drift")
    expected = _RELEASE_AUTHORITY
    count = _positive_int(inventory.get("record_count"), "record_count")
    total = _positive_int(inventory.get("total_payload_bytes"), "total_payload_bytes")
    if count != expected["retained_source_count"]:
        raise ValueError("retained record count is not independently expected")
    if total != expected["retained_payload_bytes"]:
        raise ValueError("retained payload bytes are not independently expected")
    if inventory.get("record_inventory_digest_sha256") != expected["record_inventory_digest_sha256"]:
        raise ValueError("record inventory root is not independently expected")
    if inventory.get("payload_inventory_digest_sha256") != expected["payload_inventory_digest_sha256"]:
        raise ValueError("payload inventory root is not independently expected")

    raw_rows = inventory.get("records")
    if type(raw_rows) is not list or not raw_rows:
        raise ValueError("record inventory rows missing")
    if len(raw_rows) != count:
        raise ValueError("record inventory count drift")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rows):
        if type(raw) is not dict or set(raw) != _INVENTORY_RECORD_KEYS:
            raise ValueError(f"record inventory row[{index}] schema drift")
        row = dict(raw)
        for field in ("record_id", "source_id", "family", "modality"):
            if type(row[field]) is not str or not row[field]:
                raise ValueError(f"record inventory row[{index}].{field} invalid")
        _hex64(row["payload_sha256"], f"record inventory row[{index}].payload_sha256")
        _positive_int(row["payload_bytes"], f"record inventory row[{index}].payload_bytes")
        if row["record_id"] in seen:
            raise ValueError(f"duplicate record inventory record_id: {row['record_id']}")
        seen.add(row["record_id"])
        normalized.append(row)

    if normalized != sorted(normalized, key=lambda row: row["record_id"]):
        raise ValueError("record inventory rows are not sorted")
    if sum(row["payload_bytes"] for row in normalized) != total:
        raise ValueError("record inventory payload byte sum drift")
    if len({row["source_id"] for row in normalized}) != expected["distinct_physical_source_count"]:
        raise ValueError("distinct physical source count drift")

    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in normalized
    ]
    if _sha256(_canonical(normalized)) != expected["record_inventory_digest_sha256"]:
        raise ValueError("record inventory digest does not reproduce")
    if _sha256(_canonical(payload_projection)) != expected["payload_inventory_digest_sha256"]:
        raise ValueError("payload inventory digest does not reproduce")
    return {row["record_id"]: row for row in normalized}


def _verify_evidence(evidence: Mapping[str, Any]) -> None:
    if set(evidence) != _EVIDENCE_KEYS:
        raise ValueError("materialization evidence key set drift")
    if evidence.get("schema_version") != MATERIALIZATION_SCHEMA:
        raise ValueError("materialization schema drift")
    if evidence.get("status") != "MATERIALIZED_ZERO_CREDIT":
        raise ValueError("materialization status drift")
    if evidence.get("execution_profile") != "LOCAL_FREE":
        raise ValueError("materialization profile is not LOCAL_FREE")
    if evidence.get("repeat_materialization_byte_identical") is not True:
        raise ValueError("repeat materialization proof missing")

    expected = _RELEASE_AUTHORITY
    claimed = _hex64(
        evidence.get("materialization_identity_sha256"),
        "materialization_identity_sha256",
    )
    if claimed != expected["materialization_identity_sha256"]:
        raise ValueError("materialization identity is not independently expected")
    core = dict(evidence)
    core.pop("materialization_identity_sha256")
    if _sha256(_canonical(core, newline=True)) != claimed:
        raise ValueError("materialization identity self-hash mismatch")

    input_authority = evidence.get("input")
    if type(input_authority) is not dict or set(input_authority) != _EVIDENCE_INPUT_KEYS:
        raise ValueError("materialization input authority schema drift")
    result = evidence.get("result")
    if type(result) is not dict or set(result) != _RESULT_KEYS:
        raise ValueError("materialization result schema drift")
    provenance = evidence.get("provenance_guard")
    if type(provenance) is not dict or set(provenance) != _PROVENANCE_KEYS:
        raise ValueError("materialization provenance guard schema drift")
    truth = evidence.get("truth_boundary")
    if type(truth) is not dict or set(truth) != _TRUTH_KEYS or truth != _TRUTH:
        raise ValueError("materialization truth boundary widened or drifted")

    if input_authority.get("composition_preflight_identity_sha256") != expected["composition_preflight_identity_sha256"]:
        raise ValueError("composition preflight identity drift")
    for key, wanted in (
        ("record_payload_jsonl_sha256", expected["records_jsonl_sha256"]),
        ("record_inventory_digest_sha256", expected["record_inventory_digest_sha256"]),
        ("payload_inventory_digest_sha256", expected["payload_inventory_digest_sha256"]),
        ("record_count", expected["retained_source_count"]),
        ("source_object_count", expected["distinct_physical_source_count"]),
        ("total_payload_bytes", expected["retained_payload_bytes"]),
    ):
        actual = result.get(key)
        if type(wanted) is int:
            if type(actual) is not int or actual != wanted:
                raise ValueError(f"materialization result {key} drift")
        elif actual != wanted:
            raise ValueError(f"materialization result {key} drift")

    if provenance != {
        "known_external_llm_contamination_absent": True,
        "quarantine_identity_sha256": provenance.get("quarantine_identity_sha256"),
        "whole_corpus_external_llm_cleanliness_claimed": False,
        "successor_authority_rebuild_required_for_invalidated_v5_v6_v8": True,
    }:
        raise ValueError("materialization provenance truth drift")
    _hex64(provenance.get("quarantine_identity_sha256"), "quarantine_identity_sha256")
    if evidence.get("remaining_materialization_blockers") != [
        "SUCCESSOR_CORPUS_AUTHORITY_REBUILD_REQUIRED"
    ]:
        raise ValueError("unexpected remaining materialization blockers")


def _prepare_rows(
    records_path: Path,
    by_record: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # Verify the independently expected physical byte root before parsing payload text.
    actual_raw_sha = _sha256_path(records_path)
    if actual_raw_sha != _RELEASE_AUTHORITY["records_jsonl_sha256"]:
        raise ValueError("physical records JSONL root is not independently expected")

    records: list[dict[str, Any]] = []
    projection: list[dict[str, Any]] = []
    seen: set[str] = set()
    with records_path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                raise ValueError(f"blank records JSONL line {line_number}")
            row = _strict_loads(raw, f"{records_path}:line {line_number}")
            if set(row) != _PAYLOAD_RECORD_KEYS:
                raise ValueError(f"payload row[{line_number}] schema drift")
            for field in _PAYLOAD_RECORD_KEYS:
                if type(row[field]) is not str or not row[field]:
                    raise ValueError(f"payload row[{line_number}].{field} invalid")
            record_id = row["record_id"]
            if record_id in seen:
                raise ValueError(f"duplicate payload record_id: {record_id}")
            seen.add(record_id)
            expected = by_record.get(record_id)
            if expected is None:
                raise ValueError(f"unexpected payload record_id: {record_id}")
            if (
                row["source_id"] != expected["source_id"]
                or row["family"] != expected["family"]
                or row["modality"] != expected["modality"]
            ):
                raise ValueError(f"payload metadata drift: {record_id}")
            text_raw = row["normalized_payload"].encode("utf-8")
            if len(text_raw) != expected["payload_bytes"] or _sha256(text_raw) != expected["payload_sha256"]:
                raise ValueError(f"payload identity mismatch: {record_id}")
            records.append(
                {
                    "record_id": record_id,
                    "source_id": expected["source_id"],
                    "source_family": expected["family"],
                    "modality": expected["modality"],
                    "text": row["normalized_payload"],
                }
            )
            projection.append(
                {
                    "record_id": record_id,
                    "source_id": expected["source_id"],
                    "source_family": expected["family"],
                    "modality": expected["modality"],
                    "text_sha256": expected["payload_sha256"],
                    "text_utf8_bytes": expected["payload_bytes"],
                }
            )
    missing = sorted(set(by_record) - seen)
    if missing:
        raise ValueError(f"missing payload records: {', '.join(missing[:5])}")
    if len(records) != _RELEASE_AUTHORITY["retained_source_count"]:
        raise ValueError("retained record count drift after payload binding")
    records.sort(key=lambda row: row["record_id"])
    projection.sort(key=lambda row: row["record_id"])
    return records, projection


def _handoff(projection: list[dict[str, Any]]) -> dict[str, Any]:
    handoff: dict[str, Any] = {
        "schema_version": HANDOFF_SCHEMA,
        # Compatibility fields are rebound to current clean authorities.  No old
        # PR940 identity is consumed.
        "postdedup_inventory_identity_sha256": _RELEASE_AUTHORITY[
            "materialization_identity_sha256"
        ],
        "input_survivor_authority_sha256": _RELEASE_AUTHORITY[
            "composition_preflight_identity_sha256"
        ],
        "retained_source_count": _RELEASE_AUTHORITY["retained_source_count"],
        "matcher_input_projection": projection,
        "matcher_input_projection_sha256": _sha256(_canonical(projection)),
        "raw_text_persisted_in_evidence": False,
        "final_test_payload_accessed": False,
        "final_test_outcomes_accessed": False,
        "authorized_training_exposure": 0,
    }
    handoff["handoff_identity_sha256"] = _sha256(_canonical(handoff))
    return handoff


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_bytes(_canonical(value, newline=True))


def _training_records_bytes(records: Iterable[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical(row, newline=True) for row in records)


def _build_receipt(
    *,
    carrier_git_sha: str,
    input_hashes: Mapping[str, str],
    records_raw: bytes,
    handoff_raw: bytes,
) -> dict[str, Any]:
    authority = dict(_RELEASE_AUTHORITY)
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "carrier_implementation_git_sha": _git_sha(
            carrier_git_sha, "carrier_implementation_git_sha"
        ),
        "clean_release_authority_sha256": _release_authority_identity(),
        "physical_release_authority": authority,
        "input_files_sha256": dict(input_hashes),
        "output_files_sha256": {
            TRAINING_RECORDS_NAME: _sha256(records_raw),
            TRAINING_HANDOFF_NAME: _sha256(handoff_raw),
        },
        "materialization_identity_sha256": authority["materialization_identity_sha256"],
        "composition_preflight_identity_sha256": authority[
            "composition_preflight_identity_sha256"
        ],
        "record_inventory_digest_sha256": authority["record_inventory_digest_sha256"],
        "payload_inventory_digest_sha256": authority["payload_inventory_digest_sha256"],
        "records_jsonl_sha256": authority["records_jsonl_sha256"],
        "retained_source_count": authority["retained_source_count"],
        "distinct_physical_source_count": authority["distinct_physical_source_count"],
        "retained_payload_bytes": authority["retained_payload_bytes"],
        "training_records_file_bytes": len(records_raw),
        "payload_match_proven": True,
        "durable_receipt_hash_only": True,
        "raw_text_persisted_in_receipt": False,
        "record_ids_persisted_in_receipt": False,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
        "current_corpus_external_llm_free_claimed_by_this_carrier": False,
    }
    receipt["receipt_identity_sha256"] = _sha256(_canonical(receipt))
    return receipt


def verify_receipt(receipt: Mapping[str, Any]) -> None:
    if set(receipt) != _RECEIPT_KEYS:
        raise ValueError("execution receipt key set drift")
    if receipt.get("schema_version") != RECEIPT_SCHEMA:
        raise ValueError("execution receipt schema drift")
    claimed = _hex64(receipt.get("receipt_identity_sha256"), "receipt_identity_sha256")
    core = dict(receipt)
    core.pop("receipt_identity_sha256")
    if claimed != _sha256(_canonical(core)):
        raise ValueError("execution receipt identity mismatch")
    if receipt.get("clean_release_authority_sha256") != _release_authority_identity():
        raise ValueError("clean release authority identity drift")
    if receipt.get("physical_release_authority") != _RELEASE_AUTHORITY:
        raise ValueError("physical release authority drift")
    for key in (
        "retained_source_count",
        "distinct_physical_source_count",
        "retained_payload_bytes",
        "training_records_file_bytes",
    ):
        _positive_int(receipt.get(key), f"receipt.{key}")
    if receipt.get("retained_source_count") != _RELEASE_AUTHORITY["retained_source_count"]:
        raise ValueError("receipt retained_source_count drift")
    if receipt.get("distinct_physical_source_count") != _RELEASE_AUTHORITY["distinct_physical_source_count"]:
        raise ValueError("receipt distinct physical source count drift")
    if receipt.get("retained_payload_bytes") != _RELEASE_AUTHORITY["retained_payload_bytes"]:
        raise ValueError("receipt retained payload bytes drift")
    for key in (
        "authorized_optimized_target_exposure",
        "optimizer_updates_executed_on_real_targets",
    ):
        if type(receipt.get(key)) is not int or receipt.get(key) != 0:
            raise ValueError(f"receipt {key} widened")
    for key in (
        "tokenizer_fit_authorized",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "foreign_pretrained_weights",
        "current_corpus_external_llm_free_claimed_by_this_carrier",
        "raw_text_persisted_in_receipt",
        "record_ids_persisted_in_receipt",
    ):
        if receipt.get(key) is not False:
            raise ValueError(f"receipt {key} widened")
    if receipt.get("payload_match_proven") is not True or receipt.get("durable_receipt_hash_only") is not True:
        raise ValueError("receipt proof flags drift")


def _rename_directory_no_replace(source: Path, destination: Path) -> None:
    if os.name == "nt":
        os.rename(source, destination)
        return
    libc = ctypes.CDLL(None, use_errno=True)
    try:
        renameat2 = libc.renameat2
    except AttributeError as exc:
        raise RuntimeError("atomic no-replace directory publication is unsupported") from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    AT_FDCWD = -100
    RENAME_NOREPLACE = 1
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(destination),
        RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in (errno.EEXIST, errno.ENOTEMPTY):
        raise FileExistsError(error_number, os.strerror(error_number), destination)
    if error_number in (errno.ENOSYS, errno.EINVAL):
        raise RuntimeError("atomic no-replace directory publication is unsupported")
    raise OSError(error_number, os.strerror(error_number), destination)


def prepare_and_publish(
    *,
    records_path: Path,
    inventory_path: Path,
    evidence_path: Path,
    output_dir: Path,
    carrier_git_sha: str,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite output: {output_dir}")
    inventory, inventory_file_sha = _load_json(inventory_path)
    evidence, evidence_file_sha = _load_json(evidence_path)
    by_record = _verify_inventory(inventory)
    _verify_evidence(evidence)
    if evidence["result"]["record_inventory_digest_sha256"] != inventory["record_inventory_digest_sha256"]:
        raise ValueError("evidence/inventory record root mismatch")
    if evidence["result"]["payload_inventory_digest_sha256"] != inventory["payload_inventory_digest_sha256"]:
        raise ValueError("evidence/inventory payload root mismatch")
    records, projection = _prepare_rows(records_path, by_record)
    handoff = _handoff(projection)
    records_raw = _training_records_bytes(records)
    handoff_raw = _canonical(handoff, newline=True)
    input_hashes = {
        "records_jsonl": _RELEASE_AUTHORITY["records_jsonl_sha256"],
        "record_inventory_json": inventory_file_sha,
        "materialization_evidence_json": evidence_file_sha,
    }
    receipt = _build_receipt(
        carrier_git_sha=carrier_git_sha,
        input_hashes=input_hashes,
        records_raw=records_raw,
        handoff_raw=handoff_raw,
    )
    verify_receipt(receipt)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    try:
        (temporary / TRAINING_RECORDS_NAME).write_bytes(records_raw)
        (temporary / TRAINING_HANDOFF_NAME).write_bytes(handoff_raw)
        _write_json(temporary / RECEIPT_NAME, receipt)
        _rename_directory_no_replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return receipt


def _git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def _git_path_bytes(repo_root: Path, git_sha: str, repo_path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{git_sha}:{repo_path}"],
        cwd=repo_root,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"unable to read authenticated Git bytes for {repo_path}")
    return completed.stdout


def require_exact_checkout(repo_root: Path, expected_carrier_git_sha: str) -> str:
    expected = _git_sha(expected_carrier_git_sha, "expected carrier Git SHA")
    head = _git(repo_root, "rev-parse", "--verify", "HEAD")
    if head.returncode != 0 or head.stdout.strip() != expected:
        raise RuntimeError("checked-out carrier head differs from independent expectation")
    for args in (("diff", "--quiet", "HEAD", "--"), ("diff", "--cached", "--quiet", "HEAD", "--")):
        completed = _git(repo_root, *args)
        if completed.returncode not in (0, 1):
            raise RuntimeError("unable to verify tracked working-tree cleanliness")
        if completed.returncode != 0:
            raise RuntimeError("tracked working tree differs from expected carrier head")
    physical = (repo_root / CARRIER_MODULE).resolve(strict=True).read_bytes()
    authoritative = _git_path_bytes(repo_root, expected, CARRIER_MODULE)
    if physical != authoritative:
        raise RuntimeError("physical carrier bytes differ from authenticated Git bytes")
    return expected


def require_executing_carrier(repo_root: Path, expected_carrier_git_sha: str) -> None:
    expected_path = (repo_root / CARRIER_MODULE).resolve(strict=True)
    running = Path(__file__)
    if running.is_symlink() or running.resolve(strict=True) != expected_path:
        raise RuntimeError("executing carrier path is not the authenticated repository carrier")
    authoritative = _git_path_bytes(repo_root, expected_carrier_git_sha, CARRIER_MODULE)
    if running.read_bytes() != authoritative:
        raise RuntimeError("executing carrier bytes differ from authenticated Git bytes")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--records-jsonl", type=Path, required=True)
    parser.add_argument("--record-inventory-json", type=Path, required=True)
    parser.add_argument("--materialization-evidence-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-carrier-git-sha", required=True)
    parser.add_argument("--isolated-child", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo_root = args.repo_root.resolve(strict=True)
    carrier = require_exact_checkout(repo_root, args.expected_carrier_git_sha)
    require_executing_carrier(repo_root, carrier)
    if not args.isolated_child:
        command = [
            sys.executable,
            "-I",
            "-S",
            str((repo_root / CARRIER_MODULE).resolve(strict=True)),
            "--repo-root",
            str(repo_root),
            "--records-jsonl",
            str(args.records_jsonl.resolve(strict=True)),
            "--record-inventory-json",
            str(args.record_inventory_json.resolve(strict=True)),
            "--materialization-evidence-json",
            str(args.materialization_evidence_json.resolve(strict=True)),
            "--output-dir",
            str(args.output_dir.resolve(strict=False)),
            "--expected-carrier-git-sha",
            carrier,
            "--isolated-child",
        ]
        return subprocess.run(command, check=False).returncode
    receipt = prepare_and_publish(
        records_path=args.records_jsonl,
        inventory_path=args.record_inventory_json,
        evidence_path=args.materialization_evidence_json,
        output_dir=args.output_dir,
        carrier_git_sha=carrier,
    )
    print(receipt["receipt_identity_sha256"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
