"""Bind the released clean retained payload to the incumbent DATA-232 handoff.

This module does not create another record schema and does not grant tokenizer or
training authority. The public entry point accepts only the immutable physical
release from issue 2015. Payload text is returned only in ephemeral matcher rows;
durable handoff and receipt evidence remain text-free.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from twelve_six.data.clean_g05_g06_successor_authority_v1 import (
    NOMIS_FAMILY,
    NOMIS_PAYLOAD_SHA256,
    NOMIS_RECORD_ID,
    PR462_AUTHORITY_SHA256,
)
from twelve_six.data.postdedup_decontam_handoff_v1 import HANDOFF_SCHEMA

PHYSICAL_PROOF_SCHEMA = "12-6.d03-clean-post-g05-g06-two-run-physical.v1"
INVENTORY_SCHEMA = "12-6.data526-record-inventory.v1"
EVIDENCE_SCHEMA = "12-6.d03-post-g05-g06-materialization.v2"
RECEIPT_SCHEMA = "12-6.clean-retained-data232-handoff-receipt.v1"

PHYSICAL_RUN_ID = 36026689718
PHYSICAL_JOB_ID = 107724922341
PHYSICAL_ARTIFACT_ID = 10820342689
PHYSICAL_ARTIFACT_ZIP_SHA256 = (
    "99069ce2183abbbc374749cca5c538efa259df0c64658a9b88cc96b25c0fbba0"
)
AUTHORITATIVE_MAIN_SHA = "4588b660fd7650d6ddb072e9a9b666f5cc97238c"
EXECUTION_CARRIER_HEAD_SHA = "6af889c7c3d15d38f463791c9e2ba56c8e936e16"

OUTPUT_RECORD_COUNT = 257
OUTPUT_PAYLOAD_BYTES = 5601716
OUTPUT_JSONL_SHA256 = (
    "bbeb43b3b8e3e4b0e2631c16895d700896f3733d6cd6fe3833fe678594bc86d8"
)
OUTPUT_INVENTORY_FILE_SHA256 = (
    "3804a43eba5e0bfa6ce2568782cb03bf681e53a68808742874cf89139488c69e"
)
OUTPUT_EVIDENCE_FILE_SHA256 = (
    "877b6233738e0ca6593acbbf9922f5f178ec8456bf101b95f2f4a300805509e4"
)
OUTPUT_PROOF_FILE_SHA256 = (
    "16260584c5c2bb7d8eab7a16e1b5260300e9cd2b67e9110282287e619013078f"
)
OUTPUT_RECORD_INVENTORY_DIGEST_SHA256 = (
    "dbdf741884ec1f147827647908b17a846584b145454e3f82fdb63120422c6059"
)
OUTPUT_PAYLOAD_INVENTORY_DIGEST_SHA256 = (
    "2384480c89c19b14d188aa57130ee2967463512bb54241f90b6f17a525653a1e"
)
MATERIALIZATION_IDENTITY_SHA256 = (
    "7061d74db13bf45a9a7a1266ebe50feab8e7d22c32fba7a81dd91c2be4135ade"
)
CLEAN_DATA526_EVIDENCE_IDENTITY_SHA256 = (
    "45ac421c9e4a5930330c1516af79e1b67d0535fc90b768ecfe9df54dd0a86c5c"
)
G05_EXECUTION_IDENTITY_SHA256 = (
    "b0eea4c1190cef898af60b8227dce1f0acd041da142245df5cfb21d2d670e7f3"
)
G06_EXECUTION_IDENTITY_SHA256 = (
    "31fb3daf889f4568fb11fac65357a80716e658279a4726bc654db609696e617c"
)
G06_ENVELOPE_IDENTITY_SHA256 = (
    "7e5b38fec56085188a0c7fc322ca1ac5b0d01903e7182ff67075a1975e9c9e36"
)
G06_TERMINAL_QUALIFICATION_IDENTITY_SHA256 = (
    "4872057d9cb826700b6e3b1b6210752143b18fc6a4cb96a8a0fb9f8e551335f8"
)
COMPOSITION_PREFLIGHT_IDENTITY_SHA256 = (
    "1b3adfffab2a9d65af78e88b805ef221714a0cc94fca4055105665a6a155ce95"
)

STALE_PR1836_RECORD_INVENTORY_SHA256 = (
    "766cc01d610373183e73a632e18674b1d7afc5aaa7e3888638029e0044b26103"
)
STALE_PR1836_PAYLOAD_INVENTORY_SHA256 = (
    "56ef7a457f4c4f649ad97359c2a177131632d838756d769e4b1a2ec3f3a2a278"
)

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_RECORD_KEYS = {"record_id", "source_id", "family", "modality", "normalized_payload"}
_INVENTORY_RECORD_KEYS = {
    "record_id",
    "source_id",
    "family",
    "modality",
    "payload_sha256",
    "payload_bytes",
}
_INVENTORY_KEYS = {
    "schema_version",
    "record_count",
    "total_payload_bytes",
    "record_inventory_digest_sha256",
    "payload_inventory_digest_sha256",
    "records",
}
_PROOF_KEYS = {
    "schema",
    "execution_profile",
    "authoritative_main_sha",
    "execution_carrier_head_sha",
    "input_record_count",
    "input_payload_bytes",
    "input_jsonl_sha256",
    "input_rows_sha256",
    "clean_data526_evidence_identity_sha256",
    "g05_execution_identity_sha256",
    "g06_execution_identity_sha256",
    "g06_envelope_identity_sha256",
    "g06_terminal_qualification_identity_sha256",
    "composition_preflight_identity_sha256",
    "materializer_v1_git_blob_sha1",
    "materializer_v2_git_blob_sha1",
    "privacy_implementation_git_blob_sha1",
    "privacy_policy_sha256",
    "two_fresh_cli_processes",
    "payload_byte_identical",
    "inventory_byte_identical",
    "evidence_byte_identical",
    "output_record_count",
    "output_payload_bytes",
    "output_jsonl_sha256",
    "output_inventory_file_sha256",
    "output_evidence_file_sha256",
    "output_record_inventory_digest_sha256",
    "output_payload_inventory_digest_sha256",
    "materialization_identity_sha256",
    "known_nomis_pr462_payload_absent",
    "privacy_redactions_rescanned_allow",
    "truth_boundary",
}
_PROOF_TRUTH_KEYS = {
    "current_corpus_eligible",
    "authorized_optimized_target_exposure",
    "tokenizer_fit_authorized",
    "optimizer_updates_executed_on_real_targets",
    "training_executed",
    "learned_weights_created",
    "final_test_outcomes_read",
    "paid_compute_used",
    "foreign_pretrained_weights",
    "whole_corpus_external_llm_cleanliness_claimed",
}
_EVIDENCE_TRUTH_KEYS = {
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
_RECEIPT_FILE_KEYS = {"records.jsonl", "inventory.json", "evidence.json", "proof.json"}
_RECEIPT_KEYS = {
    "schema_version",
    "physical_run_id",
    "physical_job_id",
    "physical_artifact_id",
    "physical_artifact_zip_sha256",
    "authoritative_main_sha",
    "execution_carrier_head_sha",
    "input_files_sha256",
    "record_count",
    "payload_bytes",
    "record_inventory_digest_sha256",
    "payload_inventory_digest_sha256",
    "materialization_identity_sha256",
    "clean_data526_evidence_identity_sha256",
    "g05_execution_identity_sha256",
    "g06_execution_identity_sha256",
    "g06_envelope_identity_sha256",
    "g06_terminal_qualification_identity_sha256",
    "composition_preflight_identity_sha256",
    "training_handoff_identity_sha256",
    "matcher_input_projection_sha256",
    "known_nomis_pr462_payload_absent",
    "provenance_trace_bound",
    "durable_receipt_text_free",
    "current_retained_corpus_launch_authoritative",
    "authorized_optimized_target_exposure",
    "tokenizer_fit_authorized",
    "optimizer_updates_executed_on_real_targets",
    "training_executed",
    "learned_weights_created",
    "final_test_outcomes_read",
    "paid_compute_used",
    "foreign_pretrained_weights",
    "whole_corpus_external_llm_cleanliness_claimed",
    "receipt_identity_sha256",
}


class CleanRetainedHandoffError(ValueError):
    """Raised when the clean retained handoff cannot be proved exactly."""


@dataclass(frozen=True)
class _PhysicalRelease:
    run_id: int
    job_id: int
    artifact_id: int
    artifact_zip_sha256: str
    authoritative_main_sha: str
    execution_carrier_head_sha: str
    record_count: int
    payload_bytes: int
    jsonl_sha256: str
    inventory_file_sha256: str
    evidence_file_sha256: str
    proof_file_sha256: str
    record_inventory_digest_sha256: str
    payload_inventory_digest_sha256: str
    materialization_identity_sha256: str
    clean_data526_evidence_identity_sha256: str
    g05_execution_identity_sha256: str
    g06_execution_identity_sha256: str
    g06_envelope_identity_sha256: str
    g06_terminal_qualification_identity_sha256: str
    composition_preflight_identity_sha256: str


_RELEASE = _PhysicalRelease(
    run_id=PHYSICAL_RUN_ID,
    job_id=PHYSICAL_JOB_ID,
    artifact_id=PHYSICAL_ARTIFACT_ID,
    artifact_zip_sha256=PHYSICAL_ARTIFACT_ZIP_SHA256,
    authoritative_main_sha=AUTHORITATIVE_MAIN_SHA,
    execution_carrier_head_sha=EXECUTION_CARRIER_HEAD_SHA,
    record_count=OUTPUT_RECORD_COUNT,
    payload_bytes=OUTPUT_PAYLOAD_BYTES,
    jsonl_sha256=OUTPUT_JSONL_SHA256,
    inventory_file_sha256=OUTPUT_INVENTORY_FILE_SHA256,
    evidence_file_sha256=OUTPUT_EVIDENCE_FILE_SHA256,
    proof_file_sha256=OUTPUT_PROOF_FILE_SHA256,
    record_inventory_digest_sha256=OUTPUT_RECORD_INVENTORY_DIGEST_SHA256,
    payload_inventory_digest_sha256=OUTPUT_PAYLOAD_INVENTORY_DIGEST_SHA256,
    materialization_identity_sha256=MATERIALIZATION_IDENTITY_SHA256,
    clean_data526_evidence_identity_sha256=CLEAN_DATA526_EVIDENCE_IDENTITY_SHA256,
    g05_execution_identity_sha256=G05_EXECUTION_IDENTITY_SHA256,
    g06_execution_identity_sha256=G06_EXECUTION_IDENTITY_SHA256,
    g06_envelope_identity_sha256=G06_ENVELOPE_IDENTITY_SHA256,
    g06_terminal_qualification_identity_sha256=G06_TERMINAL_QUALIFICATION_IDENTITY_SHA256,
    composition_preflight_identity_sha256=COMPOSITION_PREFLIGHT_IDENTITY_SHA256,
)


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise CleanRetainedHandoffError(message)


def _canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return raw + (b"\n" if newline else b"")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _HEX64.fullmatch(value) is not None,
        f"{label} must be lowercase SHA-256",
    )
    return str(value)


def _git_sha(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _HEX40.fullmatch(value) is not None,
        f"{label} must be lowercase Git SHA",
    )
    return str(value)


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    _require(
        type(value) is int and value >= minimum,
        f"{label} must be an exact integer >= {minimum}",
    )
    return int(value)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CleanRetainedHandoffError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise CleanRetainedHandoffError(f"non-finite JSON constant is forbidden: {value}")


def _strict_json(raw: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CleanRetainedHandoffError(f"invalid UTF-8 JSON: {label}") from exc
    _require(isinstance(value, dict), f"{label} must contain one JSON object")
    return value


def _strict_jsonl(raw: bytes) -> list[dict[str, Any]]:
    _require(bool(raw), "records.jsonl must be non-empty")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
        _require(line.endswith(b"\n"), "records.jsonl row is not newline terminated")
        _require(bool(line.strip()), f"blank JSONL row forbidden: {line_number}")
        row = _strict_json(line, f"records.jsonl:{line_number}")
        _require(
            _canonical_bytes(row, newline=True) == line,
            f"records.jsonl:{line_number} is not canonical",
        )
        rows.append(row)
    return rows


def _zero_truth(
    value: Any,
    label: str,
    *,
    expected_keys: set[str],
) -> None:
    _require(isinstance(value, Mapping), f"{label} missing")
    _require(set(value) == expected_keys, f"{label} key set drift")
    for key in (
        "current_corpus_eligible",
        "tokenizer_fit_authorized",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "foreign_pretrained_weights",
    ):
        _require(value.get(key) is False, f"{label}.{key} widened")
    if "whole_corpus_external_llm_cleanliness_claimed" in expected_keys:
        _require(
            value.get("whole_corpus_external_llm_cleanliness_claimed") is False,
            f"{label}.whole_corpus_external_llm_cleanliness_claimed widened",
        )
    for key in (
        "training_authorized_bytes",
        "authorized_unique_loss_positions",
        "authorized_optimized_target_exposure",
        "optimizer_updates_executed_on_real_targets",
    ):
        if key in expected_keys:
            _require(
                type(value.get(key)) is int and value.get(key) == 0,
                f"{label}.{key} widened",
            )


def _validate_proof(raw: bytes, *, release: _PhysicalRelease) -> dict[str, Any]:
    _require(_sha(raw) == release.proof_file_sha256, "physical proof file SHA drift")
    proof = _strict_json(raw, "proof.json")
    _require(set(proof) == _PROOF_KEYS, "physical proof key set drift")
    _require(proof.get("schema") == PHYSICAL_PROOF_SCHEMA, "physical proof schema drift")
    _require(proof.get("execution_profile") == "LOCAL_FREE", "physical proof left LOCAL_FREE")
    _require(
        proof.get("authoritative_main_sha") == release.authoritative_main_sha,
        "physical Product authority SHA drift",
    )
    _require(
        proof.get("execution_carrier_head_sha") == release.execution_carrier_head_sha,
        "physical carrier head drift",
    )
    _git_sha(proof.get("authoritative_main_sha"), "authoritative_main_sha")
    _git_sha(proof.get("execution_carrier_head_sha"), "execution_carrier_head_sha")
    for field in (
        "materializer_v1_git_blob_sha1",
        "materializer_v2_git_blob_sha1",
        "privacy_implementation_git_blob_sha1",
    ):
        _git_sha(proof.get(field), field)
    for field in ("input_jsonl_sha256", "input_rows_sha256", "privacy_policy_sha256"):
        _sha256(proof.get(field), field)
    for field in ("input_record_count", "input_payload_bytes"):
        _strict_int(proof.get(field), field, minimum=1)
    expected: dict[str, Any] = {
        "output_record_count": release.record_count,
        "output_payload_bytes": release.payload_bytes,
        "output_jsonl_sha256": release.jsonl_sha256,
        "output_inventory_file_sha256": release.inventory_file_sha256,
        "output_evidence_file_sha256": release.evidence_file_sha256,
        "output_record_inventory_digest_sha256": release.record_inventory_digest_sha256,
        "output_payload_inventory_digest_sha256": release.payload_inventory_digest_sha256,
        "materialization_identity_sha256": release.materialization_identity_sha256,
        "clean_data526_evidence_identity_sha256": release.clean_data526_evidence_identity_sha256,
        "g05_execution_identity_sha256": release.g05_execution_identity_sha256,
        "g06_execution_identity_sha256": release.g06_execution_identity_sha256,
        "g06_envelope_identity_sha256": release.g06_envelope_identity_sha256,
        "g06_terminal_qualification_identity_sha256": (
            release.g06_terminal_qualification_identity_sha256
        ),
        "composition_preflight_identity_sha256": release.composition_preflight_identity_sha256,
    }
    for key, expected_value in expected.items():
        _require(
            type(proof.get(key)) is type(expected_value) and proof.get(key) == expected_value,
            f"physical proof drift: {key}",
        )
    for key in (
        "two_fresh_cli_processes",
        "payload_byte_identical",
        "inventory_byte_identical",
        "evidence_byte_identical",
        "known_nomis_pr462_payload_absent",
        "privacy_redactions_rescanned_allow",
    ):
        _require(proof.get(key) is True, f"physical proof weakened: {key}")
    _zero_truth(
        proof.get("truth_boundary"),
        "physical proof truth",
        expected_keys=_PROOF_TRUTH_KEYS,
    )
    return proof


def _validate_inventory(
    raw: bytes,
    *,
    release: _PhysicalRelease,
) -> list[dict[str, Any]]:
    _require(_sha(raw) == release.inventory_file_sha256, "inventory file SHA drift")
    inventory = _strict_json(raw, "inventory.json")
    _require(set(inventory) == _INVENTORY_KEYS, "record inventory key set drift")
    _require(inventory.get("schema_version") == INVENTORY_SCHEMA, "record inventory schema drift")
    count = _strict_int(inventory.get("record_count"), "record_count", minimum=1)
    payload_bytes = _strict_int(
        inventory.get("total_payload_bytes"),
        "total_payload_bytes",
        minimum=1,
    )
    _require(count == release.record_count, "clean retained record count drift")
    _require(payload_bytes == release.payload_bytes, "clean retained payload byte drift")
    record_root = _sha256(
        inventory.get("record_inventory_digest_sha256"),
        "record_inventory_digest_sha256",
    )
    payload_root = _sha256(
        inventory.get("payload_inventory_digest_sha256"),
        "payload_inventory_digest_sha256",
    )
    _require(
        record_root == release.record_inventory_digest_sha256,
        "clean record inventory root drift",
    )
    _require(
        payload_root == release.payload_inventory_digest_sha256,
        "clean payload inventory root drift",
    )
    _require(
        record_root != STALE_PR1836_RECORD_INVENTORY_SHA256,
        "historical contaminated record inventory root rejected",
    )
    _require(
        payload_root != STALE_PR1836_PAYLOAD_INVENTORY_SHA256,
        "historical contaminated payload inventory root rejected",
    )
    rows = inventory.get("records")
    _require(
        isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)),
        "inventory.records must be a sequence",
    )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(rows):
        _require(isinstance(raw_row, Mapping), f"inventory.records[{index}] must be an object")
        _require(set(raw_row) == _INVENTORY_RECORD_KEYS, f"inventory.records[{index}] schema drift")
        row: dict[str, Any] = {}
        for key in ("record_id", "source_id", "family", "modality"):
            value = raw_row.get(key)
            _require(
                isinstance(value, str) and bool(value),
                f"inventory.records[{index}].{key} must be non-empty text",
            )
            row[key] = value
        record_id = str(row["record_id"])
        _require(record_id not in seen, f"duplicate clean record_id: {record_id}")
        seen.add(record_id)
        row["payload_sha256"] = _sha256(
            raw_row.get("payload_sha256"),
            f"inventory.records[{index}].payload_sha256",
        )
        row["payload_bytes"] = _strict_int(
            raw_row.get("payload_bytes"),
            f"inventory.records[{index}].payload_bytes",
            minimum=1,
        )
        normalized.append(row)
    _require(len(normalized) == count, "inventory record count does not match rows")
    _require(
        normalized == sorted(normalized, key=lambda row: row["record_id"]),
        "inventory records are not sorted by record_id",
    )
    _require(
        _sha(_canonical_bytes(normalized)) == record_root,
        "record inventory digest does not match records",
    )
    payload_projection = [
        {
            "record_id": row["record_id"],
            "payload_sha256": row["payload_sha256"],
            "payload_bytes": row["payload_bytes"],
        }
        for row in normalized
    ]
    _require(
        _sha(_canonical_bytes(payload_projection)) == payload_root,
        "payload inventory digest does not match records",
    )
    _require(
        sum(int(row["payload_bytes"]) for row in normalized) == payload_bytes,
        "inventory payload byte sum drift",
    )
    return normalized


def _validate_evidence(raw: bytes, *, release: _PhysicalRelease) -> dict[str, Any]:
    _require(_sha(raw) == release.evidence_file_sha256, "evidence file SHA drift")
    evidence = _strict_json(raw, "evidence.json")
    _require(
        evidence.get("schema_version") == EVIDENCE_SCHEMA,
        "post-G05/G06 evidence schema drift",
    )
    identity = _sha256(
        evidence.get("materialization_identity_sha256"),
        "materialization_identity_sha256",
    )
    _require(identity == release.materialization_identity_sha256, "materialization identity drift")
    body = deepcopy(evidence)
    body.pop("materialization_identity_sha256", None)
    _require(
        _sha(_canonical_bytes(body, newline=True)) == identity,
        "post-G05/G06 evidence self-hash mismatch",
    )
    _require(
        evidence.get("repeat_materialization_byte_identical") is True,
        "internal repeat materialization proof missing",
    )
    guard = evidence.get("provenance_guard")
    _require(isinstance(guard, Mapping), "provenance guard missing")
    _require(
        guard.get("known_external_llm_contamination_absent") is True,
        "known external-LLM contamination is not absent",
    )
    _require(
        guard.get("whole_corpus_external_llm_cleanliness_claimed") is False,
        "whole-corpus external-LLM cleanliness overclaimed",
    )
    _require(
        evidence.get("remaining_materialization_blockers")
        == ["SUCCESSOR_CORPUS_AUTHORITY_REBUILD_REQUIRED"],
        "unexpected post-materialization blocker set",
    )
    _zero_truth(
        evidence.get("truth_boundary"),
        "materialization evidence truth",
        expected_keys=_EVIDENCE_TRUTH_KEYS,
    )
    return evidence


def _bind_records(
    raw: bytes,
    inventory_rows: Sequence[Mapping[str, Any]],
    *,
    release: _PhysicalRelease,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    _require(_sha(raw) == release.jsonl_sha256, "records JSONL SHA drift")
    records = _strict_jsonl(raw)
    _require(len(records) == release.record_count, "records JSONL count drift")
    by_id = {str(row["record_id"]): row for row in inventory_rows}
    _require(len(by_id) == len(inventory_rows), "inventory logical id collision")
    matcher_rows: list[dict[str, str]] = []
    projection: list[dict[str, Any]] = []
    seen: set[str] = set()
    payload_total = 0
    for index, record in enumerate(records):
        _require(set(record) == _RECORD_KEYS, f"records.jsonl[{index}] schema drift")
        text_values: dict[str, str] = {}
        for key in _RECORD_KEYS:
            value = record.get(key)
            _require(
                isinstance(value, str) and bool(value),
                f"records.jsonl[{index}].{key} must be non-empty text",
            )
            text_values[key] = value
        record_id = text_values["record_id"]
        _require(record_id not in seen, f"duplicate physical record_id: {record_id}")
        seen.add(record_id)
        _require(record_id in by_id, f"physical record missing from inventory: {record_id}")
        authority = by_id[record_id]
        for key in ("source_id", "family", "modality"):
            _require(
                text_values[key] == authority[key],
                f"physical provenance drift: {record_id}:{key}",
            )
        payload = text_values["normalized_payload"].encode("utf-8")
        payload_hash = _sha(payload)
        _require(
            payload_hash == authority["payload_sha256"],
            f"physical payload SHA drift: {record_id}",
        )
        _require(
            len(payload) == authority["payload_bytes"],
            f"physical payload byte drift: {record_id}",
        )
        _require(record_id != NOMIS_RECORD_ID, "known quarantined Nomis record re-entered")
        _require(text_values["family"] != NOMIS_FAMILY, "known quarantined Nomis family re-entered")
        _require(payload_hash != NOMIS_PAYLOAD_SHA256, "known quarantined Nomis payload re-entered")
        payload_total += len(payload)
        matcher_rows.append(
            {
                "record_id": record_id,
                "source_id": text_values["source_id"],
                "source_family": text_values["family"],
                "modality": text_values["modality"],
                "text": text_values["normalized_payload"],
            }
        )
        projection.append(
            {
                "record_id": record_id,
                "source_id": text_values["source_id"],
                "source_family": text_values["family"],
                "modality": text_values["modality"],
                "text_sha256": payload_hash,
                "text_utf8_bytes": len(payload),
            }
        )
    _require(seen == set(by_id), "physical/inventory logical id coverage mismatch")
    _require(payload_total == release.payload_bytes, "physical payload byte sum drift")
    matcher_rows.sort(key=lambda row: row["record_id"])
    projection.sort(key=lambda row: row["record_id"])
    return matcher_rows, projection


def _build_handoff(
    projection: Sequence[Mapping[str, Any]],
    *,
    release: _PhysicalRelease,
) -> dict[str, Any]:
    core: dict[str, Any] = {
        "schema_version": HANDOFF_SCHEMA,
        "postdedup_inventory_identity_sha256": release.inventory_file_sha256,
        "input_survivor_authority_sha256": release.clean_data526_evidence_identity_sha256,
        "retained_source_count": release.record_count,
        "matcher_input_projection": [dict(row) for row in projection],
        "matcher_input_projection_sha256": _sha(_canonical_bytes(projection)),
        "raw_text_persisted_in_evidence": False,
        "final_test_payload_accessed": False,
        "final_test_outcomes_accessed": False,
        "authorized_training_exposure": 0,
    }
    core["handoff_identity_sha256"] = _sha(_canonical_bytes(core))
    return core


def _build_receipt(
    handoff: Mapping[str, Any],
    *,
    release: _PhysicalRelease,
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "physical_run_id": release.run_id,
        "physical_job_id": release.job_id,
        "physical_artifact_id": release.artifact_id,
        "physical_artifact_zip_sha256": release.artifact_zip_sha256,
        "authoritative_main_sha": release.authoritative_main_sha,
        "execution_carrier_head_sha": release.execution_carrier_head_sha,
        "input_files_sha256": {
            "records.jsonl": release.jsonl_sha256,
            "inventory.json": release.inventory_file_sha256,
            "evidence.json": release.evidence_file_sha256,
            "proof.json": release.proof_file_sha256,
        },
        "record_count": release.record_count,
        "payload_bytes": release.payload_bytes,
        "record_inventory_digest_sha256": release.record_inventory_digest_sha256,
        "payload_inventory_digest_sha256": release.payload_inventory_digest_sha256,
        "materialization_identity_sha256": release.materialization_identity_sha256,
        "clean_data526_evidence_identity_sha256": release.clean_data526_evidence_identity_sha256,
        "g05_execution_identity_sha256": release.g05_execution_identity_sha256,
        "g06_execution_identity_sha256": release.g06_execution_identity_sha256,
        "g06_envelope_identity_sha256": release.g06_envelope_identity_sha256,
        "g06_terminal_qualification_identity_sha256": (
            release.g06_terminal_qualification_identity_sha256
        ),
        "composition_preflight_identity_sha256": release.composition_preflight_identity_sha256,
        "training_handoff_identity_sha256": handoff["handoff_identity_sha256"],
        "matcher_input_projection_sha256": handoff["matcher_input_projection_sha256"],
        "known_nomis_pr462_payload_absent": True,
        "provenance_trace_bound": True,
        "durable_receipt_text_free": True,
        "current_retained_corpus_launch_authoritative": False,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
        "whole_corpus_external_llm_cleanliness_claimed": False,
    }
    receipt["receipt_identity_sha256"] = _sha(_canonical_bytes(receipt))
    return receipt


def verify_clean_retained_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_training_handoff_identity_sha256: str,
    expected_matcher_input_projection_sha256: str,
) -> None:
    _require(isinstance(receipt, Mapping), "clean retained receipt must be an object")
    _require(set(receipt) == _RECEIPT_KEYS, "clean retained receipt key set drift")
    _require(receipt.get("schema_version") == RECEIPT_SCHEMA, "clean retained receipt schema drift")
    claimed = _sha256(receipt.get("receipt_identity_sha256"), "receipt_identity_sha256")
    body = deepcopy(dict(receipt))
    body.pop("receipt_identity_sha256", None)
    _require(_sha(_canonical_bytes(body)) == claimed, "clean retained receipt self-hash mismatch")
    hashes = receipt.get("input_files_sha256")
    _require(isinstance(hashes, Mapping), "input_files_sha256 missing")
    _require(set(hashes) == _RECEIPT_FILE_KEYS, "input file hash map is not closed-world")
    expected_files = {
        "records.jsonl": OUTPUT_JSONL_SHA256,
        "inventory.json": OUTPUT_INVENTORY_FILE_SHA256,
        "evidence.json": OUTPUT_EVIDENCE_FILE_SHA256,
        "proof.json": OUTPUT_PROOF_FILE_SHA256,
    }
    _require(dict(hashes) == expected_files, "input file hash map is not the released clean set")
    _require(receipt.get("physical_run_id") == PHYSICAL_RUN_ID, "physical run drift")
    _require(receipt.get("physical_job_id") == PHYSICAL_JOB_ID, "physical job drift")
    _require(receipt.get("physical_artifact_id") == PHYSICAL_ARTIFACT_ID, "physical artifact drift")
    _require(
        receipt.get("physical_artifact_zip_sha256") == PHYSICAL_ARTIFACT_ZIP_SHA256,
        "physical artifact ZIP drift",
    )
    _require(receipt.get("record_count") == OUTPUT_RECORD_COUNT, "receipt record count drift")
    _require(receipt.get("payload_bytes") == OUTPUT_PAYLOAD_BYTES, "receipt payload bytes drift")
    _require(
        receipt.get("training_handoff_identity_sha256")
        == _sha256(
            expected_training_handoff_identity_sha256,
            "expected_training_handoff_identity_sha256",
        ),
        "training handoff identity is not independently expected",
    )
    _require(
        receipt.get("matcher_input_projection_sha256")
        == _sha256(
            expected_matcher_input_projection_sha256,
            "expected_matcher_input_projection_sha256",
        ),
        "matcher input projection is not independently expected",
    )
    for key in (
        "known_nomis_pr462_payload_absent",
        "provenance_trace_bound",
        "durable_receipt_text_free",
    ):
        _require(receipt.get(key) is True, f"receipt proof weakened: {key}")
    for key in (
        "current_retained_corpus_launch_authoritative",
        "tokenizer_fit_authorized",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "foreign_pretrained_weights",
        "whole_corpus_external_llm_cleanliness_claimed",
    ):
        _require(receipt.get(key) is False, f"receipt truth widened: {key}")
    for key in (
        "authorized_optimized_target_exposure",
        "optimizer_updates_executed_on_real_targets",
    ):
        _require(
            type(receipt.get(key)) is int and receipt.get(key) == 0,
            f"receipt truth widened: {key}",
        )
    _require(
        PR462_AUTHORITY_SHA256.encode("ascii") not in _canonical_bytes(receipt),
        "PR462 authority was persisted into clean receipt",
    )


def _prepare_with_release(
    records_jsonl: bytes,
    inventory_json: bytes,
    evidence_json: bytes,
    proof_json: bytes,
    *,
    release: _PhysicalRelease,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    _validate_proof(proof_json, release=release)
    inventory_rows = _validate_inventory(inventory_json, release=release)
    _validate_evidence(evidence_json, release=release)
    matcher_rows, projection = _bind_records(records_jsonl, inventory_rows, release=release)
    handoff = _build_handoff(projection, release=release)
    receipt = _build_receipt(handoff, release=release)
    return matcher_rows, handoff, receipt


def prepare_current_clean_retained_data232_rows(
    records_jsonl: bytes,
    inventory_json: bytes,
    evidence_json: bytes,
    proof_json: bytes,
) -> tuple[list[dict[str, str]], dict[str, Any], dict[str, Any]]:
    """Verify the immutable clean release and expose DATA-232 rows in memory."""
    rows, handoff, receipt = _prepare_with_release(
        records_jsonl,
        inventory_json,
        evidence_json,
        proof_json,
        release=_RELEASE,
    )
    verify_clean_retained_receipt(
        receipt,
        expected_training_handoff_identity_sha256=handoff["handoff_identity_sha256"],
        expected_matcher_input_projection_sha256=handoff[
            "matcher_input_projection_sha256"
        ],
    )
    return rows, handoff, receipt
