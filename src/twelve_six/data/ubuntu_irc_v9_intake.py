"""Authenticate the real Ubuntu IRC candidate for incumbent expanded-V9 dedup intake.

This module is deliberately an adapter, not a duplicate matcher. It binds the exact
merged Ubuntu rights/execution authority, repaired real-execution evidence, terminal
expanded-V9 implementation bytes, and ephemeral candidate JSONL before projecting
rows into the incumbent matcher input shape. Returned payload bytes are execution-local;
the receipt is text-free and grants zero corpus/training credit.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from twelve_six.ubuntu_irc_execution_rights_crossbind import (
    AUTHORITY_ID as CROSSBIND_AUTHORITY_ID,
)
from twelve_six.ubuntu_irc_execution_rights_crossbind import (
    CANDIDATE_SHA256,
    REPAIRED_EVIDENCE_BLOB,
    SOURCE_DATASET,
    SOURCE_FILE,
    SOURCE_REVISION,
    SOURCE_SHA256,
    UbuntuIrcCrossbindError,
    git_blob_sha1,
    load_json_bytes,
    validate_crossbind,
)

CROSSBIND_BLOB_SHA1 = "8cae172d3c4c86bf64ce5bedb966210448fdf3a5"
EVIDENCE_BLOB_SHA1 = REPAIRED_EVIDENCE_BLOB
INCUMBENT_V9_PRODUCT_HEAD = "5dbf143c7ca15999eadb28fecb952118b59040de"
INCUMBENT_V9_FACADE_BLOB_SHA1 = "2916d5d76d708300ad2e1794829f583bced7c84c"
CANDIDATE_RECORDS = 972
CANDIDATE_NORMALIZED_BYTES = 4_799_981
SOURCE_LABEL = "ubuntu-chat"
SOURCE_FAMILY = SOURCE_DATASET
MODALITY = "en"
RECEIPT_SCHEMA = "12-6.d03-ubuntu-irc-expanded-v9-intake-receipt.v1"
STATUS = "AUTHENTICATED_MATCHER_INPUT_READY_ZERO_CREDIT"
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATE_KEYS = {
    "record_id",
    "source_key",
    "source_label",
    "normalized_sha256",
    "normalized_bytes",
    "training_eligible",
    "evaluation_eligible",
    "text",
}
_EVIDENCE_ROOT_KEYS = {
    "schema_version",
    "worker_issue",
    "incumbent_pr",
    "execution_profile",
    "execution_head_sha",
    "execution_binding",
    "workflow_run_id",
    "workflow_job_id",
    "workflow_run_attempt",
    "workflow_conclusion",
    "runner",
    "focused_adversarial_tests",
    "materialization",
    "artifact",
    "truth_boundary",
    "next_required_gates",
}


class UbuntuIrcV9IntakeError(ValueError):
    """Raised when any authority, candidate, or incumbent-V9 binding drifts."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise UbuntuIrcV9IntakeError(message)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _strict_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        return load_json_bytes(payload, label)
    except UbuntuIrcCrossbindError as exc:
        raise UbuntuIrcV9IntakeError(str(exc)) from exc


def _expect_int(value: object, expected: int, label: str) -> None:
    _require(type(value) is int and value == expected, f"{label} drift")


def _expect_bool(value: object, expected: bool, label: str) -> None:
    _require(type(value) is bool and value is expected, f"{label} drift")


def _expect_str(value: object, expected: str, label: str) -> None:
    _require(type(value) is str and value == expected, f"{label} drift")


def _validate_execution_evidence(
    evidence: Mapping[str, Any],
    evidence_bytes: bytes,
    crossbind: Mapping[str, Any],
) -> None:
    _require(git_blob_sha1(evidence_bytes) == EVIDENCE_BLOB_SHA1, "execution evidence bytes drift")
    _require(set(evidence) == _EVIDENCE_ROOT_KEYS, "execution evidence root keys drift")
    _expect_str(
        evidence.get("schema_version"),
        "12-6.d03-common-pile-ubuntu-irc-real-execution.v1",
        "execution evidence schema",
    )
    _expect_int(evidence.get("worker_issue"), 1096, "execution worker")
    _expect_int(evidence.get("incumbent_pr"), 913, "execution incumbent PR")
    _expect_str(
        evidence.get("execution_profile"),
        "LOCAL_FREE_GITHUB_HOSTED_UBUNTU",
        "execution profile",
    )
    _expect_str(
        evidence.get("execution_head_sha"),
        "3400e2cd3c62a360b25282310390640a7f20fbc9",
        "execution head",
    )
    _expect_int(evidence.get("workflow_run_id"), 34549598850, "workflow run")
    _expect_int(evidence.get("workflow_job_id"), 103109520811, "workflow job")
    _expect_int(evidence.get("workflow_run_attempt"), 1, "workflow attempt")
    _expect_str(evidence.get("workflow_conclusion"), "success", "workflow conclusion")

    focused = evidence.get("focused_adversarial_tests")
    _require(type(focused) is dict, "focused test evidence missing")
    _expect_str(focused.get("conclusion"), "success", "focused test conclusion")
    _expect_int(focused.get("passed"), 22, "focused test count")

    material = evidence.get("materialization")
    _require(type(material) is dict, "materialization evidence missing")
    expected_material = {
        "status": "TWO_INDEPENDENT_MATERIALIZATIONS_BYTE_IDENTICAL",
        "independent_build_count": 2,
        "independent_acquisition_count": 2,
        "candidate_byte_identical": True,
        "report_byte_identical": True,
        "source_dataset": SOURCE_DATASET,
        "source_revision": SOURCE_REVISION,
        "source_file": SOURCE_FILE,
        "source_sha256": SOURCE_SHA256,
        "retained_records": CANDIDATE_RECORDS,
        "retained_normalized_utf8_bytes": CANDIDATE_NORMALIZED_BYTES,
        "candidate_payload_sha256": CANDIDATE_SHA256,
        "candidate_file_sha256": CANDIDATE_SHA256,
    }
    for key, expected in expected_material.items():
        actual = material.get(key)
        _require(
            type(actual) is type(expected) and actual == expected,
            f"materialization.{key} drift",
        )
    _expect_int(material.get("observed_source_bytes"), 154_409_587, "observed source bytes")
    _expect_int(material.get("scanned_records"), 4096, "scanned records")
    _expect_int(material.get("candidate_payload_bytes"), 5_133_301, "candidate payload bytes")

    repaired = crossbind.get("repaired_execution_authority")
    _require(type(repaired) is dict, "crossbind repaired authority missing")
    for key, evidence_key in (
        ("source_dataset", "source_dataset"),
        ("source_revision", "source_revision"),
        ("source_file", "source_file"),
        ("source_sha256", "source_sha256"),
        ("candidate_payload_sha256", "candidate_payload_sha256"),
        ("retained_records", "retained_records"),
        ("retained_normalized_utf8_bytes", "retained_normalized_utf8_bytes"),
    ):
        _require(
            type(repaired.get(key)) is type(material.get(evidence_key))
            and repaired.get(key) == material.get(evidence_key),
            f"crossbind/evidence mismatch: {key}",
        )

    truth = evidence.get("truth_boundary")
    _require(type(truth) is dict, "execution truth boundary missing")
    for field in (
        "canonical_corpus_admitted",
        "training_eligible",
        "evaluation_eligible",
        "tokenizer_fit_authorized",
        "model_training_executed",
        "learned_weights_created",
        "final_test_accessed",
        "paid_compute_used",
        "foreign_pretrained_weights_used",
        "external_llm_or_api_used_for_data_or_intelligence",
    ):
        _expect_bool(truth.get(field), False, f"truth_boundary.{field}")
    for field in (
        "training_authorized_bytes",
        "canonical_capacity_credited",
        "family_credit_added",
        "unique_causal_loss_positions_authorized",
        "optimizer_updates",
    ):
        _expect_int(truth.get(field), 0, f"truth_boundary.{field}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise UbuntuIrcV9IntakeError(f"duplicate candidate JSON key: {key}")
        result[key] = value
    return result


def _parse_candidate(candidate_bytes: bytes) -> list[dict[str, Any]]:
    _require(type(candidate_bytes) is bytes and bool(candidate_bytes), "candidate bytes missing")
    _require(_sha256(candidate_bytes) == CANDIDATE_SHA256, "candidate payload SHA-256 drift")
    _require(candidate_bytes.endswith(b"\n"), "candidate JSONL must end with newline")

    rows: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_payload_hashes: set[str] = set()
    total = 0
    for line_no, raw_line in enumerate(candidate_bytes.splitlines(), 1):
        _require(bool(raw_line), f"blank candidate JSONL line: {line_no}")
        try:
            row = json.loads(
                raw_line.decode("utf-8", errors="strict"),
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise UbuntuIrcV9IntakeError(f"candidate JSONL malformed at line {line_no}") from exc
        _require(type(row) is dict, f"candidate row {line_no} must be object")
        _require(set(row) == _CANDIDATE_KEYS, f"candidate row {line_no} keys drift")

        record_id = row.get("record_id")
        _require(
            type(record_id) is str
            and bool(record_id)
            and record_id == record_id.strip()
            and not any(ch.isspace() for ch in record_id),
            f"candidate record id invalid at line {line_no}",
        )
        _require(record_id not in seen_ids, f"duplicate candidate record id: {record_id}")
        seen_ids.add(record_id)
        _expect_str(row.get("source_key"), "ubuntu_irc", f"candidate source_key line {line_no}")
        _expect_str(row.get("source_label"), SOURCE_LABEL, f"candidate source_label line {line_no}")
        _expect_bool(row.get("training_eligible"), False, f"candidate training flag line {line_no}")
        _expect_bool(row.get("evaluation_eligible"), False, f"candidate evaluation flag line {line_no}")

        text = row.get("text")
        _require(type(text) is str and bool(text), f"candidate text invalid at line {line_no}")
        payload = text.encode("utf-8")
        claimed_bytes = row.get("normalized_bytes")
        _require(
            type(claimed_bytes) is int and claimed_bytes > 0 and claimed_bytes == len(payload),
            f"candidate normalized bytes drift at line {line_no}",
        )
        claimed_sha = row.get("normalized_sha256")
        _require(
            type(claimed_sha) is str
            and _HEX64.fullmatch(claimed_sha) is not None
            and claimed_sha == _sha256(payload),
            f"candidate normalized SHA-256 drift at line {line_no}",
        )
        _require(claimed_sha not in seen_payload_hashes, "duplicate normalized candidate payload")
        seen_payload_hashes.add(claimed_sha)
        total += claimed_bytes
        rows.append(row)

    _require(len(rows) == CANDIDATE_RECORDS, "candidate record count drift")
    _require(total == CANDIDATE_NORMALIZED_BYTES, "candidate normalized byte total drift")
    return rows


def _build_matcher_inputs(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    inventory_rows: list[dict[str, Any]] = []
    payloads: dict[str, bytes] = {}
    acquisition_url = (
        "https://huggingface.co/datasets/common-pile/ubuntu_irc/resolve/"
        f"{SOURCE_REVISION}/{SOURCE_FILE}"
    )
    for row in rows:
        record_id = row["record_id"]
        payload = row["text"].encode("utf-8")
        payload_sha = row["normalized_sha256"]
        source_id = f"ubuntu-irc:{record_id}"
        _require(source_id not in payloads, f"duplicate matcher source id: {source_id}")
        stable_origin = f"{SOURCE_DATASET}@{SOURCE_REVISION}:{SOURCE_FILE}:{record_id}"
        inventory_rows.append(
            {
                "source_id": source_id,
                "source_family": SOURCE_FAMILY,
                "stable_origin_id": stable_origin,
                "stable_object_id": f"sha256:{payload_sha}",
                "modality": MODALITY,
                "evidence_status": "DEDICATED_TERMINAL",
                "authority_ref": f"{CROSSBIND_AUTHORITY_ID}:{CANDIDATE_SHA256}",
                "declared_capacity_bytes": len(payload),
                "expected_raw_bytes": len(payload),
                "expected_raw_sha256": payload_sha,
                "acquisition_url": acquisition_url,
                "origin_key": f"ubuntu-irc:{record_id}",
            }
        )
        payloads[source_id] = payload
    return inventory_rows, payloads


def prepare_ubuntu_v9_intake(
    *,
    incumbent_v9_product_head: str,
    incumbent_v9_facade_bytes: bytes,
    crossbind_bytes: bytes,
    rights_authority_bytes: bytes,
    parent_registry_bytes: bytes,
    execution_evidence_bytes: bytes,
    candidate_bytes: bytes,
) -> tuple[list[dict[str, Any]], dict[str, bytes], dict[str, Any]]:
    """Return authenticated matcher rows/payloads plus a durable text-free receipt."""

    _expect_str(
        incumbent_v9_product_head,
        INCUMBENT_V9_PRODUCT_HEAD,
        "incumbent expanded-V9 product head",
    )
    _require(
        git_blob_sha1(incumbent_v9_facade_bytes) == INCUMBENT_V9_FACADE_BLOB_SHA1,
        "incumbent expanded-V9 facade bytes drift",
    )
    _require(git_blob_sha1(crossbind_bytes) == CROSSBIND_BLOB_SHA1, "crossbind bytes drift")

    crossbind = _strict_object(crossbind_bytes, "Ubuntu crossbind")
    rights = _strict_object(rights_authority_bytes, "Ubuntu rights authority")
    parent = _strict_object(parent_registry_bytes, "Common Pile rights registry")
    evidence = _strict_object(execution_evidence_bytes, "Ubuntu execution evidence")
    try:
        status = validate_crossbind(crossbind, rights, parent, rights_authority_bytes)
    except UbuntuIrcCrossbindError as exc:
        raise UbuntuIrcV9IntakeError(str(exc)) from exc
    _expect_str(status, "REPAIRED_EXECUTION_RIGHTS_CROSSBOUND_ZERO_CREDIT", "crossbind status")
    _validate_execution_evidence(evidence, execution_evidence_bytes, crossbind)

    rows = _parse_candidate(candidate_bytes)
    inventory_rows, payloads = _build_matcher_inputs(rows)
    projection = [
        {
            **row,
            "payload_sha256": _sha256(payloads[row["source_id"]]),
        }
        for row in inventory_rows
    ]
    matcher_input_sha256 = _sha256(_canonical(projection))
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "status": STATUS,
        "incumbent_v9_product_head": INCUMBENT_V9_PRODUCT_HEAD,
        "incumbent_v9_facade_blob_sha1": INCUMBENT_V9_FACADE_BLOB_SHA1,
        "crossbind_blob_sha1": CROSSBIND_BLOB_SHA1,
        "execution_evidence_blob_sha1": EVIDENCE_BLOB_SHA1,
        "candidate_payload_sha256": CANDIDATE_SHA256,
        "candidate_records": CANDIDATE_RECORDS,
        "candidate_normalized_utf8_bytes": CANDIDATE_NORMALIZED_BYTES,
        "matcher_input_sha256": matcher_input_sha256,
        "matcher_input_source_count": len(inventory_rows),
        "matcher_input_payload_bytes": sum(len(payload) for payload in payloads.values()),
        "truth_boundary": {
            "durable_receipt_contains_source_text": False,
            "global_dedup_executed": False,
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "unique_causal_loss_positions_authorized": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "evaluation_eligible": False,
            "final_test_accessed": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
            "upstream_source_evidence_external_llm_or_api_used": False,
            "current_corpus_external_llm_free_claimed_by_this_adapter": False,
        },
    }
    return inventory_rows, payloads, receipt
