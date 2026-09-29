"""Authenticate Franko1901 real source execution for incumbent D03 global dedup.

This module does not implement matching and does not grant corpus or training
authority.  It binds the checked-in historical terminal receipt to the separately
audited repaired-head execution authority, validates the exact candidate byte
object, and projects every accepted row one-for-one into the incumbent V3
source-object wire contract.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

SOURCE_REPOSITORY: Final = "MurzikVasilyevich/ukr-proverbs-franko"
SOURCE_REVISION: Final = "7f62a9d8f0673d325b0a565d508461f4b44dae5b"
SOURCE_PATH: Final = "franko.csv"
SOURCE_ID: Final = "ua.verba.franko1901"
SOURCE_FAMILY: Final = "ua.verba.public-domain.franko1901"
SOURCE_LANGUAGE: Final = "uk"
SOURCE_CANDIDATE_MODALITY: Final = "natural_text"
MATCHER_MODALITY: Final = "uk"
SOURCE_GIT_BLOB_SHA1: Final = "45f33ac620907e1d1ed727524975b3a0fd1a0994"
SOURCE_BYTES: Final = 6_225_761
SOURCE_SHA256: Final = "10b0db2e6451b49d63f931a607a6b27fb10c11288e65802dcd1e6793a7a2586d"
LICENSE_ID: Final = "CC0-1.0"
SOURCE_ROWS_SEEN: Final = 30_936

CANDIDATE_RECORDS: Final = 30_660
CANDIDATE_TEXT_UTF8_BYTES: Final = 1_762_005
CANDIDATE_JSONL_BYTES: Final = 10_685_784
CANDIDATE_SHA256: Final = (
    "9b00da2a2a5110cc4e384711fffbbbdaa917904462ddd133e7178e2d07dcb5ec"
)
RECORD_INVENTORY_SHA256: Final = (
    "bdedac35f8d3ef1ba6faca6f79f002d1a67da1b13b5fade2a341c526d830a4c7"
)

HISTORICAL_TERMINAL_SCHEMA: Final = "12-6.d03-franko1901-terminal-execution.v1"
HISTORICAL_TERMINAL_BLOB_SHA1: Final = "35902a3e6724d3f45ffb2578b34b9dddd51c76ab"
HISTORICAL_EXECUTION_HEAD: Final = "181a54e6f7527ed66e59f72d42290a4e0013330f"
HISTORICAL_WORKFLOW_RUN_ID: Final = 34_479_584_975
HISTORICAL_WORKFLOW_JOB_ID: Final = 102_878_704_552

FRESH_AUTHORITY_SCHEMA: Final = "12-6.d03-franko1901-fresh-execution-authority.v1"
FRESH_AUTHORITY_STATUS: Final = (
    "FRESH_REPAIRED_HEAD_EXECUTION_INDEPENDENTLY_AUDITED_ZERO_CREDIT"
)
FRESH_AUTHORITY_BLOB_SHA1: Final = "941e17e255e20440d51fc24b59bffba1e787ad2c"
UPSTREAM_PRODUCT_PR: Final = 1025
UPSTREAM_PRODUCT_HEAD: Final = "5816f0ff4ca2f53053123063cd2471442a07d974"
UPSTREAM_SHARED_CI_RUN: Final = 34_556_243_290
FRESH_WORKFLOW_RUN_ID: Final = 34_556_645_720
FRESH_WORKFLOW_JOB_ID: Final = 103_130_616_446
FRESH_ARTIFACT_ID: Final = 10_182_799_644
FRESH_RECEIPT_MEMBER_SHA256: Final = (
    "a32ecead4d704e5f9ee858c0d1e7174cf5a7b5c7d36c2746b31c1571b3b5d67e"
)
INDEPENDENT_AUDIT_ISSUE: Final = 1177
INDEPENDENT_AUDIT_VERDICT: Final = "PASS_FOR_INTEGRATION_EXECUTION_EVIDENCE"

RECEIPT_SCHEMA: Final = "12-6.d03-franko1901-dedup-intake-receipt.v1"
INCUMBENT_INVENTORY_SCHEMA: Final = "12-6.next100-065-cross-source-dedup.v3"
SWARM_CLAIM: Final = 2075
SOURCE_ACQUISITION_URL: Final = (
    "https://raw.githubusercontent.com/"
    f"{SOURCE_REPOSITORY}/{SOURCE_REVISION}/{SOURCE_PATH}"
)

_CANDIDATE_KEYS = frozenset(
    {
        "language",
        "modality",
        "record_id",
        "source_family",
        "source_id",
        "text",
        "text_sha256",
        "text_utf8_bytes",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RECORD_ID_RE = re.compile(r"^franko1901:(\d{6}):([0-9a-f]{16})$")


class FrankoDedupIntakeError(RuntimeError):
    """Fail-closed Franko1901 dedup-intake error."""


@dataclass(frozen=True)
class FrankoAuthority:
    source_rows_seen: int = SOURCE_ROWS_SEEN
    candidate_records: int = CANDIDATE_RECORDS
    candidate_text_utf8_bytes: int = CANDIDATE_TEXT_UTF8_BYTES
    candidate_jsonl_bytes: int = CANDIDATE_JSONL_BYTES
    candidate_sha256: str = CANDIDATE_SHA256
    record_inventory_sha256: str = RECORD_INVENTORY_SHA256
    historical_terminal_blob_sha1: str = HISTORICAL_TERMINAL_BLOB_SHA1
    historical_execution_head: str = HISTORICAL_EXECUTION_HEAD
    historical_workflow_run_id: int = HISTORICAL_WORKFLOW_RUN_ID
    historical_workflow_job_id: int = HISTORICAL_WORKFLOW_JOB_ID
    fresh_authority_blob_sha1: str = FRESH_AUTHORITY_BLOB_SHA1
    product_pr: int = UPSTREAM_PRODUCT_PR
    product_head: str = UPSTREAM_PRODUCT_HEAD
    shared_ci_run: int = UPSTREAM_SHARED_CI_RUN
    fresh_workflow_run_id: int = FRESH_WORKFLOW_RUN_ID
    fresh_workflow_job_id: int = FRESH_WORKFLOW_JOB_ID
    fresh_artifact_id: int = FRESH_ARTIFACT_ID
    fresh_receipt_member_sha256: str = FRESH_RECEIPT_MEMBER_SHA256
    audit_issue: int = INDEPENDENT_AUDIT_ISSUE
    audit_verdict: str = INDEPENDENT_AUDIT_VERDICT


PRODUCTION_AUTHORITY: Final = FrankoAuthority()


@dataclass(frozen=True)
class FrankoProjection:
    receipt: dict[str, Any]
    sources: tuple[dict[str, Any], ...] | None
    payloads: dict[str, bytes] | None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FrankoDedupIntakeError(message)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _git_blob_sha1(raw: bytes) -> str:
    header = f"blob {len(raw)}\0".encode("ascii")
    return hashlib.sha1(header + raw, usedforsecurity=False).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _materializer_canonical_json_bytes(value: Any) -> bytes:
    return _canonical(value) + b"\n"


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise FrankoDedupIntakeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_object(raw: bytes, *, label: str) -> dict[str, Any]:
    _require(type(raw) is bytes and bool(raw), f"{label} is empty")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrankoDedupIntakeError(f"{label} is not strict UTF-8 JSON") from exc
    _require(type(value) is dict, f"{label} root must be exact object")
    return value


def _read_bytes_once(path: Path, *, label: str) -> bytes:
    _require(isinstance(path, Path), f"{label} path must be pathlib.Path")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise FrankoDedupIntakeError(f"cannot read {label}: {path}") from exc
    _require(bool(raw), f"{label} is empty")
    return raw


def _strict_equal(actual: Any, expected: Any, *, label: str) -> None:
    _require(
        type(actual) is type(expected) and actual == expected,
        f"{label} drift",
    )


def _mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    _require(type(value) is dict, f"{label} must be exact object")
    return value


def _zero_historical_truth(boundary: Mapping[str, Any]) -> None:
    expected = {
        "candidate_only": True,
        "canonical_capacity_credit_bytes": 0,
        "family_credit_authorized": False,
        "corpus_admitted": False,
        "global_dedup": "NOT_RUN",
        "evaluation_decontamination": "NOT_RUN",
        "post_composition_quality_privacy": "NOT_RUN",
        "balance_family_caps": "NOT_RUN_FOR_THIS_ADDITION",
        "tokenizer_fit_authorized": False,
        "training_authorized_bytes": 0,
        "authorized_unique_loss_positions": 0,
        "model_training_executed": False,
        "optimizer_updates": 0,
        "final_test_accessed": False,
        "paid_compute_used": False,
        "learned_20m_promoted": False,
    }
    for key, wanted in expected.items():
        _strict_equal(boundary.get(key), wanted, label=f"historical truth {key}")


def _zero_fresh_truth(boundary: Mapping[str, Any]) -> None:
    expected = {
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
        "foreign_pretrained_weights": False,
        "external_llm_or_api_used_for_data_or_intelligence": False,
    }
    for key, wanted in expected.items():
        _strict_equal(boundary.get(key), wanted, label=f"fresh truth {key}")


def _validate_historical_terminal(
    raw: bytes,
    authority: FrankoAuthority,
) -> dict[str, Any]:
    _require(
        _git_blob_sha1(raw) == authority.historical_terminal_blob_sha1,
        "historical terminal Git blob drift",
    )
    evidence = _strict_json_object(raw, label="historical terminal evidence")
    _strict_equal(
        evidence.get("schema_version"),
        HISTORICAL_TERMINAL_SCHEMA,
        label="historical terminal schema",
    )
    _strict_equal(
        evidence.get("decision"),
        "CANDIDATE_MATERIALIZED_ZERO_CREDIT",
        label="historical decision",
    )

    execution = _mapping(evidence.get("execution"), label="historical execution")
    historical_execution_expected = {
        "execution_head_sha": authority.historical_execution_head,
        "workflow_run_id": authority.historical_workflow_run_id,
        "job_id": authority.historical_workflow_job_id,
        "execution_profile": "LOCAL_FREE",
    }
    for key, wanted in historical_execution_expected.items():
        _strict_equal(execution.get(key), wanted, label=f"historical execution {key}")

    source = _mapping(evidence.get("source"), label="historical source")
    source_expected = {
        "source_id": SOURCE_ID,
        "source_family": SOURCE_FAMILY,
        "upstream_repository": SOURCE_REPOSITORY,
        "upstream_revision": SOURCE_REVISION,
        "source_path": SOURCE_PATH,
        "source_git_blob_sha1": SOURCE_GIT_BLOB_SHA1,
        "source_bytes": SOURCE_BYTES,
        "source_sha256": SOURCE_SHA256,
        "license_id": LICENSE_ID,
    }
    for key, wanted in source_expected.items():
        _strict_equal(source.get(key), wanted, label=f"historical source {key}")

    materialization = _mapping(
        evidence.get("materialization"),
        label="historical materialization",
    )
    materialization_expected = {
        "rows_seen": authority.source_rows_seen,
        "accepted_rows": authority.candidate_records,
        "accepted_text_utf8_bytes": authority.candidate_text_utf8_bytes,
        "accepted_payload_jsonl_bytes": authority.candidate_jsonl_bytes,
        "accepted_payload_jsonl_sha256": authority.candidate_sha256,
        "record_inventory_identity_sha256": authority.record_inventory_sha256,
    }
    for key, wanted in materialization_expected.items():
        _strict_equal(
            materialization.get(key),
            wanted,
            label=f"historical materialization {key}",
        )
    _zero_historical_truth(
        _mapping(evidence.get("truth_boundary"), label="historical truth boundary")
    )
    return evidence


def _validate_fresh_authority(
    raw: bytes,
    authority: FrankoAuthority,
) -> dict[str, Any]:
    _require(
        _git_blob_sha1(raw) == authority.fresh_authority_blob_sha1,
        "fresh execution authority Git blob drift",
    )
    evidence = _strict_json_object(raw, label="fresh execution authority")
    _require(
        set(evidence)
        == {
            "schema_version",
            "status",
            "execution_profile",
            "product",
            "execution",
            "independent_audit",
            "reproduced_candidate",
            "historical_terminal_link",
            "truth_boundary",
            "authority_identity_sha256",
        },
        "fresh execution authority top-level schema drift",
    )
    _strict_equal(
        evidence.get("schema_version"),
        FRESH_AUTHORITY_SCHEMA,
        label="fresh authority schema",
    )
    _strict_equal(
        evidence.get("status"),
        FRESH_AUTHORITY_STATUS,
        label="fresh authority status",
    )
    _strict_equal(
        evidence.get("execution_profile"),
        "LOCAL_FREE",
        label="fresh execution profile",
    )

    product = _mapping(evidence.get("product"), label="fresh product")
    product_expected = {
        "pr": authority.product_pr,
        "exact_head": authority.product_head,
        "shared_exact_head_ci_run": authority.shared_ci_run,
    }
    _require(set(product) == set(product_expected), "fresh product schema drift")
    for key, wanted in product_expected.items():
        _strict_equal(product.get(key), wanted, label=f"fresh product {key}")

    execution = _mapping(evidence.get("execution"), label="fresh execution")
    execution_expected = {
        "workflow_run_id": authority.fresh_workflow_run_id,
        "workflow_job_id": authority.fresh_workflow_job_id,
        "artifact_id": authority.fresh_artifact_id,
        "receipt_member_sha256": authority.fresh_receipt_member_sha256,
        "conclusion": "success",
    }
    _require(set(execution) == set(execution_expected), "fresh execution schema drift")
    for key, wanted in execution_expected.items():
        _strict_equal(execution.get(key), wanted, label=f"fresh execution {key}")

    audit = _mapping(evidence.get("independent_audit"), label="independent audit")
    audit_expected = {
        "issue": authority.audit_issue,
        "verdict": authority.audit_verdict,
    }
    _require(set(audit) == set(audit_expected), "independent audit schema drift")
    for key, wanted in audit_expected.items():
        _strict_equal(audit.get(key), wanted, label=f"independent audit {key}")

    reproduced = _mapping(
        evidence.get("reproduced_candidate"),
        label="fresh reproduced candidate",
    )
    reproduced_expected = {
        "accepted_rows": authority.candidate_records,
        "accepted_text_utf8_bytes": authority.candidate_text_utf8_bytes,
        "accepted_payload_jsonl_bytes": authority.candidate_jsonl_bytes,
        "accepted_payload_jsonl_sha256": authority.candidate_sha256,
        "record_inventory_identity_sha256": authority.record_inventory_sha256,
    }
    _require(
        set(reproduced) == set(reproduced_expected),
        "fresh reproduced candidate schema drift",
    )
    for key, wanted in reproduced_expected.items():
        _strict_equal(reproduced.get(key), wanted, label=f"fresh reproduced {key}")

    link = _mapping(
        evidence.get("historical_terminal_link"),
        label="historical terminal link",
    )
    link_expected = {
        "schema_version": HISTORICAL_TERMINAL_SCHEMA,
        "git_blob_sha1": authority.historical_terminal_blob_sha1,
        "historical_execution_head": authority.historical_execution_head,
        "historical_workflow_run_id": authority.historical_workflow_run_id,
        "historical_workflow_job_id": authority.historical_workflow_job_id,
    }
    _require(set(link) == set(link_expected), "historical terminal link schema drift")
    for key, wanted in link_expected.items():
        _strict_equal(link.get(key), wanted, label=f"historical terminal link {key}")

    _zero_fresh_truth(
        _mapping(evidence.get("truth_boundary"), label="fresh truth boundary")
    )

    identity = evidence.get("authority_identity_sha256")
    _require(
        type(identity) is str and _SHA256_RE.fullmatch(identity) is not None,
        "fresh authority identity malformed",
    )
    core = dict(evidence)
    core.pop("authority_identity_sha256")
    _require(
        _sha256(_canonical(core)) == identity,
        "fresh authority identity drift",
    )
    return evidence


def _record_parts(record_id: str, digest: str, *, previous_index: int) -> int:
    match = _RECORD_ID_RE.fullmatch(record_id)
    _require(match is not None, "record_id format drift")
    row_index = int(match.group(1))
    suffix = match.group(2)
    _require(1 <= row_index <= SOURCE_ROWS_SEEN, "record_id row index out of range")
    _require(row_index > previous_index, "record_id row indexes must be strictly increasing")
    _require(suffix == digest[:16], "record_id digest suffix drift")
    return row_index


def _validate_row(
    row: Mapping[str, Any],
    *,
    line_number: int,
    previous_index: int,
    authority: FrankoAuthority,
) -> tuple[dict[str, Any], str, bytes, int, dict[str, Any]]:
    _require(
        type(row) is dict and set(row) == _CANDIDATE_KEYS,
        f"candidate row {line_number} schema drift",
    )
    _strict_equal(row["language"], SOURCE_LANGUAGE, label=f"row {line_number} language")
    _strict_equal(
        row["modality"],
        SOURCE_CANDIDATE_MODALITY,
        label=f"row {line_number} modality",
    )
    _strict_equal(
        row["source_id"],
        SOURCE_ID,
        label=f"row {line_number} source_id",
    )
    _strict_equal(
        row["source_family"],
        SOURCE_FAMILY,
        label=f"row {line_number} source_family",
    )

    record_id = row["record_id"]
    text = row["text"]
    digest = row["text_sha256"]
    byte_count = row["text_utf8_bytes"]
    _require(type(record_id) is str and bool(record_id), f"record_id invalid at row {line_number}")
    _require(type(text) is str and bool(text), f"text invalid at row {line_number}")
    _require(
        type(digest) is str and _SHA256_RE.fullmatch(digest) is not None,
        f"text SHA malformed at row {line_number}",
    )
    _require(
        type(byte_count) is int and byte_count > 0,
        f"text UTF-8 byte count invalid at row {line_number}",
    )
    payload = text.encode("utf-8")
    _require(
        len(payload) == byte_count,
        f"text UTF-8 byte count mismatch at row {line_number}",
    )
    _require(
        _sha256(payload) == digest,
        f"text SHA drift at row {line_number}",
    )
    row_index = _record_parts(record_id, digest, previous_index=previous_index)
    _require(
        row_index <= authority.source_rows_seen,
        "record_id row index exceeds authority rows_seen",
    )

    matcher_source_id = f"franko1901-admitted:{record_id}"
    matcher_row = {
        "source_id": matcher_source_id,
        "source_family": SOURCE_FAMILY,
        "stable_origin_id": (
            f"{SOURCE_REPOSITORY}@{SOURCE_REVISION}:{SOURCE_PATH}#{record_id}"
        ),
        "stable_object_id": f"sha256:{digest}",
        "modality": MATCHER_MODALITY,
        "evidence_status": "DEDICATED_TERMINAL",
        "authority_ref": (
            f"AUDIT{authority.audit_issue}:PR{authority.product_pr}:"
            f"{authority.product_head}:run:{authority.fresh_workflow_run_id}:"
            f"job:{authority.fresh_workflow_job_id}:"
            f"receipt-sha256:{authority.fresh_receipt_member_sha256}"
        ),
        "declared_capacity_bytes": byte_count,
        "expected_raw_bytes": byte_count,
        "expected_raw_sha256": digest,
        "acquisition_url": SOURCE_ACQUISITION_URL,
        "origin_key": f"franko1901:{record_id}",
    }
    inventory_row = {
        "record_id": record_id,
        "text_sha256": digest,
        "text_utf8_bytes": byte_count,
    }
    return matcher_row, matcher_source_id, payload, row_index, inventory_row


def _receipt(
    *,
    authority: FrankoAuthority,
    source_count: int,
    payload_bytes: int,
    distinct_payload_hashes: int,
    projection_identity_sha256: str,
) -> dict[str, Any]:
    core: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "swarm_claim": SWARM_CLAIM,
        "source_family": SOURCE_FAMILY,
        "authority_chain": {
            "historical_terminal": {
                "schema_version": HISTORICAL_TERMINAL_SCHEMA,
                "git_blob_sha1": authority.historical_terminal_blob_sha1,
                "execution_head": authority.historical_execution_head,
                "workflow_run_id": authority.historical_workflow_run_id,
                "workflow_job_id": authority.historical_workflow_job_id,
                "role": "HISTORICAL_EXECUTED_BLOB_AUTHORITY_ONLY",
            },
            "fresh_repaired_head_execution": {
                "authority_schema": FRESH_AUTHORITY_SCHEMA,
                "authority_git_blob_sha1": authority.fresh_authority_blob_sha1,
                "product_pr": authority.product_pr,
                "product_head": authority.product_head,
                "shared_exact_head_ci_run": authority.shared_ci_run,
                "workflow_run_id": authority.fresh_workflow_run_id,
                "workflow_job_id": authority.fresh_workflow_job_id,
                "artifact_id": authority.fresh_artifact_id,
                "receipt_member_sha256": authority.fresh_receipt_member_sha256,
                "independent_audit_issue": authority.audit_issue,
                "independent_audit_verdict": authority.audit_verdict,
                "role": "REPAIRED_HEAD_EXECUTION_AUTHORITY",
            },
        },
        "candidate": {
            "jsonl_bytes": authority.candidate_jsonl_bytes,
            "jsonl_sha256": authority.candidate_sha256,
            "record_count": authority.candidate_records,
            "text_utf8_bytes": authority.candidate_text_utf8_bytes,
            "record_inventory_identity_sha256": authority.record_inventory_sha256,
        },
        "projection": {
            "matcher_inventory_schema": INCUMBENT_INVENTORY_SCHEMA,
            "source_object_count": source_count,
            "payload_utf8_bytes": payload_bytes,
            "distinct_payload_sha256_count": distinct_payload_hashes,
            "exact_duplicate_payload_alias_count": source_count - distinct_payload_hashes,
            "stable_object_aliases_preserved_for_matcher": True,
            "stable_origin_is_record_specific": True,
            "origin_key_is_record_specific": True,
            "pre_dedup_or_aggregation_performed": False,
            "matcher_source_inventory_identity_sha256": projection_identity_sha256,
            "raw_text_emitted_in_receipt": False,
        },
        "execution_gate": {
            "canonical_global_dedup_executed": False,
            "matcher_invoked_by_this_adapter": False,
            "status": "READY_FOR_TERMINAL_INCUMBENT_MATCHER_AFTER_DEPENDENCY_AUTHORITY",
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
            "foreign_pretrained_weights": False,
            "external_llm_or_api_used_for_data_or_intelligence": False,
        },
    }
    return {**core, "receipt_identity_sha256": _sha256(_canonical(core))}


def _project_candidate_bytes(
    candidate_raw: bytes,
    historical_terminal_raw: bytes,
    fresh_execution_authority_raw: bytes,
    *,
    authority: FrankoAuthority,
    retain_payloads: bool,
) -> FrankoProjection:
    _require(type(retain_payloads) is bool, "retain_payloads must be exact bool")
    _validate_historical_terminal(historical_terminal_raw, authority)
    _validate_fresh_authority(fresh_execution_authority_raw, authority)

    _require(type(candidate_raw) is bytes, "candidate must be exact bytes")
    _require(
        len(candidate_raw) == authority.candidate_jsonl_bytes,
        "candidate byte length drift",
    )
    _require(
        _sha256(candidate_raw) == authority.candidate_sha256,
        "candidate SHA-256 drift",
    )

    sources: list[dict[str, Any]] | None = [] if retain_payloads else None
    payloads: dict[str, bytes] | None = {} if retain_payloads else None
    seen_record_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    payload_hashes: set[str] = set()
    inventory_projection: list[dict[str, Any]] = []
    projected_for_identity: list[dict[str, Any]] = []
    total_bytes = 0
    previous_index = 0

    lines = candidate_raw.splitlines(keepends=True)
    _require(len(lines) == authority.candidate_records, "candidate row count drift")
    for line_number, line in enumerate(lines, 1):
        _require(line.endswith(b"\n"), f"candidate newline drift at row {line_number}")
        _require(bool(line.strip()), f"blank candidate row at {line_number}")
        try:
            row = json.loads(
                line.decode("utf-8", errors="strict"),
                object_pairs_hook=_reject_duplicate_pairs,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FrankoDedupIntakeError(
                f"candidate row {line_number} is not strict UTF-8 JSON"
            ) from exc
        _require(type(row) is dict, f"candidate row {line_number} must be exact object")
        record_id = row.get("record_id")
        _require(
            type(record_id) is str and record_id not in seen_record_ids,
            f"duplicate/invalid record_id at row {line_number}",
        )
        seen_record_ids.add(record_id)
        matcher_row, source_id, payload, row_index, inventory_row = _validate_row(
            row,
            line_number=line_number,
            previous_index=previous_index,
            authority=authority,
        )
        previous_index = row_index
        _require(source_id not in seen_source_ids, f"source_id collision at row {line_number}")
        seen_source_ids.add(source_id)
        total_bytes += len(payload)
        payload_hashes.add(row["text_sha256"])
        inventory_projection.append(inventory_row)
        projected_for_identity.append(matcher_row)
        if sources is not None and payloads is not None:
            sources.append(matcher_row)
            payloads[source_id] = payload

    _require(
        total_bytes == authority.candidate_text_utf8_bytes,
        "candidate text UTF-8 byte total drift",
    )
    _require(
        len(seen_record_ids) == authority.candidate_records,
        "candidate record identity cardinality drift",
    )
    inventory_identity = _sha256(
        _materializer_canonical_json_bytes(inventory_projection)
    )
    _require(
        inventory_identity == authority.record_inventory_sha256,
        "record inventory identity drift",
    )
    projection_identity = _sha256(_canonical(projected_for_identity))
    receipt = _receipt(
        authority=authority,
        source_count=len(projected_for_identity),
        payload_bytes=total_bytes,
        distinct_payload_hashes=len(payload_hashes),
        projection_identity_sha256=projection_identity,
    )
    return FrankoProjection(
        receipt=receipt,
        sources=tuple(sources) if sources is not None else None,
        payloads=payloads,
    )


def validate_and_project_franko(
    candidate_jsonl: Path,
    historical_terminal_evidence_json: Path,
    fresh_execution_authority_json: Path,
    *,
    retain_payloads: bool = True,
) -> FrankoProjection:
    """Validate exact Franko bytes and project them one-for-one for V3 matching.

    Every file is read exactly once, avoiding a hash-then-reopen TOCTOU gap.
    This adapter never invokes the matcher and never grants downstream credit.
    """

    candidate_raw = _read_bytes_once(candidate_jsonl, label="candidate JSONL")
    historical_raw = _read_bytes_once(
        historical_terminal_evidence_json,
        label="historical terminal evidence",
    )
    fresh_raw = _read_bytes_once(
        fresh_execution_authority_json,
        label="fresh execution authority",
    )
    return _project_candidate_bytes(
        candidate_raw,
        historical_raw,
        fresh_raw,
        authority=PRODUCTION_AUTHORITY,
        retain_payloads=retain_payloads,
    )
