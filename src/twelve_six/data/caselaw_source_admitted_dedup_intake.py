"""Authenticate exact PR #1347 Caselaw rows for incumbent D03 global dedup.

This module does not implement matching. It validates the exact source-admitted
candidate and projects every row one-for-one into the incumbent V3 source-object
wire contract. Matching remains owned by the canonical global-dedup lineage.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

SOURCE_DATASET: Final = "common-pile/caselaw_access_project"
SOURCE_REVISION: Final = "31e65135501af50f8285b52489bc3b39fd0fc5d5"
SOURCE_FAMILY: Final = "en.common-pile.caselaw"
SOURCE_KIND: Final = "Caselaw Access Project"
RIGHTS_BASIS: Final = "UPSTREAM_PUBLIC_DOMAIN_METADATA_REVIEW_REQUIRED"
SOURCE_ORDERING_POLICY: Final = "PRESERVE_ORIGINAL_CAP_00044_THEN_APPEND_CAP_00043"
SOURCE_AUTHORITY_URL: Final = (
    "https://huggingface.co/datasets/common-pile/caselaw_access_project/tree/"
    + SOURCE_REVISION
)
UPSTREAM_PRODUCT_PR: Final = 1347
UPSTREAM_FINAL_HEAD: Final = "deaf0730fe04a12e9abb8f3cecb14d6ad2cc7a4d"
UPSTREAM_EVIDENCE_BLOB_SHA1: Final = "5773169a2d78f0a7cdc435ca13e05e6ce8931004"
UPSTREAM_EXECUTION_HEAD: Final = "0978645d5a8c979852a3c5c401c600cddf77e73a"
UPSTREAM_WORKFLOW_RUN_ID: Final = 34_674_050_719
UPSTREAM_WORKFLOW_JOB_ID: Final = 103_500_589_035
CANDIDATE_RECORDS: Final = 5_658
CANDIDATE_NORMALIZED_UTF8_BYTES: Final = 5_962_147
CANDIDATE_SHA256: Final = (
    "45043f528ee83cb87a94f42ca9164c16463502308bf849a6736498d3b8caa273"
)
EVIDENCE_SCHEMA: Final = "12-6.d03-caselaw-source-admission-real-execution.v1"
RECEIPT_SCHEMA: Final = "12-6.d03-caselaw-dedup-intake-receipt.v1"
INCUMBENT_INVENTORY_SCHEMA: Final = "12-6.next100-065-cross-source-dedup.v3"
SWARM_CLAIM: Final = 1906

SOURCE_OBJECTS: Final = (
    (
        "cap_00044.jsonl.gz",
        9_441_419,
        "e04bdff817b34d5fd2ab0b4aff272adfe6e863aacef7d4cd24db47e4b7c8db33",
    ),
    (
        "cap_00043.jsonl.gz",
        10_476_512,
        "f299c45effc957e1e02d3930e68c5d6dc643eb9d585093ff7445f36095c8531f",
    ),
)

_CANDIDATE_KEYS = frozenset(
    {
        "record_id",
        "normalized_sha256",
        "normalized_utf8_bytes",
        "text",
        "source_family",
        "source_kind",
        "rights_basis",
        "training_eligible",
        "evaluation_eligible",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_NEXT_GATES = (
    "current_global_exact_near_lineage_dedup",
    "fresh_reserved_evaluation_decontamination",
    "post_composition_quality_privacy",
    "balance_and_family_caps",
    "cluster_safe_split",
    "deterministic_packing_two_clean_builds",
    "positive_unique_loss_ledger",
)


class CaselawDedupIntakeError(RuntimeError):
    """Fail-closed Caselaw source-admitted dedup-intake error."""


@dataclass(frozen=True)
class CaselawAuthority:
    product_pr: int = UPSTREAM_PRODUCT_PR
    final_head: str = UPSTREAM_FINAL_HEAD
    evidence_blob_sha1: str = UPSTREAM_EVIDENCE_BLOB_SHA1
    execution_head: str = UPSTREAM_EXECUTION_HEAD
    workflow_run_id: int = UPSTREAM_WORKFLOW_RUN_ID
    workflow_job_id: int = UPSTREAM_WORKFLOW_JOB_ID
    candidate_records: int = CANDIDATE_RECORDS
    candidate_normalized_utf8_bytes: int = CANDIDATE_NORMALIZED_UTF8_BYTES
    candidate_sha256: str = CANDIDATE_SHA256


PRODUCTION_AUTHORITY: Final = CaselawAuthority()


@dataclass(frozen=True)
class CaselawProjection:
    receipt: dict[str, Any]
    sources: tuple[dict[str, Any], ...] | None
    payloads: dict[str, bytes] | None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CaselawDedupIntakeError(message)


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


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CaselawDedupIntakeError(f"duplicate JSON key: {key}")
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
        raise CaselawDedupIntakeError(f"{label} is not strict UTF-8 JSON") from exc
    _require(type(value) is dict, f"{label} root must be exact object")
    return value


def _read_bytes_once(path: Path, *, label: str) -> bytes:
    _require(isinstance(path, Path), f"{label} path must be pathlib.Path")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise CaselawDedupIntakeError(f"cannot read {label}: {path}") from exc
    _require(bool(raw), f"{label} is empty")
    return raw


def _exact(actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str) -> None:
    for key, wanted in expected.items():
        value = actual.get(key)
        _require(
            type(value) is type(wanted) and value == wanted,
            f"{label} drift: {key}",
        )


def _validate_source_objects(value: Any) -> None:
    _require(type(value) is list and len(value) == len(SOURCE_OBJECTS), "source vector drift")
    for index, (actual, expected) in enumerate(zip(value, SOURCE_OBJECTS, strict=True)):
        _require(type(actual) is dict, f"source object {index} must be exact object")
        _require(
            set(actual) == {"file", "compressed_bytes", "sha256"},
            f"source object {index} schema drift",
        )
        file_name, byte_count, digest = expected
        _exact(
            actual,
            {"file": file_name, "compressed_bytes": byte_count, "sha256": digest},
            label=f"source object {index}",
        )


def _validate_evidence(raw: bytes, authority: CaselawAuthority) -> dict[str, Any]:
    _require(
        _git_blob_sha1(raw) == authority.evidence_blob_sha1,
        "upstream evidence Git blob drift",
    )
    evidence = _strict_json_object(raw, label="upstream execution evidence")
    _exact(
        evidence,
        {
            "schema_version": EVIDENCE_SCHEMA,
            "execution_profile": "LOCAL_FREE",
            "product_pr": authority.product_pr,
        },
        label="upstream execution evidence",
    )
    execution = evidence.get("successful_execution")
    _require(type(execution) is dict, "successful_execution missing")
    _exact(
        execution,
        {
            "workflow_run_id": authority.workflow_run_id,
            "job_id": authority.workflow_job_id,
            "execution_head_sha": authority.execution_head,
            "github_run_attempt": 1,
            "github_job": "bootstrap",
            "workflow_path": ".github/workflows/ci.yml",
            "event": "pull_request",
            "conclusion": "success",
        },
        label="successful execution",
    )

    source = evidence.get("source")
    _require(type(source) is dict, "source authority missing")
    _exact(
        source,
        {
            "dataset": SOURCE_DATASET,
            "revision": SOURCE_REVISION,
            "family": SOURCE_FAMILY,
            "ordering_policy": SOURCE_ORDERING_POLICY,
        },
        label="source authority",
    )
    _validate_source_objects(source.get("objects"))

    result = evidence.get("result")
    _require(type(result) is dict, "result authority missing")
    _exact(
        result,
        {
            "historical_product_replay_reproduced": True,
            "input_records": authority.candidate_records,
            "input_normalized_utf8_bytes": authority.candidate_normalized_utf8_bytes,
            "historical_candidate_sha256": authority.candidate_sha256,
            "source_admitted_records": authority.candidate_records,
            "source_admitted_normalized_utf8_bytes": (
                authority.candidate_normalized_utf8_bytes
            ),
            "source_admitted_candidate_sha256": authority.candidate_sha256,
            "candidate_denied_reason_counts": {},
        },
        label="source-admission result",
    )

    determinism = evidence.get("determinism")
    _require(type(determinism) is dict, "determinism evidence missing")
    _exact(
        determinism,
        {
            "independent_execution_passes": 2,
            "raw_source_objects_byte_identical": True,
            "historical_candidate_bytes_identical": True,
            "admitted_candidate_bytes_identical": True,
            "materializer_report_bytes_identical": True,
            "source_admission_report_bytes_identical": True,
        },
        label="determinism evidence",
    )
    content = evidence.get("content_boundary")
    _require(type(content) is dict, "content boundary missing")
    _exact(
        content,
        {
            "candidate_or_source_text_retained_in_git": False,
            "candidate_payload_uploaded_as_artifact": False,
            "durable_evidence_text_free": True,
        },
        label="content boundary",
    )

    claim = evidence.get("claim_boundary")
    _require(type(claim) is dict, "claim boundary missing")
    _exact(
        claim,
        {
            "real_source_admission_executed": True,
            "candidate_status": "CASELAW_SOURCE_ADMITTED_CANDIDATE_ONLY_ZERO_CREDIT",
            "corpus_admitted": False,
            "evaluation_eligible": False,
            "global_dedup_on_this_candidate": "NOT_RUN",
            "reserved_evaluation_decontamination_on_this_candidate": "NOT_RUN",
            "privacy_filter_on_this_candidate": "NOT_RUN",
            "quality_filter_on_this_candidate": "NOT_RUN",
            "family_balance_on_this_candidate": "NOT_RUN",
            "cluster_safe_split_on_this_candidate": "NOT_RUN",
            "deterministic_packing_on_this_candidate": "NOT_RUN",
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "authorized_unique_loss_positions": 0,
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates_executed_on_real_targets": 0,
            "model_training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights_used": False,
            "external_llm_or_api_used_for_data_or_intelligence": False,
        },
        label="claim boundary",
    )
    _require(
        evidence.get("next_required_before_any_training_credit")
        == list(_REQUIRED_NEXT_GATES),
        "required downstream gate vector drift",
    )
    return evidence


def _record_identity(record_id: str) -> str:
    return _sha256(b"caselaw-record-id-v1\0" + record_id.encode("utf-8"))


def _validate_row(
    row: Mapping[str, Any], *, line_number: int
) -> tuple[dict[str, Any], str, bytes]:
    _require(
        type(row) is dict and set(row) == _CANDIDATE_KEYS,
        f"candidate row {line_number} schema drift",
    )
    record_id = row["record_id"]
    normalized_sha = row["normalized_sha256"]
    normalized_bytes = row["normalized_utf8_bytes"]
    text = row["text"]
    _require(
        type(record_id) is str and 1 <= len(record_id) <= 256,
        f"record_id invalid at row {line_number}",
    )
    _require(
        type(normalized_sha) is str and _SHA256_RE.fullmatch(normalized_sha) is not None,
        f"normalized SHA malformed at row {line_number}",
    )
    _require(
        type(normalized_bytes) is int and normalized_bytes > 0,
        f"normalized byte count invalid at row {line_number}",
    )
    _require(type(text) is str and bool(text), f"text invalid at row {line_number}")
    _require(
        row["source_family"] == SOURCE_FAMILY,
        f"source family drift at row {line_number}",
    )
    _require(row["source_kind"] == SOURCE_KIND, f"source kind drift at row {line_number}")
    _require(row["rights_basis"] == RIGHTS_BASIS, f"rights basis drift at row {line_number}")
    _require(
        type(row["training_eligible"]) is bool and row["training_eligible"] is False,
        f"training eligibility drift at row {line_number}",
    )
    _require(
        type(row["evaluation_eligible"]) is bool and row["evaluation_eligible"] is False,
        f"evaluation eligibility drift at row {line_number}",
    )
    payload = text.encode("utf-8")
    _require(
        len(payload) == normalized_bytes,
        f"normalized byte count mismatch at row {line_number}",
    )
    _require(_sha256(payload) == normalized_sha, f"normalized SHA drift at row {line_number}")

    record_identity = _record_identity(record_id)
    source_id = f"caselaw-source-admitted:{record_identity}"
    matcher_row = {
        "source_id": source_id,
        "source_family": SOURCE_FAMILY,
        "stable_origin_id": f"caselaw-record-id-sha256:{record_identity}",
        "stable_object_id": f"sha256:{normalized_sha}",
        "modality": "en",
        "evidence_status": "DEDICATED_TERMINAL",
        "authority_ref": (
            f"PR{UPSTREAM_PRODUCT_PR}:{UPSTREAM_FINAL_HEAD}:"
            f"blob:{UPSTREAM_EVIDENCE_BLOB_SHA1}"
        ),
        "declared_capacity_bytes": normalized_bytes,
        "expected_raw_bytes": normalized_bytes,
        "expected_raw_sha256": normalized_sha,
        "acquisition_url": SOURCE_AUTHORITY_URL,
        "origin_key": f"caselaw-unit:{record_identity}",
    }
    return matcher_row, source_id, payload


def _receipt(
    *,
    authority: CaselawAuthority,
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
        "upstream_authority": {
            "product_pr": authority.product_pr,
            "final_head": authority.final_head,
            "execution_head": authority.execution_head,
            "workflow_run_id": authority.workflow_run_id,
            "workflow_job_id": authority.workflow_job_id,
            "evidence_git_blob_sha1": authority.evidence_blob_sha1,
            "candidate_sha256": authority.candidate_sha256,
            "candidate_records": authority.candidate_records,
            "candidate_normalized_utf8_bytes": (
                authority.candidate_normalized_utf8_bytes
            ),
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
            "authorized_optimized_target_exposure": 0,
            "tokenizer_fit_authorized": False,
            "optimizer_updates_executed_on_real_targets": 0,
            "training_executed": False,
            "learned_weights_created": False,
            "final_test_outcomes_read": False,
            "paid_compute_used": False,
            "foreign_pretrained_weights": False,
            "upstream_source_evidence_external_llm_or_api_used": False,
            "current_corpus_external_llm_free_claimed_by_this_adapter": False,
        },
    }
    return {**core, "receipt_identity_sha256": _sha256(_canonical(core))}


def _project_candidate_bytes(
    candidate_raw: bytes,
    evidence_raw: bytes,
    *,
    upstream_head: str,
    authority: CaselawAuthority,
    retain_payloads: bool,
) -> CaselawProjection:
    _require(upstream_head == authority.final_head, "upstream final head drift")
    _require(type(retain_payloads) is bool, "retain_payloads must be exact bool")
    _validate_evidence(evidence_raw, authority)
    _require(_sha256(candidate_raw) == authority.candidate_sha256, "candidate SHA-256 drift")

    sources: list[dict[str, Any]] | None = [] if retain_payloads else None
    payloads: dict[str, bytes] | None = {} if retain_payloads else None
    seen_record_ids: set[str] = set()
    seen_source_ids: set[str] = set()
    payload_hashes: set[str] = set()
    projected_for_identity: list[dict[str, Any]] = []
    total_bytes = 0

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
            raise CaselawDedupIntakeError(
                f"candidate row {line_number} is not strict UTF-8 JSON"
            ) from exc
        _require(type(row) is dict, f"candidate row {line_number} must be exact object")
        record_id = row.get("record_id")
        _require(
            type(record_id) is str and record_id not in seen_record_ids,
            f"duplicate/invalid record_id at row {line_number}",
        )
        seen_record_ids.add(record_id)
        matcher_row, source_id, payload = _validate_row(row, line_number=line_number)
        _require(source_id not in seen_source_ids, f"source_id collision at row {line_number}")
        seen_source_ids.add(source_id)
        total_bytes += len(payload)
        payload_hashes.add(row["normalized_sha256"])
        projected_for_identity.append(matcher_row)
        if sources is not None and payloads is not None:
            sources.append(matcher_row)
            payloads[source_id] = payload

    _require(
        total_bytes == authority.candidate_normalized_utf8_bytes,
        "candidate normalized UTF-8 byte total drift",
    )
    _require(
        len(seen_record_ids) == authority.candidate_records,
        "candidate record identity cardinality drift",
    )
    projection_identity = _sha256(_canonical(projected_for_identity))
    receipt = _receipt(
        authority=authority,
        source_count=len(projected_for_identity),
        payload_bytes=total_bytes,
        distinct_payload_hashes=len(payload_hashes),
        projection_identity_sha256=projection_identity,
    )
    return CaselawProjection(
        receipt=receipt,
        sources=tuple(sources) if sources is not None else None,
        payloads=payloads,
    )


def validate_and_project_caselaw(
    candidate_jsonl: Path,
    execution_evidence_json: Path,
    *,
    upstream_head: str,
    retain_payloads: bool = True,
) -> CaselawProjection:
    """Validate exact PR #1347 bytes and project them one-for-one for V3 matching.

    Files are each read exactly once. This avoids a hash-then-reopen TOCTOU gap.
    The function never invokes a matcher and never grants downstream corpus credit.
    """
    candidate_raw = _read_bytes_once(candidate_jsonl, label="candidate JSONL")
    evidence_raw = _read_bytes_once(
        execution_evidence_json,
        label="upstream execution evidence",
    )
    return _project_candidate_bytes(
        candidate_raw,
        evidence_raw,
        upstream_head=upstream_head,
        authority=PRODUCTION_AUTHORITY,
        retain_payloads=retain_payloads,
    )
