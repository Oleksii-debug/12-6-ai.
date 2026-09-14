"""Authenticate and project exact PR #655 Rada-laws Q/P rows for canonical D03 dedup.

No matching science or incumbent-base authority lives here. Every accepted chunk
is preserved one-for-one for a separately authenticated canonical consumer.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

SOURCE_FAMILY: Final = "ua.rada.open-data.laws-texts"
ARCHIVE_URL: Final = "https://data.rada.gov.ua/ogd/zak/perv/text/texts.zip"
UPSTREAM_PRODUCT_PR: Final = 655
UPSTREAM_FINAL_EVIDENCE_COMMIT: Final = "43a7d62854255da4426d6a3dc4d5ad9b1e4898e9"
UPSTREAM_REPAIR_EXECUTION_HEAD: Final = "cda7c4c0ed552771ddc6313647c5baf1a4d42f47"
UPSTREAM_WORKFLOW_RUN_ID: Final = 34_669_418_910
UPSTREAM_WORKFLOW_JOB_ID: Final = 103_487_804_898
UPSTREAM_WORKFLOW_RUN_ATTEMPT: Final = 1
ACCEPTED_JSONL_FILE_BYTES: Final = 224_528_099
ACCEPTED_JSONL_SHA256: Final = (
    "48324e41385e12a5b9f590bd478516643edb21992f1a5ca5ca2c9eca75f7ab12"
)
ACCEPTED_CHUNK_COUNT: Final = 101_559
ACCEPTED_TEXT_UTF8_BYTES: Final = 192_078_166
ACCEPTED_INVENTORY_SHA256: Final = (
    "afb940b48b5de52e598d8adf026c78cf5d51b228f1fbf56c9f0fbe472d946f65"
)
EXACT_DUPLICATE_ACCEPTED_HASHES: Final = 764
ACCEPTED_SOURCE_ENCODING_COUNTS: Final = {"utf-8": 79_960, "windows-1251": 21_599}
QUALITY_REPORT_FILE_SHA256: Final = (
    "b05e1981370b1f92914997b25ba1cc4f96ba0a71cda8ad9bcf8114aec323972c"
)
QUALITY_REPORT_IDENTITY_SHA256: Final = (
    "fd7fa447c1876369b5de724deeee1bbeac741a486d737b62dd60af1ae2cbaa71"
)
EXECUTION_EVIDENCE_IDENTITY_SHA256: Final = (
    "6f4952add1a2dabaf37170fac5ec2ffae9bdd3366b0a032c6a89886a8b898e74"
)
SOURCE_ARCHIVE_SHA256: Final = (
    "0b9e8ed8fe8aa663a68d2bc4eba858a754c626391dd7b5c461d50c3b6260df63"
)
QUALITY_REPORT_SCHEMA: Final = "12-6.d03-rada-bulk-quality-privacy-report.v1"
QUALITY_REPORT_WORKER: Final = "D03-RADA-BULK-QUALITY-PRIVACY-20260911"
QUALITY_SAFE_RESULT: Final = (
    "QUALITY_PRIVACY_FILTERED_CANDIDATE_ONLY_DOWNSTREAM_GATES_REQUIRED"
)
EXECUTION_EVIDENCE_SCHEMA: Final = (
    "12-6.d03-rada-bulk-quality-privacy-real-execution-evidence.v1"
)
EXECUTION_SAFE_RESULT: Final = (
    "QUALITY_PRIVACY_FILTERED_CANDIDATE_ZERO_CREDIT_TWO_CLEAN_REAL_EXECUTION"
)
RECEIPT_SCHEMA: Final = "12-6.d03-rada-laws-qp-dedup-intake-receipt.v1"
INCUMBENT_INVENTORY_SCHEMA: Final = "12-6.next100-065-cross-source-dedup.v3"
DIRECT_RADA_ALL_PAIR_DISPATCHES: Final = (
    ACCEPTED_CHUNK_COUNT * (ACCEPTED_CHUNK_COUNT - 1) // 2
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ROW_KEYS = frozenset(
    {
        "record_id",
        "parent_record_id",
        "source_path",
        "source_encoding",
        "chunk_index",
        "normalized_bytes",
        "normalized_sha256",
        "text",
    }
)
_METADATA_KEYS = _ROW_KEYS - {"text"}
_REPORT_KEYS = frozenset(
    {
        "schema_version", "worker_id", "local_free_only", "parent_normalization",
        "policy", "input", "filter_result", "gates", "accepted_records",
        "canonical_capacity_credited", "training_authorized_bytes",
        "tokenizer_fit_authorized", "model_training_executed", "optimizer_updates",
        "paid_compute_used", "evaluation_authorized", "final_test_accessed",
        "research_corpus_v1_released", "learned_20m_claimed", "safe_result",
        "report_identity_sha256",
    }
)
_EVIDENCE_KEYS = frozenset(
    {
        "environment", "evidence_identity_sha256", "execution_class",
        "observed_quality_privacy", "parent_authority", "repair_execution_head_sha",
        "safe_result", "schema_version", "source_authority", "truth_boundary",
        "two_clean_reproducibility", "workflow_run_attempt", "workflow_run_id",
    }
)


class RadaLawsDedupIntakeError(RuntimeError):
    """Fail-closed Rada-laws Q/P dedup-intake error."""


@dataclass(frozen=True)
class RadaLawsProjection:
    receipt: dict[str, Any]
    sources: tuple[dict[str, Any], ...] | None
    payloads: dict[str, bytes] | None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RadaLawsDedupIntakeError(message)


def _exact_int(value: Any, expected: int, message: str) -> None:
    _require(type(value) is int and value == expected, message)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise RadaLawsDedupIntakeError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _strict_json_bytes(raw: bytes, *, label: str) -> dict[str, Any]:
    _require(type(raw) is bytes and bool(raw), f"{label} is empty")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RadaLawsDedupIntakeError(f"{label} is not strict UTF-8 JSON") from exc
    _require(type(value) is dict, f"{label} root must be exact object")
    return value


def _read_json(
    path: Path, *, label: str, expected_sha256: str | None = None
) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RadaLawsDedupIntakeError(f"cannot read {label}: {path}") from exc
    if expected_sha256 is not None:
        _require(_sha256(raw) == expected_sha256, f"{label} file SHA-256 drift")
    return _strict_json_bytes(raw, label=label)


def _exact_values(
    actual: Mapping[str, Any], expected: Mapping[str, Any], *, label: str
) -> None:
    for key, wanted in expected.items():
        value = actual.get(key)
        _require(
            value == wanted and type(value) is type(wanted),
            f"{label} drift: {key}",
        )


def _validate_quality_report(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    _require(type(report) is dict and set(report) == _REPORT_KEYS, "quality report schema drift")
    _exact_values(
        report,
        {
            "schema_version": QUALITY_REPORT_SCHEMA,
            "worker_id": QUALITY_REPORT_WORKER,
            "local_free_only": True,
            "safe_result": QUALITY_SAFE_RESULT,
            "canonical_capacity_credited": 0,
            "training_authorized_bytes": 0,
            "tokenizer_fit_authorized": False,
            "model_training_executed": False,
            "optimizer_updates": 0,
            "paid_compute_used": False,
            "evaluation_authorized": False,
            "final_test_accessed": False,
            "research_corpus_v1_released": False,
            "learned_20m_claimed": False,
        },
        label="quality report",
    )
    identity = report.get("report_identity_sha256")
    _require(identity == QUALITY_REPORT_IDENTITY_SHA256, "quality report identity drift")
    unsigned = copy.deepcopy(dict(report))
    unsigned.pop("report_identity_sha256", None)
    _require(_sha256(_canonical(unsigned)) == identity, "quality report self-hash mismatch")

    result = report.get("filter_result")
    _require(type(result) is dict, "quality filter_result missing")
    _exact_values(
        result,
        {
            "accepted_chunk_count": ACCEPTED_CHUNK_COUNT,
            "accepted_bytes_observed_not_credited": ACCEPTED_TEXT_UTF8_BYTES,
            "accepted_jsonl_sha256": ACCEPTED_JSONL_SHA256,
            "accepted_inventory_sha256": ACCEPTED_INVENTORY_SHA256,
            "exact_duplicate_accepted_hashes_observed_not_removed": (
                EXACT_DUPLICATE_ACCEPTED_HASHES
            ),
            "accepted_source_encoding_counts": ACCEPTED_SOURCE_ENCODING_COUNTS,
            "rejected_text_emitted": False,
            "rejected_hashes_emitted": False,
        },
        label="quality result",
    )
    gates = report.get("gates")
    _require(type(gates) is dict, "quality report gates missing")
    for key in (
        "parent_manifest_integrity", "parent_jsonl_integrity", "exact_parent_authority",
        "deterministic_chunking", "bounded_quality_privacy_filter_execution",
    ):
        _require(gates.get(key) == "PASS", f"quality prerequisite not PASS: {key}")
    for key in (
        "global_cross_source_dedup", "evaluation_decontamination", "balance_diversity",
        "corpus_materialization", "unique_loss_ledger", "d05_checkpoint_requalification",
    ):
        _require(gates.get(key) == "NOT_RUN", f"quality downstream gate drift: {key}")

    accepted = report.get("accepted_records")
    _require(type(accepted) is list, "quality accepted_records missing")
    _require(len(accepted) == ACCEPTED_CHUNK_COUNT, "quality accepted_records length drift")
    for index, metadata in enumerate(accepted):
        _require(
            type(metadata) is dict and set(metadata) == _METADATA_KEYS,
            f"quality metadata row {index} schema drift",
        )
    return accepted


def _validate_execution_evidence(evidence: Mapping[str, Any]) -> None:
    _require(
        type(evidence) is dict and set(evidence) == _EVIDENCE_KEYS,
        "execution evidence schema drift",
    )
    _exact_values(
        evidence,
        {
            "schema_version": EXECUTION_EVIDENCE_SCHEMA,
            "safe_result": EXECUTION_SAFE_RESULT,
            "execution_class": "LOCAL_FREE_GITHUB_HOSTED_ACTIONS",
            "repair_execution_head_sha": UPSTREAM_REPAIR_EXECUTION_HEAD,
            "workflow_run_id": UPSTREAM_WORKFLOW_RUN_ID,
            "workflow_run_attempt": UPSTREAM_WORKFLOW_RUN_ATTEMPT,
        },
        label="execution evidence",
    )
    identity = evidence.get("evidence_identity_sha256")
    _require(identity == EXECUTION_EVIDENCE_IDENTITY_SHA256, "execution evidence identity drift")
    unsigned = copy.deepcopy(dict(evidence))
    unsigned.pop("evidence_identity_sha256", None)
    _require(_sha256(_canonical(unsigned)) == identity, "execution evidence self-hash mismatch")

    observed = evidence.get("observed_quality_privacy")
    _require(type(observed) is dict, "observed quality/privacy missing")
    _exact_values(
        observed,
        {
            "accepted_bytes_observed_not_credited": ACCEPTED_TEXT_UTF8_BYTES,
            "accepted_chunk_count": ACCEPTED_CHUNK_COUNT,
            "accepted_inventory_sha256": ACCEPTED_INVENTORY_SHA256,
            "accepted_jsonl_file_bytes": ACCEPTED_JSONL_FILE_BYTES,
            "accepted_jsonl_sha256": ACCEPTED_JSONL_SHA256,
            "accepted_source_encoding_counts": ACCEPTED_SOURCE_ENCODING_COUNTS,
            "exact_duplicate_accepted_hashes_observed_not_removed": (
                EXACT_DUPLICATE_ACCEPTED_HASHES
            ),
            "quality_report_file_sha256": QUALITY_REPORT_FILE_SHA256,
            "quality_report_identity_sha256": QUALITY_REPORT_IDENTITY_SHA256,
        },
        label="execution observation",
    )
    source = evidence.get("source_authority")
    _require(
        type(source) is dict and source.get("archive_sha256") == SOURCE_ARCHIVE_SHA256,
        "source archive authority drift",
    )
    repeat = evidence.get("two_clean_reproducibility")
    _require(type(repeat) is dict and bool(repeat), "two-clean proof missing")
    _require(
        all(type(value) is bool and value is True for value in repeat.values()),
        "two-clean proof drift",
    )
    truth = evidence.get("truth_boundary")
    _require(type(truth) is dict, "execution truth boundary missing")
    _exact_values(
        truth,
        {
            "authorized_optimized_target_exposure": 0,
            "canonical_capacity_credited": 0,
            "optimizer_updates_executed_on_real_targets": 0,
            "training_authorized_bytes": 0,
            "current_corpus_eligible": False,
            "external_llm_api_data_or_intelligence": False,
            "final_test_accessed": False,
            "foreign_pretrained_weights": False,
            "learned_weights_created": False,
            "model_training_executed": False,
            "paid_compute_used": False,
            "tokenizer_fit_authorized": False,
        },
        label="execution truth boundary",
    )


def _validate_project_row(
    row: Mapping[str, Any],
    expected_metadata: Mapping[str, Any],
    *,
    line_number: int,
) -> tuple[dict[str, Any], str, bytes]:
    _require(type(row) is dict and set(row) == _ROW_KEYS, f"row {line_number} schema drift")
    _require(
        type(expected_metadata) is dict and set(expected_metadata) == _METADATA_KEYS,
        f"metadata {line_number} schema drift",
    )
    metadata = {key: row[key] for key in row if key != "text"}
    _require(
        _canonical(metadata) == _canonical(dict(expected_metadata)),
        f"candidate/report metadata mismatch at row {line_number}",
    )
    record_id = row["record_id"]
    parent_id = row["parent_record_id"]
    source_path = row["source_path"]
    source_encoding = row["source_encoding"]
    chunk_index = row["chunk_index"]
    text = row["text"]
    normalized_bytes = row["normalized_bytes"]
    normalized_sha = row["normalized_sha256"]
    _require(type(record_id) is str and bool(record_id), f"record_id invalid at {line_number}")
    _require(type(parent_id) is str and bool(parent_id), f"parent id invalid at {line_number}")
    _require(
        type(source_path) is str and bool(source_path),
        f"source path invalid at {line_number}",
    )
    _require(
        type(source_encoding) is str and source_encoding in {"utf-8", "windows-1251"},
        f"source_encoding invalid at {line_number}",
    )
    _require(type(chunk_index) is int and chunk_index >= 0, f"chunk_index invalid at {line_number}")
    _require(
        record_id == f"{parent_id}.q{chunk_index:05d}",
        f"record_id/chunk binding drift at {line_number}",
    )
    _require(type(text) is str and bool(text), f"text missing at {line_number}")
    payload = text.encode("utf-8")
    _require(
        type(normalized_bytes) is int
        and normalized_bytes > 0
        and normalized_bytes == len(payload),
        f"normalized byte count drift at {line_number}",
    )
    _require(
        type(normalized_sha) is str and _SHA256_RE.fullmatch(normalized_sha) is not None,
        f"normalized SHA malformed at {line_number}",
    )
    _require(_sha256(payload) == normalized_sha, f"normalized SHA drift at {line_number}")

    source_id = f"rada-laws-qp:{record_id}"
    matcher_row = {
        "source_id": source_id,
        "source_family": SOURCE_FAMILY,
        "stable_origin_id": f"rada-laws-document:{parent_id}",
        "stable_object_id": f"sha256:{normalized_sha}",
        "modality": "uk",
        "evidence_status": "DEDICATED_TERMINAL",
        "authority_ref": (
            f"PR{UPSTREAM_PRODUCT_PR}:{UPSTREAM_FINAL_EVIDENCE_COMMIT}:"
            f"{EXECUTION_EVIDENCE_IDENTITY_SHA256}"
        ),
        "declared_capacity_bytes": len(payload),
        "expected_raw_bytes": len(payload),
        "expected_raw_sha256": normalized_sha,
        "acquisition_url": ARCHIVE_URL,
        "origin_key": f"rada-laws-unit:{record_id}",
    }
    return matcher_row, source_id, payload


def _receipt(
    *,
    projection_identity_sha256: str,
    source_count: int,
    payload_bytes: int,
    duplicate_hashes: int,
    encoding_counts: Mapping[str, int],
) -> dict[str, Any]:
    core: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "execution_profile": "LOCAL_FREE",
        "swarm_claim": 1769,
        "source_family": SOURCE_FAMILY,
        "upstream_authority": {
            "product_pr": UPSTREAM_PRODUCT_PR,
            "final_evidence_commit": UPSTREAM_FINAL_EVIDENCE_COMMIT,
            "workflow_run_id": UPSTREAM_WORKFLOW_RUN_ID,
            "workflow_job_id": UPSTREAM_WORKFLOW_JOB_ID,
            "execution_evidence_identity_sha256": EXECUTION_EVIDENCE_IDENTITY_SHA256,
            "accepted_jsonl_sha256": ACCEPTED_JSONL_SHA256,
            "accepted_jsonl_file_bytes": ACCEPTED_JSONL_FILE_BYTES,
            "accepted_inventory_sha256": ACCEPTED_INVENTORY_SHA256,
            "quality_report_file_sha256": QUALITY_REPORT_FILE_SHA256,
            "quality_report_identity_sha256": QUALITY_REPORT_IDENTITY_SHA256,
        },
        "projection": {
            "matcher_inventory_schema": INCUMBENT_INVENTORY_SCHEMA,
            "source_object_count": source_count,
            "payload_utf8_bytes": payload_bytes,
            "source_encoding_counts": dict(sorted(encoding_counts.items())),
            "matcher_source_inventory_identity_sha256": projection_identity_sha256,
            "exact_duplicate_payload_hashes_preserved": duplicate_hashes,
            "stable_object_aliases_preserved_for_matcher": True,
            "origin_key_is_chunk_specific": True,
            "stable_origin_id_is_parent_document_specific": True,
            "raw_text_emitted": False,
        },
        "execution_gate": {
            "canonical_global_dedup_executed": False,
            "direct_incumbent_all_pair_dispatches_rada_only": DIRECT_RADA_ALL_PAIR_DISPATCHES,
            "performance_equivalent_executor_dependency": "#1459/#1764",
            "status": "BLOCKED_PENDING_TERMINAL_PERFORMANCE_EQUIVALENT_EXECUTOR",
            "pre_dedup_or_parent_aggregation_performed": False,
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
            "external_llm_or_api_used_for_data_or_intelligence": False,
        },
    }
    return {**core, "receipt_identity_sha256": _sha256(_canonical(core))}


def validate_and_project_rada_laws_qp(
    candidate_jsonl: Path,
    quality_report_json: Path,
    execution_evidence_json: Path,
    *,
    upstream_head: str,
    retain_payloads: bool = True,
) -> RadaLawsProjection:
    """Validate exact real PR #655 output and project every accepted chunk one-to-one."""
    _require(upstream_head == UPSTREAM_FINAL_EVIDENCE_COMMIT, "upstream head drift")
    _require(type(retain_payloads) is bool, "retain_payloads must be exact bool")

    report = _read_json(
        quality_report_json,
        label="quality report",
        expected_sha256=QUALITY_REPORT_FILE_SHA256,
    )
    expected_metadata = _validate_quality_report(report)
    evidence = _read_json(execution_evidence_json, label="execution evidence")
    _validate_execution_evidence(evidence)

    sources: list[dict[str, Any]] | None = [] if retain_payloads else None
    payloads: dict[str, bytes] | None = {} if retain_payloads else None
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    duplicate_hashes = 0
    encodings: Counter[str] = Counter()
    total_bytes = 0
    inventory_hasher = hashlib.sha256()
    transport_hasher = hashlib.sha256()
    transport_bytes = 0
    row_count = 0

    try:
        handle = candidate_jsonl.open("rb")
    except OSError as exc:
        raise RadaLawsDedupIntakeError(
            f"cannot read candidate JSONL: {candidate_jsonl}"
        ) from exc

    with handle:
        for line_number, line in enumerate(handle, 1):
            # Authenticate exactly the bytes parsed below. The candidate path is
            # opened once, so a pathname replacement cannot swap in a different
            # stream after transport authentication.
            transport_bytes += len(line)
            transport_hasher.update(line)

            _require(bool(line.strip()), f"blank candidate row at {line_number}")
            try:
                row = json.loads(
                    line.decode("utf-8", errors="strict"),
                    object_pairs_hook=_reject_duplicate_pairs,
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise RadaLawsDedupIntakeError(
                    f"invalid candidate row {line_number}"
                ) from exc
            _require(
                line_number <= len(expected_metadata),
                "candidate has more rows than authority report",
            )
            matcher_row, source_id, payload = _validate_project_row(
                row, expected_metadata[line_number - 1], line_number=line_number
            )
            _require(source_id not in seen_ids, f"duplicate source id: {source_id}")
            seen_ids.add(source_id)
            payload_sha = row["normalized_sha256"]
            if payload_sha in seen_hashes:
                duplicate_hashes += 1
            else:
                seen_hashes.add(payload_sha)
            encodings[str(row["source_encoding"])] += 1
            total_bytes += len(payload)
            inventory_hasher.update(_canonical(matcher_row) + b"\n")
            row_count += 1
            if sources is not None and payloads is not None:
                sources.append(matcher_row)
                payloads[source_id] = payload

    # Transport identity is checked only after EOF of the same single stream that
    # was parsed. Nothing is returned to a downstream consumer on mismatch.
    _exact_int(
        transport_bytes,
        ACCEPTED_JSONL_FILE_BYTES,
        "candidate transport byte drift",
    )
    _require(
        transport_hasher.hexdigest() == ACCEPTED_JSONL_SHA256,
        "candidate transport SHA drift",
    )
    _exact_int(row_count, ACCEPTED_CHUNK_COUNT, "candidate row count drift")
    _exact_int(total_bytes, ACCEPTED_TEXT_UTF8_BYTES, "candidate text byte total drift")
    _exact_int(
        duplicate_hashes,
        EXACT_DUPLICATE_ACCEPTED_HASHES,
        "candidate exact-duplicate observation drift",
    )
    _require(
        dict(sorted(encodings.items())) == ACCEPTED_SOURCE_ENCODING_COUNTS,
        "candidate encoding distribution drift",
    )
    _require(len(expected_metadata) == row_count, "report has unmatched metadata rows")
    receipt = _receipt(
        projection_identity_sha256=inventory_hasher.hexdigest(),
        source_count=row_count,
        payload_bytes=total_bytes,
        duplicate_hashes=duplicate_hashes,
        encoding_counts=encodings,
    )
    return RadaLawsProjection(
        receipt=receipt,
        sources=tuple(sources) if sources is not None else None,
        payloads=payloads,
    )
