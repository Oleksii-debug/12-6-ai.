"""Fail-closed composition of clean DATA-232, EVAL-647, G05 and G06 execution.

This module is an orchestration boundary only. It deliberately reuses the incumbent
reserved-decontamination, EVAL-647 exclusion, quality and privacy authorities. Raw
payloads exist only in caller-owned ephemeral memory; the returned receipt is
text-free and grants no corpus, tokenizer, optimizer or training authority.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from twelve_six.data.eval647_reserved_decontamination_v1 import (
    execute_eval647_reserved_decontamination,
    verify_eval647_reserved_decontamination_receipt,
)
from twelve_six.data.privacy_execution_authority import (
    build_privacy_execution_authority,
    verify_privacy_execution_authority,
)
from twelve_six.data.quality_execution_authority import (
    build_quality_execution_authority,
    verify_quality_execution_authority,
)

COMPOSITION_SCHEMA = "12-6.current-clean-decontam-quality-privacy.v1"
EXPECTED_DEPENDENCY_BLOBS = {
    "current_reserved_decontamination_v1.py": "e5c555e3cd27844e98d4ae91af0b746e427f36c9",
    "eval647_reserved_decontamination_v1.py": "ce33771c9fb4a6cc421e2f8e1f6f232c119bec71",
    "quality_execution_authority.py": "4659a9d4aba49908f372250904a54361c8d8cf46",
    "privacy_execution_authority.py": "9215287e81c0a82f05ec8405dc4f34c60313c193",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MODE_MAP = {"uk": "uk", "ua": "uk", "en": "en", "code": "code"}
_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "data232_report_sha256",
    "decontamination_execution_identity_sha256",
    "eval647_execution_receipt_identity_sha256",
    "quality_execution_identity_sha256",
    "privacy_execution_identity_sha256",
    "post_decontamination_input_rows_sha256",
    "input_training_records",
    "excluded_training_records",
    "post_decontamination_records",
    "post_decontamination_utf8_bytes",
    "dependency_git_blobs",
    "durable_evidence_hash_only",
    "current_corpus_launch_authority_promoted",
    "authorized_optimized_target_exposure",
    "tokenizer_fit_authorized",
    "optimizer_updates_executed_on_real_targets",
    "training_executed",
    "learned_weights_created",
    "final_test_outcomes_read",
    "paid_compute_used",
    "foreign_pretrained_weights",
    "receipt_identity_sha256",
}


class CurrentCleanExecutionError(RuntimeError):
    """Raised when current clean composition cannot be proven exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CurrentCleanExecutionError(message)


def _cjson(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    _require(
        isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
        f"{label} must be lowercase SHA-256",
    )
    return str(value)


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload).hexdigest()  # noqa: S324 - Git object identity


def verify_dependency_blobs() -> dict[str, str]:
    """Bind orchestration to the exact canonical dependency source bytes."""
    directory = Path(__file__).resolve(strict=True).parent
    observed: dict[str, str] = {}
    for filename, expected in EXPECTED_DEPENDENCY_BLOBS.items():
        path = (directory / filename).resolve(strict=True)
        _require(path.parent == directory, f"dependency escaped canonical data path: {filename}")
        actual = _git_blob_sha1(path.read_bytes())
        _require(actual == expected, f"dependency Git blob drift: {filename}")
        observed[filename] = actual
    return dict(sorted(observed.items()))


def _normalize_post_decontamination_records(
    training_records: Sequence[Mapping[str, Any]],
    decontamination_report: Mapping[str, Any],
) -> list[dict[str, str]]:
    excluded_raw = decontamination_report.get("excluded_records")
    _require(isinstance(excluded_raw, list), "DATA-232 excluded_records missing")
    excluded_hashes: set[str] = set()
    for index, row in enumerate(excluded_raw):
        _require(isinstance(row, Mapping), f"excluded_records[{index}] must be an object")
        record_hash = _require_sha256(
            row.get("record_id_sha256"), f"excluded_records[{index}].record_id_sha256"
        )
        _require(record_hash not in excluded_hashes, "duplicate excluded record hash")
        excluded_hashes.add(record_hash)

    normalized: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    observed_excluded: set[str] = set()
    for index, row in enumerate(training_records):
        _require(isinstance(row, Mapping), f"training_records[{index}] must be an object")
        record_id = row.get("record_id")
        text = row.get("text")
        modality = row.get("modality")
        _require(isinstance(record_id, str) and bool(record_id), "training record_id invalid")
        _require(record_id not in seen_ids, f"duplicate training record_id: {record_id}")
        _require(isinstance(text, str), f"training text invalid: {record_id}")
        _require(isinstance(modality, str), f"training modality invalid: {record_id}")
        seen_ids.add(record_id)
        record_hash = _sha256(record_id.encode("utf-8"))
        if record_hash in excluded_hashes:
            observed_excluded.add(record_hash)
            continue
        mode = _MODE_MAP.get(modality.lower())
        _require(mode is not None, f"unsupported post-decontamination modality: {modality}")
        normalized.append({"id": record_id, "text": text, "mode": mode})

    _require(
        observed_excluded == excluded_hashes,
        "DATA-232 exclusions do not map exactly to ephemeral training records",
    )
    _require(bool(normalized), "decontamination removed every training record")
    normalized.sort(key=lambda row: row["id"])
    return normalized


def _input_projection(records: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    return [
        {
            "record_id": row["id"],
            "mode": row["mode"],
            "payload_sha256": _sha256(row["text"].encode("utf-8")),
            "utf8_bytes": len(row["text"].encode("utf-8")),
        }
        for row in records
    ]


def execute_current_clean_composition(
    training_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
    *,
    training_handoff_evidence: Mapping[str, Any],
    base_reserved_binding: Mapping[str, Any],
    eval647_manifest: Mapping[str, Any],
    eval647_materialization_evidence: Mapping[str, Any],
    expected_base_reserved_binding_identity_sha256: str,
    expected_composed_reserved_binding_identity_sha256: str,
    expected_eval647_materialization_evidence_identity_sha256: str,
    expected_eval647_object_set_identity_sha256: str,
    expected_inventory_identity_sha256: str,
    expected_survivor_authority_sha256: str,
    expected_training_handoff_identity_sha256: str,
    expected_selection_validation_identity_sha256: str,
    expected_final_test_identity_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Execute clean decontamination then G05/G06 mechanics with zero scientific credit."""
    dependency_blobs = verify_dependency_blobs()
    report, decontam_evidence, eval647_receipt = execute_eval647_reserved_decontamination(
        training_records,
        evaluation_records,
        training_handoff_evidence=training_handoff_evidence,
        base_reserved_binding=base_reserved_binding,
        manifest=eval647_manifest,
        materialization_evidence=eval647_materialization_evidence,
        expected_base_reserved_binding_identity_sha256=(
            expected_base_reserved_binding_identity_sha256
        ),
        expected_composed_reserved_binding_identity_sha256=(
            expected_composed_reserved_binding_identity_sha256
        ),
        expected_eval647_materialization_evidence_identity_sha256=(
            expected_eval647_materialization_evidence_identity_sha256
        ),
        expected_eval647_object_set_identity_sha256=(
            expected_eval647_object_set_identity_sha256
        ),
        expected_inventory_identity_sha256=expected_inventory_identity_sha256,
        expected_survivor_authority_sha256=expected_survivor_authority_sha256,
        expected_training_handoff_identity_sha256=expected_training_handoff_identity_sha256,
        expected_selection_validation_identity_sha256=(
            expected_selection_validation_identity_sha256
        ),
        expected_final_test_identity_sha256=expected_final_test_identity_sha256,
    )
    verify_eval647_reserved_decontamination_receipt(
        eval647_receipt,
        expected_composed_reserved_binding_identity_sha256=(
            expected_composed_reserved_binding_identity_sha256
        ),
        expected_eval647_materialization_evidence_identity_sha256=(
            expected_eval647_materialization_evidence_identity_sha256
        ),
        expected_eval647_object_set_identity_sha256=(
            expected_eval647_object_set_identity_sha256
        ),
    )

    survivors = _normalize_post_decontamination_records(training_records, report)
    projection = _input_projection(survivors)
    input_rows_sha256 = _sha256(_cjson(projection))
    decontam_identity = _require_sha256(
        decontam_evidence.get("execution_identity_sha256"),
        "decontamination execution identity",
    )
    quality = build_quality_execution_authority(
        survivors,
        input_manifest_sha256=decontam_identity,
        expected_input_rows_sha256=input_rows_sha256,
    )
    quality_identity = _require_sha256(
        quality.get("execution_identity_sha256"), "quality execution identity"
    )
    verify_quality_execution_authority(
        quality,
        survivors,
        expected_input_manifest_sha256=decontam_identity,
        expected_input_rows_sha256=input_rows_sha256,
        expected_execution_identity_sha256=quality_identity,
    )

    privacy = build_privacy_execution_authority(
        survivors,
        expected_input_rows_sha256=input_rows_sha256,
    )
    privacy_identity = _require_sha256(
        privacy.get("execution_identity_sha256"), "privacy execution identity"
    )
    verify_privacy_execution_authority(
        privacy,
        survivors,
        expected_input_rows_sha256=input_rows_sha256,
        expected_execution_identity_sha256=privacy_identity,
    )

    total_bytes = sum(len(row["text"].encode("utf-8")) for row in survivors)
    receipt: dict[str, Any] = {
        "schema_version": COMPOSITION_SCHEMA,
        "status": "CURRENT_CLEAN_DECONTAM_G05_G06_EXECUTED_ZERO_CREDIT",
        "data232_report_sha256": _require_sha256(
            report.get("report_sha256"), "DATA-232 report identity"
        ),
        "decontamination_execution_identity_sha256": decontam_identity,
        "eval647_execution_receipt_identity_sha256": _require_sha256(
            eval647_receipt.get("receipt_identity_sha256"), "EVAL-647 receipt identity"
        ),
        "quality_execution_identity_sha256": quality_identity,
        "privacy_execution_identity_sha256": privacy_identity,
        "post_decontamination_input_rows_sha256": input_rows_sha256,
        "input_training_records": len(training_records),
        "excluded_training_records": len(training_records) - len(survivors),
        "post_decontamination_records": len(survivors),
        "post_decontamination_utf8_bytes": total_bytes,
        "dependency_git_blobs": dependency_blobs,
        "durable_evidence_hash_only": True,
        "current_corpus_launch_authority_promoted": False,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit_authorized": False,
        "optimizer_updates_executed_on_real_targets": 0,
        "training_executed": False,
        "learned_weights_created": False,
        "final_test_outcomes_read": False,
        "paid_compute_used": False,
        "foreign_pretrained_weights": False,
    }
    receipt["receipt_identity_sha256"] = _sha256(_cjson(receipt))
    return receipt, report, quality, privacy


def verify_current_clean_composition_receipt(
    receipt: Mapping[str, Any],
    *,
    expected_receipt_identity_sha256: str,
    expected_data232_report_sha256: str,
    expected_decontamination_execution_identity_sha256: str,
    expected_eval647_execution_receipt_identity_sha256: str,
    expected_quality_execution_identity_sha256: str,
    expected_privacy_execution_identity_sha256: str,
    expected_post_decontamination_input_rows_sha256: str,
) -> str:
    """Verify the receipt and every nested execution root against external pins."""
    _require(isinstance(receipt, Mapping) and set(receipt) == _RECEIPT_KEYS, "receipt schema is not closed")
    _require(receipt.get("schema_version") == COMPOSITION_SCHEMA, "receipt schema drift")
    expected = _require_sha256(expected_receipt_identity_sha256, "expected receipt identity")
    claimed = _require_sha256(receipt.get("receipt_identity_sha256"), "receipt identity")
    body = dict(receipt)
    body.pop("receipt_identity_sha256")
    _require(_sha256(_cjson(body)) == claimed, "receipt self-hash mismatch")
    _require(claimed == expected, "receipt identity is not independently expected")
    expected_nested = {
        "data232_report_sha256": expected_data232_report_sha256,
        "decontamination_execution_identity_sha256": (
            expected_decontamination_execution_identity_sha256
        ),
        "eval647_execution_receipt_identity_sha256": (
            expected_eval647_execution_receipt_identity_sha256
        ),
        "quality_execution_identity_sha256": expected_quality_execution_identity_sha256,
        "privacy_execution_identity_sha256": expected_privacy_execution_identity_sha256,
        "post_decontamination_input_rows_sha256": (
            expected_post_decontamination_input_rows_sha256
        ),
    }
    for key, value in expected_nested.items():
        if receipt.get(key) != _require_sha256(value, f"expected {key}"):
            raise CurrentCleanExecutionError(f"nested execution root drift: {key}")
    _require(
        receipt.get("dependency_git_blobs") == EXPECTED_DEPENDENCY_BLOBS,
        "dependency blob binding drift",
    )
    _require(receipt.get("durable_evidence_hash_only") is True, "durable evidence boundary weakened")
    _require(receipt.get("current_corpus_launch_authority_promoted") is False, "corpus launch authority fabricated")
    for key in ("authorized_optimized_target_exposure", "optimizer_updates_executed_on_real_targets"):
        _require(type(receipt.get(key)) is int and receipt.get(key) == 0, f"authority widened: {key}")
    for key in (
        "tokenizer_fit_authorized",
        "training_executed",
        "learned_weights_created",
        "final_test_outcomes_read",
        "paid_compute_used",
        "foreign_pretrained_weights",
    ):
        _require(receipt.get(key) is False, f"authority boundary weakened: {key}")
    for key in (
        "data232_report_sha256",
        "decontamination_execution_identity_sha256",
        "eval647_execution_receipt_identity_sha256",
        "quality_execution_identity_sha256",
        "privacy_execution_identity_sha256",
        "post_decontamination_input_rows_sha256",
    ):
        _require_sha256(receipt.get(key), key)
    for key in (
        "input_training_records",
        "excluded_training_records",
        "post_decontamination_records",
        "post_decontamination_utf8_bytes",
    ):
        _require(type(receipt.get(key)) is int and receipt.get(key) >= 0, f"invalid count: {key}")
    _require(
        receipt["input_training_records"]
        == receipt["excluded_training_records"] + receipt["post_decontamination_records"],
        "decontamination count accounting drift",
    )
    return claimed
