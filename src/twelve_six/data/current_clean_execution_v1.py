"""Fail-closed clean DATA-232 -> G05 -> G06 survivor execution.

The carrier binds the exact integrated clean PR2183 payload/handoff roots, executes
the incumbent reserved-evaluation DATA-232 path, then executes the incumbent G05
quality and G06 privacy authorities serially. Quality window retention and privacy
redaction are physically materialized in ephemeral memory. Durable authority is
text-free and remains zero-credit until later balance/split/pack/loss gates.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from twelve_six.data.eval647_reserved_decontamination_v1 import (
    execute_eval647_reserved_decontamination,
    verify_eval647_reserved_decontamination_receipt,
)
from twelve_six.data.post_g05_g06_materialization_v1 import (
    _privacy_runtime,
    _redact_text,
    canonical_record_bytes,
    materialize_record_inventory,
)
from twelve_six.data.privacy_execution_authority import (
    build_privacy_execution_authority,
    verify_privacy_execution_authority,
)
from twelve_six.data.quality_execution_authority import (
    build_quality_execution_authority,
    verify_quality_execution_authority,
)

COMPOSITION_SCHEMA = "12-6.current-clean-decontam-quality-privacy-survivor.v2"
PRODUCTION_TRAINING_RECORDS_SHA256 = (
    "3458afe0380ea45d328ad3f004b21845a188ca69f2f7a83c24999e9e53268e53"
)
PRODUCTION_TRAINING_HANDOFF_SHA256 = (
    "80bcf2dd28f0d13795ceea01b358c7149b636f55f17b29575d14313b5cee99ee"
)
PRODUCTION_POST_G05_G06_MATERIALIZATION_IDENTITY_SHA256 = (
    "7061d74db13bf45a9a7a1266ebe50feab8e7d22c32fba7a81dd91c2be4135ade"
)
PRODUCTION_COMPOSITION_PREFLIGHT_IDENTITY_SHA256 = (
    "1b3adfffab2a9d65af78e88b805ef221714a0cc94fca4055105665a6a155ce95"
)
PRODUCTION_INPUT_RECORD_COUNT = 257

EXPECTED_DEPENDENCY_BLOBS = {
    "current_reserved_decontamination_v1.py": "e5c555e3cd27844e98d4ae91af0b746e427f36c9",
    "eval647_reserved_decontamination_v1.py": "ce33771c9fb4a6cc421e2f8e1f6f232c119bec71",
    "quality_execution_authority.py": "4659a9d4aba49908f372250904a54361c8d8cf46",
    "privacy_execution_authority.py": "9215287e81c0a82f05ec8405dc4f34c60313c193",
    "post_g05_g06_materialization_v1.py": "830087f91d1fa24385c5cbc8d2f687e4cc46b419",
    "privacy_filter_v3.py": "bcc5938395724f6728ab212f98b39f2334b0f37d",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MODE_MAP = {"uk": "uk", "ua": "uk", "en": "en", "code": "code"}
_TRAINING_RECORD_KEYS = {"record_id", "source_id", "source_family", "modality", "text"}
_SURVIVOR_RECORD_KEYS = {"record_id", "source_id", "family", "modality", "normalized_payload"}
_REJECTION_KEYS = {
    "data232_excluded_records",
    "g05_reject_documents",
    "g05_partial_documents",
    "g05_rejected_units",
    "g05_rejected_utf8_bytes",
    "g06_redacted_records",
    "g06_quarantine_records",
    "g06_exclude_records",
    "g06_dropped_utf8_bytes",
}
_RECEIPT_KEYS = {
    "schema_version",
    "status",
    "clean_training_records_sha256",
    "clean_training_handoff_sha256",
    "data232_report_sha256",
    "decontamination_execution_identity_sha256",
    "eval647_execution_receipt_identity_sha256",
    "quality_execution_identity_sha256",
    "privacy_execution_identity_sha256",
    "post_decontamination_input_rows_sha256",
    "post_quality_input_rows_sha256",
    "survivor_jsonl_sha256",
    "survivor_record_inventory_digest_sha256",
    "survivor_payload_inventory_digest_sha256",
    "input_training_records",
    "post_decontamination_records",
    "post_quality_records",
    "survivor_records",
    "survivor_payload_bytes",
    "survivor_source_objects",
    "rejection_counts",
    "privacy_detector_counts",
    "dependency_git_blobs",
    "durable_evidence_hash_only",
    "terminal_post_g05_g06_authority",
    "independent_qualification_required",
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
    return hashlib.sha1(  # noqa: S324 - Git object identity, not security use
        header + payload,
        usedforsecurity=False,
    ).hexdigest()


def verify_dependency_blobs() -> dict[str, str]:
    """Bind orchestration to the exact incumbent dependency source bytes."""
    directory = Path(__file__).resolve(strict=True).parent
    observed: dict[str, str] = {}
    for filename, expected in EXPECTED_DEPENDENCY_BLOBS.items():
        path = (directory / filename).resolve(strict=True)
        _require(
            path.parent == directory,
            f"dependency escaped canonical data path: {filename}",
        )
        actual = _git_blob_sha1(path.read_bytes())
        _require(actual == expected, f"dependency Git blob drift: {filename}")
        observed[filename] = actual
    return dict(sorted(observed.items()))


def _verify_clean_release_binding(
    training_records: Sequence[Mapping[str, Any]],
    training_handoff_evidence: Mapping[str, Any],
) -> None:
    """Reject caller-selected/stale corpus roots before any evaluation payload is read."""
    _require(
        len(training_records) == PRODUCTION_INPUT_RECORD_COUNT,
        "clean training record count is not independently expected",
    )
    normalized: list[dict[str, Any]] = []
    previous_id: str | None = None
    for index, raw in enumerate(training_records):
        _require(
            isinstance(raw, Mapping) and set(raw) == _TRAINING_RECORD_KEYS,
            f"training_records[{index}] schema drift",
        )
        row = dict(raw)
        for key in _TRAINING_RECORD_KEYS:
            _require(
                isinstance(row[key], str) and bool(row[key]),
                f"training_records[{index}].{key} invalid",
            )
        record_id = row["record_id"]
        _require(
            previous_id is None or record_id > previous_id,
            "clean training records must retain exact sorted order",
        )
        previous_id = record_id
        normalized.append(row)
    raw_records = b"".join(_cjson(row) for row in normalized)
    _require(
        _sha256(raw_records) == PRODUCTION_TRAINING_RECORDS_SHA256,
        "clean training records root is not independently expected",
    )

    _require(
        isinstance(training_handoff_evidence, Mapping),
        "clean training handoff missing",
    )
    _require(
        _sha256(_cjson(dict(training_handoff_evidence)))
        == PRODUCTION_TRAINING_HANDOFF_SHA256,
        "clean training handoff root is not independently expected",
    )
    _require(
        training_handoff_evidence.get("postdedup_inventory_identity_sha256")
        == PRODUCTION_POST_G05_G06_MATERIALIZATION_IDENTITY_SHA256,
        "clean handoff materialization lineage drift",
    )
    _require(
        training_handoff_evidence.get("input_survivor_authority_sha256")
        == PRODUCTION_COMPOSITION_PREFLIGHT_IDENTITY_SHA256,
        "clean handoff preflight lineage drift",
    )
    _require(
        training_handoff_evidence.get("retained_source_count")
        == PRODUCTION_INPUT_RECORD_COUNT,
        "clean handoff retained count drift",
    )
    _require(
        training_handoff_evidence.get("raw_text_persisted_in_evidence") is False,
        "clean handoff raw-text boundary widened",
    )
    _require(
        training_handoff_evidence.get("final_test_payload_accessed") is False
        and training_handoff_evidence.get("final_test_outcomes_accessed") is False,
        "clean handoff final-test boundary widened",
    )
    _require(
        type(training_handoff_evidence.get("authorized_training_exposure")) is int
        and training_handoff_evidence.get("authorized_training_exposure") == 0,
        "clean handoff training exposure widened",
    )


def _post_decontamination_records(
    training_records: Sequence[Mapping[str, Any]],
    decontamination_report: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, dict[str, str]], int]:
    excluded_raw = decontamination_report.get("excluded_records")
    _require(isinstance(excluded_raw, list), "DATA-232 excluded_records missing")
    excluded_hashes: set[str] = set()
    for index, row in enumerate(excluded_raw):
        _require(isinstance(row, Mapping), f"excluded_records[{index}] must be an object")
        record_hash = _require_sha256(
            row.get("record_id_sha256"),
            f"excluded_records[{index}].record_id_sha256",
        )
        _require(record_hash not in excluded_hashes, "duplicate excluded record hash")
        excluded_hashes.add(record_hash)

    quality_inputs: list[dict[str, str]] = []
    metadata: dict[str, dict[str, str]] = {}
    observed_excluded: set[str] = set()
    for row in training_records:
        record_id = str(row["record_id"])
        record_hash = _sha256(record_id.encode("utf-8"))
        if record_hash in excluded_hashes:
            observed_excluded.add(record_hash)
            continue
        modality = str(row["modality"])
        mode = _MODE_MAP.get(modality.lower())
        _require(
            mode is not None,
            f"unsupported post-decontamination modality: {modality}",
        )
        quality_inputs.append(
            {"id": record_id, "text": str(row["text"]), "mode": mode}
        )
        metadata[record_id] = {
            "source_id": str(row["source_id"]),
            "family": str(row["source_family"]),
            "mode": mode,
        }

    _require(
        observed_excluded == excluded_hashes,
        "DATA-232 exclusions do not map exactly to clean training records",
    )
    _require(bool(quality_inputs), "decontamination removed every training record")
    quality_inputs.sort(key=lambda row: row["id"])
    return quality_inputs, metadata, len(excluded_hashes)


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


def _materialize_quality_survivors(
    inputs: Sequence[Mapping[str, str]],
    metadata: Mapping[str, Mapping[str, str]],
    quality: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    raw_rows = quality.get("records")
    _require(isinstance(raw_rows, list), "G05 records missing")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in raw_rows:
        _require(isinstance(row, Mapping), "G05 record is not an object")
        record_id = row.get("record_id")
        _require(isinstance(record_id, str) and bool(record_id), "G05 record_id missing")
        _require(record_id not in by_id, "duplicate G05 record_id")
        by_id[record_id] = row
    _require(
        set(by_id) == {row["id"] for row in inputs},
        "G05/input record-set drift",
    )

    output: list[dict[str, str]] = []
    seen_output_ids: set[str] = set()
    stats = {
        "g05_reject_documents": 0,
        "g05_partial_documents": 0,
        "g05_rejected_units": 0,
        "g05_rejected_utf8_bytes": 0,
    }
    for source in inputs:
        record_id = source["id"]
        text = source["text"]
        mode = source["mode"]
        row = by_id[record_id]
        payload = text.encode("utf-8")
        _require(
            row.get("payload_sha256") == _sha256(payload)
            and row.get("utf8_bytes") == len(payload)
            and row.get("mode") == mode,
            f"G05 payload binding drift: {record_id}",
        )
        status = row.get("status")
        units = row.get("units")
        _require(
            status in {"RETAIN_ALL", "RETAIN_PARTIAL", "REJECT_DOCUMENT"},
            f"G05 status drift: {record_id}",
        )
        _require(isinstance(units, list) and bool(units), f"G05 units missing: {record_id}")
        _require(
            status != "RETAIN_PARTIAL",
            f"G05 partial row lacks canonical physical materialization authority: {record_id}",
        )
        if status == "REJECT_DOCUMENT":
            stats["g05_reject_documents"] += 1
        elif status == "RETAIN_PARTIAL":
            stats["g05_partial_documents"] += 1

        accepted_bytes = 0
        rejected_bytes = 0
        for unit in units:
            _require(isinstance(unit, Mapping), "G05 unit must be an object")
            start = unit.get("start_char")
            end = unit.get("end_char")
            accepted = unit.get("accepted")
            unit_id = unit.get("unit_id")
            _require(
                type(start) is int
                and type(end) is int
                and 0 <= start < end <= len(text),
                f"G05 unit span drift: {record_id}",
            )
            _require(type(accepted) is bool, f"G05 unit accepted type drift: {record_id}")
            _require(isinstance(unit_id, str) and bool(unit_id), "G05 unit_id missing")
            piece_raw = text[start:end].encode("utf-8")
            _require(
                unit.get("payload_sha256") == _sha256(piece_raw)
                and unit.get("utf8_bytes") == len(piece_raw),
                f"G05 unit payload drift: {unit_id}",
            )
            if accepted:
                accepted_bytes += len(piece_raw)
            else:
                stats["g05_rejected_units"] += 1
                stats["g05_rejected_utf8_bytes"] += len(piece_raw)
                rejected_bytes += len(piece_raw)

        _require(
            accepted_bytes == row.get("retained_utf8_bytes")
            and rejected_bytes == row.get("rejected_utf8_bytes"),
            f"G05 retained/rejected byte accounting drift: {record_id}",
        )
        if status == "RETAIN_ALL":
            _require(
                accepted_bytes == len(payload) and rejected_bytes == 0,
                f"G05 RETAIN_ALL byte drift: {record_id}",
            )
            _require(
                record_id not in seen_output_ids,
                f"G05 materialized record-id collision: {record_id}",
            )
            seen_output_ids.add(record_id)
            meta = metadata[record_id]
            output.append(
                {
                    "record_id": record_id,
                    "source_id": meta["source_id"],
                    "family": meta["family"],
                    "modality": mode,
                    "normalized_payload": text,
                }
            )
        if status == "REJECT_DOCUMENT":
            _require(
                accepted_bytes == 0 and rejected_bytes == len(payload),
                f"G05 rejected record retained bytes: {record_id}",
            )

    _require(bool(output), "G05 removed every post-decontamination record")
    output.sort(key=lambda row: row["record_id"])
    return output, stats


def _quality_records_for_privacy(
    records: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    return [
        {
            "id": row["record_id"],
            "text": row["normalized_payload"],
            "mode": row["modality"],
        }
        for row in records
    ]


def _materialize_privacy_survivors(
    records: Sequence[Mapping[str, str]],
    privacy: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, int]]:
    raw_rows = privacy.get("records")
    binding = privacy.get("privacy_binding")
    _require(isinstance(raw_rows, list), "G06 records missing")
    _require(isinstance(binding, Mapping), "G06 privacy binding missing")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in raw_rows:
        _require(isinstance(row, Mapping), "G06 record is not an object")
        record_id = row.get("record_id")
        _require(isinstance(record_id, str) and bool(record_id), "G06 record_id missing")
        _require(record_id not in by_id, "duplicate G06 record_id")
        by_id[record_id] = row
    _require(
        set(by_id) == {row["record_id"] for row in records},
        "G06/quality-survivor record-set drift",
    )

    privacy_path = Path(__file__).resolve(strict=True).with_name("privacy_filter_v3.py")
    detect, scan, _ = _privacy_runtime(privacy_path, privacy_binding=binding)
    output: list[dict[str, str]] = []
    stats = {
        "g06_redacted_records": 0,
        "g06_quarantine_records": 0,
        "g06_exclude_records": 0,
        "g06_dropped_utf8_bytes": 0,
    }
    for record in records:
        record_id = record["record_id"]
        payload = record["normalized_payload"]
        payload_raw = payload.encode("utf-8")
        row = by_id[record_id]
        _require(
            row.get("payload_sha256") == _sha256(payload_raw)
            and row.get("utf8_bytes") == len(payload_raw)
            and row.get("mode") == record["modality"],
            f"G06 payload binding drift: {record_id}",
        )
        action = row.get("action")
        _require(
            action in {"ALLOW", "REDACT", "QUARANTINE", "EXCLUDE"},
            f"G06 action drift: {record_id}",
        )
        if action == "ALLOW":
            output.append(dict(record))
            continue
        if action == "REDACT":
            transformed = _redact_text(payload, detect(payload))
            rescanned = scan(transformed.encode("utf-8"))
            _require(
                getattr(rescanned, "action", None) == "ALLOW",
                f"G06 redaction did not rescan ALLOW: {record_id}",
            )
            changed = dict(record)
            changed["normalized_payload"] = transformed
            output.append(changed)
            stats["g06_redacted_records"] += 1
            continue
        stats[f"g06_{action.lower()}_records"] += 1
        stats["g06_dropped_utf8_bytes"] += len(payload_raw)

    _require(bool(output), "G06 removed every quality survivor")
    output.sort(key=lambda row: row["record_id"])
    return output, stats


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
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, str]],
    dict[str, Any],
]:
    """Execute clean decontamination, G05, G06, and physical survivor materialization."""
    dependency_blobs = verify_dependency_blobs()
    _verify_clean_release_binding(training_records, training_handoff_evidence)

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

    quality_inputs, metadata, decontam_excluded = _post_decontamination_records(
        training_records,
        report,
    )
    decontam_projection = _input_projection(quality_inputs)
    post_decontam_rows_sha256 = _sha256(_cjson(decontam_projection))
    decontam_identity = _require_sha256(
        decontam_evidence.get("execution_identity_sha256"),
        "decontamination execution identity",
    )
    quality = build_quality_execution_authority(
        quality_inputs,
        input_manifest_sha256=decontam_identity,
        expected_input_rows_sha256=post_decontam_rows_sha256,
    )
    quality_identity = _require_sha256(
        quality.get("execution_identity_sha256"),
        "quality execution identity",
    )
    verify_quality_execution_authority(
        quality,
        quality_inputs,
        expected_input_manifest_sha256=decontam_identity,
        expected_input_rows_sha256=post_decontam_rows_sha256,
        expected_execution_identity_sha256=quality_identity,
    )

    quality_survivors, quality_stats = _materialize_quality_survivors(
        quality_inputs,
        metadata,
        quality,
    )
    privacy_inputs = _quality_records_for_privacy(quality_survivors)
    privacy_projection = _input_projection(privacy_inputs)
    post_quality_rows_sha256 = _sha256(_cjson(privacy_projection))
    privacy = build_privacy_execution_authority(
        privacy_inputs,
        expected_input_rows_sha256=post_quality_rows_sha256,
    )
    privacy_identity = _require_sha256(
        privacy.get("execution_identity_sha256"),
        "privacy execution identity",
    )
    verify_privacy_execution_authority(
        privacy,
        privacy_inputs,
        expected_input_rows_sha256=post_quality_rows_sha256,
        expected_execution_identity_sha256=privacy_identity,
    )

    final_survivors, privacy_stats = _materialize_privacy_survivors(
        quality_survivors,
        privacy,
    )
    survivor_inventory = materialize_record_inventory(final_survivors)
    survivor_jsonl_sha256 = _sha256(canonical_record_bytes(final_survivors))
    _require(
        survivor_inventory.get("record_count") == len(final_survivors),
        "survivor inventory record count drift",
    )
    _require(
        survivor_inventory.get("total_payload_bytes")
        == sum(
            len(row["normalized_payload"].encode("utf-8"))
            for row in final_survivors
        ),
        "survivor inventory byte count drift",
    )

    rejection_counts = {
        "data232_excluded_records": decontam_excluded,
        **quality_stats,
        **privacy_stats,
    }
    _require(
        set(rejection_counts) == _REJECTION_KEYS,
        "rejection-count schema drift",
    )
    detector_counts = privacy.get("detector_counts")
    _require(isinstance(detector_counts, Mapping), "G06 detector counts missing")
    privacy_detector_counts: dict[str, int] = {}
    for key, value in detector_counts.items():
        _require(
            isinstance(key, str)
            and bool(key)
            and type(value) is int
            and value >= 0,
            "G06 detector count malformed",
        )
        privacy_detector_counts[key] = value

    receipt: dict[str, Any] = {
        "schema_version": COMPOSITION_SCHEMA,
        "status": "CLEAN_SURVIVOR_MATERIALIZED_PENDING_INDEPENDENT_QUALIFICATION",
        "clean_training_records_sha256": PRODUCTION_TRAINING_RECORDS_SHA256,
        "clean_training_handoff_sha256": PRODUCTION_TRAINING_HANDOFF_SHA256,
        "data232_report_sha256": _require_sha256(
            report.get("report_sha256"),
            "DATA-232 report identity",
        ),
        "decontamination_execution_identity_sha256": decontam_identity,
        "eval647_execution_receipt_identity_sha256": _require_sha256(
            eval647_receipt.get("receipt_identity_sha256"),
            "EVAL-647 receipt identity",
        ),
        "quality_execution_identity_sha256": quality_identity,
        "privacy_execution_identity_sha256": privacy_identity,
        "post_decontamination_input_rows_sha256": post_decontam_rows_sha256,
        "post_quality_input_rows_sha256": post_quality_rows_sha256,
        "survivor_jsonl_sha256": survivor_jsonl_sha256,
        "survivor_record_inventory_digest_sha256": _require_sha256(
            survivor_inventory.get("record_inventory_digest_sha256"),
            "survivor record inventory",
        ),
        "survivor_payload_inventory_digest_sha256": _require_sha256(
            survivor_inventory.get("payload_inventory_digest_sha256"),
            "survivor payload inventory",
        ),
        "input_training_records": len(training_records),
        "post_decontamination_records": len(quality_inputs),
        "post_quality_records": len(quality_survivors),
        "survivor_records": len(final_survivors),
        "survivor_payload_bytes": survivor_inventory["total_payload_bytes"],
        "survivor_source_objects": len(
            {row["source_id"] for row in final_survivors}
        ),
        "rejection_counts": rejection_counts,
        "privacy_detector_counts": dict(sorted(privacy_detector_counts.items())),
        "dependency_git_blobs": dependency_blobs,
        "durable_evidence_hash_only": True,
        "terminal_post_g05_g06_authority": False,
        "independent_qualification_required": True,
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
    return (
        receipt,
        report,
        decontam_evidence,
        eval647_receipt,
        quality,
        privacy,
        final_survivors,
        survivor_inventory,
    )


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
    expected_post_quality_input_rows_sha256: str,
    expected_survivor_jsonl_sha256: str,
    expected_survivor_record_inventory_digest_sha256: str,
    expected_survivor_payload_inventory_digest_sha256: str,
) -> str:
    """Verify zero-credit survivor candidate evidence against independent pins."""
    _require(
        isinstance(receipt, Mapping) and set(receipt) == _RECEIPT_KEYS,
        "receipt schema is not closed",
    )
    _require(receipt.get("schema_version") == COMPOSITION_SCHEMA, "receipt schema drift")
    _require(
        receipt.get("status")
        == "CLEAN_SURVIVOR_MATERIALIZED_PENDING_INDEPENDENT_QUALIFICATION",
        "receipt status drift",
    )
    expected = _require_sha256(
        expected_receipt_identity_sha256,
        "expected receipt identity",
    )
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
        "post_quality_input_rows_sha256": expected_post_quality_input_rows_sha256,
        "survivor_jsonl_sha256": expected_survivor_jsonl_sha256,
        "survivor_record_inventory_digest_sha256": (
            expected_survivor_record_inventory_digest_sha256
        ),
        "survivor_payload_inventory_digest_sha256": (
            expected_survivor_payload_inventory_digest_sha256
        ),
    }
    for key, value in expected_nested.items():
        if receipt.get(key) != _require_sha256(value, f"expected {key}"):
            raise CurrentCleanExecutionError(f"nested execution root drift: {key}")

    _require(
        receipt.get("clean_training_records_sha256")
        == PRODUCTION_TRAINING_RECORDS_SHA256,
        "clean training root drift",
    )
    _require(
        receipt.get("clean_training_handoff_sha256")
        == PRODUCTION_TRAINING_HANDOFF_SHA256,
        "clean handoff root drift",
    )
    _require(
        receipt.get("dependency_git_blobs") == EXPECTED_DEPENDENCY_BLOBS,
        "dependency blob binding drift",
    )
    _require(
        receipt.get("durable_evidence_hash_only") is True,
        "durable evidence boundary weakened",
    )
    _require(
        receipt.get("terminal_post_g05_g06_authority") is False,
        "terminal post-G05/G06 authority fabricated",
    )
    _require(
        receipt.get("independent_qualification_required") is True,
        "independent qualification requirement erased",
    )
    _require(
        receipt.get("current_corpus_launch_authority_promoted") is False,
        "corpus launch authority fabricated",
    )
    for key in (
        "authorized_optimized_target_exposure",
        "optimizer_updates_executed_on_real_targets",
    ):
        _require(
            type(receipt.get(key)) is int and receipt.get(key) == 0,
            f"authority widened: {key}",
        )
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
        "input_training_records",
        "post_decontamination_records",
        "post_quality_records",
        "survivor_records",
        "survivor_payload_bytes",
        "survivor_source_objects",
    ):
        _require(
            type(receipt.get(key)) is int and receipt.get(key) > 0,
            f"invalid positive count: {key}",
        )
    _require(
        receipt["input_training_records"] == PRODUCTION_INPUT_RECORD_COUNT,
        "production input count drift",
    )
    _require(
        receipt["input_training_records"]
        >= receipt["post_decontamination_records"]
        >= receipt["post_quality_records"]
        >= receipt["survivor_records"],
        "survivor count monotonicity drift",
    )

    rejection_counts = receipt.get("rejection_counts")
    _require(
        isinstance(rejection_counts, Mapping)
        and set(rejection_counts) == _REJECTION_KEYS,
        "rejection-count schema drift",
    )
    for key, value in rejection_counts.items():
        _require(
            type(value) is int and value >= 0,
            f"invalid rejection count: {key}",
        )
    detector_counts = receipt.get("privacy_detector_counts")
    _require(isinstance(detector_counts, Mapping), "privacy detector counts missing")
    for key, value in detector_counts.items():
        _require(
            isinstance(key, str)
            and bool(key)
            and type(value) is int
            and value >= 0,
            "privacy detector count malformed",
        )
    return claimed
