"""Payload-bound G05 quality execution authority.

The frozen D03 quality threshold and granularity policies remain authoritative.
This module only records deterministic, text-free execution evidence over an
exact externally bound input graph. It grants no corpus or training authority.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from twelve_six.data.document_quality import default_quality_policy
from twelve_six.data.quality_granularity import (
    apply_frozen_granularity,
    frozen_granularity_policy,
)

QUALITY_EXECUTION_SCHEMA = "12-6.g05-quality-execution-authority.v1"
QUALITY_EXECUTION_AUTHORITY_CLASS = "G05_QUALITY_EXECUTION_ZERO_CREDIT"
_TRUTH_BOUNDARY = {
    "current_corpus_eligible": False,
    "training_authorized_bytes": 0,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates_executed": 0,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights": False,
    "external_llm_or_api_used_for_data_or_intelligence": False,
    "source_text_retained_in_authority": False,
}
_INPUT_KEYS = frozenset({"id", "text", "mode"})
_MODES = frozenset({"uk", "en", "code"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_AUTHORITY_KEYS = frozenset(
    {
        "schema_version",
        "authority_class",
        "input_manifest_sha256",
        "input_rows_sha256",
        "quality_threshold_policy",
        "quality_granularity_policy",
        "execution_rows_sha256",
        "counts",
        "bytes",
        "records",
        "truth_boundary",
        "execution_identity_sha256",
    }
)
_POLICY_KEYS = frozenset({"policy_id", "policy_sha256"})
_COUNTS_KEYS = frozenset(
    {
        "records",
        "retain_all",
        "retain_partial",
        "reject_document",
        "quality_units",
        "accepted_quality_units",
        "rejected_quality_units",
    }
)
_BYTES_KEYS = frozenset(
    {"input_utf8_bytes", "retained_utf8_bytes", "rejected_utf8_bytes"}
)
_EXECUTION_ROW_KEYS = frozenset(
    {
        "record_id",
        "mode",
        "payload_sha256",
        "utf8_bytes",
        "authoritative_unit",
        "status",
        "retained_utf8_bytes",
        "rejected_utf8_bytes",
        "quality_result_sha256",
        "units",
    }
)
_UNIT_KEYS = frozenset(
    {
        "unit_id",
        "start_char",
        "end_char",
        "payload_sha256",
        "utf8_bytes",
        "accepted",
        "decision_sha256",
    }
)
_STATUSES = frozenset({"RETAIN_ALL", "RETAIN_PARTIAL", "REJECT_DOCUMENT"})


class QualityExecutionAuthorityError(ValueError):
    """Raised when G05 execution authority is malformed or unverifiable."""


def _cjson(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise QualityExecutionAuthorityError(
            f"{field} must be lowercase SHA-256 hex"
        )
    return value


def _require_exact_int(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise QualityExecutionAuthorityError(
            f"{field} must be a non-negative integer"
        )
    return value


def _require_exact_bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise QualityExecutionAuthorityError(f"{field} must be a boolean")
    return value


def _require_nonempty_str(value: Any, field: str) -> str:
    if type(value) is not str or not value:
        raise QualityExecutionAuthorityError(f"{field} must be a non-empty string")
    return value


def _require_exact_dict(
    value: Any,
    *,
    field: str,
    keys: frozenset[str],
) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise QualityExecutionAuthorityError(f"{field} schema is not closed")
    return value


def _require_exact_list(value: Any, field: str) -> list[Any]:
    if type(value) is not list:
        raise QualityExecutionAuthorityError(f"{field} must be a JSON list")
    return value


def _normalize_inputs(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise QualityExecutionAuthorityError("records must be a sequence")
    if not records:
        raise QualityExecutionAuthorityError("records must not be empty")

    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(records):
        if not isinstance(row, Mapping) or set(row) != _INPUT_KEYS:
            raise QualityExecutionAuthorityError(
                f"records[{index}] must contain exactly id/text/mode"
            )
        record_id = row["id"]
        text = row["text"]
        mode = row["mode"]
        if not isinstance(record_id, str) or not record_id:
            raise QualityExecutionAuthorityError(
                f"records[{index}].id must be a non-empty string"
            )
        if record_id in seen:
            raise QualityExecutionAuthorityError(
                f"duplicate record id: {record_id}"
            )
        if not isinstance(text, str):
            raise QualityExecutionAuthorityError(
                f"records[{index}].text must be a string"
            )
        if not isinstance(mode, str) or mode not in _MODES:
            raise QualityExecutionAuthorityError(
                f"records[{index}].mode must be uk/en/code"
            )
        try:
            text.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise QualityExecutionAuthorityError(
                f"records[{index}].text must encode as strict UTF-8"
            ) from exc
        seen.add(record_id)
        normalized.append({"id": record_id, "text": text, "mode": mode})
    return sorted(normalized, key=lambda row: row["id"])


def _input_row_projection(
    rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    return [
        {
            "record_id": row["id"],
            "mode": row["mode"],
            "payload_sha256": _sha256(row["text"].encode("utf-8")),
            "utf8_bytes": len(row["text"].encode("utf-8")),
        }
        for row in rows
    ]


def _quality_units(
    record_id: str,
    text: str,
    result: Mapping[str, Any],
) -> list[dict[str, Any]]:
    windows = result["windows"]
    if not windows:
        diagnostic = result["full_document_diagnostic"]
        accepted = result["status"] == "RETAIN_ALL"
        payload = text.encode("utf-8")
        return [
            {
                "unit_id": record_id,
                "start_char": 0,
                "end_char": len(text),
                "payload_sha256": _sha256(payload),
                "utf8_bytes": len(payload),
                "accepted": accepted,
                "decision_sha256": _sha256(_cjson(diagnostic)),
            }
        ]

    units: list[dict[str, Any]] = []
    for window in windows:
        index = _require_exact_int(window["index"], "window.index")
        start = _require_exact_int(window["start_char"], "window.start_char")
        end = _require_exact_int(window["end_char"], "window.end_char")
        if not start < end <= len(text):
            raise QualityExecutionAuthorityError(
                "canonical quality window span is invalid"
            )
        decision = window["decision"]
        if not isinstance(decision.get("accepted"), bool):
            raise QualityExecutionAuthorityError(
                "canonical quality decision must expose bool accepted"
            )
        payload = text[start:end].encode("utf-8")
        if _require_exact_int(
            window["utf8_bytes"], "window.utf8_bytes"
        ) != len(payload):
            raise QualityExecutionAuthorityError(
                "canonical quality window byte accounting drift"
            )
        units.append(
            {
                "unit_id": f"{record_id}#quality-window-{index:04d}",
                "start_char": start,
                "end_char": end,
                "payload_sha256": _sha256(payload),
                "utf8_bytes": len(payload),
                "accepted": decision["accepted"],
                "decision_sha256": _sha256(_cjson(decision)),
            }
        )
    return units


def build_quality_execution_authority(
    records: Sequence[Mapping[str, Any]],
    *,
    input_manifest_sha256: str,
    expected_input_rows_sha256: str,
) -> dict[str, Any]:
    """Run frozen G05 quality semantics over an independently bound row set."""
    input_manifest = _require_sha256(
        input_manifest_sha256, "input_manifest_sha256"
    )
    expected_input_rows = _require_sha256(
        expected_input_rows_sha256, "expected_input_rows_sha256"
    )
    rows = _normalize_inputs(records)
    input_rows = _input_row_projection(rows)
    input_rows_sha256 = _sha256(_cjson(input_rows))
    if input_rows_sha256 != expected_input_rows:
        raise QualityExecutionAuthorityError("input rows authority mismatch")
    quality = default_quality_policy()
    granularity = frozen_granularity_policy()
    quality_manifest = quality.manifest()
    granularity_manifest = granularity.manifest()
    if (
        quality_manifest["policy_sha256"]
        != granularity.quality_threshold_policy_sha256
    ):
        raise QualityExecutionAuthorityError(
            "quality threshold/granularity policy binding drift"
        )

    execution_rows: list[dict[str, Any]] = []
    status_counts = {
        "RETAIN_ALL": 0,
        "RETAIN_PARTIAL": 0,
        "REJECT_DOCUMENT": 0,
    }
    input_bytes = retained_bytes = rejected_bytes = 0
    accepted_units = rejected_units = 0

    for row in rows:
        record_id = row["id"]
        text = row["text"]
        mode = row["mode"]
        payload = text.encode("utf-8")
        payload_sha256 = _sha256(payload)
        result = apply_frozen_granularity(
            record_id,
            text,
            mode,
            quality_policy=quality,
            granularity_policy=granularity,
        )
        status = result["status"]
        if status not in status_counts:
            raise QualityExecutionAuthorityError(
                "canonical quality status is unsupported"
            )
        retained = _require_exact_int(
            result["retained_utf8_bytes"], "retained_utf8_bytes"
        )
        rejected = _require_exact_int(
            result["rejected_utf8_bytes"], "rejected_utf8_bytes"
        )
        if retained + rejected != len(payload):
            raise QualityExecutionAuthorityError(
                "canonical record byte accounting drift"
            )

        units = _quality_units(record_id, text, result)
        if sum(unit["utf8_bytes"] for unit in units) != len(payload):
            raise QualityExecutionAuthorityError(
                "quality-unit byte accounting drift"
            )
        if sum(
            unit["utf8_bytes"] for unit in units if unit["accepted"]
        ) != retained:
            raise QualityExecutionAuthorityError(
                "quality-unit retained-byte accounting drift"
            )

        execution_rows.append(
            {
                "record_id": record_id,
                "mode": mode,
                "payload_sha256": payload_sha256,
                "utf8_bytes": len(payload),
                "authoritative_unit": result["authoritative_unit"],
                "status": status,
                "retained_utf8_bytes": retained,
                "rejected_utf8_bytes": rejected,
                "quality_result_sha256": _sha256(_cjson(result)),
                "units": units,
            }
        )
        status_counts[status] += 1
        input_bytes += len(payload)
        retained_bytes += retained
        rejected_bytes += rejected
        accepted_units += sum(unit["accepted"] for unit in units)
        rejected_units += sum(not unit["accepted"] for unit in units)

    counts = {
        "records": len(execution_rows),
        "retain_all": status_counts["RETAIN_ALL"],
        "retain_partial": status_counts["RETAIN_PARTIAL"],
        "reject_document": status_counts["REJECT_DOCUMENT"],
        "quality_units": accepted_units + rejected_units,
        "accepted_quality_units": accepted_units,
        "rejected_quality_units": rejected_units,
    }
    byte_summary = {
        "input_utf8_bytes": input_bytes,
        "retained_utf8_bytes": retained_bytes,
        "rejected_utf8_bytes": rejected_bytes,
    }
    core = {
        "schema_version": QUALITY_EXECUTION_SCHEMA,
        "authority_class": QUALITY_EXECUTION_AUTHORITY_CLASS,
        "input_manifest_sha256": input_manifest,
        "input_rows_sha256": input_rows_sha256,
        "quality_threshold_policy": {
            "policy_id": quality.policy_id,
            "policy_sha256": quality_manifest["policy_sha256"],
        },
        "quality_granularity_policy": {
            "policy_id": granularity.policy_id,
            "policy_sha256": granularity_manifest["policy_sha256"],
        },
        "execution_rows_sha256": _sha256(_cjson(execution_rows)),
        "counts": counts,
        "bytes": byte_summary,
        "records": execution_rows,
        "truth_boundary": dict(_TRUTH_BOUNDARY),
    }
    return {
        **core,
        "execution_identity_sha256": _sha256(_cjson(core)),
    }


def _verify_quality_execution_structure(
    authority: Mapping[str, Any],
    *,
    expected_input_manifest_sha256: str,
    expected_input_rows_sha256: str,
) -> str:
    expected_input = _require_sha256(
        expected_input_manifest_sha256, "expected_input_manifest_sha256"
    )
    expected_input_rows = _require_sha256(
        expected_input_rows_sha256, "expected_input_rows_sha256"
    )
    root = _require_exact_dict(
        authority,
        field="quality execution root",
        keys=_AUTHORITY_KEYS,
    )
    if type(root["schema_version"]) is not str or (
        root["schema_version"] != QUALITY_EXECUTION_SCHEMA
    ):
        raise QualityExecutionAuthorityError("quality execution schema drift")
    if type(root["authority_class"]) is not str or (
        root["authority_class"] != QUALITY_EXECUTION_AUTHORITY_CLASS
    ):
        raise QualityExecutionAuthorityError(
            "quality execution authority class drift"
        )
    if _require_sha256(
        root["input_manifest_sha256"], "input_manifest_sha256"
    ) != expected_input:
        raise QualityExecutionAuthorityError("input manifest authority mismatch")
    if _require_sha256(
        root["input_rows_sha256"], "input_rows_sha256"
    ) != expected_input_rows:
        raise QualityExecutionAuthorityError("input rows authority mismatch")

    quality = default_quality_policy()
    granularity = frozen_granularity_policy()
    expected_quality_policy = {
        "policy_id": quality.policy_id,
        "policy_sha256": quality.manifest()["policy_sha256"],
    }
    expected_granularity_policy = {
        "policy_id": granularity.policy_id,
        "policy_sha256": granularity.manifest()["policy_sha256"],
    }
    _require_exact_dict(
        root["quality_threshold_policy"],
        field="quality_threshold_policy",
        keys=_POLICY_KEYS,
    )
    if _cjson(root["quality_threshold_policy"]) != _cjson(
        expected_quality_policy
    ):
        raise QualityExecutionAuthorityError(
            "quality threshold policy identity drift"
        )
    _require_exact_dict(
        root["quality_granularity_policy"],
        field="quality_granularity_policy",
        keys=_POLICY_KEYS,
    )
    if _cjson(root["quality_granularity_policy"]) != _cjson(
        expected_granularity_policy
    ):
        raise QualityExecutionAuthorityError(
            "quality granularity policy identity drift"
        )

    truth_boundary = _require_exact_dict(
        root["truth_boundary"],
        field="truth_boundary",
        keys=frozenset(_TRUTH_BOUNDARY),
    )
    if _cjson(truth_boundary) != _cjson(_TRUTH_BOUNDARY):
        raise QualityExecutionAuthorityError("quality execution truth boundary drift")

    records = _require_exact_list(root["records"], "records")
    if not records:
        raise QualityExecutionAuthorityError("records must not be empty")

    input_rows: list[dict[str, Any]] = []
    status_counts = {status: 0 for status in _STATUSES}
    input_bytes = retained_bytes = rejected_bytes = 0
    accepted_units = rejected_units = 0
    previous_record_id: str | None = None

    for row_index, raw_row in enumerate(records):
        row = _require_exact_dict(
            raw_row,
            field=f"records[{row_index}]",
            keys=_EXECUTION_ROW_KEYS,
        )
        record_id = _require_nonempty_str(
            row["record_id"], f"records[{row_index}].record_id"
        )
        if previous_record_id is not None and record_id <= previous_record_id:
            raise QualityExecutionAuthorityError(
                "records must be strictly ordered by unique record_id"
            )
        previous_record_id = record_id
        mode = _require_nonempty_str(
            row["mode"], f"records[{row_index}].mode"
        )
        if mode not in _MODES:
            raise QualityExecutionAuthorityError(
                f"records[{row_index}].mode must be uk/en/code"
            )
        payload_sha256 = _require_sha256(
            row["payload_sha256"],
            f"records[{row_index}].payload_sha256",
        )
        utf8_bytes = _require_exact_int(
            row["utf8_bytes"], f"records[{row_index}].utf8_bytes"
        )
        _require_nonempty_str(
            row["authoritative_unit"],
            f"records[{row_index}].authoritative_unit",
        )
        status = _require_nonempty_str(
            row["status"], f"records[{row_index}].status"
        )
        if status not in _STATUSES:
            raise QualityExecutionAuthorityError(
                f"records[{row_index}].status is unsupported"
            )
        retained = _require_exact_int(
            row["retained_utf8_bytes"],
            f"records[{row_index}].retained_utf8_bytes",
        )
        rejected = _require_exact_int(
            row["rejected_utf8_bytes"],
            f"records[{row_index}].rejected_utf8_bytes",
        )
        if retained + rejected != utf8_bytes:
            raise QualityExecutionAuthorityError(
                f"records[{row_index}] byte accounting drift"
            )
        _require_sha256(
            row["quality_result_sha256"],
            f"records[{row_index}].quality_result_sha256",
        )

        units = _require_exact_list(
            row["units"], f"records[{row_index}].units"
        )
        if not units:
            raise QualityExecutionAuthorityError(
                f"records[{row_index}].units must not be empty"
            )
        row_unit_bytes = row_accepted_bytes = row_rejected_bytes = 0
        unit_ids: set[str] = set()
        for unit_index, raw_unit in enumerate(units):
            unit = _require_exact_dict(
                raw_unit,
                field=f"records[{row_index}].units[{unit_index}]",
                keys=_UNIT_KEYS,
            )
            unit_id = _require_nonempty_str(
                unit["unit_id"],
                f"records[{row_index}].units[{unit_index}].unit_id",
            )
            if unit_id in unit_ids:
                raise QualityExecutionAuthorityError(
                    f"records[{row_index}] has duplicate unit_id"
                )
            unit_ids.add(unit_id)
            start_char = _require_exact_int(
                unit["start_char"],
                f"records[{row_index}].units[{unit_index}].start_char",
            )
            end_char = _require_exact_int(
                unit["end_char"],
                f"records[{row_index}].units[{unit_index}].end_char",
            )
            if end_char < start_char:
                raise QualityExecutionAuthorityError(
                    f"records[{row_index}] has invalid unit character span"
                )
            _require_sha256(
                unit["payload_sha256"],
                f"records[{row_index}].units[{unit_index}].payload_sha256",
            )
            unit_bytes = _require_exact_int(
                unit["utf8_bytes"],
                f"records[{row_index}].units[{unit_index}].utf8_bytes",
            )
            accepted = _require_exact_bool(
                unit["accepted"],
                f"records[{row_index}].units[{unit_index}].accepted",
            )
            _require_sha256(
                unit["decision_sha256"],
                f"records[{row_index}].units[{unit_index}].decision_sha256",
            )
            row_unit_bytes += unit_bytes
            if accepted:
                row_accepted_bytes += unit_bytes
                accepted_units += 1
            else:
                row_rejected_bytes += unit_bytes
                rejected_units += 1

        if row_unit_bytes != utf8_bytes:
            raise QualityExecutionAuthorityError(
                f"records[{row_index}] quality-unit byte accounting drift"
            )
        if row_accepted_bytes != retained or row_rejected_bytes != rejected:
            raise QualityExecutionAuthorityError(
                f"records[{row_index}] quality-unit disposition accounting drift"
            )

        input_rows.append(
            {
                "record_id": record_id,
                "mode": mode,
                "payload_sha256": payload_sha256,
                "utf8_bytes": utf8_bytes,
            }
        )
        status_counts[status] += 1
        input_bytes += utf8_bytes
        retained_bytes += retained
        rejected_bytes += rejected

    observed_input_rows = _sha256(_cjson(input_rows))
    if observed_input_rows != root["input_rows_sha256"]:
        raise QualityExecutionAuthorityError("input rows self-hash mismatch")
    observed_execution_rows = _sha256(_cjson(records))
    if _require_sha256(
        root["execution_rows_sha256"], "execution_rows_sha256"
    ) != observed_execution_rows:
        raise QualityExecutionAuthorityError("execution rows self-hash mismatch")

    expected_counts = {
        "records": len(records),
        "retain_all": status_counts["RETAIN_ALL"],
        "retain_partial": status_counts["RETAIN_PARTIAL"],
        "reject_document": status_counts["REJECT_DOCUMENT"],
        "quality_units": accepted_units + rejected_units,
        "accepted_quality_units": accepted_units,
        "rejected_quality_units": rejected_units,
    }
    counts = _require_exact_dict(
        root["counts"], field="counts", keys=_COUNTS_KEYS
    )
    for key in _COUNTS_KEYS:
        _require_exact_int(counts[key], f"counts.{key}")
    if _cjson(counts) != _cjson(expected_counts):
        raise QualityExecutionAuthorityError("quality execution count accounting drift")

    expected_bytes = {
        "input_utf8_bytes": input_bytes,
        "retained_utf8_bytes": retained_bytes,
        "rejected_utf8_bytes": rejected_bytes,
    }
    byte_summary = _require_exact_dict(
        root["bytes"], field="bytes", keys=_BYTES_KEYS
    )
    for key in _BYTES_KEYS:
        _require_exact_int(byte_summary[key], f"bytes.{key}")
    if _cjson(byte_summary) != _cjson(expected_bytes):
        raise QualityExecutionAuthorityError("quality execution byte accounting drift")

    observed = _require_sha256(
        root["execution_identity_sha256"], "execution_identity_sha256"
    )
    core = dict(root)
    del core["execution_identity_sha256"]
    if observed != _sha256(_cjson(core)):
        raise QualityExecutionAuthorityError(
            "quality execution self-hash mismatch"
        )
    return observed


def verify_quality_execution_root(
    authority: Mapping[str, Any],
    *,
    expected_input_manifest_sha256: str,
    expected_input_rows_sha256: str,
    expected_execution_identity_sha256: str,
) -> str:
    """Verify a text-free root already pinned by an independent executor/auditor."""
    expected_identity = _require_sha256(
        expected_execution_identity_sha256,
        "expected_execution_identity_sha256",
    )
    observed = _verify_quality_execution_structure(
        authority,
        expected_input_manifest_sha256=expected_input_manifest_sha256,
        expected_input_rows_sha256=expected_input_rows_sha256,
    )
    if observed != expected_identity:
        raise QualityExecutionAuthorityError(
            "quality execution authority root mismatch"
        )
    return observed


def verify_quality_execution_authority(
    authority: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    expected_input_manifest_sha256: str,
    expected_input_rows_sha256: str,
    expected_execution_identity_sha256: str | None = None,
) -> str:
    """Re-execute frozen quality semantics and require byte-exact JSON equality."""
    expected = build_quality_execution_authority(
        records,
        input_manifest_sha256=expected_input_manifest_sha256,
        expected_input_rows_sha256=expected_input_rows_sha256,
    )
    observed = _verify_quality_execution_structure(
        authority,
        expected_input_manifest_sha256=expected_input_manifest_sha256,
        expected_input_rows_sha256=expected_input_rows_sha256,
    )
    if _cjson(authority) != _cjson(expected):
        raise QualityExecutionAuthorityError(
            "quality execution authority does not match canonical re-execution"
        )
    if expected_execution_identity_sha256 is not None:
        expected_identity = _require_sha256(
            expected_execution_identity_sha256,
            "expected_execution_identity_sha256",
        )
        if observed != expected_identity:
            raise QualityExecutionAuthorityError(
                "quality execution authority root mismatch"
            )
    return observed
