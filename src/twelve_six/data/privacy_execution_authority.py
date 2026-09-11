"""Payload-bound G06 privacy execution authority.

This module wraps the already-qualified privacy_filter_v3 mechanics without
changing detector semantics. It emits deterministic text-free execution
evidence over an exact externally bound input graph and grants no corpus or
training authority by itself.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PRIVACY_EXECUTION_SCHEMA = "12-6.g06-privacy-execution-authority.v1"
PRIVACY_EXECUTION_AUTHORITY_CLASS = "G06_PRIVACY_EXECUTION_ZERO_CREDIT"
PRIVACY_MODULE = "twelve_six.data.privacy_filter_v3"
PRIVACY_SOURCE_RELATIVE_PATH = "privacy_filter_v3.py"
EXPECTED_PRIVACY_IMPLEMENTATION_GIT_BLOB_SHA1 = (
    "bcc5938395724f6728ab212f98b39f2334b0f37d"
)

_INPUT_KEYS = frozenset({"id", "text", "mode"})
_INVENTORY_ROW_KEYS = frozenset(
    {
        "record_id",
        "source_id",
        "family",
        "modality",
        "payload_sha256",
        "payload_bytes",
    }
)
_EXECUTION_ROW_KEYS = frozenset(
    {
        "record_id",
        "mode",
        "payload_sha256",
        "utf8_bytes",
        "action",
        "detector_counts",
        "privacy_result_sha256",
    }
)
_MODES = frozenset({"uk", "en", "code"})
_ACTIONS = frozenset({"ALLOW", "REDACT", "QUARANTINE", "EXCLUDE"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SHA1_RE = re.compile(r"[0-9a-f]{40}")

_TRUTH_BOUNDARY = {
    "dependency_bound_g06_executed": True,
    "canonical_current_corpus_g06": False,
    "current_corpus_eligible": False,
    "training_authorized_bytes": 0,
    "authorized_unique_loss_positions": 0,
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
    "text_previews_retained_in_authority": False,
    "matched_values_retained_in_authority": False,
    "matched_value_hashes_retained_in_authority": False,
}


class PrivacyExecutionAuthorityError(ValueError):
    """Raised when G06 execution authority is malformed or unverifiable."""


def _cjson(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _git_blob_sha1(payload: bytes) -> str:
    header = f"blob {len(payload)}\0".encode("ascii")
    return hashlib.sha1(header + payload, usedforsecurity=False).hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise PrivacyExecutionAuthorityError(
            f"{field} must be lowercase SHA-256 hex"
        )
    return value


def _require_sha1(value: Any, field: str) -> str:
    if not isinstance(value, str) or _SHA1_RE.fullmatch(value) is None:
        raise PrivacyExecutionAuthorityError(
            f"{field} must be lowercase Git SHA-1 hex"
        )
    return value


def _require_exact_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise PrivacyExecutionAuthorityError(
            f"{field} must be an integer >= {minimum}"
        )
    return value


def _require_bool(value: Any, field: str) -> bool:
    if type(value) is not bool:
        raise PrivacyExecutionAuthorityError(f"{field} must be bool")
    return value


def _normalize_inputs(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise PrivacyExecutionAuthorityError("records must be a sequence")
    if not records:
        raise PrivacyExecutionAuthorityError("records must not be empty")

    normalized: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, row in enumerate(records):
        if not isinstance(row, Mapping) or set(row) != _INPUT_KEYS:
            raise PrivacyExecutionAuthorityError(
                f"records[{index}] must contain exactly id/text/mode"
            )
        record_id = row["id"]
        text = row["text"]
        mode = row["mode"]
        if not isinstance(record_id, str) or not record_id:
            raise PrivacyExecutionAuthorityError(
                f"records[{index}].id must be a non-empty string"
            )
        if record_id in seen:
            raise PrivacyExecutionAuthorityError(
                f"duplicate record id: {record_id}"
            )
        if not isinstance(text, str):
            raise PrivacyExecutionAuthorityError(
                f"records[{index}].text must be a string"
            )
        if not isinstance(mode, str) or mode not in _MODES:
            raise PrivacyExecutionAuthorityError(
                f"records[{index}].mode must be uk/en/code"
            )
        try:
            text.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise PrivacyExecutionAuthorityError(
                f"records[{index}].text must encode as strict UTF-8"
            ) from exc
        seen.add(record_id)
        normalized.append({"id": record_id, "text": text, "mode": mode})
    return sorted(normalized, key=lambda row: row["id"])


def _input_rows_from_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = _normalize_inputs(records)
    result: list[dict[str, Any]] = []
    for row in rows:
        payload = row["text"].encode("utf-8")
        result.append(
            {
                "record_id": row["id"],
                "mode": row["mode"],
                "payload_sha256": _sha256(payload),
                "utf8_bytes": len(payload),
            }
        )
    return result


def input_rows_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    """Return the exact G06 input-row root computed from raw payloads."""
    return _sha256(_cjson(_input_rows_from_records(records)))


def input_rows_sha256_from_text_free_inventory(
    rows: Sequence[Mapping[str, Any]],
) -> str:
    """Bridge a DATA526 text-free inventory into the exact G06 input root."""
    if isinstance(rows, (str, bytes)) or not isinstance(rows, Sequence):
        raise PrivacyExecutionAuthorityError("inventory rows must be a sequence")
    if not rows:
        raise PrivacyExecutionAuthorityError("inventory rows must not be empty")

    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _INVENTORY_ROW_KEYS:
            raise PrivacyExecutionAuthorityError(
                f"inventory[{index}] schema drift"
            )
        record_id = row["record_id"]
        mode = row["modality"]
        payload_sha256 = row["payload_sha256"]
        payload_bytes = row["payload_bytes"]
        if not isinstance(record_id, str) or not record_id:
            raise PrivacyExecutionAuthorityError(
                f"inventory[{index}].record_id must be non-empty string"
            )
        if record_id in seen:
            raise PrivacyExecutionAuthorityError(
                f"duplicate inventory record id: {record_id}"
            )
        if not isinstance(mode, str) or mode not in _MODES:
            raise PrivacyExecutionAuthorityError(
                f"inventory[{index}].modality must be uk/en/code"
            )
        _require_sha256(payload_sha256, f"inventory[{index}].payload_sha256")
        nbytes = _require_exact_int(
            payload_bytes, f"inventory[{index}].payload_bytes"
        )
        for field in ("source_id", "family"):
            if not isinstance(row[field], str) or not row[field]:
                raise PrivacyExecutionAuthorityError(
                    f"inventory[{index}].{field} must be non-empty string"
                )
        seen.add(record_id)
        normalized.append(
            {
                "record_id": record_id,
                "mode": mode,
                "payload_sha256": payload_sha256,
                "utf8_bytes": nbytes,
            }
        )
    normalized.sort(key=lambda row: row["record_id"])
    return _sha256(_cjson(normalized))


def _privacy_binding() -> tuple[Any, dict[str, str]]:
    module = importlib.import_module(PRIVACY_MODULE)
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file:
        raise PrivacyExecutionAuthorityError(
            "canonical privacy module has no source path"
        )
    try:
        observed_path = Path(module_file).resolve(strict=True)
        expected_path = Path(__file__).with_name(PRIVACY_SOURCE_RELATIVE_PATH).resolve(
            strict=True
        )
    except OSError as exc:
        raise PrivacyExecutionAuthorityError(
            "cannot resolve canonical privacy implementation"
        ) from exc
    if observed_path != expected_path:
        raise PrivacyExecutionAuthorityError(
            "privacy implementation resolved outside canonical repository path"
        )
    observed_blob = _git_blob_sha1(observed_path.read_bytes())
    if observed_blob != EXPECTED_PRIVACY_IMPLEMENTATION_GIT_BLOB_SHA1:
        raise PrivacyExecutionAuthorityError(
            "canonical privacy implementation Git blob drift"
        )

    scan = getattr(module, "hash_safe_scan", None)
    policy_provider = getattr(module, "policy_manifest", None)
    if not callable(scan) or getattr(scan, "__module__", None) != PRIVACY_MODULE:
        raise PrivacyExecutionAuthorityError(
            "privacy scanner callable provenance drift"
        )
    if (
        not callable(policy_provider)
        or getattr(policy_provider, "__module__", None) != PRIVACY_MODULE
    ):
        raise PrivacyExecutionAuthorityError(
            "privacy policy callable provenance drift"
        )

    manifest = policy_provider()
    expected_keys = {
        "schema_version",
        "coverage_claim",
        "detector_actions",
        "evidence_policy",
        "policy_sha256",
    }
    if not isinstance(manifest, Mapping) or set(manifest) != expected_keys:
        raise PrivacyExecutionAuthorityError("privacy policy manifest schema drift")
    claimed_policy = _require_sha256(
        manifest["policy_sha256"], "privacy policy_sha256"
    )
    core = dict(manifest)
    del core["policy_sha256"]
    if claimed_policy != _sha256(_cjson(core)):
        raise PrivacyExecutionAuthorityError("privacy policy self-hash mismatch")
    schema_version = manifest["schema_version"]
    if not isinstance(schema_version, str) or not schema_version:
        raise PrivacyExecutionAuthorityError("privacy policy schema is malformed")

    return scan, {
        "module": PRIVACY_MODULE,
        "relative_path": PRIVACY_SOURCE_RELATIVE_PATH,
        "implementation_git_blob_sha1": observed_blob,
        "policy_schema_version": schema_version,
        "policy_sha256": claimed_policy,
    }


def _validate_detector_counts(value: Any, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise PrivacyExecutionAuthorityError(f"{field} must be an object")
    result: dict[str, int] = {}
    for key, count in value.items():
        if not isinstance(key, str) or not key:
            raise PrivacyExecutionAuthorityError(
                f"{field} detector names must be non-empty strings"
            )
        result[key] = _require_exact_int(count, f"{field}.{key}")
    return dict(sorted(result.items()))


def build_privacy_execution_authority(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_input_rows_sha256: str,
) -> dict[str, Any]:
    """Run canonical G06 privacy mechanics over an externally bound input set."""
    expected_input = _require_sha256(
        expected_input_rows_sha256, "expected_input_rows_sha256"
    )
    rows = _normalize_inputs(records)
    input_rows = _input_rows_from_records(rows)
    observed_input = _sha256(_cjson(input_rows))
    if observed_input != expected_input:
        raise PrivacyExecutionAuthorityError(
            "raw payload set does not match externally expected input-row root"
        )

    scan, privacy_binding = _privacy_binding()
    execution_rows: list[dict[str, Any]] = []
    actions: Counter[str] = Counter({action: 0 for action in _ACTIONS})
    detectors: Counter[str] = Counter()
    total_bytes = 0

    for row, input_row in zip(rows, input_rows, strict=True):
        result = scan(row["text"])
        evidence = result.evidence()
        if not isinstance(evidence, Mapping) or set(evidence) != {
            "input_sha256",
            "input_bytes",
            "action",
            "detector_counts",
        }:
            raise PrivacyExecutionAuthorityError(
                "canonical privacy scan evidence schema drift"
            )
        if evidence["input_sha256"] != input_row["payload_sha256"]:
            raise PrivacyExecutionAuthorityError(
                "privacy scanner payload hash accounting drift"
            )
        if evidence["input_bytes"] != input_row["utf8_bytes"]:
            raise PrivacyExecutionAuthorityError(
                "privacy scanner byte accounting drift"
            )
        action = evidence["action"]
        if not isinstance(action, str) or action not in _ACTIONS:
            raise PrivacyExecutionAuthorityError(
                "canonical privacy action is unsupported"
            )
        detector_counts = _validate_detector_counts(
            evidence["detector_counts"],
            f"privacy_result[{row['id']}].detector_counts",
        )
        result_core = {
            "input_sha256": evidence["input_sha256"],
            "input_bytes": evidence["input_bytes"],
            "action": action,
            "detector_counts": detector_counts,
        }
        execution_rows.append(
            {
                **input_row,
                "action": action,
                "detector_counts": detector_counts,
                "privacy_result_sha256": _sha256(_cjson(result_core)),
            }
        )
        actions[action] += 1
        detectors.update(detector_counts)
        total_bytes += input_row["utf8_bytes"]

    counts = {
        "records": len(execution_rows),
        "allow": actions["ALLOW"],
        "redact": actions["REDACT"],
        "quarantine": actions["QUARANTINE"],
        "exclude": actions["EXCLUDE"],
    }
    core = {
        "schema_version": PRIVACY_EXECUTION_SCHEMA,
        "authority_class": PRIVACY_EXECUTION_AUTHORITY_CLASS,
        "expected_input_rows_sha256": expected_input,
        "observed_input_rows_sha256": observed_input,
        "privacy_binding": privacy_binding,
        "execution_rows_sha256": _sha256(_cjson(execution_rows)),
        "counts": counts,
        "total_input_utf8_bytes": total_bytes,
        "detector_counts": dict(sorted(detectors.items())),
        "records": execution_rows,
        "truth_boundary": dict(_TRUTH_BOUNDARY),
    }
    return {
        **core,
        "execution_identity_sha256": _sha256(_cjson(core)),
    }


def _validate_execution_rows(
    records: Any,
) -> tuple[list[dict[str, Any]], Counter[str], Counter[str], int]:
    if not isinstance(records, list) or not records:
        raise PrivacyExecutionAuthorityError(
            "privacy execution records must be a non-empty list"
        )
    normalized: list[dict[str, Any]] = []
    actions: Counter[str] = Counter({action: 0 for action in _ACTIONS})
    detectors: Counter[str] = Counter()
    total_bytes = 0
    seen: set[str] = set()
    previous_id: str | None = None

    for index, row in enumerate(records):
        if not isinstance(row, Mapping) or set(row) != _EXECUTION_ROW_KEYS:
            raise PrivacyExecutionAuthorityError(
                f"privacy execution record {index} schema drift"
            )
        record_id = row["record_id"]
        mode = row["mode"]
        payload_sha256 = row["payload_sha256"]
        nbytes = row["utf8_bytes"]
        action = row["action"]
        if not isinstance(record_id, str) or not record_id:
            raise PrivacyExecutionAuthorityError(
                f"privacy execution record {index} id malformed"
            )
        if record_id in seen:
            raise PrivacyExecutionAuthorityError(
                f"duplicate privacy execution record id: {record_id}"
            )
        if previous_id is not None and record_id <= previous_id:
            raise PrivacyExecutionAuthorityError(
                "privacy execution records must be strictly record-id sorted"
            )
        if not isinstance(mode, str) or mode not in _MODES:
            raise PrivacyExecutionAuthorityError(
                f"privacy execution record {index} mode drift"
            )
        _require_sha256(
            payload_sha256, f"privacy execution record {index} payload_sha256"
        )
        nbytes = _require_exact_int(
            nbytes, f"privacy execution record {index} utf8_bytes"
        )
        if not isinstance(action, str) or action not in _ACTIONS:
            raise PrivacyExecutionAuthorityError(
                f"privacy execution record {index} action drift"
            )
        detector_counts = _validate_detector_counts(
            row["detector_counts"],
            f"privacy execution record {index} detector_counts",
        )
        result_core = {
            "input_sha256": payload_sha256,
            "input_bytes": nbytes,
            "action": action,
            "detector_counts": detector_counts,
        }
        result_hash = _require_sha256(
            row["privacy_result_sha256"],
            f"privacy execution record {index} privacy_result_sha256",
        )
        if result_hash != _sha256(_cjson(result_core)):
            raise PrivacyExecutionAuthorityError(
                f"privacy execution record {index} result self-hash mismatch"
            )
        normalized.append(
            {
                "record_id": record_id,
                "mode": mode,
                "payload_sha256": payload_sha256,
                "utf8_bytes": nbytes,
                "action": action,
                "detector_counts": detector_counts,
                "privacy_result_sha256": result_hash,
            }
        )
        seen.add(record_id)
        previous_id = record_id
        actions[action] += 1
        detectors.update(detector_counts)
        total_bytes += nbytes

    return normalized, actions, detectors, total_bytes


def verify_privacy_execution_root(
    authority: Mapping[str, Any],
    *,
    expected_input_rows_sha256: str,
    expected_execution_identity_sha256: str,
) -> str:
    """Verify a text-free G06 root against independent expected identities."""
    expected_input = _require_sha256(
        expected_input_rows_sha256, "expected_input_rows_sha256"
    )
    expected_identity = _require_sha256(
        expected_execution_identity_sha256,
        "expected_execution_identity_sha256",
    )
    required = {
        "schema_version",
        "authority_class",
        "expected_input_rows_sha256",
        "observed_input_rows_sha256",
        "privacy_binding",
        "execution_rows_sha256",
        "counts",
        "total_input_utf8_bytes",
        "detector_counts",
        "records",
        "truth_boundary",
        "execution_identity_sha256",
    }
    if not isinstance(authority, Mapping) or set(authority) != required:
        raise PrivacyExecutionAuthorityError(
            "privacy execution authority schema is not closed"
        )
    if authority["schema_version"] != PRIVACY_EXECUTION_SCHEMA:
        raise PrivacyExecutionAuthorityError("privacy execution schema drift")
    if authority["authority_class"] != PRIVACY_EXECUTION_AUTHORITY_CLASS:
        raise PrivacyExecutionAuthorityError(
            "privacy execution authority class drift"
        )
    if authority["expected_input_rows_sha256"] != expected_input:
        raise PrivacyExecutionAuthorityError(
            "expected input-row authority mismatch"
        )
    if authority["observed_input_rows_sha256"] != expected_input:
        raise PrivacyExecutionAuthorityError(
            "observed input-row authority mismatch"
        )

    _, current_binding = _privacy_binding()
    if authority["privacy_binding"] != current_binding:
        raise PrivacyExecutionAuthorityError(
            "privacy implementation/policy binding drift"
        )
    if authority["truth_boundary"] != _TRUTH_BOUNDARY:
        raise PrivacyExecutionAuthorityError(
            "privacy execution truth boundary drift"
        )

    rows, actions, detectors, total_bytes = _validate_execution_rows(
        authority["records"]
    )
    input_rows = [
        {
            "record_id": row["record_id"],
            "mode": row["mode"],
            "payload_sha256": row["payload_sha256"],
            "utf8_bytes": row["utf8_bytes"],
        }
        for row in rows
    ]
    if _sha256(_cjson(input_rows)) != expected_input:
        raise PrivacyExecutionAuthorityError(
            "privacy execution records do not match expected input-row root"
        )
    execution_rows_sha256 = _require_sha256(
        authority["execution_rows_sha256"], "execution_rows_sha256"
    )
    if execution_rows_sha256 != _sha256(_cjson(rows)):
        raise PrivacyExecutionAuthorityError(
            "privacy execution rows hash mismatch"
        )

    expected_counts = {
        "records": len(rows),
        "allow": actions["ALLOW"],
        "redact": actions["REDACT"],
        "quarantine": actions["QUARANTINE"],
        "exclude": actions["EXCLUDE"],
    }
    if authority["counts"] != expected_counts:
        raise PrivacyExecutionAuthorityError(
            "privacy execution action/count accounting drift"
        )
    if authority["detector_counts"] != dict(sorted(detectors.items())):
        raise PrivacyExecutionAuthorityError(
            "privacy execution detector accounting drift"
        )
    if _require_exact_int(
        authority["total_input_utf8_bytes"], "total_input_utf8_bytes"
    ) != total_bytes:
        raise PrivacyExecutionAuthorityError(
            "privacy execution byte accounting drift"
        )

    observed = _require_sha256(
        authority["execution_identity_sha256"],
        "execution_identity_sha256",
    )
    core = dict(authority)
    del core["execution_identity_sha256"]
    if observed != _sha256(_cjson(core)):
        raise PrivacyExecutionAuthorityError(
            "privacy execution authority self-hash mismatch"
        )
    if observed != expected_identity:
        raise PrivacyExecutionAuthorityError(
            "privacy execution authority root mismatch"
        )
    return observed


def verify_privacy_execution_authority(
    authority: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    expected_input_rows_sha256: str,
    expected_execution_identity_sha256: str | None = None,
) -> str:
    """Re-execute canonical privacy semantics and require byte-exact equality."""
    expected = build_privacy_execution_authority(
        records,
        expected_input_rows_sha256=expected_input_rows_sha256,
    )
    if _cjson(authority) != _cjson(expected):
        raise PrivacyExecutionAuthorityError(
            "privacy execution authority does not match canonical re-execution"
        )
    identity = expected["execution_identity_sha256"]
    if expected_execution_identity_sha256 is not None:
        expected_identity = _require_sha256(
            expected_execution_identity_sha256,
            "expected_execution_identity_sha256",
        )
        if identity != expected_identity:
            raise PrivacyExecutionAuthorityError(
                "privacy execution authority root mismatch"
            )
    return identity
