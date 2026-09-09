"""Fail-closed record-to-family-vector bridge for the learned-20M data spine.

This module deliberately does not implement balance allocation.  It converts an exact
post-decontamination record graph into a text-free family/stratum capacity authority
that the already-canonical NEXT100-106 balance gate can consume.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "12-6.d03-postdecontam-family-vector.v1"
BINDING_SCHEMA = "12-6.d03-final-record-decontamination-binding.v1"
FAMILY_MAP_SCHEMA = "12-6.d03-family-stratum-map.v1"
ALLOWED_VERDICTS = frozenset({"PASS_CLEAN", "PASS_WITH_EXCLUSIONS"})
ALLOWED_STRATA = frozenset({"uk", "en", "code"})
_HEX = frozenset("0123456789abcdef")


class ProjectionError(ValueError):
    """Raised when a post-decontamination projection is not scientifically bound."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_hex_digest(value: Any, field: str, length: int) -> str:
    if (
        not isinstance(value, str)
        or len(value) != length
        or any(ch not in _HEX for ch in value)
    ):
        raise ProjectionError(f"{field} must be a lowercase {length * 4}-bit hex digest")
    return value


def _require_sha256(value: Any, field: str) -> str:
    return _require_hex_digest(value, field, 64)


def _require_git_sha(value: Any, field: str) -> str:
    return _require_hex_digest(value, field, 40)


def _require_nonempty_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProjectionError(f"{field} must be a non-empty string")
    return value


def _require_bool_false(value: Any, field: str) -> None:
    if value is not False:
        raise ProjectionError(f"{field} must be false")


def _require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ProjectionError(f"{field} must be a non-negative integer")
    return value


def _self_hash(document: Mapping[str, Any], identity_field: str) -> str:
    clean = dict(document)
    clean.pop(identity_field, None)
    return _sha256_bytes(_canonical_bytes(clean))


def _verify_self_hash(document: Mapping[str, Any], identity_field: str) -> str:
    claimed = _require_sha256(document.get(identity_field), identity_field)
    actual = _self_hash(document, identity_field)
    if claimed != actual:
        raise ProjectionError(f"{identity_field} self-hash mismatch")
    return claimed


@dataclass(frozen=True)
class Record:
    record_id: str
    source_id: str
    family: str
    modality: str
    payload_sha256: str
    payload_bytes: int


def read_records_jsonl(path: Path) -> tuple[list[Record], str]:
    raw = path.read_bytes()
    file_sha256 = _sha256_bytes(raw)
    records: list[Record] = []
    seen_record_ids: set[str] = set()

    for line_number, raw_line in enumerate(raw.splitlines(), start=1):
        if not raw_line.strip():
            raise ProjectionError(f"blank line in records JSONL at line {line_number}")
        try:
            row = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProjectionError(f"invalid UTF-8/JSON at line {line_number}") from exc
        if not isinstance(row, dict):
            raise ProjectionError(f"record at line {line_number} must be an object")

        record_id = _require_nonempty_string(row.get("record_id"), "record_id")
        source_id = _require_nonempty_string(row.get("source_id"), "source_id")
        family = _require_nonempty_string(row.get("family"), "family")
        modality = _require_nonempty_string(row.get("modality"), "modality")
        payload = row.get("normalized_payload")
        if not isinstance(payload, str) or not payload:
            raise ProjectionError("normalized_payload must be a non-empty string")
        if record_id in seen_record_ids:
            raise ProjectionError(f"duplicate record_id: {record_id}")
        seen_record_ids.add(record_id)

        payload_bytes = len(payload.encode("utf-8"))
        payload_sha256 = _sha256_bytes(payload.encode("utf-8"))

        declared_bytes = row.get("normalized_payload_bytes")
        if declared_bytes is not None and declared_bytes != payload_bytes:
            raise ProjectionError(f"normalized_payload_bytes drift for {record_id}")
        declared_sha = row.get("normalized_payload_sha256")
        if declared_sha is not None:
            _require_sha256(declared_sha, "normalized_payload_sha256")
            if declared_sha != payload_sha256:
                raise ProjectionError(f"normalized_payload_sha256 drift for {record_id}")

        if row.get("training_eligible") not in (None, False):
            raise ProjectionError("input record cannot already claim training eligibility")
        if row.get("evaluation_eligible") not in (None, False):
            raise ProjectionError("input record cannot claim evaluation eligibility")

        records.append(
            Record(
                record_id=record_id,
                source_id=source_id,
                family=family,
                modality=modality,
                payload_sha256=payload_sha256,
                payload_bytes=payload_bytes,
            )
        )

    if not records:
        raise ProjectionError("records JSONL must be non-empty")
    return records, file_sha256


def verify_family_map(document: Mapping[str, Any]) -> tuple[dict[str, str], str]:
    if document.get("schema") != FAMILY_MAP_SCHEMA:
        raise ProjectionError("unsupported family map schema")
    identity = _verify_self_hash(document, "family_map_identity_sha256")
    _require_bool_false(
        document.get("training_authorized_by_this_mapping"),
        "training_authorized_by_this_mapping",
    )

    entries = document.get("families")
    if not isinstance(entries, list) or not entries:
        raise ProjectionError("family map must contain families")
    mapping: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ProjectionError("family map entries must be objects")
        family = _require_nonempty_string(entry.get("family"), "family")
        stratum = entry.get("stratum")
        if stratum not in ALLOWED_STRATA:
            raise ProjectionError(f"unsupported stratum for family {family}")
        if family in mapping:
            raise ProjectionError(f"duplicate family mapping: {family}")
        mapping[family] = stratum
    return mapping, identity


def verify_decontamination_binding(
    document: Mapping[str, Any],
    *,
    expected_records_sha256: str,
    expected_record_count: int,
    expected_input_payload_bytes: int,
) -> tuple[set[str], str, str]:
    if document.get("schema") != BINDING_SCHEMA:
        raise ProjectionError("unsupported decontamination binding schema")

    authority = _verify_self_hash(document, "decontamination_authority_sha256")
    retained_inventory = _require_sha256(
        document.get("retained_inventory_identity_sha256"),
        "retained_inventory_identity_sha256",
    )
    if document.get("verdict") not in ALLOWED_VERDICTS:
        raise ProjectionError("decontamination verdict is not terminal PASS")
    if document.get("records_jsonl_sha256") != expected_records_sha256:
        raise ProjectionError("decontamination binding references different records JSONL")
    if document.get("input_record_count") != expected_record_count:
        raise ProjectionError("decontamination input record count mismatch")
    if document.get("input_payload_bytes") != expected_input_payload_bytes:
        raise ProjectionError("decontamination input payload bytes mismatch")

    _require_bool_false(
        document.get("final_test_outcomes_read"),
        "final_test_outcomes_read",
    )
    _require_bool_false(
        document.get("model_selection_performed"),
        "model_selection_performed",
    )
    _require_bool_false(
        document.get("training_authorized_by_this_report"),
        "training_authorized_by_this_report",
    )
    if document.get("authorized_optimized_target_exposure") != 0:
        raise ProjectionError("decontamination binding must keep optimized-target exposure at zero")

    excluded = document.get("excluded_record_id_sha256")
    if not isinstance(excluded, list):
        raise ProjectionError("excluded_record_id_sha256 must be a list")
    excluded_set: set[str] = set()
    for index, value in enumerate(excluded):
        value = _require_sha256(value, f"excluded_record_id_sha256[{index}]")
        if value in excluded_set:
            raise ProjectionError("duplicate excluded record hash")
        excluded_set.add(value)

    expected_survivors = expected_record_count - len(excluded_set)
    if document.get("survivor_record_count") != expected_survivors:
        raise ProjectionError("survivor record count mismatch")

    return excluded_set, authority, retained_inventory


def _record_id_hash(record_id: str) -> str:
    return _sha256_bytes(record_id.encode("utf-8"))


def build_family_vector(
    *,
    records: Sequence[Record],
    records_jsonl_sha256: str,
    decontamination_binding: Mapping[str, Any],
    family_map: Mapping[str, Any],
    source_git_sha: str,
) -> dict[str, Any]:
    source_git_sha = _require_git_sha(source_git_sha, "source_git_sha")
    family_to_stratum, family_map_identity = verify_family_map(family_map)

    input_payload_bytes = sum(record.payload_bytes for record in records)
    excluded_hashes, decontam_authority, retained_inventory = verify_decontamination_binding(
        decontamination_binding,
        expected_records_sha256=records_jsonl_sha256,
        expected_record_count=len(records),
        expected_input_payload_bytes=input_payload_bytes,
    )

    known_record_hashes = {_record_id_hash(record.record_id) for record in records}
    unknown_exclusions = excluded_hashes - known_record_hashes
    if unknown_exclusions:
        raise ProjectionError("decontamination binding excludes unknown records")

    family_bytes: dict[tuple[str, str], int] = defaultdict(int)
    family_records: dict[tuple[str, str], int] = defaultdict(int)
    survivor_id_hashes: list[str] = []
    survivor_payload_hashes: list[str] = []
    survivor_payload_bytes = 0

    for record in records:
        record_hash = _record_id_hash(record.record_id)
        if record_hash in excluded_hashes:
            continue
        try:
            stratum = family_to_stratum[record.family]
        except KeyError as exc:
            raise ProjectionError(f"unmapped surviving family: {record.family}") from exc
        key = (stratum, record.family)
        family_bytes[key] += record.payload_bytes
        family_records[key] += 1
        survivor_id_hashes.append(record_hash)
        survivor_payload_hashes.append(record.payload_sha256)
        survivor_payload_bytes += record.payload_bytes

    declared_survivor_bytes = decontamination_binding.get("survivor_payload_bytes")
    if declared_survivor_bytes != survivor_payload_bytes:
        raise ProjectionError("survivor payload bytes mismatch")

    family_rows = [
        {
            "stratum": stratum,
            "family": family,
            "record_count": family_records[(stratum, family)],
            "capacity_bytes": capacity_bytes,
        }
        for stratum, family in sorted(family_bytes)
        for capacity_bytes in [family_bytes[(stratum, family)]]
    ]
    if not family_rows:
        raise ProjectionError("decontamination left no surviving records")

    stratum_bytes = {
        stratum: sum(
            row["capacity_bytes"] for row in family_rows if row["stratum"] == stratum
        )
        for stratum in sorted(ALLOWED_STRATA)
    }
    stratum_family_counts = {
        stratum: sum(1 for row in family_rows if row["stratum"] == stratum)
        for stratum in sorted(ALLOWED_STRATA)
    }

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS",
        "source_git_sha": source_git_sha,
        "records_jsonl_sha256": records_jsonl_sha256,
        "retained_inventory_identity_sha256": retained_inventory,
        "decontamination_authority_sha256": decontam_authority,
        "family_map_identity_sha256": family_map_identity,
        "input_record_count": len(records),
        "input_payload_bytes": input_payload_bytes,
        "excluded_record_count": len(excluded_hashes),
        "survivor_record_count": len(survivor_id_hashes),
        "survivor_payload_bytes": survivor_payload_bytes,
        "survivor_record_id_membership_sha256": _sha256_bytes(
            _canonical_bytes(sorted(survivor_id_hashes))
        ),
        "survivor_payload_membership_sha256": _sha256_bytes(
            _canonical_bytes(sorted(survivor_payload_hashes))
        ),
        "families": family_rows,
        "stratum_capacity_bytes": stratum_bytes,
        "stratum_family_counts": stratum_family_counts,
        "next_gate": "NEXT100-106_BALANCE_FAMILY_CAP",
        "training_eligible": False,
        "evaluation_eligible": False,
        "tokenizer_fit_authorized": False,
        "model_training_authorized": False,
        "authorized_optimized_target_exposure": 0,
        "training_authorized_by_this_report": False,
    }
    result["family_vector_identity_sha256"] = _self_hash(
        result, "family_vector_identity_sha256"
    )
    return result


def verify_family_vector(document: Mapping[str, Any]) -> str:
    if document.get("schema") != SCHEMA:
        raise ProjectionError("unsupported family vector schema")
    identity = _verify_self_hash(document, "family_vector_identity_sha256")
    if document.get("status") != "PASS":
        raise ProjectionError("family vector is not PASS")
    if document.get("next_gate") != "NEXT100-106_BALANCE_FAMILY_CAP":
        raise ProjectionError("family vector does not hand off to NEXT100-106")
    _require_bool_false(document.get("training_eligible"), "training_eligible")
    _require_bool_false(document.get("evaluation_eligible"), "evaluation_eligible")
    _require_bool_false(
        document.get("tokenizer_fit_authorized"),
        "tokenizer_fit_authorized",
    )
    _require_bool_false(
        document.get("model_training_authorized"),
        "model_training_authorized",
    )
    _require_bool_false(
        document.get("training_authorized_by_this_report"),
        "training_authorized_by_this_report",
    )
    if document.get("authorized_optimized_target_exposure") != 0:
        raise ProjectionError("family vector must keep optimized-target exposure at zero")

    families = document.get("families")
    if not isinstance(families, list) or not families:
        raise ProjectionError("family vector must contain families")
    seen: set[tuple[str, str]] = set()
    total = 0
    by_stratum: dict[str, int] = defaultdict(int)
    count_by_stratum: dict[str, int] = defaultdict(int)
    for row in families:
        if not isinstance(row, dict):
            raise ProjectionError("family rows must be objects")
        stratum = row.get("stratum")
        if stratum not in ALLOWED_STRATA:
            raise ProjectionError("invalid family stratum")
        family = _require_nonempty_string(row.get("family"), "family")
        key = (stratum, family)
        if key in seen:
            raise ProjectionError("duplicate family row")
        seen.add(key)
        records = _require_nonnegative_int(row.get("record_count"), "record_count")
        capacity = _require_nonnegative_int(row.get("capacity_bytes"), "capacity_bytes")
        if records == 0 or capacity == 0:
            raise ProjectionError("surviving family rows must be positive")
        total += capacity
        by_stratum[stratum] += capacity
        count_by_stratum[stratum] += 1

    if total != document.get("survivor_payload_bytes"):
        raise ProjectionError("family vector total bytes mismatch")
    family_record_total = sum(row["record_count"] for row in families)
    if family_record_total != document.get("survivor_record_count"):
        raise ProjectionError("family vector record-count mismatch")
    input_record_count = _require_nonnegative_int(
        document.get("input_record_count"), "input_record_count"
    )
    excluded_record_count = _require_nonnegative_int(
        document.get("excluded_record_count"), "excluded_record_count"
    )
    survivor_record_count = _require_nonnegative_int(
        document.get("survivor_record_count"), "survivor_record_count"
    )
    if excluded_record_count + survivor_record_count != input_record_count:
        raise ProjectionError("input/excluded/survivor record arithmetic mismatch")
    input_payload_bytes = _require_nonnegative_int(
        document.get("input_payload_bytes"), "input_payload_bytes"
    )
    survivor_payload_bytes = _require_nonnegative_int(
        document.get("survivor_payload_bytes"), "survivor_payload_bytes"
    )
    if survivor_payload_bytes > input_payload_bytes:
        raise ProjectionError("survivor payload bytes exceed input payload bytes")
    expected_strata = {
        stratum: by_stratum.get(stratum, 0) for stratum in sorted(ALLOWED_STRATA)
    }
    if document.get("stratum_capacity_bytes") != expected_strata:
        raise ProjectionError("stratum capacity arithmetic mismatch")
    expected_counts = {
        stratum: count_by_stratum.get(stratum, 0) for stratum in sorted(ALLOWED_STRATA)
    }
    if document.get("stratum_family_counts") != expected_counts:
        raise ProjectionError("stratum family-count arithmetic mismatch")
    return identity


def write_family_vector(path: Path, result: Mapping[str, Any]) -> None:
    verify_family_vector(result)
    path.write_bytes(_canonical_bytes(result) + b"\n")


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectionError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ProjectionError(f"JSON document must be an object: {path}")
    return value
