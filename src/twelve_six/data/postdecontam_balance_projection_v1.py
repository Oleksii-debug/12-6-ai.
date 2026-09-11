"""Fail-closed record-to-family-vector bridge for the learned-20M data spine.

This module deliberately does not implement balance allocation. It converts an exact
post-decontamination record graph into a text-free family/stratum capacity authority
that the already-canonical NEXT100-106 balance gate can consume.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "12-6.d03-postdecontam-family-vector.v1"
BINDING_SCHEMA = "12-6.d03-final-record-decontamination-binding.v1"
FAMILY_MAP_SCHEMA = "12-6.d03-family-stratum-map.v1"
FAMILY_PROVENANCE_SCHEMA = "12-6.d03-family-provenance-authority.v1"
G05_G06_COVERAGE_SCHEMA = "12-6.d03-final-g05-g06-coverage.v1"
QUALITY_POLICY_IDENTITY_SHA256 = (
    "97b9fe1452b22c6275a27f85524f670253a7f4012377361c4cb007004aeccd1d"
)
QUALITY_GRANULARITY_IDENTITY_SHA256 = (
    "e8685c2c6b265b9b289ded7a5245888d8d16ae4d6e881f6229f3bc777601f857"
)
ALLOWED_VERDICTS = frozenset({"PASS_CLEAN", "PASS_WITH_EXCLUSIONS"})
ALLOWED_STRATA = frozenset({"uk", "en", "code"})
_HEX = frozenset("0123456789abcdef")


class ProjectionError(ValueError):
    """Raised when a post-decontamination projection is not scientifically bound."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


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


def _verify_expected_identity(
    document: Mapping[str, Any],
    *,
    identity_field: str,
    expected_identity_sha256: str,
    authority_name: str,
) -> str:
    expected = _require_sha256(
        expected_identity_sha256,
        f"expected_{identity_field}",
    )
    actual = _verify_self_hash(document, identity_field)
    if actual != expected:
        raise ProjectionError(
            f"{authority_name} does not match independently expected identity"
        )
    return actual


@dataclass(frozen=True)
class Record:
    record_id: str
    source_id: str
    family: str
    modality: str
    payload_sha256: str
    payload_bytes: int


@dataclass(frozen=True)
class FamilyProvenance:
    family: str
    source_family_identity_sha256: str
    language: str
    modalities: frozenset[str]
    stratum: str


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
        if declared_bytes is not None:
            declared_bytes = _require_nonnegative_int(
                declared_bytes,
                "normalized_payload_bytes",
            )
            if declared_bytes != payload_bytes:
                raise ProjectionError(f"normalized_payload_bytes drift for {record_id}")
        declared_sha = row.get("normalized_payload_sha256")
        if declared_sha is not None:
            _require_sha256(declared_sha, "normalized_payload_sha256")
            if declared_sha != payload_sha256:
                raise ProjectionError(f"normalized_payload_sha256 drift for {record_id}")

        if "training_eligible" in row and row["training_eligible"] is not False:
            raise ProjectionError("input record cannot already claim training eligibility")
        if "evaluation_eligible" in row and row["evaluation_eligible"] is not False:
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


def verify_family_map(
    document: Mapping[str, Any],
    *,
    expected_identity_sha256: str,
    expected_families: set[str],
) -> tuple[dict[str, str], str]:
    if document.get("schema") != FAMILY_MAP_SCHEMA:
        raise ProjectionError("unsupported family map schema")
    identity = _verify_expected_identity(
        document,
        identity_field="family_map_identity_sha256",
        expected_identity_sha256=expected_identity_sha256,
        authority_name="family map",
    )
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
        if set(entry) != {"family", "stratum"}:
            raise ProjectionError("family map entries must contain only family and stratum")
        family = _require_nonempty_string(entry.get("family"), "family")
        stratum = entry.get("stratum")
        if stratum not in ALLOWED_STRATA:
            raise ProjectionError(f"unsupported stratum for family {family}")
        if family in mapping:
            raise ProjectionError(f"duplicate family mapping: {family}")
        mapping[family] = stratum
    if set(mapping) != expected_families:
        raise ProjectionError("family map must exactly cover surviving families")
    return mapping, identity


def verify_family_provenance(
    document: Mapping[str, Any],
    *,
    expected_identity_sha256: str,
    expected_families: set[str],
    survivor_records: Sequence[Record],
) -> tuple[dict[str, FamilyProvenance], str]:
    if document.get("schema") != FAMILY_PROVENANCE_SCHEMA:
        raise ProjectionError("unsupported family provenance schema")
    identity = _verify_expected_identity(
        document,
        identity_field="family_provenance_identity_sha256",
        expected_identity_sha256=expected_identity_sha256,
        authority_name="family provenance",
    )
    _require_bool_false(
        document.get("training_authorized_by_this_provenance"),
        "training_authorized_by_this_provenance",
    )

    entries = document.get("families")
    if not isinstance(entries, list) or not entries:
        raise ProjectionError("family provenance must contain families")
    provenance: dict[str, FamilyProvenance] = {}
    required_fields = {
        "family",
        "source_family_identity_sha256",
        "language",
        "modalities",
        "stratum",
    }
    for entry in entries:
        if not isinstance(entry, dict):
            raise ProjectionError("family provenance entries must be objects")
        if set(entry) != required_fields:
            raise ProjectionError("family provenance entry fields are not closed-world")
        family = _require_nonempty_string(entry.get("family"), "family")
        if family in provenance:
            raise ProjectionError(f"duplicate family provenance: {family}")
        source_family_identity = _require_sha256(
            entry.get("source_family_identity_sha256"),
            "source_family_identity_sha256",
        )
        language = _require_nonempty_string(entry.get("language"), "language")
        stratum = entry.get("stratum")
        if stratum not in ALLOWED_STRATA:
            raise ProjectionError(f"unsupported provenance stratum for family {family}")
        raw_modalities = entry.get("modalities")
        if not isinstance(raw_modalities, list) or not raw_modalities:
            raise ProjectionError("family provenance modalities must be a non-empty list")
        modalities: set[str] = set()
        for raw_modality in raw_modalities:
            modality = _require_nonempty_string(raw_modality, "modality")
            if modality in modalities:
                raise ProjectionError(f"duplicate modality in provenance for {family}")
            modalities.add(modality)
        provenance[family] = FamilyProvenance(
            family=family,
            source_family_identity_sha256=source_family_identity,
            language=language,
            modalities=frozenset(modalities),
            stratum=stratum,
        )

    if set(provenance) != expected_families:
        raise ProjectionError("family provenance must exactly cover surviving families")
    for record in survivor_records:
        authority = provenance[record.family]
        if record.modality not in authority.modalities:
            raise ProjectionError(
                f"record modality is not admitted by provenance for {record.family}"
            )
    return provenance, identity


def verify_decontamination_binding(
    document: Mapping[str, Any],
    *,
    expected_records_sha256: str,
    expected_record_count: int,
    expected_input_payload_bytes: int,
) -> tuple[set[str], str, str, str]:
    if document.get("schema") != BINDING_SCHEMA:
        raise ProjectionError("unsupported decontamination binding schema")

    authority = _verify_self_hash(document, "decontamination_authority_sha256")
    retained_inventory = _require_sha256(
        document.get("retained_inventory_identity_sha256"),
        "retained_inventory_identity_sha256",
    )
    dedup_evidence = _require_sha256(
        document.get("dedup_evidence_identity_sha256"),
        "dedup_evidence_identity_sha256",
    )
    if document.get("verdict") not in ALLOWED_VERDICTS:
        raise ProjectionError("decontamination verdict is not terminal PASS")
    if document.get("records_jsonl_sha256") != expected_records_sha256:
        raise ProjectionError("decontamination binding references different records JSONL")
    input_record_count = _require_nonnegative_int(
        document.get("input_record_count"),
        "input_record_count",
    )
    if input_record_count != expected_record_count:
        raise ProjectionError("decontamination input record count mismatch")
    input_payload_bytes = _require_nonnegative_int(
        document.get("input_payload_bytes"),
        "input_payload_bytes",
    )
    if input_payload_bytes != expected_input_payload_bytes:
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
    optimized_target_exposure = _require_nonnegative_int(
        document.get("authorized_optimized_target_exposure"),
        "authorized_optimized_target_exposure",
    )
    if optimized_target_exposure != 0:
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
    survivor_record_count = _require_nonnegative_int(
        document.get("survivor_record_count"),
        "survivor_record_count",
    )
    if survivor_record_count != expected_survivors:
        raise ProjectionError("survivor record count mismatch")

    return excluded_set, authority, retained_inventory, dedup_evidence


def _record_id_hash(record_id: str) -> str:
    return _sha256_bytes(record_id.encode("utf-8"))


def _coverage_rows(records: Sequence[Record]) -> list[dict[str, Any]]:
    rows = [
        {
            "record_id_sha256": _record_id_hash(record.record_id),
            "payload_sha256": record.payload_sha256,
            "payload_bytes": record.payload_bytes,
        }
        for record in records
    ]
    return sorted(rows, key=lambda row: row["record_id_sha256"])


def verify_final_g05_g06_coverage(
    document: Mapping[str, Any],
    *,
    expected_identity_sha256: str,
    expected_privacy_policy_identity_sha256: str,
    expected_records_sha256: str,
    expected_retained_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    survivor_records: Sequence[Record],
) -> str:
    if document.get("schema") != G05_G06_COVERAGE_SCHEMA:
        raise ProjectionError("unsupported final G05/G06 coverage schema")
    identity = _verify_expected_identity(
        document,
        identity_field="g05_g06_coverage_identity_sha256",
        expected_identity_sha256=expected_identity_sha256,
        authority_name="final G05/G06 coverage",
    )
    if document.get("status") != "PASS":
        raise ProjectionError("final G05/G06 coverage is not terminal PASS")
    if document.get("records_jsonl_sha256") != expected_records_sha256:
        raise ProjectionError("G05/G06 coverage references different records JSONL")
    if (
        document.get("retained_inventory_identity_sha256")
        != expected_retained_inventory_identity_sha256
    ):
        raise ProjectionError("G05/G06 coverage retained-inventory lineage mismatch")
    if (
        document.get("decontamination_authority_sha256")
        != expected_decontamination_authority_sha256
    ):
        raise ProjectionError("G05/G06 coverage decontamination lineage mismatch")
    if document.get("quality_policy_identity_sha256") != QUALITY_POLICY_IDENTITY_SHA256:
        raise ProjectionError("G05/G06 coverage uses noncanonical quality policy")
    if (
        document.get("quality_granularity_identity_sha256")
        != QUALITY_GRANULARITY_IDENTITY_SHA256
    ):
        raise ProjectionError("G05/G06 coverage uses noncanonical quality granularity")
    expected_privacy = _require_sha256(
        expected_privacy_policy_identity_sha256,
        "expected_privacy_policy_identity_sha256",
    )
    if document.get("privacy_policy_identity_sha256") != expected_privacy:
        raise ProjectionError("G05/G06 coverage privacy policy identity mismatch")

    _require_bool_false(
        document.get("final_test_outcomes_read"),
        "final_test_outcomes_read",
    )
    _require_bool_false(
        document.get("training_authorized_by_this_coverage"),
        "training_authorized_by_this_coverage",
    )
    optimized_target_exposure = _require_nonnegative_int(
        document.get("authorized_optimized_target_exposure"),
        "authorized_optimized_target_exposure",
    )
    if optimized_target_exposure != 0:
        raise ProjectionError("G05/G06 coverage must keep optimized-target exposure at zero")

    expected_rows = _coverage_rows(survivor_records)
    covered_rows = document.get("covered_records")
    if not isinstance(covered_rows, list):
        raise ProjectionError("G05/G06 covered_records must be a list")
    required_fields = {"record_id_sha256", "payload_sha256", "payload_bytes"}
    normalized_rows: list[dict[str, Any]] = []
    seen_record_hashes: set[str] = set()
    for index, row in enumerate(covered_rows):
        if not isinstance(row, dict) or set(row) != required_fields:
            raise ProjectionError("G05/G06 coverage rows are not closed-world")
        record_hash = _require_sha256(
            row.get("record_id_sha256"),
            f"covered_records[{index}].record_id_sha256",
        )
        if record_hash in seen_record_hashes:
            raise ProjectionError("duplicate record in G05/G06 coverage")
        seen_record_hashes.add(record_hash)
        payload_sha = _require_sha256(
            row.get("payload_sha256"),
            f"covered_records[{index}].payload_sha256",
        )
        payload_bytes = _require_nonnegative_int(
            row.get("payload_bytes"),
            f"covered_records[{index}].payload_bytes",
        )
        normalized_rows.append(
            {
                "record_id_sha256": record_hash,
                "payload_sha256": payload_sha,
                "payload_bytes": payload_bytes,
            }
        )
    normalized_rows.sort(key=lambda row: row["record_id_sha256"])
    if normalized_rows != expected_rows:
        raise ProjectionError("G05/G06 coverage does not exactly cover final survivors")
    covered_record_count = _require_nonnegative_int(
        document.get("covered_record_count"),
        "covered_record_count",
    )
    if covered_record_count != len(expected_rows):
        raise ProjectionError("G05/G06 covered record count mismatch")
    expected_bytes = sum(record.payload_bytes for record in survivor_records)
    covered_payload_bytes = _require_nonnegative_int(
        document.get("covered_payload_bytes"),
        "covered_payload_bytes",
    )
    if covered_payload_bytes != expected_bytes:
        raise ProjectionError("G05/G06 covered payload bytes mismatch")
    return identity


def build_family_vector(
    *,
    records: Sequence[Record],
    records_jsonl_sha256: str,
    decontamination_binding: Mapping[str, Any],
    g05_g06_coverage: Mapping[str, Any],
    expected_g05_g06_coverage_identity_sha256: str,
    expected_privacy_policy_identity_sha256: str,
    family_map: Mapping[str, Any],
    expected_family_map_identity_sha256: str,
    family_provenance: Mapping[str, Any],
    expected_family_provenance_identity_sha256: str,
    source_git_sha: str,
) -> dict[str, Any]:
    source_git_sha = _require_git_sha(source_git_sha, "source_git_sha")
    _require_sha256(records_jsonl_sha256, "records_jsonl_sha256")

    input_payload_bytes = sum(record.payload_bytes for record in records)
    (
        excluded_hashes,
        decontam_authority,
        retained_inventory,
        dedup_evidence,
    ) = verify_decontamination_binding(
        decontamination_binding,
        expected_records_sha256=records_jsonl_sha256,
        expected_record_count=len(records),
        expected_input_payload_bytes=input_payload_bytes,
    )

    known_record_hashes = {_record_id_hash(record.record_id) for record in records}
    unknown_exclusions = excluded_hashes - known_record_hashes
    if unknown_exclusions:
        raise ProjectionError("decontamination binding excludes unknown records")

    survivor_records = [
        record for record in records if _record_id_hash(record.record_id) not in excluded_hashes
    ]
    if not survivor_records:
        raise ProjectionError("decontamination left no surviving records")
    survivor_payload_bytes = sum(record.payload_bytes for record in survivor_records)
    declared_survivor_bytes = _require_nonnegative_int(
        decontamination_binding.get("survivor_payload_bytes"),
        "survivor_payload_bytes",
    )
    if declared_survivor_bytes != survivor_payload_bytes:
        raise ProjectionError("survivor payload bytes mismatch")

    g05_g06_coverage_identity = verify_final_g05_g06_coverage(
        g05_g06_coverage,
        expected_identity_sha256=expected_g05_g06_coverage_identity_sha256,
        expected_privacy_policy_identity_sha256=expected_privacy_policy_identity_sha256,
        expected_records_sha256=records_jsonl_sha256,
        expected_retained_inventory_identity_sha256=retained_inventory,
        expected_decontamination_authority_sha256=decontam_authority,
        survivor_records=survivor_records,
    )

    survivor_families = {record.family for record in survivor_records}
    family_to_stratum, family_map_identity = verify_family_map(
        family_map,
        expected_identity_sha256=expected_family_map_identity_sha256,
        expected_families=survivor_families,
    )
    provenance, family_provenance_identity = verify_family_provenance(
        family_provenance,
        expected_identity_sha256=expected_family_provenance_identity_sha256,
        expected_families=survivor_families,
        survivor_records=survivor_records,
    )
    for family, stratum in family_to_stratum.items():
        if provenance[family].stratum != stratum:
            raise ProjectionError(f"family map reclassifies canonical provenance for {family}")

    family_bytes: dict[tuple[str, str], int] = defaultdict(int)
    family_records: dict[tuple[str, str], int] = defaultdict(int)
    survivor_id_hashes: list[str] = []
    survivor_payload_hashes: list[str] = []

    for record in survivor_records:
        stratum = family_to_stratum[record.family]
        key = (stratum, record.family)
        family_bytes[key] += record.payload_bytes
        family_records[key] += 1
        survivor_id_hashes.append(_record_id_hash(record.record_id))
        survivor_payload_hashes.append(record.payload_sha256)

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
        "dedup_evidence_identity_sha256": dedup_evidence,
        "g05_g06_coverage_identity_sha256": g05_g06_coverage_identity,
        "quality_policy_identity_sha256": QUALITY_POLICY_IDENTITY_SHA256,
        "quality_granularity_identity_sha256": QUALITY_GRANULARITY_IDENTITY_SHA256,
        "privacy_policy_identity_sha256": _require_sha256(
            expected_privacy_policy_identity_sha256,
            "expected_privacy_policy_identity_sha256",
        ),
        "family_map_identity_sha256": family_map_identity,
        "family_provenance_identity_sha256": family_provenance_identity,
        "input_record_count": len(records),
        "input_payload_bytes": input_payload_bytes,
        "excluded_record_count": len(excluded_hashes),
        "survivor_record_count": len(survivor_records),
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
        result,
        "family_vector_identity_sha256",
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
    optimized_target_exposure = _require_nonnegative_int(
        document.get("authorized_optimized_target_exposure"),
        "authorized_optimized_target_exposure",
    )
    if optimized_target_exposure != 0:
        raise ProjectionError("family vector must keep optimized-target exposure at zero")

    for field in (
        "records_jsonl_sha256",
        "retained_inventory_identity_sha256",
        "decontamination_authority_sha256",
        "dedup_evidence_identity_sha256",
        "g05_g06_coverage_identity_sha256",
        "privacy_policy_identity_sha256",
        "family_map_identity_sha256",
        "family_provenance_identity_sha256",
        "survivor_record_id_membership_sha256",
        "survivor_payload_membership_sha256",
    ):
        _require_sha256(document.get(field), field)
    _require_git_sha(document.get("source_git_sha"), "source_git_sha")
    if document.get("quality_policy_identity_sha256") != QUALITY_POLICY_IDENTITY_SHA256:
        raise ProjectionError("family vector quality policy identity mismatch")
    if (
        document.get("quality_granularity_identity_sha256")
        != QUALITY_GRANULARITY_IDENTITY_SHA256
    ):
        raise ProjectionError("family vector quality granularity identity mismatch")

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
        record_count = _require_nonnegative_int(row.get("record_count"), "record_count")
        capacity = _require_nonnegative_int(row.get("capacity_bytes"), "capacity_bytes")
        if record_count == 0 or capacity == 0:
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
        document.get("input_record_count"),
        "input_record_count",
    )
    excluded_record_count = _require_nonnegative_int(
        document.get("excluded_record_count"),
        "excluded_record_count",
    )
    survivor_record_count = _require_nonnegative_int(
        document.get("survivor_record_count"),
        "survivor_record_count",
    )
    if excluded_record_count + survivor_record_count != input_record_count:
        raise ProjectionError("input/excluded/survivor record arithmetic mismatch")
    input_payload_bytes = _require_nonnegative_int(
        document.get("input_payload_bytes"),
        "input_payload_bytes",
    )
    survivor_payload_bytes = _require_nonnegative_int(
        document.get("survivor_payload_bytes"),
        "survivor_payload_bytes",
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
