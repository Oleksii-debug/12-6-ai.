"""Bind an exact balanced zero-credit record set to canonical split mechanics.

This module is deliberately not a balance allocator and not a split algorithm. It
verifies an externally selected record authority, projects the exact selected raw
records into the already-canonical ``twelve_six.split_robustness`` API, and emits a
text-free split-application authority. The temporary ``SplitRecord``
``training_eligible=True`` value exists only to satisfy that canonical mechanics API;
it is never persisted as data/training authorization.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from twelve_six.split_robustness import (
    SplitFamilySpec,
    SplitRecord,
    build_split_family,
    verify_split_family_manifest,
)

SELECTION_SCHEMA = "12-6.d03-balanced-selection-authority.v1"
APPLICATION_SCHEMA = "12-6.d03-balanced-split-application.v1"
SPLIT_SPEC_SCHEMA = "12-6.d03-split-spec-authority.v1"
CANONICAL_SPLIT_GIT_BLOB_SHA1 = "5a5395748bed6b666391268b605e428af18baf0c"
CANONICAL_SPLIT_ALGORITHM = "cluster-hash-ranked-greedy-v1"
CANONICAL_SPLIT_VARIANT_SEEDS = ("split-a", "split-b", "split-c")
CANONICAL_SPLIT_VALIDATION_FRACTION = 0.2
CANONICAL_SPLIT_SPEC_IDENTITY_SHA256 = (
    "b0b745ab890343b705c7f02222b3541058513fdaaf5813940b7f8ac9c7bf63e7"
)
_ALLOWED_PURPOSES = frozenset({"pretraining", "pretraining_eligible", "training_eligible"})
_FORBIDDEN_PURPOSES = frozenset(
    {"benchmark", "evaluation", "evaluation_test", "heldout_test", "test", "probe_test"}
)
_ALLOWED_STRATA = frozenset({"uk", "ua", "en", "code"})
_HEX = frozenset("0123456789abcdef")
_SELECTION_FIELDS = frozenset(
    {
        "schema",
        "terminal",
        "status",
        "balanced_selection_identity_sha256",
        "retained_inventory_identity_sha256",
        "decontamination_authority_sha256",
        "dedup_authority_sha256",
        "balance_policy_identity_sha256",
        "balance_result_identity_sha256",
        "records",
        "totals",
        "claim_boundary",
    }
)
_SELECTION_ROW_FIELDS = frozenset(
    {
        "record_id",
        "source_id",
        "family",
        "stratum",
        "modality",
        "payload_sha256",
        "payload_bytes",
        "near_duplicate_cluster_id",
        "purpose",
        "training_eligible",
        "evaluation_eligible",
        "evaluation_reserved",
    }
)
_APPLICATION_FIELDS = frozenset(
    {
        "schema",
        "status",
        "balanced_selection_identity_sha256",
        "retained_inventory_identity_sha256",
        "decontamination_authority_sha256",
        "dedup_authority_sha256",
        "balance_policy_identity_sha256",
        "balance_result_identity_sha256",
        "canonical_split_git_blob_sha1",
        "split_spec_identity_sha256",
        "selected_record_count",
        "selected_source_bytes",
        "selected_family_source_bytes",
        "selected_stratum_source_bytes",
        "split_family",
        "claim_boundary",
        "application_identity_sha256",
    }
)


class BalancedSplitApplicationError(ValueError):
    """Raised when split application cannot be proven fail-closed."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical_split_spec_authority() -> dict[str, Any]:
    return {
        "schema": SPLIT_SPEC_SCHEMA,
        "canonical_split_git_blob_sha1": CANONICAL_SPLIT_GIT_BLOB_SHA1,
        "algorithm": CANONICAL_SPLIT_ALGORITHM,
        "variant_seeds": list(CANONICAL_SPLIT_VARIANT_SEEDS),
        "validation_fraction": CANONICAL_SPLIT_VALIDATION_FRACTION,
    }


def _require_canonical_split_spec(
    variant_seeds: Sequence[str], validation_fraction: float
) -> str:
    if isinstance(variant_seeds, (str, bytes)) or tuple(variant_seeds) != CANONICAL_SPLIT_VARIANT_SEEDS:
        raise BalancedSplitApplicationError(
            "variant_seeds do not match independently fixed canonical split spec"
        )
    if isinstance(validation_fraction, bool) or not isinstance(validation_fraction, (int, float)):
        raise BalancedSplitApplicationError(
            "validation_fraction does not match independently fixed canonical split spec"
        )
    if float(validation_fraction) != CANONICAL_SPLIT_VALIDATION_FRACTION:
        raise BalancedSplitApplicationError(
            "validation_fraction does not match independently fixed canonical split spec"
        )
    identity = _sha256_bytes(_canonical_bytes(_canonical_split_spec_authority()))
    if identity != CANONICAL_SPLIT_SPEC_IDENTITY_SHA256:
        raise BalancedSplitApplicationError("canonical split spec authority identity drift")
    return identity


def _require_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value == "UNRESOLVED":
        raise BalancedSplitApplicationError(f"{field} must be resolved non-empty text")
    return value.strip()


def _require_sha256(value: Any, field: str) -> str:
    text = _require_text(value, field)
    if len(text) != 64 or text != text.lower() or any(ch not in _HEX for ch in text):
        raise BalancedSplitApplicationError(f"{field} must be lowercase SHA-256 hex")
    return text


def _require_sha1(value: Any, field: str) -> str:
    text = _require_text(value, field)
    if len(text) != 40 or text != text.lower() or any(ch not in _HEX for ch in text):
        raise BalancedSplitApplicationError(f"{field} must be lowercase SHA-1 hex")
    return text


def _require_nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BalancedSplitApplicationError(f"{field} must be a non-negative integer")
    return value


def _require_nonnegative_int_map(value: Any, field: str) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise BalancedSplitApplicationError(f"{field} must be a mapping")
    validated: dict[str, int] = {}
    for key, count in value.items():
        _require_text(key, f"{field}.key")
        validated[key] = _require_nonnegative_int(count, f"{field}[{key!r}]")
    return validated


def _self_hash(document: Mapping[str, Any], identity_field: str) -> str:
    core = dict(document)
    core.pop(identity_field, None)
    return _sha256_bytes(_canonical_bytes(core))


def _verify_expected_identity(
    document: Mapping[str, Any], *, identity_field: str, expected: str
) -> str:
    expected = _require_sha256(expected, f"expected_{identity_field}")
    claimed = _require_sha256(document.get(identity_field), identity_field)
    if claimed != _self_hash(document, identity_field):
        raise BalancedSplitApplicationError(f"{identity_field} self-hash mismatch")
    if claimed != expected:
        raise BalancedSplitApplicationError(f"{identity_field} does not match expected identity")
    return claimed


def _require_zero_credit_boundary(boundary: Any) -> None:
    expected_boolean_fields = {
        "training_eligible",
        "evaluation_eligible",
        "tokenizer_fit_authorized",
        "model_training_authorized",
        "paid_compute_authorized",
        "final_test_outcomes_read",
    }
    expected_fields = expected_boolean_fields | {"authorized_optimized_target_exposure"}
    if not isinstance(boundary, Mapping) or set(boundary) != expected_fields:
        raise BalancedSplitApplicationError("balanced selection claim boundary is not closed-world")
    for field in expected_boolean_fields:
        value = boundary.get(field)
        if not isinstance(value, bool) or value is not False:
            raise BalancedSplitApplicationError(f"balanced selection claim boundary {field} widened")
    exposure = _require_nonnegative_int(
        boundary.get("authorized_optimized_target_exposure"),
        "claim_boundary.authorized_optimized_target_exposure",
    )
    if exposure != 0:
        raise BalancedSplitApplicationError("balanced selection claim boundary exposure widened")


def _record_id(row: Mapping[str, Any], index: int) -> str:
    return _require_text(row.get("record_id"), f"records[{index}].record_id")


def verify_balanced_selection(
    selection: Mapping[str, Any],
    *,
    expected_selection_identity_sha256: str,
    expected_retained_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    expected_dedup_authority_sha256: str,
    expected_balance_policy_identity_sha256: str,
    expected_balance_result_identity_sha256: str,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Verify closed-world terminal balanced-selection authority and accounting."""

    if set(selection) != _SELECTION_FIELDS:
        raise BalancedSplitApplicationError("balanced selection fields are not closed-world")
    if selection.get("schema") != SELECTION_SCHEMA:
        raise BalancedSplitApplicationError("unsupported balanced selection schema")
    if selection.get("terminal") is not True or selection.get("status") != "PASS":
        raise BalancedSplitApplicationError("balanced selection is not terminal PASS")

    _verify_expected_identity(
        selection,
        identity_field="balanced_selection_identity_sha256",
        expected=expected_selection_identity_sha256,
    )
    expected_pairs = (
        ("retained_inventory_identity_sha256", expected_retained_inventory_identity_sha256),
        ("decontamination_authority_sha256", expected_decontamination_authority_sha256),
        ("dedup_authority_sha256", expected_dedup_authority_sha256),
        ("balance_policy_identity_sha256", expected_balance_policy_identity_sha256),
        ("balance_result_identity_sha256", expected_balance_result_identity_sha256),
    )
    for field, expected in expected_pairs:
        if _require_sha256(selection.get(field), field) != _require_sha256(expected, f"expected_{field}"):
            raise BalancedSplitApplicationError(f"{field} does not match expected identity")
    _require_zero_credit_boundary(selection.get("claim_boundary"))

    rows = selection.get("records")
    if not isinstance(rows, list) or len(rows) < 4:
        raise BalancedSplitApplicationError("balanced selection requires at least four records")

    by_id: dict[str, dict[str, Any]] = {}
    family_bytes: defaultdict[str, int] = defaultdict(int)
    stratum_bytes: defaultdict[str, int] = defaultdict(int)
    source_bytes = 0
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping) or set(raw_row) != _SELECTION_ROW_FIELDS:
            raise BalancedSplitApplicationError("balanced selection record fields are not closed-world")
        row = dict(raw_row)
        record_id = _record_id(row, index)
        if record_id in by_id:
            raise BalancedSplitApplicationError(f"duplicate balanced record_id: {record_id}")
        for field in ("source_id", "family", "modality", "near_duplicate_cluster_id"):
            _require_text(row.get(field), f"records[{index}].{field}")
        stratum = _require_text(row.get("stratum"), f"records[{index}].stratum")
        if stratum not in _ALLOWED_STRATA:
            raise BalancedSplitApplicationError(f"unsupported stratum: {stratum}")
        purpose = _require_text(row.get("purpose"), f"records[{index}].purpose").lower()
        if purpose in _FORBIDDEN_PURPOSES or purpose not in _ALLOWED_PURPOSES:
            raise BalancedSplitApplicationError(f"forbidden/non-training purpose: {purpose}")
        _require_sha256(row.get("payload_sha256"), f"records[{index}].payload_sha256")
        payload_bytes = _require_nonnegative_int(
            row.get("payload_bytes"), f"records[{index}].payload_bytes"
        )
        if payload_bytes <= 0:
            raise BalancedSplitApplicationError("selected payload_bytes must be positive")
        if row.get("training_eligible") is not False:
            raise BalancedSplitApplicationError("input row must remain zero-credit training_eligible=false")
        if row.get("evaluation_eligible") is not False:
            raise BalancedSplitApplicationError("input row must remain evaluation_eligible=false")
        if row.get("evaluation_reserved") is not False:
            raise BalancedSplitApplicationError("evaluation-reserved rows cannot enter balanced split")
        row["purpose"] = purpose
        by_id[record_id] = row
        source_bytes += payload_bytes
        family_bytes[row["family"]] += payload_bytes
        stratum_bytes[stratum] += payload_bytes

    totals = selection.get("totals")
    if not isinstance(totals, Mapping) or set(totals) != {
        "record_count",
        "source_bytes",
        "family_source_bytes",
        "stratum_source_bytes",
    }:
        raise BalancedSplitApplicationError("balanced selection totals are not closed-world")
    record_count = _require_nonnegative_int(totals.get("record_count"), "totals.record_count")
    declared_source_bytes = _require_nonnegative_int(
        totals.get("source_bytes"), "totals.source_bytes"
    )
    declared_family_bytes = _require_nonnegative_int_map(
        totals.get("family_source_bytes"), "totals.family_source_bytes"
    )
    declared_stratum_bytes = _require_nonnegative_int_map(
        totals.get("stratum_source_bytes"), "totals.stratum_source_bytes"
    )
    if record_count != len(by_id) or declared_source_bytes != source_bytes:
        raise BalancedSplitApplicationError("balanced selection total count/bytes mismatch")
    if declared_family_bytes != dict(sorted(family_bytes.items())):
        raise BalancedSplitApplicationError("balanced selection family byte accounting mismatch")
    if declared_stratum_bytes != dict(sorted(stratum_bytes.items())):
        raise BalancedSplitApplicationError("balanced selection stratum byte accounting mismatch")
    return by_id, dict(totals)


def _project_records(
    selected_by_id: Mapping[str, Mapping[str, Any]], raw_records: Sequence[Mapping[str, Any]]
) -> list[SplitRecord]:
    seen: set[str] = set()
    projected: list[SplitRecord] = []
    for index, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise BalancedSplitApplicationError(f"raw_records[{index}] must be an object")
        record_id = _require_text(raw.get("record_id"), f"raw_records[{index}].record_id")
        if record_id in seen:
            raise BalancedSplitApplicationError(f"duplicate raw record_id: {record_id}")
        seen.add(record_id)
        selected = selected_by_id.get(record_id)
        if selected is None:
            raise BalancedSplitApplicationError(f"extra raw record outside selection: {record_id}")
        payload = raw.get("normalized_payload")
        if not isinstance(payload, str) or not payload:
            raise BalancedSplitApplicationError(f"{record_id}: normalized_payload must be non-empty")
        payload_bytes = len(payload.encode())
        payload_sha256 = _sha256_bytes(payload.encode())
        equality_fields = (
            "source_id",
            "family",
            "stratum",
            "modality",
            "near_duplicate_cluster_id",
        )
        for field in equality_fields:
            if raw.get(field) != selected[field]:
                raise BalancedSplitApplicationError(f"{record_id}: {field} drift")
        if raw.get("purpose") != selected["purpose"]:
            raise BalancedSplitApplicationError(f"{record_id}: purpose drift")
        if raw.get("training_eligible") is not False or raw.get("evaluation_eligible") is not False:
            raise BalancedSplitApplicationError(f"{record_id}: raw record widened eligibility")
        if raw.get("evaluation_reserved") is not False:
            raise BalancedSplitApplicationError(f"{record_id}: raw record is evaluation-reserved")
        if payload_bytes != selected["payload_bytes"] or payload_sha256 != selected["payload_sha256"]:
            raise BalancedSplitApplicationError(f"{record_id}: payload bytes/hash drift")

        # Canonical #938 uses this flag as a mechanics precondition. It is an in-memory
        # projection only; the returned authority below explicitly remains zero-credit.
        projected.append(
            SplitRecord(
                id=record_id,
                text=payload,
                source_id=selected["source_id"],
                modality=selected["modality"],
                content_sha256=payload_sha256,
                near_duplicate_cluster_id=selected["near_duplicate_cluster_id"],
                training_eligible=True,
                purpose=selected["purpose"],
            )
        )
    missing = set(selected_by_id) - seen
    if missing:
        raise BalancedSplitApplicationError(
            f"selected records missing from raw input: {sorted(missing)!r}"
        )
    return sorted(projected, key=lambda record: record.id)


def build_balanced_split_application(
    selection: Mapping[str, Any],
    raw_records: Sequence[Mapping[str, Any]],
    *,
    expected_selection_identity_sha256: str,
    expected_retained_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    expected_dedup_authority_sha256: str,
    expected_balance_policy_identity_sha256: str,
    expected_balance_result_identity_sha256: str,
    expected_split_git_blob_sha1: str,
    variant_seeds: Sequence[str],
    validation_fraction: float,
) -> dict[str, Any]:
    """Build a deterministic zero-credit split authority using canonical #938 only."""

    expected_split_blob = _require_sha1(expected_split_git_blob_sha1, "expected_split_git_blob_sha1")
    if expected_split_blob != CANONICAL_SPLIT_GIT_BLOB_SHA1:
        raise BalancedSplitApplicationError("canonical split mechanics blob does not match merged #938")
    split_spec_identity = _require_canonical_split_spec(variant_seeds, validation_fraction)
    selected_by_id, totals = verify_balanced_selection(
        selection,
        expected_selection_identity_sha256=expected_selection_identity_sha256,
        expected_retained_inventory_identity_sha256=expected_retained_inventory_identity_sha256,
        expected_decontamination_authority_sha256=expected_decontamination_authority_sha256,
        expected_dedup_authority_sha256=expected_dedup_authority_sha256,
        expected_balance_policy_identity_sha256=expected_balance_policy_identity_sha256,
        expected_balance_result_identity_sha256=expected_balance_result_identity_sha256,
    )
    projected = _project_records(selected_by_id, raw_records)

    from twelve_six.split_robustness import dedup_relations_identity, eligible_corpus_identity

    spec = SplitFamilySpec(
        eligible_corpus_sha256=eligible_corpus_identity(projected),
        dedup_relations_sha256=dedup_relations_identity(projected),
        variant_seeds=CANONICAL_SPLIT_VARIANT_SEEDS,
        validation_fraction=CANONICAL_SPLIT_VALIDATION_FRACTION,
        algorithm=CANONICAL_SPLIT_ALGORITHM,
    )
    split_family = build_split_family(projected, spec)
    verify_split_family_manifest(projected, split_family)

    core: dict[str, Any] = {
        "schema": APPLICATION_SCHEMA,
        "status": "PASS_ZERO_CREDIT",
        "balanced_selection_identity_sha256": selection["balanced_selection_identity_sha256"],
        "retained_inventory_identity_sha256": selection["retained_inventory_identity_sha256"],
        "decontamination_authority_sha256": selection["decontamination_authority_sha256"],
        "dedup_authority_sha256": selection["dedup_authority_sha256"],
        "balance_policy_identity_sha256": selection["balance_policy_identity_sha256"],
        "balance_result_identity_sha256": selection["balance_result_identity_sha256"],
        "canonical_split_git_blob_sha1": CANONICAL_SPLIT_GIT_BLOB_SHA1,
        "split_spec_identity_sha256": split_spec_identity,
        "selected_record_count": totals["record_count"],
        "selected_source_bytes": totals["source_bytes"],
        "selected_family_source_bytes": totals["family_source_bytes"],
        "selected_stratum_source_bytes": totals["stratum_source_bytes"],
        "split_family": split_family,
        "claim_boundary": {
            "training_eligible": False,
            "evaluation_eligible": False,
            "tokenizer_fit_authorized": False,
            "model_training_authorized": False,
            "paid_compute_authorized": False,
            "final_test_outcomes_read": False,
            "authorized_optimized_target_exposure": 0,
        },
    }
    return {**core, "application_identity_sha256": _self_hash(core, "application_identity_sha256")}


def verify_balanced_split_application(
    application: Mapping[str, Any],
    selection: Mapping[str, Any],
    raw_records: Sequence[Mapping[str, Any]],
    *,
    expected_selection_identity_sha256: str,
    expected_retained_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    expected_dedup_authority_sha256: str,
    expected_balance_policy_identity_sha256: str,
    expected_balance_result_identity_sha256: str,
    expected_split_git_blob_sha1: str,
    variant_seeds: Sequence[str],
    validation_fraction: float,
) -> None:
    """Independently rebuild and compare every semantic split-application field."""

    if set(application) != _APPLICATION_FIELDS:
        raise BalancedSplitApplicationError("split application fields are not closed-world")
    claimed = _require_sha256(application.get("application_identity_sha256"), "application_identity_sha256")
    if claimed != _self_hash(application, "application_identity_sha256"):
        raise BalancedSplitApplicationError("split application self-hash mismatch")
    split_spec_identity = _require_sha256(
        application.get("split_spec_identity_sha256"), "split_spec_identity_sha256"
    )
    if split_spec_identity != CANONICAL_SPLIT_SPEC_IDENTITY_SHA256:
        raise BalancedSplitApplicationError("split application does not bind canonical split spec authority")
    _require_zero_credit_boundary(application.get("claim_boundary"))
    expected = build_balanced_split_application(
        selection,
        raw_records,
        expected_selection_identity_sha256=expected_selection_identity_sha256,
        expected_retained_inventory_identity_sha256=expected_retained_inventory_identity_sha256,
        expected_decontamination_authority_sha256=expected_decontamination_authority_sha256,
        expected_dedup_authority_sha256=expected_dedup_authority_sha256,
        expected_balance_policy_identity_sha256=expected_balance_policy_identity_sha256,
        expected_balance_result_identity_sha256=expected_balance_result_identity_sha256,
        expected_split_git_blob_sha1=expected_split_git_blob_sha1,
        variant_seeds=variant_seeds,
        validation_fraction=validation_fraction,
    )
    if _canonical_bytes(dict(application)) != _canonical_bytes(expected):
        raise BalancedSplitApplicationError("split application semantic content mismatch")


def balanced_record_counts(selection: Mapping[str, Any]) -> Counter[str]:
    """Small text-free observability helper; not a selection or split policy."""

    rows = selection.get("records")
    if not isinstance(rows, list):
        return Counter()
    return Counter(
        row.get("stratum")
        for row in rows
        if isinstance(row, Mapping) and isinstance(row.get("stratum"), str)
    )
