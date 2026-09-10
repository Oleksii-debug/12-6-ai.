"""Bind complete G05/G06 evidence to the exact final D03 survivor graph.

This module is authority plumbing only. It does not implement quality thresholds,
privacy detectors, redaction, deduplication, decontamination, balancing, or split
selection. Trust comes from independently expected authority identities supplied by
the caller; self-hashes alone never confer scientific authority.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

COVERAGE_SCHEMA = "12-6.d03-final-g05-g06-coverage.v1"
INVENTORY_SCHEMA = "twelve-six.expanded-postdedup-inventory.v1"
DECONTAM_SCHEMA = "12-6.d03-final-record-decontamination-binding.v1"
QUALITY_AUTHORITY_SCHEMA = "12-6.d03-final-g05-quality-authority.v1"
PRIVACY_AUTHORITY_SCHEMA = "12-6.d03-final-g06-privacy-authority.v1"
QUALITY_THRESHOLD_POLICY_SHA256 = (
    "97b9fe1452b22c6275a27f85524f670253a7f4012377361c4cb007004aeccd1d"
)
QUALITY_GRANULARITY_POLICY_SHA256 = (
    "e8685c2c6b265b9b289ded7a5245888d8d16ae4d6e881f6229f3bc777601f857"
)
FINAL_SCOPE = "FINAL_SURVIVOR_SET"
PREDECONTAM_SCOPE = "PRE_DECONTAM_INVENTORY_SUBSET_PROOF"
ALLOWED_SCOPES = frozenset({FINAL_SCOPE, PREDECONTAM_SCOPE})
ALLOWED_DECONTAM_VERDICTS = frozenset({"PASS_CLEAN", "PASS_WITH_EXCLUSIONS"})
HEX64 = frozenset("0123456789abcdef")
HEX40 = frozenset("0123456789abcdef")

ZERO_TRUTH = {
    "training_eligible": False,
    "evaluation_eligible": False,
    "tokenizer_fit_authorized": False,
    "model_training_authorized": False,
    "final_test_outcomes_accessed": False,
    "model_selection_performed": False,
    "paid_compute_used": False,
    "authorized_optimized_target_exposure": 0,
}


class CoverageError(ValueError):
    """Raised when a G05/G06 coverage claim is not externally bound."""


def canonical_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes used by all identities here."""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    """Hash a JSON-compatible value with this module's canonical encoding."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def authority_identity(document: Mapping[str, Any], identity_field: str) -> str:
    """Compute an authority self-hash without treating it as trust."""
    payload = dict(document)
    payload.pop(identity_field, None)
    return sha256_json(payload)


def _require_sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(ch not in HEX64 for ch in value)
    ):
        raise CoverageError(f"{field} must be lowercase SHA-256 hex")
    return value


def _require_git_sha(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(ch not in HEX40 for ch in value)
    ):
        raise CoverageError(f"{field} must be lowercase 40-hex git SHA")
    return value


def _require_nonempty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CoverageError(f"{field} must be a non-empty string")
    return value


def _require_int(value: Any, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise CoverageError(f"{field} must be an integer >= {minimum}")
    return value


def _require_false(value: Any, field: str) -> None:
    if value is not False:
        raise CoverageError(f"{field} must be false")


def _verify_zero_truth(value: Any, field: str) -> None:
    if not isinstance(value, Mapping):
        raise CoverageError(f"{field} must be an object")
    for key, expected in ZERO_TRUTH.items():
        if value.get(key) != expected:
            raise CoverageError(f"{field}.{key} must remain {expected!r}")


def _verify_external_self_hash(
    document: Mapping[str, Any],
    *,
    identity_field: str,
    expected_identity: str,
    label: str,
) -> str:
    expected = _require_sha256(expected_identity, f"expected_{label}_identity_sha256")
    claimed = _require_sha256(document.get(identity_field), f"{label}.{identity_field}")
    observed = authority_identity(document, identity_field)
    if claimed != observed:
        raise CoverageError(f"{label} self-hash mismatch")
    if claimed != expected:
        raise CoverageError(f"{label} external identity mismatch")
    return claimed


def _record_id_sha256(record_id: str) -> str:
    return hashlib.sha256(record_id.encode("utf-8")).hexdigest()


def _inventory_records(
    inventory: Mapping[str, Any],
    *,
    expected_inventory_identity_sha256: str,
) -> tuple[dict[str, dict[str, Any]], str]:
    if inventory.get("schema") != INVENTORY_SCHEMA:
        raise CoverageError("unsupported retained inventory schema")
    identity = _verify_external_self_hash(
        inventory,
        identity_field="inventory_identity_sha256",
        expected_identity=expected_inventory_identity_sha256,
        label="inventory",
    )
    _verify_zero_truth(inventory.get("truth_boundary"), "inventory.truth_boundary")

    raw_rows = inventory.get("records")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise CoverageError("inventory.records must be a non-empty list")

    by_hash: dict[str, dict[str, Any]] = {}
    raw_ids: set[str] = set()
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            raise CoverageError(f"inventory.records[{index}] must be an object")
        record_id = _require_nonempty(
            row.get("record_id"),
            f"inventory.records[{index}].record_id",
        )
        if record_id in raw_ids:
            raise CoverageError("duplicate record_id in retained inventory")
        raw_ids.add(record_id)
        record_hash = _record_id_sha256(record_id)
        if record_hash in by_hash:
            raise CoverageError("record_id hash collision in retained inventory")
        payload_sha = _require_sha256(
            row.get("payload_sha256"),
            f"inventory.records[{index}].payload_sha256",
        )
        payload_bytes = _require_int(
            row.get("payload_bytes"),
            f"inventory.records[{index}].payload_bytes",
            minimum=1,
        )
        _require_false(
            row.get("training_eligible"),
            f"inventory.records[{index}].training_eligible",
        )
        _require_false(
            row.get("evaluation_eligible"),
            f"inventory.records[{index}].evaluation_eligible",
        )
        by_hash[record_hash] = {
            "record_id_sha256": record_hash,
            "payload_sha256": payload_sha,
            "payload_bytes": payload_bytes,
        }

    if inventory.get("record_count") != len(by_hash):
        raise CoverageError("inventory record_count mismatch")
    payload_sum = sum(row["payload_bytes"] for row in by_hash.values())
    if inventory.get("retained_payload_bytes") != payload_sum:
        raise CoverageError("inventory retained_payload_bytes mismatch")
    return by_hash, identity


def _decontam_survivors(
    decontam: Mapping[str, Any],
    inventory_rows: Mapping[str, Mapping[str, Any]],
    *,
    expected_decontamination_authority_sha256: str,
    inventory_identity_sha256: str,
) -> tuple[dict[str, dict[str, Any]], str]:
    if decontam.get("schema") != DECONTAM_SCHEMA:
        raise CoverageError("unsupported decontamination binding schema")
    identity = _verify_external_self_hash(
        decontam,
        identity_field="decontamination_authority_sha256",
        expected_identity=expected_decontamination_authority_sha256,
        label="decontamination",
    )
    if decontam.get("retained_inventory_identity_sha256") != inventory_identity_sha256:
        raise CoverageError("decontamination/inventory identity mismatch")
    if decontam.get("verdict") not in ALLOWED_DECONTAM_VERDICTS:
        raise CoverageError("decontamination verdict is not terminal PASS")
    _require_false(
        decontam.get("final_test_outcomes_read"),
        "decontamination.final_test_outcomes_read",
    )
    _require_false(
        decontam.get("model_selection_performed"),
        "decontamination.model_selection_performed",
    )
    _require_false(
        decontam.get("training_authorized_by_this_report"),
        "decontamination.training_authorized_by_this_report",
    )
    if decontam.get("authorized_optimized_target_exposure") != 0:
        raise CoverageError("decontamination widens optimized-target exposure")

    input_count = len(inventory_rows)
    input_bytes = sum(row["payload_bytes"] for row in inventory_rows.values())
    if decontam.get("input_record_count") != input_count:
        raise CoverageError("decontamination input_record_count mismatch")
    if decontam.get("input_payload_bytes") != input_bytes:
        raise CoverageError("decontamination input_payload_bytes mismatch")

    excluded_raw = decontam.get("excluded_record_id_sha256")
    if not isinstance(excluded_raw, list):
        raise CoverageError("excluded_record_id_sha256 must be a list")
    excluded: set[str] = set()
    for index, value in enumerate(excluded_raw):
        value = _require_sha256(value, f"excluded_record_id_sha256[{index}]")
        if value in excluded:
            raise CoverageError("duplicate decontamination exclusion")
        if value not in inventory_rows:
            raise CoverageError("decontamination excludes unknown record")
        excluded.add(value)

    survivors = {
        key: dict(value) for key, value in inventory_rows.items() if key not in excluded
    }
    if not survivors:
        raise CoverageError("decontamination left no final survivors")
    if decontam.get("survivor_record_count") != len(survivors):
        raise CoverageError("decontamination survivor_record_count mismatch")
    survivor_bytes = sum(row["payload_bytes"] for row in survivors.values())
    if decontam.get("survivor_payload_bytes") != survivor_bytes:
        raise CoverageError("decontamination survivor_payload_bytes mismatch")
    return survivors, identity


def _evidence_rows(
    authority: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    raw_rows = authority.get("records")
    if not isinstance(raw_rows, list) or not raw_rows:
        raise CoverageError(f"{label}.records must be a non-empty list")
    rows: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(raw_rows):
        if not isinstance(row, Mapping):
            raise CoverageError(f"{label}.records[{index}] must be an object")
        record_hash = _require_sha256(
            row.get("record_id_sha256"),
            f"{label}.records[{index}].record_id_sha256",
        )
        if record_hash in rows:
            raise CoverageError(f"duplicate record evidence in {label}")
        rows[record_hash] = {
            "record_id_sha256": record_hash,
            "payload_sha256": _require_sha256(
                row.get("payload_sha256"),
                f"{label}.records[{index}].payload_sha256",
            ),
            "payload_bytes": _require_int(
                row.get("payload_bytes"),
                f"{label}.records[{index}].payload_bytes",
                minimum=1,
            ),
            "decision": row.get("decision"),
        }
    return rows


def _verify_authority_coverage(
    authority: Mapping[str, Any],
    *,
    label: str,
    schema: str,
    identity_field: str,
    expected_identity: str,
    inventory_rows: Mapping[str, Mapping[str, Any]],
    survivor_rows: Mapping[str, Mapping[str, Any]],
    inventory_identity_sha256: str,
    decontamination_authority_sha256: str,
    accepted_decision: str,
) -> tuple[str, str, bool]:
    if authority.get("schema") != schema:
        raise CoverageError(f"unsupported {label} authority schema")
    identity = _verify_external_self_hash(
        authority,
        identity_field=identity_field,
        expected_identity=expected_identity,
        label=label,
    )
    _verify_zero_truth(authority.get("truth_boundary"), f"{label}.truth_boundary")
    if authority.get("retained_inventory_identity_sha256") != inventory_identity_sha256:
        raise CoverageError(f"{label} retained-inventory binding mismatch")

    scope = authority.get("coverage_scope")
    if scope not in ALLOWED_SCOPES:
        raise CoverageError(f"unsupported {label} coverage_scope")
    if scope == FINAL_SCOPE:
        if (
            authority.get("decontamination_authority_sha256")
            != decontamination_authority_sha256
        ):
            raise CoverageError(f"{label} decontamination binding mismatch")
        required = survivor_rows
        subset_proved = False
    else:
        required = inventory_rows
        subset_proved = True

    evidence = _evidence_rows(authority, label=label)
    if set(evidence) != set(required):
        raise CoverageError(f"{label} evidence does not exactly cover its declared scope")
    for record_hash, expected in required.items():
        row = evidence[record_hash]
        if row["payload_sha256"] != expected["payload_sha256"]:
            raise CoverageError(f"{label} payload SHA drift")
        if row["payload_bytes"] != expected["payload_bytes"]:
            raise CoverageError(f"{label} payload byte drift")
        if row["decision"] != accepted_decision:
            raise CoverageError(f"{label} includes non-{accepted_decision} decision")
    return identity, scope, subset_proved


def bind_final_g05_g06_coverage(
    *,
    inventory: Mapping[str, Any],
    decontamination_binding: Mapping[str, Any],
    quality_authority: Mapping[str, Any],
    privacy_authority: Mapping[str, Any],
    expected_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    expected_quality_authority_identity_sha256: str,
    expected_privacy_authority_identity_sha256: str,
    expected_privacy_policy_sha256: str,
    source_git_sha: str,
) -> dict[str, Any]:
    """Bind exact complete quality/privacy evidence to final survivor payloads."""
    source_git_sha = _require_git_sha(source_git_sha, "source_git_sha")
    expected_privacy_policy_sha256 = _require_sha256(
        expected_privacy_policy_sha256,
        "expected_privacy_policy_sha256",
    )

    inventory_rows, inventory_identity = _inventory_records(
        inventory,
        expected_inventory_identity_sha256=expected_inventory_identity_sha256,
    )
    survivors, decontam_identity = _decontam_survivors(
        decontamination_binding,
        inventory_rows,
        expected_decontamination_authority_sha256=(
            expected_decontamination_authority_sha256
        ),
        inventory_identity_sha256=inventory_identity,
    )

    if (
        quality_authority.get("quality_threshold_policy_sha256")
        != QUALITY_THRESHOLD_POLICY_SHA256
    ):
        raise CoverageError("quality threshold policy identity mismatch")
    if (
        quality_authority.get("quality_granularity_policy_sha256")
        != QUALITY_GRANULARITY_POLICY_SHA256
    ):
        raise CoverageError("quality granularity policy identity mismatch")
    if privacy_authority.get("privacy_policy_sha256") != expected_privacy_policy_sha256:
        raise CoverageError("privacy policy identity mismatch")

    quality_identity, quality_scope, quality_subset = _verify_authority_coverage(
        quality_authority,
        label="quality",
        schema=QUALITY_AUTHORITY_SCHEMA,
        identity_field="quality_authority_identity_sha256",
        expected_identity=expected_quality_authority_identity_sha256,
        inventory_rows=inventory_rows,
        survivor_rows=survivors,
        inventory_identity_sha256=inventory_identity,
        decontamination_authority_sha256=decontam_identity,
        accepted_decision="RETAIN",
    )
    privacy_identity, privacy_scope, privacy_subset = _verify_authority_coverage(
        privacy_authority,
        label="privacy",
        schema=PRIVACY_AUTHORITY_SCHEMA,
        identity_field="privacy_authority_identity_sha256",
        expected_identity=expected_privacy_authority_identity_sha256,
        inventory_rows=inventory_rows,
        survivor_rows=survivors,
        inventory_identity_sha256=inventory_identity,
        decontamination_authority_sha256=decontam_identity,
        accepted_decision="ALLOW",
    )

    sorted_survivors = sorted(
        survivors.values(),
        key=lambda row: row["record_id_sha256"],
    )
    report: dict[str, Any] = {
        "schema": COVERAGE_SCHEMA,
        "status": "PASS",
        "source_git_sha": source_git_sha,
        "retained_inventory_identity_sha256": inventory_identity,
        "decontamination_authority_sha256": decontam_identity,
        "quality_authority_identity_sha256": quality_identity,
        "privacy_authority_identity_sha256": privacy_identity,
        "quality_threshold_policy_sha256": QUALITY_THRESHOLD_POLICY_SHA256,
        "quality_granularity_policy_sha256": QUALITY_GRANULARITY_POLICY_SHA256,
        "privacy_policy_sha256": expected_privacy_policy_sha256,
        "quality_coverage_scope": quality_scope,
        "privacy_coverage_scope": privacy_scope,
        "quality_subset_preservation_proved": quality_subset,
        "privacy_subset_preservation_proved": privacy_subset,
        "survivor_record_count": len(sorted_survivors),
        "survivor_payload_bytes": sum(
            row["payload_bytes"] for row in sorted_survivors
        ),
        "survivor_record_id_membership_sha256": sha256_json(
            [row["record_id_sha256"] for row in sorted_survivors]
        ),
        "survivor_payload_membership_sha256": sha256_json(
            [
                {
                    "record_id_sha256": row["record_id_sha256"],
                    "payload_sha256": row["payload_sha256"],
                    "payload_bytes": row["payload_bytes"],
                }
                for row in sorted_survivors
            ]
        ),
        "next_gate": "POSTDECONTAM_BALANCE_FAMILY_CAP",
        "truth_boundary": dict(ZERO_TRUTH),
    }
    report["coverage_identity_sha256"] = authority_identity(
        report,
        "coverage_identity_sha256",
    )
    return report


def verify_coverage_report(
    report: Mapping[str, Any],
    *,
    expected_coverage_identity_sha256: str,
    expected_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    expected_quality_authority_identity_sha256: str,
    expected_privacy_authority_identity_sha256: str,
) -> None:
    """Verify the compact downstream handoff without widening authority."""
    if report.get("schema") != COVERAGE_SCHEMA or report.get("status") != "PASS":
        raise CoverageError("unsupported or non-PASS G05/G06 coverage report")
    _verify_external_self_hash(
        report,
        identity_field="coverage_identity_sha256",
        expected_identity=expected_coverage_identity_sha256,
        label="coverage",
    )
    required_bindings = {
        "retained_inventory_identity_sha256": expected_inventory_identity_sha256,
        "decontamination_authority_sha256": expected_decontamination_authority_sha256,
        "quality_authority_identity_sha256": expected_quality_authority_identity_sha256,
        "privacy_authority_identity_sha256": expected_privacy_authority_identity_sha256,
    }
    for field, expected in required_bindings.items():
        expected = _require_sha256(expected, f"expected_{field}")
        if report.get(field) != expected:
            raise CoverageError(f"coverage {field} mismatch")
    if report.get("quality_threshold_policy_sha256") != QUALITY_THRESHOLD_POLICY_SHA256:
        raise CoverageError("coverage quality threshold identity drift")
    if report.get("quality_granularity_policy_sha256") != QUALITY_GRANULARITY_POLICY_SHA256:
        raise CoverageError("coverage quality granularity identity drift")
    if report.get("next_gate") != "POSTDECONTAM_BALANCE_FAMILY_CAP":
        raise CoverageError("coverage report next_gate mismatch")
    _verify_zero_truth(report.get("truth_boundary"), "coverage.truth_boundary")
    _require_int(
        report.get("survivor_record_count"),
        "survivor_record_count",
        minimum=1,
    )
    _require_int(
        report.get("survivor_payload_bytes"),
        "survivor_payload_bytes",
        minimum=1,
    )
    _require_sha256(
        report.get("survivor_record_id_membership_sha256"),
        "survivor_record_id_membership_sha256",
    )
    _require_sha256(
        report.get("survivor_payload_membership_sha256"),
        "survivor_payload_membership_sha256",
    )
