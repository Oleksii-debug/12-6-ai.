"""Fail-closed final G05/G06 coverage binder for expanded-V9.

No quality or privacy detector is implemented here. The binder proves that every
final post-decontamination survivor is unchanged and covered by independently
expected, text-free G05/G06 qualification evidence.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

SCHEMA = "12-6.d03-final-g05-g06-coverage.v1"
INVENTORY_SCHEMA = "twelve-six.expanded-postdedup-inventory.v1"
DECONTAM_SCHEMA = "12-6.d03-final-record-decontamination-binding.v1"
QUALIFICATION_SCHEMA = "12-6.d03-g05-g06-qualification-authority.v1"
QUALITY_POLICY_SHA256 = (
    "97b9fe1452b22c6275a27f85524f670253a7f4012377361c4cb007004aeccd1d"
)
QUALITY_GRANULARITY_SHA256 = (
    "e8685c2c6b265b9b289ded7a5245888d8d16ae4d6e881f6229f3bc777601f857"
)
QUALIFICATION_SCOPE = "PRE_DECONTAM_UNCHANGED_PAYLOAD_SUBSET_V1"
PASS_VERDICTS = frozenset({"PASS_CLEAN", "PASS_WITH_EXCLUSIONS"})

# Public aliases match the names used by adjacent D03 authority packages.
RETAINED_INVENTORY_SCHEMA = INVENTORY_SCHEMA
DECONTAMINATION_BINDING_SCHEMA = DECONTAM_SCHEMA
QUALIFICATION_AUTHORITY_SCHEMA = QUALIFICATION_SCHEMA
QUALITY_POLICY_IDENTITY_SHA256 = QUALITY_POLICY_SHA256
QUALITY_GRANULARITY_IDENTITY_SHA256 = QUALITY_GRANULARITY_SHA256

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")


class CoverageError(ValueError):
    """Raised when the final quality/privacy authority is not fail-closed."""


FinalG05G06CoverageError = CoverageError


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise CoverageError(message)


def _cjson(value: Any, *, newline: bool = False) -> bytes:
    text = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return (text + ("\n" if newline else "")).encode("utf-8")


def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _self_hash(
    document: Mapping[str, Any],
    field: str,
    *,
    newline: bool = False,
) -> str:
    body = dict(document)
    body.pop(field, None)
    return _sha(_cjson(body, newline=newline))


def _sha256(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
        raise CoverageError(f"{field} must be lowercase SHA-256")
    return value


def _git_sha(value: Any, field: str) -> str:
    if not isinstance(value, str) or _HEX40.fullmatch(value) is None:
        raise CoverageError(f"{field} must be a lowercase 40-hex Git object id")
    return value


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CoverageError(f"{field} must be a non-negative integer")
    return value


def _false(value: Any, field: str) -> None:
    if value is not False:
        raise CoverageError(f"{field} must be false")


def _record_hash(record_id: str) -> str:
    return _sha(record_id.encode("utf-8"))


def _inventory(
    document: Mapping[str, Any],
    *,
    expected_identity: str,
) -> tuple[dict[str, tuple[str, int]], str, int]:
    _require(document.get("schema") == INVENTORY_SCHEMA, "inventory schema drift")
    expected = _sha256(expected_identity, "expected inventory identity")
    claimed = _sha256(
        document.get("inventory_identity_sha256"),
        "inventory identity",
    )
    _require(claimed == expected, "inventory is not independently expected")
    _require(
        claimed == _self_hash(document, "inventory_identity_sha256", newline=True),
        "inventory self-hash mismatch",
    )
    truth = document.get("truth_boundary")
    _require(isinstance(truth, Mapping), "inventory truth boundary missing")
    expected_truth = {
        "training_eligible": False,
        "evaluation_eligible": False,
        "authorized_optimized_target_exposure": 0,
        "tokenizer_fit": False,
        "optimizer_updates": 0,
        "model_training": False,
        "final_test_outcomes_accessed": False,
        "paid_compute_used": False,
    }
    for key, value in expected_truth.items():
        _require(truth.get(key) == value, f"inventory truth drift: {key}")

    rows = document.get("records")
    _require(isinstance(rows, list) and bool(rows), "inventory records missing")
    required = {
        "record_id",
        "source_id",
        "family",
        "modality",
        "payload_sha256",
        "payload_bytes",
        "comparison_policy_id",
        "comparison_sha256",
        "comparison_bytes",
        "training_eligible",
        "evaluation_eligible",
    }
    result: dict[str, tuple[str, int]] = {}
    record_ids: list[str] = []
    payload_total = 0
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"inventory record[{index}] is not an object")
        _require(set(row) == required, f"inventory record[{index}] schema drift")
        record_id = row.get("record_id")
        _require(isinstance(record_id, str) and bool(record_id), "record_id missing")
        _require(record_id not in record_ids, f"duplicate record_id: {record_id}")
        record_ids.append(record_id)
        for field in ("source_id", "family", "modality", "comparison_policy_id"):
            _require(
                isinstance(row.get(field), str) and bool(row[field]),
                f"{field} missing for {record_id}",
            )
        payload_sha = _sha256(row.get("payload_sha256"), "payload_sha256")
        payload_bytes = _nonnegative_int(row.get("payload_bytes"), "payload_bytes")
        _require(payload_bytes > 0, "payload_bytes must be positive")
        _sha256(row.get("comparison_sha256"), "comparison_sha256")
        _nonnegative_int(row.get("comparison_bytes"), "comparison_bytes")
        _false(row.get("training_eligible"), "record.training_eligible")
        _false(row.get("evaluation_eligible"), "record.evaluation_eligible")
        result[_record_hash(record_id)] = (payload_sha, payload_bytes)
        payload_total += payload_bytes

    _require(record_ids == sorted(record_ids), "inventory records are not sorted")
    _require(document.get("record_count") == len(rows), "inventory count mismatch")
    _require(
        document.get("retained_payload_bytes") == payload_total,
        "inventory byte total mismatch",
    )
    return result, claimed, payload_total


def _decontam(
    document: Mapping[str, Any],
    *,
    expected_identity: str,
    expected_records_sha256: str,
    inventory_identity: str,
    records: Mapping[str, tuple[str, int]],
    input_bytes: int,
) -> tuple[set[str], str, int]:
    _require(document.get("schema") == DECONTAM_SCHEMA, "decontam schema drift")
    expected = _sha256(expected_identity, "expected decontam identity")
    claimed = _sha256(
        document.get("decontamination_authority_sha256"),
        "decontamination authority",
    )
    _require(claimed == expected, "decontam authority is not independently expected")
    _require(
        claimed == _self_hash(document, "decontamination_authority_sha256"),
        "decontam self-hash mismatch",
    )
    records_sha = _sha256(expected_records_sha256, "expected records JSONL SHA-256")
    _require(document.get("records_jsonl_sha256") == records_sha, "records identity drift")
    _require(
        document.get("retained_inventory_identity_sha256") == inventory_identity,
        "decontam inventory lineage drift",
    )
    _require(document.get("verdict") in PASS_VERDICTS, "decontam is not terminal PASS")
    _require(document.get("input_record_count") == len(records), "decontam count drift")
    _require(document.get("input_payload_bytes") == input_bytes, "decontam byte drift")
    _false(document.get("final_test_outcomes_read"), "final_test_outcomes_read")
    _false(document.get("model_selection_performed"), "model_selection_performed")
    _false(
        document.get("training_authorized_by_this_report"),
        "training_authorized_by_this_report",
    )
    _require(
        document.get("authorized_optimized_target_exposure") == 0,
        "decontam grants optimized-target exposure",
    )

    excluded_raw = document.get("excluded_record_id_sha256")
    _require(isinstance(excluded_raw, list), "decontam exclusions missing")
    excluded: set[str] = set()
    excluded_bytes = 0
    for index, value in enumerate(excluded_raw):
        digest = _sha256(value, f"excluded[{index}]")
        _require(digest not in excluded, "duplicate decontam exclusion")
        _require(digest in records, "decontam excludes unknown record")
        excluded.add(digest)
        excluded_bytes += records[digest][1]
    survivor_count = len(records) - len(excluded)
    survivor_bytes = input_bytes - excluded_bytes
    _require(survivor_count > 0, "decontam left no survivors")
    _require(
        document.get("survivor_record_count") == survivor_count,
        "decontam survivor count drift",
    )
    _require(
        document.get("survivor_payload_bytes") == survivor_bytes,
        "decontam survivor byte drift",
    )
    return excluded, claimed, excluded_bytes


def _qualification(
    document: Mapping[str, Any],
    *,
    expected_identity: str,
    expected_privacy_policy: str,
    expected_privacy_implementation: str,
    records: Mapping[str, tuple[str, int]],
) -> tuple[dict[str, tuple[str, int]], str]:
    _require(
        document.get("schema") == QUALIFICATION_SCHEMA,
        "qualification schema drift",
    )
    expected = _sha256(expected_identity, "expected qualification identity")
    claimed = _sha256(
        document.get("qualification_authority_identity_sha256"),
        "qualification authority identity",
    )
    _require(claimed == expected, "qualification is not independently expected")
    _require(
        claimed == _self_hash(document, "qualification_authority_identity_sha256"),
        "qualification self-hash mismatch",
    )
    _require(document.get("status") == "PASS", "qualification is not terminal PASS")
    _require(document.get("coverage_scope") == QUALIFICATION_SCOPE, "scope drift")
    _require(
        document.get("quality_policy_identity_sha256") == QUALITY_POLICY_SHA256,
        "noncanonical quality policy",
    )
    _require(
        document.get("quality_granularity_identity_sha256")
        == QUALITY_GRANULARITY_SHA256,
        "noncanonical quality granularity",
    )
    privacy_policy = _sha256(expected_privacy_policy, "expected privacy policy")
    privacy_impl = _git_sha(
        expected_privacy_implementation,
        "expected privacy implementation",
    )
    _require(
        document.get("privacy_policy_identity_sha256") == privacy_policy,
        "privacy policy identity drift",
    )
    _require(
        document.get("privacy_implementation_git_blob_sha") == privacy_impl,
        "privacy implementation identity mismatch",
    )
    _false(document.get("final_test_outcomes_read"), "qualification outcomes boundary")
    _false(
        document.get("training_authorized_by_this_authority"),
        "qualification training boundary",
    )
    _require(
        document.get("authorized_optimized_target_exposure") == 0,
        "qualification grants optimized-target exposure",
    )

    rows = document.get("records")
    _require(isinstance(rows, list) and bool(rows), "qualification records missing")
    required = {
        "record_id_sha256",
        "payload_sha256",
        "payload_bytes",
        "quality_decision",
        "privacy_action",
    }
    result: dict[str, tuple[str, int]] = {}
    order: list[str] = []
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"qualification row[{index}] not an object")
        _require(set(row) == required, f"qualification row[{index}] schema drift")
        record_hash = _sha256(row.get("record_id_sha256"), "record_id_sha256")
        _require(record_hash not in result, "duplicate qualification record")
        _require(record_hash in records, "qualification record outside retained inventory")
        payload_sha = _sha256(row.get("payload_sha256"), "payload_sha256")
        payload_bytes = _nonnegative_int(row.get("payload_bytes"), "payload_bytes")
        _require(
            (payload_sha, payload_bytes) == records[record_hash],
            "qualification payload SHA-256 mismatch or byte mismatch",
        )
        _require(row.get("quality_decision") == "ACCEPT", "non-accepted quality decision")
        _require(row.get("privacy_action") == "ALLOW", "non-ALLOW privacy decision")
        result[record_hash] = (payload_sha, payload_bytes)
        order.append(record_hash)
    _require(order == sorted(order), "qualification records are not sorted")
    _require(document.get("qualified_record_count") == len(result), "qualified count drift")
    _require(
        document.get("qualified_payload_bytes") == sum(value[1] for value in result.values()),
        "qualified byte total drift",
    )
    return result, claimed


def build_final_g05_g06_coverage(
    *,
    retained_inventory: Mapping[str, Any],
    expected_retained_inventory_identity_sha256: str,
    decontamination_binding: Mapping[str, Any],
    expected_decontamination_authority_sha256: str,
    expected_records_jsonl_sha256: str,
    qualification_authorities: Sequence[Mapping[str, Any]],
    expected_qualification_authority_identities_sha256: Sequence[str],
    expected_privacy_policy_identity_sha256: str,
    expected_privacy_implementation_git_blob_sha: str,
) -> dict[str, Any]:
    """Build the text-free coverage authority consumed by the balance bridge."""
    records, inventory_identity, input_bytes = _inventory(
        retained_inventory,
        expected_identity=expected_retained_inventory_identity_sha256,
    )
    excluded, decontam_identity, excluded_bytes = _decontam(
        decontamination_binding,
        expected_identity=expected_decontamination_authority_sha256,
        expected_records_sha256=expected_records_jsonl_sha256,
        inventory_identity=inventory_identity,
        records=records,
        input_bytes=input_bytes,
    )
    _require(
        isinstance(qualification_authorities, Sequence)
        and not isinstance(qualification_authorities, (str, bytes))
        and bool(qualification_authorities),
        "qualification authorities missing",
    )
    expected_ids = list(expected_qualification_authority_identities_sha256)
    _require(len(expected_ids) == len(qualification_authorities), "authority count drift")
    _require(len(expected_ids) == len(set(expected_ids)), "duplicate expected authority")

    covered: dict[str, tuple[str, int]] = {}
    authority_ids: list[str] = []
    for authority, expected_identity in zip(
        qualification_authorities,
        expected_ids,
        strict=True,
    ):
        rows, identity = _qualification(
            authority,
            expected_identity=expected_identity,
            expected_privacy_policy=expected_privacy_policy_identity_sha256,
            expected_privacy_implementation=expected_privacy_implementation_git_blob_sha,
            records=records,
        )
        for record_hash, payload_identity in rows.items():
            _require(
                record_hash not in covered,
                "record covered by multiple qualification authorities",
            )
            covered[record_hash] = payload_identity
        authority_ids.append(identity)

    survivors = set(records) - excluded
    _require(survivors <= set(covered), "final survivor missing G05/G06 coverage")
    _require(set(covered) <= survivors | excluded, "coverage is not an unchanged subset")
    rows = [
        {
            "record_id_sha256": record_hash,
            "payload_sha256": records[record_hash][0],
            "payload_bytes": records[record_hash][1],
        }
        for record_hash in sorted(survivors)
    ]
    privacy_policy = _sha256(
        expected_privacy_policy_identity_sha256,
        "expected privacy policy",
    )
    privacy_impl = _git_sha(
        expected_privacy_implementation_git_blob_sha,
        "expected privacy implementation",
    )
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS",
        "records_jsonl_sha256": _sha256(
            expected_records_jsonl_sha256,
            "expected records JSONL SHA-256",
        ),
        "retained_inventory_identity_sha256": inventory_identity,
        "decontamination_authority_sha256": decontam_identity,
        "quality_policy_identity_sha256": QUALITY_POLICY_SHA256,
        "quality_granularity_identity_sha256": QUALITY_GRANULARITY_SHA256,
        "privacy_policy_identity_sha256": privacy_policy,
        "privacy_implementation_git_blob_sha": privacy_impl,
        "qualification_authority_identities_sha256": sorted(authority_ids),
        "qualification_authority_set_identity_sha256": _sha(
            _cjson(sorted(authority_ids))
        ),
        "projection_rule": QUALIFICATION_SCOPE,
        "predecontam_record_count": len(records),
        "predecontam_payload_bytes": input_bytes,
        "excluded_record_count": len(excluded),
        "excluded_payload_bytes": excluded_bytes,
        "covered_record_count": len(rows),
        "covered_payload_bytes": sum(row["payload_bytes"] for row in rows),
        "covered_records": rows,
        "final_test_outcomes_read": False,
        "training_authorized_by_this_coverage": False,
        "tokenizer_fit_authorized": False,
        "model_training_authorized": False,
        "authorized_optimized_target_exposure": 0,
        "paid_compute_used": False,
    }
    result["g05_g06_coverage_identity_sha256"] = _self_hash(
        result,
        "g05_g06_coverage_identity_sha256",
    )
    return result


def verify_final_g05_g06_coverage(
    document: Mapping[str, Any],
    *,
    expected_identity_sha256: str,
    expected_retained_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    expected_records_jsonl_sha256: str,
    expected_privacy_policy_identity_sha256: str,
    expected_privacy_implementation_git_blob_sha: str,
) -> None:
    """Verify an already-materialized durable coverage authority."""
    _require(document.get("schema") == SCHEMA, "coverage schema drift")
    expected = _sha256(expected_identity_sha256, "expected coverage identity")
    claimed = _sha256(
        document.get("g05_g06_coverage_identity_sha256"),
        "coverage identity",
    )
    _require(claimed == expected, "coverage is not independently expected")
    _require(
        claimed == _self_hash(document, "g05_g06_coverage_identity_sha256"),
        "coverage self-hash mismatch",
    )
    _require(document.get("status") == "PASS", "coverage is not terminal PASS")
    expected_pairs = {
        "retained_inventory_identity_sha256": _sha256(
            expected_retained_inventory_identity_sha256,
            "expected inventory identity",
        ),
        "decontamination_authority_sha256": _sha256(
            expected_decontamination_authority_sha256,
            "expected decontam identity",
        ),
        "records_jsonl_sha256": _sha256(
            expected_records_jsonl_sha256,
            "expected records JSONL SHA-256",
        ),
        "privacy_policy_identity_sha256": _sha256(
            expected_privacy_policy_identity_sha256,
            "expected privacy policy",
        ),
        "privacy_implementation_git_blob_sha": _git_sha(
            expected_privacy_implementation_git_blob_sha,
            "expected privacy implementation",
        ),
    }
    for key, value in expected_pairs.items():
        message = (
            "privacy implementation identity mismatch"
            if key == "privacy_implementation_git_blob_sha"
            else f"coverage identity drift: {key}"
        )
        _require(document.get(key) == value, message)
    _require(
        document.get("quality_policy_identity_sha256") == QUALITY_POLICY_SHA256,
        "noncanonical quality policy",
    )
    _require(
        document.get("quality_granularity_identity_sha256")
        == QUALITY_GRANULARITY_SHA256,
        "noncanonical quality granularity",
    )
    for key in (
        "final_test_outcomes_read",
        "training_authorized_by_this_coverage",
        "tokenizer_fit_authorized",
        "model_training_authorized",
        "paid_compute_used",
    ):
        _false(document.get(key), key)
    _require(
        document.get("authorized_optimized_target_exposure") == 0,
        "coverage grants optimized-target exposure",
    )
    rows = document.get("covered_records")
    _require(isinstance(rows, list) and bool(rows), "covered records missing")
    required = {"record_id_sha256", "payload_sha256", "payload_bytes"}
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"covered row[{index}] not an object")
        _require(set(row) == required, f"covered row[{index}] schema drift")
        record_hash = _sha256(row.get("record_id_sha256"), "record_id_sha256")
        _require(record_hash not in seen, "duplicate covered record")
        seen.add(record_hash)
        normalized.append(
            {
                "record_id_sha256": record_hash,
                "payload_sha256": _sha256(row.get("payload_sha256"), "payload_sha256"),
                "payload_bytes": _nonnegative_int(
                    row.get("payload_bytes"),
                    "payload_bytes",
                ),
            }
        )
    _require(
        normalized == sorted(normalized, key=lambda row: row["record_id_sha256"]),
        "covered records are not sorted",
    )
    _require(document.get("covered_record_count") == len(rows), "covered count drift")
    _require(
        document.get("covered_payload_bytes")
        == sum(row["payload_bytes"] for row in normalized),
        "covered byte total drift",
    )


# AUD1041-001 hardening: verify every durable semantic field independently before
# delegating to the original exact-identity/self-hash verifier above.
_legacy_verify_final_g05_g06_coverage = verify_final_g05_g06_coverage
_COVERAGE_ROOT_KEYS = frozenset(
    {
        "schema",
        "status",
        "records_jsonl_sha256",
        "retained_inventory_identity_sha256",
        "decontamination_authority_sha256",
        "quality_policy_identity_sha256",
        "quality_granularity_identity_sha256",
        "privacy_policy_identity_sha256",
        "privacy_implementation_git_blob_sha",
        "qualification_authority_identities_sha256",
        "qualification_authority_set_identity_sha256",
        "projection_rule",
        "predecontam_record_count",
        "predecontam_payload_bytes",
        "excluded_record_count",
        "excluded_payload_bytes",
        "covered_record_count",
        "covered_payload_bytes",
        "covered_records",
        "final_test_outcomes_read",
        "training_authorized_by_this_coverage",
        "tokenizer_fit_authorized",
        "model_training_authorized",
        "authorized_optimized_target_exposure",
        "paid_compute_used",
        "g05_g06_coverage_identity_sha256",
    }
)


def verify_final_g05_g06_coverage(
    document: Mapping[str, Any],
    *,
    expected_identity_sha256: str,
    expected_retained_inventory_identity_sha256: str,
    expected_decontamination_authority_sha256: str,
    expected_records_jsonl_sha256: str,
    expected_privacy_policy_identity_sha256: str,
    expected_privacy_implementation_git_blob_sha: str,
) -> None:
    """Verify durable G05/G06 coverage with closed-world provenance accounting."""
    _require(set(document) == _COVERAGE_ROOT_KEYS, "coverage root schema drift")
    _require(
        document.get("projection_rule") == QUALIFICATION_SCOPE,
        "coverage projection rule drift",
    )

    raw_authority_ids = document.get("qualification_authority_identities_sha256")
    _require(
        isinstance(raw_authority_ids, list) and bool(raw_authority_ids),
        "qualification authority roots missing",
    )
    authority_ids = [
        _sha256(value, f"qualification authority identity[{index}]")
        for index, value in enumerate(raw_authority_ids)
    ]
    _require(
        authority_ids == sorted(authority_ids),
        "qualification authority roots are not sorted",
    )
    _require(
        len(authority_ids) == len(set(authority_ids)),
        "duplicate qualification authority root",
    )
    authority_set_identity = _sha256(
        document.get("qualification_authority_set_identity_sha256"),
        "qualification authority set identity",
    )
    _require(
        authority_set_identity == _sha(_cjson(authority_ids)),
        "qualification authority set identity drift",
    )

    pre_count = _nonnegative_int(
        document.get("predecontam_record_count"),
        "predecontam_record_count",
    )
    pre_bytes = _nonnegative_int(
        document.get("predecontam_payload_bytes"),
        "predecontam_payload_bytes",
    )
    excluded_count = _nonnegative_int(
        document.get("excluded_record_count"),
        "excluded_record_count",
    )
    excluded_bytes = _nonnegative_int(
        document.get("excluded_payload_bytes"),
        "excluded_payload_bytes",
    )
    covered_count = _nonnegative_int(
        document.get("covered_record_count"),
        "covered_record_count",
    )
    covered_bytes = _nonnegative_int(
        document.get("covered_payload_bytes"),
        "covered_payload_bytes",
    )
    _require(pre_count > 0 and pre_bytes > 0, "predecontam accounting must be positive")
    _require(excluded_count <= pre_count, "excluded record accounting exceeds input")
    _require(excluded_bytes <= pre_bytes, "excluded byte accounting exceeds input")
    _require(covered_count > 0 and covered_bytes > 0, "coverage accounting must be positive")
    _require(
        pre_count - excluded_count == covered_count,
        "record projection accounting drift",
    )
    _require(
        pre_bytes - excluded_bytes == covered_bytes,
        "byte projection accounting drift",
    )

    exposure = _nonnegative_int(
        document.get("authorized_optimized_target_exposure"),
        "authorized_optimized_target_exposure",
    )
    _require(exposure == 0, "coverage grants optimized-target exposure")

    rows = document.get("covered_records")
    _require(isinstance(rows, list) and bool(rows), "covered records missing")
    for index, row in enumerate(rows):
        _require(isinstance(row, Mapping), f"covered row[{index}] not an object")
        payload_bytes = _nonnegative_int(
            row.get("payload_bytes"),
            f"covered row[{index}].payload_bytes",
        )
        _require(payload_bytes > 0, f"covered row[{index}] payload_bytes must be positive")

    _legacy_verify_final_g05_g06_coverage(
        document,
        expected_identity_sha256=expected_identity_sha256,
        expected_retained_inventory_identity_sha256=(
            expected_retained_inventory_identity_sha256
        ),
        expected_decontamination_authority_sha256=(
            expected_decontamination_authority_sha256
        ),
        expected_records_jsonl_sha256=expected_records_jsonl_sha256,
        expected_privacy_policy_identity_sha256=(
            expected_privacy_policy_identity_sha256
        ),
        expected_privacy_implementation_git_blob_sha=(
            expected_privacy_implementation_git_blob_sha
        ),
    )
