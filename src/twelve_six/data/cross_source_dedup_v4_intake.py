"""Fail-closed object-level intake for NEXT100-065 successor global dedup V4."""
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

CONFIG_SCHEMA = "12-6.next100-065-cross-source-dedup-v4-intake.v1"
CONVERGENCE_SCHEMA = "12-6.next100-063-source-registry-convergence.v1"
READY = "READY_FOR_GLOBAL_DEDUP_OBJECT_COMPARISON"
BLOCKED_UPSTREAM = "BLOCKED_UPSTREAM_SOURCE_CONVERGENCE_NONTERMINAL"
BLOCKED_OBJECTS = "BLOCKED_OBJECT_LEVEL_HANDOFF"
ALLOWED_MODALITIES = {"uk", "en", "code"}
ALLOWED_COMPARISON_NORMALIZATIONS = {
    "DATA232_GENERIC_FROM_RAW",
    "RUST_BOOK_SOURCE_MARKDOWN_PROSE_ONLY_V1",
    "NEXT100_026_KMU_AUTHORITY_NORMALIZED_V1",
    "NEXT100_034_NIST_PDF_BODY_NFKC_BOUNDED_V1",
    "MDN_PROSE_ONLY_MARKDOWN_V1",
    "NEXT100_027_VERBA_NOMIS1864_NORMALIZED_V1",
}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA1_RE = re.compile(r"^[0-9a-f]{40}$")


class CrossSourceV4IntakeError(RuntimeError):
    """Raised when the V4 intake contract itself is malformed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CrossSourceV4IntakeError(message)


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_vector(vector: Mapping[str, Any], name: str) -> None:
    capacity = vector.get("numeric_capacity_bytes")
    families = vector.get("independent_family_counts")
    _require(isinstance(capacity, Mapping), f"{name}: numeric_capacity_bytes missing")
    _require(isinstance(families, Mapping), f"{name}: independent_family_counts missing")
    for key in ("uk", "en", "code", "total"):
        _require(_nonnegative_int(capacity.get(key)), f"{name}: invalid capacity {key}")
        _require(_nonnegative_int(families.get(key)), f"{name}: invalid family count {key}")
    _require(
        capacity["uk"] + capacity["en"] + capacity["code"] == capacity["total"],
        f"{name}: capacity total mismatch",
    )
    _require(
        families["uk"] + families["en"] + families["code"] == families["total"],
        f"{name}: family total mismatch",
    )


def _validate_object_record(record: Mapping[str, Any], expected_modality: str) -> list[str]:
    blockers: list[str] = []
    for key in (
        "source_id",
        "source_family",
        "stable_origin_id",
        "stable_object_id",
        "origin_key",
    ):
        if not isinstance(record.get(key), str) or not record[key].strip():
            blockers.append(f"object_record_missing_{key}")
    modality = record.get("modality")
    if modality not in ALLOWED_MODALITIES:
        blockers.append("object_record_invalid_modality")
    elif modality != expected_modality:
        blockers.append("object_record_authority_modality_mismatch")
    if not _positive_int(record.get("declared_capacity_bytes")):
        blockers.append("object_record_invalid_declared_capacity_bytes")

    raw_sha = record.get("expected_raw_sha256")
    git_sha = record.get("expected_git_blob_sha1")
    has_raw_sha = isinstance(raw_sha, str) and SHA256_RE.fullmatch(raw_sha) is not None
    has_git_sha = isinstance(git_sha, str) and GIT_SHA1_RE.fullmatch(git_sha) is not None
    if not has_raw_sha and not has_git_sha:
        blockers.append("object_record_missing_content_identity")

    normalization = record.get("comparison_normalization", "DATA232_GENERIC_FROM_RAW")
    if normalization not in ALLOWED_COMPARISON_NORMALIZATIONS:
        blockers.append("object_record_unsupported_comparison_normalization")
    if normalization != "DATA232_GENERIC_FROM_RAW":
        comparison_sha = record.get("expected_comparison_sha256")
        comparison_bytes = record.get("expected_comparison_bytes")
        if not isinstance(comparison_sha, str) or SHA256_RE.fullmatch(comparison_sha) is None:
            blockers.append("object_record_missing_comparison_sha256")
        if not _positive_int(comparison_bytes):
            blockers.append("object_record_missing_comparison_bytes")
    return blockers


def evaluate_v4_intake(
    config: Mapping[str, Any],
    convergence: Mapping[str, Any],
) -> dict[str, Any]:
    """Evaluate whether exact object-level inputs are safe to hand to global dedup.

    READY means only that the successor dedup comparison may execute. It never means
    that post-dedup capacity, corpus readiness, tokenizer readiness, or training
    readiness has been established.
    """
    _require(config.get("schema_version") == CONFIG_SCHEMA, "unsupported V4 intake schema")
    _require(config.get("local_free_only") is True, "V4 intake must be LOCAL_FREE only")
    for flag in (
        "model_training_executed",
        "tokenizer_fit_executed",
        "paid_compute_used",
        "final_test_payload_read",
    ):
        _require(config.get(flag) is False, f"{flag} must be false")
    _require(
        convergence.get("schema_version") == CONVERGENCE_SCHEMA,
        "unsupported upstream convergence schema",
    )
    _require(convergence.get("local_free_only") is True, "upstream must be LOCAL_FREE only")
    _require(convergence.get("model_training_executed") is False, "upstream trained a model")

    upstream = config.get("upstream_source_convergence")
    _require(isinstance(upstream, Mapping), "upstream_source_convergence missing")
    _require(upstream.get("schema_version") == CONVERGENCE_SCHEMA, "upstream schema binding mismatch")
    _require(isinstance(upstream.get("head_sha"), str), "upstream head_sha missing")
    _require(isinstance(upstream.get("workflow_run"), int), "upstream workflow_run missing")

    expected_vector = config.get("expected_pre_dedup_vector")
    actual_vector = convergence.get("converged_pre_successor_dedup_vector")
    _require(isinstance(expected_vector, Mapping), "expected_pre_dedup_vector missing")
    _require(isinstance(actual_vector, Mapping), "upstream converged vector missing")
    _validate_vector(expected_vector, "expected_pre_dedup_vector")
    _validate_vector(actual_vector, "upstream converged vector")
    _require(
        expected_vector["numeric_capacity_bytes"] == actual_vector["numeric_capacity_bytes"],
        "upstream capacity vector drifted",
    )
    _require(
        expected_vector["independent_family_counts"] == actual_vector["independent_family_counts"],
        "upstream family vector drifted",
    )

    blockers: list[str] = []
    if upstream.get("workflow_status") != "completed" or upstream.get("workflow_conclusion") != "success":
        blockers.append("upstream_source_convergence_ci_not_terminal_success")

    late_raw = convergence.get("late_authorities")
    _require(isinstance(late_raw, list), "upstream late_authorities missing")
    late_by_worker: dict[str, Mapping[str, Any]] = {}
    for raw in late_raw:
        _require(isinstance(raw, Mapping), "late authority row must be an object")
        worker_id = raw.get("worker_id")
        _require(isinstance(worker_id, str) and worker_id, "late authority worker_id missing")
        _require(worker_id not in late_by_worker, f"duplicate late authority {worker_id}")
        late_by_worker[worker_id] = raw

    required_raw = config.get("required_positive_credit_late_authorities")
    _require(isinstance(required_raw, list), "required_positive_credit_late_authorities missing")
    required_by_worker: dict[str, Mapping[str, Any]] = {}
    for required in required_raw:
        _require(isinstance(required, Mapping), "required authority row must be an object")
        worker_id = required.get("worker_id")
        _require(isinstance(worker_id, str) and worker_id, "required authority worker_id missing")
        _require(worker_id not in required_by_worker, f"duplicate required authority {worker_id}")
        required_by_worker[worker_id] = required
        upstream_row = late_by_worker.get(worker_id)
        if upstream_row is None:
            blockers.append(f"missing_upstream_late_authority:{worker_id}")
            continue
        for key in (
            "authority_identity",
            "family_id",
            "stratum",
            "numeric_capacity_bytes",
            "capacity_object_count",
        ):
            if required.get(key) != upstream_row.get(key):
                blockers.append(f"late_authority_binding_mismatch:{worker_id}:{key}")
        if upstream_row.get("training_authorized") is not True:
            blockers.append(f"late_authority_not_training_authorized:{worker_id}")
        if not _positive_int(upstream_row.get("numeric_capacity_bytes")):
            blockers.append(f"late_authority_nonpositive_credit:{worker_id}")
        if upstream_row.get("independent_family_credit") != 1:
            blockers.append(f"late_authority_family_credit_not_one:{worker_id}")

    for worker_id, upstream_row in late_by_worker.items():
        numeric_credit = upstream_row.get("numeric_capacity_bytes")
        family_credit = upstream_row.get("independent_family_credit")
        has_positive_credit = _positive_int(numeric_credit) or (
            isinstance(family_credit, int) and not isinstance(family_credit, bool) and family_credit > 0
        )
        if has_positive_credit and worker_id not in required_by_worker:
            blockers.append(f"unbound_positive_credit_late_authority:{worker_id}")

    manifests_raw = config.get("object_manifests")
    _require(isinstance(manifests_raw, list), "object_manifests must be a list")
    manifests: dict[str, Mapping[str, Any]] = {}
    global_source_ids: set[str] = set()
    global_object_ids: set[str] = set()
    object_count = 0
    object_capacity = 0

    for manifest in manifests_raw:
        _require(isinstance(manifest, Mapping), "object manifest must be an object")
        worker_id = manifest.get("worker_id")
        _require(isinstance(worker_id, str) and worker_id, "object manifest worker_id missing")
        if worker_id in manifests:
            blockers.append(f"duplicate_object_manifest:{worker_id}")
            continue
        manifests[worker_id] = manifest
        required = required_by_worker.get(worker_id)
        if required is None:
            blockers.append(f"unexpected_positive_credit_object_manifest:{worker_id}")
            continue
        if manifest.get("authority_identity") != required.get("authority_identity"):
            blockers.append(f"object_manifest_authority_identity_mismatch:{worker_id}")
        rows = manifest.get("source_objects")
        if not isinstance(rows, list) or not rows:
            blockers.append(f"object_manifest_empty:{worker_id}")
            continue
        expected_count = required["capacity_object_count"]
        if len(rows) != expected_count:
            blockers.append(f"object_manifest_count_mismatch:{worker_id}")
        manifest_capacity = 0
        for row in rows:
            if not isinstance(row, Mapping):
                blockers.append(f"object_record_not_object:{worker_id}")
                continue
            for reason in _validate_object_record(row, str(required["stratum"])):
                blockers.append(f"{reason}:{worker_id}:{row.get('source_id', '<unknown>')}")
            if row.get("source_family") != required.get("family_id"):
                blockers.append(f"object_record_family_mismatch:{worker_id}:{row.get('source_id', '<unknown>')}")
            source_id = row.get("source_id")
            stable_object_id = row.get("stable_object_id")
            if isinstance(source_id, str):
                if source_id in global_source_ids:
                    blockers.append(f"duplicate_global_source_id:{source_id}")
                global_source_ids.add(source_id)
            if isinstance(stable_object_id, str):
                if stable_object_id in global_object_ids:
                    blockers.append(f"duplicate_global_stable_object_id:{stable_object_id}")
                global_object_ids.add(stable_object_id)
            capacity = row.get("declared_capacity_bytes")
            if _positive_int(capacity):
                manifest_capacity += capacity
        if manifest_capacity != required["numeric_capacity_bytes"]:
            blockers.append(f"object_manifest_capacity_mismatch:{worker_id}")
        object_count += len(rows)
        object_capacity += manifest_capacity

    for worker_id in required_by_worker:
        if worker_id not in manifests:
            blockers.append(f"missing_positive_credit_object_manifest:{worker_id}")

    zero_credit_raw = config.get("zero_credit_late_authorities", [])
    _require(isinstance(zero_credit_raw, list), "zero_credit_late_authorities must be a list")
    for zero in zero_credit_raw:
        _require(isinstance(zero, Mapping), "zero-credit row must be an object")
        worker_id = zero.get("worker_id")
        if not isinstance(worker_id, str) or not worker_id:
            raise CrossSourceV4IntakeError("zero-credit worker_id missing")
        upstream_row = late_by_worker.get(worker_id)
        if upstream_row is None:
            blockers.append(f"missing_zero_credit_upstream_authority:{worker_id}")
            continue
        if upstream_row.get("numeric_capacity_bytes") != 0 or upstream_row.get("independent_family_credit") != 0:
            blockers.append(f"zero_credit_authority_gained_unbound_credit:{worker_id}")
        if worker_id in manifests:
            blockers.append(f"zero_credit_authority_must_not_enter_dedup:{worker_id}")

    if blockers:
        status = BLOCKED_UPSTREAM if blockers == ["upstream_source_convergence_ci_not_terminal_success"] else BLOCKED_OBJECTS
    else:
        status = READY
    return {
        "schema_version": "12-6.next100-065-cross-source-dedup-v4-intake-report.v1",
        "status": status,
        "ready_for_global_dedup_object_comparison": status == READY,
        "blockers": sorted(set(blockers)),
        "bound_upstream_head_sha": upstream["head_sha"],
        "bound_upstream_workflow_run": upstream["workflow_run"],
        "validated_positive_credit_authority_count": len(required_by_worker),
        "validated_object_manifest_count": len(manifests),
        "validated_object_count": object_count,
        "validated_object_capacity_bytes": object_capacity,
        "post_dedup_capacity_claimed": False,
        "corpus_identity_claimed": False,
        "tokenizer_fit_authorized": False,
        "training_authorized": False,
    }


def require_v4_intake_ready(
    config: Mapping[str, Any],
    convergence: Mapping[str, Any],
) -> dict[str, Any]:
    report = evaluate_v4_intake(config, convergence)
    if not report["ready_for_global_dedup_object_comparison"]:
        raise CrossSourceV4IntakeError(
            f"V4 successor dedup intake blocked: {', '.join(report['blockers'])}"
        )
    return report
