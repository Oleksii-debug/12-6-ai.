"""Fail-closed validation for the Research Corpus V1 scalable acquisition plan.

Planning requests are deliberately not corpus capacity.  This module exists to
make that firewall machine-checkable while the project expands legal, diverse,
reproducible source coverage toward the 20 MB balanced target.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

SCHEMA = "12-6.research-corpus-v1-scalable-acquisition-plan.v1"
STRATA = ("ua", "en", "code")
ALLOWED_CANDIDATE_STATUSES = {
    "REQUIRES_DEDICATED_ADMISSION",
    "RIGHTS_AND_AUTHORSHIP_RESEARCH_REQUIRED",
    "WORK_LEVEL_PUBLIC_DOMAIN_REVIEW_REQUIRED",
    "REUSE_EXISTING_TERMINAL_RIGHTS_BUT_NEW_BYTES_REQUIRE_ADMISSION",
    "RIGHTS_RESEARCH_REQUIRED",
}


class CorpusAcquisitionPlanError(RuntimeError):
    """Raised when a planning artifact crosses the training-capacity firewall."""


def _positive_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise CorpusAcquisitionPlanError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise CorpusAcquisitionPlanError(f"{field} must be a non-negative integer")
    return value


def _validate_stratum_vector(
    value: Any,
    *,
    field: str,
    positive: bool,
) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise CorpusAcquisitionPlanError(f"{field} must be a mapping")
    expected = {*STRATA, "total"}
    if set(value) != expected:
        raise CorpusAcquisitionPlanError(f"{field} must contain exactly {sorted(expected)}")
    parser = _positive_int if positive else _nonnegative_int
    parsed = {key: parser(value[key], field=f"{field}.{key}") for key in expected}
    if parsed["total"] != sum(parsed[key] for key in STRATA):
        raise CorpusAcquisitionPlanError(f"{field}.total does not equal the stratum sum")
    return parsed


def validate_acquisition_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one acquisition plan without granting any training capacity."""

    if plan.get("schema_version") != SCHEMA:
        raise CorpusAcquisitionPlanError("unsupported acquisition-plan schema")
    if plan.get("local_free_only") is not True:
        raise CorpusAcquisitionPlanError("acquisition plan must be LOCAL_FREE only")
    if plan.get("model_training_executed") is not False:
        raise CorpusAcquisitionPlanError("acquisition planning must not execute model training")

    baseline = plan.get("baseline")
    target = plan.get("target")
    firewall = plan.get("capacity_firewall")
    if not isinstance(baseline, Mapping) or not isinstance(target, Mapping):
        raise CorpusAcquisitionPlanError("baseline and target must be mappings")
    if not isinstance(firewall, Mapping):
        raise CorpusAcquisitionPlanError("capacity_firewall must be a mapping")

    observed = _validate_stratum_vector(
        baseline.get("observed_pre_dedup_bytes"),
        field="baseline.observed_pre_dedup_bytes",
        positive=True,
    )
    targets = _validate_stratum_vector(target.get("bytes"), field="target.bytes", positive=True)
    gaps = _validate_stratum_vector(
        target.get("planning_gap_from_observed_pre_dedup"),
        field="target.planning_gap_from_observed_pre_dedup",
        positive=True,
    )
    for stratum in STRATA:
        expected_gap = targets[stratum] - observed[stratum]
        if expected_gap <= 0 or gaps[stratum] != expected_gap:
            raise CorpusAcquisitionPlanError(
                f"planning gap for {stratum} must equal target minus observed pre-dedup bytes"
            )
    if gaps["total"] != targets["total"] - observed["total"]:
        raise CorpusAcquisitionPlanError("total planning gap is inconsistent")

    if _nonnegative_int(
        baseline.get("training_authorized_bytes"),
        field="baseline.training_authorized_bytes",
    ) != 0:
        raise CorpusAcquisitionPlanError("candidate baseline cannot authorize training bytes")
    if firewall.get("candidate_planning_bytes_are_capacity") is not False:
        raise CorpusAcquisitionPlanError("planning bytes must never be treated as capacity")
    if firewall.get("observed_pre_dedup_bytes_are_training_authorized") is not False:
        raise CorpusAcquisitionPlanError("pre-dedup bytes cannot be training-authorized")
    if _nonnegative_int(
        firewall.get("current_training_authorized_bytes"),
        field="capacity_firewall.current_training_authorized_bytes",
    ) != 0:
        raise CorpusAcquisitionPlanError("current training-authorized bytes must remain zero")
    if firewall.get("long_training_authorized") is not False:
        raise CorpusAcquisitionPlanError("long training must remain blocked at planning stage")

    rows = plan.get("candidate_streams")
    if not isinstance(rows, list) or not rows:
        raise CorpusAcquisitionPlanError("candidate_streams must be a non-empty list")
    candidate_ids: set[str] = set()
    planning_by_stratum: Counter[str] = Counter()
    families_by_stratum: dict[str, set[str]] = {stratum: set() for stratum in STRATA}
    for row in rows:
        if not isinstance(row, Mapping):
            raise CorpusAcquisitionPlanError("candidate stream must be an object")
        candidate_id = row.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in candidate_ids:
            raise CorpusAcquisitionPlanError("candidate_id must be non-empty and unique")
        candidate_ids.add(candidate_id)
        stratum = row.get("stratum")
        if stratum not in STRATA:
            raise CorpusAcquisitionPlanError(f"{candidate_id}: invalid stratum")
        status = row.get("status")
        if status not in ALLOWED_CANDIDATE_STATUSES:
            raise CorpusAcquisitionPlanError(f"{candidate_id}: unsupported nonterminal status")
        family = row.get("family_candidate")
        if not isinstance(family, str) or not family:
            raise CorpusAcquisitionPlanError(f"{candidate_id}: family_candidate is required")
        request = _positive_int(
            row.get("planning_request_bytes"), field=f"{candidate_id}.planning_request_bytes"
        )
        credited = _nonnegative_int(row.get("credited_bytes"), field=f"{candidate_id}.credited_bytes")
        if credited != 0:
            raise CorpusAcquisitionPlanError(
                f"{candidate_id}: candidate planning stream cannot credit corpus capacity"
            )
        for field in ("source_scope", "rights_anchor", "rights_basis_candidate"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise CorpusAcquisitionPlanError(f"{candidate_id}: {field} is required")
        retests = row.get("required_retests")
        if not isinstance(retests, list) or len(retests) < 3:
            raise CorpusAcquisitionPlanError(
                f"{candidate_id}: at least three explicit admission retests are required"
            )
        if any(not isinstance(item, str) or not item.strip() for item in retests):
            raise CorpusAcquisitionPlanError(f"{candidate_id}: invalid required_retests entry")
        planning_by_stratum[stratum] += request
        families_by_stratum[stratum].add(family)

    for stratum in STRATA:
        if planning_by_stratum[stratum] < gaps[stratum]:
            raise CorpusAcquisitionPlanError(
                f"{stratum}: planning requests do not cover the current planning gap"
            )
        if len(families_by_stratum[stratum]) < 3:
            raise CorpusAcquisitionPlanError(
                f"{stratum}: plan must span at least three candidate families"
            )

    gates = plan.get("mandatory_admission_gates")
    required_gates = {
        "source_or_object_specific_training_rights_basis",
        "cross_source_exact_and_near_duplicate_audit",
        "train_selection_final_eval_decontamination",
        "two_clean_build_determinism",
        "unique_nonignored_causal_position_ledger",
        "dedicated_exact_head_workflow_success",
    }
    if not isinstance(gates, list) or not required_gates.issubset(set(gates)):
        raise CorpusAcquisitionPlanError("mandatory admission gates are incomplete")

    execution_order = plan.get("next_execution_order")
    if not isinstance(execution_order, list) or len(execution_order) < 5:
        raise CorpusAcquisitionPlanError("next_execution_order is incomplete")

    return {
        "schema_version": SCHEMA,
        "status": "VALID_PLANNING_ARTIFACT_CAPACITY_ZERO",
        "observed_pre_dedup_bytes": observed,
        "target_bytes": targets,
        "planning_gap_bytes": gaps,
        "planning_request_bytes": {
            **{stratum: planning_by_stratum[stratum] for stratum in STRATA},
            "total": sum(planning_by_stratum.values()),
        },
        "candidate_family_counts": {
            **{stratum: len(families_by_stratum[stratum]) for stratum in STRATA},
            "total": len({family for families in families_by_stratum.values() for family in families}),
        },
        "training_authorized_bytes": 0,
        "long_training_authorized": False,
    }


def load_and_validate_acquisition_plan(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8")
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CorpusAcquisitionPlanError("acquisition plan is not valid JSON") from exc
    if not isinstance(plan, Mapping):
        raise CorpusAcquisitionPlanError("acquisition plan root must be an object")
    return validate_acquisition_plan(plan)
