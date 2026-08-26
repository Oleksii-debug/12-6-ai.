from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

STRATA = ("uk", "en", "code")
SCHEMA_VERSION = "12-6.research-corpus-v1-acquisition.v1"
SAFE_BASE_RESULT = "SOURCE_AUTHORITY_VECTOR_CONVERGED_FOR_NEXT_DEDUP_ITERATION"
HEX40_RE = re.compile(r"^[0-9a-f]{40}$")
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class AcquisitionContractError(ValueError):
    """Raised when Research Corpus V1 acquisition invariants are violated."""


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AcquisitionContractError(message)


def _require_hex(value: Any, pattern: re.Pattern[str], field: str) -> str:
    _require(isinstance(value, str) and pattern.fullmatch(value) is not None, f"{field}: invalid hash")
    return value


def _byte_vector(value: Any, field: str) -> dict[str, int]:
    _require(isinstance(value, Mapping), f"{field}: expected object")
    expected = {*STRATA, "total"}
    _require(set(value) == expected, f"{field}: expected keys {sorted(expected)}")
    out: dict[str, int] = {}
    for key in (*STRATA, "total"):
        item = value[key]
        _require(type(item) is int and item >= 0, f"{field}.{key}: expected non-negative integer")
        out[key] = item
    _require(out["total"] == sum(out[key] for key in STRATA), f"{field}: total arithmetic drift")
    return out


def compute_plan_identity(plan: Mapping[str, Any]) -> str:
    """Return a stable identity without relying on a self-referential stored digest."""
    identity_payload = {key: value for key, value in plan.items() if key != "plan_identity_sha256"}
    return hashlib.sha256(_canonical_json_bytes(identity_payload)).hexdigest()


def validate_terminal_authority(authority: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the minimum handoff contract required before source bytes receive credit."""
    _require(isinstance(authority, Mapping), "terminal_authority: expected object")
    _require(authority.get("status") == "ADMIT", "terminal_authority.status must be ADMIT")
    _require(authority.get("training_authorized") is True, "terminal authority must authorize training")
    _require(authority.get("evaluation_authorized") is False, "training authority cannot authorize evaluation")
    _require_hex(authority.get("exact_head_sha"), HEX40_RE, "terminal_authority.exact_head_sha")
    _require_hex(authority.get("authority_identity_sha256"), HEX64_RE, "terminal_authority.authority_identity_sha256")
    _require_hex(authority.get("capacity_ledger_sha256"), HEX64_RE, "terminal_authority.capacity_ledger_sha256")

    execution = authority.get("execution")
    _require(isinstance(execution, Mapping), "terminal_authority.execution is required")
    _require(execution.get("conclusion") == "success", "terminal authority execution must be terminal success")
    _require(type(execution.get("run_id")) is int and execution["run_id"] > 0, "terminal authority run_id is required")

    rights = authority.get("rights")
    _require(isinstance(rights, Mapping), "terminal_authority.rights is required")
    _require(rights.get("training_decision") == "ALLOW", "terminal authority rights must explicitly ALLOW training")
    _require(rights.get("provenance_review") == "PASS", "terminal authority provenance review must PASS")
    _require(bool(rights.get("evidence_reference")), "terminal authority rights evidence reference is required")

    stratum = authority.get("stratum")
    _require(stratum in STRATA, "terminal_authority.stratum is invalid")
    family_id = authority.get("family_id")
    _require(isinstance(family_id, str) and family_id.strip(), "terminal authority family_id is required")

    capacity_bytes = authority.get("capacity_bytes")
    _require(type(capacity_bytes) is int and capacity_bytes > 0, "terminal authority capacity_bytes must be positive")
    objects = authority.get("objects")
    _require(isinstance(objects, list) and objects, "terminal authority objects are required")
    object_ids: set[str] = set()
    object_bytes = 0
    for index, obj in enumerate(objects):
        _require(isinstance(obj, Mapping), f"terminal_authority.objects[{index}]: expected object")
        object_id = obj.get("object_id")
        _require(isinstance(object_id, str) and object_id, f"terminal_authority.objects[{index}].object_id is required")
        _require(object_id not in object_ids, f"terminal authority duplicate object_id: {object_id}")
        object_ids.add(object_id)
        _require_hex(obj.get("content_sha256"), HEX64_RE, f"terminal_authority.objects[{index}].content_sha256")
        eligible_bytes = obj.get("eligible_bytes")
        _require(type(eligible_bytes) is int and eligible_bytes > 0, f"terminal_authority.objects[{index}].eligible_bytes must be positive")
        object_bytes += eligible_bytes
    _require(object_bytes == capacity_bytes, "terminal authority object ledger does not equal capacity_bytes")

    return {"stratum": stratum, "family_id": family_id, "capacity_bytes": capacity_bytes}


def validate_plan(plan: Mapping[str, Any]) -> dict[str, Any]:
    _require(isinstance(plan, Mapping), "plan must be an object")
    _require(plan.get("schema_version") == SCHEMA_VERSION, "unsupported acquisition plan schema")
    _require(plan.get("local_free_only") is True, "acquisition planning must remain LOCAL_FREE")
    for field in ("model_training_executed", "optimizer_update_executed", "tokenizer_fit_executed", "paid_compute_used", "final_test_payload_read"):
        _require(plan.get(field) is False, f"{field} must be false")

    base = plan.get("base_authority")
    _require(isinstance(base, Mapping), "base_authority is required")
    _require_hex(base.get("head_sha"), HEX40_RE, "base_authority.head_sha")
    _require(base.get("safe_result") == SAFE_BASE_RESULT, "base authority is not the source-vector convergence boundary")
    _require(isinstance(base.get("config_path"), str) and base["config_path"], "base_authority.config_path is required")
    _require_hex(base.get("config_blob_sha1"), HEX40_RE, "base_authority.config_blob_sha1")

    targets = _byte_vector(plan.get("frozen_targets_bytes"), "frozen_targets_bytes")
    credited = _byte_vector(plan.get("credited_pre_successor_dedup_bytes"), "credited_pre_successor_dedup_bytes")
    declared_gap = _byte_vector(plan.get("remaining_gap_bytes"), "remaining_gap_bytes")
    expected_gap = {key: targets[key] - credited[key] for key in STRATA}
    _require(all(value >= 0 for value in expected_gap.values()), "credited bytes exceed frozen target")
    expected_gap["total"] = sum(expected_gap.values())
    _require(declared_gap == expected_gap, "remaining_gap_bytes does not match target minus credited vector")

    policy = plan.get("planning_policy")
    _require(isinstance(policy, Mapping), "planning_policy is required")
    survival_floor = policy.get("planning_survival_floor")
    _require(type(survival_floor) in {int, float} and 0 < float(survival_floor) <= 1, "planning_survival_floor must be in (0, 1]")
    survival_floor = float(survival_floor)
    _require(policy.get("survival_floor_is_evidence") is False, "planning survival floor cannot be represented as measured evidence")
    share_cap = policy.get("max_single_package_share_per_stratum")
    _require(type(share_cap) in {int, float} and 0 < float(share_cap) <= 1, "package share cap must be in (0, 1]")
    share_cap = float(share_cap)
    min_families = policy.get("minimum_planned_independent_families_per_stratum")
    _require(type(min_families) is int and min_families >= 2, "minimum planned family count must be >= 2")

    required_gross = {key: math.ceil(expected_gap[key] / survival_floor) for key in STRATA}
    required_gross["total"] = sum(required_gross.values())
    declared_required = _byte_vector(plan.get("buffered_gross_required_bytes"), "buffered_gross_required_bytes")
    _require(declared_required == required_gross, "buffered gross requirement arithmetic drift")

    packages = plan.get("work_packages")
    _require(isinstance(packages, list) and packages, "work_packages are required")
    package_ids: set[str] = set()
    gross_by_stratum: defaultdict[str, int] = defaultdict(int)
    families_by_stratum: defaultdict[str, int] = defaultdict(int)
    package_gross: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
    terminal_credit_by_stratum: defaultdict[str, int] = defaultdict(int)

    for index, package in enumerate(packages):
        _require(isinstance(package, Mapping), f"work_packages[{index}] must be an object")
        package_id = package.get("package_id")
        _require(isinstance(package_id, str) and package_id, f"work_packages[{index}].package_id is required")
        _require(package_id not in package_ids, f"duplicate package_id: {package_id}")
        package_ids.add(package_id)
        stratum = package.get("stratum")
        _require(stratum in STRATA, f"{package_id}: invalid stratum")
        stage = package.get("stage")
        _require(stage in {"PROSPECT", "QUALIFYING", "TERMINAL_ADMIT"}, f"{package_id}: invalid stage")
        planned_gross = package.get("planned_gross_bytes")
        _require(type(planned_gross) is int and planned_gross > 0, f"{package_id}: planned_gross_bytes must be positive")
        authority_credit = package.get("authority_credit_bytes")
        _require(type(authority_credit) is int and authority_credit >= 0, f"{package_id}: invalid authority_credit_bytes")
        family_budget = package.get("minimum_independent_families")
        _require(type(family_budget) is int and family_budget >= 1, f"{package_id}: minimum_independent_families must be positive")
        _require(package.get("evaluation_authorized") is False, f"{package_id}: evaluation leakage is forbidden")
        _require(package.get("public_availability_is_training_authority") is False, f"{package_id}: public availability cannot be training authority")
        _require(package.get("rights_state") in {"REVIEW_REQUIRED", "EVIDENCE_REQUIRED", "PASS"}, f"{package_id}: invalid rights_state")
        _require(package.get("provenance_state") in {"REVIEW_REQUIRED", "EVIDENCE_REQUIRED", "PASS"}, f"{package_id}: invalid provenance_state")

        if stage != "TERMINAL_ADMIT":
            _require(authority_credit == 0, f"{package_id}: nonterminal prospect cannot receive capacity credit")
            _require(package.get("terminal_authority") in {None, False}, f"{package_id}: nonterminal package cannot carry terminal authority")
        else:
            authority = package.get("terminal_authority")
            _require(isinstance(authority, Mapping), f"{package_id}: terminal_authority is required")
            terminal = validate_terminal_authority(authority)
            _require(terminal["stratum"] == stratum, f"{package_id}: terminal authority stratum mismatch")
            _require(terminal["capacity_bytes"] == authority_credit, f"{package_id}: terminal authority credit mismatch")
            _require(package.get("rights_state") == "PASS", f"{package_id}: terminal rights_state must PASS")
            _require(package.get("provenance_state") == "PASS", f"{package_id}: terminal provenance_state must PASS")
            terminal_credit_by_stratum[stratum] += authority_credit

        gross_by_stratum[stratum] += planned_gross
        families_by_stratum[stratum] += family_budget
        package_gross[stratum].append((package_id, planned_gross))

    for stratum in STRATA:
        _require(gross_by_stratum[stratum] >= required_gross[stratum], f"{stratum}: planned gross acquisition is below buffered requirement")
        _require(families_by_stratum[stratum] >= min_families, f"{stratum}: insufficient planned independent-family budget")
        for package_id, gross in package_gross[stratum]:
            share = gross / gross_by_stratum[stratum]
            _require(share <= share_cap + 1e-12, f"{package_id}: package concentration exceeds planner share cap")

    claim = plan.get("claim_boundary")
    _require(isinstance(claim, Mapping), "claim_boundary is required")
    for field in ("post_dedup_capacity_claimed", "research_corpus_v1_released", "tokenizer_fit_claimed", "learned_20m_checkpoint_claimed", "learned_100m_checkpoint_claimed"):
        _require(claim.get(field) is False, f"claim_boundary.{field} must be false")

    downstream = plan.get("downstream_gates")
    _require(isinstance(downstream, Mapping), "downstream_gates are required")
    for gate in ("GLOBAL_CROSS_SOURCE_DEDUP", "CORPUS_MATERIALIZATION", "DECONTAMINATION", "UNIQUE_LOSS_LEDGER", "TOKENIZER_FIT", "LEARNED_20M_CAMPAIGN"):
        _require(downstream.get(gate) in {"REQUIRED", "BLOCKED"}, f"downstream gate {gate} cannot be promoted by acquisition planning")

    identity = compute_plan_identity(plan)
    stored_identity = plan.get("plan_identity_sha256")
    if stored_identity is not None:
        _require_hex(stored_identity, HEX64_RE, "plan_identity_sha256")
        _require(stored_identity == identity, "stored plan identity does not match canonical plan content")

    planned = {key: gross_by_stratum[key] for key in STRATA}
    planned["total"] = sum(planned.values())
    headroom = {key: planned[key] - required_gross[key] for key in STRATA}
    headroom["total"] = sum(headroom.values())
    terminal_credit = {key: terminal_credit_by_stratum[key] for key in STRATA}
    terminal_credit["total"] = sum(terminal_credit.values())

    return {
        "status": "PASS_PLANNING_CONTRACT_ONLY",
        "plan_identity_sha256": identity,
        "remaining_gap_bytes": expected_gap,
        "buffered_gross_required_bytes": required_gross,
        "planned_gross_bytes": planned,
        "planning_headroom_bytes": headroom,
        "terminal_package_credit_bytes": terminal_credit,
        "package_count": len(packages),
        "planned_minimum_independent_families": {key: families_by_stratum[key] for key in STRATA},
        "next_gate": "SOURCE_ACQUISITION_AND_TERMINAL_AUTHORITY",
        "research_corpus_v1_released": False,
        "training_authorized_by_this_contract": False,
    }


def load_and_validate(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    return validate_plan(plan)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate the Research Corpus V1 bulk acquisition planning contract.")
    parser.add_argument("plan", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = load_and_validate(args.plan)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
