#!/usr/bin/env python3
"""Fail-closed validator for the NEXT100-102 20M acquisition portfolio."""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

DEFAULT_PATH = Path("configs/data/next100_102_20m_acquisition_portfolio_v1.json")
SCHEMA = "12-6.next100-102-20m-acquisition-portfolio.v1"
WORKER = "NEXT100-102-20M-ACQUISITION-PORTFOLIO"
STRATA = ("uk", "en", "code")
EXPECTED_SOURCE_HEAD = "9ad8f74b12a2e991b7934356a88dd9a1f6ff3f41"
EXPECTED_CAPACITY = {"uk": 100856, "en": 144151, "code": 69133}
EXPECTED_TARGETS = {"uk": 9000000, "en": 7000000, "code": 4000000}
EXPECTED_BUDGETS = {"uk": 10000000, "en": 7200000, "code": 4200000}
REQUIRED_RESEARCH_EVIDENCE = {
    "UA-RADA-PORTAL-CCBY4",
    "UA-DATA-GOV-REUSE",
    "US-17-USC-105",
    "SQLALCHEMY-MIT",
}
REQUIRED_QUALIFICATION_STEPS = {
    "PIN_EXACT_UPSTREAM_OR_OFFICIAL_OBJECT_IDENTITY",
    "PIN_EXACT_RIGHTS_EVIDENCE_AND_PURPOSE_DECISION",
    "MATERIALIZE_EXACT_BYTES_AND_NORMALIZATION_IDENTITY",
    "RUN_QUALITY_LANGUAGE_PRIVACY_SECRET_FILTERS",
    "RUN_EXACT_NEAR_LINEAGE_AWARE_GLOBAL_DEDUP",
    "PROVE_SELECTION_FINAL_TEST_RESERVATION_SEPARATION_WITHOUT_READING_FINAL_OUTCOMES",
    "REQUIRE_TERMINAL_SOURCE_AUTHORITY_BEFORE_ANY_CAPACITY_CREDIT",
    "RERUN_SUCCESSOR_GLOBAL_DEDUP_BEFORE_BALANCE_OR_CORPUS_PROMOTION",
}
EXPECTED_DOWNSTREAM_ORDER = [
    "SOURCE_QUALIFICATION",
    "SOURCE_REGISTRY_CONVERGENCE",
    "GLOBAL_CROSS_SOURCE_DEDUP",
    "BALANCE_DIVERSITY_RETEST",
    "IMMUTABLE_PRE_DECONTAMINATION_CORPUS_IDENTITY",
    "EVALUATION_DECONTAMINATION",
    "QUALITY_PRIVACY_SPLIT_TWO_CLEAN_BUILD",
    "UNIQUE_POSTPACK_LOSS_LEDGER",
    "TOKENIZER_FIT_AUTHORIZATION",
    "MODEL341_20M_TRAINING_AUTHORIZATION",
]


class ValidationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _positive_int(value: Any, name: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0, f"{name} must be a positive integer")
    return value


def _sum_strata(mapping: dict[str, Any]) -> int:
    return sum(int(mapping[key]) for key in STRATA)


def _fraction(value: Any, name: str) -> Fraction:
    _require(isinstance(value, str), f"{name} must be a fraction string")
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValidationError(f"invalid fraction for {name}: {value}") from exc
    _require(0 < result <= 1, f"{name} must be within (0, 1]")
    return result


def validate(data: dict[str, Any]) -> dict[str, Any]:
    _require(data.get("schema_version") == SCHEMA, "schema mismatch")
    _require(data.get("worker_id") == WORKER, "worker mismatch")
    _require(data.get("issue") == 551, "issue binding changed")
    _require(data.get("execution_class") == "LOCAL_FREE", "execution class must stay LOCAL_FREE")
    for key, expected in (
        ("model_training_executed", False),
        ("tokenizer_fit_executed", False),
        ("paid_compute_used", False),
        ("final_test_payload_read", False),
    ):
        _require(data.get(key) is expected, f"unsafe boundary changed: {key}")
    _require(data.get("optimizer_steps_executed") == 0, "optimizer steps must remain zero")

    planning = data.get("planning_input")
    _require(isinstance(planning, dict), "planning_input missing")
    _require(planning.get("source_convergence_pr") == 527, "source convergence PR binding changed")
    _require(planning.get("source_convergence_head_sha") == EXPECTED_SOURCE_HEAD, "source convergence head changed")
    _require(
        planning.get("source_convergence_status_at_claim") == "OPEN_NONTERMINAL_INPUT_DO_NOT_PROMOTE",
        "nonterminal source input was promoted",
    )
    _require(planning.get("capacity_stage") == "PRE_SUCCESSOR_GLOBAL_DEDUP", "capacity stage changed")
    capacity = planning.get("capacity_bytes")
    _require(isinstance(capacity, dict), "capacity vector missing")
    _require({key: capacity.get(key) for key in STRATA} == EXPECTED_CAPACITY, "capacity vector changed")
    _require(capacity.get("total") == _sum_strata(capacity) == 314140, "capacity total mismatch")

    policy = data.get("frozen_policy")
    _require(isinstance(policy, dict), "frozen policy missing")
    _require(policy.get("authority") == "DATA-295-BALANCE-POLICY-20M-V1", "balance authority changed")
    targets = policy.get("target_bytes")
    _require(isinstance(targets, dict), "target bytes missing")
    _require({key: targets.get(key) for key in STRATA} == EXPECTED_TARGETS, "target vector changed")
    _require(targets.get("total") == _sum_strata(targets) == policy.get("target_total_source_bytes") == 20000000, "target total mismatch")
    global_fraction = _fraction(policy.get("max_family_fraction_total"), "max_family_fraction_total")
    own_fraction = _fraction(policy.get("max_family_fraction_own_stratum"), "max_family_fraction_own_stratum")
    _require(global_fraction == Fraction(1, 4), "global family cap changed")
    _require(own_fraction == Fraction(3, 5), "own-stratum family cap changed")
    _require(policy.get("minimum_independent_families_per_stratum") == 2, "hard family minimum changed")
    _require(policy.get("replay_or_duplication_to_meet_quota") is False, "replay was enabled")
    _require(policy.get("model_metric_guided_mixture_retuning") is False, "model-guided mixture retuning was enabled")

    geometry = data.get("derived_acquisition_geometry")
    _require(isinstance(geometry, dict), "derived geometry missing")
    gaps = geometry.get("remaining_gap_bytes")
    caps = geometry.get("max_bytes_per_single_family_at_target")
    minima = geometry.get("minimum_new_families_if_all_gap_bytes_come_from_new_families")
    _require(isinstance(gaps, dict) and isinstance(caps, dict) and isinstance(minima, dict), "derived vectors missing")
    expected_gaps: dict[str, int] = {}
    expected_caps: dict[str, int] = {}
    expected_minima: dict[str, int] = {}
    global_cap = int(Fraction(20000000) * global_fraction)
    for stratum in STRATA:
        expected_gaps[stratum] = EXPECTED_TARGETS[stratum] - EXPECTED_CAPACITY[stratum]
        own_cap = int(Fraction(EXPECTED_TARGETS[stratum]) * own_fraction)
        expected_caps[stratum] = min(global_cap, own_cap)
        expected_minima[stratum] = (expected_gaps[stratum] + expected_caps[stratum] - 1) // expected_caps[stratum]
    _require({key: gaps.get(key) for key in STRATA} == expected_gaps, "gap arithmetic mismatch")
    _require(gaps.get("total") == sum(expected_gaps.values()) == 19685860, "gap total mismatch")
    _require(caps == expected_caps, "single-family cap arithmetic mismatch")
    _require(minima == expected_minima == {"uk": 2, "en": 2, "code": 2}, "minimum new-family geometry mismatch")
    _require("zero credited capacity" in geometry.get("note", ""), "zero-credit truth boundary missing")

    lanes = data.get("portfolio_lanes")
    _require(isinstance(lanes, list) and lanes, "portfolio lanes missing")
    lane_ids: set[str] = set()
    budget_by_stratum = {key: 0 for key in STRATA}
    family_slots_by_stratum = {key: 0 for key in STRATA}
    for row in lanes:
        _require(isinstance(row, dict), "lane must be an object")
        lane_id = row.get("lane_id")
        _require(isinstance(lane_id, str) and lane_id and lane_id not in lane_ids, "lane ids must be unique")
        lane_ids.add(lane_id)
        stratum = row.get("stratum")
        _require(stratum in STRATA, f"{lane_id}: invalid stratum")
        budget = _positive_int(row.get("planning_budget_bytes"), f"{lane_id}.planning_budget_bytes")
        families = _positive_int(row.get("minimum_distinct_families"), f"{lane_id}.minimum_distinct_families")
        single = row.get("single_family_lane")
        _require(isinstance(single, bool), f"{lane_id}: single_family_lane must be bool")
        if single:
            _require(families == 1, f"{lane_id}: single-family lane must require exactly one family")
            _require(budget <= expected_caps[stratum], f"{lane_id}: single-family planning budget exceeds family cap")
        else:
            _require(families >= 2, f"{lane_id}: multi-family lane must require at least two families")
        _require(row.get("candidate_authority_status") == "RESEARCH_ONLY_NOT_ADMITTED", f"{lane_id}: candidate was promoted")
        _require(row.get("capacity_credit_now_bytes") == 0, f"{lane_id}: unqualified capacity received credit")
        _require(isinstance(row.get("research_basis"), str) and row["research_basis"].strip(), f"{lane_id}: research basis missing")
        filters = row.get("hard_filters")
        _require(isinstance(filters, list) and "GLOBAL_DEDUP" in filters and "EVAL_SEPARATION" in filters, f"{lane_id}: mandatory dedup/eval filters missing")
        budget_by_stratum[stratum] += budget
        family_slots_by_stratum[stratum] += families

    summary = data.get("portfolio_budget_summary")
    _require(isinstance(summary, dict), "portfolio summary missing")
    summary_budget = summary.get("planning_budget_bytes")
    _require(isinstance(summary_budget, dict), "portfolio budget vector missing")
    _require(budget_by_stratum == EXPECTED_BUDGETS, "lane budget vector changed")
    _require({key: summary_budget.get(key) for key in STRATA} == budget_by_stratum, "summary budget mismatch")
    _require(summary_budget.get("total") == sum(budget_by_stratum.values()) == 21400000, "summary total mismatch")
    for stratum in STRATA:
        _require(budget_by_stratum[stratum] >= expected_gaps[stratum], f"{stratum}: planning budget does not cover current gap")
        _require(summary.get("budget_exceeds_current_gap", {}).get(stratum) is True, f"{stratum}: gap coverage flag must be true")
        _require(family_slots_by_stratum[stratum] >= expected_minima[stratum], f"{stratum}: insufficient planned family topology")
    _require(summary.get("credited_capacity_from_portfolio_now_bytes") == 0, "portfolio received premature capacity credit")

    evidence = data.get("research_evidence")
    _require(isinstance(evidence, list), "research evidence missing")
    evidence_ids = set()
    for row in evidence:
        _require(isinstance(row, dict), "research evidence row must be object")
        evidence_id = row.get("evidence_id")
        _require(isinstance(evidence_id, str) and evidence_id not in evidence_ids, "research evidence ids must be unique")
        evidence_ids.add(evidence_id)
        _require(str(row.get("authority_effect", "")).startswith("RESEARCH_ONLY_"), f"{evidence_id}: research evidence became authority")
        _require(str(row.get("url", "")).startswith("https://"), f"{evidence_id}: official evidence URL missing")
    _require(evidence_ids == REQUIRED_RESEARCH_EVIDENCE, "research evidence set changed")

    contract = data.get("qualification_contract")
    _require(isinstance(contract, list) and set(contract) == REQUIRED_QUALIFICATION_STEPS, "qualification contract changed")
    _require(data.get("downstream_order") == EXPECTED_DOWNSTREAM_ORDER, "downstream gate order changed")

    boundary = data.get("claim_boundary")
    _require(isinstance(boundary, dict), "claim boundary missing")
    for key in (
        "candidate_source_admitted",
        "post_dedup_capacity_claimed",
        "research_corpus_v1_released",
        "tokenizer_fit_authorized",
        "long_training_authorized",
        "learned_20m_checkpoint_claimed",
        "learned_100m_checkpoint_claimed",
    ):
        _require(boundary.get(key) is False, f"premature claim: {key}")
    _require(boundary.get("safe_result") == "ACQUISITION_PORTFOLIO_READY_FOR_PARALLEL_SOURCE_QUALIFICATION", "safe result changed")

    return {
        "status": "PASS",
        "current_pre_dedup_capacity_bytes": {**EXPECTED_CAPACITY, "total": sum(EXPECTED_CAPACITY.values())},
        "remaining_gap_bytes": {**expected_gaps, "total": sum(expected_gaps.values())},
        "planning_budget_bytes": {**budget_by_stratum, "total": sum(budget_by_stratum.values())},
        "planned_minimum_family_slots": family_slots_by_stratum,
        "credited_capacity_from_portfolio_now_bytes": 0,
        "next_action": "PARALLEL_EXACT_SOURCE_QUALIFICATION_WITH_ZERO_PREMATURE_CREDIT",
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else DEFAULT_PATH
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        report = validate(data)
    except (OSError, json.JSONDecodeError, ValidationError, KeyError, TypeError, ValueError) as exc:
        print(f"NEXT100-102 FAIL: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
