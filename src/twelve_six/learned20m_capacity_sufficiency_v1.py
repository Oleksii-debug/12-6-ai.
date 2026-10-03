"""Fail-closed learned-20M capacity snapshot validator."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

SCHEMA = "12-6.learned20m-capacity-sufficiency.v3"
MAIN = "bcc8f75090dd31a5e42a9045d96044d9721bebae"
REPORT_ID = "LEARNED20M-CAPACITY-SUFFICIENCY-20261003-SWARM2570-MAINBCC8F750-V3"
REPORT_SHA = "f82705d54692a9e8e6b4e7631eeac39adbf75a72bec600354204b9f7e0410121"
POLICY_SHA = "9a9242f47981c25e754fc95e2650050da4e4195aa1ef3a78f2c293f9e25d7ff7"
POLICY_BLOB = "b5a2577aeb1a2e56ebff1a4b46ac325d99dd8f8f"
EXECUTOR_HEAD = "2ad5b63bf7107465d6fa7deb25bf8c2f2fa03171"
EXECUTOR_MERGE = "bd2d445dfd8fbd7ec6759c1398bb913f4e0c0093"
CLEAN_ID = "7061d74db13bf45a9a7a1266ebe50feab8e7d22c32fba7a81dd91c2be4135ade"
CONTROL_ISSUES = [548, 723, 2011, 2021, 2570]
STRATA = {"ua": (9, 20), "en": (7, 20), "code": (1, 5)}
TRUTH = {
    "current_retained_corpus_launch_authoritative": False,
    "authorized_postpack_unique_loss_positions": 0,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates_executed_on_real_targets": 0,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights": False,
}
TOP = {
    "schema_version", "report_id", "source_main_sha", "claim_issue", "control_issues",
    "scientific_truth", "terminal_physical_clean_supply", "terminal_clean_post_qp_supply",
    "en_family_cap_necessary_condition", "balance_policy",
    "indexed_executor", "pending_existing_high_yield_work",
    "existing_independent_family_backlog", "downstream_authority_backlog",
    "optimistic_source_mixture_bound", "decision",
}
HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class CapacityReportError(ValueError):
    pass


def fail(message: str) -> None:
    raise CapacityReportError(message)


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{name} must be an integer, not bool")
    if value < minimum:
        fail(f"{name} must be >= {minimum}")
    return value


def boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        fail(f"{name} must be a JSON boolean")
    return value


def hexid(value: Any, name: str, pattern: re.Pattern[str]) -> str:
    if not isinstance(value, str) or pattern.fullmatch(value) is None:
        fail(f"{name} must be lowercase hex")
    return value


def json_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        fail(f"{name} must be a JSON object")
    return value


def object_list(value: Any, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        fail(f"{name} must be a JSON array of objects")
    return value


def fraction(value: Any, name: str, numerator: int, denominator: int) -> None:
    value = json_object(value, name)
    if set(value) != {"numerator", "denominator"}:
        fail(f"{name} keys mismatch")
    if integer(value["numerator"], f"{name}.numerator") != numerator:
        fail(f"{name} numerator drifted")
    if integer(value["denominator"], f"{name}.denominator", 1) != denominator:
        fail(f"{name} denominator drifted")


def canonical_sha(report: dict[str, Any]) -> str:
    data = json.dumps(
        report, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    return hashlib.sha256(data).hexdigest()


def _validate_clean_post_qp(report: dict[str, Any]) -> None:
    """Keep post-QP physical capacity separate from earlier pre-QP supply."""
    clean = json_object(report["terminal_physical_clean_supply"], "pre-QP")
    qp = json_object(report["terminal_clean_post_qp_supply"], "post-QP")
    expected = {
        "classification", "source_pr", "audit_issue", "integration_merge",
        "source_head", "run_id", "job_id", "artifact_id", "artifact_zip_sha256",
        "records", "distinct_source_ids", "payload_utf8_bytes",
        "records_jsonl_sha256", "record_inventory_sha256", "payload_inventory_sha256",
        "upstream_pre_qp_materialization_identity_sha256", "by_stratum_bytes",
        "postpack_unique_loss_positions", "capacity_credit_bytes",
        "launch_authoritative",
    }
    if set(qp) != expected:
        fail("post-QP evidence keys mismatch")
    if qp["classification"] != "PHYSICAL_TERMINAL_ZERO_CREDIT_POST_RESERVED_EVAL_QP_PRE_BALANCE":
        fail("post-QP physical evidence classification drifted")
    for key, value in {
        "source_pr": 2211, "audit_issue": 2455, "run_id": 36893337849,
        "job_id": 110485577178, "artifact_id": 11181030848,
        "records": 254, "distinct_source_ids": 241,
        "payload_utf8_bytes": 5_428_358,
    }.items():
        if integer(qp[key], "post-QP." + key, 1) != value:
            fail("post-QP immutable physical count or run drifted: " + key)
    for key, value in {
        "integration_merge": "dcd85658547818f0835c819e6b0ea8379eb9ca0a",
        "source_head": "d232e2003d18627534d5436a55ea919322c0a08f",
    }.items():
        if hexid(qp[key], "post-QP." + key, HEX40) != value:
            fail("post-QP immutable Git authority drifted: " + key)
    for key, value in {
        "artifact_zip_sha256": (
            "3eef740b52e9db962b01b63a0a48474f735694dca4c3848e3e9c53d8cd7f1988"
        ),
        "records_jsonl_sha256": (
            "3aa6235e07da4fa50c9642639cc0d93d7da22a6b602d822c7c7ca6e1f4a7486a"
        ),
        "record_inventory_sha256": (
            "0807f46418fb5a16a1c553c86a6f6967e5c2854d0016e3f6fe24c1d9c05c628b"
        ),
        "payload_inventory_sha256": (
            "2d2f0f5695ee07dfbba2e49b2d14983ede91051644770804fb734235125353af"
        ),
        "upstream_pre_qp_materialization_identity_sha256": CLEAN_ID,
    }.items():
        if hexid(qp[key], "post-QP." + key, HEX64) != value:
            fail("post-QP physical root drifted: " + key)
    if qp["upstream_pre_qp_materialization_identity_sha256"] != (
        clean["materialization_identity_sha256"]
    ):
        fail("post-QP physical evidence ancestry mismatch")
    strata = json_object(qp["by_stratum_bytes"], "post-QP strata")
    if set(strata) != set(STRATA):
        fail("post-QP stratum keys mismatch")
    for name, value in {"ua": 10_632, "en": 1_753_479, "code": 3_664_247}.items():
        if integer(strata[name], "post-QP." + name) != value:
            fail("post-QP stratum bytes drifted: " + name)
    if sum(strata.values()) != qp["payload_utf8_bytes"]:
        fail("post-QP stratum bytes do not sum to authenticated payload")
    if qp["postpack_unique_loss_positions"] is not None:
        fail("post-QP source bytes are not postpack loss positions")
    if integer(qp["capacity_credit_bytes"], "post-QP capacity credit"):
        fail("post-QP source bytes cannot grant canonical capacity")
    if boolean(qp["launch_authoritative"], "post-QP launch authority"):
        fail("post-QP physical evidence cannot authorize launch")

    condition = json_object(
        report["en_family_cap_necessary_condition"], "EN necessary condition"
    )
    if set(condition) != {
        "classification", "en_target_bytes", "en_clean_baseline_bytes",
        "en_gap_before_new_candidates_bytes",
        "caselaw_max_optimistic_new_clean_increment_bytes",
        "en_total_if_caselaw_reaches_full_family_cap_bytes",
        "minimum_other_independent_clean_en_bytes",
        "code_clean_gap_before_new_candidates_bytes",
        "candidate_bytes_are_not_observed_clean_survivors",
        "authoritative_capacity_credit_bytes", "proves_balance_feasibility",
    }:
        fail("EN necessary-condition keys mismatch")
    if condition["classification"] != "CLEAN_POST_QP_NECESSARY_CONDITION_ZERO_CREDIT":
        fail("EN necessary-condition stage drifted")
    policy = json_object(report["balance_policy"], "balance policy")
    target = integer(policy["target_total_source_bytes"], "policy target", 1)
    en_target = target * 7 // 20
    code_target = target // 5
    en_family_cap = min(target // 4, en_target * 3 // 5)
    en_baseline = strata["en"]
    remaining = en_target - en_baseline
    expected_values = {
        "en_target_bytes": en_target,
        "en_clean_baseline_bytes": en_baseline,
        "en_gap_before_new_candidates_bytes": remaining,
        "caselaw_max_optimistic_new_clean_increment_bytes": en_family_cap,
        "en_total_if_caselaw_reaches_full_family_cap_bytes": en_baseline + en_family_cap,
        "minimum_other_independent_clean_en_bytes": max(0, remaining - en_family_cap),
        "code_clean_gap_before_new_candidates_bytes": code_target - strata["code"],
    }
    for key, value in expected_values.items():
        if integer(condition[key], "EN condition." + key) != value:
            fail("EN family-cap necessary-condition arithmetic drifted: " + key)
    if not boolean(
        condition["candidate_bytes_are_not_observed_clean_survivors"],
        "EN condition no candidate-as-survivor",
    ):
        fail("EN candidate bytes cannot be counted as observed clean survivors")
    if integer(
        condition["authoritative_capacity_credit_bytes"], "EN condition credit"
    ):
        fail("EN necessary-condition arithmetic cannot grant capacity credit")
    if boolean(condition["proves_balance_feasibility"], "EN condition feasibility"):
        fail("EN necessary condition alone cannot prove balance feasibility")


def validate_report(report: dict[str, Any], *, expected_main_sha: str | None = None) -> None:
    report = json_object(report, "report")
    if set(report) != TOP:
        fail("report keys mismatch")
    if report["schema_version"] != SCHEMA or report["report_id"] != REPORT_ID:
        fail("report identity drifted")
    main = hexid(report["source_main_sha"], "source_main_sha", HEX40)
    if main != MAIN:
        fail("report source_main_sha drifted")
    if expected_main_sha is not None and main != expected_main_sha:
        fail(f"stale report root: {main} != {expected_main_sha}")
    if integer(report["claim_issue"], "claim_issue", 1) != 2570:
        fail("claim_issue drifted")
    if report["control_issues"] != CONTROL_ISSUES:
        fail("control_issues drifted")

    truth = report["scientific_truth"]
    if not isinstance(truth, dict) or set(truth) != set(TRUTH):
        fail("scientific_truth keys mismatch")
    for key, expected in TRUTH.items():
        actual = truth[key]
        if isinstance(expected, bool):
            boolean(actual, f"scientific_truth.{key}")
        else:
            integer(actual, f"scientific_truth.{key}")
        if actual != expected:
            fail(f"scientific_truth.{key} widens authority")

    clean = report["terminal_physical_clean_supply"]
    if clean["classification"] != "PHYSICAL_TERMINAL_ZERO_CREDIT_PRE_DOWNSTREAM_AUTHORITY":
        fail("terminal clean classification drifted")
    roots = (
        integer(clean["audit_issue"], "clean.audit", 1),
        integer(clean["execution_pr"], "clean.pr", 1),
    )
    if roots != (2174, 2153):
        fail("terminal clean authority root drifted")
    clean_head = hexid(clean["execution_head"], "clean.head", HEX40)
    if clean_head != "6af889c7c3d15d38f463791c9e2ba56c8e936e16":
        fail("terminal clean execution head drifted")
    physical = (
        integer(clean["run_id"], "clean.run", 1),
        integer(clean["artifact_id"], "clean.artifact", 1),
    )
    if physical != (36026689718, 10820342689):
        fail("terminal clean physical run drifted")
    if hexid(clean["materialization_identity_sha256"], "clean.identity", HEX64) != CLEAN_ID:
        fail("terminal clean materialization_identity_sha256 drifted")
    counts = (
        integer(clean["records"], "clean.records", 1),
        integer(clean["distinct_source_ids"], "clean.sources", 1),
    )
    if counts != (257, 244):
        fail("terminal clean counts drifted")
    clean_bytes = integer(clean["payload_utf8_bytes"], "clean.bytes", 1)
    if clean_bytes != 5_601_716:
        fail("terminal clean payload bytes drifted")
    if clean["postpack_unique_loss_positions"] is not None:
        fail("pre-downstream bytes cannot assert postpack unique-loss positions")
    if integer(clean["capacity_credit_bytes"], "clean.capacity_credit_bytes"):
        fail("pre-balance physical bytes cannot receive canonical capacity credit")
    if boolean(clean["launch_authoritative"], "clean.launch"):
        fail("terminal clean supply is not launch-authoritative")

    _validate_clean_post_qp(report)

    policy = json_object(report["balance_policy"], "balance_policy")
    if policy.get("source_path") != "configs/data/next100_106_balance_gate_policy_v1.json":
        fail("balance policy source path drifted")
    if hexid(policy["git_blob_sha"], "policy.blob", HEX40) != POLICY_BLOB:
        fail("balance policy blob drifted")
    if hexid(policy["policy_identity_sha256"], "policy.identity", HEX64) != POLICY_SHA:
        fail("balance policy identity drifted")
    if integer(policy["target_total_source_bytes"], "policy.target", 1) != 20_000_000:
        fail("balance target drifted")
    strata = json_object(policy["strata"], "policy.strata")
    if set(strata) != set(STRATA):
        fail("balance strata set drifted")
    for stratum, expected in STRATA.items():
        fraction(strata[stratum], f"policy.strata.{stratum}", *expected)
    fraction(policy["max_family_fraction_total"], "policy.max_family_total", 1, 4)
    fraction(policy["max_family_fraction_own_stratum"], "policy.max_family_stratum", 3, 5)
    if integer(policy["minimum_independent_families_per_stratum"], "policy.family_min", 1) != 2:
        fail("minimum independent-family rule drifted")
    if integer(policy["budget_quantum_bytes"], "policy.budget_quantum", 1) != 100:
        fail("balance budget quantum drifted")
    if boolean(policy["replay_or_duplication_to_meet_quota"], "policy.replay"):
        fail("replay/duplication cannot satisfy balance")
    if not boolean(policy["source_bytes_are_not_unique_loss_positions"], "policy.byte_rule"):
        fail("source bytes cannot be relabeled as unique loss positions")

    executor = report["indexed_executor"]
    if integer(executor["pr"], "executor.pr", 1) != 1459:
        fail("indexed executor PR drifted")
    if hexid(executor["head"], "executor.head", HEX40) != EXECUTOR_HEAD:
        fail("indexed executor head drifted")
    if hexid(executor["merge_commit"], "executor.merge", HEX40) != EXECUTOR_MERGE:
        fail("indexed executor merge drifted")
    if not boolean(executor["integrated"], "executor.integrated"):
        fail("indexed executor must remain integrated")
    if boolean(executor["defines_new_matcher_science"], "executor.new_science"):
        fail("indexed executor cannot claim new matcher science")
    if integer(executor["capacity_credit_bytes"], "executor.credit"):
        fail("execution mechanics cannot receive capacity credit")

    pending = object_list(report["pending_existing_high_yield_work"], "pending work")
    pending_ids = {"rada_two_clean", "franko1901_two_clean", "caselaw_two_clean"}
    if len(pending) != len(pending_ids) or {row["id"] for row in pending} != pending_ids:
        fail("pending high-yield work set mismatch")
    franko_bytes = 0
    for row in pending:
        ident = row["id"]
        if "balance_credit_bytes" in row:
            fail("legacy balance_credit_bytes field is forbidden")
        if row["post_global_dedup_survivor_bytes"] is not None:
            fail("pending execution cannot assert terminal survivor bytes")
        if integer(row["capacity_credit_bytes"], f"pending.{ident}.credit"):
            fail("pending lanes cannot receive canonical capacity credit")
        if boolean(row["launch_authoritative"], f"pending.{ident}.launch"):
            fail("pending lanes cannot become launch-authoritative")
        optimistic_bound = integer(
            row["optimistic_balance_upper_bound_bytes"],
            f"pending.{ident}.optimistic_bound",
        )
        if ident == "rada_two_clean" and optimistic_bound != 5_000_000:
            fail("Rada optimistic balance bound must equal the one-family global cap")
        if ident != "rada_two_clean" and optimistic_bound:
            fail("non-Rada pending lanes cannot claim an optimistic balance bound")
        if ident == "franko1901_two_clean":
            franko_bytes = integer(row["candidate_utf8_bytes"], "Franko candidate bytes", 1)
    if franko_bytes != 1_762_005:
        fail("Franko candidate evidence drifted")

    backlog = object_list(report["existing_independent_family_backlog"], "family backlog")
    family_ids = {"lesia1892", "edrnpa", "nbu"}
    if len(backlog) != len(family_ids) or {row["id"] for row in backlog} != family_ids:
        fail("independent-family backlog set mismatch")
    for row in backlog:
        hexid(row["observed_head"], "family observed_head", HEX40)
        if integer(row["capacity_credit_bytes"], "family capacity credit"):
            fail("unexecuted independent-family lanes cannot receive capacity credit")
    downstream = object_list(report["downstream_authority_backlog"], "downstream backlog")
    expected_downstream = {
        "clean_current_composition",
        "balance_execution",
        "packing",
        "postpack_unique_loss",
    }
    if (
        len(downstream) != len(expected_downstream)
        or {row["id"] for row in downstream} != expected_downstream
    ):
        fail("downstream authority backlog set mismatch")
    for row in downstream:
        actual = boolean(row["terminal_authority"], "downstream terminal")
        if actual != (row["id"] == "clean_current_composition"):
            fail("downstream terminal flag inconsistent with audited clean composition")

    bound = report["optimistic_source_mixture_bound"]
    expected_total = clean_bytes + 5_000_000 + franko_bytes
    if bound["classification"] != "ILLUSTRATIVE_UPPER_BOUND_ONLY_NOT_CAPACITY_CREDIT":
        fail("optimistic bound classification drifted")
    optimistic_total = integer(
        bound["optimistic_total_before_overlap_or_further_loss"],
        "optimistic total",
    )
    if optimistic_total != expected_total:
        fail("optimistic source-mixture arithmetic drifted")
    optimistic_deficit = integer(
        bound["deficit_to_20m_balance_target"],
        "optimistic deficit",
    )
    if optimistic_deficit != 20_000_000 - expected_total:
        fail("optimistic source-mixture deficit drifted")
    if integer(bound["authoritative_capacity_credit_bytes"], "optimistic credit"):
        fail("optimistic arithmetic drifted: authoritative capacity credit")
    if boolean(bound["proves_balance_feasibility"], "optimistic proves balance"):
        fail("optimistic bound cannot prove balance feasibility")
    if bound.get("source_stage") != "PRE_RESERVED_EVAL_QP_NOT_ADDITIVE_WITH_CLEAN_POST_QP":
        fail("optimistic bound source stage drifted")
    if integer(bound["terminal_clean_payload_bytes"], "pre-QP clean bytes") != clean_bytes:
        fail("optimistic bound incorrectly mixes clean stages")
    if boolean(bound["proves_unique_loss_sufficiency"], "optimistic proves unique loss"):
        fail("optimistic bound cannot prove unique-loss sufficiency")

    decision = report["decision"]
    if boolean(decision["open_new_distant_source_families"], "decision.open_distant"):
        fail("do not open distant source families while existing high-yield work remains")
    if boolean(decision["stop_new_source_acquisition_finish_pipeline"], "decision.stop"):
        fail("STOP is not authorized before terminal postpack unique-loss sufficiency")
    if integer(decision["meaningful_unique_loss_floor"], "decision.floor", 1) != 10_000_000:
        fail("meaningful unique-loss floor drifted")
    if integer(decision["authorized_postpack_unique_loss_positions"], "decision.authorized"):
        fail("current snapshot has zero authorized postpack unique-loss positions")
    if integer(decision["authorized_unique_loss_deficit"], "decision.deficit") != 10_000_000:
        fail("authorized unique-loss deficit drifted")
    if canonical_sha(report) != REPORT_SHA:
        fail("closed-world capacity report identity drifted")


def _duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            fail(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    fail(f"nonfinite JSON number: {value}")


def _float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        fail(f"nonfinite JSON number: {value}")
    return parsed


def load_and_validate(path: str | Path, *, expected_main_sha: str | None = None) -> dict[str, Any]:
    report = json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_duplicates,
        parse_constant=_nonfinite,
        parse_float=_float,
    )
    if not isinstance(report, dict):
        fail("report root must be an object")
    validate_report(report, expected_main_sha=expected_main_sha)
    return report
