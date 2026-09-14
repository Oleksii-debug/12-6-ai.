"""Fail-closed learned-20M capacity snapshot validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "12-6.learned20m-capacity-sufficiency.v1"
CONTINUE = "CONTINUE_ONLY_NAMED_HIGH_YIELD_EXECUTIONS"
STOP = "STOP_NEW_SOURCE_ACQUISITION_FINISH_PIPELINE"
MAIN = "fc57531f7f8e7ef03323df641d8196af6c573c8b"
DEDUP_HEAD = "e9aeea146325e80bbbbb4e01580e5c7893a6c208"

TRUTH = {
    "current_retained_corpus_launch_authoritative": False,
    "authorized_optimized_target_exposure": 0,
    "tokenizer_fit_authorized": False,
    "optimizer_updates_executed_on_real_targets": 0,
    "training_executed": False,
    "learned_weights_created": False,
    "final_test_outcomes_read": False,
    "paid_compute_used": False,
    "foreign_pretrained_weights": False,
    "clean_successor_external_llm_free_authorized": False,
}
CANDIDATES = {
    "caselaw": (1917, "bb1350d8821af2bebe244f8d2f32cb8ee215c9fd", 5658, 5962147),
    "pep_loc": (1345, "62be1740045ed4a0bffec3647f2658d4f856333a", 114, 9487683),
    "ubuntu_irc": (1791, "1748c728c7d22f87fd25140b8e27c2e8c17794bc", 972, 4799981),
    "arxiv_languk": (1851, "13afb887a3c42519d8b6ba2d71b8949ac5f87d10", 1280, 3949184),
}
CLASSES = {
    "caselaw": "PHYSICAL_ZERO_CREDIT_SOURCE_ADMITTED",
    "pep_loc": "PHYSICAL_ZERO_CREDIT_COMPOSED_CANDIDATE",
    "ubuntu_irc": "PHYSICAL_ZERO_CREDIT_SOURCE_ADMITTED",
    "arxiv_languk": "PHYSICAL_ZERO_CREDIT_REMATERIALIZATION_CANDIDATE",
}


class CapacityReportError(ValueError):
    pass


def fail(message: str) -> None:
    raise CapacityReportError(message)


def obj(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail(f"{name} must be an object")
    return value


def integer(value: Any, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        fail(f"{name} must be an integer, not bool")
    if value < minimum:
        fail(f"{name} must be >= {minimum}")
    return value


def boolean(value: Any, name: str) -> bool:
    if type(value) is not bool:
        fail(f"{name} must be a JSON boolean")
    return value


def hexid(value: Any, name: str, length: int) -> str:
    if not isinstance(value, str) or len(value) != length:
        fail(f"{name} must be lowercase {length}-hex")
    if value.lower() != value or any(c not in "0123456789abcdef" for c in value):
        fail(f"{name} must be lowercase {length}-hex")
    return value


def exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(value) != expected:
        fail(f"{name} keys mismatch")


def validate_report(report: Mapping[str, Any], *, expected_main_sha: str | None = None) -> None:
    report = obj(report, "report")
    exact_keys(report, {
        "schema_version", "report_id", "source_main_sha", "claim_issue", "control_issues",
        "scientific_truth", "floor", "clean_base_oracle", "invalidated_physical_baseline",
        "candidates", "capacity_math", "decision", "evidence_roots",
    }, "report")
    if report["schema_version"] != SCHEMA:
        fail("unsupported schema_version")
    if report["report_id"] != "LEARNED20M-CAPACITY-SUFFICIENCY-20260914-SWARM2048-V1":
        fail("unexpected report_id")
    root = hexid(report["source_main_sha"], "source_main_sha", 40)
    if expected_main_sha is not None and root != expected_main_sha:
        fail(f"stale report root: {root} != {expected_main_sha}")
    if integer(report["claim_issue"], "claim_issue", 1) != 2048:
        fail("claim_issue drifted")
    if report["control_issues"] != [548, 723, 2011, 2021]:
        fail("control_issues drifted")

    truth = obj(report["scientific_truth"], "scientific_truth")
    exact_keys(truth, set(TRUTH), "scientific_truth")
    for key, expected in TRUTH.items():
        (boolean if isinstance(expected, bool) else integer)(truth[key], f"truth.{key}")
        if truth[key] != expected:
            fail(f"scientific_truth.{key} widens authority")

    floor = obj(report["floor"], "floor")
    integer(floor.get("meaningful_unique_loss_positions"), "floor.meaningful", 1)
    integer(floor.get("requested_unique_loss_positions"), "floor.requested", 1)
    boolean(floor.get("source_bytes_are_not_unique_loss_positions"), "floor.byte_rule")
    if floor != {
        "unit": "unique_causal_loss_positions",
        "meaningful_unique_loss_positions": 10000000,
        "requested_unique_loss_positions": 20000000,
        "source_bytes_are_not_unique_loss_positions": True,
        "authority_issue": 548,
    }:
        fail("LEARN-345 floor authority drifted")

    oracle = obj(report["clean_base_oracle"], "clean_base_oracle")
    if oracle.get("authority_class") != "NONAUTHORITATIVE_PHYSICAL_DERIVATIVE_ORACLE":
        fail("clean_base_oracle cannot be promoted to a terminal corpus authority")
    if (oracle.get("issue"), oracle.get("status")) != (
        1881, "PREQUAL_INVENTORY_ORACLE_READY_RAW_JSONL_ORACLE_BLOCKED"
    ):
        fail("clean_base_oracle terminal authority mismatch")
    oracle_identity = (
        integer(oracle.get("run_id"), "oracle.run"),
        integer(oracle.get("artifact_id"), "oracle.artifact"),
    )
    if oracle_identity != (34783440822, 10325234135):
        fail("clean_base_oracle physical identity mismatch")
    if hexid(oracle.get("artifact_zip_sha256"), "oracle.zip", 64) != (
        "b65df0b4b508e2fc46679043c50426756d1d6efef8e8aaaea47010c50ef38793"
    ):
        fail("clean_base_oracle artifact digest mismatch")
    counts = tuple(integer(oracle.get(k), f"oracle.{k}") for k in (
        "records", "payload_bytes", "families", "source_objects"
    ))
    if counts != (254, 5428358, 18, 241):
        fail("clean_base_oracle physical accounting mismatch")
    if hexid(oracle.get("record_inventory_digest_sha256"), "oracle.record_root", 64) != (
        "0807f46418fb5a16a1c553c86a6f6967e5c2854d0016e3f6fe24c1d9c05c628b"
    ):
        fail("clean_base_oracle record root mismatch")
    if hexid(oracle.get("payload_inventory_digest_sha256"), "oracle.payload_root", 64) != (
        "2d2f0f5695ee07dfbba2e49b2d14983ede91051644770804fb734235125353af"
    ):
        fail("clean_base_oracle payload root mismatch")
    if boolean(oracle.get("raw_jsonl_identity_available"), "oracle.raw_jsonl"):
        fail("#1881 explicitly blocks a raw JSONL successor identity")
    if integer(oracle.get("canonical_unique_loss_positions_credited"), "oracle.credit"):
        fail("the derivative oracle cannot receive canonical unique-loss credit")
    if boolean(oracle.get("launch_authoritative"), "oracle.launch"):
        fail("the derivative oracle cannot become launch-authoritative")

    baseline = obj(report["invalidated_physical_baseline"], "invalidated_physical_baseline")
    if baseline.get("status") != "INVALIDATED_NOMIS1864_SONNET_PROVENANCE":
        fail("invalidated physical baseline identity or truth drifted")
    if tuple(integer(baseline.get(k), f"baseline.{k}") for k in (
        "records", "payload_bytes", "source_objects"
    )) != (255, 5430017, 242):
        fail("invalidated physical baseline identity or truth drifted")
    roots = (
        (
            "record_inventory_digest_sha256",
            "7f936f5518db1087578ef25a964e50796ff4572c0c9b666b83df78673af3ff4a",
        ),
        (
            "payload_inventory_digest_sha256",
            "4ff5424fe95d335459f0d58a818859f89106683768decd1ed291f16f7b0012b3",
        ),
        (
            "retained_jsonl_sha256",
            "b1ec0433fbd9675645b7e29c1e32b406e638fa081868cad7a56afec8f9a601cc",
        ),
    )
    for key, expected in roots:
        if hexid(baseline.get(key), f"baseline.{key}", 64) != expected:
            fail("invalidated physical baseline identity or truth drifted")
    if baseline.get("contamination") != "NOMIS1864_SONNET_EXTERNAL_LLM_PROVENANCE":
        fail("invalidated physical baseline identity or truth drifted")
    if integer(baseline.get("canonical_unique_loss_positions_credited"), "baseline.credit"):
        fail("invalidated physical bytes cannot receive canonical capacity credit")
    if boolean(baseline.get("launch_authoritative"), "baseline.launch"):
        fail("invalidated physical baseline cannot become launch-authoritative")

    candidates = report["candidates"]
    if not isinstance(candidates, list) or len(candidates) != 4:
        fail("candidates must contain the four named physical zero-credit inputs")
    seen, raw_bytes = set(), 0
    for i, candidate in enumerate(candidates):
        candidate = obj(candidate, f"candidates[{i}]")
        ident = candidate.get("id")
        if ident in seen or ident not in CANDIDATES:
            fail(f"unexpected or duplicate candidate id: {ident!r}")
        seen.add(ident)
        pr, head, records, byte_count = CANDIDATES[ident]
        actual = (
            integer(candidate.get("source_pr"), f"candidate[{i}].pr", 1),
            hexid(candidate.get("exact_head"), f"candidate[{i}].head", 40),
            integer(candidate.get("records"), f"candidate[{i}].records", 1),
            integer(candidate.get("physical_candidate_bytes"), f"candidate[{i}].bytes", 1),
        )
        if actual != (pr, head, records, byte_count):
            fail(f"candidates[{i}] evidence root or physical accounting drifted")
        if candidate.get("authority_class") != CLASSES[ident]:
            fail(f"candidates[{i}].authority_class drifted")
        if boolean(candidate.get("global_dedup_executed"), f"candidate[{i}].dedup"):
            fail("this report cannot promote a named candidate to executed global dedup")
        credited = integer(
            candidate.get("canonical_unique_loss_positions_credited"),
            f"candidate[{i}].credit",
        )
        if credited:
            fail("zero-credit candidates cannot receive unique-loss credit")
        if boolean(candidate.get("launch_authoritative"), f"candidate[{i}].launch"):
            fail("named pending candidates cannot become launch-authoritative")
        if not isinstance(candidate.get("blocker"), str) or not candidate["blocker"]:
            fail(f"candidates[{i}].blocker must be non-empty")
        raw_bytes += byte_count
    if seen != set(CANDIDATES):
        fail("candidate set mismatch")

    math = obj(report["capacity_math"], "capacity_math")
    combined = 5428358 + raw_bytes
    expected_math = {
        "authorized_unique_loss_positions_now": 0,
        "authorized_deficit_to_meaningful_floor": 10000000,
        "clean_oracle_one_pass_position_ceiling": 5428358,
        "minimum_deficit_if_oracle_becomes_exact_clean_successor": 4571642,
        "named_candidate_raw_bytes": raw_bytes,
        "clean_oracle_plus_named_candidate_raw_bytes": combined,
        "post_filter_unique_loss_expected_range_authorized": None,
        "post_filter_mathematical_lower_bound": 0,
        "post_filter_mathematical_upper_bound": combined,
    }
    if math.get("post_filter_unique_loss_expected_range_authorized") is not None:
        fail("current evidence does not authorize a numeric expected post-filter range")
    for key, value in expected_math.items():
        if value is not None:
            integer(math.get(key), f"capacity_math.{key}")
    if dict(math) != expected_math:
        fail("capacity arithmetic mismatch")

    decision = obj(report["decision"], "decision")
    if decision.get("routing") == STOP:
        fail("STOP decision is illegal while terminal authorized supply is below floor")
    if decision.get("routing") != CONTINUE:
        fail("decision.routing is not the current evidence-bound routing")
    if boolean(decision.get("open_new_source_families"), "decision.open_new_source_families"):
        fail("v1 decision must not fan out new source families while named supply is pending")
    if decision.get("recommended_execution_order") != [
        "pep_loc", "caselaw", "ubuntu_irc", "arxiv_languk"
    ]:
        fail("recommended execution order drifted")
    triggers = [
        f"different_worker_audit_pr1459_{DEDUP_HEAD}",
        "lawful_integration_pr1459_after_terminal_audit",
        "finish_clean_successor_issue_1951",
        "execute_pep_loc_then_caselaw_through_incumbent_matcher",
    ]
    if decision.get("immediate_triggers") != triggers:
        fail("decision.immediate_triggers drifted from the current shortest-path routing")
    if decision.get("stop_condition") != (
        "terminal post-pack unique-loss authority >= 10000000 after complete downstream gates"
    ):
        fail("decision.stop_condition drifted")
    if not isinstance(decision.get("rationale"), str) or not decision["rationale"]:
        fail("decision.rationale must be non-empty")

    evidence = obj(report["evidence_roots"], "evidence_roots")
    expected_evidence = {
        "scientific_control_issue": 548,
        "capacity_task_issue": 2021,
        "acceleration_board_issue": 2011,
        "candidate_queue_issue": 2020,
        "clean_oracle_issue": 1881,
        "clean_successor_owner_issue": 1951,
        "global_dedup_pr": 1459,
        "global_dedup_released_head": DEDUP_HEAD,
        "global_dedup_exact_head_ci_run": 34873834969,
        "global_dedup_exact_head_ci_conclusion": "SUCCESS",
    }
    hexid(evidence.get("global_dedup_released_head"), "evidence.dedup_head", 40)
    if evidence.get("global_dedup_released_head") != DEDUP_HEAD:
        fail("global dedup released head drifted from the late-bound PR1459 release")
    if evidence.get("global_dedup_exact_head_ci_conclusion") != "SUCCESS":
        fail("global dedup exact-head CI is not the observed released-head SUCCESS")
    for key in expected_evidence:
        if key.endswith("_issue") or key in {"global_dedup_pr", "global_dedup_exact_head_ci_run"}:
            integer(evidence.get(key), f"evidence.{key}", 1)
    if dict(evidence) != expected_evidence:
        fail("evidence_roots drifted")


def load_and_validate(path: str | Path, *, expected_main_sha: str | None = None) -> dict[str, Any]:
    def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                fail(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    document = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=no_duplicates)
    validate_report(document, expected_main_sha=expected_main_sha)
    return document
